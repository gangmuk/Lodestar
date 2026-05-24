#!/usr/bin/env python3
"""
Plot distributions of input tokens, output tokens, RPS, inter-arrival times,
and prefix sharing for both workload.jsonl and data-processed.csv.
"""

import csv
import json
import statistics
import os
import sys
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <workload.jsonl>")
    print(f"  e.g. python3 {sys.argv[0]} ./conversation-2/workload.jsonl")
    sys.exit(1)

WORKLOAD_FILE = os.path.abspath(sys.argv[1])
TARGET_DIR = os.path.dirname(WORKLOAD_FILE)
TRAINING_FILE = os.path.join(TARGET_DIR, "data-processed.csv")
workload_name = os.path.splitext(os.path.basename(WORKLOAD_FILE))[0]
OUTPUT_BASE = os.path.join(TARGET_DIR, f"data_distribution-{workload_name}")
# Increase all font sizes globally
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
})


def load_workload():
    """Load workload.jsonl and return list of request dicts."""
    entries = []
    with open(WORKLOAD_FILE) as f:
        for line in f:
            obj = json.loads(line)
            ts = obj["timestamp"]
            for req in obj["requests"]:
                entries.append({
                    "timestamp_ms": ts,
                    "input_tokens": req["Prompt Length"],
                    "output_tokens": req["Output Length"],
                    "hash_ids": req.get("hash_ids", []),
                    "prefix_group": req.get("prefix_group", ""),
                    "prompt": req.get("prompt", ""),
                })
    return entries


def load_training_data():
    """Load data-processed.csv and return list of row dicts."""
    with open(TRAINING_FILE) as f:
        return list(csv.DictReader(f))


def separate_training_runs(rows):
    """
    Separate training data into individual experiment runs.
    Greedy assignment: for each (algo, iter_bin), assign duplicate request_ids
    to different runs by time order.
    """
    WORKLOAD_SIZE = 5083
    runs = []
    for algo in sorted(set(r["subAlgorithm"] for r in rows)):
        for iter_num in range(2):
            subset = [
                r for r in rows
                if r["subAlgorithm"] == algo
                and int(r["request_id"]) // WORKLOAD_SIZE == iter_num
            ]
            if not subset:
                continue
            id_buckets = defaultdict(list)
            for r in subset:
                id_buckets[int(r["request_id"])].append(r)
            for rid in id_buckets:
                id_buckets[rid].sort(key=lambda r: float(r["request_start_time"]))
            max_runs = max(len(v) for v in id_buckets.values())
            for run_idx in range(max_runs):
                run_rows = []
                for rid in sorted(id_buckets.keys()):
                    bucket = id_buckets[rid]
                    if run_idx < len(bucket):
                        run_rows.append(bucket[run_idx])
                run_rows.sort(key=lambda r: float(r["request_start_time"]))
                runs.append({
                    "algo": algo,
                    "iter": iter_num,
                    "run_idx": run_idx,
                    "rows": run_rows,
                    "label": f"{algo}_iter{iter_num}_run{run_idx}",
                })
    return runs


def compute_inter_arrival_times_ms(times_us):
    """Compute inter-arrival times in ms from sorted timestamps in microseconds."""
    sorted_times = sorted(times_us)
    return [
        (sorted_times[i + 1] - sorted_times[i]) / 1e3
        for i in range(len(sorted_times) - 1)
    ]


def compute_rps_timeseries(times_us, bin_size_s=10):
    """Compute RPS in fixed-width time bins."""
    if not times_us:
        return [], []
    times_s = [t / 1e6 for t in times_us]
    t_min, t_max = min(times_s), max(times_s)
    bins = np.arange(t_min, t_max + bin_size_s, bin_size_s)
    counts, edges = np.histogram(times_s, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    rps = counts / bin_size_s
    return centers, rps


def compute_unique_groups_timeseries(entries_sorted, window_ms=30000, step_ms=1000):
    """Count unique prefix groups in a sliding window over time.

    Uses a two-pointer sweep for O(n) efficiency instead of re-scanning
    all entries for every time step.
    """
    if not entries_sorted:
        return [], []
    t_min = entries_sorted[0]["timestamp_ms"]
    t_max = entries_sorted[-1]["timestamp_ms"]
    n = len(entries_sorted)
    xs, ys = [], []
    left = 0
    right = 0
    group_counts = Counter()
    for t in range(int(t_min), int(t_max) + step_ms, step_ms):
        w_start = t - window_ms
        while right < n and entries_sorted[right]["timestamp_ms"] <= t:
            group_counts[entries_sorted[right]["prefix_group"]] += 1
            right += 1
        while left < n and entries_sorted[left]["timestamp_ms"] <= w_start:
            pg = entries_sorted[left]["prefix_group"]
            group_counts[pg] -= 1
            if group_counts[pg] == 0:
                del group_counts[pg]
            left += 1
        xs.append((t - t_min) / 1000)
        ys.append(len(group_counts))
    return xs, ys


class TrieNode:
    """Trie node for prefix matching on word-chunk sequences."""
    __slots__ = ['children']
    def __init__(self):
        self.children = {}


def compute_prefix_hit_ratios(entries_sorted, chunk_size=10):
    """
    Compute prefix hit ratio for each request in temporal order using
    actual prompt text.

    Chunks each prompt into fixed-size word blocks, hashes each block,
    and matches against a trie of previously seen prompts — simulating
    an online KV-cache.  chunk_size=10 words gives ~1% precision on the
    prefix boundary while keeping memory reasonable.

    Returns list of hit_ratios (one per request, in order).
    """
    import hashlib

    root = TrieNode()
    hit_ratios = []

    for entry in entries_sorted:
        words = entry["prompt"].split()
        # Chunk into fixed-size word blocks and hash each block
        chunks = []
        for i in range(0, len(words), chunk_size):
            block = " ".join(words[i:i + chunk_size])
            chunks.append(hashlib.md5(block.encode()).hexdigest())

        total_blocks = len(chunks)

        node = root
        matched_blocks = 0
        for h in chunks:
            if h in node.children:
                matched_blocks += 1
                node = node.children[h]
            else:
                break

        hit_ratio = matched_blocks / total_blocks if total_blocks > 0 else 0.0
        hit_ratios.append(hit_ratio)

        node = root
        for h in chunks:
            if h not in node.children:
                node.children[h] = TrieNode()
            node = node.children[h]

    return hit_ratios


def verify_prompt_text_sharing(entries_sorted, sample_size=200):
    """Spot-check text-level prefix sharing within prefix groups.
    Samples evenly across all groups to cover the full workload."""
    groups = defaultdict(list)
    for e in entries_sorted:
        groups[e["prefix_group"]].append(e)

    # Sample evenly across all groups (not just the first N)
    group_keys = [k for k, v in groups.items() if len(v) >= 2]
    if len(group_keys) > sample_size:
        step = len(group_keys) / sample_size
        sampled_keys = [group_keys[int(i * step)] for i in range(sample_size)]
    else:
        sampled_keys = group_keys

    checked = 0
    text_ratios = []
    for pg in sampled_keys:
        reqs = groups[pg]
        p1 = reqs[0]["prompt"]
        p2 = reqs[1]["prompt"]
        common_chars = 0
        for c1, c2 in zip(p1, p2):
            if c1 == c2:
                common_chars += 1
            else:
                break
        text_ratio = common_chars / min(len(p1), len(p2)) if min(len(p1), len(p2)) > 0 else 0
        text_ratios.append(text_ratio)
        checked += 1

    avg_text = statistics.mean(text_ratios) if text_ratios else 0
    print(f"  Prompt-text verification: {checked} groups checked, "
          f"avg text sharing ratio={avg_text:.3f}")


def save_workload_csv(workload_sorted, hit_ratios, out_path):
    """Save per-request time-series CSV with all plotted metrics."""
    # Compute per-request inter-arrival time
    iats = [0.0]  # first request has no IAT
    for i in range(1, len(workload_sorted)):
        iat_ms = workload_sorted[i]["timestamp_ms"] - workload_sorted[i - 1]["timestamp_ms"]
        iats.append(iat_ms)

    # Compute RPS: assign each request to a 1s time bin
    t0_ms = workload_sorted[0]["timestamp_ms"]
    bin_size_ms = 1000
    # Count requests per bin
    bin_counts = Counter()
    for e in workload_sorted:
        b = int((e["timestamp_ms"] - t0_ms) // bin_size_ms)
        bin_counts[b] += 1
    # Map each request to its bin's RPS
    rps_per_req = []
    for e in workload_sorted:
        b = int((e["timestamp_ms"] - t0_ms) // bin_size_ms)
        rps_per_req.append(bin_counts[b] / (bin_size_ms / 1000.0))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "request_idx", "timestamp_ms", "timestamp_s",
            "input_tokens", "output_tokens",
            "inter_arrival_time_ms", "rps_1s_bin",
            "prefix_hit_ratio", "prefix_group", "num_hash_blocks",
        ])
        for i, e in enumerate(workload_sorted):
            writer.writerow([
                i,
                e["timestamp_ms"],
                f"{e['timestamp_ms'] / 1000.0:.3f}",
                e["input_tokens"],
                e["output_tokens"],
                f"{iats[i]:.3f}",
                f"{rps_per_req[i]:.1f}",
                f"{hit_ratios[i]:.6f}",
                e["prefix_group"],
                len(e["hash_ids"]),
            ])
    print(f"Saved per-request CSV: {out_path}")


def main():
    print("Loading data...")
    workload = load_workload()

    has_training = os.path.isfile(TRAINING_FILE)
    if has_training:
        training_rows = load_training_data()
        runs = separate_training_runs(training_rows)
        runs = [r for r in runs if len(r["rows"]) > 100]
        print(f"Workload: {len(workload)} requests")
        print(f"Training: {len(training_rows)} rows, {len(runs)} runs (>100 rows each):")
        for r in runs:
            print(f"  {r['label']}: {len(r['rows'])} rows")
    else:
        training_rows, runs = [], []
        print(f"Workload: {len(workload)} requests")
        print(f"No training data found ({TRAINING_FILE}), plotting workload only.")

    # --- Extract data ---
    wl_input = [e["input_tokens"] for e in workload]
    wl_output = [e["output_tokens"] for e in workload]
    wl_times_ms = sorted(e["timestamp_ms"] for e in workload)
    wl_iat_ms = [wl_times_ms[i + 1] - wl_times_ms[i] for i in range(len(wl_times_ms) - 1)]

    if has_training:
        tr_input = [int(r["input_tokens"]) for r in training_rows]
        tr_output = [int(r["output_tokens"]) for r in training_rows]

        # Per-run inter-arrival times
        run_iats = {}
        for run in runs:
            times_us = [float(r["request_start_time"]) for r in run["rows"]]
            run_iats[run["label"]] = compute_inter_arrival_times_ms(times_us)

    # --- Prefix sharing ---
    print("Computing prefix hit ratios (temporal ordering)...")
    workload_sorted = sorted(workload, key=lambda e: e["timestamp_ms"])
    wl_hit_ratios = compute_prefix_hit_ratios(workload_sorted)
    verify_prompt_text_sharing(workload_sorted)

    # Also compute hit ratio over time (rolling window)
    window_size = 10
    hit_ratio_rolling = []
    for i in range(window_size, len(wl_hit_ratios)):
        window_mean = statistics.mean(wl_hit_ratios[i - window_size:i])
        hit_ratio_rolling.append((i, window_mean))

    # --- Unique prefix groups over time (30s sliding window) ---
    print("Computing unique prefix groups over time (30s window)...")
    wl_sorted_by_time = sorted(workload, key=lambda e: e["timestamp_ms"])
    wl_upg_x, wl_upg_y = compute_unique_groups_timeseries(wl_sorted_by_time)

    # --- Plot ---
    fig, axes = plt.subplots(1, 6, figsize=(36, 5))

    # 1. Input token distribution
    ax = axes[0]
    if has_training:
        bins_in = np.linspace(0, max(max(wl_input), max(tr_input)), 60)
        ax.hist(wl_input, bins=bins_in, alpha=0.6, label=f"workload (n={len(wl_input)})", density=True)
        ax.hist(tr_input, bins=bins_in, alpha=0.6, label=f"training (n={len(tr_input)})", density=True)
    else:
        bins_in = np.linspace(0, max(wl_input), 60)
        ax.hist(wl_input, bins=bins_in, alpha=0.6, label=f"n={len(wl_input)}", density=True)
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Density")
    ax.set_title("Input Token Distribution")
    ax.legend()

    # 2. Output token distribution
    ax = axes[1]
    if has_training:
        bins_out = np.linspace(0, max(max(wl_output), max(tr_output)), 60)
        ax.hist(wl_output, bins=bins_out, alpha=0.6, label=f"workload (n={len(wl_output)})", density=True)
        ax.hist(tr_output, bins=bins_out, alpha=0.6, label=f"training (n={len(tr_output)})", density=True)
    else:
        bins_out = np.linspace(0, max(wl_output), 60)
        ax.hist(wl_output, bins=bins_out, alpha=0.6, label=f"n={len(wl_output)}", density=True)
    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Density")
    ax.set_title("Output Token Distribution")
    ax.legend()

    # 3. Inter-arrival time
    ax = axes[2]
    if has_training:
        colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
        all_training_iat = []
        for i, run in enumerate(runs):
            iat = run_iats[run["label"]]
            if iat:
                all_training_iat.extend(iat)
                ax.hist(iat, bins=80, alpha=0.4, label=run["label"], density=True, color=colors[i])
        ax.set_title("Inter-arrival Time: Training (per run)")
        if all_training_iat:
            ax.set_xlim(0, np.percentile(all_training_iat, 99))
        ax.legend(fontsize=10)
    else:
        if wl_iat_ms:
            ax.hist(wl_iat_ms, bins=80, alpha=0.6, density=True)
            ax.set_xlim(0, np.percentile(wl_iat_ms, 99))
        ax.set_title("Inter-arrival Time (workload)")
    ax.set_xlabel("Inter-arrival Time (ms)")
    ax.set_ylabel("Density")

    # 4. Prefix hit ratio over time (rolling mean)
    ax = axes[3]
    ax.scatter(range(len(wl_hit_ratios)), wl_hit_ratios, alpha=0.15, s=8, color="C2",
               label="per-request hit ratio")
    if hit_ratio_rolling:
        roll_x = [r[0] for r in hit_ratio_rolling]
        roll_y = [r[1] for r in hit_ratio_rolling]
        ax.plot(roll_x, roll_y, color="red", alpha=0.8, linewidth=1.2,
                label=f"sliding window avg (window={window_size})")
    ax.set_xlabel("Request Index")
    ax.set_ylabel("Prefix Hit Ratio")
    ax.set_title("Prefix Hit Ratio Over Time")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)

    # 5. Unique prefix groups over time (30s sliding window)
    ax = axes[4]
    if wl_upg_x:
        ax.plot(wl_upg_x, wl_upg_y, color="C3", alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unique Prefix Groups")
    ax.set_title("Unique Prefix Groups (30s window)")

    # 6. KV cache hit ratio distribution (histogram)
    # Hit ratios are computed temporally: each request can only hit prefixes
    # that were inserted by earlier requests (simulating online KV cache).
    ax = axes[5]
    hr_arr = np.array(wl_hit_ratios)
    # Exclude the very first request (always 0 hit, cold cache) from stats
    hr_nonzero_eligible = hr_arr[1:]  # all requests after the first
    bins_hr = np.linspace(0, 1, 51)
    ax.hist(hr_arr, bins=bins_hr, alpha=0.7, color="C2", edgecolor="white", linewidth=0.5,
            label=f"all (n={len(hr_arr)})")
    ax.axvline(hr_arr.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={hr_arr.mean():.3f}")
    ax.axvline(np.median(hr_arr), color="orange", linestyle=":", linewidth=1.5,
               label=f"median={np.median(hr_arr):.3f}")
    zero_pct = (hr_arr == 0).mean() * 100
    full_pct = (hr_arr == 1.0).mean() * 100
    ax.text(0.97, 0.95, f"zero-hit: {zero_pct:.1f}%\nfull-hit: {full_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlabel("KV Cache Hit Ratio")
    ax.set_ylabel("Count")
    ax.set_title("KV Cache Hit Ratio Distribution")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=11)

    title = "Workload vs Training Data Distribution" if has_training else "Workload Distribution"
    plt.suptitle(f"{title} ({os.path.basename(TARGET_DIR)})", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE + ".pdf", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_BASE}.pdf")

    # Save per-request time-series CSV
    csv_path = OUTPUT_BASE + "_per_request.csv"
    save_workload_csv(workload_sorted, wl_hit_ratios, csv_path)


if __name__ == "__main__":
    main()
