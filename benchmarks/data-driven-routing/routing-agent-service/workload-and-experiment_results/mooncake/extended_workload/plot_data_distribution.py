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
TOKENS_PER_BLOCK = 50  # each hash_id block = 50 tokens

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


class TrieNode:
    """Trie node for prefix matching on hash_id sequences."""
    __slots__ = ['children']
    def __init__(self):
        self.children = {}


def compute_prefix_hit_ratios(entries_sorted):
    """
    Compute prefix hit ratio for each request in temporal order.

    For each request at time T, find the longest prefix match from any
    previously seen request by walking a trie built on hash_ids.

    Hit ratio = (matched_blocks * TOKENS_PER_BLOCK) / input_tokens

    Returns list of hit_ratios (one per request, in order).
    """
    root = TrieNode()
    hit_ratios = []

    for entry in entries_sorted:
        hids = entry["hash_ids"]
        input_tokens = entry["input_tokens"]

        # Walk trie to find longest prefix match from prior requests
        node = root
        matched_blocks = 0
        for h in hids:
            if h in node.children:
                matched_blocks += 1
                node = node.children[h]
            else:
                break

        shared_tokens = matched_blocks * TOKENS_PER_BLOCK
        hit_ratio = shared_tokens / input_tokens if input_tokens > 0 else 0.0
        hit_ratios.append(hit_ratio)

        # Insert this request's hash_ids into trie
        node = root
        for h in hids:
            if h not in node.children:
                node.children[h] = TrieNode()
            node = node.children[h]

    return hit_ratios


def main():
    print("Loading data...")
    workload = load_workload()
    training_rows = load_training_data()
    runs = separate_training_runs(training_rows)
    runs = [r for r in runs if len(r["rows"]) > 100]

    print(f"Workload: {len(workload)} requests")
    print(f"Training: {len(training_rows)} rows, {len(runs)} runs (>100 rows each):")
    for r in runs:
        print(f"  {r['label']}: {len(r['rows'])} rows")

    # --- Extract data ---
    wl_input = [e["input_tokens"] for e in workload]
    wl_output = [e["output_tokens"] for e in workload]
    wl_times_ms = sorted(e["timestamp_ms"] for e in workload)
    wl_iat_ms = [wl_times_ms[i + 1] - wl_times_ms[i] for i in range(len(wl_times_ms) - 1)]

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

    # Also compute hit ratio over time (rolling window)
    window_size = 10
    hit_ratio_rolling = []
    for i in range(window_size, len(wl_hit_ratios)):
        window_mean = statistics.mean(wl_hit_ratios[i - window_size:i])
        hit_ratio_rolling.append((i, window_mean))

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # 1. Input token distribution
    ax = axes[0]
    bins_in = np.linspace(0, max(max(wl_input), max(tr_input)), 60)
    ax.hist(wl_input, bins=bins_in, alpha=0.6, label=f"workload (n={len(wl_input)})", density=True)
    ax.hist(tr_input, bins=bins_in, alpha=0.6, label=f"training (n={len(tr_input)})", density=True)
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Density")
    ax.set_title("Input Token Distribution")
    ax.legend()

    # 2. Output token distribution
    ax = axes[1]
    bins_out = np.linspace(0, max(max(wl_output), max(tr_output)), 60)
    ax.hist(wl_output, bins=bins_out, alpha=0.6, label=f"workload (n={len(wl_output)})", density=True)
    ax.hist(tr_output, bins=bins_out, alpha=0.6, label=f"training (n={len(tr_output)})", density=True)
    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Density")
    ax.set_title("Output Token Distribution")
    ax.legend()

    # 3. Inter-arrival time (training, per run)
    ax = axes[2]
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    all_training_iat = []
    for i, run in enumerate(runs):
        iat = run_iats[run["label"]]
        if iat:
            all_training_iat.extend(iat)
            ax.hist(iat, bins=80, alpha=0.4, label=run["label"], density=True, color=colors[i])
    ax.set_xlabel("Inter-arrival Time (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Inter-arrival Time: Training (per run)")
    if all_training_iat:
        ax.set_xlim(0, np.percentile(all_training_iat, 99))
    ax.legend(fontsize=10)

    # 4. Prefix hit ratio over time (rolling mean)
    ax = axes[3]
    if hit_ratio_rolling:
        roll_x = [r[0] for r in hit_ratio_rolling]
        roll_y = [r[1] for r in hit_ratio_rolling]
        # ax.plot(roll_x, roll_y, color="black", alpha=0.8, linewidth=1.2, label=f"rolling mean (window={window_size})")
    ax.scatter(range(len(wl_hit_ratios)), wl_hit_ratios, alpha=0.15, s=8, color="C2",
               label="per-request hit ratio")
    ax.set_xlabel("Request Index")
    ax.set_ylabel("Prefix Hit Ratio")
    ax.set_title("Prefix Hit Ratio Over Time")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)

    plt.suptitle(f"Workload vs Training Data Distribution ({os.path.basename(TARGET_DIR)})", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(OUTPUT_BASE + ".pdf", dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {OUTPUT_BASE}.png")
    print(f"Saved plot to {OUTPUT_BASE}.pdf")

    # --- Summary stats ---
    print("\n=== Token Distribution Summary ===")
    fmt = "{:30s} | {:>12s} | {:>12s}"
    print(fmt.format("", "Workload", "Training"))
    print(f"{'-'*30}-+-{'-'*12}-+-{'-'*12}")
    print(fmt.format("Count", str(len(wl_input)), str(len(tr_input))))
    print(fmt.format("Input tokens (mean)", f"{statistics.mean(wl_input):.0f}", f"{statistics.mean(tr_input):.0f}"))
    print(fmt.format("Input tokens (median)", f"{statistics.median(wl_input):.0f}", f"{statistics.median(tr_input):.0f}"))
    print(fmt.format("Output tokens (mean)", f"{statistics.mean(wl_output):.0f}", f"{statistics.mean(tr_output):.0f}"))
    print(fmt.format("Output tokens (median)", f"{statistics.median(wl_output):.0f}", f"{statistics.median(tr_output):.0f}"))

    print("\n=== Prefix Hit Ratio Summary (workload, temporal) ===")
    hr = wl_hit_ratios
    print(f"  Requests with hit > 0: {sum(1 for r in hr if r > 0)} / {len(hr)} ({100*sum(1 for r in hr if r > 0)/len(hr):.1f}%)")
    print(f"  Mean:   {statistics.mean(hr):.3f}")
    print(f"  Median: {statistics.median(hr):.3f}")
    pcts = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    sorted_hr = sorted(hr)
    print(f"  Percentiles: " + ", ".join(
        f"p{p}={sorted_hr[min(int(p/100*len(sorted_hr)), len(sorted_hr)-1)]:.3f}" for p in pcts))

    # Histogram
    bins_hr = [(0, 0.01), (0.01, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5),
               (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    print(f"\n  Hit ratio histogram:")
    for lo, hi in bins_hr:
        count = sum(1 for r in hr if lo <= r < hi)
        pct = 100 * count / len(hr)
        bar = '#' * int(pct)
        print(f"    [{lo:.2f}, {hi:.2f}): {count:5d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
