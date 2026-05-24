#!/usr/bin/env python3
"""
Verify prefix hit ratios with temporal dependency (trie-based prefix cache).

Each hash_id = 1 block = 100 tokens (all same token).
A prefix "hit" means matching consecutive hash_ids from the start
that were already inserted into the trie by earlier requests.

This simulates what a real KV-cache prefix matcher does:
  - Process requests in timestamp order
  - For each request, walk the trie matching hash_ids from the start
  - Count matched blocks / total blocks = prefix hit ratio
  - Then insert the full hash_id chain into the trie
"""

import json
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


def load_workload(path):
    """Load workload.jsonl, return list of (timestamp, hash_ids, input_tokens)."""
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            req = obj["requests"][0]
            records.append({
                "timestamp": obj["timestamp"],
                "hash_ids": req["hash_ids"],
                "input_tokens": req["Prompt Length"],
                "output_tokens": req["Output Length"],
                "prefix_group": req.get("prefix_group", ""),
            })
    # Already sorted by timestamp in workload files, but ensure it
    records.sort(key=lambda r: r["timestamp"])
    return records


def compute_prefix_hits_temporal(records):
    """Simulate trie-based prefix cache, return per-request hit ratios."""

    class TrieNode:
        __slots__ = ['children']
        def __init__(self):
            self.children = {}

    root = TrieNode()
    hit_ratios = []
    matched_blocks_list = []
    total_blocks_list = []

    for rec in records:
        hids = rec["hash_ids"]
        total_blocks = len(hids)

        # Walk trie to count matched prefix blocks
        node = root
        matched = 0
        for h in hids:
            if h in node.children:
                matched += 1
                node = node.children[h]
            else:
                break

        hit_ratio = matched / total_blocks if total_blocks > 0 else 0.0
        hit_ratios.append(hit_ratio)
        matched_blocks_list.append(matched)
        total_blocks_list.append(total_blocks)

        # Insert full chain into trie
        node = root
        for h in hids:
            if h not in node.children:
                node.children[h] = TrieNode()
            node = node.children[h]

    return hit_ratios, matched_blocks_list, total_blocks_list


def windowed_stats(timestamps, values, window_s=30):
    """Compute windowed average of values over time."""
    ts_arr = np.array(timestamps) / 1000.0  # to seconds
    val_arr = np.array(values)
    max_t = ts_arr[-1]
    windows = []
    means = []
    t = 0
    while t < max_t:
        mask = (ts_arr >= t) & (ts_arr < t + window_s)
        if mask.any():
            means.append(val_arr[mask].mean())
        else:
            means.append(np.nan)
        windows.append(t + window_s / 2)
        t += window_s
    return windows, means


def analyze_workload(name, path):
    """Full analysis of a single workload."""
    print(f"\n{'='*70}")
    print(f"  {name}: {path}")
    print(f"{'='*70}")

    records = load_workload(path)
    print(f"  Total requests: {len(records)}")
    print(f"  Duration: {records[-1]['timestamp']/1000:.1f}s")

    hit_ratios, matched_blocks, total_blocks = compute_prefix_hits_temporal(records)

    timestamps = [r["timestamp"] for r in records]

    # Overall stats
    hr = np.array(hit_ratios)
    print(f"\n  Prefix hit ratio (block-level, temporal trie):")
    print(f"    Mean:   {hr.mean():.4f}")
    print(f"    Median: {np.median(hr):.4f}")
    print(f"    Std:    {hr.std():.4f}")
    print(f"    P25:    {np.percentile(hr, 25):.4f}")
    print(f"    P75:    {np.percentile(hr, 75):.4f}")
    print(f"    P90:    {np.percentile(hr, 90):.4f}")
    print(f"    Zero-hit requests: {(hr == 0).sum()} / {len(hr)} ({(hr == 0).mean()*100:.1f}%)")
    print(f"    Full-hit requests: {(hr == 1.0).sum()} / {len(hr)} ({(hr == 1.0).mean()*100:.1f}%)")

    # Matched blocks stats
    mb = np.array(matched_blocks)
    tb = np.array(total_blocks)
    print(f"\n  Block counts:")
    print(f"    Total blocks/req:   mean={tb.mean():.1f}, median={np.median(tb):.0f}")
    print(f"    Matched blocks/req: mean={mb.mean():.1f}, median={np.median(mb):.0f}")

    # Token-weighted hit ratio (what fraction of total input tokens are cache hits)
    total_input_tokens = tb.sum() * 100  # 100 tokens per block
    total_hit_tokens = mb.sum() * 100
    print(f"\n  Token-weighted prefix hit ratio: {total_hit_tokens/total_input_tokens:.4f}")

    # Temporal analysis: windowed hit ratio
    print(f"\n  Temporal prefix hit ratio (30s windows):")
    windows, w_means = windowed_stats(timestamps, hit_ratios, window_s=30)
    w_means_clean = [m for m in w_means if not np.isnan(m)]

    # Split into head (first 70%) and tail (last 30%) by time
    duration_s = records[-1]["timestamp"] / 1000.0
    head_cutoff_s = duration_s * 0.7

    head_hrs = [hr for hr, r in zip(hit_ratios, records) if r["timestamp"]/1000.0 <= head_cutoff_s]
    tail_hrs = [hr for hr, r in zip(hit_ratios, records) if r["timestamp"]/1000.0 > head_cutoff_s]

    print(f"    Head (0-{head_cutoff_s:.0f}s, {len(head_hrs)} reqs): mean={np.mean(head_hrs):.4f}")
    print(f"    Tail ({head_cutoff_s:.0f}s-end, {len(tail_hrs)} reqs):  mean={np.mean(tail_hrs):.4f}")

    # Prefix group analysis
    pg_members = defaultdict(list)
    for i, r in enumerate(records):
        pg_members[r["prefix_group"]].append(i)

    pg_sizes = [len(v) for v in pg_members.values()]
    print(f"\n  Prefix groups: {len(pg_members)}")
    print(f"    Size: mean={np.mean(pg_sizes):.1f}, median={np.median(pg_sizes):.0f}, max={max(pg_sizes)}")

    # Within-group hit analysis: for requests that are NOT the first in their group
    within_group_hrs = []
    first_in_group_hrs = []
    for pg, indices in pg_members.items():
        for j, idx in enumerate(indices):
            if j == 0:
                first_in_group_hrs.append(hit_ratios[idx])
            else:
                within_group_hrs.append(hit_ratios[idx])

    if within_group_hrs:
        print(f"\n  First-in-group hit ratio:  mean={np.mean(first_in_group_hrs):.4f} ({len(first_in_group_hrs)} reqs)")
        print(f"  Follow-up-in-group hit ratio: mean={np.mean(within_group_hrs):.4f} ({len(within_group_hrs)} reqs)")

    # Hash ID reuse: how many unique hash_ids vs total hash_id slots
    all_hids = []
    for r in records:
        all_hids.extend(r["hash_ids"])
    unique_hids = len(set(all_hids))
    print(f"\n  Hash ID usage:")
    print(f"    Total hash_id slots: {len(all_hids)}")
    print(f"    Unique hash_ids:     {unique_hids}")
    print(f"    Reuse ratio:         {len(all_hids)/unique_hids:.2f}x")

    # Check for synthesized request artifacts:
    # How many requests share >90% of their hash_ids with another request?
    # (Sample first 2000 for perf)
    sample = records[:min(2000, len(records))]
    near_duplicate_count = 0
    for i in range(1, len(sample)):
        hids_i = sample[i]["hash_ids"]
        # Check against previous 50 requests
        for j in range(max(0, i-50), i):
            hids_j = sample[j]["hash_ids"]
            shared = 0
            for a, b in zip(hids_i, hids_j):
                if a == b:
                    shared += 1
                else:
                    break
            overlap = shared / min(len(hids_i), len(hids_j)) if min(len(hids_i), len(hids_j)) > 0 else 0
            if overlap > 0.9:
                near_duplicate_count += 1
                break
    print(f"\n  Near-duplicate check (first {len(sample)} reqs, >90% prefix overlap with prev 50):")
    print(f"    {near_duplicate_count} / {len(sample)} ({near_duplicate_count/len(sample)*100:.1f}%)")

    return {
        "name": name,
        "records": records,
        "hit_ratios": hit_ratios,
        "matched_blocks": matched_blocks,
        "total_blocks": total_blocks,
        "timestamps": timestamps,
        "windows": windows,
        "w_means": w_means,
    }


def save_csv(result, out_dir):
    """Save CSVs: per-request timeseries, prefix group sizes, intra-group distances."""
    os.makedirs(out_dir, exist_ok=True)
    records = result["records"]

    # 1) Per-request timeseries CSV
    ts_path = os.path.join(out_dir, "kv_hit_ratio_timeseries.csv")
    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_idx", "timestamp_ms", "timestamp_s", "input_tokens",
                         "output_tokens", "prefix_group", "total_blocks", "matched_blocks",
                         "prefix_hit_ratio"])
        for i, rec in enumerate(records):
            writer.writerow([
                i,
                rec["timestamp"],
                rec["timestamp"] / 1000.0,
                rec["input_tokens"],
                rec["output_tokens"],
                rec["prefix_group"],
                result["total_blocks"][i],
                result["matched_blocks"][i],
                f"{result['hit_ratios'][i]:.6f}",
            ])
    print(f"  Saved: {ts_path}")

    # 2) Prefix group size distribution CSV
    pg_members = defaultdict(list)
    for i, r in enumerate(records):
        pg_members[r["prefix_group"]].append(i)

    pg_path = os.path.join(out_dir, "prefix_group_size_distribution.csv")
    with open(pg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix_group", "num_requests"])
        for pg in sorted(pg_members.keys()):
            writer.writerow([pg, len(pg_members[pg])])
    print(f"  Saved: {pg_path}")

    # 3) Intra-group request distance distribution CSV
    dist_path = os.path.join(out_dir, "intra_prefix_group_request_distance_distribution.csv")
    with open(dist_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix_group", "req_idx_a", "req_idx_b", "distance_s"])
        for pg, indices in sorted(pg_members.items()):
            if len(indices) < 2:
                continue
            for j in range(1, len(indices)):
                dt_s = (records[indices[j]]["timestamp"] - records[indices[j - 1]]["timestamp"]) / 1000.0
                writer.writerow([pg, indices[j - 1], indices[j], f"{dt_s:.3f}"])
    print(f"  Saved: {dist_path}")


def plot_timeseries(results, out_dir):
    """Plot time-series of prefix hit ratio, input length, and prefix group stats in one figure."""
    os.makedirs(out_dir, exist_ok=True)

    for res in results:
        ts_s = np.array(res["timestamps"]) / 1000.0
        hr = np.array(res["hit_ratios"])
        input_tokens = np.array([r["input_tokens"] for r in res["records"]])
        records = res["records"]

        fig = plt.figure(figsize=(14, 14))
        fig.suptitle(res["name"], fontsize=13)
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # Row 1 col 1: prefix hit ratio time series
        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(ts_s, hr, alpha=0.15, s=4, color="steelblue", label="per-request")
        w_t = np.array(res["windows"])
        w_m = np.array(res["w_means"])
        valid = ~np.isnan(w_m)
        ax1.plot(w_t[valid], w_m[valid], color="red", linewidth=2, label="30s windowed mean")
        ax1.set_ylabel("Prefix Hit Ratio")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Row 2 full width: input length time series
        ax2 = fig.add_subplot(gs[1, :])
        ax2.scatter(ts_s, input_tokens, alpha=0.15, s=4, color="darkorange", label="per-request")
        w_t_inp, w_m_inp = windowed_stats(res["timestamps"], input_tokens.tolist(), window_s=30)
        w_t_inp = np.array(w_t_inp)
        w_m_inp = np.array(w_m_inp)
        valid_inp = ~np.isnan(w_m_inp)
        ax2.plot(w_t_inp[valid_inp], w_m_inp[valid_inp], color="red", linewidth=2, label="30s windowed mean")
        ax2.set_ylabel("Input Tokens")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Compute prefix group stats
        pg_members = defaultdict(list)
        for i, r in enumerate(records):
            pg_members[r["prefix_group"]].append(i)

        pg_sizes = [len(indices) for indices in pg_members.values()]

        intra_distances_s = []
        for indices in pg_members.values():
            if len(indices) < 2:
                continue
            for j in range(1, len(indices)):
                dt_s = (records[indices[j]]["timestamp"] - records[indices[j - 1]]["timestamp"]) / 1000.0
                intra_distances_s.append(dt_s)

        # Row 3 left: prefix group size distribution
        ax3 = fig.add_subplot(gs[2, 0])
        pg_sizes_arr = np.array(pg_sizes)
        max_bin = min(int(np.percentile(pg_sizes_arr, 99)) + 2, max(pg_sizes) + 1)
        bins_size = np.arange(1, max_bin + 1) - 0.5
        ax3.hist(pg_sizes_arr, bins=bins_size, alpha=0.7, color="C0", edgecolor="white")
        ax3.axvline(np.mean(pg_sizes_arr), color="red", linestyle="--", linewidth=1.5,
                    label=f"mean={np.mean(pg_sizes_arr):.1f}")
        ax3.axvline(np.median(pg_sizes_arr), color="orange", linestyle=":", linewidth=1.5,
                    label=f"median={np.median(pg_sizes_arr):.0f}")
        ax3.set_xlabel("Requests per Prefix Group")
        ax3.set_ylabel("Count (groups)")
        ax3.set_title(f"Prefix Group Size (n={len(pg_sizes)} groups)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Row 3 right: intra-group request distance distribution
        ax4 = fig.add_subplot(gs[2, 1])
        if intra_distances_s:
            dist_arr = np.array(intra_distances_s)
            p99 = np.percentile(dist_arr, 99)
            bins_dist = np.linspace(0, p99, 60)
            ax4.hist(dist_arr, bins=bins_dist, alpha=0.7, color="C1", edgecolor="white")
            ax4.axvline(np.mean(dist_arr), color="red", linestyle="--", linewidth=1.5,
                        label=f"mean={np.mean(dist_arr):.1f}s")
            ax4.axvline(np.median(dist_arr), color="orange", linestyle=":", linewidth=1.5,
                        label=f"median={np.median(dist_arr):.1f}s")
            ax4.set_xlabel("Time Between Same-Group Requests (s)")
            ax4.set_ylabel("Count")
            ax4.set_title(f"Intra-Group Distance (n={len(intra_distances_s)} pairs)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No groups with >1 request", transform=ax4.transAxes,
                     ha="center", va="center")
            ax4.set_title("Intra-Group Request Distance")

        plot_path = os.path.join(out_dir, "kv_hit_characteristics.pdf")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {plot_path}")

    # Comparison plot if multiple results
    if len(results) > 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle("Comparison: Prefix Hit Ratio & Input Length", fontsize=13)
        colors = ["steelblue", "darkorange", "green", "purple"]
        for i, res in enumerate(results):
            c = colors[i % len(colors)]
            w_t = np.array(res["windows"])
            w_m = np.array(res["w_means"])
            valid = ~np.isnan(w_m)
            ax1.plot(w_t[valid], w_m[valid], linewidth=2, color=c, label=res["name"])

            input_tokens = [r["input_tokens"] for r in res["records"]]
            w_t_inp, w_m_inp = windowed_stats(res["timestamps"], input_tokens, window_s=30)
            w_t_inp = np.array(w_t_inp)
            w_m_inp = np.array(w_m_inp)
            valid_inp = ~np.isnan(w_m_inp)
            ax2.plot(w_t_inp[valid_inp], w_m_inp[valid_inp], linewidth=2, color=c, label=res["name"])

        ax1.set_ylabel("Prefix Hit Ratio (30s mean)")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax2.set_ylabel("Input Tokens (30s mean)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        cmp_path = os.path.join(out_dir, "comparison_timeseries.pdf")
        fig.savefig(cmp_path, dpi=150)
        plt.close(fig)
        print(f"  Saved comparison plot: {cmp_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <workload.jsonl> [workload2.jsonl ...]")
        print(f"  Output files are saved to each workload's parent directory.")
        sys.exit(1)

    workload_paths = [os.path.abspath(p) for p in sys.argv[1:]]
    results = []
    for path in workload_paths:
        name = os.path.basename(os.path.dirname(path))
        r = analyze_workload(name, path)
        results.append(r)

        # Save CSVs and plots to the workload's directory
        out_dir = os.path.dirname(path)
        print(f"\n  Saving outputs to: {out_dir}")
        save_csv(r, out_dir)
        plot_timeseries([r], out_dir)

    # If multiple workloads, print comparison and save comparison plot
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*70}")
        header_fmt = "  {:40s}" + " | {:>12s}" * len(results)
        row_fmt = "  {:40s}" + " | {:>12s}" * len(results)
        print(header_fmt.format("", *[r["name"][:12] for r in results]))
        print(f"  {'-'*40}" + ("-+-" + "-"*12) * len(results))

        hrs = [np.array(r["hit_ratios"]) for r in results]
        metric_rows = [
            ("Requests", [f"{len(r['records'])}" for r in results]),
            ("Duration (s)", [f"{r['records'][-1]['timestamp']/1000:.0f}" for r in results]),
            ("Mean prefix hit ratio", [f"{h.mean():.4f}" for h in hrs]),
            ("Median prefix hit ratio", [f"{np.median(h):.4f}" for h in hrs]),
            ("P90 prefix hit ratio", [f"{np.percentile(h, 90):.4f}" for h in hrs]),
            ("Zero-hit %", [f"{(h==0).mean()*100:.1f}%" for h in hrs]),
            ("Full-hit %", [f"{(h==1.0).mean()*100:.1f}%" for h in hrs]),
        ]
        for label, vals in metric_rows:
            print(row_fmt.format(label, *vals))

        # Save comparison plot to first workload's directory
        plot_timeseries(results, os.path.dirname(workload_paths[0]))
