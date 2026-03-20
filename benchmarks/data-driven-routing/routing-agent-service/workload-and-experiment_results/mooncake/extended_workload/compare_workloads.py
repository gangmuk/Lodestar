#!/usr/bin/env python3
"""
Compare original and extended workload traces side by side.

Usage:
    python3 compare_workloads.py <original.jsonl> <extended.jsonl>
"""

import json
import sys
import os
import statistics
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

TOKENS_PER_BLOCK = None  # auto-detected


def load_workload(path):
    entries = []
    with open(path) as f:
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
    entries.sort(key=lambda e: e["timestamp_ms"])
    return entries


def compute_prefix_hits(entries, tokens_per_block):
    class TrieNode:
        __slots__ = ['children']
        def __init__(self):
            self.children = {}

    root = TrieNode()
    hit_ratios = []
    for e in entries:
        node = root
        matched = 0
        for h in e["hash_ids"]:
            if h in node.children:
                matched += 1
                node = node.children[h]
            else:
                break
        hr = (matched * tokens_per_block) / e["input_tokens"] if e["input_tokens"] > 0 else 0
        hit_ratios.append(hr)
        node = root
        for h in e["hash_ids"]:
            if h not in node.children:
                node.children[h] = TrieNode()
            node = node.children[h]
    return hit_ratios


def compute_group_stats(entries):
    pg = defaultdict(list)
    for e in entries:
        pg[e["prefix_group"]].append(e)
    sizes = [len(v) for v in pg.values()]

    intra_iats = []
    for members in pg.values():
        if len(members) < 2:
            continue
        ms = sorted(members, key=lambda e: e["timestamp_ms"])
        for i in range(len(ms) - 1):
            intra_iats.append(ms[i+1]["timestamp_ms"] - ms[i]["timestamp_ms"])
    return sizes, intra_iats


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <original.jsonl> <extended.jsonl>")
        sys.exit(1)

    orig_path = sys.argv[1]
    ext_path = sys.argv[2]
    out_dir = os.path.dirname(os.path.abspath(ext_path))
    out_file = os.path.join(out_dir, "workload_comparison.png")
    label = os.path.basename(out_dir)

    print(f"Loading {orig_path}...")
    orig = load_workload(orig_path)
    print(f"  {len(orig)} requests")

    print(f"Loading {ext_path}...")
    ext = load_workload(ext_path)
    print(f"  {len(ext)} requests")

    # Auto-detect tokens per block
    global TOKENS_PER_BLOCK
    TOKENS_PER_BLOCK = orig[0]["input_tokens"] / len(orig[0]["hash_ids"]) if orig[0]["hash_ids"] else 50
    print(f"  Tokens/block: {TOKENS_PER_BLOCK:.0f}")

    # Compute everything
    print("Computing prefix hit ratios...")
    orig_hits = compute_prefix_hits(orig, TOKENS_PER_BLOCK)
    ext_hits = compute_prefix_hits(ext, TOKENS_PER_BLOCK)

    orig_inputs = [e["input_tokens"] for e in orig]
    ext_inputs = [e["input_tokens"] for e in ext]
    orig_outputs = [e["output_tokens"] for e in orig]
    ext_outputs = [e["output_tokens"] for e in ext]

    orig_times = [e["timestamp_ms"] for e in orig]
    ext_times = [e["timestamp_ms"] for e in ext]
    orig_iats = [orig_times[i+1] - orig_times[i] for i in range(len(orig_times)-1)]
    ext_iats = [ext_times[i+1] - ext_times[i] for i in range(len(ext_times)-1)]

    orig_gsizes, orig_giats = compute_group_stats(orig)
    ext_gsizes, ext_giats = compute_group_stats(ext)

    # Rolling prefix hit
    window = 200
    def rolling_mean(vals, w):
        out = []
        for i in range(w, len(vals)):
            out.append((i, statistics.mean(vals[i-w:i])))
        return out
    orig_roll = rolling_mean(orig_hits, window)
    ext_roll = rolling_mean(ext_hits, window)

    # --- Compute group liveness ---
    def compute_active_groups(entries, window_ms=10000, step_ms=5000):
        pg_times = defaultdict(list)
        for e in entries:
            pg_times[e["prefix_group"]].append(e["timestamp_ms"])
        times = sorted(e["timestamp_ms"] for e in entries)
        t_min, t_max = int(times[0]), int(times[-1])
        xs, ys = [], []
        for t in range(t_min, t_max, step_ms):
            active = sum(1 for pg, ts in pg_times.items() if any(t - window_ms <= tt <= t for tt in ts))
            xs.append((t - t_min) / 1000)
            ys.append(active)
        return xs, ys

    print("Computing active prefix groups over time...")
    orig_active_x, orig_active_y = compute_active_groups(orig)
    ext_active_x, ext_active_y = compute_active_groups(ext)

    # --- Plot: 3 rows x 4 cols ---
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))

    # Row 1: distributions
    # 1. Input tokens
    ax = axes[0, 0]
    bins = np.linspace(0, max(max(orig_inputs), max(ext_inputs)), 50)
    ax.hist(orig_inputs, bins=bins, alpha=0.6, density=True, label=f"original (n={len(orig_inputs)})")
    ax.hist(ext_inputs, bins=bins, alpha=0.6, density=True, label=f"extended (n={len(ext_inputs)})")
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Density")
    ax.set_title("Input Token Distribution")
    ax.legend()

    # 2. Output tokens
    ax = axes[0, 1]
    bins = np.linspace(0, max(max(orig_outputs), max(ext_outputs)), 50)
    ax.hist(orig_outputs, bins=bins, alpha=0.6, density=True, label="original")
    ax.hist(ext_outputs, bins=bins, alpha=0.6, density=True, label="extended")
    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Density")
    ax.set_title("Output Token Distribution")
    ax.legend()

    # 3. Global inter-arrival time
    ax = axes[0, 2]
    p99 = max(np.percentile(orig_iats, 99), np.percentile(ext_iats, 99))
    bins = np.linspace(0, p99, 60)
    ax.hist(orig_iats, bins=bins, alpha=0.6, density=True, label="original")
    ax.hist(ext_iats, bins=bins, alpha=0.6, density=True, label="extended")
    ax.set_xlabel("Inter-arrival Time (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Global Inter-arrival Time")
    ax.legend()

    # 4. Prefix hit ratio histogram
    ax = axes[0, 3]
    bins = np.linspace(0, 1, 40)
    ax.hist(orig_hits, bins=bins, alpha=0.6, density=True, label=f"original (mean={np.mean(orig_hits):.3f})")
    ax.hist(ext_hits, bins=bins, alpha=0.6, density=True, label=f"extended (mean={np.mean(ext_hits):.3f})")
    ax.set_xlabel("Prefix Hit Ratio")
    ax.set_ylabel("Density")
    ax.set_title("Prefix Hit Ratio Distribution")
    ax.legend()

    # Row 2: temporal and group structure
    # 5. Prefix hit ratio over time
    ax = axes[1, 0]
    ax.scatter(range(len(orig_hits)), orig_hits, alpha=0.1, s=6, color="C0", label="original (per-req)")
    ax.scatter(range(len(ext_hits)), ext_hits, alpha=0.1, s=6, color="C1", label="extended (per-req)")
    if orig_roll:
        ax.plot([r[0] for r in orig_roll], [r[1] for r in orig_roll],
                color="C0", alpha=0.9, linewidth=1.5, label=f"original (rolling w={window})")
    if ext_roll:
        ax.plot([r[0] for r in ext_roll], [r[1] for r in ext_roll],
                color="C1", alpha=0.9, linewidth=1.5, label=f"extended (rolling w={window})")
    ax.set_xlabel("Request Index")
    ax.set_ylabel("Prefix Hit Ratio")
    ax.set_title("Prefix Hit Ratio Over Time")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    # 6. Group size distribution
    ax = axes[1, 1]
    max_size = min(max(max(orig_gsizes), max(ext_gsizes)), 30)
    bins = np.arange(0.5, max_size + 1.5, 1)
    ax.hist(orig_gsizes, bins=bins, alpha=0.6, density=True, label="original")
    ax.hist(ext_gsizes, bins=bins, alpha=0.6, density=True, label="extended")
    ax.set_xlabel("Group Size (requests)")
    ax.set_ylabel("Density")
    ax.set_title("Prefix Group Size Distribution")
    ax.set_xlim(0, max_size + 1)
    ax.legend()

    # 7. Intra-group inter-arrival time
    ax = axes[1, 2]
    if orig_giats and ext_giats:
        p99 = max(np.percentile(orig_giats, 99), np.percentile(ext_giats, 99))
        bins = np.linspace(0, p99, 50)
        ax.hist(orig_giats, bins=bins, alpha=0.6, density=True, label="original")
        ax.hist(ext_giats, bins=bins, alpha=0.6, density=True, label="extended")
    ax.set_xlabel("Intra-group IAT (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Intra-group Inter-arrival Time")
    ax.legend()

    # 8. RPS over time
    ax = axes[1, 3]
    def rps_timeseries(times_ms, bin_s=10):
        times_s = [t / 1000 for t in times_ms]
        bins = np.arange(min(times_s), max(times_s) + bin_s, bin_s)
        counts, edges = np.histogram(times_s, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, counts / bin_s

    oc, orps = rps_timeseries(orig_times)
    ec, erps = rps_timeseries(ext_times)
    ax.plot(oc, orps, alpha=0.7, label="original", color="C0")
    ax.plot(ec, erps, alpha=0.7, label="extended", color="C1")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPS")
    ax.set_title("RPS Over Time")
    ax.legend()

    # Row 3: prefix group temporal behavior
    # 9. Active prefix groups over time (10s window)
    ax = axes[2, 0]
    ax.plot(orig_active_x, orig_active_y, alpha=0.7, color="C0", label="original")
    ax.plot(ext_active_x, ext_active_y, alpha=0.7, color="C1", label="extended")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Active Groups")
    ax.set_title("Active Prefix Groups Over Time (10s window)")
    ax.legend()

    # 10. New group creation rate over time
    def new_group_rate(entries, bin_s=60):
        first_seen = {}
        for e in entries:
            pg = e["prefix_group"]
            if pg not in first_seen:
                first_seen[pg] = e["timestamp_ms"]
        times_s = [t / 1000 for t in first_seen.values()]
        if not times_s:
            return [], []
        bins_arr = np.arange(min(times_s), max(times_s) + bin_s, bin_s)
        counts, edges = np.histogram(times_s, bins=bins_arr)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, counts

    ax = axes[2, 1]
    oc, ocounts = new_group_rate(orig)
    ec, ecounts = new_group_rate(ext)
    ax.plot(oc, ocounts, alpha=0.7, color="C0", label="original")
    ax.plot(ec, ecounts, alpha=0.7, color="C1", label="extended")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("New Groups per 60s")
    ax.set_title("New Prefix Group Creation Rate")
    ax.legend()

    # 11. Group lifespan distribution
    def group_lifespans(entries):
        pg_times = defaultdict(list)
        for e in entries:
            pg_times[e["prefix_group"]].append(e["timestamp_ms"])
        spans = []
        for pg, ts in pg_times.items():
            if len(ts) >= 2:
                spans.append((max(ts) - min(ts)) / 1000)  # seconds
        return spans

    ax = axes[2, 2]
    orig_spans = group_lifespans(orig)
    ext_spans = group_lifespans(ext)
    if orig_spans and ext_spans:
        p99 = max(np.percentile(orig_spans, 99), np.percentile(ext_spans, 99))
        bins = np.linspace(0, p99, 50)
        ax.hist(orig_spans, bins=bins, alpha=0.6, density=True, label="original")
        ax.hist(ext_spans, bins=bins, alpha=0.6, density=True, label="extended")
    ax.set_xlabel("Group Lifespan (s)")
    ax.set_ylabel("Density")
    ax.set_title("Prefix Group Lifespan Distribution")
    ax.legend()

    # 12. Unique groups per 1s window (co-arrival mixing)
    def groups_per_second(entries):
        bins_map = defaultdict(set)
        t0 = entries[0]["timestamp_ms"]
        for e in entries:
            bins_map[(e["timestamp_ms"] - t0) // 1000].add(e["prefix_group"])
        return [len(v) for v in bins_map.values()]

    ax = axes[2, 3]
    orig_gps = groups_per_second(orig)
    ext_gps = groups_per_second(ext)
    max_gps = max(max(orig_gps), max(ext_gps))
    bins = np.arange(0.5, max_gps + 1.5, 1)
    ax.hist(orig_gps, bins=bins, alpha=0.6, density=True, label=f"original (mean={np.mean(orig_gps):.1f})")
    ax.hist(ext_gps, bins=bins, alpha=0.6, density=True, label=f"extended (mean={np.mean(ext_gps):.1f})")
    ax.set_xlabel("Unique Groups per 1s Window")
    ax.set_ylabel("Density")
    ax.set_title("Co-arrival Group Mixing")
    ax.legend()

    plt.suptitle(f"Workload Comparison: Original vs Extended ({label})", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
