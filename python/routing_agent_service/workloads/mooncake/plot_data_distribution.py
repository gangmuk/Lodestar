#!/usr/bin/env python3
"""
Multi-workload facet plot: two rows per workload.jsonl, six panels per row.

Per workload (rows 2i, 2i+1):
  Row 1 (distributions and prefix-group structure):
    1) Input token distribution
    2) Output token distribution
    3) Prefix / KV cache hit ratio distribution (temporal trie over hash_ids)
    4) Unique prefix groups in a 30s sliding window (1s step)
    5) Prefix group size distribution (requests per prefix_group)
    6) Intra-group distance (consecutive same-group request spacing, seconds)
  Row 2 (temporal trends and extra distributions):
    1) RPS over time (1s bins)
    2) Average input tokens over time (1s bins)
    3) Average output tokens over time (1s bins)
    4) Input vs Output token scatter
    5) Per-second RPS distribution (histogram)
    6) Hash blocks per request distribution

Usage:
  python3 plot_data_distribution.py [-o OUT.pdf] workload1.jsonl [workload2.jsonl ...]

Default output: workload_distribution.pdf next to this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Large fonts for paper figures
plt.rcParams.update(
    {
        "font.size": 22,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "figure.titlesize": 28,
    }
)


def load_workload(path: str) -> list[dict]:
    """Load workload.jsonl; expand all requests per line; sort by timestamp."""
    entries = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ts = obj["timestamp"]
            for req in obj["requests"]:
                entries.append(
                    {
                        "timestamp_ms": ts,
                        "input_tokens": req["Prompt Length"],
                        "output_tokens": req["Output Length"],
                        "hash_ids": req.get("hash_ids", []),
                        "prefix_group": req.get("prefix_group", ""),
                    }
                )
    entries.sort(key=lambda e: e["timestamp_ms"])
    return entries


def compute_unique_groups_timeseries(
    entries_sorted: list[dict], window_ms: int = 30000, step_ms: int = 1000
):
    """Unique prefix_group count in sliding window."""
    if not entries_sorted:
        return [], []
    t_min = entries_sorted[0]["timestamp_ms"]
    t_max = entries_sorted[-1]["timestamp_ms"]
    n = len(entries_sorted)
    xs, ys = [], []
    left = 0
    right = 0
    group_counts: Counter = Counter()
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


def compute_sliding_windows(entries_sorted: list[dict], hit_ratios: list[float], window: int = 100):
    """Sliding-window (step=1) means over `window` consecutive requests for input tokens,
    output tokens, and the per-request prefix hit ratio.

    x-axis values are the timestamp (seconds, relative to trace start) of the LAST
    request in each window — so each point summarises the most recent `window`
    requests up to that moment, which preserves the temporal-only semantics of
    the trie-based prefix hit ratio.

    Returns (times_s, sw_input, sw_output, sw_hr); all lists are length
    max(0, N - window + 1).
    """
    n = len(entries_sorted)
    if n < window:
        return [], [], [], []
    t0 = entries_sorted[0]["timestamp_ms"]
    inputs = np.array([e["input_tokens"] for e in entries_sorted], dtype=float)
    outputs = np.array([e["output_tokens"] for e in entries_sorted], dtype=float)
    hr = np.array(hit_ratios, dtype=float) if hit_ratios else np.zeros(n)
    kernel = np.ones(window) / window
    sw_in = np.convolve(inputs, kernel, mode="valid").tolist()
    sw_out = np.convolve(outputs, kernel, mode="valid").tolist()
    sw_hr = np.convolve(hr, kernel, mode="valid").tolist()
    times_s = [
        (entries_sorted[i + window - 1]["timestamp_ms"] - t0) / 1000.0
        for i in range(len(sw_in))
    ]
    return times_s, sw_in, sw_out, sw_hr


class TrieNode:
    __slots__ = ("children",)

    def __init__(self):
        self.children = {}


def compute_prefix_hit_ratios(entries_sorted: list[dict]) -> list[float]:
    """Temporal trie prefix hit ratio per request."""
    root = TrieNode()
    hit_ratios = []
    for entry in entries_sorted:
        hids = entry["hash_ids"]
        total_blocks = len(hids)
        node = root
        matched_blocks = 0
        for h in hids:
            if h in node.children:
                matched_blocks += 1
                node = node.children[h]
            else:
                break
        hit_ratio = matched_blocks / total_blocks if total_blocks > 0 else 0.0
        hit_ratios.append(hit_ratio)
        node = root
        for h in hids:
            if h not in node.children:
                node.children[h] = TrieNode()
            node = node.children[h]
    return hit_ratios


def prefix_group_and_intra_distance(entries_sorted: list[dict]):
    """Prefix group sizes and consecutive same-group gaps in seconds."""
    pg_members: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(entries_sorted):
        pg_members[r["prefix_group"]].append(i)

    pg_sizes = [len(indices) for indices in pg_members.values()]
    intra_distances_s = []
    for indices in pg_members.values():
        if len(indices) < 2:
            continue
        for j in range(1, len(indices)):
            dt_s = (
                entries_sorted[indices[j]]["timestamp_ms"]
                - entries_sorted[indices[j - 1]]["timestamp_ms"]
            ) / 1000.0
            intra_distances_s.append(dt_s)
    return pg_sizes, intra_distances_s


def workload_label(path: str) -> str:
    """Short label for a row: parent dir name, or file stem."""
    parent = os.path.basename(os.path.dirname(path))
    if parent:
        return parent
    return os.path.splitext(os.path.basename(path))[0]


def analyze(path: str) -> dict:
    entries = load_workload(path)
    if not entries:
        return {
            "label": workload_label(path),
            "path": path,
            "entries": entries,
            "wl_input": [],
            "wl_output": [],
            "hit_ratios": [],
            "upg_x": [],
            "upg_y": [],
            "pg_sizes": [],
            "intra_distances_s": [],
            "sw_times_s": [],
            "sw_input": [],
            "sw_output": [],
            "sw_hr": [],
        }

    hit_ratios = compute_prefix_hit_ratios(entries)
    upg_x, upg_y = compute_unique_groups_timeseries(entries)
    pg_sizes, intra_distances_s = prefix_group_and_intra_distance(entries)
    sw_times_s, sw_input, sw_output, sw_hr = compute_sliding_windows(entries, hit_ratios, window=100)

    return {
        "label": workload_label(path),
        "path": path,
        "entries": entries,
        "wl_input": [e["input_tokens"] for e in entries],
        "wl_output": [e["output_tokens"] for e in entries],
        "hit_ratios": hit_ratios,
        "upg_x": upg_x,
        "upg_y": upg_y,
        "pg_sizes": pg_sizes,
        "intra_distances_s": intra_distances_s,
        "sw_times_s": sw_times_s,
        "sw_input": sw_input,
        "sw_output": sw_output,
        "sw_hr": sw_hr,
    }


def plot_facets(analyses: list[dict], out_pdf: str) -> None:
    n = len(analyses)
    ncol = 6
    rows_per_workload = 2
    row_h = 5.8
    fig_w = 38
    fig_h = max(1, n) * rows_per_workload * row_h + 1.2
    fig, axes = plt.subplots(
        n * rows_per_workload, ncol, figsize=(fig_w, fig_h), squeeze=False
    )

    col_titles_dist = [
        "Input tokens",
        "Output tokens",
        "Prefix hit ratio",
        "Unique prefix groups (30 s)",
        "Prefix group size",
        "Intra-group distance (s)",
    ]
    col_titles_temp = [
        "Avg input (100-req sliding window)",
        "Avg output (100-req sliding window)",
        "Prefix hit ratio over time (100-req SW)",
        "Input vs Output",
    ]

    for i, data in enumerate(analyses):
        dist_axes = axes[2 * i]
        temp_axes = axes[2 * i + 1]
        wl_in = data["wl_input"]
        wl_out = data["wl_output"]
        hr = np.array(data["hit_ratios"]) if data["hit_ratios"] else np.array([])
        upg_x, upg_y = data["upg_x"], data["upg_y"]
        pg_sizes = data["pg_sizes"]
        intra = data["intra_distances_s"]
        sw_times_s = data["sw_times_s"]
        sw_input = data["sw_input"]
        sw_output = data["sw_output"]
        sw_hr = data["sw_hr"]

        # === Row 1: distributions & prefix-group structure ===

        # 0: input tokens
        ax = dist_axes[0]
        if wl_in:
            bins_in = np.linspace(0, max(wl_in), 60)
            ax.hist(wl_in, bins=bins_in, alpha=0.75, color="C0", density=True, edgecolor="white")
        ax.set_ylabel("Density")
        if i == 0:
            ax.set_title(col_titles_dist[0])
        ax.text(
            0.02,
            0.98,
            data["label"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.92),
        )

        # 1: output tokens
        ax = dist_axes[1]
        if wl_out:
            bins_out = np.linspace(0, max(wl_out), 60)
            ax.hist(wl_out, bins=bins_out, alpha=0.75, color="C1", density=True, edgecolor="white")
        ax.set_ylabel("Density")
        if i == 0:
            ax.set_title(col_titles_dist[1])

        # 2: hit ratio
        ax = dist_axes[2]
        if hr.size:
            bins_hr = np.linspace(0, 1, 51)
            ax.hist(hr, bins=bins_hr, alpha=0.75, color="C2", edgecolor="white", linewidth=0.5)
            ax.axvline(
                float(hr.mean()),
                color="red",
                linestyle="--",
                linewidth=2.0,
                label=f"mean={hr.mean():.3f}",
            )
            ax.axvline(
                float(np.median(hr)),
                color="orange",
                linestyle=":",
                linewidth=2.0,
                label=f"median={np.median(hr):.3f}",
            )
            zero_pct = (hr == 0).mean() * 100
            full_pct = (hr == 1.0).mean() * 100
            ax.text(
                0.97,
                0.95,
                f"zero: {zero_pct:.1f}%\nfull: {full_pct:.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9),
            )
            ax.legend(loc="upper left", fontsize=14)
        ax.set_ylabel("Count")
        ax.set_xlim(-0.02, 1.02)
        if i == 0:
            ax.set_title(col_titles_dist[2])

        # 3: unique prefix groups
        ax = dist_axes[3]
        if upg_x:
            ax.plot(upg_x, upg_y, color="C3", linewidth=2.0)
        ax.set_ylabel("Unique groups")
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_title(col_titles_dist[3])

        # 4: prefix group size
        ax = dist_axes[4]
        if pg_sizes:
            pg_arr = np.array(pg_sizes)
            max_bin = min(int(np.percentile(pg_arr, 99)) + 2, int(pg_arr.max()) + 1)
            bins_size = np.arange(1, max_bin + 1) - 0.5
            ax.hist(pg_arr, bins=bins_size, alpha=0.75, color="C0", edgecolor="white")
            ax.axvline(
                float(np.mean(pg_arr)),
                color="red",
                linestyle="--",
                linewidth=2.0,
                label=f"mean={np.mean(pg_arr):.1f}",
            )
            ax.axvline(
                float(np.median(pg_arr)),
                color="orange",
                linestyle=":",
                linewidth=2.0,
                label=f"med={np.median(pg_arr):.0f}",
            )
            ax.legend(loc="upper right", fontsize=14)
            lo, hi = ax.get_xlim()
            t0 = max(1, int(np.floor(lo)))
            t1 = int(np.ceil(hi))
            ax.set_xticks(np.arange(t0, t1 + 1))
        ax.set_ylabel("Groups")
        ax.set_xlabel("Requests / group")
        if i == 0:
            ax.set_title(col_titles_dist[4])

        # 5: intra-group distance
        ax = dist_axes[5]
        if intra:
            dist_arr = np.array(intra)
            p99 = np.percentile(dist_arr, 99)
            bins_dist = np.linspace(0, max(p99, 1e-6), 60)
            ax.hist(dist_arr, bins=bins_dist, alpha=0.75, color="C1", edgecolor="white")
            ax.axvline(
                float(np.mean(dist_arr)),
                color="red",
                linestyle="--",
                linewidth=2.0,
                label=f"mean={np.mean(dist_arr):.1f}s",
            )
            ax.axvline(
                float(np.median(dist_arr)),
                color="orange",
                linestyle=":",
                linewidth=2.0,
                label=f"med={np.median(dist_arr):.1f}s",
            )
            ax.legend(loc="upper right", fontsize=14)
        else:
            ax.text(
                0.5,
                0.5,
                "No multi-req groups",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=18,
            )
        ax.set_ylabel("Pairs")
        ax.set_xlabel("Δt (s)")
        if i == 0:
            ax.set_title(col_titles_dist[5])

        # === Row 2: 100-req sliding-window trends + Input vs Output ===

        # 0: Avg input over a 100-request sliding window
        ax = temp_axes[0]
        if sw_times_s:
            ax.plot(sw_times_s, sw_input, color="C2", linewidth=1.5)
        ax.set_ylabel("Avg input tokens")
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_title(col_titles_temp[0])

        # 1: Avg output over a 100-request sliding window
        ax = temp_axes[1]
        if sw_times_s:
            ax.plot(sw_times_s, sw_output, color="C3", linewidth=1.5)
        ax.set_ylabel("Avg output tokens")
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_title(col_titles_temp[1])

        # 2: Prefix hit ratio over time
        # Per-request hit ratios come from a temporal trie (each request only sees
        # previously-arrived requests), so plotting them in time order preserves
        # the causal "only-hit-earlier-prompts" semantics. We show the 100-req
        # sliding mean so the trend is readable instead of a noisy point cloud.
        ax = temp_axes[2]
        if sw_times_s:
            ax.plot(sw_times_s, sw_hr, color="C2", linewidth=1.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Hit ratio (100-req mean)")
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_title(col_titles_temp[2])

        # 3: Input vs Output scatter
        ax = temp_axes[3]
        if wl_in and wl_out:
            ax.scatter(wl_in, wl_out, alpha=0.3, s=10, color="C4")
        ax.set_xlabel("Input tokens")
        ax.set_ylabel("Output tokens")
        if i == 0:
            ax.set_title(col_titles_temp[3])

        # 4, 5: unused
        for unused in temp_axes[4:]:
            unused.set_visible(False)

    fig.suptitle("Workload characteristics (two rows per trace)", fontsize=30, y=1.005)
    plt.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-row workload facet plots → PDF.")
    p.add_argument(
        "-o",
        "--output",
        default=os.path.join(SCRIPT_DIR, "workload_distribution.pdf"),
        help="Output PDF path (default: next to this script)",
    )
    p.add_argument("workloads", nargs="+", help="One or more workload.jsonl paths")
    args = p.parse_args()

    paths = [os.path.abspath(x) for x in args.workloads]
    for path in paths:
        if not os.path.isfile(path):
            print(f"Error: not a file: {path}", file=sys.stderr)
            return 1

    analyses = [analyze(path) for path in paths]
    for a in analyses:
        n = len(a["entries"])
        print(f"{a['label']}: {n} requests ({a['path']})")

    out_pdf = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    plot_facets(analyses, out_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
