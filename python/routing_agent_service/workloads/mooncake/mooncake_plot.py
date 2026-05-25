#!/usr/bin/env python3
"""
Multi-workload facet plot for raw Mooncake traces.

Each row is one trace file. Panels per row:
  1) Input token distribution
  2) Output token distribution
  3) Prefix / KV cache hit ratio distribution (temporal trie over hash_ids)
  4) Intra-group distance (consecutive same-group request spacing, seconds)

Prefix groups are derived from hash_ids: two requests belong to the same
group if they share the first MIN_PREFIX_MATCH hash_ids (matching from the
beginning). Requests with fewer hash_ids are ungrouped.

Usage:
  python3 mooncake_plot.py [-o OUT.pdf] trace1.jsonl [trace2.jsonl ...]

Default: plots all three Mooncake traces next to this script.
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
import matplotlib.ticker as mticker
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Minimum number of matching hash_ids from the beginning to count as shared prefix.
# Used for: prefix group key, trie hit ratio threshold.
MIN_PREFIX_MATCH = 2

plt.rcParams.update(
    {
        "font.size": 34,
        "axes.titlesize": 44,
        "axes.labelsize": 38,
        "xtick.labelsize": 36,
        "ytick.labelsize": 36,
        "legend.fontsize": 32,
        "figure.titlesize": 42,
    }
)

textbox_fontsize = 36
main_title_fontsize = 58

def load_workload(path: str) -> list[dict]:
    """Load raw Mooncake trace (flat jsonl, one request per line). Sort by timestamp."""
    entries = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            hids = obj.get("hash_ids", [])
            # prefix group = first MIN_PREFIX_MATCH hash_ids
            if len(hids) >= MIN_PREFIX_MATCH:
                pg = "_".join(str(h) for h in hids[:MIN_PREFIX_MATCH])
            else:
                pg = ""
            entries.append(
                {
                    "timestamp_ms": obj["timestamp"],
                    "input_tokens": obj["input_length"],
                    "output_tokens": obj["output_length"],
                    "hash_ids": hids,
                    "prefix_group": pg,
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


class TrieNode:
    __slots__ = ("children",)

    def __init__(self):
        self.children = {}


def compute_prefix_hit_ratios(
    entries_sorted: list[dict], min_match: int = MIN_PREFIX_MATCH
) -> list[float]:
    """Temporal trie prefix hit ratio per request.

    Only counts a hit when >= min_match hash_ids match from the beginning.
    """
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
        # Require at least min_match blocks to count as a prefix hit
        if matched_blocks < min_match:
            matched_blocks = 0
        hit_ratio = matched_blocks / total_blocks if total_blocks > 0 else 0.0
        hit_ratios.append(hit_ratio)
        node = root
        for h in hids:
            if h not in node.children:
                node.children[h] = TrieNode()
            node = node.children[h]
    return hit_ratios


def prefix_group_and_intra_distance(entries_sorted: list[dict]):
    """Prefix group sizes and consecutive same-group gaps (in number of requests between)."""
    pg_members: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(entries_sorted):
        if r["prefix_group"]:
            pg_members[r["prefix_group"]].append(i)

    pg_sizes = [len(indices) for indices in pg_members.values()]
    intra_distances = []
    for indices in pg_members.values():
        if len(indices) < 2:
            continue
        for j in range(1, len(indices)):
            # number of other requests between consecutive same-group requests
            intra_distances.append(indices[j] - indices[j - 1] - 1)
    return pg_sizes, intra_distances


def workload_label(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    # Strip common prefixes for brevity
    for prefix in ("Mooncake_", "mooncake_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    for suffix in ("_trace",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


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
            "intra_distances": [],
        }

    hit_ratios = compute_prefix_hit_ratios(entries)
    upg_x, upg_y = compute_unique_groups_timeseries(entries)
    pg_sizes, intra_distances = prefix_group_and_intra_distance(entries)

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
        "intra_distances": intra_distances,
    }


def plot_facets(analyses: list[dict], out_pdf: str) -> None:
    n = len(analyses)
    ncol = 4
    cell_size = 7  # square subplots, slightly larger with fewer columns
    fig_w = ncol * cell_size
    fig_h = max(1, n) * cell_size + 1.2
    fig, axes = plt.subplots(n, ncol, figsize=(fig_w, fig_h), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_aspect("auto")
            ax.set_box_aspect(1)  # force square

    col_titles = [
        "Input tokens",
        "Output tokens",
        "Prefix hit ratio",
        "Prefix reuse distance",
    ]

    for i, data in enumerate(analyses):
        row_axes = axes[i]
        wl_in = data["wl_input"]
        wl_out = data["wl_output"]
        hr = np.array(data["hit_ratios"]) if data["hit_ratios"] else np.array([])
        pg_sizes = data["pg_sizes"]
        intra = data["intra_distances"]

        # 0: input
        ax = row_axes[0]
        if wl_in:
            bins_in = np.linspace(0, 100000, 60)
            ax.hist(wl_in, bins=bins_in, alpha=0.75, color="C0", edgecolor="white")
            in_arr = np.array(wl_in)
            ax.axvline(float(in_arr.mean()), color="red", linestyle="--", linewidth=2.0, label=f"mean={in_arr.mean()/1000:.1f}K")
            ax.legend(loc="upper center")
        ax.set_xlim(0, 100000)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
        ax.set_ylabel("Count")
        if i == n - 1:
            ax.set_xlabel("Tokens")
        if i == 0:
            ax.set_title(col_titles[0])
        ax.annotate(
            data["label"],
            xy=(-0.55, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=main_title_fontsize,
        )

        # 1: output
        ax = row_axes[1]
        if wl_out:
            bins_out = np.linspace(0, 1000, 60)
            ax.hist(wl_out, bins=bins_out, alpha=0.75, color="C1", edgecolor="white")
            out_arr = np.array(wl_out)
            ax.axvline(float(out_arr.mean()), color="red", linestyle="--", linewidth=2.0, label=f"mean={out_arr.mean():.0f}")
            ax.legend(loc="upper right")
        ax.set_xlim(0, 1000)
        if i == n - 1:
            ax.set_xlabel("Tokens")
        if i == 0:
            ax.set_title(col_titles[1])

        # 2: hit ratio histogram
        ax = row_axes[2]
        if hr.size:
            bins_hr = np.linspace(0, 1, 51)
            ax.hist(
                hr,
                bins=bins_hr,
                alpha=0.75,
                color="C2",
                edgecolor="white",
                linewidth=0.5,
            )
            ax.axvline(float(hr.mean()), color="red", linestyle="--", linewidth=2.0, label=f"mean={hr.mean():.2f}")
            ax.legend(loc="upper left")
        if i == n - 1:
            ax.set_xlabel("Hit ratio")
        ax.set_xlim(-0.02, 1.02)
        if i == 0:
            ax.set_title(col_titles[2])

        # 3: intra-group distance
        ax = row_axes[3]
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
                label=f"mean={np.mean(dist_arr):.0f}",
            )
            ax.legend(loc="upper right")
        else:
            ax.text(0.5, 0.5, "No multi-req groups", transform=ax.transAxes, ha="center", va="center", fontsize=18)
        if i == n - 1:
            ax.set_xlabel("Requests between")
        if i == 0:
            ax.set_title(col_titles[3])

    plt.tight_layout(w_pad=0.1, h_pad=0.1)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


def main() -> int:
    default_traces = [
        os.path.join(_SCRIPT_DIR, "Mooncake_conversation_trace.jsonl"),
        os.path.join(_SCRIPT_DIR, "Mooncake_toolagent_trace.jsonl"),
        os.path.join(_SCRIPT_DIR, "Mooncake_synthetic_trace.jsonl"),
    ]

    p = argparse.ArgumentParser(description="Multi-row Mooncake trace facet plots → PDF.")
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output PDF path (default: mooncake_workload_summary.pdf next to this script)",
    )
    p.add_argument(
        "workloads",
        nargs="*",
        help="One or more trace .jsonl paths (default: all 3 Mooncake traces)",
    )
    args = p.parse_args()

    paths = [os.path.abspath(x) for x in args.workloads] if args.workloads else default_traces
    for path in paths:
        if not os.path.isfile(path):
            print(f"Error: not a file: {path}", file=sys.stderr)
            return 1

    analyses = [analyze(path) for path in paths]
    for a in analyses:
        n = len(a["entries"])
        pg_count = len(a["pg_sizes"])
        print(f"{a['label']}: {n} requests, {pg_count} prefix groups ({a['path']})")

    out_pdf = args.output if args.output else os.path.join(_SCRIPT_DIR, "mooncake_workload_summary.pdf")
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    plot_facets(analyses, out_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
