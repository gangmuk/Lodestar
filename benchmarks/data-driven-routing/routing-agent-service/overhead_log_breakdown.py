#!/usr/bin/env python3
"""
Parse overhead_log lines from routing-agent-service logs and plot breakdowns.

Usage:
  python overhead_log_breakdown.py /path/to/all-routing-agent-service.log.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


OVERHEAD_LINE_RE = re.compile(r"overhead_log: oh, (.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_.]+):\s*(-?\d+(?:\.\d+)?)ms")

# Sentinel values seen in logs for unavailable timings
SENTINEL_VALUES = {-1000.0}

# Exclude total-like entries from category totals to avoid double counting
EXCLUDE_FROM_CATEGORY_TOTAL = ("end_to_end", "total_inference")
WARMUP_RECORDS = 600


def parse_overhead_lines(log_path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = OVERHEAD_LINE_RE.search(line)
            if not match:
                continue
            payload = match.group(1)
            pairs = KV_RE.findall(payload)
            if not pairs:
                continue
            record: Dict[str, float] = {}
            for key, value in pairs:
                try:
                    record[key] = float(value)
                except ValueError:
                    continue
            records.append(record)
    return records


def to_long_df(records: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["record_id"] = np.arange(len(df))
    long_df = df.melt(id_vars=["record_id"], var_name="component", value_name="ms")
    long_df = long_df.dropna(subset=["ms"])
    long_df = long_df[~long_df["ms"].isin(SENTINEL_VALUES)]
    return long_df


def classify_category(component: str) -> str:
    if component.startswith("handle_infer_"):
        return "handle_infer"
    if component.startswith("encode_"):
        return "encode"
    if component.startswith("preprocess_"):
        return "preprocess"
    if component.startswith("infer_from_tensor_"):
        return "infer_from_tensor"
    return "other"


def classify_subgroup(component: str) -> str:
    if "." in component:
        return component.split(".", 1)[0]
    parts = component.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return component


def summarize_components(long_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_df.groupby("component")["ms"]
        .agg(count="count", mean_ms="mean", median_ms="median",
             p95_ms=lambda s: np.percentile(s, 95),
             min_ms="min", max_ms="max")
        .reset_index()
        .sort_values("mean_ms", ascending=False)
    )
    return summary


def summarize_categories(long_df: pd.DataFrame) -> pd.DataFrame:
    def is_excluded(component: str) -> bool:
        return any(component.endswith(suffix) for suffix in EXCLUDE_FROM_CATEGORY_TOTAL)

    filtered = long_df[~long_df["component"].apply(is_excluded)].copy()
    filtered["category"] = filtered["component"].apply(classify_category)
    category_totals = (
        filtered.groupby(["record_id", "category"])["ms"]
        .sum()
        .reset_index()
    )
    category_summary = (
        category_totals.groupby("category")["ms"]
        .agg(count="count", mean_ms="mean", median_ms="median",
             p95_ms=lambda s: np.percentile(s, 95))
        .reset_index()
        .sort_values("mean_ms", ascending=False)
    )
    return category_summary


def summarize_subgroups(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["subgroup"] = df["component"].apply(classify_subgroup)
    df["category"] = df["component"].apply(classify_category)
    subgroup_summary = (
        df.groupby(["subgroup", "category"])["ms"]
        .mean()
        .reset_index()
        .rename(columns={"ms": "mean_ms"})
        .sort_values("mean_ms", ascending=False)
    )
    return subgroup_summary


def plot_breakdown(
    component_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    subgroup_summary: pd.DataFrame,
    output_path: Path,
    top_n_components: int = 15,
    top_n_subgroups: int = 12,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: Top components
    top_components = component_summary.head(top_n_components).copy()
    top_components["category"] = top_components["component"].apply(classify_category)
    sns.barplot(
        data=top_components,
        x="mean_ms",
        y="component",
        hue="category",
        dodge=False,
        ax=axes[0],
        palette="tab10",
    )
    axes[0].set_title("Top Component Overheads (Mean)")
    axes[0].set_xlabel("Mean Overhead (ms)")
    axes[0].set_ylabel("Component")
    axes[0].legend(title="Category", loc="lower right")

    # Panel 2: Category totals
    sns.barplot(
        data=category_summary,
        x="mean_ms",
        y="category",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("Category Totals (Mean of Sum per Record)")
    axes[1].set_xlabel("Mean Overhead (ms)")
    axes[1].set_ylabel("Category")

    # Panel 3: Top subgroups
    top_subgroups = subgroup_summary.head(top_n_subgroups)
    sns.barplot(
        data=top_subgroups,
        x="mean_ms",
        y="subgroup",
        hue="category",
        dodge=False,
        ax=axes[2],
        palette="tab20",
    )
    axes[2].set_title("Top Subgroup Overheads (Mean)")
    axes[2].set_xlabel("Mean Overhead (ms)")
    axes[2].set_ylabel("Subgroup")
    axes[2].legend(title="Category", loc="lower right")

    for ax in axes:
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Overhead Breakdown from routing-agent-service Logs", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


def run_analysis(records: List[Dict[str, float]], outdir: Path, label: str) -> None:
    if not records:
        return

    long_df = to_long_df(records)
    component_summary = summarize_components(long_df)
    category_summary = summarize_categories(long_df)
    subgroup_summary = summarize_subgroups(long_df)

    # Save data tables
    component_summary.to_csv(outdir / f"overhead_component_summary_{label}.csv", index=False)
    category_summary.to_csv(outdir / f"overhead_category_summary_{label}.csv", index=False)
    subgroup_summary.to_csv(outdir / f"overhead_subgroup_summary_{label}.csv", index=False)
    long_df.to_csv(outdir / f"overhead_long_format_{label}.csv", index=False)

    plot_path = outdir / f"overhead_breakdown_{label}.png"
    plot_breakdown(
        component_summary,
        category_summary,
        subgroup_summary,
        plot_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze overhead_log entries and plot breakdowns."
    )
    parser.add_argument("log_path", type=Path, help="Path to log file.")
    args = parser.parse_args()

    log_path = args.log_path
    outdir = log_path.parent

    records = parse_overhead_lines(log_path)
    if not records:
        raise SystemExit(f"No overhead_log entries found in {log_path}")

    warmup_records = records[:WARMUP_RECORDS]
    post_warmup_records = records[WARMUP_RECORDS:]

    run_analysis(records, outdir, "full")
    run_analysis(warmup_records, outdir, "warmup")
    run_analysis(post_warmup_records, outdir, "postwarmup")

    print(f"Parsed {len(records)} overhead_log entries")
    print(f"Warmup records: {len(warmup_records)}")
    print(f"Post-warmup records: {len(post_warmup_records)}")
    print(f"Saved summaries and plots to: {outdir}")


if __name__ == "__main__":
    main()

