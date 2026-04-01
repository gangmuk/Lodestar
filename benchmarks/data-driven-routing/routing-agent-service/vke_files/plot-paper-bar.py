#!/usr/bin/env python3
"""
Plot bar-only gateway comparison pages for paper figures.

This script is a lightweight variant of trendline_plot_from_gateway_log.py:
- Loads routing_strategy_metrics_gateway.csv files
- Groups workloads by category (same workload config, varying RPS)
- Produces ONLY the top-row bar plots (Avg TTFT vs P99 TTFT) into a PDF

Usage:
    python plot-paper-bar.py <base_dir> [--output-dir <output_dir>]
    python plot-paper-bar.py <base_dir> --target-dirs-file <file>
"""

import os
import sys
import argparse
import logging
import re
import subprocess
import concurrent.futures
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from trendline_plot_from_gateway_log import (
    find_gateway_metrics_files,
    merge_gateway_metrics_files,
    group_workloads_by_category,
    _short_group_label,
    generate_policy_colors,
    order_policies,
    extract_routing_policy,
)


def _set_paper_style():
    """Apply matplotlib defaults that are cleaner for paper-ready figures."""
    # Keep output clean: font subsetting can emit verbose INFO logs.
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 20,
            "pdf.fonttype": 42,  # Embed editable TrueType fonts
            "ps.fonttype": 42,
        }
    )


_DISPLAY_NAME_OVERRIDES = {
    "cb_ttft_conv2_tool2-onlinelearning_0": "Quicksilver-without-onlinelearning",
    "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1": "Quicksilver",
}

# Policies matching any of these substrings are excluded from plots by default.
_DEFAULT_EXCLUDE_PATTERNS = [
    "cb_ttft_conv2_tool2-onlinelearning_1",
]


def _compact_policy_label(policy: str, max_len: int = 56) -> str:
    """Make long policy names more readable in legends."""
    if policy in _DISPLAY_NAME_OVERRIDES:
        return _DISPLAY_NAME_OVERRIDES[policy]
    text = (
        policy.replace("contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_", "cb_ttft_")
        .replace("prefix_hit_threshold_or_least_request_threshold_", "prefix_hit_or_lr_")
        .replace("_conversation_2", "_conv2")
        .replace("_toolagent_2", "_tool2")
    )
    if len(text) <= max_len:
        return text
    return f"{text[:max_len - 1]}…"


def _order_policies_for_paper(policies):
    """Apply paper-specific ordering, especially for contextual-bandit variants."""
    base = order_policies(policies)

    def _is_cb(policy: str) -> bool:
        pl = policy.lower()
        return ("contextual_bandit" in pl) or pl.startswith("cb_ttft_")

    def _cb_rank(policy: str):
        pl = policy.lower()
        is_random = "random" in pl
        is_ol0 = "onlinelearning_0" in pl
        is_ol1 = "onlinelearning_1" in pl

        # Requested order:
        #   1) *-onlinelearning_0
        #   2) *-onlinelearning_1
        #   3) cb_ttft_random-onlinelearning_1 (last among CB)
        if is_ol0:
            return (0, pl)
        if is_ol1 and not is_random:
            return (1, pl)
        if is_ol1 and is_random:
            return (2, pl)
        return (3, pl)

    non_cb = [p for p in base if not _is_cb(p)]
    cb = [p for p in base if _is_cb(p)]
    cb_sorted = sorted(cb, key=_cb_rank)
    return non_cb + cb_sorted


def _mean_std_positive(series):
    vals = [v for v in series.dropna().tolist() if v > 0]
    if not vals:
        return 0.0, 0.0
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean_v, std_v


def _is_quicksilver_policy(policy: str) -> bool:
    p = str(policy).lower()
    return ("quicksilver" in p) or ("contextual_bandit" in p) or p.startswith("cb_ttft_")


def _is_quicksilver_main_policy(policy: str) -> bool:
    # Main Quicksilver corresponds to the onlinelearning_1 random-init CB policy.
    p = str(policy).lower()
    return ("random-onlinelearning_1" in p) or (_compact_policy_label(policy) == "Quicksilver")


def _is_prefix_cache_1_policy(policy: str) -> bool:
    return "prefix_cache_1" in str(policy).lower()


def _lighten(color, factor=1.35):
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    t = max(0.0, min(factor - 1.0, 1.0))
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def _extract_rps_int(workload: str):
    """Extract single integer RPS from workload path (e.g., rps12-benchmark -> 12)."""
    m = re.search(r"rps(\d+)", workload, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _remove_rps_segment_local(workload: str) -> str:
    """Remove the rps* path segment from workload path."""
    parts = workload.split("/")
    filtered = [p for p in parts if not re.match(r"^rps[\d,]+", p, re.IGNORECASE)]
    return "/".join(filtered)


def _merge_cb_conv2_tool2_onlinelearning_policies(df):
    """Merge conv2/tool2 contextual-bandit policies per onlinelearning_k into one label."""
    df = df.copy()
    rp = df["routing_policy"].fillna("").astype(str).str.lower()
    merge_mask = (
        rp.str.contains("contextual_bandit")
        & rp.str.contains("onlinelearning_")
        & (rp.str.contains("conversation_2") | rp.str.contains("toolagent_2"))
    )
    merged_count = int(merge_mask.sum())
    if merged_count == 0:
        return df, 0

    def _rewrite_label(policy: str) -> str:
        policy_s = str(policy)
        m = re.search(r"(onlinelearning_\d+)", policy_s, re.IGNORECASE)
        suffix = m.group(1).lower() if m else "onlinelearning"
        return f"cb_ttft_conv2_tool2-{suffix}"

    df.loc[merge_mask, "routing_policy"] = (
        df.loc[merge_mask, "routing_policy"].apply(_rewrite_label)
    )
    return df, merged_count


def export_paper_csv(df, output_dir, exclude_patterns=None):
    """Export clean TTFT summary CSV for plotted policies/RPS from aggregated CSVs."""
    import pandas as pd

    exclude_patterns = exclude_patterns or []
    df_plot = df.copy()
    if exclude_patterns:
        df_plot = df_plot[
            ~df_plot["routing_policy"].apply(
                lambda p: any(pat in p for pat in exclude_patterns)
            )
        ]
    if df_plot.empty:
        print("No rows to export after exclusion filters.")
        return None

    # Build export rows directly from aggregated metrics rows.
    rows = []
    for _, row in df_plot.iterrows():
        workload = row.get("workload", "")
        routing_policy = row.get("routing_policy", "")
        if not isinstance(workload, str) or not workload.strip():
            continue
        if not isinstance(routing_policy, str) or not routing_policy.strip():
            continue
        rows.append(
            {
                "workload_group": _short_group_label(_remove_rps_segment_local(workload)),
                "workload": workload,
                "rps": _extract_rps_int(workload),
                "routing_policy": routing_policy,
                "strategy_full_name": row.get("strategy_full_name", ""),
                "avg_ttft": row.get("avg_ttft", np.nan),
                "p99_ttft": row.get("p99_ttft", np.nan),
                "p999_ttft": row.get("p999_ttft", np.nan),
                "num_requests": row.get("num_requests", np.nan),
            }
        )

    if not rows:
        print("No valid rows available for CSV export.")
        return None

    runs_df = pd.DataFrame(rows)
    for c in ["avg_ttft", "p99_ttft", "p999_ttft", "num_requests"]:
        runs_df[c] = pd.to_numeric(runs_df[c], errors="coerce")
    runs_df = runs_df[runs_df["avg_ttft"] > 0]

    if runs_df.empty:
        print("No positive TTFT rows available for CSV export.")
        return None

    # Final table: one row per (routing_policy, rps) (within workload group/workload).
    agg = (
        runs_df.groupby(["workload_group", "workload", "rps", "routing_policy"], dropna=False)
        .agg(
            runs=("strategy_full_name", "nunique"),
            mean_ttft=("avg_ttft", "mean"),
            p99_ttft=("p99_ttft", "mean"),
            p999_ttft=("p999_ttft", "mean"),
            num_requests=("num_requests", "sum"),
        )
        .reset_index()
    )
    # p50/p90 TTFT are unavailable from aggregated gateway metrics without raw per-request logs.
    agg["p50_ttft"] = np.nan
    agg["p90_ttft"] = np.nan
    agg["stats_source"] = "aggregated_gateway_csv"
    agg = agg[
        [
            "workload_group",
            "workload",
            "rps",
            "routing_policy",
            "runs",
            "num_requests",
            "mean_ttft",
            "p50_ttft",
            "p90_ttft",
            "p99_ttft",
            "p999_ttft",
            "stats_source",
        ]
    ]

    metric_cols = ["mean_ttft", "p99_ttft", "p999_ttft"]
    for col in metric_cols:
        agg[col] = agg[col].round(2)

    agg = agg.sort_values(
        by=["workload_group", "rps", "routing_policy"], kind="stable"
    ).reset_index(drop=True)

    out_path = os.path.join(output_dir, "paper_bar_ttft_summary.csv")
    agg.to_csv(out_path, index=False)
    print(
        f"Saved paper CSV to {out_path} ({len(agg)} rows). "
        "Note: p50/p90 require per-request logs and are left blank."
    )
    return out_path


def _plot_bars_twin_y_paper(ax, df_group, rps_workload_pair, policies, policy_colors, annotate_values=False):
    """Paper-focused bar plotting: cleaner visuals and less subplot clutter."""
    _, workload = rps_workload_pair
    avg_vals, avg_errs, p99_vals, p99_errs, p999_vals, p999_errs, colors = [], [], [], [], [], [], []

    for policy in policies:
        rows = df_group[
            (df_group["workload"] == workload) & (df_group["routing_policy"] == policy)
        ]
        avg_mean, avg_std = _mean_std_positive(rows["avg_ttft"]) if "avg_ttft" in rows.columns else (0, 0)
        p99_mean, p99_std = _mean_std_positive(rows["p99_ttft"]) if "p99_ttft" in rows.columns else (0, 0)
        p999_mean, p999_std = _mean_std_positive(rows["p999_ttft"]) if "p999_ttft" in rows.columns else (0, 0)
        avg_vals.append(avg_mean)
        avg_errs.append(avg_std)
        p99_vals.append(p99_mean)
        p99_errs.append(p99_std)
        p999_vals.append(p999_mean)
        p999_errs.append(p999_std)
        colors.append(policy_colors.get(policy, "#7f7f7f"))

    if not any(v > 0 for v in avg_vals + p99_vals + p999_vals):
        ax.set_visible(False)
        return None

    ax2 = ax.twinx()
    n = len(policies)
    # Make bars much thicker and keep substantial spacing between policy groups.
    x = np.arange(n) * 2.35
    width = 0.84
    p99_colors = [_lighten(c, 1.45) for c in colors]
    p999_colors = [_lighten(c, 1.75) for c in colors]

    b1 = ax.bar(
        x - width,
        avg_vals,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        yerr=[e if e > 0 else np.nan for e in avg_errs],
        capsize=2,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "color": "black"},
        zorder=3,
    )
    b2 = ax2.bar(
        x,
        p99_vals,
        width=width,
        color=p99_colors,
        edgecolor="black",
        linewidth=0.6,
        yerr=[e if e > 0 else np.nan for e in p99_errs],
        capsize=2,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "color": "black"},
        zorder=2,
    )
    b3 = ax2.bar(
        x + width,
        p999_vals,
        width=width,
        color=p999_colors,
        edgecolor="black",
        linewidth=0.6,
        yerr=[e if e > 0 else np.nan for e in p999_errs],
        capsize=2,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "color": "black"},
        zorder=1,
    )

    max_avg = max((v + e for v, e in zip(avg_vals, avg_errs)), default=1.0)
    max_tail = max(
        [v + e for v, e in zip(p99_vals, p99_errs)] + [v + e for v, e in zip(p999_vals, p999_errs)],
        default=1.0,
    )
    ylim_scale = 1.34 if annotate_values else 1.22
    ax.set_ylim(0, max(1.0, max_avg * ylim_scale))
    ax2.set_ylim(0, max(1.0, max_tail * ylim_scale))

    if annotate_values:
        prefix_idx = next((i for i, p in enumerate(policies) if _is_prefix_cache_1_policy(p)), None)
        quicksilver_idx = next((i for i, p in enumerate(policies) if _is_quicksilver_main_policy(p)), None)
        ratio_label = None
        ratio_label_p99 = None
        if (
            prefix_idx is not None
            and quicksilver_idx is not None
            and avg_vals[prefix_idx] > 0
            and avg_vals[quicksilver_idx] > 0
        ):
            ratio_label = f"{avg_vals[quicksilver_idx] / avg_vals[prefix_idx]:.2f}"
        if (
            prefix_idx is not None
            and quicksilver_idx is not None
            and p99_vals[prefix_idx] > 0
            and p99_vals[quicksilver_idx] > 0
        ):
            ratio_label_p99 = f"{p99_vals[quicksilver_idx] / p99_vals[prefix_idx]:.2f}"

        for i, (bar, v, e) in enumerate(zip(b1, avg_vals, avg_errs)):
            if v > 0:
                label = f"{v:.0f}"
                if (
                    quicksilver_idx is not None
                    and ratio_label is not None
                    and i == quicksilver_idx
                ):
                    label = f"{v:.0f}({ratio_label})"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + e + max_avg * 0.015,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=90,
                )
        for i, (bar, v, e) in enumerate(zip(b2, p99_vals, p99_errs)):
            if v > 0:
                label = f"{v:.0f}"
                if (
                    quicksilver_idx is not None
                    and ratio_label_p99 is not None
                    and i == quicksilver_idx
                ):
                    label = f"{v:.0f}({ratio_label_p99})"
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + e + max_tail * 0.015,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=90,
                    color="dimgray",
                )
        for bar, v, e in zip(b3, p999_vals, p999_errs):
            if v > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + e + max_tail * 0.015,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=90,
                    color="dimgray",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n)])
    ax.set_xlabel("Policy index")
    ax.set_ylabel("Avg TTFT (ms)")
    ax2.set_ylabel("P99/P999 TTFT (ms)", color="dimgray")
    ax2.tick_params(axis="y", labelcolor="dimgray")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    return ax2


def plot_bar_only(df, output_dir, exclude_patterns=None, annotate_values=False):
    """Generate bar-only pages and save to a single PDF."""
    exclude_patterns = exclude_patterns or []

    # Defensive cleanup: some merged CSVs may contain NaN/non-string workload rows.
    df = df.copy()
    df = df[df["workload"].notna()]
    df = df[df["workload"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    if df.empty:
        print("No valid workload rows available after filtering.")
        return

    workloads = df["workload"].unique().tolist()
    groups = group_workloads_by_category(workloads)

    sorted_group_keys = sorted(groups.keys())
    all_policies = _order_policies_for_paper(df["routing_policy"].unique())
    policies = [p for p in all_policies if not any(pat in p for pat in exclude_patterns)]

    if exclude_patterns:
        excluded = [p for p in all_policies if p not in policies]
        if excluded:
            print(f"  Excluded policies: {excluded}")

    _set_paper_style()
    policy_colors = generate_policy_colors(policies)
    pdf_path = os.path.join(output_dir, "paper_bar_from_gateway_log.pdf")

    with PdfPages(pdf_path) as pdf:
        for gk in sorted_group_keys:
            rps_workload_pairs = groups[gk]
            group_workloads = [w for _, w in rps_workload_pairs]
            df_group = df[df["workload"].isin(group_workloads)]

            if "avg_ttft" not in df_group.columns and "p99_ttft" not in df_group.columns:
                continue

            group_policies = [p for p in policies if p in df_group["routing_policy"].values]
            if not group_policies:
                continue

            n_rows = len(rps_workload_pairs)
            n_policies = len(group_policies)

            # Layout tuned for publication readability.
            ncol_policy_legend = min(4, max(2, int(np.ceil(np.sqrt(n_policies)))))
            n_legend_rows = int(np.ceil(n_policies / ncol_policy_legend))
            # One RPS per row for better readability with thicker bars.
            fig_height = (3.9 * n_rows) + 2.4 + max(0, n_legend_rows - 1) * 0.30
            fig_width = 14.5

            fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height))
            if n_rows == 1:
                axes = [axes]

            short_label = _short_group_label(gk)
            fig.suptitle(f"TTFT Comparison Across RPS - {short_label}", y=0.985)

            for ri, rps_pair in enumerate(rps_workload_pairs):
                ax2 = _plot_bars_twin_y_paper(
                    axes[ri],
                    df_group,
                    rps_pair,
                    group_policies,
                    policy_colors,
                    annotate_values=annotate_values,
                )
                rps, _ = rps_pair
                axes[ri].set_title(f"RPS {rps}")

            from matplotlib.patches import Patch

            # Shared metric-style legend
            metric_handles = [
                Patch(facecolor="#666666", edgecolor="black", label="Avg TTFT (left axis)"),
                Patch(facecolor="#bbbbbb", edgecolor="black", label="P99 TTFT (right axis, lighter shade)"),
                Patch(facecolor="#d9d9d9", edgecolor="black", label="P999 TTFT (right axis, lightest shade)"),
            ]
            fig.legend(
                handles=metric_handles,
                loc="upper center",
                ncol=3,
                bbox_to_anchor=(0.5, 0.955),
                framealpha=0.9,
                title="Bar semantics",
            )

            # Shared policy-color legend
            legend_labels = [f"{i + 1}. {_compact_policy_label(p)}" for i, p in enumerate(group_policies)]
            legend_colors = [policy_colors.get(p, "#7f7f7f") for p in group_policies]
            legend_handles = [
                Patch(facecolor=c, edgecolor="black", label=lbl)
                for c, lbl in zip(legend_colors, legend_labels)
            ]
            if legend_handles:
                fig.legend(
                    handles=legend_handles,
                    loc="lower center",
                    ncol=ncol_policy_legend,
                    bbox_to_anchor=(0.5, 0.02),
                    framealpha=0.9,
                    title="Routing policies",
                )

            # Leave compact top space and bottom space for policy legend.
            bottom_frac = 0.095 + max(0, n_legend_rows - 1) * 0.03
            fig.tight_layout(rect=[0.03, bottom_frac, 0.995, 0.925])
            fig.subplots_adjust(hspace=0.38)
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)

    print(f"Saved bar-only PDF to {pdf_path}")


def _run_compare_routing_strategies(target_dirs):
    """Run compare_routing_strategies.py in parallel on each target directory.

    Mirrors the Step-1 behaviour of compare2: processes all dirs concurrently
    and raises RuntimeError if any subprocess fails.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_routing_strategies.py")

    def _run_one(d):
        print(f"  [compare] Processing: {d}")
        result = subprocess.run(
            [sys.executable, script, d, "--from-request", "1000"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [compare] ERROR in {d}:\n{result.stderr.strip()}")
            return d, False
        return d, True

    print(f"\n=== Running compare_routing_strategies.py on {len(target_dirs)} dir(s) (parallel) ===")
    failed = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(_run_one, d): d for d in target_dirs}
        for future in concurrent.futures.as_completed(futures):
            d, ok = future.result()
            if not ok:
                failed.append(d)

    if failed:
        raise RuntimeError(
            f"compare_routing_strategies.py failed for {len(failed)} dir(s):\n"
            + "\n".join(f"  {d}" for d in failed)
        )
    print("=== compare_routing_strategies.py finished ===\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot bar-only gateway routing comparison from "
            "routing_strategy_metrics_gateway.csv"
        )
    )
    parser.add_argument(
        "base_dir",
        help="Base directory for output (and recursive search if --target-dirs-file not provided)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for plots (default: base_dir)",
    )
    parser.add_argument(
        "--target-dirs-file",
        "-t",
        default=None,
        help="File containing list of target directories (one per line).",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="+",
        default=[],
        metavar="PATTERN",
        help=(
            "Exclude routing policies whose name contains any of these substrings. "
            "Example: --exclude e2e_latency_negative_linear"
        ),
    )
    parser.add_argument(
        "--value-labels",
        action="store_true",
        default=False,
        help="Deprecated alias; labels are shown by default.",
    )
    parser.add_argument(
        "--no-value-labels",
        action="store_true",
        default=False,
        help="Disable numeric labels above bars.",
    )
    parser.add_argument(
        "--include-non-paper",
        action="store_true",
        default=False,
        help="Include CSV files under paths containing 'non-used-for-paper'.",
    )
    parser.add_argument(
        "--include-multi-rps",
        action="store_true",
        default=False,
        help="Include multi-RPS benchmarks (e.g., rps9,11-benchmark).",
    )

    args = parser.parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir if args.output_dir else base_dir
    os.makedirs(output_dir, exist_ok=True)

    target_dirs = None
    if args.target_dirs_file:
        if os.path.exists(args.target_dirs_file):
            with open(args.target_dirs_file, "r", encoding="utf-8") as f:
                target_dirs = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            print(f"Using {len(target_dirs)} target directories from {args.target_dirs_file}")
        else:
            print(f"Error: Target dirs file not found: {args.target_dirs_file}")
            sys.exit(1)

    # Discover which directories will be used so we can regenerate their CSVs
    # before loading them.  When target_dirs is not provided, find them by
    # scanning for existing CSVs (or rps*-benchmark subdirs) under base_dir.
    if target_dirs is not None:
        dirs_to_process = list(target_dirs)
    else:
        # Discover rps*-benchmark/without_bitsandbytes dirs under base_dir.
        import glob as _glob
        dirs_to_process = sorted(
            d for d in _glob.glob(
                os.path.join(base_dir, "rps*-benchmark", "without_bitsandbytes"),
                recursive=False,
            )
            if os.path.isdir(d)
        )
        if not dirs_to_process:
            # Fall back: derive dirs from any pre-existing CSV paths.
            _existing = find_gateway_metrics_files(base_dir, None)
            dirs_to_process = sorted({os.path.dirname(f) for f in _existing})

    if dirs_to_process:
        try:
            _run_compare_routing_strategies(dirs_to_process)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
    else:
        print("Warning: no target directories found; skipping compare_routing_strategies.py.")

    files = find_gateway_metrics_files(base_dir, target_dirs)
    if not files:
        print("No routing_strategy_metrics_gateway.csv files found")
        sys.exit(1)

    if not args.include_non_paper:
        before = len(files)
        files = [
            f for f in files if "non-used-for-paper" not in f.replace("\\", "/")
        ]
        removed = before - len(files)
        if removed > 0:
            print(f"Filtered out {removed} non-paper metrics file(s).")

    if not args.include_multi_rps:
        before = len(files)
        files = [
            f
            for f in files
            if not re.search(r"/rps\d+(?:,\d+)+-benchmark/", f.replace("\\", "/"))
        ]
        removed = before - len(files)
        if removed > 0:
            print(f"Filtered out {removed} multi-RPS metrics file(s).")

    if not files:
        print("No metrics files left after filtering.")
        sys.exit(1)

    print(f"Found {len(files)} metrics files")
    df = merge_gateway_metrics_files(files)
    if df is None or len(df) == 0:
        print("No data to process")
        sys.exit(1)

    # Match trendline script behavior for policy parsing.
    df["routing_policy"] = df["strategy_full_name"].apply(extract_routing_policy)
    df, merged_cb = _merge_cb_conv2_tool2_onlinelearning_policies(df)
    if merged_cb > 0:
        print(
            f"Merged {merged_cb} CB conv2/tool2 onlinelearning rows "
            "into shared policy bars per onlinelearning_k."
        )

    exclude_patterns = list(dict.fromkeys(_DEFAULT_EXCLUDE_PATTERNS + args.exclude))
    if exclude_patterns:
        print(f"Excluding from plots policies matching: {exclude_patterns}")

    # Export final numeric table (same policy filters as plotted figure).
    export_paper_csv(
        df,
        output_dir,
        exclude_patterns=exclude_patterns,
    )

    plot_bar_only(
        df,
        output_dir,
        exclude_patterns=exclude_patterns,
        annotate_values=not args.no_value_labels,
    )
    print("Done!")


if __name__ == "__main__":
    main()
