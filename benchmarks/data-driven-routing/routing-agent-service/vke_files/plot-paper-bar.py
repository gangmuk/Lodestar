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


# -- Font size variables (bump all by changing _FS_BUMP) --
_FS_BUMP = 4
FS_DEFAULT = 13 + _FS_BUMP        # matplotlib font.size
FS_AXES_TITLE = 16 + _FS_BUMP     # axes.titlesize
FS_AXES_LABEL = 13 + _FS_BUMP + 5 # axes.labelsize (+2+3)
FS_YLABEL = FS_AXES_LABEL + 2
FS_XLABEL = FS_AXES_LABEL + 3
FS_TICK = 11 + _FS_BUMP + 4 + 3   # xtick / ytick labelsize (+2+2, then +3)
FS_LEGEND = 11 + _FS_BUMP + 12 + 2    # legend.fontsize (+4+4+4, then +2)
FS_FIG_TITLE = 20 + _FS_BUMP + 10 # figure.titlesize / suptitle (+10)
FS_BAR_ANNOTATION = 10 + _FS_BUMP + 1 # value labels on bars (+1)
FS_SUBPLOT_TITLE = 11 + _FS_BUMP + 7 + 3  # per-subplot title (+5+2, then +3)
FS_LEGEND_BOX = 10 + _FS_BUMP + 12 + 2 # legend boxes in single-page mode (+4+4+4, then +2)


def _set_paper_style():
    """Apply matplotlib defaults that are cleaner for paper-ready figures."""
    # Keep output clean: font subsetting can emit verbose INFO logs.
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FS_DEFAULT,
            "axes.titlesize": FS_AXES_TITLE,
            "axes.labelsize": FS_AXES_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "figure.titlesize": FS_FIG_TITLE,
            "pdf.fonttype": 42,  # Embed editable TrueType fonts
            "ps.fonttype": 42,
        }
    )


_DISPLAY_NAME_OVERRIDES = {
    "cb_ttft_conv2_tool2-onlinelearning_0": "Quicksilver offline train only",
    "cb_ttft_conv2_tool2-onlinelearning_1": "Quicksilver",
    "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1": "Quicksilver",
    "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random": "Quicksilver",
    "least_request": "Least request",
    "prefix_cache_1": "Prefix-and-load-aware",
}

# Policies matching any of these substrings are excluded from plots by default.
_DEFAULT_EXCLUDE_PATTERNS = []


def _compact_policy_label(policy: str, max_len: int = 56) -> str:
    """Make long policy names more readable in legends."""
    if policy in _DISPLAY_NAME_OVERRIDES:
        return _DISPLAY_NAME_OVERRIDES[policy]
    # Pattern-based: any CB random-init variant with onlinelearning_1 is Quicksilver.
    pl = policy.lower()
    if "contextual_bandit" in pl and "random" in pl and "onlinelearning_1" in pl:
        return "Quicksilver"
    # prefix_hit_threshold_or_least_request_threshold_XX -> Prefix hit or least request (tau=XX)
    m = re.match(r"prefix_hit_threshold_or_least_request_threshold_(.+)", policy)
    if m:
        return f"Prefix hit or least request (tau={m.group(1)})"
    text = (
        policy.replace("contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_", "cb_ttft_")
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
    return mean_v, 0.0


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


def _merge_cb_random_quicksilver_policies(df):
    """Merge all CB-random Quicksilver variants (incl. ablation variants) per onlinelearning_k.

    Variants like `..._random_K2_iter1-onlinelearning_1`,
    `..._random_all_data_maxnumtrains_7_K2-onlinelearning_1`,
    `..._random_fifo_replay_..._onlinelearning_1`, etc. collapse into a single
    `quicksilver-onlinelearning_N` label so they render as one bar with an
    error bar from the mean/std across variants.
    """
    df = df.copy()
    rp = df["routing_policy"].fillna("").astype(str).str.lower()
    merge_mask = (
        rp.str.contains("contextual_bandit")
        & rp.str.contains("random")
        & rp.str.contains("onlinelearning_")
        & ~rp.str.contains("conversation_2")
        & ~rp.str.contains("toolagent_2")
        & ~rp.str.contains("no_candidate_filtering")
    )
    merged_count = int(merge_mask.sum())
    if merged_count == 0:
        return df, 0

    def _rewrite_label(policy: str) -> str:
        m = re.search(r"(onlinelearning_\d+)", str(policy), re.IGNORECASE)
        suffix = m.group(1).lower() if m else "onlinelearning"
        # Keep "contextual_bandit" + "random" in the label so existing color
        # categorization and Quicksilver legend-label rules still apply.
        return f"contextual_bandit_random-{suffix}"

    df.loc[merge_mask, "routing_policy"] = (
        df.loc[merge_mask, "routing_policy"].apply(_rewrite_label)
    )
    return df, merged_count


def _merge_cb_conv2_tool2_onlinelearning_policies(df):
    """Merge conv2/tool2 contextual-bandit policies per onlinelearning_k into one label."""
    df = df.copy()
    rp = df["routing_policy"].fillna("").astype(str).str.lower()
    merge_mask = (
        rp.str.contains("contextual_bandit")
        & rp.str.contains("onlinelearning_")
        & (rp.str.contains("conversation_2") | rp.str.contains("toolagent_2"))
        & ~rp.str.contains("no_candidate_filtering")
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
        if "strategy_full_name" in df_plot.columns:
            df_plot = df_plot[
                ~df_plot["strategy_full_name"].apply(
                    lambda s: any(pat in s for pat in exclude_patterns) if isinstance(s, str) else False
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
    avg_vals, avg_errs, p99_vals, p99_errs, colors = [], [], [], [], []

    for policy in policies:
        rows = df_group[
            (df_group["workload"] == workload) & (df_group["routing_policy"] == policy)
        ]
        avg_mean, avg_std = _mean_std_positive(rows["avg_ttft"]) if "avg_ttft" in rows.columns else (0, 0)
        p99_mean, p99_std = _mean_std_positive(rows["p99_ttft"]) if "p99_ttft" in rows.columns else (0, 0)
        avg_vals.append(avg_mean)
        avg_errs.append(avg_std)
        p99_vals.append(p99_mean)
        p99_errs.append(p99_std)
        colors.append(policy_colors.get(policy, "#7f7f7f"))

    if not any(v > 0 for v in avg_vals + p99_vals):
        ax.set_visible(False)
        return None

    ax2 = ax.twinx()
    n = len(policies)
    # Two bars per policy group (Avg / P99), centred around each tick.
    x = np.arange(n) * 1.85
    width = 0.62
    p99_colors = [_lighten(c, 1.45) for c in colors]

    b1 = ax.bar(
        x - width / 2,
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
        x + width / 2,
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

    max_avg = max((v + e for v, e in zip(avg_vals, avg_errs)), default=1.0)
    max_right = max((v + e for v, e in zip(p99_vals, p99_errs)), default=1.0)
    ylim_scale = 1.34 if annotate_values else 1.22
    ax.set_ylim(0, max(1.0, max_avg * ylim_scale))
    ax2.set_ylim(0, max(1.0, max_right * ylim_scale))

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
            if v <= 0:
                continue
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = v + e + max_avg * 0.015
            val_str = f"{v:.0f}"
            if quicksilver_idx is not None and ratio_label is not None and i == quicksilver_idx:
                val_str = f"{v:.0f} ({ratio_label})"
            ax.text(x_pos, y_pos, val_str, ha="center", va="bottom",
                    fontsize=FS_BAR_ANNOTATION, rotation=90,
                    fontweight="bold" if (quicksilver_idx is not None and i == quicksilver_idx and ratio_label) else "normal")

        for i, (bar, v, e) in enumerate(zip(b2, p99_vals, p99_errs)):
            if v <= 0:
                continue
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = v + e + max_right * 0.015
            val_str = f"{v:.0f}"
            if quicksilver_idx is not None and ratio_label_p99 is not None and i == quicksilver_idx:
                val_str = f"{v:.0f} ({ratio_label_p99})"
            ax2.text(x_pos, y_pos, val_str, ha="center", va="bottom",
                     fontsize=FS_BAR_ANNOTATION, rotation=90,
                     fontweight="bold" if (quicksilver_idx is not None and i == quicksilver_idx and ratio_label_p99) else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n)])
    ax.set_xlabel("Policy index", fontsize=FS_XLABEL)
    ax.set_ylabel("Avg TTFT (ms)", fontsize=FS_YLABEL)
    ax2.set_ylabel("P99 TTFT (ms)", color="black", fontsize=FS_YLABEL)
    ax2.tick_params(axis="y", labelcolor="black")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    return ax2


def plot_bar_only(df, output_dir, exclude_patterns=None, annotate_values=False,
                  single_page=False, single_row=False, ncol=3,
                  no_bar_semantics_legend=False, no_routing_policies_legend=False,
                  no_suptitle=False, output_pdf=None):
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
    # Filter out policies matching exclude patterns (check routing_policy name).
    policies = [p for p in all_policies if not any(pat in p for pat in exclude_patterns)]
    # Also drop individual rows whose strategy_full_name matches an exclude pattern.
    if exclude_patterns and "strategy_full_name" in df.columns:
        mask = df["strategy_full_name"].apply(
            lambda s: any(pat in s for pat in exclude_patterns) if isinstance(s, str) else False
        )
        if mask.any():
            print(f"  Excluded {mask.sum()} row(s) by strategy_full_name match.")
            df = df[~mask]
    # Remove policies that have no remaining rows after filtering.
    remaining_policies = set(df["routing_policy"].unique())
    policies = [p for p in policies if p in remaining_policies]

    if exclude_patterns:
        excluded = [p for p in all_policies if p not in policies]
        if excluded:
            print(f"  Excluded policies: {excluded}")

    _set_paper_style()
    policy_colors = generate_policy_colors(policies)
    pdf_path = output_pdf if output_pdf else os.path.join(output_dir, "paper_bar_from_gateway_log.pdf")

    if single_page:
        _plot_bar_single_page(
            df, groups, sorted_group_keys, policies, policy_colors,
            annotate_values, pdf_path, single_row=single_row, ncol=ncol,
            no_bar_semantics_legend=no_bar_semantics_legend,
            no_routing_policies_legend=no_routing_policies_legend,
            no_suptitle=no_suptitle,
        )
        return

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

            n_cols = len(rps_workload_pairs)
            n_policies = len(group_policies)

            # Layout tuned for publication readability.
            ncol_policy_legend = min(4, max(2, int(np.ceil(np.sqrt(n_policies)))))
            n_legend_rows = int(np.ceil(n_policies / ncol_policy_legend))
            fig_height = 5.70 + max(0, n_legend_rows - 1) * 0.32
            fig_width = max(11.5, 6.60 * n_cols)

            fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height))
            if n_cols == 1:
                axes = [axes]

            short_label = _short_group_label(gk)
            if not no_suptitle:
                fig.suptitle(f"TTFT Comparison Across RPS - {short_label}", y=0.985)

            for ci, rps_pair in enumerate(rps_workload_pairs):
                ax2 = _plot_bars_twin_y_paper(
                    axes[ci],
                    df_group,
                    rps_pair,
                    group_policies,
                    policy_colors,
                    annotate_values=annotate_values,
                )
                rps, _ = rps_pair
                axes[ci].set_title(f"RPS {rps}")
                if ci > 0:
                    axes[ci].set_ylabel("")
                if ax2 is not None and ci < n_cols - 1:
                    ax2.set_ylabel("")

            from matplotlib.patches import Patch

            # Shared metric-style legend
            if not no_bar_semantics_legend:
                metric_handles = [
                    Patch(facecolor="#666666", edgecolor="black", label="Avg TTFT (left axis)"),
                    Patch(facecolor="#bbbbbb", edgecolor="black", label="P99 TTFT (right axis, lighter shade)"),
                ]
                fig.legend(
                    handles=metric_handles,
                    loc="upper center",
                    ncol=2,
                    bbox_to_anchor=(0.5, 0.955),
                    framealpha=0.9,
                    # title="Bar semantics",
                )

            # Shared policy-color legend
            if not no_routing_policies_legend:
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
                        # title="Routing policies",
                    )

            # Leave compact top space and bottom space for policy legend.
            bottom_frac = 0.135 + max(0, n_legend_rows - 1) * 0.035
            fig.tight_layout(rect=[0.02, bottom_frac, 0.995, 0.885])
            fig.subplots_adjust(wspace=0.40)
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)

    print(f"Saved bar-only PDF to {pdf_path}")


def _plot_bar_single_page(df, groups, sorted_group_keys, policies, policy_colors,
                          annotate_values, pdf_path, single_row=False, ncol=3,
                          no_bar_semantics_legend=False, no_routing_policies_legend=False,
                          no_suptitle=False):
    """All workload categories on a single page: one row per category, columns per RPS."""
    import pandas as pd
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec

    # Collect valid groups and their data.
    valid_groups = []
    for gk in sorted_group_keys:
        rps_workload_pairs = groups[gk]
        group_workloads = [w for _, w in rps_workload_pairs]
        df_group = df[df["workload"].isin(group_workloads)]
        if "avg_ttft" not in df_group.columns and "p99_ttft" not in df_group.columns:
            continue
        group_policies = [p for p in policies if p in df_group["routing_policy"].values]
        if not group_policies:
            continue
        valid_groups.append((gk, rps_workload_pairs, df_group, group_policies))

    # Order: Conversation first, ToolAgent second, Synthetic third.
    # For gangmuk-prefix: 71% → 47% → 28% → 9% → Mixed.
    _ROW_ORDER = {
        "conversation": 0, "toolagent": 1, "synthetic": 2,
        "sharingratio71": 0, "sharingratio47": 1,
        "sharingratio28": 2, "sharingratio9%": 3,
        "mixedsharingratio": 4,
    }

    def _row_sort_key(item):
        gk = item[0].lower()
        for key, rank in _ROW_ORDER.items():
            if key in gk:
                return rank
        return 99

    valid_groups.sort(key=_row_sort_key)

    if not valid_groups:
        print("No valid groups for single-page plot.")
        return

    # Flatten all groups into a single row: each group+RPS becomes one column.
    # Store per-column group keys for titles.
    _single_row_gks = None
    if single_row:
        all_pairs = []
        _single_row_gks = []
        combined_df = pd.concat([dg for _, _, dg, _ in valid_groups], ignore_index=True)
        combined_policies = [p for p in policies if p in combined_df["routing_policy"].values]
        for gk, rps_workload_pairs, df_group, group_policies in valid_groups:
            for rps_pair in rps_workload_pairs:
                all_pairs.append(rps_pair)
                _single_row_gks.append(gk)
        valid_groups = [("__single_row__", all_pairs, combined_df, combined_policies)]

    n_rows = len(valid_groups)
    n_cols = max(len(pairs) for _, pairs, _, _ in valid_groups)
    n_policies = len(policies)

    # Figure sizing: balance compactness with readability.
    row_height = 3.2 if single_row else 4.2
    col_width = max(5.5, 1.3 * n_policies)
    ncol_policy_legend = ncol
    n_legend_rows = int(np.ceil(n_policies / ncol_policy_legend))
    # Top area: build up from visible elements only.
    top_header_inches = 0.2  # base top padding
    if not no_suptitle:
        top_header_inches += 1.2
    if not no_bar_semantics_legend:
        top_header_inches += 1.4
    if not no_routing_policies_legend:
        top_header_inches += 1.4 + max(0, n_legend_rows - 1) * 0.45
    fig_width = col_width * n_cols + 1.2
    fig_height = row_height * n_rows + top_header_inches + 0.3

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Reserve fractional space at top for title + legends.
    top_header_frac = top_header_inches / fig_height
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        top=1.0 - top_header_frac, bottom=0.03,
        left=0.07, right=0.96,
        hspace=0.45, wspace=0.65 if single_row else 0.30,
    )

    # Bar-semantics legend: just below title.
    if not no_bar_semantics_legend:
        metric_handles = [
            Patch(facecolor="#666666", edgecolor="black", label="Avg TTFT (left axis)"),
            Patch(facecolor="#bbbbbb", edgecolor="black", label="P99 TTFT (right axis, lighter shade)"),
        ]
        metric_legend_y = 1.0 - 1.0 / fig_height
        leg1 = fig.legend(
            handles=metric_handles,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, metric_legend_y),
            framealpha=0.9,
            fontsize=FS_LEGEND_BOX,
        )

    # Policy-color legend: below bar-semantics with clear gap, above plots.
    if not no_routing_policies_legend:
        legend_labels = [f"{i + 1}. {_compact_policy_label(p)}" for i, p in enumerate(policies)]
        legend_colors = [policy_colors.get(p, "#7f7f7f") for p in policies]
        legend_handles = [
            Patch(facecolor=c, edgecolor="black", label=lbl)
            for c, lbl in zip(legend_colors, legend_labels)
        ]
        if legend_handles:
            # Offset from top: shrink when title / bar-semantics legend are hidden.
            _policy_legend_top_offset = 1.7
            if no_suptitle:
                _policy_legend_top_offset -= 1.0
            if no_bar_semantics_legend:
                _policy_legend_top_offset -= 1.6
            _policy_legend_top_offset = max(0.2, _policy_legend_top_offset)
            policy_legend_y = 1.0 - _policy_legend_top_offset / fig_height
            leg2 = fig.legend(
                handles=legend_handles,
                loc="upper center",
                ncol=ncol_policy_legend,
                bbox_to_anchor=(0.5, policy_legend_y),
                framealpha=0.9,
                # title="Routing policies",
                fontsize=FS_LEGEND_BOX,
            )
            leg2.get_title().set_fontsize(FS_LEGEND_BOX)

    for ri, (gk, rps_workload_pairs, df_group, group_policies) in enumerate(valid_groups):
        for ci, rps_pair in enumerate(rps_workload_pairs):
            ax = fig.add_subplot(gs[ri, ci])
            ax2 = _plot_bars_twin_y_paper(
                ax, df_group, rps_pair, group_policies, policy_colors,
                annotate_values=annotate_values,
            )
            rps, _ = rps_pair
            # In single_row mode, use per-column group key for the title.
            col_gk = _single_row_gks[ci] if _single_row_gks else gk
            short_label = _short_group_label(col_gk)
            clean_label = short_label.replace("mooncake/", "")
            _TITLE_MAP = {
                "conversation-2-extended-ver1": "Conversation",
                "synthetic_3x-numtokens_100": "Synthetic",
                "toolagent-2-extended-ver1": "ToolAgent",
            }
            clean_label = _TITLE_MAP.get(clean_label, clean_label)
            ax.set_title(f"{clean_label} — RPS {rps}", fontsize=FS_SUBPLOT_TITLE)
            if ci > 0:
                ax.set_ylabel("")
            if ax2 is not None and ci < len(rps_workload_pairs) - 1:
                ax2.set_ylabel("")
        # Hide unused columns in this row.
        for ci in range(len(rps_workload_pairs), n_cols):
            ax = fig.add_subplot(gs[ri, ci])
            ax.set_visible(False)

    with PdfPages(pdf_path) as pdf_out:
        pdf_out.savefig(fig, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved single-page bar PDF to {pdf_path}")


def _run_compare_routing_strategies(target_dirs, from_request=1000, append=False):
    """Run compare_routing_strategies.py in parallel on each target directory.

    Mirrors the Step-1 behaviour of compare2: processes all dirs concurrently
    and raises RuntimeError if any subprocess fails.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_routing_strategies.py")

    def _run_one(d):
        print(f"  [compare] Processing: {d}")
        cmd = [sys.executable, script, d, "--from-request", str(from_request)]
        if append:
            cmd.append("--append")
        result = subprocess.run(
            cmd,
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
        "--search-dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help=(
            "Recursively search for CSV files only under these directories "
            "(instead of searching the entire base_dir)."
        ),
    )
    parser.add_argument(
        "--output-pdf",
        default=None,
        metavar="FILENAME",
        help="Output PDF filename (default: paper_bar_from_gateway_log.pdf). Relative to output-dir.",
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
        "--rps",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Only include these RPS values (e.g., --rps 6 8). If not given, include all.",
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
    parser.add_argument(
        "--single-page",
        action="store_true",
        default=False,
        help="Combine all workload categories into a single-page PDF (one row per category).",
    )
    parser.add_argument(
        "--ncol",
        type=int,
        default=3,
        metavar="N",
        help="Number of columns in the policy legend (default: 3).",
    )
    parser.add_argument(
        "--single-row",
        action="store_true",
        default=False,
        help="With --single-page, flatten all workloads into a single row (one column per workload+RPS).",
    )
    parser.add_argument(
        "--no-bar-semantics-legend",
        action="store_true",
        default=False,
        help="Hide the bar-semantics legend (Avg TTFT / P99 TTFT).",
    )
    parser.add_argument(
        "--no-suptitle",
        action="store_true",
        default=False,
        help="Hide the main figure title (suptitle).",
    )
    parser.add_argument(
        "--no-routing-policies-legend",
        action="store_true",
        default=False,
        help="Hide the routing-policies color legend.",
    )
    parser.add_argument(
        "--from-request",
        type=int,
        default=1000,
        metavar="N",
        help=(
            "Skip the first N requests when computing metrics "
            "(passed to compare_routing_strategies.py). Default: 1000."
        ),
    )
    parser.add_argument(
        "--run-compare-routing-strategies",
        type=int,
        default=1,
        metavar="{0,1}",
        help=(
            "Whether to run compare_routing_strategies.py before plotting. "
            "1 (default): regenerate routing_strategy_metrics_gateway.csv from raw logs. "
            "0: skip and use existing CSV files as-is."
        ),
    )
    parser.add_argument(
        "--compare-append",
        type=int,
        default=0,
        metavar="{0,1}",
        help=(
            "Forward --append to compare_routing_strategies.py. "
            "1: append to existing CSV. 0 (default): overwrite."
        ),
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
    # before loading them.
    if target_dirs is not None:
        dirs_to_process = list(target_dirs)
    else:
        import glob as _glob
        # When --search-dirs is given, only look inside those directories.
        # Otherwise fall back to base_dir.
        _search_roots = args.search_dirs if args.search_dirs else [base_dir]
        dirs_to_process = []
        for root in _search_roots:
            # If root is already a without_bitsandbytes dir, use it directly.
            if os.path.basename(root) == "without_bitsandbytes":
                if os.path.isdir(root):
                    dirs_to_process.append(root)
            # If root is already a rps*-benchmark dir, look for without_bitsandbytes inside.
            elif re.search(r"rps[\d,]+-benchmark$", os.path.basename(root)):
                wb = os.path.join(root, "without_bitsandbytes")
                if os.path.isdir(wb):
                    dirs_to_process.append(wb)
            else:
                # Glob recursively for rps*-benchmark/without_bitsandbytes.
                dirs_to_process.extend(
                    d for d in _glob.glob(
                        os.path.join(root, "**", "rps*-benchmark", "without_bitsandbytes"),
                        recursive=True,
                    )
                    if os.path.isdir(d)
                )
        dirs_to_process = sorted(set(dirs_to_process))

    # Skip empty dirs — compare_routing_strategies.py exits 1 when it finds no logs.
    _before_empty = len(dirs_to_process)
    dirs_to_process = [d for d in dirs_to_process if os.listdir(d)]
    _empty_removed = _before_empty - len(dirs_to_process)
    if _empty_removed > 0:
        print(f"Skipped {_empty_removed} empty dir(s).")

    # Filter dirs by --rps if given.
    if args.rps:
        _rps_set = set(args.rps)
        before = len(dirs_to_process)
        dirs_to_process = [
            d for d in dirs_to_process
            if any(re.search(rf"/rps{r}-benchmark/", d) for r in _rps_set)
        ]
        removed = before - len(dirs_to_process)
        if removed > 0:
            print(f"Filtered out {removed} dir(s) not matching --rps {args.rps}")

    if args.run_compare_routing_strategies:
        if dirs_to_process:
            try:
                _run_compare_routing_strategies(
                    dirs_to_process,
                    from_request=args.from_request,
                    append=bool(args.compare_append),
                )
            except RuntimeError as exc:
                print(f"ERROR: {exc}")
                sys.exit(1)
        else:
            print("Warning: no target directories found; skipping compare_routing_strategies.py.")
    else:
        print("Skipping compare_routing_strategies.py (--run-compare-routing-strategies 0).")

    if args.search_dirs:
        # Only load CSVs directly in the specified dirs (no recursion into subdirs).
        files = find_gateway_metrics_files(None, args.search_dirs)
    else:
        files = find_gateway_metrics_files(base_dir, target_dirs)
    if not files:
        print("No routing_strategy_metrics_gateway.csv files found")
        sys.exit(1)

    if args.rps:
        _rps_set = set(args.rps)
        before = len(files)
        files = [
            f for f in files
            if any(re.search(rf"/rps{r}-benchmark/", f) for r in _rps_set)
        ]
        removed = before - len(files)
        if removed > 0:
            print(f"Filtered out {removed} CSV file(s) not matching --rps {args.rps}")

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
    df, merged_qs = _merge_cb_random_quicksilver_policies(df)
    if merged_qs > 0:
        print(
            f"Merged {merged_qs} CB-random Quicksilver rows (incl. ablation variants) "
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
        single_page=args.single_page,
        single_row=args.single_row,
        ncol=args.ncol,
        no_bar_semantics_legend=args.no_bar_semantics_legend,
        no_routing_policies_legend=args.no_routing_policies_legend,
        no_suptitle=args.no_suptitle,
        output_pdf=args.output_pdf,
    )
    print("Done!")


if __name__ == "__main__":
    main()
