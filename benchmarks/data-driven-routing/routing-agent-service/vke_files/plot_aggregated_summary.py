import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import sibling module when running from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Reuse the triple-axis plotting utility and routing policy constants
import compare_routing_strategies as crs


# def get_strategy_priority(routing_policy: str, strategy_name: str):
#     """Define ordering across strategies/categories to keep rows consistent."""
#     rp = (routing_policy or "").lower()
#     if crs.rl_naive_routing in rp:
#         return (0, strategy_name)
#     if crs.e2e_latency_predictor_routing in rp:
#         return (1, strategy_name)
#     if crs.ttft_latency_predictor_routing in rp:
#         return (2, strategy_name)
#     if crs.avg_tpot_latency_predictor_routing in rp:
#         return (3, strategy_name)
#     if crs.prefix_cache_1_routing in rp:
#         return (4, strategy_name)
#     if crs.prefix_cache_2_routing in rp:
#         return (5, strategy_name)
#     if crs.preble_routing in rp:
#         return (6, strategy_name)
#     if crs.random_routing in rp:
#         return (7, strategy_name)
#     return (8, strategy_name)


def plot_group_rows(aggregated_csv: str, output_pdf: str):
    if not os.path.exists(aggregated_csv):
        raise FileNotFoundError(f"Aggregated CSV not found: {aggregated_csv}")

    df = pd.read_csv(aggregated_csv)

    required_cols = {
        'filename', 'routing_policy', 'group',
        'avg_ttft', 'avg_tpot', 'avg_end_to_end',
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in aggregated CSV: {missing}")

    groups = [g for g in df['group'].fillna("").unique() if g != ""]
    if not groups:
        # If no group values, treat all rows as one group
        groups = ["all"]
        df['group'] = "all"

    n_rows = len(groups)
    fig_height = 6 * n_rows
    fig = plt.figure(figsize=(22, fig_height))

    for i, group in enumerate(sorted(groups)):
        ax = fig.add_subplot(n_rows, 1, i + 1)
        df_g = df[df['group'] == group].copy()
        if df_g.empty:
            ax.text(0.5, 0.5, f"No data for group: {group}", ha='center', va='center')
            continue

        # Build the DataFrame expected by plot_triple_axis_comparison
        metrics_df = pd.DataFrame({
            'strategy': df_g['filename'].values,
            'avg_ttft': df_g['avg_ttft'].values,
            'avg_tpot': df_g['avg_tpot'].values,
            'avg_end_to_end': df_g['avg_end_to_end'].values,
        })

        # Strategy order by routing policy priority, then by strategy name
        df_g['__priority__'] = df_g.apply(
            lambda r: crs.get_strategy_priority(str(r['routing_policy'])), axis=1
        )
        ordered = df_g.sort_values(by='__priority__')['filename'].tolist()

        # Build color dict by routing policy
        color_dict = {}
        category_counts = {}
        for fname, rp in zip(df_g['filename'], df_g['routing_policy']):
            key = (rp or 'other').lower()
            if key not in category_counts:
                category_counts[key] = 0
            color = crs.get_strategy_color(key, category_counts[key])
            color_dict[fname] = color
            category_counts[key] += 1

        # Call the reusable triple-axis plotter
        crs.plot_triple_axis_comparison(ax, metrics_df, ordered, color_dict)
        ax.set_title(f"{group} - Average TTFT (left), TPOT (middle), End-to-End (right) Comparison", fontsize=20, pad=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    print(f"Saved aggregated comparison plot to {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description='Plot aggregated summary per group using triple-axis comparison')
    parser.add_argument('--csv', default='../workload-and-experiment_results/aggregated_summary.csv', help='Path to aggregated_summary.csv')
    parser.add_argument('--out', default='../workload-and-experiment_results/aggregated_groups_comparison.pdf', help='Output PDF path')
    args = parser.parse_args()

    plot_group_rows(args.csv, args.out)


if __name__ == '__main__':
    main()


