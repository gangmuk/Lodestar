#!/usr/bin/env python3
"""
Plot per-pod request count time series from aibrix gateway log.
Parses pod_request_count log lines and visualizes the drift into negative values.

Usage:
    python plot_pod_request_count.py [--log LOG_FILE] [--output OUTPUT_FILE]
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib import rcParams

# Set publication-quality style using matplotlib only
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times', 'Palatino']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 16
rcParams['axes.linewidth'] = 1.2
rcParams['grid.linewidth'] = 0.8
rcParams['grid.alpha'] = 0.3
rcParams['lines.linewidth'] = 1.5
rcParams['patch.linewidth'] = 0.5
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.minor.width'] = 0.8
rcParams['ytick.minor.width'] = 0.8
rcParams['axes.grid'] = True
rcParams['axes.axisbelow'] = True
rcParams['axes.facecolor'] = 'white'
rcParams['figure.facecolor'] = 'white'
rcParams['axes.edgecolor'] = '#333333'
rcParams['axes.labelcolor'] = '#000000'
rcParams['xtick.color'] = '#333333'
rcParams['ytick.color'] = '#333333'
rcParams['text.color'] = '#000000'


def parse_log_file(log_file):
    """Parse pod_request_count lines from the gateway log file."""
    timestamps = []
    pod_series = defaultdict(list)
    pod_names_ordered = None

    with open(log_file) as f:
        for line in f:
            if "pod_request_count:" not in line:
                continue

            # Extract timestamp
            ts_match = re.search(r"I\d+ (\d+:\d+:\d+\.\d+)", line)
            if not ts_match:
                continue
            ts = datetime.strptime(ts_match.group(1), "%H:%M:%S.%f")

            # Extract pod:value pairs
            pairs = re.findall(r"([\w-]+):(-?\d+)", line.split("map[")[1].split("]")[0])
            if not pairs:
                continue

            # Use short pod names (last segment of pod name)
            current = {}
            for pod_full, val in pairs:
                short = pod_full.rsplit("-", 1)[-1]
                current[short] = int(val)

            if pod_names_ordered is None:
                pod_names_ordered = sorted(current.keys())

            timestamps.append(ts)
            for pod in pod_names_ordered:
                pod_series[pod].append(current.get(pod, 0))

    return timestamps, pod_series, pod_names_ordered


def plot_per_pod(timestamps, pod_series, pod_names, output_file):
    """Create a per-pod time series plot with a sum line."""
    # Professional color palette - using a more sophisticated scheme
    # Using a combination of distinct, publication-friendly colors
    professional_colors = [
        '#2E86AB',  # Blue
        '#A23B72',  # Purple
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#6A994E',  # Green
        '#BC4749',  # Dark red
        '#219EBC',  # Cyan
        '#FFB703',  # Yellow
        '#8B5A3C',  # Brown
        '#7209B7',  # Violet
    ]
    
    # Create figure with better proportions - three subplots
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1], hspace=0.35, 
                          left=0.1, right=0.88, top=0.95, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])  # Individual pod lines
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Min, max, avg, std
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # Sum of all pods

    # --- Subplot 1: Individual per-pod lines ---
    # Calculate aggregate statistics across all pods (for subplot 2)
    avg_values = []
    min_values = []
    max_values = []
    std_values = []
    for i in range(len(timestamps)):
        values_at_t = [pod_series[p][i] for p in pod_names]
        if values_at_t:
            mean = sum(values_at_t) / len(values_at_t)
            variance = sum((x - mean) ** 2 for x in values_at_t) / len(values_at_t)
            std = variance ** 0.5  # Standard deviation
            avg_values.append(mean)
            min_values.append(min(values_at_t))
            max_values.append(max(values_at_t))
            std_values.append(std)
        else:
            avg_values.append(0)
            min_values.append(0)
            max_values.append(0)
            std_values.append(0)
    
    # Plot each pod with distinct styling
    for i, pod in enumerate(pod_names):
        color = professional_colors[i % len(professional_colors)]
        ax1.plot(timestamps, pod_series[pod], label=pod,
                 color=color, linewidth=1.8, alpha=0.9, 
                 marker='', antialiased=True, zorder=1)

    # Zero reference line with professional styling
    ax1.axhline(y=0, color="#D32F2F", linestyle="--", linewidth=1.5, 
                alpha=0.8, zorder=0, label="Zero reference")

    # Shade negative region with subtle styling
    y_min = min(min(v) for v in pod_series.values())
    y_max = max(max(v) for v in pod_series.values())
    if y_min < 0:
        ax1.axhspan(y_min - abs(y_max - y_min) * 0.05, 0, 
                   alpha=0.12, color="#D32F2F", zorder=0)
    
    # Professional axis labels and title
    ax1.set_ylabel("Request Count\n(realtime_num_requests_running)", 
                   fontsize=12, fontweight='medium')
    ax1.set_title("Per-Pod Request Count Time Series", 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Enhanced legend
    legend = ax1.legend(loc="upper left", fontsize=9, ncol=min(4, len(pod_names) + 1),
                       frameon=True, fancybox=True, shadow=True,
                       framealpha=0.95, edgecolor='gray', facecolor='white')
    legend.get_frame().set_linewidth(0.8)
    
    # Professional grid
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.7, which='major')
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, which='minor')
    ax1.set_axisbelow(True)
    
    # Clean up spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#333333')
    ax1.spines['bottom'].set_color('#333333')
    
    # Set y-axis limits with padding
    y_range = y_max - y_min
    ax1.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

    # --- Subplot 2: Aggregate statistics (Min, Max, Avg, Std) ---
    # Plot aggregate lines
    ax2.plot(timestamps, max_values, color="#1A1A1A", linewidth=2.5, 
             linestyle='-', alpha=0.9, label="Max", zorder=2, antialiased=True)
    ax2.plot(timestamps, avg_values, color="#666666", linewidth=2.5, 
             linestyle='--', alpha=0.9, label="Average", zorder=2, antialiased=True)
    ax2.plot(timestamps, min_values, color="#1A1A1A", linewidth=2.5, 
             linestyle=':', alpha=0.9, label="Min", zorder=2, antialiased=True)
    
    # Zero reference line
    ax2.axhline(y=0, color="#D32F2F", linestyle="--", linewidth=1.5, 
                alpha=0.8, zorder=0)
    
    # Create right y-axis for standard deviation
    ax2_right = ax2.twinx()
    
    # Plot std on right y-axis
    std_color = "#8B5A3C"  # Brown color for std
    ax2_right.plot(timestamps, std_values, color=std_color, 
                   linewidth=2.0, linestyle='-.', alpha=0.85, 
                   label="Std Dev", zorder=3, antialiased=True)
    ax2_right.set_ylabel("Standard Deviation\n(between pods)", 
                        fontsize=12, fontweight='medium', color=std_color)
    ax2_right.tick_params(axis='y', labelcolor=std_color, labelsize=10)
    
    # Clean up right axis spines
    ax2_right.spines['top'].set_visible(False)
    ax2_right.spines['left'].set_visible(False)
    ax2_right.spines['right'].set_color(std_color)
    ax2_right.spines['bottom'].set_visible(False)
    
    # Professional axis labels and title
    ax2.set_ylabel("Request Count", fontsize=12, fontweight='medium')
    ax2.set_title("Aggregate Statistics (Min, Max, Average, Std Dev)", 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Enhanced legend - combine both axes
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2_right, labels2_right = ax2_right.get_legend_handles_labels()
    legend2 = ax2.legend(lines2 + lines2_right, labels2 + labels2_right, 
                        loc="upper left", fontsize=9, ncol=4,
                        frameon=True, fancybox=True, shadow=True,
                        framealpha=0.95, edgecolor='gray', facecolor='white')
    legend2.get_frame().set_linewidth(0.8)
    
    # Professional grid
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.7, which='major')
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, which='minor')
    ax2.set_axisbelow(True)
    
    # Clean up spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#333333')
    ax2.spines['bottom'].set_color('#333333')
    
    # Set y-axis limits with padding for ax2
    agg_y_min = min(min_values)
    agg_y_max = max(max_values)
    agg_y_range = agg_y_max - agg_y_min
    ax2.set_ylim(agg_y_min - agg_y_range * 0.05, agg_y_max + agg_y_range * 0.05)

    # --- Subplot 3: Sum of all pods ---
    total = [sum(pod_series[p][i] for p in pod_names) for i in range(len(timestamps))]
    ax3.plot(timestamps, total, color="#1A1A1A", linewidth=2.0, 
             label="Sum (all pods)", marker='', antialiased=True, zorder=3)
    ax3.axhline(y=0, color="#D32F2F", linestyle="--", linewidth=1.5, 
                alpha=0.8, zorder=0)
    
    ax3.set_ylabel("Sum of All Pods", fontsize=12, fontweight='medium')
    ax3.set_xlabel("Time", fontsize=12, fontweight='medium')
    
    # Professional legend for sum plot
    legend3 = ax3.legend(loc="upper left", fontsize=9, frameon=True, 
                         fancybox=True, shadow=True, framealpha=0.95, 
                         edgecolor='gray', facecolor='white')
    legend3.get_frame().set_linewidth(0.8)
    
    # Professional grid for bottom plot
    ax3.grid(True, alpha=0.4, linestyle='-', linewidth=0.7, which='major')
    ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, which='minor')
    ax3.set_axisbelow(True)
    
    # Clean up spines for bottom plot
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#333333')
    ax3.spines['bottom'].set_color('#333333')

    # Professional time formatting
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax3.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    
    # Rotate x-axis labels for better readability
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Save with publication-quality settings
    # Auto-detect format from extension, default to PDF for paper quality
    output_format = 'pdf' if output_file.lower().endswith('.pdf') else None
    plt.savefig(output_file, dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none', format=output_format)
    print(f"Saved plot to {output_file}")
    plt.close()


def print_summary(timestamps, pod_series, pod_names):
    """Print summary statistics."""
    print(f"Total entries: {len(timestamps)}")
    print(f"Time range: {timestamps[0].strftime('%H:%M:%S.%f')} -> {timestamps[-1].strftime('%H:%M:%S.%f')}")

    negative_entries = 0
    for i in range(len(timestamps)):
        if any(pod_series[p][i] < 0 for p in pod_names):
            negative_entries += 1
    print(f"Entries with negative values: {negative_entries} / {len(timestamps)} ({100*negative_entries/len(timestamps):.1f}%)")

    print(f"\n{'Pod':<8} {'Min':>6} {'Max':>6} {'Final':>6}")
    print("-" * 30)
    for pod in pod_names:
        vals = pod_series[pod]
        print(f"{pod:<8} {min(vals):>6} {max(vals):>6} {vals[-1]:>6}")
    final_sum = sum(pod_series[p][-1] for p in pod_names)
    print(f"{'SUM':<8} {'':>6} {'':>6} {final_sum:>6}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-pod request count time series")
    parser.add_argument("log", default="all-aibrix-gateway-plugins.log.txt",
                        help="Path to gateway log file")
    args = parser.parse_args()

    timestamps, pod_series, pod_names = parse_log_file(args.log)
    if not timestamps:
        print("No pod_request_count entries found in log file.")
        return

    print_summary(timestamps, pod_series, pod_names)
    input_file_dir = os.path.dirname(args.log)
    plot_per_pod(timestamps, pod_series, pod_names, os.path.join(input_file_dir, "pod_request_count_timeseries.pdf"))


if __name__ == "__main__":
    main()
