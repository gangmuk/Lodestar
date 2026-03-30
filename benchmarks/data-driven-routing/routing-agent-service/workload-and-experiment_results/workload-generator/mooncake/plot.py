import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Resolve paths relative to this notebook's directory
_NOTEBOOK_DIR = os.path.dirname(os.path.abspath('mooncake-plot.ipynb'))


def plot_timeseries_analysis(df, workload_name, save_path=None, num_tokens_per_hash_id=100):
    """Plot 4x2 time series analysis figure.

    Args:
        df: DataFrame with columns: timestamp (ms), input_length, output_length, hash_ids
        workload_name: Title prefix for the figure
        save_path: If given, save PDF to this path. Otherwise call plt.show().
        num_tokens_per_hash_id: Tokens per hash_id block (for label only)
    """
    df = df.copy()
    df['timestamp_seconds'] = df['timestamp'] / 1000
    df['num_token_blocks'] = df['hash_ids'].apply(len)

    max_time = int(df['timestamp_seconds'].max()) + 1
    time_bins = range(0, max_time + 1)

    fig, axes = plt.subplots(4, 2, figsize=(10, 8))
    fig.suptitle(f'{workload_name} (total {len(df)} requests)', fontsize=16)

    # 1. RPS over time with per-minute average trend line
    rps_by_second = df.groupby(df['timestamp_seconds'].astype(int)).size().reindex(time_bins, fill_value=0)

    axes[0, 0].plot(rps_by_second.index, rps_by_second.values, linewidth=1, alpha=0.5, color='blue', label='Per-second')
    # Per-minute average RPS trend line
    max_minute_rps = max_time // 60 + 1
    minute_centers = []
    minute_avg_rps = []
    rps_values = rps_by_second.values
    for m in range(max_minute_rps + 1):
        start = m * 60
        end = min((m + 1) * 60, len(rps_values))
        if start < len(rps_values):
            avg = np.mean(rps_values[start:end])
            minute_centers.append((start + end) / 2)
            minute_avg_rps.append(avg)
    axes[0, 0].plot(minute_centers, minute_avg_rps, linewidth=2, alpha=0.9, color='red', label='Per-minute avg')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Requests per Second')
    axes[0, 0].set_title('RPS Over Time')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, max_time)

    # 2. Total input tokens per second over time
    total_input_by_second = df.groupby(df['timestamp_seconds'].astype(int))['input_length'].sum().reindex(time_bins, fill_value=0)

    axes[0, 1].plot(total_input_by_second.index, total_input_by_second.values, linewidth=1, alpha=0.5, color='green', label='Per-second')
    # Per-minute average trend line
    input_vals = total_input_by_second.values
    minute_centers_input = []
    minute_avg_input = []
    for m in range(max_minute_rps + 1):
        start = m * 60
        end = min((m + 1) * 60, len(input_vals))
        if start < len(input_vals):
            avg = np.mean(input_vals[start:end])
            minute_centers_input.append((start + end) / 2)
            minute_avg_input.append(avg)
    axes[0, 1].plot(minute_centers_input, minute_avg_input, linewidth=2, alpha=0.9, color='darkgreen', label='Per-minute avg')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Total Input Tokens')
    axes[0, 1].set_title('Total Input Tokens Per Second')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, max_time)

    # 3. Total output tokens per second over time
    total_output_by_second = df.groupby(df['timestamp_seconds'].astype(int))['output_length'].sum().reindex(time_bins, fill_value=0)

    axes[1, 0].plot(total_output_by_second.index, total_output_by_second.values, linewidth=1, alpha=0.5, color='red', label='Per-second')
    # Per-minute average trend line
    output_vals = total_output_by_second.values
    minute_centers_output = []
    minute_avg_output = []
    for m in range(max_minute_rps + 1):
        start = m * 60
        end = min((m + 1) * 60, len(output_vals))
        if start < len(output_vals):
            avg = np.mean(output_vals[start:end])
            minute_centers_output.append((start + end) / 2)
            minute_avg_output.append(avg)
    axes[1, 0].plot(minute_centers_output, minute_avg_output, linewidth=2, alpha=0.9, color='darkred', label='Per-minute avg')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Total Output Tokens')
    axes[1, 0].set_title('Total Output Tokens Per Second')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, max_time)

    # 4. Prefix sharing ratio over time (10-second windows + per-request)
    def find_longest_common_prefix(seq1, seq2):
        min_len = min(len(seq1), len(seq2))
        for i in range(min_len):
            if seq1[i] != seq2[i]:
                return i
        return min_len

    def calculate_prefix_sharing_ratios(requests_with_timestamps):
        """Return (window_avg, list_of_(timestamp, ratio) pairs)."""
        if len(requests_with_timestamps) <= 1:
            return 0.0, []

        per_request = []
        for i, (ts, current_hash_ids) in enumerate(requests_with_timestamps):
            if len(current_hash_ids) == 0:
                continue
            max_prefix_length = 0
            for j in range(i):
                _, prev_hash_ids = requests_with_timestamps[j]
                prefix_length = find_longest_common_prefix(current_hash_ids, prev_hash_ids)
                max_prefix_length = max(max_prefix_length, prefix_length)
            ratio = max_prefix_length / len(current_hash_ids)
            per_request.append((ts, ratio))

        ratios = [r for _, r in per_request]
        avg = sum(ratios) / len(ratios) if ratios else 0.0
        return avg, per_request

    window_sec = 10
    max_window = int(df['timestamp_seconds'].max() / window_sec) + 1
    window_bins = range(0, max_window + 1)

    window_ratios = []
    window_timestamps = []
    per_request_times = []
    per_request_ratios = []

    for w in window_bins:
        t_start = w * window_sec
        t_end = (w + 1) * window_sec
        window_data = df[(df['timestamp_seconds'] >= t_start) & (df['timestamp_seconds'] < t_end)]
        if len(window_data) > 1:
            window_data_sorted = window_data.sort_values('timestamp_seconds')
            requests_with_timestamps = list(zip(window_data_sorted['timestamp_seconds'], window_data_sorted['hash_ids']))
            avg, per_req = calculate_prefix_sharing_ratios(requests_with_timestamps)
            window_ratios.append(avg)
            window_timestamps.append(t_start + window_sec / 2)
            per_request_times.extend([ts for ts, _ in per_req])
            per_request_ratios.extend([r for _, r in per_req])

    axes[1, 1].scatter(per_request_times, per_request_ratios, s=2, alpha=0.15, color='violet', label='Per-request')
    axes[1, 1].plot(window_timestamps, window_ratios, linewidth=2, alpha=0.8, color='purple', marker='o', markersize=3, label=f'{window_sec}s window avg')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Prefix Sharing Ratio')
    axes[1, 1].set_title(f'Prefix Cache Hit Ratio Over Time ({window_sec}s windows)')
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, max_time)
    axes[1, 1].set_ylim(0, 1)

    # 5. Token blocks per request distribution
    axes[2, 0].hist(df['num_token_blocks'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[2, 0].set_xlabel('Number of Token Blocks')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title(f'Token Blocks per Request Distribution ({num_tokens_per_hash_id} tokens/block)')
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Input token length histogram
    axes[2, 1].hist(df['input_length'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[2, 1].set_xlabel('Input Token Length')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].set_title('Input Token Length Distribution')
    axes[2, 1].grid(True, alpha=0.3)

    # 7. Output token length histogram
    axes[3, 0].hist(df['output_length'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[3, 0].set_xlabel('Output Token Length')
    axes[3, 0].set_ylabel('Frequency')
    axes[3, 0].set_title('Output Token Length Distribution')
    axes[3, 0].grid(True, alpha=0.3)

    # Hide the empty subplot
    axes[3, 1].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"* saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def load_mooncake_data(input_file):
    # If the path is already absolute, use it as-is; otherwise resolve relative to notebook dir
    if os.path.isabs(input_file):
        filepath = input_file
    else:
        filepath = os.path.join(_NOTEBOOK_DIR, input_file)
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)

    # Convert timestamp from milliseconds to seconds
    df['timestamp_seconds'] = df['timestamp'] / 1000

    workload_name = os.path.basename(input_file).split('.')[0]

    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(f'{workload_name} - Distribution Analysis', fontsize=16)

    # 1. RPS calculation and distribution
    # Group by second and count requests
    rps_data = df.groupby(df['timestamp_seconds'].astype(int)).size()
    print(f"RPS stats: min={rps_data.min()}, max={rps_data.max()}, mean={rps_data.mean():.2f}, std={rps_data.std():.2f}")

    # RPS distribution
    axes[0, 0].hist(rps_data.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Requests per Second')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('RPS Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Input token length distribution
    axes[0, 1].hist(df['input_length'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Input Token Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Input Token Length Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Output token length distribution
    axes[1, 0].hist(df['output_length'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Output Token Length')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Output Token Length Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Scatter plot of input vs output length
    axes[1, 1].scatter(df['input_length'], df['output_length'], alpha=0.5, s=20, c='purple')
    axes[1, 1].set_xlabel('Input Token Length')
    axes[1, 1].set_ylabel('Output Token Length')
    axes[1, 1].set_title('Input vs Output Token Length')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Time series analysis
    plot_timeseries_analysis(df, workload_name, save_path=f'plot_{workload_name}.pdf')
    print(f"Total requests: {len(df)}")


if __name__ == "__main__":
    load_mooncake_data('Mooncake_conversation_trace.jsonl')
    load_mooncake_data('Mooncake_toolagent_trace.jsonl')
    load_mooncake_data('Mooncake_synthetic_trace.jsonl')
