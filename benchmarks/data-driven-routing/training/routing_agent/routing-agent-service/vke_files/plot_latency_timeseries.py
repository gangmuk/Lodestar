import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import pandas as pd
# import seaborn as sns
import re
import json
# from logger import logger

def parse_log_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    data = []
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or "**@latency_metrics@" not in line:
            continue
        try:
            entry = parse_metrics_line(line)
            if entry:
                data.append(entry)
        except Exception as e:
            print(f"Error parsing line {line_num}: {e}")
            continue
    data.sort(key=lambda x: x.get('start_time', 0) or 0)
    if data:
        valid_start_times = [item['start_time'] for item in data if item.get('start_time')]
        if valid_start_times:
            base_time = min(valid_start_times)
            for item in data:
                if item.get('start_time') is not None:
                    item['relative_time'] = (item['start_time'] - base_time) / 1000000
                else:
                    print(f"Error: Missing start_time in entry: {item}")
                    print(f"Error: data: {data}")
                    assert False
                # if item.get('end_time'):
                if item.get('end_time') is not None:
                    item['relative_end_time'] = (item['end_time'] - base_time) / 1000000
                    item['duration'] = (item['end_time'] - item['start_time']) / 1000000
                else:
                    print(f"Error: Missing end_time in entry: {item}")
                    print(f"Error: data: {data}")
                    assert False
        else:
            print("Error: No valid start times found in the data.")
            print(f"Error: data: {data}")
            assert False
    else:
        print("Error: No valid entries found in the log file.")
        assert False
    
    print(f"Parsed {len(data)} entries")
    return data

def parse_metrics_line(line):
    """Parse a single metrics line"""
    # Find the metrics section
    start_marker = "**@latency_metrics@"
    start_idx = line.find(start_marker)
    if start_idx == -1:
        return None
    
    # Extract the metrics string
    metrics_str = line[start_idx + len(start_marker):]
    
    # Parse using a more robust approach
    parsed_data = parse_metrics_string(metrics_str)
    
    # Create final entry with standardized field names
    final_entry = {
        'request_id': parsed_data.get('requestID'),
        'start_time': parsed_data.get('request_start_time'),
        'end_time': parsed_data.get('request_end_time'),
        'ttft': parsed_data.get('ttft'),
        'avg_tpot': parsed_data.get('avg_tpot'),
        'pod': parsed_data.get('selectedpod'),
        'total_decode_time': parsed_data.get('total_decode_time'),
        'e2e': parsed_data.get('e2e'),
        'num_input_tokens': parsed_data.get('numInputTokens'),
        'num_output_tokens': parsed_data.get('numOutputTokens'),
        'num_trains': parsed_data.get('numTrains'),
        'num_flush': parsed_data.get('numFlush'),
    }
    
    # Only require request_id and start_time as essential
    if final_entry.get('request_id') is not None and final_entry.get('start_time') is not None:
        return final_entry
    else:
        print(f"Skipping entry - missing required fields. request_id: {final_entry.get('request_id')}, start_time: {final_entry.get('start_time')}")
        return None

def parse_metrics_string(metrics_str):
    """Parse the metrics string into key-value pairs"""
    parsed = {}
    
    # Split by @ but handle JSON values that contain @
    parts = split_metrics_string(metrics_str)
    
    # Process key-value pairs
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        
        # JSON fields that might span multiple parts
        json_keys = {'allPodsKvCacheHitRatios', 'vllmGPUKVCacheUsage', 'vllmCPUKVCacheUsage', 
                     'vllmNumRequestsRunning', 'vllmNumRequestsWaiting', 'podMetricsLastSecond',
                     'numInflightRequestsAllPods', 'numPrefillTokensForAllPods', 'numDecodeTokensForAllPods'}
        
        if key in json_keys:
            # Handle JSON value
            json_value, consumed_parts = extract_json_value(parts, i + 1)
            if json_value is not None:
                try:
                    parsed[key] = json.loads(json_value)
                except json.JSONDecodeError:
                    parsed[key] = json_value
            i += consumed_parts + 1
        else:
            # Handle regular value
            if i + 1 < len(parts):
                value = parts[i + 1]
                # Try to convert to int
                try:
                    parsed[key] = int(value)
                except ValueError:
                    parsed[key] = value
            i += 2
    
    return parsed

def split_metrics_string(metrics_str):
    """Split metrics string by @ but preserve JSON structure"""
    parts = []
    current_part = ""
    in_json = False
    brace_count = 0
    in_string = False
    escaped = False
    
    i = 0
    while i < len(metrics_str):
        char = metrics_str[i]
        
        if escaped:
            current_part += char
            escaped = False
            i += 1
            continue
        
        if char == '\\' and in_json and in_string:
            current_part += char
            escaped = True
            i += 1
            continue
        
        if char == '"' and in_json:
            in_string = not in_string
            current_part += char
        elif char == '{' and not in_string:
            if not in_json:
                in_json = True
                brace_count = 1
            else:
                brace_count += 1
            current_part += char
        elif char == '}' and in_json and not in_string:
            brace_count -= 1
            current_part += char
            if brace_count == 0:
                in_json = False
        elif char == '@' and not in_json:
            if current_part:
                parts.append(current_part)
                current_part = ""
        else:
            current_part += char
        
        i += 1
    
    if current_part:
        parts.append(current_part)
    
    return parts

def extract_json_value(parts, start_idx):
    """Extract JSON value that might span multiple parts"""
    if start_idx >= len(parts):
        return None, 0
    
    value = parts[start_idx]
    
    # If it doesn't start with {, it's not JSON
    if not value.startswith('{'):
        return value, 1
    
    # Check if JSON is complete in this part
    brace_count = value.count('{') - value.count('}')
    if brace_count == 0:
        return value, 1
    
    # JSON spans multiple parts - reconstruct
    json_parts = [value]
    consumed = 1
    
    for i in range(start_idx + 1, len(parts)):
        part = parts[i]
        json_parts.append('@' + part)  # Add back the @ separator
        brace_count += part.count('{') - part.count('}')
        consumed += 1
        
        if brace_count <= 0:
            break
    
    return ''.join(json_parts), consumed


def get_numtrains_transitions(data):
    """
    Get the first occurrence time of each new numTrains value
    """
    transitions = []
    seen_trains = set()
    
    for item in data:
        num_trains = item['num_trains']
        if num_trains not in seen_trains:
            transitions.append({
                'num_trains': num_trains,
                'relative_time': item['relative_time']
            })
            seen_trains.add(num_trains)
    
    return transitions

def get_numflush_transitions(data):
    """
    Get the first occurrence time of each new numFlush value
    """
    transitions = []
    seen_flush = set()
    
    for item in data:
        num_flush = item['num_flush']
        if num_flush is not None and num_flush not in seen_flush:
            transitions.append({
                'num_flush': num_flush,
                'relative_time': item['relative_time']
            })
            seen_flush.add(num_flush)
    
    return transitions

def calculate_ttft_reward(ttft, slo_ttft=500):
    """Calculate TTFT reward based on the given formula"""
    if ttft <= 0:
        return 0.5
    elif 0 < ttft <= slo_ttft:
        return 0.5 - 0.4 * (ttft / slo_ttft)
    else:  # ttft > slo_ttft
        return -0.1 - 0.4 * min(1, (ttft - slo_ttft) / slo_ttft)

def calculate_tpot_reward(tpot, slo_tpot=50):
    """Calculate TPOT reward based on the given formula"""
    if tpot <= 0:
        return -0.5
    elif 0 < tpot <= slo_tpot:
        return 0.1 + 0.4 * (1 - tpot / slo_tpot)
    else:  # tpot > slo_tpot
        return -0.1 - 0.4 * min(1, (tpot - slo_tpot) / slo_tpot)

def calculate_total_reward(ttft, tpot, slo_ttft=500, slo_tpot=50):
    """Calculate total reward as sum of TTFT and TPOT rewards"""
    return calculate_ttft_reward(ttft, slo_ttft) + calculate_tpot_reward(tpot, slo_tpot)

def calculate_slo_satisfaction(df, slo_ttft=500, slo_tpot=50):
    """Calculate SLO satisfaction statistics"""
    ttft_satisfied = (df['ttft'] <= slo_ttft).sum()
    tpot_satisfied = (df['avg_tpot'] <= slo_tpot).sum()
    both_satisfied = ((df['ttft'] <= slo_ttft) & (df['avg_tpot'] <= slo_tpot)).sum()
    
    total_requests = len(df)
    
    return {
        'ttft_satisfied': ttft_satisfied,
        'tpot_satisfied': tpot_satisfied,
        'both_satisfied': both_satisfied,
        'total_requests': total_requests,
        'ttft_satisfaction_rate': ttft_satisfied / total_requests * 100,
        'tpot_satisfaction_rate': tpot_satisfied / total_requests * 100,
        'both_satisfaction_rate': both_satisfied / total_requests * 100
    }

def calculate_cluster_wise_metrics(df):
    """
    Calculate cluster-wise TTFT and TPOT statistics for each second interval
    """
    # Create time bins (1-second intervals)
    df['time_bin'] = np.floor(df['relative_time']).astype(int)
    
    # Group by time bin and calculate statistics
    cluster_stats = []
    
    for time_bin in sorted(df['time_bin'].unique()):
        bin_data = df[df['time_bin'] == time_bin]
        
        # Overall statistics for this time bin
        overall_stats = {
            'time_bin': time_bin,
            'total_requests': len(bin_data),
            'mean_ttft': bin_data['ttft'].mean(),
            'median_ttft': bin_data['ttft'].median(),
            'std_ttft': bin_data['ttft'].std(),
            'min_ttft': bin_data['ttft'].min(),
            'max_ttft': bin_data['ttft'].max(),
            'p95_ttft': bin_data['ttft'].quantile(0.95),
            'p99_ttft': bin_data['ttft'].quantile(0.99),
            'mean_tpot': bin_data['avg_tpot'].mean(),
            'median_tpot': bin_data['avg_tpot'].median(),
            'std_tpot': bin_data['avg_tpot'].std(),
            'min_tpot': bin_data['avg_tpot'].min(),
            'max_tpot': bin_data['avg_tpot'].max(),
            'p95_tpot': bin_data['avg_tpot'].quantile(0.95),
            'p99_tpot': bin_data['avg_tpot'].quantile(0.99)
        }
        
        # Pod-wise statistics for this time bin
        pod_stats = {}
        for pod in bin_data['pod'].unique():
            pod_data = bin_data[bin_data['pod'] == pod]
            pod_stats[f'pod_{pod}'] = {
                'count': len(pod_data),
                'mean_ttft': pod_data['ttft'].mean(),
                'median_ttft': pod_data['ttft'].median(),
                'std_ttft': pod_data['ttft'].std() if len(pod_data) > 1 else 0,
                'min_ttft': pod_data['ttft'].min(),
                'max_ttft': pod_data['ttft'].max(),
                'mean_tpot': pod_data['avg_tpot'].mean(),
                'median_tpot': pod_data['avg_tpot'].median(),
                'std_tpot': pod_data['avg_tpot'].std() if len(pod_data) > 1 else 0,
                'min_tpot': pod_data['avg_tpot'].min(),
                'max_tpot': pod_data['avg_tpot'].max()
            }
        
        overall_stats['pod_stats'] = pod_stats
        cluster_stats.append(overall_stats)
    
    return cluster_stats

def create_enhanced_plot(data, log_dir, setylim, slo_ttft=1000, slo_tpot=50):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    if len(df) == 0:
        print("Error, No valid data to plot.")
        exit()
    cluster_stats = calculate_cluster_wise_metrics(df)
    train_transitions = get_numtrains_transitions(data)
    flush_transitions = get_numflush_transitions(data)

    df['ttft_reward'] = df['ttft'].apply(lambda x: calculate_ttft_reward(x, slo_ttft))
    df['tpot_reward'] = df['avg_tpot'].apply(lambda x: calculate_tpot_reward(x, slo_tpot))
    df['total_reward'] = df['ttft_reward'] + df['tpot_reward']
    slo_stats = calculate_slo_satisfaction(df, slo_ttft, slo_tpot)

    # Create a more complex figure with GridSpec
    fig = plt.figure(figsize=(15, 24))  # Increased height further
    gs = GridSpec(9, 2, figure=fig, height_ratios=[1, 1, 1, 1, 0.8, 0.8, 0.8, 1, 0.8])
    
    # Define the main time series plots
    ax1 = fig.add_subplot(gs[0, :])  # TTFT plot
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)  # TPOT plot
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)  # E2E Duration plot
    # ax1.set_xlim(left=0)
    # ax2.set_xlim(left=0)
    # ax3.set_xlim(left=0)
    # Define the pod analysis plots
    ax4 = fig.add_subplot(gs[3, 0])  # Average TTFT by Pod
    ax5 = fig.add_subplot(gs[3, 1])  # Average TPOT by Pod
    
    # Define the distribution plots - each in its own row
    ax6 = fig.add_subplot(gs[4, :])  # TTFT distribution
    ax7 = fig.add_subplot(gs[5, :])  # TPOT distribution
    ax8 = fig.add_subplot(gs[6, :])  # E2E distribution
    
    # NEW PLOTS - Add these after the existing plots
    ax9 = fig.add_subplot(gs[7, :])  # Reward time series
    ax10 = fig.add_subplot(gs[8, :])  # SLO satisfaction bar chart

   # Color mapping for different pods
    unique_pods = list(df['pod'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_pods)))

    # # Define different marker styles
    # markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']

    pod_colors = dict(zip(unique_pods, colors))
    # pod_marker = dict(zip(unique_pods, markers))

    ## I want to get cluster-wise ttft of all pods for each second.

    # TTFT Plot (ax1)
    for pod in unique_pods:
        pod_data = df[df['pod'] == pod]
        ax1.scatter(pod_data['relative_time'], pod_data['ttft'], s=100, 
                   color=pod_colors[pod], edgecolor='black', linewidth=1, alpha=0.8, label=f'Pod: {pod}')
    # ax1.plot(df['relative_time'], df['ttft'], 'k--', alpha=0.5)

    # Add cluster-wise TTFT trend line
    time_bins = [stat['time_bin'] for stat in cluster_stats]
    ttft_means = [stat['mean_ttft'] for stat in cluster_stats]
    ax1.plot(time_bins, ttft_means, 'r-', linewidth=2, alpha=0.8, label='Cluster Mean TTFT', zorder=10)
    
    # Add vertical lines for numTrains transitions
    for transition in train_transitions:
        ax1.axvline(x=transition['relative_time'], color='purple', linewidth=2, alpha=0.8, zorder=5)
    for transition in flush_transitions:
        ax1.axvline(x=transition['relative_time'], color='orange', linewidth=2, alpha=0.8, zorder=5)

    # TPOT Plot (ax2)
    for pod in unique_pods:
        pod_data = df[df['pod'] == pod]
        ax2.scatter(pod_data['relative_time'], pod_data['avg_tpot'], s=100, color=pod_colors[pod], edgecolor='black', linewidth=1, alpha=0.8)

    # Add cluster-wise TPOT trend line
    tpot_means = [stat['mean_tpot'] for stat in cluster_stats]
    ax2.plot(time_bins, tpot_means, 'r-', linewidth=2, alpha=0.8, label='Cluster Mean TPOT', zorder=10)
    
    # Add vertical lines for numTrains transitions
    for transition in train_transitions:
        ax2.axvline(x=transition['relative_time'], color='purple', linewidth=2, alpha=0.8, zorder=5)
    for transition in flush_transitions:
        ax2.axvline(x=transition['relative_time'], color='orange', linewidth=2, alpha=0.8, zorder=5)
    # E2E Duration Plot (ax3) - Now following the same format as TTFT and TPOT
    for pod in unique_pods:
        pod_data = df[df['pod'] == pod]
        ax3.scatter(pod_data['relative_time'], pod_data['e2e'], s=100, color=pod_colors[pod], edgecolor='black', linewidth=1, alpha=0.8)
    
    # Average TTFT by Pod (ax4)
    pod_avg_ttft = df.groupby('pod')['ttft'].mean().sort_values(ascending=False)
    pod_counts = df.groupby('pod').size()
    
    # Create bars for TTFT plot
    ttft_bars = ax4.bar(range(len(pod_avg_ttft)), pod_avg_ttft.values, 
                      color=[pod_colors[pod] for pod in pod_avg_ttft.index], 
                      edgecolor='black', alpha=0.8, width=0.7)
    
    # Add annotations
    for i, (pod, ttft) in enumerate(pod_avg_ttft.items()):
        ax4.text(i, ttft + 5, f'{ttft:.1f} ms', ha='center', fontsize=12, fontweight='bold')
        ax4.text(i, 10, f'n={pod_counts[pod]}', ha='center', fontsize=10)
    
    ax4.set_xticks(range(len(pod_avg_ttft)))
    ax4.set_xticklabels([f'Pod {pod}' for pod in pod_avg_ttft.index], rotation=45, ha='right', fontsize=11)
    ax4.set_ylabel('Average TTFT (ms)', fontsize=12, fontweight='bold')
    ax4.set_title('Average TTFT by Pod', fontsize=14, fontweight='bold', pad=10)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Average TPOT by Pod (ax5)
    pod_avg_tpot = df.groupby('pod')['avg_tpot'].mean().sort_values(ascending=False)
    
    # Create bars for TPOT plot
    tpot_bars = ax5.bar(range(len(pod_avg_tpot)), pod_avg_tpot.values,
                      color=[pod_colors[pod] for pod in pod_avg_tpot.index], 
                      edgecolor='black', alpha=0.8, width=0.7)
    
    # Add annotations
    for i, (pod, tpot) in enumerate(pod_avg_tpot.items()):
        ax5.text(i, tpot + 1, f'{tpot:.1f} ms', ha='center', fontsize=12, fontweight='bold')
        ax5.text(i, 0.5, f'n={pod_counts[pod]}', ha='center', fontsize=10)
    
    ax5.set_xticks(range(len(pod_avg_tpot)))
    ax5.set_xticklabels([f'Pod {pod}' for pod in pod_avg_tpot.index], rotation=45, ha='right', fontsize=11)
    ax5.set_ylabel('Average TPOT (ms)', fontsize=12, fontweight='bold')
    ax5.set_title('Average TPOT by Pod', fontsize=14, fontweight='bold', pad=10)
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    
    # TTFT/TPOT/E2E distributions - now in separate plots (ax6, ax7, ax8)
    # Distribution for TTFT (ax6)
    ax6.hist(df['ttft'], bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax6.set_xlabel('Time (ms)', fontsize=12)
    ax6.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax6.set_title('TTFT Distribution', fontsize=14, fontweight='bold', pad=10)
    
    # Add vertical line for mean
    ttft_mean = df['ttft'].mean()
    ax6.axvline(ttft_mean, color='red', linestyle='--', alpha=0.7)
    ax6.text(ttft_mean + 5, ax6.get_ylim()[1]*0.9, 
            f'Mean: {ttft_mean:.1f} ms', color='red', 
            fontsize=10, backgroundcolor='white', alpha=0.9)
    
    # TPOT Distribution (ax7)
    ax7.hist(df['avg_tpot'], bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
    ax7.set_xlabel('Time (ms)', fontsize=12)
    ax7.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax7.set_title('TPOT Distribution', fontsize=14, fontweight='bold', pad=10)
    
    # Add vertical line for mean
    tpot_mean = df['avg_tpot'].mean()
    ax7.axvline(tpot_mean, color='red', linestyle='--', alpha=0.7)
    ax7.text(tpot_mean + 0.5, ax7.get_ylim()[1]*0.9, 
            f'Mean: {tpot_mean:.1f} ms', color='red', 
            fontsize=10, backgroundcolor='white', alpha=0.9)
    
    # E2E Distribution (ax8) - now full width
    ax8.hist(df['e2e'], bins=30, density=True, alpha=0.7, color='purple', edgecolor='black')
    ax8.set_xlabel('Time (ms)', fontsize=12)
    ax8.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax8.set_title('E2E Duration Distribution', fontsize=14, fontweight='bold', pad=10)
    
    # Add vertical line for mean
    e2e_mean = df['e2e'].mean()
    ax8.axvline(e2e_mean, color='red', linestyle='--', alpha=0.7)
    ax8.text(e2e_mean + 50, ax8.get_ylim()[1]*0.9, 
            f'Mean: {e2e_mean:.1f} ms', color='red', 
            fontsize=10, backgroundcolor='white', alpha=0.9)
    
    # Set titles and labels for main plots
    ax1.set_title('Time to First Token (TTFT) for Each Request', fontsize=16, fontweight='bold', pad=10)
    # set ylim
    ax1.set_ylabel('TTFT (ms)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # ax1.legend(fontsize=12)
    from matplotlib.lines import Line2D
    legend_elements = ax1.get_legend_handles_labels()
    legend_elements[0].append(Line2D([0], [0], color='purple', linewidth=2, label='numTrains transition'))
    legend_elements[1].append('numTrains transition')
    legend_elements[0].append(Line2D([0], [0], color='orange', linewidth=2, label='numFlush transition'))
    legend_elements[1].append('numFlush transition')
    ax1.legend(legend_elements[0], legend_elements[1], fontsize=12)
    
    ax2.set_title('Average Time Per Output Token (TPOT) for Each Request', fontsize=16, fontweight='bold', pad=10)
    ax2.set_ylabel('Average TPOT (ms)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('End-to-End Request Duration for Each Request', fontsize=16, fontweight='bold', pad=10)
    ax3.set_xlabel('Relative Time (seconds)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('End-to-End Duration (ms)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Set font sizes for tick labels
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Improve x-axis formatting
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # # Set y-axis limits with some padding
    if setylim:
        ax1.set_ylim(0, 2000)
        ax2.set_ylim(0, 200)
        ax3.set_ylim(0, 10000)
    else:
        ax1.set_ylim(0, df['ttft'].max() * 1.1)
        ax2.set_ylim(0, df['avg_tpot'].max() * 1.1)
        ax3.set_ylim(0, df['e2e'].max() * 1.1)


    ax9.scatter(df['relative_time'], df['ttft_reward'], s=50, alpha=0.6, color='blue', label='TTFT Reward')
    ax9.scatter(df['relative_time'], df['tpot_reward'], s=50, alpha=0.6, color='green', label='TPOT Reward')
    ax9.scatter(df['relative_time'], df['total_reward'], s=50, alpha=0.6, color='red', label='Total Reward')
    
    # Add trend lines for rewards
    time_bins = [stat['time_bin'] for stat in cluster_stats]
    ttft_reward_means = [df[df['time_bin'] == time_bin]['ttft_reward'].mean() for time_bin in time_bins]
    tpot_reward_means = [df[df['time_bin'] == time_bin]['tpot_reward'].mean() for time_bin in time_bins]
    total_reward_means = [df[df['time_bin'] == time_bin]['total_reward'].mean() for time_bin in time_bins]
    
    ax9.plot(time_bins, ttft_reward_means, 'b-', linewidth=2, alpha=0.8, label='TTFT Reward Mean')
    ax9.plot(time_bins, tpot_reward_means, 'g-', linewidth=2, alpha=0.8, label='TPOT Reward Mean')
    ax9.plot(time_bins, total_reward_means, 'r-', linewidth=2, alpha=0.8, label='Total Reward Mean')
    
    # Add vertical lines for numTrains transitions
    for transition in train_transitions:
        ax9.axvline(x=transition['relative_time'], color='purple', linewidth=2, alpha=0.8, zorder=5)
    for transition in flush_transitions:
        ax9.axvline(x=transition['relative_time'], color='orange', linewidth=2, alpha=0.8, zorder=5)
    # Add horizontal lines for reward boundaries
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax9.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax9.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax9.set_title(f'Reward Time Series (SLO: TTFT≤{slo_ttft}ms, TPOT≤{slo_tpot}ms)', fontsize=16, fontweight='bold', pad=10)
    ax9.set_xlabel('Relative Time (seconds)', fontsize=14, fontweight='bold')
    ax9.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax9.set_ylim(-1.1, 1.1)
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=10, loc='upper right')
    
    # NEW SLO SATISFACTION BAR CHART (ax10)
    categories = ['TTFT SLO\nSatisfied', 'TPOT SLO\nSatisfied', 'Both SLOs\nSatisfied']
    satisfied_counts = [slo_stats['ttft_satisfied'], slo_stats['tpot_satisfied'], slo_stats['both_satisfied']]
    satisfaction_rates = [slo_stats['ttft_satisfaction_rate'], slo_stats['tpot_satisfaction_rate'], slo_stats['both_satisfaction_rate']]
    
    bars = ax10.bar(categories, satisfied_counts, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
    
    # Add percentage labels on bars
    for i, (bar, rate) in enumerate(zip(bars, satisfaction_rates)):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}\n({rate:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax10.set_title(f'SLO Satisfaction Summary (Total Requests: {slo_stats["total_requests"]})', fontsize=16, fontweight='bold', pad=10)
    ax10.set_ylabel('Number of Requests', fontsize=14, fontweight='bold')
    ax10.set_ylim(0, slo_stats['total_requests'] * 1.1)
    ax10.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add SLO values as text
    ax10.text(0.02, 0.98, f'SLO Thresholds:\nTTFT ≤ {slo_ttft} ms\nTPOT ≤ {slo_tpot} ms', 
             transform=ax10.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Update the layout and save - modify the existing code
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]:
        ax.tick_params(axis='both', which='major', labelsize=11)
        if ax != ax10:  # Don't add grid to bar chart
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add reward statistics to the summary print
    print(f"TTFT: {slo_stats['ttft_satisfied']}/{slo_stats['total_requests']} ({slo_stats['ttft_satisfaction_rate']:.1f}%)")
    print(f"TPOT: {slo_stats['tpot_satisfied']}/{slo_stats['total_requests']} ({slo_stats['tpot_satisfaction_rate']:.1f}%)")
    print(f"Both: {slo_stats['both_satisfied']}/{slo_stats['total_requests']} ({slo_stats['both_satisfaction_rate']:.1f}%)")
    
    # Add a super title
    fig.suptitle(f'Latency Metrics Analysis (#request: {len(data)})', fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save files
    plt_fn = f"{log_dir}/latency_metrics_analysis.pdf"
    plt.savefig(plt_fn, bbox_inches='tight')
    print(f"* Saving plot to: {plt_fn}")
    
    return fig, plt_fn

if __name__ == "__main__":
    # Parse the log file
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_latency_timeseries.py <log_file>")
        sys.exit(1)

    if len(sys.argv) == 2:
        setylim = False
    else:
        setylim = int(sys.argv[2]) == 1

    slo_ttft = 1000 if len(sys.argv) <= 3 else int(sys.argv[3])
    slo_tpot = 50 if len(sys.argv) <= 4 else int(sys.argv[4])
    
    log_file = sys.argv[1]
    log_dir = log_file.rsplit('/', 1)[0]
    # Parse the log file to get data
    data = parse_log_file(log_file)
    
    if not data:
        print(f"Error: No valid latency metrics found in {log_file}. Please check the file format.")
        assert False
    
    print(f"Found {len(data)} log entries with latency metrics")
    
    # Create and save the enhanced plot
    fig, output_file = create_enhanced_plot(data, log_dir, setylim, slo_ttft, slo_tpot)
    
    # Print summary statistics
    df = pd.DataFrame(data)
    print("\nSummary Statistics:")
    print(f"TTFT - Min: {df['ttft'].min()} ms, Max: {df['ttft'].max()} ms, Avg: {df['ttft'].mean():.2f} ms")
    print(f"TPOT - Min: {df['avg_tpot'].min()} ms, Max: {df['avg_tpot'].max()} ms, Avg: {df['avg_tpot'].mean():.2f} ms")
    print(f"E2E  - Min: {df['e2e'].min()} ms, Max: {df['e2e'].max()} ms, Avg: {df['e2e'].mean():.2f} ms")