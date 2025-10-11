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
from matplotlib.lines import Line2D
import numpy as np
from numpy.polynomial.polynomial import polyfit
# from logger import logger

linewidth = 1.5
edgecolor = 'gray'
alpha = 0.7
marker_size = 50
edgewidth=0.5

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
        'selectedpod': parsed_data.get('selectedpod'),
        'total_decode_time': parsed_data.get('total_decode_time'),
        'e2e': parsed_data.get('e2e'),
        'num_input_tokens': parsed_data.get('numInputTokens'),
        'num_output_tokens': parsed_data.get('numOutputTokens'),
        'num_trains': parsed_data.get('numTrains'),
        'num_flush': parsed_data.get('numFlush'),
        'vllm_num_requests_waiting': parsed_data.get('vllmNumRequestsWaiting'),
        'all_pods_kv_cache_hit_ratios': parsed_data.get('allPodsKvCacheHitRatios'),
        'vllm_num_requests_running': parsed_data.get('vllmNumRequestsRunning'),  # ADD THIS
        'num_prefill_tokens_for_all_pods': parsed_data.get('numPrefillTokensForAllPods'),  # ADD THIS
        'num_decode_tokens_for_all_pods': parsed_data.get('numDecodeTokensForAllPods'),  # ADD THIS
        'vllm_gpu_kv_cache_usage': parsed_data.get('vllmGPUKVCacheUsage'),  # ADD THIS
        'exploration': parsed_data.get('exploration'),  # ADD THIS - routing exploration flag
        'exploration_enabled': parsed_data.get('explorationEnabled'),  # ADD THIS - exploration enabled flag
        'predicted_latencies': parsed_data.get('predictedLatencies'),  # ADD THIS - predicted latencies for all pods
        'chosen_pod_predicted_latency': float(parsed_data.get('chosenPodPredictedLatency', 0)) if parsed_data.get('chosenPodPredictedLatency') else None,  # ADD THIS - predicted latency for chosen pod
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
                     'numInflightRequestsAllPods', 'numPrefillTokensForAllPods', 'numDecodeTokensForAllPods',
                     'predictedLatencies'}
        
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
    Calculate cluster-wise TTFT, TPOT, E2E, and Reward statistics for each second interval
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
            'p99_tpot': bin_data['avg_tpot'].quantile(0.99),
            'mean_e2e': bin_data['e2e'].mean(),
            'min_e2e': bin_data['e2e'].min(),
            'max_e2e': bin_data['e2e'].max(),
            'mean_reward': bin_data['total_reward'].mean(),
            'min_reward': bin_data['total_reward'].min(),
            'max_reward': bin_data['total_reward'].max()
        }
        
        # Pod-wise statistics for this time bin
        pod_stats = {}
        for pod in bin_data['selectedpod'].unique():
            pod_data = bin_data[bin_data['selectedpod'] == pod]
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


def prepare_plot_data(df, unique_pods):
    """Prepare all data needed for plotting"""
    # Calculate request rates for the new subplots
    df['time_bin'] = np.floor(df['relative_time']).astype(int)
    
    # Total requests per second
    total_requests_per_sec = df.groupby('time_bin').size().reset_index(name='total_requests')
    
    # Total input tokens per second
    if 'num_input_tokens' in df.columns:
        input_tokens_per_sec = df.groupby('time_bin')['num_input_tokens'].sum().reset_index()
    else:
        print(f"Error: 'num_input_tokens' column not found in data. Cannot calculate total tokens per second.")
        exit()
    if 'num_output_tokens' in df.columns:
        output_tokens_per_sec = df.groupby('time_bin')['num_output_tokens'].sum().reset_index()
    else:
        print(f"Error: 'num_output_tokens' column not found in data. Cannot calculate total tokens per second.")
        exit()
    
    # Calculate total waiting requests per second
    waiting_requests_per_sec = []
    for time_bin in sorted(df['time_bin'].unique()):
        bin_data = df[df['time_bin'] == time_bin]
        # Get the last entry in this time bin to get the most recent waiting count
        if not bin_data.empty:
            last_entry = bin_data.iloc[-1]
            if last_entry['vllm_num_requests_waiting'] is not None:
                total_waiting = sum(last_entry['vllm_num_requests_waiting'].values())
                waiting_requests_per_sec.append({'time_bin': time_bin, 'total_waiting': total_waiting})
    
    waiting_requests_df = pd.DataFrame(waiting_requests_per_sec)
    
    # Calculate total running requests per second
    running_requests_per_sec = []
    for time_bin in sorted(df['time_bin'].unique()):
        bin_data = df[df['time_bin'] == time_bin]
        # Get the last entry in this time bin to get the most recent running count
        if not bin_data.empty:
            last_entry = bin_data.iloc[-1]
            if last_entry['vllm_num_requests_running'] is not None:
                total_running = sum(last_entry['vllm_num_requests_running'].values())
                running_requests_per_sec.append({'time_bin': time_bin, 'total_running': total_running})
    
    running_requests_df = pd.DataFrame(running_requests_per_sec)
    
    # Calculate total prefill tokens per second across all pods
    prefill_tokens_per_sec = []
    for time_bin in sorted(df['time_bin'].unique()):
        bin_data = df[df['time_bin'] == time_bin]
        if not bin_data.empty:
            last_entry = bin_data.iloc[-1]
            if last_entry['num_prefill_tokens_for_all_pods'] is not None:
                total_prefill = sum(last_entry['num_prefill_tokens_for_all_pods'].values())
                prefill_tokens_per_sec.append({'time_bin': time_bin, 'total_prefill': total_prefill})
    
    prefill_tokens_df = pd.DataFrame(prefill_tokens_per_sec)
    
    # Calculate total decode tokens per second across all pods
    decode_tokens_per_sec = []
    for time_bin in sorted(df['time_bin'].unique()):
        bin_data = df[df['time_bin'] == time_bin]
        if not bin_data.empty:
            last_entry = bin_data.iloc[-1]
            if last_entry['num_decode_tokens_for_all_pods'] is not None:
                total_decode = sum(last_entry['num_decode_tokens_for_all_pods'].values())
                decode_tokens_per_sec.append({'time_bin': time_bin, 'total_decode': total_decode})
    
    decode_tokens_df = pd.DataFrame(decode_tokens_per_sec)
    
    # Requests per second by pod
    requests_per_pod_per_sec = df.groupby(['time_bin', 'selectedpod']).size().reset_index(name='requests')
    
    return {
        'total_requests_per_sec': total_requests_per_sec,
        'input_tokens_per_sec': input_tokens_per_sec,
        'output_tokens_per_sec': output_tokens_per_sec,
        'waiting_requests_df': waiting_requests_df,
        'running_requests_df': running_requests_df,
        'prefill_tokens_df': prefill_tokens_df,
        'decode_tokens_df': decode_tokens_df,
        'requests_per_pod_per_sec': requests_per_pod_per_sec
    }

def extract_pod_specific_data(df, unique_pods):
    """Extract pod-specific data for various metrics"""
    # Extract KV cache hit ratios for selected pods and cluster-wide statistics
    kv_cache_data = []
    for _, row in df.iterrows():
        if row['all_pods_kv_cache_hit_ratios'] is not None and row['selectedpod'] is not None:
            # Extract the hit ratio for the selected pod
            selected_pod_hit_ratio = row['all_pods_kv_cache_hit_ratios'].get(row['selectedpod'], 0)
            # Calculate cluster-wide statistics
            all_values = list(row['all_pods_kv_cache_hit_ratios'].values())
            cluster_avg = sum(all_values) / len(all_values) if all_values else None
            cluster_min = min(all_values) if all_values else None
            cluster_max = max(all_values) if all_values else None
            
            if selected_pod_hit_ratio is not None:
                kv_cache_data.append({
                    'relative_time': row['relative_time'],
                    'selectedpod': row['selectedpod'],
                    'selectedpod_kv_cache_hit_ratio': selected_pod_hit_ratio,
                    'cluster_avg_kv_cache': cluster_avg,
                    'cluster_min_kv_cache': cluster_min,
                    'cluster_max_kv_cache': cluster_max
                })
    
    kv_cache_df = pd.DataFrame(kv_cache_data)
    
    # 1. vllmNumRequestsRunning for selected pod and cluster-wide statistics
    running_requests_data = []
    for _, row in df.iterrows():
        if row['vllm_num_requests_running'] is not None and row['selectedpod'] is not None:
            selected_pod_running = row['vllm_num_requests_running'].get(row['selectedpod'], 0)
            # Calculate cluster-wide statistics
            all_values = list(row['vllm_num_requests_running'].values())
            cluster_avg = sum(all_values) / len(all_values) if all_values else None
            cluster_min = min(all_values) if all_values else None
            cluster_max = max(all_values) if all_values else None
            
            if selected_pod_running is not None:
                running_requests_data.append({
                    'relative_time': row['relative_time'],
                    'selectedpod': row['selectedpod'],
                    'running_requests': selected_pod_running,
                    'cluster_avg_running': cluster_avg,
                    'cluster_min_running': cluster_min,
                    'cluster_max_running': cluster_max
                })
    running_requests_df = pd.DataFrame(running_requests_data)
    
    # 2. numPrefillTokensForAllPods for selected pod and cluster-wide statistics
    prefill_tokens_data = []
    for _, row in df.iterrows():
        if row['num_prefill_tokens_for_all_pods'] is not None and row['selectedpod'] is not None:
            selected_pod_prefill = row['num_prefill_tokens_for_all_pods'].get(row['selectedpod'], 0)
            # Calculate cluster-wide statistics
            all_values = list(row['num_prefill_tokens_for_all_pods'].values())
            cluster_avg = sum(all_values) / len(all_values) if all_values else None
            cluster_min = min(all_values) if all_values else None
            cluster_max = max(all_values) if all_values else None
            
            if selected_pod_prefill is not None:
                prefill_tokens_data.append({
                    'relative_time': row['relative_time'],
                    'selectedpod': row['selectedpod'],
                    'prefill_tokens': selected_pod_prefill,
                    'cluster_avg_prefill': cluster_avg,
                    'cluster_min_prefill': cluster_min,
                    'cluster_max_prefill': cluster_max
                })
    prefill_tokens_df = pd.DataFrame(prefill_tokens_data)
    
    # 3. numDecodeTokensForAllPods for selected pod and cluster-wide statistics
    decode_tokens_data = []
    for _, row in df.iterrows():
        if row['num_decode_tokens_for_all_pods'] is not None and row['selectedpod'] is not None:
            selected_pod_decode = row['num_decode_tokens_for_all_pods'].get(row['selectedpod'], 0)
            # Calculate cluster-wide statistics
            all_values = list(row['num_decode_tokens_for_all_pods'].values())
            cluster_avg = sum(all_values) / len(all_values) if all_values else None
            cluster_min = min(all_values) if all_values else None
            cluster_max = max(all_values) if all_values else None
            
            if selected_pod_decode is not None:
                decode_tokens_data.append({
                    'relative_time': row['relative_time'],
                    'selectedpod': row['selectedpod'],
                    'decode_tokens': selected_pod_decode,
                    'cluster_avg_decode': cluster_avg,
                    'cluster_min_decode': cluster_min,
                    'cluster_max_decode': cluster_max
                })
    decode_tokens_df = pd.DataFrame(decode_tokens_data)
    
    # 4. vllmGPUKVCacheUsage for selected pod and cluster-wide statistics
    gpu_cache_usage_data = []
    for _, row in df.iterrows():
        if row['vllm_gpu_kv_cache_usage'] is not None and row['selectedpod'] is not None:
            selected_pod_gpu_usage = row['vllm_gpu_kv_cache_usage'].get(row['selectedpod'], 0)
            # Calculate cluster-wide statistics
            all_values = list(row['vllm_gpu_kv_cache_usage'].values())
            cluster_avg = sum(all_values) / len(all_values) if all_values else None
            cluster_min = min(all_values) if all_values else None
            cluster_max = max(all_values) if all_values else None
            
            if selected_pod_gpu_usage is not None:
                gpu_cache_usage_data.append({
                    'relative_time': row['relative_time'],
                    'selectedpod': row['selectedpod'],
                    'gpu_cache_usage': selected_pod_gpu_usage,
                    'cluster_avg_gpu': cluster_avg,
                    'cluster_min_gpu': cluster_min,
                    'cluster_max_gpu': cluster_max
                })
    gpu_cache_usage_df = pd.DataFrame(gpu_cache_usage_data)
    
    # 5. vllmNumRequestsWaiting for selected pod and cluster-wide statistics
    waiting_selected_pod_data = []
    for _, row in df.iterrows():
        if row['vllm_num_requests_waiting'] is not None and row['selectedpod'] is not None:
            selected_pod_waiting = row['vllm_num_requests_waiting'].get(row['selectedpod'], 0)
            # Calculate cluster-wide statistics
            all_values = list(row['vllm_num_requests_waiting'].values())
            cluster_avg = sum(all_values) / len(all_values) if all_values else None
            cluster_min = min(all_values) if all_values else None
            cluster_max = max(all_values) if all_values else None
            
            if selected_pod_waiting is not None:
                waiting_selected_pod_data.append({
                    'relative_time': row['relative_time'],
                    'selectedpod': row['selectedpod'],
                    'waiting_requests_selected': selected_pod_waiting,
                    'cluster_avg_waiting': cluster_avg,
                    'cluster_min_waiting': cluster_min,
                    'cluster_max_waiting': cluster_max
                })
    waiting_selected_pod_df = pd.DataFrame(waiting_selected_pod_data)
    
    # Debug: Print info about waiting_selected_pod_df
    print(f"Waiting selected pod data points: {len(waiting_selected_pod_df)}")
    # Save for debugging
    waiting_selected_pod_df.to_csv('waiting_selected_pod_df.csv')
    
    return {
        'kv_cache_df': kv_cache_df,
        'running_requests_df': running_requests_df,
        'prefill_tokens_df': prefill_tokens_df,
        'decode_tokens_df': decode_tokens_df,
        'gpu_cache_usage_df': gpu_cache_usage_df,
        'waiting_selected_pod_df': waiting_selected_pod_df
    }

def add_transition_lines(ax, train_transitions, flush_transitions):
    """Add vertical lines for numTrains transitions only (numFlush transitions removed)"""
    for transition in train_transitions:
        ax.axvline(x=transition['relative_time'], color='purple', linewidth=linewidth, alpha=alpha, zorder=5)

def plot_request_rate_subplots(fig, gs, plot_data, train_transitions, flush_transitions, unique_pods, pod_colors):
    """Plot the request rate analysis subplots"""
    # Request rate analysis subplots
    ax_total_rate = fig.add_subplot(gs[0, :])  # Total requests per second
    ax_token_rate = fig.add_subplot(gs[1, :], sharex=ax_total_rate)  # Total input tokens per second
    ax_pod_rate = fig.add_subplot(gs[2, :], sharex=ax_total_rate)  # Requests per pod per second
    
    # SUBPLOT 1: Total Requests Per Second (ax_total_rate)
    ax_total_rate.plot(plot_data['total_requests_per_sec']['time_bin'], plot_data['total_requests_per_sec']['total_requests'], 
                      '-', color='blue', linewidth=linewidth, alpha=alpha, label='Total RPS')
    add_transition_lines(ax_total_rate, train_transitions, flush_transitions)
    ax_total_rate.set_title('Total Requests Per Second', fontsize=16, fontweight='bold', pad=10)
    ax_total_rate.set_ylabel('Requests/sec', fontsize=14, fontweight='bold')
    ax_total_rate.grid(True, alpha=alpha)
    ax_total_rate.tick_params(axis='both', which='major', labelsize=11)

    # SUBPLOT 2: Total Input Tokens Per Second (ax_token_rate)
    ax_token_rate.plot(plot_data['input_tokens_per_sec']['time_bin'], plot_data['input_tokens_per_sec']['num_input_tokens'], 
                      '-', color='green', linewidth=linewidth, alpha=alpha, label='Total Tokens/sec')
    add_transition_lines(ax_token_rate, train_transitions, flush_transitions)
    ax_token_rate.set_title('Total Input Tokens Per Second', fontsize=16, fontweight='bold', pad=10)
    ax_token_rate.set_ylabel('Tokens/sec', fontsize=14, fontweight='bold')
    ax_token_rate.grid(True, alpha=alpha)
    ax_token_rate.tick_params(axis='both', which='major', labelsize=11)

    # SUBPLOT 3: Requests Per Pod Per Second (ax_pod_rate)
    # Create line plots for requests per pod per second
    pivot_data = plot_data['requests_per_pod_per_sec'].pivot(index='time_bin', columns='selectedpod', values='requests').fillna(0)
    
    # Plot trend lines for each pod
    for pod in unique_pods:
        if pod in pivot_data.columns:
            ax_pod_rate.plot(pivot_data.index, pivot_data[pod], '-', 
                           label=f'Pod {pod}', color=pod_colors[pod], 
                           linewidth=linewidth, alpha=alpha)
    
    add_transition_lines(ax_pod_rate, train_transitions, flush_transitions)
    ax_pod_rate.set_title('Requests Per Second by Pod', fontsize=16, fontweight='bold', pad=10)
    ax_pod_rate.set_ylabel('Requests/sec', fontsize=14, fontweight='bold')
    ax_pod_rate.grid(True, alpha=alpha)
    ax_pod_rate.legend(fontsize=10, loc='upper right')
    ax_pod_rate.tick_params(axis='both', which='major', labelsize=11)

    
    return ax_total_rate, ax_token_rate, ax_pod_rate

def plot_main_metrics_subplots(fig, gs, df, pod_data, cluster_stats, train_transitions, flush_transitions, unique_pods, pod_colors, plot_data):
    """Plot the main metrics (TTFT, KV Cache, Running Requests, etc.)"""
    # Define the main time series plots (starting from row 3 after removing empty rows)
    # Group request-level metrics together
    ax1 = fig.add_subplot(gs[3, :])  # TTFT plot
    ax2 = fig.add_subplot(gs[4, :], sharex=ax1)  # TPOT plot
    ax3 = fig.add_subplot(gs[5, :], sharex=ax1)  # E2E Duration plot
    ax_reward = fig.add_subplot(gs[6, :], sharex=ax1)  # Reward plot
    
    # Pod-level system metrics - grouped with their cluster-wide counterparts
    ax_kv_cache = fig.add_subplot(gs[7, :], sharex=ax1)  # KV Cache Hit Ratio plot
    ax_running_total = fig.add_subplot(gs[8, :], sharex=ax1)  # Total running requests (cluster-wide)
    ax_running = fig.add_subplot(gs[9, :], sharex=ax1)  # Running requests (selected pod)
    ax_prefill_total = fig.add_subplot(gs[10, :], sharex=ax1)  # Total prefill tokens (cluster-wide)
    ax_prefill = fig.add_subplot(gs[11, :], sharex=ax1)  # Prefill tokens (selected pod)
    ax_decode_total = fig.add_subplot(gs[12, :], sharex=ax1)  # Total decode tokens (cluster-wide)
    ax_decode = fig.add_subplot(gs[13, :], sharex=ax1)  # Decode tokens (selected pod)
    ax_gpu_usage = fig.add_subplot(gs[14, :], sharex=ax1)  # GPU cache usage
    ax_waiting = fig.add_subplot(gs[15, :], sharex=ax1)  # Total waiting requests (cluster-wide)
    ax_waiting_selected = fig.add_subplot(gs[16, :], sharex=ax1)  # Waiting requests (selected pod)
    
    # TTFT Plot (ax1)
    for pod in unique_pods:
        pod_df = df[df['selectedpod'] == pod]
        ax1.scatter(pod_df['relative_time'], pod_df['ttft'], s=marker_size, 
                   color=pod_colors[pod], edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha, label=f'Pod: {pod}')

    # Note: Removed cluster-wise statistics as they are redundant with sliding window average
    # Each request belongs to one pod, so cluster min/max across requests per time bin is not meaningful
    
    # Add sliding window average for all TTFT values per second
    df['time_bin'] = np.floor(df['relative_time']).astype(int)
    ttft_avg_per_sec = df.groupby('time_bin')['ttft'].mean().reset_index()
    ax1.plot(ttft_avg_per_sec['time_bin'], ttft_avg_per_sec['ttft'], 'red', '-', linewidth=linewidth, alpha=alpha, label='Avg TTFT (per sec)', zorder=10)
    
    add_transition_lines(ax1, train_transitions, flush_transitions)

    # NEW SUBPLOT: KV Cache Hit Ratio (ax_kv_cache)
    if not pod_data['kv_cache_df'].empty:
        for pod in unique_pods:
            pod_kv_data = pod_data['kv_cache_df'][pod_data['kv_cache_df']['selectedpod'] == pod]
            if not pod_kv_data.empty:
                ax_kv_cache.scatter(pod_kv_data['relative_time'], pod_kv_data['selectedpod_kv_cache_hit_ratio'], s=marker_size, color=pod_colors[pod], edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha) #, label=f'Pod: {pod}')
        
        # Plot cluster-wide statistics
        ax_kv_cache.plot(pod_data['kv_cache_df']['relative_time'], pod_data['kv_cache_df']['cluster_avg_kv_cache'], label='Cluster Avg KV Cache', linewidth=linewidth, alpha=alpha, color='blue')
        ax_kv_cache.plot(pod_data['kv_cache_df']['relative_time'], pod_data['kv_cache_df']['cluster_min_kv_cache'], label='Cluster Min KV Cache', linewidth=linewidth, alpha=alpha, color='green')
        ax_kv_cache.plot(pod_data['kv_cache_df']['relative_time'], pod_data['kv_cache_df']['cluster_max_kv_cache'], label='Cluster Max KV Cache', linewidth=linewidth, alpha=alpha, color='orange')
        
    # Add average KV cache hit ratio per second for selected pod
    if not pod_data['kv_cache_df'].empty:
        # Create time bins for KV cache data
        pod_data['kv_cache_df']['time_bin'] = np.floor(pod_data['kv_cache_df']['relative_time']).astype(int)
        
        # Calculate average KV cache hit ratio per second for selected pod
        avg_kv_cache_per_sec = pod_data['kv_cache_df'].groupby('time_bin')['selectedpod_kv_cache_hit_ratio'].mean().reset_index()
        
        # Plot the average line for selected pod
        ax_kv_cache.plot(avg_kv_cache_per_sec['time_bin'], avg_kv_cache_per_sec['selectedpod_kv_cache_hit_ratio'], 'red', '-', linewidth=linewidth, alpha=alpha, label='Avg Selected Pod KV Cache (per sec)')

    pod_data['kv_cache_df'].to_csv('kv_cache_df.csv')
    add_transition_lines(ax_kv_cache, train_transitions, flush_transitions)
    ax_kv_cache.set_title('KV Cache Hit Ratio for Selected Pod per Request', fontsize=16, fontweight='bold', pad=10)
    ax_kv_cache.set_ylabel('KV Cache Hit Ratio', fontsize=14, fontweight='bold')
    ax_kv_cache.set_ylim(0, 110)  # Hit ratio is between 0 and 100
    ax_kv_cache.grid(True, alpha=alpha)
    ax_kv_cache.tick_params(axis='both', which='major', labelsize=11)
    # bring legend forward
    legend_object = ax_kv_cache.legend(loc='upper right', fontsize=10)
    legend_object.set_zorder(10)

    # Plot other metrics subplots (Running Requests, Prefill Tokens, etc.)
    _plot_pod_metric_subplot(ax_running, pod_data['running_requests_df'], 'running_requests', 'Running Requests', 
                           'vllmNumRequestsRunning for Selected Pod per Request', unique_pods, pod_colors, 
                           train_transitions, flush_transitions)
    
    _plot_pod_metric_subplot(ax_prefill, pod_data['prefill_tokens_df'], 'prefill_tokens', 'Prefill Tokens',
                           'numPrefillTokensForAllPods for Selected Pod per Request', unique_pods, pod_colors,
                           train_transitions, flush_transitions)
    
    _plot_pod_metric_subplot(ax_decode, pod_data['decode_tokens_df'], 'decode_tokens', 'Decode Tokens',
                           'numDecodeTokensForAllPods for Selected Pod per Request', unique_pods, pod_colors,
                           train_transitions, flush_transitions)
    
    _plot_pod_metric_subplot(ax_gpu_usage, pod_data['gpu_cache_usage_df'], 'gpu_cache_usage', 'GPU Cache Usage',
                           'vllmGPUKVCacheUsage for Selected Pod per Request', unique_pods, pod_colors,
                           train_transitions, flush_transitions, ylim=(0, 1.1))
    
    if not pod_data['waiting_selected_pod_df'].empty:
        for pod in unique_pods:
            pod_waiting_data = pod_data['waiting_selected_pod_df'][pod_data['waiting_selected_pod_df']['selectedpod'] == pod]
            if not pod_waiting_data.empty:
                print(f"Plotting {len(pod_waiting_data)} points for pod {pod}")
                ax_waiting_selected.scatter(pod_waiting_data['relative_time'], pod_waiting_data['waiting_requests_selected'], 
                                          s=marker_size, color=pod_colors[pod], edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha, label=f'Pod: {pod}')
        
        # Add average waiting requests for selected pod per second
        pod_data['waiting_selected_pod_df']['time_bin'] = np.floor(pod_data['waiting_selected_pod_df']['relative_time']).astype(int)
        avg_waiting_selected_per_sec = pod_data['waiting_selected_pod_df'].groupby('time_bin')['waiting_requests_selected'].mean().reset_index()
        print(f"Average data points: {len(avg_waiting_selected_per_sec)}")
        if not avg_waiting_selected_per_sec.empty:
            ax_waiting_selected.plot(avg_waiting_selected_per_sec['time_bin'], avg_waiting_selected_per_sec['waiting_requests_selected'], 
                                   'red', '-', linewidth=linewidth, alpha=alpha, label='Average Waiting Requests (Selected Pod)', zorder=10)
    else:
        print("No data to plot for waiting selected pod - adding placeholder text")
        ax_waiting_selected.text(0.5, 0.5, 'No Data Available', transform=ax_waiting_selected.transAxes, 
                               ha='center', va='center', fontsize=16, alpha=alpha)
    
    add_transition_lines(ax_waiting_selected, train_transitions, flush_transitions)
    ax_waiting_selected.set_title('vllmNumRequestsWaiting for Selected Pod per Request', fontsize=16, fontweight='bold', pad=10)
    ax_waiting_selected.set_ylabel('Waiting Requests (Selected Pod)', fontsize=14, fontweight='bold')
    ax_waiting_selected.grid(True, alpha=alpha)
    ax_waiting_selected.tick_params(axis='both', which='major', labelsize=11)

    # TPOT Plot (ax2)
    for pod in unique_pods:
        pod_df = df[df['selectedpod'] == pod]
        ax2.scatter(pod_df['relative_time'], pod_df['avg_tpot'], s=marker_size, color=pod_colors[pod], edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha)

    # Note: Removed cluster-wise statistics as they are redundant with sliding window average
    # Each request belongs to one pod, so cluster min/max across requests per time bin is not meaningful
    
    # Add sliding window average for all TPOT values per second
    tpot_avg_per_sec = df.groupby('time_bin')['avg_tpot'].mean().reset_index()
    ax2.plot(tpot_avg_per_sec['time_bin'], tpot_avg_per_sec['avg_tpot'], 'red', '-', linewidth=linewidth, alpha=alpha, label='Avg TPOT (per sec)', zorder=10)
    
    add_transition_lines(ax2, train_transitions, flush_transitions)

    # E2E Duration Plot (ax3)
    for pod in unique_pods:
        pod_df = df[df['selectedpod'] == pod]
        ax3.scatter(pod_df['relative_time'], pod_df['e2e'], s=marker_size, color=pod_colors[pod], edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha)
    
    # Note: Removed cluster-wise statistics as they are redundant with sliding window average  
    # Each request belongs to one pod, so cluster min/max across requests per time bin is not meaningful
    
    # Add sliding window average for all E2E values per second
    e2e_avg_per_sec = df.groupby('time_bin')['e2e'].mean().reset_index()
    ax3.plot(e2e_avg_per_sec['time_bin'], e2e_avg_per_sec['e2e'], 'red', '-', linewidth=linewidth, alpha=alpha, label='Avg E2E (per sec)', zorder=10)
    
    add_transition_lines(ax3, train_transitions, flush_transitions)
    
    # Set titles and labels for main plots
    ax1.set_title('Time to First Token (TTFT) for Each Request', fontsize=16, fontweight='bold', pad=10)
    ax1.set_ylabel('TTFT (ms)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=alpha)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.tick_params(axis='y', which='major', labelsize=11, labelleft=True)
    
    # Add legend for TTFT - only meaningful metrics, no pod labels
    legend_elements_ttft = []
    legend_labels_ttft = []
    legend_elements_ttft.extend([
        Line2D([0], [0], color='red', linewidth=linewidth, label='Avg TTFT (per sec)'),
        Line2D([0], [0], color='purple', linewidth=linewidth, label='numTrains transition')
    ])
    legend_labels_ttft.extend(['Avg TTFT (per sec)', 'numTrains transition'])
    ax1.legend(legend_elements_ttft, legend_labels_ttft, fontsize=10, loc='upper right', ncol=2)
    
    ax2.set_title('Average Time Per Output Token (TPOT) for Each Request', fontsize=16, fontweight='bold', pad=10)
    ax2.set_ylabel('Average TPOT (ms)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=alpha)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    ax2.tick_params(axis='y', which='major', labelsize=11, labelleft=True)
    
    # Add legend for TPOT - only meaningful metrics
    legend_elements_tpot = [
        Line2D([0], [0], color='red', linewidth=linewidth, label='Avg TPOT (per sec)')
    ]
    ax2.legend(legend_elements_tpot, [elem.get_label() for elem in legend_elements_tpot], fontsize=10, loc='upper right')
    
    ax3.set_title('End-to-End Request Duration for Each Request', fontsize=16, fontweight='bold', pad=10)
    # ax3.set_xlabel('Relative Time (seconds)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('E2E (ms)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=alpha)
    ax3.tick_params(axis='both', which='major', labelsize=11)
    ax3.tick_params(axis='y', which='major', labelsize=11, labelleft=True)
    
    # Add legend for E2E - only meaningful metrics
    legend_elements_e2e = [
        Line2D([0], [0], color='red', linewidth=linewidth, label='Avg E2E (per sec)')
    ]
    ax3.legend(legend_elements_e2e, [elem.get_label() for elem in legend_elements_e2e], fontsize=10, loc='upper right')
    
    # Reward Plot (ax_reward)
    for pod in unique_pods:
        pod_df = df[df['selectedpod'] == pod]
        ax_reward.scatter(pod_df['relative_time'], pod_df['total_reward'], s=marker_size, 
                         color=pod_colors[pod], edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha, label=f'Pod: {pod}')
    
    # Note: Removed cluster-wise statistics as they are redundant with sliding window average
    # Each request belongs to one pod, so cluster min/max across requests per time bin is not meaningful
    
    # Add sliding window average for all Reward values per second
    reward_avg_per_sec = df.groupby('time_bin')['total_reward'].mean().reset_index()
    ax_reward.plot(reward_avg_per_sec['time_bin'], reward_avg_per_sec['total_reward'], 'red', '-', linewidth=linewidth, alpha=alpha, label='Avg Reward (per sec)', zorder=10)
    
    add_transition_lines(ax_reward, train_transitions, flush_transitions)
    
    ax_reward.set_title('Total Reward for Each Request', fontsize=16, fontweight='bold', pad=10)
    ax_reward.set_ylabel('Total Reward', fontsize=14, fontweight='bold')
    ax_reward.grid(True, alpha=alpha)
    ax_reward.tick_params(axis='both', which='major', labelsize=11)
    ax_reward.tick_params(axis='y', which='major', labelsize=11, labelleft=True)
    
    # Add legend for Reward - only meaningful metrics
    legend_elements_reward = [
        Line2D([0], [0], color='red', linewidth=linewidth, label='Avg Reward (per sec)')
    ]
    ax_reward.legend(legend_elements_reward, [elem.get_label() for elem in legend_elements_reward], fontsize=10, loc='upper right')
    
    # Plot the grouped cluster-wide total plots
    
    # Total Running Requests (ax_running_total)
    if 'running_requests_df' in plot_data and not plot_data['running_requests_df'].empty:
        ax_running_total.plot(plot_data['running_requests_df']['time_bin'], plot_data['running_requests_df']['total_running'], 
                             '-', color='orange', linewidth=linewidth, alpha=alpha, label='Total Running Requests')
    add_transition_lines(ax_running_total, train_transitions, flush_transitions)
    ax_running_total.set_title('Total vllmNumRequestsRunning Across All Pods', fontsize=16, fontweight='bold', pad=10)
    ax_running_total.set_ylabel('Running Requests', fontsize=14, fontweight='bold')
    ax_running_total.grid(True, alpha=alpha)
    ax_running_total.tick_params(axis='both', which='major', labelsize=11)
    
    # Running Requests for Selected Pod (ax_running)
    _plot_pod_metric_subplot(ax_running, pod_data['running_requests_df'], 'running_requests', 'Running Requests', 
                           'vllmNumRequestsRunning for Selected Pod per Request', unique_pods, pod_colors, train_transitions, flush_transitions)
    
    # Total Prefill Tokens (ax_prefill_total)
    if 'prefill_tokens_df' in plot_data and not plot_data['prefill_tokens_df'].empty:
        ax_prefill_total.plot(plot_data['prefill_tokens_df']['time_bin'], plot_data['prefill_tokens_df']['total_prefill'], 
                             '-', color='purple', linewidth=linewidth, alpha=alpha, label='Total Prefill Tokens')
    add_transition_lines(ax_prefill_total, train_transitions, flush_transitions)
    ax_prefill_total.set_title('Total numPrefillTokensForAllPods Across All Pods', fontsize=16, fontweight='bold', pad=10)
    ax_prefill_total.set_ylabel('Prefill Tokens', fontsize=14, fontweight='bold')
    ax_prefill_total.grid(True, alpha=alpha)
    ax_prefill_total.tick_params(axis='both', which='major', labelsize=11)
    
    # Prefill Tokens for Selected Pod (ax_prefill)
    _plot_pod_metric_subplot(ax_prefill, pod_data['prefill_tokens_df'], 'prefill_tokens', 'Prefill Tokens', 
                           'numPrefillTokensForAllPods for Selected Pod per Request', unique_pods, pod_colors, train_transitions, flush_transitions)

    # Total Decode Tokens (ax_decode_total)
    if 'decode_tokens_df' in plot_data and not plot_data['decode_tokens_df'].empty:
        ax_decode_total.plot(plot_data['decode_tokens_df']['time_bin'], plot_data['decode_tokens_df']['total_decode'], 
                            '-', color='brown', linewidth=linewidth, alpha=alpha, label='Total Decode Tokens')
    add_transition_lines(ax_decode_total, train_transitions, flush_transitions)
    ax_decode_total.set_title('Total numDecodeTokensForAllPods Across All Pods', fontsize=16, fontweight='bold', pad=10)
    ax_decode_total.set_ylabel('Decode Tokens', fontsize=14, fontweight='bold')
    ax_decode_total.grid(True, alpha=alpha)
    ax_decode_total.tick_params(axis='both', which='major', labelsize=11)
    
    # Decode Tokens for Selected Pod (ax_decode)
    _plot_pod_metric_subplot(ax_decode, pod_data['decode_tokens_df'], 'decode_tokens', 'Decode Tokens', 
                           'numDecodeTokensForAllPods for Selected Pod per Request', unique_pods, pod_colors, train_transitions, flush_transitions)

    # Total Waiting Requests (ax_waiting)
    if 'waiting_requests_df' in plot_data and not plot_data['waiting_requests_df'].empty:
        ax_waiting.plot(plot_data['waiting_requests_df']['time_bin'], plot_data['waiting_requests_df']['total_waiting'], 
                       '-', color='red', linewidth=linewidth, alpha=alpha, label='Total Waiting Requests')
    add_transition_lines(ax_waiting, train_transitions, flush_transitions)
    ax_waiting.set_title('Total vllmNumRequestsWaiting Across All Pods', fontsize=16, fontweight='bold', pad=10)
    ax_waiting.set_ylabel('Waiting Requests', fontsize=14, fontweight='bold')
    ax_waiting.grid(True, alpha=alpha)
    ax_waiting.tick_params(axis='both', which='major', labelsize=11)
    
    # Waiting Requests for Selected Pod (ax_waiting_selected)
    _plot_pod_metric_subplot(ax_waiting_selected, pod_data['waiting_selected_pod_df'], 'waiting_requests_selected', 'Waiting Requests', 
                           'vllmNumRequestsWaiting for Selected Pod per Request', unique_pods, pod_colors, train_transitions, flush_transitions)
    
    return ax1, ax2, ax3, ax_reward, ax_kv_cache, ax_running_total, ax_running, ax_prefill_total, ax_prefill, ax_decode_total, ax_decode, ax_gpu_usage, ax_waiting, ax_waiting_selected

def _plot_pod_metric_subplot(ax, data_df, metric_col, metric_name, title, unique_pods, pod_colors, train_transitions, flush_transitions, ylim=None):
    """Helper function to plot pod metric subplots"""
    # Define cluster column names first
    if metric_col == 'waiting_requests_selected':
        cluster_avg_col = 'cluster_avg_waiting'
        cluster_min_col = 'cluster_min_waiting'
        cluster_max_col = 'cluster_max_waiting'
    elif metric_col == 'gpu_cache_usage':
        cluster_avg_col = 'cluster_avg_gpu'
        cluster_min_col = 'cluster_min_gpu'
        cluster_max_col = 'cluster_max_gpu'
    elif metric_col == 'running_requests':
        cluster_avg_col = 'cluster_avg_running'
        cluster_min_col = 'cluster_min_running'
        cluster_max_col = 'cluster_max_running'
    elif metric_col == 'prefill_tokens':
        cluster_avg_col = 'cluster_avg_prefill'
        cluster_min_col = 'cluster_min_prefill'
        cluster_max_col = 'cluster_max_prefill'
    elif metric_col == 'decode_tokens':
        cluster_avg_col = 'cluster_avg_decode'
        cluster_min_col = 'cluster_min_decode'
        cluster_max_col = 'cluster_max_decode'
    else:
        # Extract the last part of the metric column name for other metrics
        suffix = metric_col.split("_")[-1]
        cluster_avg_col = f'cluster_avg_{suffix}'
        cluster_min_col = f'cluster_min_{suffix}'
        cluster_max_col = f'cluster_max_{suffix}'
    
    if not data_df.empty:
        for pod in unique_pods:
            pod_data = data_df[data_df['selectedpod'] == pod]
            if not pod_data.empty:
                ax.scatter(pod_data['relative_time'], pod_data[metric_col], 
                         s=marker_size, color=pod_colors[pod], edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha, label=f'Pod: {pod}')
        
        # Plot cluster-wide statistics
        
        if cluster_avg_col in data_df.columns:
            ax.plot(data_df['relative_time'], data_df[cluster_avg_col], label=f'Cluster Avg {metric_name}', linewidth=linewidth, alpha=alpha, color='blue')
        if cluster_min_col in data_df.columns:
            ax.plot(data_df['relative_time'], data_df[cluster_min_col], label=f'Cluster Min {metric_name}', linewidth=linewidth, alpha=alpha, color='green')
        if cluster_max_col in data_df.columns:
            ax.plot(data_df['relative_time'], data_df[cluster_max_col], label=f'Cluster Max {metric_name}', linewidth=linewidth, alpha=alpha, color='orange')
        
        # Add average per second for selected pod
        data_df['time_bin'] = np.floor(data_df['relative_time']).astype(int)
        avg_per_sec = data_df.groupby('time_bin')[metric_col].mean().reset_index()
        ax.plot(avg_per_sec['time_bin'], avg_per_sec[metric_col], 
               'red', '-', linewidth=linewidth, alpha=alpha, label=f'Avg Selected Pod {metric_name} (per sec)', zorder=10)
    
    add_transition_lines(ax, train_transitions, flush_transitions)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add legend - only cluster metrics, no pod labels
    legend_elements = []
    if cluster_avg_col in data_df.columns:
        legend_elements.append(Line2D([0], [0], color='blue', linewidth=linewidth, label=f'Cluster Avg {metric_name}'))
    if cluster_min_col in data_df.columns:
        legend_elements.append(Line2D([0], [0], color='green', linewidth=linewidth, label=f'Cluster Min {metric_name}'))
    if cluster_max_col in data_df.columns:
        legend_elements.append(Line2D([0], [0], color='orange', linewidth=linewidth, label=f'Cluster Max {metric_name}'))
    if len(legend_elements) > 0:
        legend_elements.append(Line2D([0], [0], color='red', linewidth=linewidth, label=f'Avg Selected Pod {metric_name} (per sec)'))
        ax.legend(legend_elements, [elem.get_label() for elem in legend_elements], fontsize=10, loc='upper right')

def plot_analysis_subplots(fig, gs, df, slo_stats, slo_ttft, slo_tpot, unique_pods, pod_colors):
    """Plot the pod analysis and CDF distribution subplots"""
    # Define the pod analysis plots (updated indices)
    ax4 = fig.add_subplot(gs[17, 0])  # Average TTFT by Pod
    ax5 = fig.add_subplot(gs[17, 1])  # Average TPOT by Pod
    ax_slo = fig.add_subplot(gs[17, 2])  # SLO satisfaction
    
    # Define the CDF distribution plots  
    ax6 = fig.add_subplot(gs[18, 0])  # TTFT CDF
    ax7 = fig.add_subplot(gs[18, 1])  # TPOT CDF
    ax8 = fig.add_subplot(gs[18, 2])  # E2E CDF
    
    # Average TTFT by Pod (ax4)
    pod_avg_ttft = df.groupby('selectedpod')['ttft'].mean().sort_values(ascending=False)
    pod_counts = df.groupby('selectedpod').size()
    
    # Create bars for TTFT plot
    ttft_bars = ax4.bar(range(len(pod_avg_ttft)), pod_avg_ttft.values, 
                      color=[pod_colors[pod] for pod in pod_avg_ttft.index], 
                      edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha, width=0.7)
    
    # Add annotations
    for i, (pod, ttft) in enumerate(pod_avg_ttft.items()):
        ax4.text(i, ttft + 5, f'{ttft:.1f} ms', ha='center', fontsize=12, fontweight='bold')
        ax4.text(i, 10, f'n={pod_counts[pod]}', ha='center', fontsize=10)
    
    ax4.set_xticks(range(len(pod_avg_ttft)))
    ax4.set_xticklabels([f'Pod {pod}' for pod in pod_avg_ttft.index], rotation=45, ha='right', fontsize=11)
    ax4.set_ylabel('Average TTFT (ms)', fontsize=12, fontweight='bold')
    ax4.set_title('Average TTFT by Pod', fontsize=14, fontweight='bold', pad=10)
    ax4.grid(axis='y', linestyle='--', alpha=alpha)
    
    # Average TPOT by Pod (ax5)
    pod_avg_tpot = df.groupby('selectedpod')['avg_tpot'].mean().sort_values(ascending=False)
    
    # Create bars for TPOT plot
    tpot_bars = ax5.bar(range(len(pod_avg_tpot)), pod_avg_tpot.values,
                      color=[pod_colors[pod] for pod in pod_avg_tpot.index], 
                      edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha, width=0.7)
    
    # Add annotations
    for i, (pod, tpot) in enumerate(pod_avg_tpot.items()):
        ax5.text(i, tpot + 1, f'{tpot:.1f} ms', ha='center', fontsize=12, fontweight='bold')
        ax5.text(i, 0.5, f'n={pod_counts[pod]}', ha='center', fontsize=10)
    
    ax5.set_xticks(range(len(pod_avg_tpot)))
    ax5.set_xticklabels([f'Pod {pod}' for pod in pod_avg_tpot.index], rotation=45, ha='right', fontsize=11)
    ax5.set_ylabel('Avg TPOT (ms)', fontsize=12, fontweight='bold')
    ax5.set_title('Average TPOT by Pod', fontsize=14, fontweight='bold', pad=10)
    ax5.grid(axis='y', linestyle='--', alpha=alpha)

    # SLO SATISFACTION BAR CHART (ax_slo)
    categories = ['TTFT SLO\nSatisfied', 'TPOT SLO\nSatisfied', 'Both SLOs\nSatisfied']
    satisfied_counts = [slo_stats['ttft_satisfied'], slo_stats['tpot_satisfied'], slo_stats['both_satisfied']]
    satisfaction_rates = [slo_stats['ttft_satisfaction_rate'], slo_stats['tpot_satisfaction_rate'], slo_stats['both_satisfaction_rate']]
    
    bars = ax_slo.bar(categories, satisfied_counts, color=['blue', 'green', 'red'], alpha=alpha, edgecolor=edgecolor)
    
    # Add percentage labels on bars
    for i, (bar, rate) in enumerate(zip(bars, satisfaction_rates)):
        height = bar.get_height()
        ax_slo.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}\n({rate:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax_slo.set_title(f'SLO Satisfaction Summary\n(Total: {slo_stats["total_requests"]})', fontsize=12, fontweight='bold', pad=10)
    ax_slo.set_ylabel('# Requests', fontsize=10, fontweight='bold')
    ax_slo.set_ylim(0, slo_stats['total_requests'] * 1.1)
    ax_slo.grid(axis='y', linestyle='--', alpha=alpha)
    ax_slo.tick_params(axis='x', rotation=45, labelsize=9)
    
    # Add SLO values as text
    ax_slo.text(0.02, 0.98, f'SLO Thresholds:\nTTFT ≤ {slo_ttft} ms\nTPOT ≤ {slo_tpot} ms', 
             transform=ax_slo.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=alpha))
    
    # CDF Distribution plots (ax6, ax7, ax8)
    # TTFT CDF (ax6)
    sorted_ttft = np.sort(df['ttft'])
    y_ttft = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
    ax6.plot(sorted_ttft, y_ttft, color='blue', linewidth=linewidth, alpha=alpha)
    ax6.set_xlabel('TTFT (ms)', fontsize=10)
    ax6.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax6.set_title('TTFT CDF', fontsize=12, fontweight='bold', pad=10)
    ax6.grid(True, alpha=alpha)
    
    # Add percentile lines and average
    p50_ttft = np.percentile(df['ttft'], 50)
    p95_ttft = np.percentile(df['ttft'], 95)
    p99_ttft = np.percentile(df['ttft'], 99)
    avg_ttft = df['ttft'].mean()
    ax6.axvline(p50_ttft, color='red', linestyle='--', alpha=alpha, label=f'P50: {p50_ttft:.1f}ms')
    ax6.axvline(p95_ttft, color='orange', linestyle='--', alpha=alpha, label=f'P95: {p95_ttft:.1f}ms')
    ax6.axvline(p99_ttft, color='purple', linestyle='--', alpha=alpha, label=f'P99: {p99_ttft:.1f}ms')
    ax6.axvline(avg_ttft, color='green', linestyle='-', alpha=alpha, label=f'Avg: {avg_ttft:.1f}ms')
    ax6.legend(fontsize=8)
    
    # TPOT CDF (ax7)
    sorted_tpot = np.sort(df['avg_tpot'])
    y_tpot = np.arange(1, len(sorted_tpot) + 1) / len(sorted_tpot)
    ax7.plot(sorted_tpot, y_tpot, color='green', linewidth=linewidth, alpha=alpha)
    ax7.set_xlabel('TPOT (ms)', fontsize=10)
    ax7.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax7.set_title('TPOT CDF', fontsize=12, fontweight='bold', pad=10)
    ax7.grid(True, alpha=alpha)
    
    # Add percentile lines and average
    p50_tpot = np.percentile(df['avg_tpot'], 50)
    p95_tpot = np.percentile(df['avg_tpot'], 95)
    p99_tpot = np.percentile(df['avg_tpot'], 99)
    avg_tpot = df['avg_tpot'].mean()
    ax7.axvline(p50_tpot, color='red', linestyle='--', alpha=alpha, label=f'P50: {p50_tpot:.1f}ms')
    ax7.axvline(p95_tpot, color='orange', linestyle='--', alpha=alpha, label=f'P95: {p95_tpot:.1f}ms')
    ax7.axvline(p99_tpot, color='purple', linestyle='--', alpha=alpha, label=f'P99: {p99_tpot:.1f}ms')
    ax7.axvline(avg_tpot, color='green', linestyle='-', alpha=alpha, label=f'Avg: {avg_tpot:.1f}ms')
    ax7.legend(fontsize=8)
    
    # E2E CDF (ax8)
    sorted_e2e = np.sort(df['e2e'])
    y_e2e = np.arange(1, len(sorted_e2e) + 1) / len(sorted_e2e)
    ax8.plot(sorted_e2e, y_e2e, color='purple', linewidth=linewidth, alpha=alpha)
    ax8.set_xlabel('E2E (ms)', fontsize=10)
    ax8.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax8.set_title('E2E CDF', fontsize=12, fontweight='bold', pad=10)
    ax8.grid(True, alpha=alpha)
    
    # Add percentile lines and average
    p50_e2e = np.percentile(df['e2e'], 50)
    p95_e2e = np.percentile(df['e2e'], 95)
    p99_e2e = np.percentile(df['e2e'], 99)
    avg_e2e = df['e2e'].mean()
    ax8.axvline(p50_e2e, color='red', linestyle='--', alpha=alpha, label=f'P50: {p50_e2e:.1f}ms')
    ax8.axvline(p95_e2e, color='orange', linestyle='--', alpha=alpha, label=f'P95: {p95_e2e:.1f}ms')
    ax8.axvline(p99_e2e, color='purple', linestyle='--', alpha=alpha, label=f'P99: {p99_e2e:.1f}ms')
    ax8.axvline(avg_e2e, color='green', linestyle='-', alpha=alpha, label=f'Avg: {avg_e2e:.1f}ms')
    ax8.legend(fontsize=8)
    
    # New subplots for numTrains analysis (updated indices)
    ax9 = fig.add_subplot(gs[19, 0])  # TTFT CDF by numTrains
    ax10 = fig.add_subplot(gs[19, 1])  # TPOT CDF by numTrains  
    ax11 = fig.add_subplot(gs[19, 2])  # Trend plot
    
    # Get unique num_trains values
    unique_num_trains = sorted(df['num_trains'].unique())
    ttft_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_num_trains)))
    tpot_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_num_trains)))
    num_trains_ttft_colors = dict(zip(unique_num_trains, ttft_colors))
    num_trains_tpot_colors = dict(zip(unique_num_trains, tpot_colors))
    
    # TTFT CDF by numTrains (ax9)
    for num_trains in unique_num_trains:
        subset = df[df['num_trains'] == num_trains]
        if len(subset) > 0:
            sorted_ttft = np.sort(subset['ttft'])
            y = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
            
            # Calculate stats
            avg_ttft = subset['ttft'].mean()
            p99_ttft = np.percentile(subset['ttft'], 99)
            
            # Plot CDF line with combined label
            ax9.plot(sorted_ttft, y, color=num_trains_ttft_colors[num_trains], 
                    linewidth=linewidth, alpha=alpha, label=f'numTrains={num_trains}, avg: {avg_ttft:.0f}ms, p99: {p99_ttft:.0f}ms')
            
            # Add vertical lines without labels
            ax9.axvline(avg_ttft, color=num_trains_ttft_colors[num_trains], linestyle='-', alpha=alpha, linewidth=linewidth)
            ax9.axvline(p99_ttft, color=num_trains_ttft_colors[num_trains], linestyle='--', alpha=alpha, linewidth=linewidth)
    
    ax9.set_xlabel('TTFT (ms)', fontsize=10)
    ax9.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax9.set_title('TTFT CDF by numTrains', fontsize=12, fontweight='bold', pad=10)
    ax9.grid(True, alpha=alpha)
    ax9.legend(fontsize=6)
    
    # TPOT CDF by numTrains (ax10)
    for num_trains in unique_num_trains:
        subset = df[df['num_trains'] == num_trains]
        if len(subset) > 0:
            sorted_tpot = np.sort(subset['avg_tpot'])
            y = np.arange(1, len(sorted_tpot) + 1) / len(sorted_tpot)
            
            # Calculate stats
            avg_tpot = subset['avg_tpot'].mean()
            p99_tpot = np.percentile(subset['avg_tpot'], 99)
            
            # Plot CDF line with combined label
            ax10.plot(sorted_tpot, y, color=num_trains_tpot_colors[num_trains], 
                     linewidth=linewidth, alpha=alpha, label=f'numTrains={num_trains}, avg: {avg_tpot:.0f}ms, p99: {p99_tpot:.0f}ms')
            
            # Add vertical lines without labels
            ax10.axvline(avg_tpot, color=num_trains_tpot_colors[num_trains], linestyle='-', alpha=alpha, linewidth=linewidth)
            ax10.axvline(p99_tpot, color=num_trains_tpot_colors[num_trains], linestyle='--', alpha=alpha, linewidth=linewidth)
    
    ax10.set_xlabel('TPOT (ms)', fontsize=10)
    ax10.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax10.set_title('TPOT CDF by numTrains', fontsize=12, fontweight='bold', pad=10)
    ax10.grid(True, alpha=alpha)
    ax10.legend(fontsize=6)
    
    # Trend plot: avg TTFT, p99 TTFT, avg TPOT, p99 TPOT vs numTrains (ax11)
    num_trains_stats = []
    for num_trains in unique_num_trains:
        subset = df[df['num_trains'] == num_trains]
        if len(subset) > 0:
            avg_ttft = subset['ttft'].mean()
            p99_ttft = np.percentile(subset['ttft'], 99)
            avg_tpot = subset['avg_tpot'].mean()
            p99_tpot = np.percentile(subset['avg_tpot'], 99)
            num_trains_stats.append((num_trains, avg_ttft, p99_ttft, avg_tpot, p99_tpot))
    
    if num_trains_stats:
        num_trains_vals, avg_ttft_vals, p99_ttft_vals, avg_tpot_vals, p99_tpot_vals = zip(*num_trains_stats)
        
        # Compute overall statistics for legend
        overall_avg_ttft = df['ttft'].mean()
        overall_p99_ttft = np.percentile(df['ttft'], 99)
        overall_avg_tpot = df['avg_tpot'].mean()
        overall_p99_tpot = np.percentile(df['avg_tpot'], 99)
        
        # Create dual y-axis plot
        # Left axis (ax11) for TTFT
        ax11.plot(num_trains_vals, avg_ttft_vals, marker='o', linestyle='-', color='blue', linewidth=linewidth, alpha=alpha, label=f'Avg TTFT: {overall_avg_ttft:.1f}ms')
        ax11.plot(num_trains_vals, p99_ttft_vals, marker='x', linestyle='--', color='blue', linewidth=linewidth, alpha=alpha, label=f'P99 TTFT: {overall_p99_ttft:.1f}ms')
        ax11.set_xlabel('numTrains', fontsize=10)
        ax11.set_ylabel('TTFT (ms)', fontsize=10, fontweight='bold', color='blue')
        ax11.tick_params(axis='y', labelcolor='blue')
        
        # Right axis for TPOT
        ax11_right = ax11.twinx()
        ax11_right.plot(num_trains_vals, avg_tpot_vals, marker='o', linestyle='-', color='green', linewidth=linewidth, alpha=alpha, label=f'Avg TPOT: {overall_avg_tpot:.1f}ms')
        ax11_right.plot(num_trains_vals, p99_tpot_vals, marker='x', linestyle='--', color='green', linewidth=linewidth, alpha=alpha, label=f'P99 TPOT: {overall_p99_tpot:.1f}ms')
        ax11_right.set_ylabel('TPOT (ms)', fontsize=10, fontweight='bold', color='green')
        ax11_right.tick_params(axis='y', labelcolor='green')
        
        ax11.set_title('Latency Trends by numTrains', fontsize=12, fontweight='bold', pad=10)
        ax11.grid(True, alpha=alpha)
        ax11.set_xticks(num_trains_vals)
        
        # Combine legends from both axes
        lines1, labels1 = ax11.get_legend_handles_labels()
        lines2, labels2 = ax11_right.get_legend_handles_labels()
        ax11.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    
    
    return [ax4, ax5, ax_slo, ax6, ax7, ax8, ax9, ax10, ax11]

def plot_prediction_analysis_subplots(fig, gs, df, train_transitions, flush_transitions, unique_pods, pod_colors, ax1, routing_policy):
    """Plot prediction analysis subplots: actual vs predicted latency comparison and time series"""
    # Determine the target latency metric based on routing policy
    if 'latency_predictor_ttft' in routing_policy:
        actual_col = 'ttft'
        metric_name = 'TTFT'
        ylabel = 'TTFT (ms)'
        title_scatter = 'Actual vs Predicted TTFT Comparison'
        title_timeseries = 'TTFT Time Series with Predictions'
    elif 'latency_predictor_avg_tpot' in routing_policy:
        actual_col = 'avg_tpot'
        metric_name = 'TPOT'
        ylabel = 'TPOT (ms)'
        title_scatter = 'Actual vs Predicted TPOT Comparison'
        title_timeseries = 'TPOT Time Series with Predictions'
    elif 'latency_predictor_e2e_latency' in routing_policy:
        actual_col = 'e2e'
        metric_name = 'E2E Latency'
        ylabel = 'E2E Latency (ms)'
        title_scatter = 'Actual vs Predicted E2E Latency Comparison'
        title_timeseries = 'E2E Latency Time Series with Predictions'
    else:
        # Fallback to E2E for any other latency predictor
        actual_col = 'e2e'
        metric_name = 'E2E Latency'
        ylabel = 'E2E Latency (ms)'
        title_scatter = 'Actual vs Predicted E2E Latency Comparison'
        title_timeseries = 'E2E Latency Time Series with Predictions'

    # Define the prediction analysis plots (rows 20-21)
    ax_pred_scatter = fig.add_subplot(gs[20, :])  # Actual vs Predicted Latency Scatter Plot (full width)
    ax_pred_timeseries = fig.add_subplot(gs[21, :], sharex=ax1)  # Prediction Time Series (share x-axis with other time series)

    # SUBPLOT 1: Actual vs Predicted Latency Scatter Plot (ax_pred_scatter)
    # Filter out entries where predicted latency is None or 0 (no prediction made)
    valid_predictions = df[(df['chosen_pod_predicted_latency'].notna()) &
                          (df['chosen_pod_predicted_latency'] > 0) &
                          (df[actual_col].notna()) &
                          (df[actual_col] > 0)]

    if not valid_predictions.empty:
        # Scatter plot of actual vs predicted
        ax_pred_scatter.scatter(valid_predictions[actual_col], valid_predictions['chosen_pod_predicted_latency'],
                               s=10, color='tab:pink', alpha=0.6, marker='.',
                               label='Predictions')

        # Add diagonal line for perfect prediction
        max_val = max(valid_predictions[actual_col].max(), valid_predictions['chosen_pod_predicted_latency'].max())
        min_val = min(valid_predictions[actual_col].min(), valid_predictions['chosen_pod_predicted_latency'].min())
        ax_pred_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=linewidth, alpha=alpha, label='Perfect Prediction')

        ## Add regression line (optional)
        # try:
        #     b, m = polyfit(valid_predictions[actual_col], valid_predictions['chosen_pod_predicted_latency'], 1)
        #     x_range = np.linspace(min_val, max_val, 100)
        #     ax_pred_scatter.plot(x_range, m * x_range + b, 'g-', linewidth=linewidth, alpha=alpha, label=f'Linear Fit: y={m:.2f}x+{b:.2f}')
        # except:
        #     pass  # Skip regression if it fails

        # Set same range for x and y axes, starting from 0
        ax_pred_scatter.set_xlim(0, max_val)
        ax_pred_scatter.set_ylim(0, max_val)

        # Set same grid intervals for both axes to create square grid cells
        # Calculate a reasonable number of grid lines (around 10-15 total)
        import math
        n_grid_lines = 10
        grid_interval = math.ceil(max_val / n_grid_lines / 1000) * 1000  # Round to nearest 1000ms

        # Create explicit tick positions that are exactly the same for both axes
        tick_positions = [i * grid_interval for i in range(0, int(max_val / grid_interval) + 2)]
        tick_positions = [pos for pos in tick_positions if pos <= max_val]

        # Clear any existing locators and formatters
        ax_pred_scatter.xaxis.set_major_locator(ticker.NullLocator())
        ax_pred_scatter.yaxis.set_major_locator(ticker.NullLocator())

        # Set exact same tick positions for both axes
        ax_pred_scatter.set_xticks(tick_positions)
        ax_pred_scatter.set_yticks(tick_positions)

        # Force the same tick labels
        ax_pred_scatter.set_xticklabels([f'{int(tick)}' for tick in tick_positions])
        ax_pred_scatter.set_yticklabels([f'{int(tick)}' for tick in tick_positions])

        # Use FixedLocator to prevent matplotlib from changing ticks
        ax_pred_scatter.xaxis.set_major_locator(ticker.FixedLocator(tick_positions))
        ax_pred_scatter.yaxis.set_major_locator(ticker.FixedLocator(tick_positions))

        ax_pred_scatter.set_xlabel(f'Actual {metric_name} (ms)', fontsize=10, fontweight='bold')
        ax_pred_scatter.set_ylabel('Predicted Latency (ms)', fontsize=10, fontweight='bold')
        ax_pred_scatter.set_title(title_scatter, fontsize=16, fontweight='bold', pad=10)
        ax_pred_scatter.grid(True, alpha=alpha, which='major')
        ax_pred_scatter.legend(fontsize=8, loc='upper right')

        # Rotate x-axis tick labels by 45 degrees
        ax_pred_scatter.tick_params(axis='x', rotation=45)

        # Calculate and display prediction accuracy metrics
        mse = ((valid_predictions[actual_col] - valid_predictions['chosen_pod_predicted_latency']) ** 2).mean()
        mae = (valid_predictions[actual_col] - valid_predictions['chosen_pod_predicted_latency']).abs().mean()
        mape = ((valid_predictions[actual_col] - valid_predictions['chosen_pod_predicted_latency']).abs() / valid_predictions[actual_col]).mean() * 100

        # ax_pred_scatter.text(0.02, 0.98, '.2f',
        #                     transform=ax_pred_scatter.transAxes, fontsize=10, verticalalignment='top',
        #                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=alpha))

        # Set equal aspect ratio for better visualization
        ax_pred_scatter.set_aspect('equal', adjustable='box')

    else:
        ax_pred_scatter.text(0.5, 0.5, 'No Valid Prediction Data Available', transform=ax_pred_scatter.transAxes,
                            ha='center', va='center', fontsize=16, alpha=alpha)
        ax_pred_scatter.set_title(title_scatter, fontsize=16, fontweight='bold', pad=10)

    # SUBPLOT 2: Prediction Time Series (ax_pred_timeseries)
    # Plot actual target latency metric for each pod (following same format as other time series)
    for pod in unique_pods:
        pod_df = df[df['selectedpod'] == pod]
        ax_pred_timeseries.scatter(pod_df['relative_time'], pod_df[actual_col], s=marker_size,
                                  color=pod_colors[pod], edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha)

    # Plot predicted latency where available (as overlay)
    valid_pred_timeseries = df[(df['chosen_pod_predicted_latency'].notna()) &
                              (df['chosen_pod_predicted_latency'] > 0)]

    # if not valid_pred_timeseries.empty:
    #     ax_pred_timeseries.scatter(valid_pred_timeseries['relative_time'], valid_pred_timeseries['chosen_pod_predicted_latency'],
    #                               s=10, color='tab:pink', linewidth=linewidth, alpha=0.7,
    #                               marker='x', label='Predicted Latency')

    # Add average predicted latency per second (using valid predictions only)
    valid_pred_avg = df[(df['chosen_pod_predicted_latency'].notna()) & (df['chosen_pod_predicted_latency'] > 0)].copy()
    if not valid_pred_avg.empty:
        valid_pred_avg['time_bin'] = np.floor(valid_pred_avg['relative_time']).astype(int)
        pred_avg_per_sec = valid_pred_avg.groupby('time_bin')['chosen_pod_predicted_latency'].mean().reset_index()
        ax_pred_timeseries.plot(pred_avg_per_sec['time_bin'], pred_avg_per_sec['chosen_pod_predicted_latency'],
                               color='tab:pink', linestyle='-', linewidth=linewidth+0.5, alpha=1,
                               label='Avg Predicted (per sec)', zorder=10)

    # Add sliding window average for all target metric values per second
    df['time_bin'] = np.floor(df['relative_time']).astype(int)
    actual_avg_per_sec = df.groupby('time_bin')[actual_col].mean().reset_index()
    ax_pred_timeseries.plot(actual_avg_per_sec['time_bin'], actual_avg_per_sec[actual_col], 'tab:orange', '-', linewidth=linewidth+0.5, alpha=1, label=f'Avg {metric_name} (per sec)', zorder=10)

    add_transition_lines(ax_pred_timeseries, train_transitions, flush_transitions)
    ax_pred_timeseries.set_xlabel('Relative Time (seconds)', fontsize=14, fontweight='bold')
    ax_pred_timeseries.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax_pred_timeseries.set_title(title_timeseries, fontsize=16, fontweight='bold', pad=10)
    ax_pred_timeseries.grid(True, alpha=alpha)

    # Create legend with only the labeled lines we want (exclude any automatic blue label)
    legend_elements = []
    if not valid_pred_avg.empty:
        legend_elements.append(Line2D([0], [0], color='tab:pink', linewidth=linewidth+0.5, label='Avg Predicted (per sec)'))
    legend_elements.append(Line2D([0], [0], color='tab:orange', linewidth=linewidth+0.5, label=f'Avg {metric_name} (per sec)'))
    ax_pred_timeseries.legend(handles=legend_elements, fontsize=10, loc='upper left')

    return ax_pred_scatter, ax_pred_timeseries


def create_enhanced_plot(data, log_dir, setylim, slo_ttft, slo_tpot, routing_policy):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    if len(df) == 0:
        print("Error, No valid data to plot.")
        exit()
    
    # Calculate reward columns first before cluster statistics
    df['ttft_reward'] = df['ttft'].apply(lambda x: calculate_ttft_reward(x, slo_ttft))
    df['tpot_reward'] = df['avg_tpot'].apply(lambda x: calculate_tpot_reward(x, slo_tpot))
    df['total_reward'] = df['ttft_reward'] + df['tpot_reward']
    
    # Now calculate cluster statistics (which need total_reward column)
    cluster_stats = calculate_cluster_wise_metrics(df)
    train_transitions = get_numtrains_transitions(data)
    flush_transitions = get_numflush_transitions(data)
    
    slo_stats = calculate_slo_satisfaction(df, slo_ttft, slo_tpot)

    unique_pods = list(df['selectedpod'].unique())
    
    df.to_csv("df.csv")
    
    # Prepare all plot data
    plot_data = prepare_plot_data(df, unique_pods)
    pod_data = extract_pod_specific_data(df, unique_pods)

    # Color mapping for different pods
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_pods)))
    pod_colors = dict(zip(unique_pods, colors))

    # Determine number of rows based on whether prediction plots are needed
    if 'latency_predictor' in routing_policy:
        n_rows = 22
        height_ratios = [0.8, 0.8, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2.0, 2.0, 2.0, 2.0, 2.0]
        fig_height = 55
    else:
        n_rows = 20  # Skip the last 2 rows for prediction plots
        height_ratios = [0.8, 0.8, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2.0, 2.0, 2.0]
        fig_height = 50

    # Create a more complex figure with GridSpec - Updated with additional subplots
    fig = plt.figure(figsize=(15, fig_height))
    gs = GridSpec(n_rows, 3, figure=fig, height_ratios=height_ratios, hspace=1.0, top=0.96)
    
    # Plot all subplots
    ax_total_rate, ax_token_rate, ax_pod_rate = plot_request_rate_subplots(
        fig, gs, plot_data, train_transitions, flush_transitions, unique_pods, pod_colors)
    
    main_metrics_axes = plot_main_metrics_subplots(
        fig, gs, df, pod_data, cluster_stats, train_transitions, flush_transitions, unique_pods, pod_colors, plot_data)
    ax1, ax2, ax3, ax_reward = main_metrics_axes[:4]  # Extract the first 4 axes for backward compatibility
    
    analysis_axes = plot_analysis_subplots(
        fig, gs, df, slo_stats, slo_ttft, slo_tpot, unique_pods, pod_colors)

    if 'latency_predictor' in routing_policy:
        prediction_axes = plot_prediction_analysis_subplots(fig, gs, df, train_transitions, flush_transitions, unique_pods, pod_colors, ax1, routing_policy)
    else:
        prediction_axes = None

    # Unpack prediction axes for conditional formatting below (only if prediction plots were created)
    if prediction_axes is not None:
        ax_pred_scatter, ax_pred_timeseries = prediction_axes

    # Set font sizes for tick labels
    all_axes = [ax_total_rate, ax_token_rate, ax_pod_rate] + list(main_metrics_axes) + analysis_axes
    if prediction_axes is not None:
        all_axes += list(prediction_axes)
    for ax in all_axes:
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Force y-axis tick generation for axes with data
        ylim = ax.get_ylim()
        # Do not override custom ticks on the Actual vs Predicted scatter plot
        if prediction_axes is not None and ax is ax_pred_scatter:
            continue
        if ylim[1] > ylim[0] + 1:  # Only if there's a meaningful range
            # Force matplotlib to generate proper y-ticks
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.tick_params(axis='y', which='major', labelsize=11, labelleft=True)
    
    # Improve x-axis formatting
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # Set y-axis limits with some padding
    if setylim:
        ax1.set_ylim(0, 2000)
        ax2.set_ylim(0, 200)
        ax3.set_ylim(0, 10000)
        ax_reward.set_ylim(0, 2.2)
    else:
        ax1.set_ylim(0, df['ttft'].max() * 1.1)
        ax2.set_ylim(0, df['avg_tpot'].max() * 1.1)
        ax3.set_ylim(0, df['e2e'].max() * 1.1)
        ax_reward.set_ylim(df['total_reward'].min() - 1.1, df['total_reward'].max() * 1.5)
    
    # Add reward statistics to the summary print
    print(f"TTFT: {slo_stats['ttft_satisfied']}/{slo_stats['total_requests']} ({slo_stats['ttft_satisfaction_rate']:.1f}%)")
    print(f"TPOT: {slo_stats['tpot_satisfied']}/{slo_stats['total_requests']} ({slo_stats['tpot_satisfaction_rate']:.1f}%)")
    print(f"Both: {slo_stats['both_satisfied']}/{slo_stats['total_requests']} ({slo_stats['both_satisfaction_rate']:.1f}%)")
    
    # Add a super title
    fig.suptitle(f'Latency Metrics Analysis (#request: {len(data)})', fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout - minimize white space
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save files
    plt_pdf_fn = f"{log_dir}/latency_metrics_analysis.pdf"
    plt_png_fn = f"{log_dir}/latency_metrics_analysis.png"
    plt.savefig(plt_pdf_fn, bbox_inches='tight')
    plt.savefig(plt_png_fn, bbox_inches='tight')
    print(f"*****************************")
    print(f"** Saving plot to: {plt_pdf_fn}")
    print(f"** Saving plot to: {plt_png_fn}")
    print(f"*****************************")
    
    return fig

import argparse

parser = argparse.ArgumentParser(description='Plot latency metrics analysis')
parser.add_argument('log_file', type=str, help='Path to the log file')
parser.add_argument('--setylim', type=int, default=0, help='Set y-axis limits')
parser.add_argument('--slo_ttft', type=int, default=1000, help='SLO TTFT')
parser.add_argument('--slo_tpot', type=int, default=50, help='SLO TPOT')


if __name__ == "__main__":
    args = parser.parse_args()

    log_file = args.log_file
    log_dir = log_file.rsplit('/', 1)[0]
    setylim = args.setylim
    slo_ttft = args.slo_ttft
    slo_tpot = args.slo_tpot
    routing_policy = log_file.split('/')[-2].split('-')[0]
    print(f"routing_policy: {routing_policy}")
    data = parse_log_file(log_file)
    
    if not data:
        print(f"Error: No valid latency metrics found in {log_file}. Please check the file format.")
        assert False
    
    print(f"Found {len(data)} log entries with latency metrics")
    
    # Create and save the enhanced plot
    fig = create_enhanced_plot(data, log_dir, setylim, slo_ttft, slo_tpot, routing_policy)
    
    # Print summary statistics
    df = pd.DataFrame(data)
    print("\nSummary Statistics:")
    print(f"TTFT - Min: {df['ttft'].min()} ms, Max: {df['ttft'].max()} ms, Avg: {df['ttft'].mean():.2f} ms")
    print(f"TPOT - Min: {df['avg_tpot'].min()} ms, Max: {df['avg_tpot'].max()} ms, Avg: {df['avg_tpot'].mean():.2f} ms")
    print(f"E2E  - Min: {df['e2e'].min()} ms, Max: {df['e2e'].max()} ms, Avg: {df['e2e'].mean():.2f} ms")