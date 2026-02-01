#!/usr/bin/env python3

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
import os
# from logger import logger

linewidth = 1.5
transition_linewidth = 1.5
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
        'iteration': parsed_data.get('iteration'),  # ADD THIS - iteration from the data
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
                    # Handle Go formatting errors like "%!d(float64=-99)" or "%!d(string={})"
                    # Try to extract number from Go error format
                    if value.startswith('%!') and '(' in value:
                        # Extract number from format like "%!d(float64=-99)"
                        num_match = re.search(r'-?\d+', value)
                        if num_match:
                            try:
                                parsed[key] = int(num_match.group())
                            except ValueError:
                                parsed[key] = value
                        else:
                            # Handle "%!d(string={})" - treat as None/NaN
                            parsed[key] = None
                    else:
                        # Try to convert to float if int fails
                        try:
                            parsed[key] = float(value)
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

def get_iteration_transitions(data):
    """
    Get the first occurrence time of each new iteration value
    """
    transitions = []
    seen_iterations = set()
    
    for item in data:
        iteration = item.get('iteration')
        if iteration is not None and iteration not in seen_iterations:
            transitions.append({
                'iteration': iteration,
                'relative_time': item['relative_time']
            })
            seen_iterations.add(iteration)
    
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

def add_transition_lines(ax, train_transitions, flush_transitions, iteration_transitions=None):
    """Add vertical lines for numTrains and iteration transitions"""
    # Add numTrains transition lines (purple)
    for transition in train_transitions:
        ax.axvline(x=transition['relative_time'], color='red', linewidth=transition_linewidth, zorder=5, linestyle='--')
    
    # Add iteration transition lines (orange) if provided
    if iteration_transitions:
        for transition in iteration_transitions:
            ax.axvline(x=transition['relative_time'], color='blue', linewidth=transition_linewidth, zorder=5, linestyle='-.')

def plot_request_rate_subplots(fig, gs, plot_data, train_transitions, flush_transitions, iteration_transitions, unique_pods, pod_colors):
    """Plot the request rate analysis subplots"""
    # Request rate analysis subplots
    ax_total_rate = fig.add_subplot(gs[0, :])  # Total requests per second
    ax_token_rate = fig.add_subplot(gs[1, :], sharex=ax_total_rate)  # Total input tokens per second
    ax_pod_rate = fig.add_subplot(gs[2, :], sharex=ax_total_rate)  # Requests per pod per second
    
    # SUBPLOT 1: Total Requests Per Second (ax_total_rate)
    ax_total_rate.plot(plot_data['total_requests_per_sec']['time_bin'], plot_data['total_requests_per_sec']['total_requests'], 
                      '-', color='blue', linewidth=linewidth, alpha=alpha, label='Total RPS')
    add_transition_lines(ax_total_rate, train_transitions, flush_transitions, iteration_transitions)
    ax_total_rate.set_title('Total Requests Per Second', fontsize=16, fontweight='bold', pad=10)
    ax_total_rate.set_ylabel('Requests/sec', fontsize=14, fontweight='bold')
    ax_total_rate.grid(True, alpha=alpha)
    ax_total_rate.tick_params(axis='both', which='major', labelsize=11)

    # SUBPLOT 2: Total Input Tokens Per Second (ax_token_rate)
    ax_token_rate.plot(plot_data['input_tokens_per_sec']['time_bin'], plot_data['input_tokens_per_sec']['num_input_tokens'], 
                      '-', color='green', linewidth=linewidth, alpha=alpha, label='Total Tokens/sec')
    add_transition_lines(ax_token_rate, train_transitions, flush_transitions, iteration_transitions)
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
    
    add_transition_lines(ax_pod_rate, train_transitions, flush_transitions, iteration_transitions)
    ax_pod_rate.set_title('Requests Per Second by Pod', fontsize=16, fontweight='bold', pad=10)
    ax_pod_rate.set_ylabel('Requests/sec', fontsize=14, fontweight='bold')
    ax_pod_rate.grid(True, alpha=alpha)
    ax_pod_rate.legend(fontsize=10, loc='upper right')
    ax_pod_rate.tick_params(axis='both', which='major', labelsize=11)

    
    return ax_total_rate, ax_token_rate, ax_pod_rate

def plot_main_metrics_subplots(fig, gs, df, pod_data, cluster_stats, train_transitions, flush_transitions, iteration_transitions, unique_pods, pod_colors, plot_data):
    """Plot the main metrics (TTFT, KV Cache, Running Requests, etc.)"""
    # Define the main time series plots (starting from row 3 after removing empty rows)
    # Group request-level metrics together
    ax1 = fig.add_subplot(gs[3, :])  # TTFT plot
    ax2 = fig.add_subplot(gs[4, :], sharex=ax1)  # TPOT plot
    ax3 = fig.add_subplot(gs[5, :], sharex=ax1)  # E2E Duration plot
    # ax_reward = fig.add_subplot(gs[6, :], sharex=ax1)  # Reward plot - REMOVED
    
    # Pod-level system metrics - grouped with their cluster-wide counterparts
    ax_kv_cache = fig.add_subplot(gs[6, :], sharex=ax1)  # KV Cache Hit Ratio plot (moved up from row 7)
    ax_running_total = fig.add_subplot(gs[7, :], sharex=ax1)  # Total running requests (cluster-wide)
    ax_running = fig.add_subplot(gs[8, :], sharex=ax1)  # Running requests (selected pod)
    ax_prefill_total = fig.add_subplot(gs[9, :], sharex=ax1)  # Total prefill tokens (cluster-wide)
    ax_prefill = fig.add_subplot(gs[10, :], sharex=ax1)  # Prefill tokens (selected pod)
    ax_decode_total = fig.add_subplot(gs[11, :], sharex=ax1)  # Total decode tokens (cluster-wide)
    ax_decode = fig.add_subplot(gs[12, :], sharex=ax1)  # Decode tokens (selected pod)
    ax_gpu_usage = fig.add_subplot(gs[13, :], sharex=ax1)  # GPU cache usage
    ax_waiting = fig.add_subplot(gs[14, :], sharex=ax1)  # Total waiting requests (cluster-wide)
    ax_waiting_selected = fig.add_subplot(gs[15, :], sharex=ax1)  # Waiting requests (selected pod)
    
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
    
    add_transition_lines(ax1, train_transitions, flush_transitions, iteration_transitions)

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
    add_transition_lines(ax_kv_cache, train_transitions, flush_transitions, iteration_transitions)
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
                           train_transitions, flush_transitions, iteration_transitions)
    
    _plot_pod_metric_subplot(ax_prefill, pod_data['prefill_tokens_df'], 'prefill_tokens', 'Prefill Tokens',
                           'numPrefillTokensForAllPods for Selected Pod per Request', unique_pods, pod_colors,
                           train_transitions, flush_transitions, iteration_transitions)
    
    _plot_pod_metric_subplot(ax_decode, pod_data['decode_tokens_df'], 'decode_tokens', 'Decode Tokens',
                           'numDecodeTokensForAllPods for Selected Pod per Request', unique_pods, pod_colors,
                           train_transitions, flush_transitions, iteration_transitions)
    
    _plot_pod_metric_subplot(ax_gpu_usage, pod_data['gpu_cache_usage_df'], 'gpu_cache_usage', 'GPU Cache Usage',
                           'vllmGPUKVCacheUsage for Selected Pod per Request', unique_pods, pod_colors,
                           train_transitions, flush_transitions, iteration_transitions, ylim=(0, 1.1), show_legend=False)
    
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
    
    add_transition_lines(ax_waiting_selected, train_transitions, flush_transitions, iteration_transitions)
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
    
    add_transition_lines(ax2, train_transitions, flush_transitions, iteration_transitions)

    # E2E Duration Plot (ax3)
    for pod in unique_pods:
        pod_df = df[df['selectedpod'] == pod]
        ax3.scatter(pod_df['relative_time'], pod_df['e2e'], s=marker_size, color=pod_colors[pod], edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha)
    
    # Note: Removed cluster-wise statistics as they are redundant with sliding window average  
    # Each request belongs to one pod, so cluster min/max across requests per time bin is not meaningful
    
    # Add sliding window average for all E2E values per second
    e2e_avg_per_sec = df.groupby('time_bin')['e2e'].mean().reset_index()
    ax3.plot(e2e_avg_per_sec['time_bin'], e2e_avg_per_sec['e2e'], 'red', '-', linewidth=linewidth, alpha=alpha, label='Avg E2E (per sec)', zorder=10)
    
    add_transition_lines(ax3, train_transitions, flush_transitions, iteration_transitions)
    
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
        Line2D([0], [0], color='purple', linewidth=linewidth, label='numTrains transition'),
        Line2D([0], [0], color='orange', linewidth=linewidth, linestyle='--', label='Iteration transition')
    ])
    legend_labels_ttft.extend(['Avg TTFT (per sec)', 'numTrains transition', 'Iteration transition'])
    ax1.legend(legend_elements_ttft, legend_labels_ttft, fontsize=10, loc='upper right', ncol=3)
    
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
    
    # Reward Plot - REMOVED
    
    # Plot the grouped cluster-wide total plots
    
    # Total Running Requests (ax_running_total)
    if 'running_requests_df' in plot_data and not plot_data['running_requests_df'].empty:
        ax_running_total.plot(plot_data['running_requests_df']['time_bin'], plot_data['running_requests_df']['total_running'], 
                             '-', color='orange', linewidth=linewidth, alpha=alpha, label='Total Running Requests')
    add_transition_lines(ax_running_total, train_transitions, flush_transitions, iteration_transitions)
    ax_running_total.set_title('Total vllmNumRequestsRunning Across All Pods', fontsize=16, fontweight='bold', pad=10)
    ax_running_total.set_ylabel('Running Requests', fontsize=14, fontweight='bold')
    ax_running_total.grid(True, alpha=alpha)
    ax_running_total.tick_params(axis='both', which='major', labelsize=11)
    
    # Running Requests for Selected Pod (ax_running)
    _plot_pod_metric_subplot(ax_running, pod_data['running_requests_df'], 'running_requests', 'Running Requests', 
                           'vllmNumRequestsRunning for Selected Pod per Request', unique_pods, pod_colors, train_transitions, flush_transitions, iteration_transitions)
    
    # Total Prefill Tokens (ax_prefill_total)
    if 'prefill_tokens_df' in plot_data and not plot_data['prefill_tokens_df'].empty:
        ax_prefill_total.plot(plot_data['prefill_tokens_df']['time_bin'], plot_data['prefill_tokens_df']['total_prefill'], 
                             '-', color='purple', linewidth=linewidth, alpha=alpha, label='Total Prefill Tokens')
    add_transition_lines(ax_prefill_total, train_transitions, flush_transitions, iteration_transitions)
    ax_prefill_total.set_title('Total numPrefillTokensForAllPods Across All Pods', fontsize=16, fontweight='bold', pad=10)
    ax_prefill_total.set_ylabel('Prefill Tokens', fontsize=14, fontweight='bold')
    ax_prefill_total.grid(True, alpha=alpha)
    ax_prefill_total.tick_params(axis='both', which='major', labelsize=11)
    
    # Prefill Tokens for Selected Pod (ax_prefill)
    _plot_pod_metric_subplot(ax_prefill, pod_data['prefill_tokens_df'], 'prefill_tokens', 'Prefill Tokens', 
                           'numPrefillTokensForAllPods for Selected Pod per Request', unique_pods, pod_colors, train_transitions, flush_transitions, iteration_transitions)

    # Total Decode Tokens (ax_decode_total)
    if 'decode_tokens_df' in plot_data and not plot_data['decode_tokens_df'].empty:
        ax_decode_total.plot(plot_data['decode_tokens_df']['time_bin'], plot_data['decode_tokens_df']['total_decode'], 
                            '-', color='brown', linewidth=linewidth, alpha=alpha, label='Total Decode Tokens')
    add_transition_lines(ax_decode_total, train_transitions, flush_transitions, iteration_transitions)
    ax_decode_total.set_title('Total numDecodeTokensForAllPods Across All Pods', fontsize=16, fontweight='bold', pad=10)
    ax_decode_total.set_ylabel('Decode Tokens', fontsize=14, fontweight='bold')
    ax_decode_total.grid(True, alpha=alpha)
    ax_decode_total.tick_params(axis='both', which='major', labelsize=11)
    
    # Decode Tokens for Selected Pod (ax_decode)
    _plot_pod_metric_subplot(ax_decode, pod_data['decode_tokens_df'], 'decode_tokens', 'Decode Tokens', 
                           'numDecodeTokensForAllPods for Selected Pod per Request', unique_pods, pod_colors, train_transitions, flush_transitions, iteration_transitions)

    # Total Waiting Requests (ax_waiting)
    if 'waiting_requests_df' in plot_data and not plot_data['waiting_requests_df'].empty:
        ax_waiting.plot(plot_data['waiting_requests_df']['time_bin'], plot_data['waiting_requests_df']['total_waiting'], 
                       '-', color='red', linewidth=linewidth, alpha=alpha, label='Total Waiting Requests')
    add_transition_lines(ax_waiting, train_transitions, flush_transitions, iteration_transitions)
    ax_waiting.set_title('Total vllmNumRequestsWaiting Across All Pods', fontsize=16, fontweight='bold', pad=10)
    ax_waiting.set_ylabel('Waiting Requests', fontsize=14, fontweight='bold')
    ax_waiting.grid(True, alpha=alpha)
    ax_waiting.tick_params(axis='both', which='major', labelsize=11)
    
    # Waiting Requests for Selected Pod (ax_waiting_selected)
    _plot_pod_metric_subplot(ax_waiting_selected, pod_data['waiting_selected_pod_df'], 'waiting_requests_selected', 'Waiting Requests', 
                           'vllmNumRequestsWaiting for Selected Pod per Request', unique_pods, pod_colors, train_transitions, flush_transitions, iteration_transitions, show_legend=False)
    
    return ax1, ax2, ax3, ax_kv_cache, ax_running_total, ax_running, ax_prefill_total, ax_prefill, ax_decode_total, ax_decode, ax_gpu_usage, ax_waiting, ax_waiting_selected

def _plot_pod_metric_subplot(ax, data_df, metric_col, metric_name, title, unique_pods, pod_colors, train_transitions, flush_transitions, iteration_transitions=None, ylim=None, show_legend=True):
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
    
    add_transition_lines(ax, train_transitions, flush_transitions, iteration_transitions)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add legend - only cluster metrics, no pod labels (if show_legend is True)
    if show_legend:
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
    # Define the pod analysis plots (updated indices - shifted up by 1 after removing reward plot)
    ax4 = fig.add_subplot(gs[16, 0])  # Average TTFT by Pod
    ax5 = fig.add_subplot(gs[16, 1])  # Average TPOT by Pod
    ax_slo = fig.add_subplot(gs[16, 2])  # SLO satisfaction
    
    # Define the CDF distribution plots  
    ax6 = fig.add_subplot(gs[17, 0])  # TTFT CDF
    ax7 = fig.add_subplot(gs[17, 1])  # TPOT CDF
    ax8 = fig.add_subplot(gs[17, 2])  # E2E CDF
    
    # Average TTFT by Pod (ax4)
    pod_avg_ttft = df.groupby('selectedpod')['ttft'].mean().sort_values(ascending=False)
    pod_counts = df.groupby('selectedpod').size()
    
    # Create bars for TTFT plot
    ttft_bars = ax4.bar(range(len(pod_avg_ttft)), pod_avg_ttft.values, 
                      color=[pod_colors[pod] for pod in pod_avg_ttft.index], 
                      edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha, width=0.7)
    
    # Add annotations
    for i, (pod, ttft) in enumerate(pod_avg_ttft.items()):
        ax4.text(i, ttft + 7, f'{ttft:.0f} ms', ha='center', fontsize=6, fontweight='bold', rotation=45)
        ax4.text(i, 14, f'n={pod_counts[pod]}', ha='center', fontsize=6, rotation=90, fontweight='bold')
    
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
    
    return [ax4, ax5, ax_slo, ax6, ax7, ax8]

def plot_numtrains_analysis_subplots(fig, gs, df, start_row):
    """Plot numTrains-based latency trends and CDFs for latency_predictor policies"""
    linewidth = 1.5
    alpha = 0.7
    
    # Get unique numTrains
    unique_num_trains = sorted(df['num_trains'].unique())
    
    # Define colors for numTrains (used across all numTrains plots)
    num_trains_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_num_trains)))
    num_trains_color_map = dict(zip(unique_num_trains, num_trains_colors))
    
    # Row start_row: TTFT Trends (Avg + P99 with dual y-axis) | TTFT CDF
    gs_row_ttft = gs[start_row, :].subgridspec(1, 2, wspace=0.3)
    ax_nt_ttft_trends = fig.add_subplot(gs_row_ttft[0])
    ax_nt_ttft_cdf = fig.add_subplot(gs_row_ttft[1])
    
    # Row start_row+1: TPOT Trends (Avg + P99 with dual y-axis) | TPOT CDF
    gs_row_tpot = gs[start_row+1, :].subgridspec(1, 2, wspace=0.3)
    ax_nt_tpot_trends = fig.add_subplot(gs_row_tpot[0])
    ax_nt_tpot_cdf = fig.add_subplot(gs_row_tpot[1])
    
    # Row start_row+2: E2E Trends (Avg + P99 with dual y-axis) | E2E CDF
    gs_row_e2e = gs[start_row+2, :].subgridspec(1, 2, wspace=0.3)
    ax_nt_e2e_trends = fig.add_subplot(gs_row_e2e[0])
    ax_nt_e2e_cdf = fig.add_subplot(gs_row_e2e[1])
    
    # Calculate numTrains statistics
    nt_stats = []
    for num_trains in unique_num_trains:
        subset = df[df['num_trains'] == num_trains]
        if len(subset) > 0:
            nt_stats.append({
                'num_trains': num_trains,
                'avg_ttft': subset['ttft'].mean(),
                'p99_ttft': np.percentile(subset['ttft'], 99),
                'avg_tpot': subset['avg_tpot'].mean(),
                'p99_tpot': np.percentile(subset['avg_tpot'], 99),
                'avg_e2e': subset['e2e'].mean(),
                'p99_e2e': np.percentile(subset['e2e'], 99),
            })
    
    if nt_stats:
        nt_vals = [s['num_trains'] for s in nt_stats]
        avg_ttft_vals = [s['avg_ttft'] for s in nt_stats]
        p99_ttft_vals = [s['p99_ttft'] for s in nt_stats]
        avg_tpot_vals = [s['avg_tpot'] for s in nt_stats]
        p99_tpot_vals = [s['p99_tpot'] for s in nt_stats]
        avg_e2e_vals = [s['avg_e2e'] for s in nt_stats]
        p99_e2e_vals = [s['p99_e2e'] for s in nt_stats]
        
        # Compute overall statistics for legend
        overall_avg_ttft = df['ttft'].mean()
        overall_p99_ttft = np.percentile(df['ttft'], 99)
        overall_avg_tpot = df['avg_tpot'].mean()
        overall_p99_tpot = np.percentile(df['avg_tpot'], 99)
        overall_avg_e2e = df['e2e'].mean()
        overall_p99_e2e = np.percentile(df['e2e'], 99)
        
        # TTFT Trends (Avg + P99 with dual y-axis)
        ax_nt_ttft_trends.plot(nt_vals, avg_ttft_vals, marker='o', linestyle='-', color='blue', linewidth=linewidth, alpha=alpha, label=f'Avg: {overall_avg_ttft:.1f}ms')
        # Add value labels on each dot for average
        for nt, val in zip(nt_vals, avg_ttft_vals):
            ax_nt_ttft_trends.text(nt, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='blue', fontweight='bold')
        ax_nt_ttft_trends.set_xlabel('numTrains', fontsize=10)
        ax_nt_ttft_trends.set_ylabel('Average TTFT (ms)', fontsize=10, fontweight='bold', color='blue')
        ax_nt_ttft_trends.tick_params(axis='y', labelcolor='blue')
        
        ax_nt_ttft_trends_right = ax_nt_ttft_trends.twinx()
        ax_nt_ttft_trends_right.plot(nt_vals, p99_ttft_vals, marker='x', linestyle='--', color='darkblue', linewidth=linewidth, alpha=alpha, label=f'P99: {overall_p99_ttft:.1f}ms')
        # Add value labels on each dot for P99 (positioned slightly to the right)
        x_range = max(nt_vals) - min(nt_vals) if len(nt_vals) > 1 else 1
        x_offset = x_range * 0.08  # Offset to the right
        for nt, val in zip(nt_vals, p99_ttft_vals):
            ax_nt_ttft_trends_right.text(nt + x_offset, val, f'{val:.0f}', ha='left', va='center', fontsize=8, color='darkblue', fontweight='bold')
        ax_nt_ttft_trends_right.set_ylabel('P99 TTFT (ms)', fontsize=10, fontweight='bold', color='darkblue')
        ax_nt_ttft_trends_right.tick_params(axis='y', labelcolor='darkblue')
        
        ax_nt_ttft_trends.set_ylim(0, max(avg_ttft_vals) * 1.4)
        ax_nt_ttft_trends_right.set_ylim(0, max(p99_ttft_vals) * 1.4)
        ax_nt_ttft_trends.set_title('TTFT Trends by numTrains', fontsize=12, fontweight='bold', pad=10)
        ax_nt_ttft_trends.grid(True, alpha=alpha)
        ax_nt_ttft_trends.set_xticks(nt_vals)
        
        lines1, labels1 = ax_nt_ttft_trends.get_legend_handles_labels()
        lines2, labels2 = ax_nt_ttft_trends_right.get_legend_handles_labels()
        ax_nt_ttft_trends.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left', ncol=2)
        
        # TPOT Trends (Avg + P99 with dual y-axis)
        ax_nt_tpot_trends.plot(nt_vals, avg_tpot_vals, marker='o', linestyle='-', color='green', linewidth=linewidth, alpha=alpha, label=f'Avg: {overall_avg_tpot:.1f}ms')
        # Add value labels on each dot for average
        for nt, val in zip(nt_vals, avg_tpot_vals):
            ax_nt_tpot_trends.text(nt, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
        ax_nt_tpot_trends.set_xlabel('numTrains', fontsize=10)
        ax_nt_tpot_trends.set_ylabel('Average TPOT (ms)', fontsize=10, fontweight='bold', color='green')
        ax_nt_tpot_trends.tick_params(axis='y', labelcolor='green')
        
        ax_nt_tpot_trends_right = ax_nt_tpot_trends.twinx()
        ax_nt_tpot_trends_right.plot(nt_vals, p99_tpot_vals, marker='x', linestyle='--', color='darkgreen', linewidth=linewidth, alpha=alpha, label=f'P99: {overall_p99_tpot:.1f}ms')
        # Add value labels on each dot for P99 (positioned slightly to the right)
        x_range = max(nt_vals) - min(nt_vals) if len(nt_vals) > 1 else 1
        x_offset = x_range * 0.08  # Offset to the right
        for nt, val in zip(nt_vals, p99_tpot_vals):
            ax_nt_tpot_trends_right.text(nt + x_offset, val, f'{val:.0f}', ha='left', va='center', fontsize=8, color='darkgreen', fontweight='bold')
        ax_nt_tpot_trends_right.set_ylabel('P99 TPOT (ms)', fontsize=10, fontweight='bold', color='darkgreen')
        ax_nt_tpot_trends_right.tick_params(axis='y', labelcolor='darkgreen')
        
        ax_nt_tpot_trends.set_ylim(0, max(avg_tpot_vals) * 1.4)
        ax_nt_tpot_trends_right.set_ylim(0, max(p99_tpot_vals) * 1.4)
        ax_nt_tpot_trends.set_title('TPOT Trends by numTrains', fontsize=12, fontweight='bold', pad=10)
        ax_nt_tpot_trends.grid(True, alpha=alpha)
        ax_nt_tpot_trends.set_xticks(nt_vals)
        
        lines1, labels1 = ax_nt_tpot_trends.get_legend_handles_labels()
        lines2, labels2 = ax_nt_tpot_trends_right.get_legend_handles_labels()
        ax_nt_tpot_trends.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left', ncol=2)
        
        # E2E Trends (Avg + P99 with dual y-axis)
        ax_nt_e2e_trends.plot(nt_vals, avg_e2e_vals, marker='o', linestyle='-', color='purple', linewidth=linewidth, alpha=alpha, label=f'Avg: {overall_avg_e2e:.1f}ms')
        # Add value labels on each dot for average
        for nt, val in zip(nt_vals, avg_e2e_vals):
            ax_nt_e2e_trends.text(nt, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='purple', fontweight='bold')
        ax_nt_e2e_trends.set_xlabel('numTrains', fontsize=10)
        ax_nt_e2e_trends.set_ylabel('Average E2E (ms)', fontsize=10, fontweight='bold', color='purple')
        ax_nt_e2e_trends.tick_params(axis='y', labelcolor='purple')
        
        ax_nt_e2e_trends_right = ax_nt_e2e_trends.twinx()
        ax_nt_e2e_trends_right.plot(nt_vals, p99_e2e_vals, marker='x', linestyle='--', color='indigo', linewidth=linewidth, alpha=alpha, label=f'P99: {overall_p99_e2e:.1f}ms')
        # Add value labels on each dot for P99 (positioned slightly to the right)
        x_range = max(nt_vals) - min(nt_vals) if len(nt_vals) > 1 else 1
        x_offset = x_range * 0.08  # Offset to the right
        for nt, val in zip(nt_vals, p99_e2e_vals):
            ax_nt_e2e_trends_right.text(nt + x_offset, val, f'{val:.0f}', ha='left', va='center', fontsize=8, color='indigo', fontweight='bold')
        ax_nt_e2e_trends_right.set_ylabel('P99 E2E (ms)', fontsize=10, fontweight='bold', color='indigo')
        ax_nt_e2e_trends_right.tick_params(axis='y', labelcolor='indigo')
        
        ax_nt_e2e_trends.set_ylim(0, max(avg_e2e_vals) * 1.4)
        ax_nt_e2e_trends_right.set_ylim(0, max(p99_e2e_vals) * 1.4)
        ax_nt_e2e_trends.set_title('E2E Latency Trends by numTrains', fontsize=12, fontweight='bold', pad=10)
        ax_nt_e2e_trends.grid(True, alpha=alpha)
        ax_nt_e2e_trends.set_xticks(nt_vals)
        
        lines1, labels1 = ax_nt_e2e_trends.get_legend_handles_labels()
        lines2, labels2 = ax_nt_e2e_trends_right.get_legend_handles_labels()
        ax_nt_e2e_trends.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left', ncol=2)
    
    # TTFT CDF by numTrains
    for num_trains in unique_num_trains:
        subset = df[df['num_trains'] == num_trains]
        if len(subset) > 0:
            sorted_ttft = np.sort(subset['ttft'])
            cdf = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
            avg_ttft = subset['ttft'].mean()
            p99_ttft = np.percentile(subset['ttft'], 99)
            ax_nt_ttft_cdf.plot(sorted_ttft, cdf, label=f'numTrains={num_trains}, avg: {avg_ttft:.0f}ms, p99: {p99_ttft:.0f}ms',
                                  color=num_trains_color_map[num_trains], linewidth=1.5, alpha=0.7)
    
    ax_nt_ttft_cdf.set_xlabel('TTFT (ms)', fontsize=10, fontweight='bold')
    ax_nt_ttft_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_nt_ttft_cdf.set_title('TTFT CDF by numTrains', fontsize=12, fontweight='bold', pad=10)
    ax_nt_ttft_cdf.grid(True, alpha=0.3)
    ax_nt_ttft_cdf.legend(fontsize=6, loc='lower right')
    
    # TPOT CDF by numTrains
    for num_trains in unique_num_trains:
        subset = df[df['num_trains'] == num_trains]
        if len(subset) > 0:
            sorted_tpot = np.sort(subset['avg_tpot'])
            cdf = np.arange(1, len(sorted_tpot) + 1) / len(sorted_tpot)
            avg_tpot = subset['avg_tpot'].mean()
            p99_tpot = np.percentile(subset['avg_tpot'], 99)
            ax_nt_tpot_cdf.plot(sorted_tpot, cdf, label=f'numTrains={num_trains}, avg: {avg_tpot:.0f}ms, p99: {p99_tpot:.0f}ms',
                                  color=num_trains_color_map[num_trains], linewidth=1.5, alpha=0.7)
    
    ax_nt_tpot_cdf.set_xlabel('TPOT (ms)', fontsize=10, fontweight='bold')
    ax_nt_tpot_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_nt_tpot_cdf.set_title('TPOT CDF by numTrains', fontsize=12, fontweight='bold', pad=10)
    ax_nt_tpot_cdf.grid(True, alpha=0.3)
    ax_nt_tpot_cdf.legend(fontsize=6, loc='lower right')
    
    # E2E CDF by numTrains
    for num_trains in unique_num_trains:
        subset = df[df['num_trains'] == num_trains]
        if len(subset) > 0:
            sorted_e2e = np.sort(subset['e2e'])
            cdf = np.arange(1, len(sorted_e2e) + 1) / len(sorted_e2e)
            avg_e2e = subset['e2e'].mean()
            p99_e2e = np.percentile(subset['e2e'], 99)
            ax_nt_e2e_cdf.plot(sorted_e2e, cdf, label=f'numTrains={num_trains}, avg: {avg_e2e:.0f}ms, p99: {p99_e2e:.0f}ms',
                                 color=num_trains_color_map[num_trains], linewidth=1.5, alpha=0.7)
    
    ax_nt_e2e_cdf.set_xlabel('E2E Latency (ms)', fontsize=10, fontweight='bold')
    ax_nt_e2e_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_nt_e2e_cdf.set_title('E2E CDF by numTrains', fontsize=12, fontweight='bold', pad=10)
    ax_nt_e2e_cdf.grid(True, alpha=0.3)
    ax_nt_e2e_cdf.legend(fontsize=6, loc='lower right')
    
    return ax_nt_ttft_trends, ax_nt_tpot_trends, ax_nt_e2e_trends, ax_nt_ttft_cdf, ax_nt_tpot_cdf, ax_nt_e2e_cdf


def plot_iteration_analysis_subplots(fig, gs, df, start_row):
    """Plot iteration-based latency trends and CDFs for all routing policies"""
    linewidth = 1.5
    alpha = 0.7
    
    # Get unique iterations
    unique_iterations = sorted(df['iteration'].unique())
    
    # Define colors for iterations (used across all iteration plots)
    iteration_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_iterations)))
    iteration_color_map = dict(zip(unique_iterations, iteration_colors))
    
    # Row start_row: TTFT Trends (Avg + P99 with dual y-axis) | TTFT CDF
    gs_row_ttft = gs[start_row, :].subgridspec(1, 2, wspace=0.3)
    ax_iter_ttft_trends = fig.add_subplot(gs_row_ttft[0])
    ax_iter_ttft_cdf = fig.add_subplot(gs_row_ttft[1])
    
    # Row start_row+1: TPOT Trends (Avg + P99 with dual y-axis) | TPOT CDF
    gs_row_tpot = gs[start_row+1, :].subgridspec(1, 2, wspace=0.3)
    ax_iter_tpot_trends = fig.add_subplot(gs_row_tpot[0])
    ax_iter_tpot_cdf = fig.add_subplot(gs_row_tpot[1])
    
    # Row start_row+2: E2E Trends (Avg + P99 with dual y-axis) | E2E CDF
    gs_row_e2e = gs[start_row+2, :].subgridspec(1, 2, wspace=0.3)
    ax_iter_e2e_trends = fig.add_subplot(gs_row_e2e[0])
    ax_iter_e2e_cdf = fig.add_subplot(gs_row_e2e[1])
    
    # Calculate iteration statistics
    iter_stats = []
    for iteration in unique_iterations:
        subset = df[df['iteration'] == iteration]
        if len(subset) > 0:
            iter_stats.append({
                'iteration': iteration,
                'avg_ttft': subset['ttft'].mean(),
                'p99_ttft': np.percentile(subset['ttft'], 99),
                'avg_tpot': subset['avg_tpot'].mean(),
                'p99_tpot': np.percentile(subset['avg_tpot'], 99),
                'avg_e2e': subset['e2e'].mean(),
                'p99_e2e': np.percentile(subset['e2e'], 99),
            })
    
    if iter_stats:
        iter_vals = [s['iteration'] for s in iter_stats]
        avg_ttft_vals = [s['avg_ttft'] for s in iter_stats]
        p99_ttft_vals = [s['p99_ttft'] for s in iter_stats]
        avg_tpot_vals = [s['avg_tpot'] for s in iter_stats]
        p99_tpot_vals = [s['p99_tpot'] for s in iter_stats]
        avg_e2e_vals = [s['avg_e2e'] for s in iter_stats]
        p99_e2e_vals = [s['p99_e2e'] for s in iter_stats]
        
        # Compute overall statistics for legend
        overall_avg_ttft = df['ttft'].mean()
        overall_p99_ttft = np.percentile(df['ttft'], 99)
        overall_avg_tpot = df['avg_tpot'].mean()
        overall_p99_tpot = np.percentile(df['avg_tpot'], 99)
        overall_avg_e2e = df['e2e'].mean()
        overall_p99_e2e = np.percentile(df['e2e'], 99)
        
        # TTFT Trends (Avg + P99 with dual y-axis)
        ax_iter_ttft_trends.plot(iter_vals, avg_ttft_vals, marker='o', linestyle='-', color='blue', linewidth=linewidth, alpha=alpha, label=f'Avg: {overall_avg_ttft:.1f}ms')
        # Add value labels on each dot for average
        for it, val in zip(iter_vals, avg_ttft_vals):
            ax_iter_ttft_trends.text(it, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='blue', fontweight='bold')
        ax_iter_ttft_trends.set_xlabel('Iteration', fontsize=10)
        ax_iter_ttft_trends.set_ylabel('Average TTFT (ms)', fontsize=10, fontweight='bold', color='blue')
        ax_iter_ttft_trends.tick_params(axis='y', labelcolor='blue')
        
        ax_iter_ttft_trends_right = ax_iter_ttft_trends.twinx()
        ax_iter_ttft_trends_right.plot(iter_vals, p99_ttft_vals, marker='x', linestyle='--', color='darkblue', linewidth=linewidth, alpha=alpha, label=f'P99: {overall_p99_ttft:.1f}ms')
        # Add value labels on each dot for P99 (positioned slightly to the right)
        x_range = max(iter_vals) - min(iter_vals) if len(iter_vals) > 1 else 1
        x_offset = x_range * 0.08  # Offset to the right
        for it, val in zip(iter_vals, p99_ttft_vals):
            ax_iter_ttft_trends_right.text(it, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='darkblue', fontweight='bold')
        ax_iter_ttft_trends_right.set_ylabel('P99 TTFT (ms)', fontsize=10, fontweight='bold', color='darkblue')
        ax_iter_ttft_trends_right.tick_params(axis='y', labelcolor='darkblue')
        
        ax_iter_ttft_trends.set_ylim(0, max(avg_ttft_vals) * 1.4)
        ax_iter_ttft_trends_right.set_ylim(0, max(p99_ttft_vals) * 1.4)
        ax_iter_ttft_trends.set_title('TTFT Trends by Iterations', fontsize=12, fontweight='bold', pad=10)
        ax_iter_ttft_trends.grid(True, alpha=alpha)
        
        lines1, labels1 = ax_iter_ttft_trends.get_legend_handles_labels()
        lines2, labels2 = ax_iter_ttft_trends_right.get_legend_handles_labels()
        ax_iter_ttft_trends.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left', ncol=2)
        
        # TPOT Trends (Avg + P99 with dual y-axis)
        ax_iter_tpot_trends.plot(iter_vals, avg_tpot_vals, marker='o', linestyle='-', color='green', linewidth=linewidth, alpha=alpha, label=f'Avg: {overall_avg_tpot:.1f}ms')
        # Add value labels on each dot for average
        for it, val in zip(iter_vals, avg_tpot_vals):
            ax_iter_tpot_trends.text(it, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
        ax_iter_tpot_trends.set_xlabel('Iteration', fontsize=10)
        ax_iter_tpot_trends.set_ylabel('Average TPOT (ms)', fontsize=10, fontweight='bold', color='green')
        ax_iter_tpot_trends.tick_params(axis='y', labelcolor='green')
        
        ax_iter_tpot_trends_right = ax_iter_tpot_trends.twinx()
        ax_iter_tpot_trends_right.plot(iter_vals, p99_tpot_vals, marker='x', linestyle='--', color='darkgreen', linewidth=linewidth, alpha=alpha, label=f'P99: {overall_p99_tpot:.1f}ms')
        # Add value labels on each dot for P99 (positioned slightly to the right)
        x_range = max(iter_vals) - min(iter_vals) if len(iter_vals) > 1 else 1
        x_offset = x_range * 0.08  # Offset to the right
        for it, val in zip(iter_vals, p99_tpot_vals):
            ax_iter_tpot_trends_right.text(it, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='darkgreen', fontweight='bold')
        ax_iter_tpot_trends_right.set_ylabel('P99 TPOT (ms)', fontsize=10, fontweight='bold', color='darkgreen')
        ax_iter_tpot_trends_right.tick_params(axis='y', labelcolor='darkgreen')
        
        ax_iter_tpot_trends.set_ylim(0, max(avg_tpot_vals) * 1.4)
        ax_iter_tpot_trends_right.set_ylim(0, max(p99_tpot_vals) * 1.4)
        ax_iter_tpot_trends.set_title('TPOT Trends by Iterations', fontsize=12, fontweight='bold', pad=10)
        ax_iter_tpot_trends.grid(True, alpha=alpha)
        
        lines1, labels1 = ax_iter_tpot_trends.get_legend_handles_labels()
        lines2, labels2 = ax_iter_tpot_trends_right.get_legend_handles_labels()
        ax_iter_tpot_trends.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left', ncol=2)
        
        # E2E Trends (Avg + P99 with dual y-axis)
        ax_iter_e2e_trends.plot(iter_vals, avg_e2e_vals, marker='o', linestyle='-', color='purple', linewidth=linewidth, alpha=alpha, label=f'Avg: {overall_avg_e2e:.1f}ms')
        # Add value labels on each dot for average
        for it, val in zip(iter_vals, avg_e2e_vals):
            ax_iter_e2e_trends.text(it, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='purple', fontweight='bold')
        ax_iter_e2e_trends.set_xlabel('Iteration', fontsize=10)
        ax_iter_e2e_trends.set_ylabel('Average E2E (ms)', fontsize=10, fontweight='bold', color='purple')
        ax_iter_e2e_trends.tick_params(axis='y', labelcolor='purple')
        
        ax_iter_e2e_trends_right = ax_iter_e2e_trends.twinx()
        ax_iter_e2e_trends_right.plot(iter_vals, p99_e2e_vals, marker='x', linestyle='--', color='indigo', linewidth=linewidth, alpha=alpha, label=f'P99: {overall_p99_e2e:.1f}ms')
        # Add value labels on each dot for P99 (positioned slightly to the right)
        x_range = max(iter_vals) - min(iter_vals) if len(iter_vals) > 1 else 1
        x_offset = x_range * 0.08  # Offset to the right
        for it, val in zip(iter_vals, p99_e2e_vals):
            ax_iter_e2e_trends_right.text(it, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='indigo', fontweight='bold')
        ax_iter_e2e_trends_right.set_ylabel('P99 E2E (ms)', fontsize=10, fontweight='bold', color='indigo')
        ax_iter_e2e_trends_right.tick_params(axis='y', labelcolor='indigo')
        
        ax_iter_e2e_trends.set_ylim(0, max(avg_e2e_vals) * 1.4)
        ax_iter_e2e_trends_right.set_ylim(0, max(p99_e2e_vals) * 1.4)
        ax_iter_e2e_trends.set_title('E2E Latency Trends by Iterations', fontsize=12, fontweight='bold', pad=10)
        ax_iter_e2e_trends.grid(True, alpha=alpha)
        
        lines1, labels1 = ax_iter_e2e_trends.get_legend_handles_labels()
        lines2, labels2 = ax_iter_e2e_trends_right.get_legend_handles_labels()
        ax_iter_e2e_trends.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left', ncol=2)
    
    # TTFT CDF by Iterations
    for iteration in unique_iterations:
        subset = df[df['iteration'] == iteration]
        if len(subset) > 0:
            sorted_ttft = np.sort(subset['ttft'])
            cdf = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
            avg_ttft = subset['ttft'].mean()
            p99_ttft = np.percentile(subset['ttft'], 99)
            ax_iter_ttft_cdf.plot(sorted_ttft, cdf, label=f'Iter {iteration}, avg: {avg_ttft:.0f}ms, p99: {p99_ttft:.0f}ms',
                                  color=iteration_color_map[iteration], linewidth=1.5, alpha=0.7)
    
    ax_iter_ttft_cdf.set_xlabel('TTFT (ms)', fontsize=10, fontweight='bold')
    ax_iter_ttft_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_iter_ttft_cdf.set_title('TTFT CDF by Iterations', fontsize=12, fontweight='bold', pad=10)
    ax_iter_ttft_cdf.grid(True, alpha=0.3)
    ax_iter_ttft_cdf.legend(fontsize=7, loc='lower right', ncol=2)
    
    # TPOT CDF by Iterations
    for iteration in unique_iterations:
        subset = df[df['iteration'] == iteration]
        if len(subset) > 0:
            sorted_tpot = np.sort(subset['avg_tpot'])
            cdf = np.arange(1, len(sorted_tpot) + 1) / len(sorted_tpot)
            avg_tpot = subset['avg_tpot'].mean()
            p99_tpot = np.percentile(subset['avg_tpot'], 99)
            ax_iter_tpot_cdf.plot(sorted_tpot, cdf, label=f'Iter {iteration}, avg: {avg_tpot:.0f}ms, p99: {p99_tpot:.0f}ms',
                                  color=iteration_color_map[iteration], linewidth=1.5, alpha=0.7)
    
    ax_iter_tpot_cdf.set_xlabel('TPOT (ms)', fontsize=10, fontweight='bold')
    ax_iter_tpot_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_iter_tpot_cdf.set_title('TPOT CDF by Iterations', fontsize=12, fontweight='bold', pad=10)
    ax_iter_tpot_cdf.grid(True, alpha=0.3)
    ax_iter_tpot_cdf.legend(fontsize=7, loc='lower right', ncol=2)
    
    # E2E CDF by Iterations
    for iteration in unique_iterations:
        subset = df[df['iteration'] == iteration]
        if len(subset) > 0:
            sorted_e2e = np.sort(subset['e2e'])
            cdf = np.arange(1, len(sorted_e2e) + 1) / len(sorted_e2e)
            avg_e2e = subset['e2e'].mean()
            p99_e2e = np.percentile(subset['e2e'], 99)
            ax_iter_e2e_cdf.plot(sorted_e2e, cdf, label=f'Iter {iteration}, avg: {avg_e2e:.0f}ms, p99: {p99_e2e:.0f}ms',
                                 color=iteration_color_map[iteration], linewidth=1.5, alpha=0.7)
    
    ax_iter_e2e_cdf.set_xlabel('E2E Latency (ms)', fontsize=10, fontweight='bold')
    ax_iter_e2e_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_iter_e2e_cdf.set_title('E2E CDF by Iterations', fontsize=12, fontweight='bold', pad=10)
    ax_iter_e2e_cdf.grid(True, alpha=0.3)
    ax_iter_e2e_cdf.legend(fontsize=7, loc='lower right', ncol=2)
    
    return ax_iter_ttft_trends, ax_iter_tpot_trends, ax_iter_e2e_trends, ax_iter_ttft_cdf, ax_iter_tpot_cdf, ax_iter_e2e_cdf


def plot_prediction_analysis_subplots(fig, gs, df, train_transitions, flush_transitions, iteration_transitions, unique_pods, pod_colors, ax1, routing_policy):
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

    # Define the prediction analysis plots (rows 22-24, shifted up by 1 after removing reward plot)
    # Row 22: Prediction Accuracy by numTrains (full width)
    ax_pred_bar = fig.add_subplot(gs[22, :])
    
    # Row 23: Centered scatter plot (bigger, square)
    ax_pred_scatter = fig.add_subplot(gs[23, :])  # Actual vs Predicted Scatter Plot (full width, will use aspect ratio)
    
    # Row 24: Time series
    ax_pred_timeseries = fig.add_subplot(gs[24, :], sharex=ax1)  # Prediction Time Series (share x-axis with other time series)

    # SUBPLOT: Prediction Accuracy Bar Chart by numTrains (ax_pred_bar)
    # Filter out entries where predicted latency is None or 0 (no prediction made)
    valid_predictions = df[(df['chosen_pod_predicted_latency'].notna()) &
                          (df['chosen_pod_predicted_latency'] > 0) &
                          (df[actual_col].notna()) &
                          (df[actual_col] > 0)]

    if not valid_predictions.empty:
        # Calculate accuracy metrics for each numTrains group
        unique_num_trains = sorted(valid_predictions['num_trains'].unique())
        num_trains_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_num_trains)))
        
        mae_values = []
        mape_values = []
        rmse_values = []
        count_values = []
        
        for num_trains in unique_num_trains:
            subset = valid_predictions[valid_predictions['num_trains'] == num_trains]
            if len(subset) > 0:
                # Calculate metrics
                mae = (subset[actual_col] - subset['chosen_pod_predicted_latency']).abs().mean()
                mape = ((subset[actual_col] - subset['chosen_pod_predicted_latency']).abs() / subset[actual_col]).mean() * 100
                rmse = np.sqrt(((subset[actual_col] - subset['chosen_pod_predicted_latency']) ** 2).mean())
                
                mae_values.append(mae)
                mape_values.append(mape)
                rmse_values.append(rmse)
                count_values.append(len(subset))
        
        # Create bar chart with grouped bars for MAE and MAPE
        x = np.arange(len(unique_num_trains))
        width = 0.35
        
        # Create twin axis for MAPE (percentage)
        ax_pred_bar_right = ax_pred_bar.twinx()
        
        # Plot MAE bars (left axis)
        bars1 = ax_pred_bar.bar(x - width/2, mae_values, width, 
                                    color=[num_trains_colors[i] for i in range(len(unique_num_trains))],
                                    alpha=0.8, edgecolor=edgecolor, label='MAE (ms)')
        
        # Plot MAPE bars (right axis)
        bars2 = ax_pred_bar_right.bar(x + width/2, mape_values, width,
                                         color=[num_trains_colors[i] for i in range(len(unique_num_trains))],
                                         alpha=0.5, edgecolor=edgecolor, hatch='//', label='MAPE (%)')
        
        # Add value labels on bars
        for i, (bar1, bar2, mae, mape, count) in enumerate(zip(bars1, bars2, mae_values, mape_values, count_values)):
            # MAE label
            ax_pred_bar.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max(mae_values)*0.02,
                               f'{mae:.0f}', ha='center', va='bottom', fontsize=9, rotation=90)
            # MAPE label
            ax_pred_bar_right.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max(mape_values)*0.02,
                                      f'{mape:.0f}%', ha='center', va='bottom', fontsize=9, rotation=90)
            # Count label at bottom
            ax_pred_bar.text(i, 0, f'n={count}', ha='center', va='bottom', fontsize=8, color='black')
        
        # Set labels and title
        ax_pred_bar.set_xlabel('numTrains', fontsize=14, fontweight='bold')
        ax_pred_bar.set_ylabel('MAE (ms)', fontsize=12, fontweight='bold', color='navy')
        ax_pred_bar_right.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold', color='darkred')
        ax_pred_bar.set_title(f'Prediction Accuracy by Training Iteration ({metric_name})', 
                                 fontsize=16, fontweight='bold', pad=10)
        
        # Set x-axis
        ax_pred_bar.set_xticks(x)
        ax_pred_bar.set_xticklabels([f'{nt}' for nt in unique_num_trains], fontsize=11)
        
        # Set y-axis colors
        ax_pred_bar.tick_params(axis='y', labelcolor='navy', labelsize=11)
        ax_pred_bar_right.tick_params(axis='y', labelcolor='darkred', labelsize=11)
        
        # Add grid
        ax_pred_bar.grid(True, alpha=0.3, axis='y')
        
        # Add legends
        lines1, labels1 = ax_pred_bar.get_legend_handles_labels()
        lines2, labels2 = ax_pred_bar_right.get_legend_handles_labels()
        ax_pred_bar.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left', ncol=2)
        
        # Set y-limits for both axes with same relative padding (20% extra space on top)
        ax_pred_bar.set_ylim(0, max(mae_values) * 1.4)
        ax_pred_bar_right.set_ylim(0, max(mape_values) * 1.4)
        
        # SUBPLOT 2: Actual vs Predicted Scatter Plot (ax_pred_scatter)
        # Create color map for scatter plot
        num_trains_color_map = dict(zip(unique_num_trains, num_trains_colors))
        
        # Scatter plot of actual vs predicted, colored by numTrains
        for num_trains in unique_num_trains:
            subset = valid_predictions[valid_predictions['num_trains'] == num_trains]
            if len(subset) > 0:
                ax_pred_scatter.scatter(subset[actual_col], subset['chosen_pod_predicted_latency'],
                                       s=10, color=num_trains_color_map[num_trains], alpha=0.4, marker='.',
                                       label=f'{num_trains}')

        # Add diagonal line for perfect prediction
        max_val = max(valid_predictions[actual_col].max(), valid_predictions['chosen_pod_predicted_latency'].max())
        min_val = min(valid_predictions[actual_col].min(), valid_predictions['chosen_pod_predicted_latency'].min())
        ax_pred_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=linewidth, alpha=alpha, label='Perfect')

        # Set same range for x and y axes, starting from 0
        ax_pred_scatter.set_xlim(0, max_val)
        ax_pred_scatter.set_ylim(0, max_val)

        # Set same grid intervals for both axes to create square grid cells
        import math
        n_grid_lines = 5  # Fewer grid lines for smaller subplot
        grid_interval = math.ceil(max_val / n_grid_lines / 1000) * 1000

        tick_positions = [i * grid_interval for i in range(0, int(max_val / grid_interval) + 2)]
        tick_positions = [pos for pos in tick_positions if pos <= max_val]

        ax_pred_scatter.set_xticks(tick_positions)
        ax_pred_scatter.set_yticks(tick_positions)
        ax_pred_scatter.set_xticklabels([f'{int(tick)}' for tick in tick_positions], fontsize=9)
        ax_pred_scatter.set_yticklabels([f'{int(tick)}' for tick in tick_positions], fontsize=9)

        ax_pred_scatter.xaxis.set_major_locator(ticker.FixedLocator(tick_positions))
        ax_pred_scatter.yaxis.set_major_locator(ticker.FixedLocator(tick_positions))

        ax_pred_scatter.set_xlabel(f'Actual {metric_name} (ms)', fontsize=10, fontweight='bold')
        ax_pred_scatter.set_ylabel('Predicted (ms)', fontsize=10, fontweight='bold')
        ax_pred_scatter.set_title(title_scatter, fontsize=14, fontweight='bold', pad=10)
        ax_pred_scatter.grid(True, alpha=0.3, which='major')
        # ax_pred_scatter.legend(fontsize=7, loc='upper left', title='numTrains', title_fontsize=7)
        ax_pred_scatter.tick_params(axis='x', rotation=45, labelsize=9)
        ax_pred_scatter.set_aspect('equal', adjustable='box')

    else:
        ax_pred_bar.text(0.5, 0.5, 'No Valid Prediction Data Available', transform=ax_pred_bar.transAxes,
                            ha='center', va='center', fontsize=16, alpha=alpha)
        ax_pred_bar.set_title(f'Prediction Accuracy by Training Iteration ({metric_name})', 
                                 fontsize=16, fontweight='bold', pad=10)
        ax_pred_scatter.text(0.5, 0.5, 'No Valid Prediction Data Available', transform=ax_pred_scatter.transAxes,
                            ha='center', va='center', fontsize=16, alpha=alpha)
        ax_pred_scatter.set_title(title_scatter, fontsize=14, fontweight='bold', pad=10)

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

    add_transition_lines(ax_pred_timeseries, train_transitions, flush_transitions, iteration_transitions)
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

    # ========== PREDICTION ACCURACY BY ITERATIONS ==========
    # Row 28: Prediction Accuracy by Iterations (full width, shifted up by 1 after removing reward plot)
    ax_iter_pred_bar = fig.add_subplot(gs[28, :])
    
    # Row 29: Centered scatter plot (bigger, square)
    ax_iter_pred_scatter = fig.add_subplot(gs[29, :])  # Actual vs Predicted Scatter Plot by Iterations (full width)
    
    # Get unique iterations
    unique_iterations = sorted(df['iteration'].unique())
    
    # Define colors for iterations (used across prediction plots)
    iteration_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_iterations)))
    iteration_color_map = dict(zip(unique_iterations, iteration_colors))
    
    # SUBPLOT: Prediction Accuracy Bar Chart by Iterations (ax_iter_pred_bar)
    if not valid_predictions.empty:
        # Calculate accuracy metrics for each iteration
        mae_values_iter = []
        mape_values_iter = []
        count_values_iter = []
        iterations_with_data = []  # Track which iterations have data
        
        for iteration in unique_iterations:
            subset = valid_predictions[valid_predictions['iteration'] == iteration]
            if len(subset) > 0:
                mae = (subset[actual_col] - subset['chosen_pod_predicted_latency']).abs().mean()
                mape = ((subset[actual_col] - subset['chosen_pod_predicted_latency']).abs() / subset[actual_col]).mean() * 100
                
                mae_values_iter.append(mae)
                mape_values_iter.append(mape)
                count_values_iter.append(len(subset))
                iterations_with_data.append(iteration)  # Track this iteration
        
        if mae_values_iter:
            # Create bar chart with grouped bars for MAE and MAPE - only for iterations with data
            x = np.arange(len(iterations_with_data))
            width = 0.35
            
            # Create twin axis for MAPE
            ax_iter_pred_bar_right = ax_iter_pred_bar.twinx()
            
            # Plot MAE bars (left axis)
            bars1 = ax_iter_pred_bar.bar(x - width/2, mae_values_iter, width, 
                                        color=[iteration_color_map[it] for it in iterations_with_data],
                                        alpha=0.8, edgecolor=edgecolor, label='MAE (ms)')
            
            # Plot MAPE bars (right axis)
            bars2 = ax_iter_pred_bar_right.bar(x + width/2, mape_values_iter, width,
                                             color=[iteration_color_map[it] for it in iterations_with_data],
                                             alpha=0.5, edgecolor=edgecolor, hatch='//', label='MAPE (%)')
            
            # Add value labels on bars
            for i, (bar1, bar2, mae, mape, count) in enumerate(zip(bars1, bars2, mae_values_iter, mape_values_iter, count_values_iter)):
                # MAE label
                ax_iter_pred_bar.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max(mae_values_iter)*0.02,
                                   f'{mae:.0f}', ha='center', va='bottom', fontsize=9, rotation=90)
                # MAPE label
                ax_iter_pred_bar_right.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max(mape_values_iter)*0.02,
                                          f'{mape:.0f}%', ha='center', va='bottom', fontsize=9, rotation=90)
                # Count label at bottom
                ax_iter_pred_bar.text(i, 0, f'n={count}', ha='center', va='bottom', fontsize=8, color='black')
            
            # Set labels and title
            ax_iter_pred_bar.set_xlabel('Iteration', fontsize=14, fontweight='bold')
            ax_iter_pred_bar.set_ylabel('MAE (ms)', fontsize=12, fontweight='bold', color='navy')
            ax_iter_pred_bar_right.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold', color='darkred')
            ax_iter_pred_bar.set_title(f'Prediction Accuracy by Iterations ({metric_name})', 
                                     fontsize=16, fontweight='bold', pad=10)
            
            # Set x-axis
            ax_iter_pred_bar.set_xticks(x)
            ax_iter_pred_bar.set_xticklabels([f'{it}' for it in iterations_with_data], fontsize=11)
            
            # Set y-axis colors
            ax_iter_pred_bar.tick_params(axis='y', labelcolor='navy', labelsize=11)
            ax_iter_pred_bar_right.tick_params(axis='y', labelcolor='darkred', labelsize=11)
            
            # Add grid
            ax_iter_pred_bar.grid(True, alpha=0.3, axis='y')
            
            # Add legends
            lines1, labels1 = ax_iter_pred_bar.get_legend_handles_labels()
            lines2, labels2 = ax_iter_pred_bar_right.get_legend_handles_labels()
            ax_iter_pred_bar.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left', ncol=2)
            
            # Set y-limits for both axes with same relative padding
            ax_iter_pred_bar.set_ylim(0, max(mae_values_iter) * 1.4)
            ax_iter_pred_bar_right.set_ylim(0, max(mape_values_iter) * 1.4)
    
    # SUBPLOT: Actual vs Predicted Scatter Plot colored by Iterations (ax_iter_pred_scatter)
    if not valid_predictions.empty:
        # Scatter plot of actual vs predicted, colored by iteration
        for iteration in unique_iterations:
            subset = valid_predictions[valid_predictions['iteration'] == iteration]
            if len(subset) > 0:
                ax_iter_pred_scatter.scatter(subset[actual_col], subset['chosen_pod_predicted_latency'],
                                       s=10, color=iteration_color_map[iteration], alpha=0.4, marker='.',
                                       label=f'{iteration}')
        
        # Add diagonal line for perfect prediction
        max_val = max(valid_predictions[actual_col].max(), valid_predictions['chosen_pod_predicted_latency'].max())
        min_val = min(valid_predictions[actual_col].min(), valid_predictions['chosen_pod_predicted_latency'].min())
        ax_iter_pred_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=linewidth, alpha=alpha, label='Perfect')
        
        # Set same range for x and y axes
        ax_iter_pred_scatter.set_xlim(0, max_val)
        ax_iter_pred_scatter.set_ylim(0, max_val)
        
        # Set same grid intervals for both axes
        import math
        n_grid_lines = 5
        grid_interval = math.ceil(max_val / n_grid_lines / 1000) * 1000
        
        tick_positions = [i * grid_interval for i in range(0, int(max_val / grid_interval) + 2)]
        tick_positions = [pos for pos in tick_positions if pos <= max_val]
        
        ax_iter_pred_scatter.set_xticks(tick_positions)
        ax_iter_pred_scatter.set_yticks(tick_positions)
        ax_iter_pred_scatter.set_xticklabels([f'{int(tick)}' for tick in tick_positions], fontsize=9)
        ax_iter_pred_scatter.set_yticklabels([f'{int(tick)}' for tick in tick_positions], fontsize=9)
        
        ax_iter_pred_scatter.xaxis.set_major_locator(ticker.FixedLocator(tick_positions))
        ax_iter_pred_scatter.yaxis.set_major_locator(ticker.FixedLocator(tick_positions))
        
        ax_iter_pred_scatter.set_xlabel(f'Actual {metric_name} (ms)', fontsize=10, fontweight='bold')
        ax_iter_pred_scatter.set_ylabel('Predicted (ms)', fontsize=10, fontweight='bold')
        ax_iter_pred_scatter.set_title(f'Actual vs Predicted {metric_name} by Iterations', fontsize=14, fontweight='bold', pad=10)
        ax_iter_pred_scatter.grid(True, alpha=0.3, which='major')
        ax_iter_pred_scatter.legend(fontsize=5, loc='upper left', title='Iteration', title_fontsize=7, ncol=4)
        ax_iter_pred_scatter.tick_params(axis='x', rotation=45, labelsize=9)
        ax_iter_pred_scatter.set_aspect('equal', adjustable='box')

    return ax_pred_bar, ax_pred_scatter, ax_pred_timeseries, ax_iter_pred_bar, ax_iter_pred_scatter


def create_enhanced_plot(data, log_dir, setylim, slo_ttft, slo_tpot, routing_policy):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    if len(df) == 0:
        print("Error, No valid data to plot.")
        exit()
    
    # Filter out rows with negative iteration or numTrains values
    # But allow -99 as a sentinel value meaning "not set" (common in non-learning policies like prefix_cache)
    original_count = len(df)
    if 'iteration' in df.columns:
        # Convert to numeric, handling string values and Go formatting errors
        df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')
        # Filter out negative values EXCEPT -99 (sentinel for "not set")
        # Keep: >= 0, == -99, or NaN (invalid/unparseable values)
        df = df[(df['iteration'] >= 0) | (df['iteration'] == -99) | (df['iteration'].isna())]
    if 'num_trains' in df.columns:
        # Convert to numeric, handling string values like "%!d(string={})"
        df['num_trains'] = pd.to_numeric(df['num_trains'], errors='coerce')
        # Filter out negative values, but allow NaN (from parsing errors) and -99
        df = df[(df['num_trains'] >= 0) | (df['num_trains'] == -99) | (df['num_trains'].isna())]
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows with negative iteration or numTrains values ({len(df)} rows remaining)")
    
    if len(df) == 0:
        print("Error, No valid data remaining after filtering negative values.")
        print(f"Note: This might happen if all rows have invalid iteration/numTrains values.")
        print(f"For non-learning policies (like prefix_cache), iteration=-99 is normal and should be allowed.")
        exit()
    
    # Calculate reward columns first before cluster statistics
    df['ttft_reward'] = df['ttft'].apply(lambda x: calculate_ttft_reward(x, slo_ttft))
    df['tpot_reward'] = df['avg_tpot'].apply(lambda x: calculate_tpot_reward(x, slo_tpot))
    df['total_reward'] = df['ttft_reward'] + df['tpot_reward']
    
    # Use iteration from data if available, otherwise create bins based on order
    if 'iteration' not in df.columns or df['iteration'].isna().all():
        # Fallback: Create iteration bins if not in data
        iteration_size = max(100, len(df) // 20)
        df['iteration'] = df.index // iteration_size
        print("Warning: 'iteration' field not found in data. Creating artificial iterations based on request order.")
    else:
        print(f"Using 'iteration' field from data. Found {df['iteration'].nunique()} unique iterations.")
    
    # Now calculate cluster statistics (which need total_reward column)
    cluster_stats = calculate_cluster_wise_metrics(df)
    
    # Convert filtered df back to list format for transition calculations (to exclude negative values)
    filtered_data = df.to_dict('records')
    train_transitions = get_numtrains_transitions(filtered_data)
    flush_transitions = get_numflush_transitions(filtered_data)
    iteration_transitions = get_iteration_transitions(filtered_data)
    
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
    if 'latency_predictor' in routing_policy or 'contextual_bandit' in routing_policy:
        n_rows = 30  # Includes numTrains analysis + iteration analysis + prediction plots (removed reward plot, so 31->30)
        # Rows: 0-2 (request rate), 3-5 (TTFT/TPOT/E2E), 6-15 (pod metrics), 16-17 (analysis/CDF), 19-21 (numTrains), 22-24 (pred by numTrains), 25-27 (iterations), 28-29 (pred by iterations)
        height_ratios = [0.8, 0.8, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0]  # Rows 23 & 29 (scatter plots) are 3.0
        fig_height = 75
    else:
        n_rows = 23  # Includes iteration analysis only (no numTrains or prediction plots, removed reward plot, so 24->23)
        height_ratios = [0.8, 0.8, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # Iteration analysis at rows 20-22
        fig_height = 57

    # Create a more complex figure with GridSpec - Updated with additional subplots
    fig = plt.figure(figsize=(15, fig_height))
    gs = GridSpec(n_rows, 3, figure=fig, height_ratios=height_ratios, hspace=1.0, top=0.96)
    
    # Plot all subplots
    ax_total_rate, ax_token_rate, ax_pod_rate = plot_request_rate_subplots(
        fig, gs, plot_data, train_transitions, flush_transitions, iteration_transitions, unique_pods, pod_colors)
    
    main_metrics_axes = plot_main_metrics_subplots(
        fig, gs, df, pod_data, cluster_stats, train_transitions, flush_transitions, iteration_transitions, unique_pods, pod_colors, plot_data)
    ax1, ax2, ax3 = main_metrics_axes[:3]  # Extract the first 3 axes for backward compatibility
    
    analysis_axes = plot_analysis_subplots(
        fig, gs, df, slo_stats, slo_ttft, slo_tpot, unique_pods, pod_colors)

    # Plot numTrains analysis and prediction analysis only for latency_predictor policies
    if 'latency_predictor' in routing_policy or 'contextual_bandit' in routing_policy:
        # numTrains analysis: rows 19-21 (shifted up by 1 after removing reward plot)
        numtrains_axes = plot_numtrains_analysis_subplots(fig, gs, df, start_row=19)
        
        # Iteration analysis: rows 25-27
        iteration_axes = plot_iteration_analysis_subplots(fig, gs, df, start_row=25)
        
        if 'latency_predictor' in routing_policy:
            # Prediction analysis: rows 22-24 (numTrains) and rows 28-29 (iterations)
            prediction_axes = plot_prediction_analysis_subplots(fig, gs, df, train_transitions, flush_transitions, iteration_transitions, unique_pods, pod_colors, ax1, routing_policy)
            ax_pred_bar, ax_pred_scatter, ax_pred_timeseries, ax_iter_pred_bar, ax_iter_pred_scatter = prediction_axes
        else:
            prediction_axes = None
    else:
        # Only iteration analysis for non-predictor policies: rows 20-22
        numtrains_axes = None
        iteration_axes = plot_iteration_analysis_subplots(fig, gs, df, start_row=20)
        prediction_axes = None

    # Set font sizes for tick labels
    all_axes = [ax_total_rate, ax_token_rate, ax_pod_rate] + list(main_metrics_axes) + analysis_axes + list(iteration_axes)
    if numtrains_axes is not None:
        all_axes += list(numtrains_axes)
    if prediction_axes is not None:
        all_axes += list(prediction_axes)
    
    # Create a set of axes to skip for tick formatting (only if prediction_axes exist)
    skip_axes_set = set()
    if prediction_axes is not None and 'latency_predictor' in routing_policy:
        ax_pred_bar, ax_pred_scatter, ax_pred_timeseries, ax_iter_pred_bar, ax_iter_pred_scatter = prediction_axes
        skip_axes_set = {ax_pred_scatter, ax_iter_pred_scatter}
    
    for ax in all_axes:
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Force y-axis tick generation for axes with data
        ylim = ax.get_ylim()
        # Do not override custom ticks on the Actual vs Predicted scatter plots
        if ax in skip_axes_set:
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
    else:
        ax1.set_ylim(0, df['ttft'].max() * 1.1)
        ax2.set_ylim(0, df['avg_tpot'].max() * 1.1)
        ax3.set_ylim(0, df['e2e'].max() * 1.1)
    
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
    plt.savefig(plt_png_fn, bbox_inches='tight', dpi=200)
    print(f"*****************************")
    print(f"** Saving plot to: {plt_pdf_fn}")
    print(f"** Saving plot to: {plt_png_fn}")
    print(f"*****************************")
    
    return fig

def create_simple_timeseries_plot(data, log_dir, slo_ttft, slo_tpot, routing_policy):
    """Create a simplified plot with TTFT and TPOT time series, CDFs, and bar plots"""
    # Convert to DataFrame
    df = pd.DataFrame(data)
    if len(df) == 0:
        print("Error: No valid data to plot.")
        return None
    
    # Filter out negative values (same as main plot)
    original_count = len(df)
    if 'iteration' in df.columns:
        df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')
        df = df[(df['iteration'] >= 0) | (df['iteration'] == -99) | (df['iteration'].isna())]
    if 'num_trains' in df.columns:
        df['num_trains'] = pd.to_numeric(df['num_trains'], errors='coerce')
        df = df[(df['num_trains'] >= 0) | (df['num_trains'] == -99) | (df['num_trains'].isna())]
    
    if len(df) == 0:
        print("Error: No valid data remaining after filtering.")
        return None
    
    # Get transitions
    filtered_data = df.to_dict('records')
    train_transitions = get_numtrains_transitions(filtered_data)
    iteration_transitions = get_iteration_transitions(filtered_data)
    
    # Calculate 1-second sliding window averages
    df['time_bin'] = np.floor(df['relative_time']).astype(int)
    ttft_avg_per_sec = df.groupby('time_bin')['ttft'].mean().reset_index()
    tpot_avg_per_sec = df.groupby('time_bin')['avg_tpot'].mean().reset_index()
    
    # Get unique pods and create color map
    unique_pods = list(df['selectedpod'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_pods)))
    pod_colors = dict(zip(unique_pods, colors))
    pod_counts = df.groupby('selectedpod').size()
    
    # Calculate RPS per second
    total_requests_per_sec = df.groupby('time_bin').size().reset_index(name='total_requests')
    
    # Determine layout based on routing policy
    if 'latency_predictor' in routing_policy or 'contextual_bandit' in routing_policy:
        # With numTrains analysis: 9 rows (added RPS row)
        n_rows = 9
        height_ratios = [0.8, 1.2, 1.2, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2]  # RPS, time series, bar plots, CDFs for both TTFT and TPOT
        fig_height = 30
    else:
        # Without numTrains analysis: 7 rows (added RPS row)
        n_rows = 7
        height_ratios = [0.8, 1.2, 1.2, 1.5, 1.5, 1.2, 1.2]  # RPS, time series, bar plots, CDFs by iteration only
        fig_height = 24
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, fig_height))
    gs = GridSpec(n_rows, 3, figure=fig, height_ratios=height_ratios, hspace=0.4, wspace=0.3)
    
    # Row 0: RPS Time Series (full width)
    ax_rps = fig.add_subplot(gs[0, :])
    ax_rps.plot(total_requests_per_sec['time_bin'], total_requests_per_sec['total_requests'], 
                '-', color='blue', linewidth=linewidth, alpha=alpha, label='Total RPS')
    
    # Add transition lines
    for transition in train_transitions:
        ax_rps.axvline(x=transition['relative_time'], color='red', linewidth=transition_linewidth, 
                       linestyle='--', zorder=5)
    if iteration_transitions:
        for transition in iteration_transitions:
            ax_rps.axvline(x=transition['relative_time'], color='blue', linewidth=transition_linewidth, 
                          linestyle='-.', zorder=5)
    
    ax_rps.set_ylabel('Requests/sec', fontsize=14, fontweight='bold')
    ax_rps.set_title('Requests Per Second (RPS)', fontsize=16, fontweight='bold', pad=10)
    ax_rps.grid(True, alpha=alpha)
    ax_rps.tick_params(axis='both', which='major', labelsize=11)
    
    # Row 1: TTFT Time Series (full width)
    ax_ttft = fig.add_subplot(gs[1, :], sharex=ax_rps)
    ax_ttft.plot(ttft_avg_per_sec['time_bin'], ttft_avg_per_sec['ttft'], 
                 color='red', linewidth=linewidth, alpha=alpha, label='Avg TTFT (per sec)', zorder=10)
    
    # Add transition lines
    for transition in train_transitions:
        ax_ttft.axvline(x=transition['relative_time'], color='red', linewidth=transition_linewidth, 
                       linestyle='--', zorder=5)
    if iteration_transitions:
        for transition in iteration_transitions:
            ax_ttft.axvline(x=transition['relative_time'], color='blue', linewidth=transition_linewidth, 
                          linestyle='-.', zorder=5)
    
    ax_ttft.set_ylabel('TTFT (ms)', fontsize=14, fontweight='bold')
    ax_ttft.set_title('Time to First Token (TTFT) - 1-Second Sliding Window Average', 
                      fontsize=16, fontweight='bold', pad=10)
    ax_ttft.grid(True, alpha=alpha)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=linewidth, label='Avg TTFT (per sec)'),
        Line2D([0], [0], color='red', linewidth=linewidth, linestyle='--', label='numTrains transition'),
        Line2D([0], [0], color='blue', linewidth=linewidth, linestyle='-.', label='Iteration transition')
    ]
    ax_ttft.legend(handles=legend_elements, fontsize=10, loc='upper right', ncol=3)
    ax_ttft.tick_params(axis='both', which='major', labelsize=11)
    
    # Row 2: TPOT Time Series (full width)
    ax_tpot = fig.add_subplot(gs[2, :], sharex=ax_ttft)
    ax_tpot.plot(tpot_avg_per_sec['time_bin'], tpot_avg_per_sec['avg_tpot'], 
                 color='red', linewidth=linewidth, alpha=alpha, label='Avg TPOT (per sec)', zorder=10)
    
    # Add transition lines
    for transition in train_transitions:
        ax_tpot.axvline(x=transition['relative_time'], color='red', linewidth=transition_linewidth, 
                       linestyle='--', zorder=5)
    if iteration_transitions:
        for transition in iteration_transitions:
            ax_tpot.axvline(x=transition['relative_time'], color='blue', linewidth=transition_linewidth, 
                          linestyle='-.', zorder=5)
    
    ax_tpot.set_xlabel('Relative Time (seconds)', fontsize=14, fontweight='bold')
    ax_tpot.set_ylabel('TPOT (ms)', fontsize=14, fontweight='bold')
    ax_tpot.set_title('Time Per Output Token (TPOT) - 1-Second Sliding Window Average', 
                      fontsize=16, fontweight='bold', pad=10)
    ax_tpot.grid(True, alpha=alpha)
    
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=linewidth, label='Avg TPOT (per sec)')
    ]
    ax_tpot.legend(handles=legend_elements, fontsize=10, loc='upper right')
    ax_tpot.tick_params(axis='both', which='major', labelsize=11)
    
    # Row 3: Bar plots - TTFT by Pod | TPOT by Pod | empty
    ax_ttft_bar = fig.add_subplot(gs[3, 0])
    ax_tpot_bar = fig.add_subplot(gs[3, 1])
    
    # TTFT by Pod
    pod_avg_ttft = df.groupby('selectedpod')['ttft'].mean().sort_values(ascending=False)
    bars = ax_ttft_bar.bar(range(len(pod_avg_ttft)), pod_avg_ttft.values,
                           color=[pod_colors[pod] for pod in pod_avg_ttft.index],
                           edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha, width=0.7)
    for i, (pod, ttft) in enumerate(pod_avg_ttft.items()):
        ax_ttft_bar.text(i, ttft + ttft*0.02, f'{ttft:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax_ttft_bar.text(i, ttft*0.05, f'n={pod_counts[pod]}', ha='center', fontsize=8)
    ax_ttft_bar.set_xticks(range(len(pod_avg_ttft)))
    ax_ttft_bar.set_xticklabels([f'Pod {pod}' for pod in pod_avg_ttft.index], rotation=45, ha='right', fontsize=10)
    ax_ttft_bar.set_ylabel('Average TTFT (ms)', fontsize=12, fontweight='bold')
    ax_ttft_bar.set_title('Average TTFT by Pod', fontsize=14, fontweight='bold', pad=10)
    ax_ttft_bar.grid(axis='y', alpha=alpha)
    
    # TPOT by Pod
    pod_avg_tpot = df.groupby('selectedpod')['avg_tpot'].mean().sort_values(ascending=False)
    bars = ax_tpot_bar.bar(range(len(pod_avg_tpot)), pod_avg_tpot.values,
                           color=[pod_colors[pod] for pod in pod_avg_tpot.index],
                           edgecolor=edgecolor, linewidth=edgewidth, alpha=alpha, width=0.7)
    for i, (pod, tpot) in enumerate(pod_avg_tpot.items()):
        ax_tpot_bar.text(i, tpot + tpot*0.02, f'{tpot:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax_tpot_bar.text(i, tpot*0.05, f'n={pod_counts[pod]}', ha='center', fontsize=8)
    ax_tpot_bar.set_xticks(range(len(pod_avg_tpot)))
    ax_tpot_bar.set_xticklabels([f'Pod {pod}' for pod in pod_avg_tpot.index], rotation=45, ha='right', fontsize=10)
    ax_tpot_bar.set_ylabel('Average TPOT (ms)', fontsize=12, fontweight='bold')
    ax_tpot_bar.set_title('Average TPOT by Pod', fontsize=14, fontweight='bold', pad=10)
    ax_tpot_bar.grid(axis='y', alpha=alpha)
    
    # Row 4: CDFs - TTFT CDF | TPOT CDF | empty
    ax_ttft_cdf = fig.add_subplot(gs[4, 0])
    ax_tpot_cdf = fig.add_subplot(gs[4, 1])
    
    # TTFT CDF
    sorted_ttft = np.sort(df['ttft'])
    y_ttft = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
    ax_ttft_cdf.plot(sorted_ttft, y_ttft, color='blue', linewidth=linewidth, alpha=alpha)
    p50_ttft = np.percentile(df['ttft'], 50)
    p95_ttft = np.percentile(df['ttft'], 95)
    p99_ttft = np.percentile(df['ttft'], 99)
    avg_ttft = df['ttft'].mean()
    ax_ttft_cdf.axvline(p50_ttft, color='red', linestyle='--', alpha=alpha, label=f'P50: {p50_ttft:.1f}ms')
    ax_ttft_cdf.axvline(p95_ttft, color='orange', linestyle='--', alpha=alpha, label=f'P95: {p95_ttft:.1f}ms')
    ax_ttft_cdf.axvline(p99_ttft, color='purple', linestyle='--', alpha=alpha, label=f'P99: {p99_ttft:.1f}ms')
    ax_ttft_cdf.axvline(avg_ttft, color='green', linestyle='-', alpha=alpha, label=f'Avg: {avg_ttft:.1f}ms')
    ax_ttft_cdf.set_xlabel('TTFT (ms)', fontsize=10, fontweight='bold')
    ax_ttft_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_ttft_cdf.set_title('TTFT CDF', fontsize=12, fontweight='bold', pad=10)
    ax_ttft_cdf.grid(True, alpha=alpha)
    ax_ttft_cdf.legend(fontsize=8)
    
    # TPOT CDF
    sorted_tpot = np.sort(df['avg_tpot'])
    y_tpot = np.arange(1, len(sorted_tpot) + 1) / len(sorted_tpot)
    ax_tpot_cdf.plot(sorted_tpot, y_tpot, color='green', linewidth=linewidth, alpha=alpha)
    p50_tpot = np.percentile(df['avg_tpot'], 50)
    p95_tpot = np.percentile(df['avg_tpot'], 95)
    p99_tpot = np.percentile(df['avg_tpot'], 99)
    avg_tpot = df['avg_tpot'].mean()
    ax_tpot_cdf.axvline(p50_tpot, color='red', linestyle='--', alpha=alpha, label=f'P50: {p50_tpot:.1f}ms')
    ax_tpot_cdf.axvline(p95_tpot, color='orange', linestyle='--', alpha=alpha, label=f'P95: {p95_tpot:.1f}ms')
    ax_tpot_cdf.axvline(p99_tpot, color='purple', linestyle='--', alpha=alpha, label=f'P99: {p99_tpot:.1f}ms')
    ax_tpot_cdf.axvline(avg_tpot, color='green', linestyle='-', alpha=alpha, label=f'Avg: {avg_tpot:.1f}ms')
    ax_tpot_cdf.set_xlabel('TPOT (ms)', fontsize=10, fontweight='bold')
    ax_tpot_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_tpot_cdf.set_title('TPOT CDF', fontsize=12, fontweight='bold', pad=10)
    ax_tpot_cdf.grid(True, alpha=alpha)
    ax_tpot_cdf.legend(fontsize=8)
    
    current_row = 5
    
    # Add numTrains analysis if applicable
    if 'latency_predictor' in routing_policy or 'contextual_bandit' in routing_policy:
        unique_num_trains = sorted(df['num_trains'].unique())
        num_trains_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_num_trains)))
        num_trains_color_map = dict(zip(unique_num_trains, num_trains_colors))
        
        # Row 5: TTFT by numTrains | TPOT by numTrains | empty
        ax_nt_ttft_cdf = fig.add_subplot(gs[current_row, 0])
        ax_nt_tpot_cdf = fig.add_subplot(gs[current_row, 1])
        
        # TTFT CDF by numTrains
        for num_trains in unique_num_trains:
            subset = df[df['num_trains'] == num_trains]
            if len(subset) > 0:
                sorted_ttft = np.sort(subset['ttft'])
                cdf = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
                avg_ttft = subset['ttft'].mean()
                p99_ttft = np.percentile(subset['ttft'], 99)
                ax_nt_ttft_cdf.plot(sorted_ttft, cdf, label=f'numTrains={num_trains}, avg: {avg_ttft:.0f}ms, p99: {p99_ttft:.0f}ms',
                                    color=num_trains_color_map[num_trains], linewidth=1.5, alpha=0.7)
        ax_nt_ttft_cdf.set_xlabel('TTFT (ms)', fontsize=10, fontweight='bold')
        ax_nt_ttft_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
        ax_nt_ttft_cdf.set_title('TTFT CDF by numTrains', fontsize=12, fontweight='bold', pad=10)
        ax_nt_ttft_cdf.grid(True, alpha=0.3)
        ax_nt_ttft_cdf.legend(fontsize=6, loc='lower right')
        
        # TPOT CDF by numTrains
        for num_trains in unique_num_trains:
            subset = df[df['num_trains'] == num_trains]
            if len(subset) > 0:
                sorted_tpot = np.sort(subset['avg_tpot'])
                cdf = np.arange(1, len(sorted_tpot) + 1) / len(sorted_tpot)
                avg_tpot = subset['avg_tpot'].mean()
                p99_tpot = np.percentile(subset['avg_tpot'], 99)
                ax_nt_tpot_cdf.plot(sorted_tpot, cdf, label=f'numTrains={num_trains}, avg: {avg_tpot:.0f}ms, p99: {p99_tpot:.0f}ms',
                                    color=num_trains_color_map[num_trains], linewidth=1.5, alpha=0.7)
        ax_nt_tpot_cdf.set_xlabel('TPOT (ms)', fontsize=10, fontweight='bold')
        ax_nt_tpot_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
        ax_nt_tpot_cdf.set_title('TPOT CDF by numTrains', fontsize=12, fontweight='bold', pad=10)
        ax_nt_tpot_cdf.grid(True, alpha=0.3)
        ax_nt_tpot_cdf.legend(fontsize=6, loc='lower right')
        
        current_row += 1
    
    # Iteration analysis CDFs
    unique_iterations = sorted(df['iteration'].unique())
    iteration_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_iterations)))
    iteration_color_map = dict(zip(unique_iterations, iteration_colors))
    
    # Row 6 or current_row: TTFT by Iteration | TPOT by Iteration | empty
    ax_iter_ttft_cdf = fig.add_subplot(gs[current_row, 0])
    ax_iter_tpot_cdf = fig.add_subplot(gs[current_row, 1])
    
    # TTFT CDF by Iterations
    for iteration in unique_iterations:
        subset = df[df['iteration'] == iteration]
        if len(subset) > 0:
            sorted_ttft = np.sort(subset['ttft'])
            cdf = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
            avg_ttft = subset['ttft'].mean()
            p99_ttft = np.percentile(subset['ttft'], 99)
            ax_iter_ttft_cdf.plot(sorted_ttft, cdf, label=f'Iter {iteration}, avg: {avg_ttft:.0f}ms, p99: {p99_ttft:.0f}ms',
                                  color=iteration_color_map[iteration], linewidth=1.5, alpha=0.7)
    ax_iter_ttft_cdf.set_xlabel('TTFT (ms)', fontsize=10, fontweight='bold')
    ax_iter_ttft_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_iter_ttft_cdf.set_title('TTFT CDF by Iterations', fontsize=12, fontweight='bold', pad=10)
    ax_iter_ttft_cdf.grid(True, alpha=0.3)
    ax_iter_ttft_cdf.legend(fontsize=7, loc='lower right', ncol=2)
    
    # TPOT CDF by Iterations
    for iteration in unique_iterations:
        subset = df[df['iteration'] == iteration]
        if len(subset) > 0:
            sorted_tpot = np.sort(subset['avg_tpot'])
            cdf = np.arange(1, len(sorted_tpot) + 1) / len(sorted_tpot)
            avg_tpot = subset['avg_tpot'].mean()
            p99_tpot = np.percentile(subset['avg_tpot'], 99)
            ax_iter_tpot_cdf.plot(sorted_tpot, cdf, label=f'Iter {iteration}, avg: {avg_tpot:.0f}ms, p99: {p99_tpot:.0f}ms',
                                  color=iteration_color_map[iteration], linewidth=1.5, alpha=0.7)
    ax_iter_tpot_cdf.set_xlabel('TPOT (ms)', fontsize=10, fontweight='bold')
    ax_iter_tpot_cdf.set_ylabel('CDF', fontsize=10, fontweight='bold')
    ax_iter_tpot_cdf.set_title('TPOT CDF by Iterations', fontsize=12, fontweight='bold', pad=10)
    ax_iter_tpot_cdf.grid(True, alpha=0.3)
    ax_iter_tpot_cdf.legend(fontsize=7, loc='lower right', ncol=2)
    
    # Add super title
    fig.suptitle(f'Latency Metrics Summary (Total Requests: {len(data)})', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save files
    plt_pdf_fn = f"{log_dir}/latency_timeseries_simple.pdf"
    plt_png_fn = f"{log_dir}/latency_timeseries_simple.png"
    plt.savefig(plt_pdf_fn, bbox_inches='tight')
    plt.savefig(plt_png_fn, bbox_inches='tight', dpi=200)
    print(f"*****************************")
    print(f"** Saving simple timeseries plot to: {plt_pdf_fn}")
    print(f"** Saving simple timeseries plot to: {plt_png_fn}")
    print(f"*****************************")
    
    return fig

import argparse

parser = argparse.ArgumentParser(description='Plot latency metrics analysis')
parser.add_argument('log_file', type=str, help='Path to the log file')
parser.add_argument('--setylim', type=int, default=0, help='Set y-axis limits')
parser.add_argument('--slo_ttft', type=int, default=1000, help='SLO TTFT')
parser.add_argument('--slo_tpot', type=int, default=50, help='SLO TPOT')
parser.add_argument('--skip-first-seconds', type=float, default=0, help='Skip/truncate the first X seconds of data (default: 30s)')


if __name__ == "__main__":
    args = parser.parse_args()

    log_file = args.log_file
    # Get absolute path to handle relative paths correctly
    log_file = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else '.'
    setylim = args.setylim
    slo_ttft = args.slo_ttft
    slo_tpot = args.slo_tpot
    skip_first_seconds = args.skip_first_seconds
    
    # Extract routing policy from path structure more robustly
    path_parts = log_file.split('/')
    if len(path_parts) >= 2:
        # Try to extract from second-to-last directory
        try:
            routing_policy = path_parts[-2].split('-')[0]
        except (IndexError, AttributeError):
            routing_policy = "unknown"
    else:
        # If no directory structure, try to extract from filename
        filename = os.path.basename(log_file)
        try:
            routing_policy = filename.split('-')[0]
        except (IndexError, AttributeError):
            routing_policy = "unknown"
    
    print(f"routing_policy: {routing_policy}")
    data = parse_log_file(log_file)
    
    if not data:
        print(f"Error: No valid latency metrics found in {log_file}. Please check the file format.")
        assert False
    
    print(f"Found {len(data)} log entries with latency metrics")
    
    # Filter out first X seconds if specified
    if skip_first_seconds > 0:
        original_count = len(data)
        data = [entry for entry in data if entry.get('relative_time', 0) >= skip_first_seconds]
        filtered_count = original_count - len(data)
        print(f"Skipped first {skip_first_seconds} seconds: removed {filtered_count} entries, {len(data)} entries remaining")
        
        if not data:
            print(f"Error: No data remaining after skipping first {skip_first_seconds} seconds.")
            assert False
    
    # Create and save the enhanced plot
    fig = create_enhanced_plot(data, log_dir, setylim, slo_ttft, slo_tpot, routing_policy)
    
    # Create and save the simple timeseries plot
    fig_simple = create_simple_timeseries_plot(data, log_dir, slo_ttft, slo_tpot, routing_policy)
    
    # Print summary statistics
    df = pd.DataFrame(data)
    print("\nSummary Statistics:")
    print(f"TTFT - Min: {df['ttft'].min()} ms, Max: {df['ttft'].max()} ms, Avg: {df['ttft'].mean():.2f} ms")
    print(f"TPOT - Min: {df['avg_tpot'].min()} ms, Max: {df['avg_tpot'].max()} ms, Avg: {df['avg_tpot'].mean():.2f} ms")
    print(f"E2E  - Min: {df['e2e'].min()} ms, Max: {df['e2e'].max()} ms, Avg: {df['e2e'].mean():.2f} ms")