import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

def parse_log_file(log_dir, filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    # Splitting by log entries
    entries = re.findall(r'I\d+.*?@latency_metrics@.*?@numDecodeTokensForAllPods@\{.*?\}', content, re.DOTALL)
    
    data = []
    for entry in entries:
        request_id = re.search(r'@requestID@(\d+)@', entry)
        start_time = re.search(r'@request_start_time@(\d+)@', entry)
        end_time = re.search(r'@request_end_time@(\d+)@', entry)
        ttft = re.search(r'@ttft@(\d+)@', entry)
        avg_tpot = re.search(r'@avg_tpot@(\d+)@', entry)
        selected_pod = re.search(r'@selectedpod@([\d\.]+)@', entry)
        total_decode_time = re.search(r'@total_decode_time@(\d+)@', entry)
        e2e = re.search(r'@e2e@(\d+)@', entry)
        num_input_tokens = re.search(r'@numInputTokens@(\d+)@', entry)
        num_output_tokens = re.search(r'@numOutputTokens@(\d+)@', entry)
        
        if all([request_id, start_time, ttft, avg_tpot, selected_pod, total_decode_time, e2e, num_input_tokens, num_output_tokens]):
            data.append({
                'request_id': int(request_id.group(1)),
                'start_time': int(start_time.group(1)),
                'end_time': int(end_time.group(1)) if end_time else None,
                'ttft': int(ttft.group(1)),
                'avg_tpot': int(avg_tpot.group(1)),
                'pod': selected_pod.group(1),
                'total_decode_time': int(total_decode_time.group(1)),
                'e2e': int(e2e.group(1)),
                'num_input_tokens': int(num_input_tokens.group(1)),
                'num_output_tokens': int(num_output_tokens.group(1))
            })
    
    # Sort data by start time
    data.sort(key=lambda x: x['start_time'])
    
    # Calculate relative time in seconds
    if data:
        base_time = data[0]['start_time']
        for item in data:
            item['relative_time'] = (item['start_time'] - base_time) / 1000000  # Convert to seconds
            if item['end_time']:
                item['relative_end_time'] = (item['end_time'] - base_time) / 1000000
                item['duration'] = (item['end_time'] - item['start_time']) / 1000000  # Duration in seconds
    
    return data

def create_enhanced_plot(data):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Create a more complex figure with GridSpec
    fig = plt.figure(figsize=(15, 20))  # Increased height to accommodate all plots
    gs = GridSpec(7, 2, figure=fig, height_ratios=[1, 1, 1, 1, 0.8, 0.8, 0.8])
    
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
    ax6 = fig.add_subplot(gs[4, :])  # TTFT/TPOT distribution
    
   # Color mapping for different pods
    unique_pods = list(df['pod'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_pods)))

    # Define different marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']

    pod_colors = dict(zip(unique_pods, colors))
    pod_marker = dict(zip(unique_pods, markers))
    # TTFT Plot (ax1)
    for pod in unique_pods:
        pod_data = df[df['pod'] == pod]
        ax1.scatter(pod_data['relative_time'], pod_data['ttft'], s=100, 
                #    color=pod_colors[pod], edgecolor='black', linewidth=1.5, alpha=0.8,
                   color=pod_colors[pod], alpha=0.7, edgecolor='gray', linewidth=0.5, marker=pod_marker[pod],
                #    color=pod_colors[pod], alpha=0.5,
                   label=f'Pod: {pod}')
    # ax1.plot(df['relative_time'], df['ttft'], 'k--', alpha=0.5)
    
    # TPOT Plot (ax2)
    for pod in unique_pods:
        pod_data = df[df['pod'] == pod]
        ax2.scatter(pod_data['relative_time'], pod_data['avg_tpot'], s=100, 
                   color=pod_colors[pod], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.plot(df['relative_time'], df['avg_tpot'], 'k--', alpha=0.5)
    
    # E2E Duration Plot (ax3) - Now following the same format as TTFT and TPOT
    for pod in unique_pods:
        pod_data = df[df['pod'] == pod]
        ax3.scatter(pod_data['relative_time'], pod_data['e2e'], s=100, 
                   color=pod_colors[pod], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.plot(df['relative_time'], df['e2e'], 'k--', alpha=0.5)
    
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
    sns.kdeplot(df['ttft'], ax=ax6, fill=True, alpha=0.7, color='blue')
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
    sns.kdeplot(df['avg_tpot'], ax=ax7, fill=True, alpha=0.7, color='green')
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
    sns.kdeplot(df['e2e'], ax=ax8, fill=True, alpha=0.7, color='purple')
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
    ax1.set_ylabel('TTFT (ms)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
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
    
    # Set y-axis limits with some padding
    ax1.set_ylim(0, df['ttft'].max() * 1.1)
    ax2.set_ylim(0, df['avg_tpot'].max() * 1.1)
    ax3.set_ylim(0, df['e2e'].max() * 1.1)
    
    # Add a super title
    fig.suptitle('Latency Metrics Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save files
    plt.savefig("latency_metrics_analysis.png", dpi=300, bbox_inches='tight')
    plt_fn = f"{log_dir}/latency_metrics_analysis.pdf"
    plt.savefig(plt_fn, bbox_inches='tight')
    
    return fig, plt_fn

if __name__ == "__main__":
    # Parse the log file
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_latency_timeseries.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    log_dir = log_file.rsplit('/', 1)[0]
    # Parse the log file to get data
    data = parse_log_file(log_dir, log_file)
    
    if not data:
        print(f"No valid latency metrics found in {log_file}. Please check the file format.")
        sys.exit(1)
    
    print(f"Found {len(data)} log entries with latency metrics")
    
    # Create and save the enhanced plot
    fig, output_file = create_enhanced_plot(data)
    
    print(f"Plot saved to: {output_file}")
    
    # Print summary statistics
    df = pd.DataFrame(data)
    print("\nSummary Statistics:")
    print(f"TTFT - Min: {df['ttft'].min()} ms, Max: {df['ttft'].max()} ms, Avg: {df['ttft'].mean():.2f} ms")
    print(f"TPOT - Min: {df['avg_tpot'].min()} ms, Max: {df['avg_tpot'].max()} ms, Avg: {df['avg_tpot'].mean():.2f} ms")
    print(f"E2E  - Min: {df['e2e'].min()} ms, Max: {df['e2e'].max()} ms, Avg: {df['e2e'].mean():.2f} ms")
    
    # Show the plot
    plt.show()