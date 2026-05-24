#!/usr/bin/env python3
"""
Analyze latency distribution from training data to inform reward function design.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_latency_distribution(csv_path):
    """Analyze latency distribution and visualize for reward function design."""
    
    # Read data
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Basic statistics
    print("\n" + "="*80)
    print("LATENCY STATISTICS")
    print("="*80)
    
    print(f"\nTTFT (ms):")
    print(f"  Min: {df['ttft'].min():.1f}")
    print(f"  P5: {df['ttft'].quantile(0.05):.1f}")
    print(f"  P25: {df['ttft'].quantile(0.25):.1f}")
    print(f"  Median: {df['ttft'].median():.1f}")
    print(f"  P75: {df['ttft'].quantile(0.75):.1f}")
    print(f"  P95: {df['ttft'].quantile(0.95):.1f}")
    print(f"  P99: {df['ttft'].quantile(0.99):.1f}")
    print(f"  Max: {df['ttft'].max():.1f}")
    print(f"  Mean: {df['ttft'].mean():.1f}")
    print(f"  Std: {df['ttft'].std():.1f}")
    
    print(f"\nInput Tokens:")
    print(f"  Min: {df['input_tokens'].min():.0f}")
    print(f"  P25: {df['input_tokens'].quantile(0.25):.0f}")
    print(f"  Median: {df['input_tokens'].median():.0f}")
    print(f"  P75: {df['input_tokens'].quantile(0.75):.0f}")
    print(f"  Max: {df['input_tokens'].max():.0f}")
    
    # Analyze by input length buckets
    print("\n" + "="*80)
    print("LATENCY BY INPUT LENGTH BUCKETS")
    print("="*80)
    
    # Define buckets
    input_quantiles = df['input_tokens'].quantile([0, 0.33, 0.67, 1.0]).values
    bucket_names = [
        f"Short ({input_quantiles[0]:.0f}-{input_quantiles[1]:.0f} tokens)",
        f"Medium ({input_quantiles[1]:.0f}-{input_quantiles[2]:.0f} tokens)",
        f"Long ({input_quantiles[2]:.0f}-{input_quantiles[3]:.0f} tokens)"
    ]
    
    for i, (low, high) in enumerate([(input_quantiles[0], input_quantiles[1]), 
                                      (input_quantiles[1], input_quantiles[2]), 
                                      (input_quantiles[2], input_quantiles[3])]):
        mask = (df['input_tokens'] >= low) & (df['input_tokens'] < high) if i < 2 else (df['input_tokens'] >= low)
        bucket_data = df[mask]['ttft']
        
        print(f"\n{bucket_names[i]}:")
        print(f"  Count: {len(bucket_data)}")
        print(f"  Min: {bucket_data.min():.1f}")
        print(f"  P50: {bucket_data.median():.1f}")
        print(f"  P95: {bucket_data.quantile(0.95):.1f}")
        print(f"  Max: {bucket_data.max():.1f}")
    
    # Test reward functions
    print("\n" + "="*80)
    print("REWARD FUNCTION COMPARISON")
    print("="*80)
    
    ttft = df['ttft'].values
    
    # Calculate rewards using different functions
    reward_log = -np.log(ttft + 1.0)
    reward_reciprocal = -1000.0 / np.maximum(ttft, 1.0)
    reward_linear = -ttft / 1000.0
    reward_squared = -np.square(ttft / 1000.0)
    
    reward_functions = {
        'simple_latency_minimization (-log)': reward_log,
        'negative_reciprocal (-1000/lat)': reward_reciprocal,
        'negative_linear (-lat/1000)': reward_linear,
        'negative_squared (-(lat/1000)^2)': reward_squared
    }
    
    print("\nReward Statistics:")
    for name, rewards in reward_functions.items():
        print(f"\n{name}:")
        print(f"  Range: [{rewards.min():.3f}, {rewards.max():.3f}]")
        print(f"  Span: {rewards.max() - rewards.min():.3f}")
        print(f"  Mean: {rewards.mean():.3f}")
        print(f"  Std: {rewards.std():.3f}")
        
        # Check differentiation by input length
        spreads = []
        for i, (low, high) in enumerate([(input_quantiles[0], input_quantiles[1]), 
                                          (input_quantiles[1], input_quantiles[2]), 
                                          (input_quantiles[2], input_quantiles[3])]):
            mask = (df['input_tokens'] >= low) & (df['input_tokens'] < high) if i < 2 else (df['input_tokens'] >= low)
            bucket_rewards = rewards[mask]
            spreads.append(bucket_rewards.max() - bucket_rewards.min())
        
        print(f"  Spread by bucket: {[f'{s:.3f}' for s in spreads]}")
        print(f"  Avg spread: {np.mean(spreads):.3f}")
    
    # Create visualization
    print("\n" + "="*80)
    print("Creating visualization...")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Latency distribution (overall)
    plt.subplot(3, 4, 1)
    plt.hist(ttft, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.median(ttft), color='r', linestyle='--', linewidth=2, label=f'Median: {np.median(ttft):.0f}ms')
    plt.axvline(np.percentile(ttft, 95), color='orange', linestyle='--', linewidth=2, label=f'P95: {np.percentile(ttft, 95):.0f}ms')
    plt.xlabel('TTFT (ms)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('1. Latency Distribution', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Latency by input length (stratified)
    plt.subplot(3, 4, 2)
    colors = ['green', 'orange', 'red']
    for i, (low, high) in enumerate([(input_quantiles[0], input_quantiles[1]), 
                                      (input_quantiles[1], input_quantiles[2]), 
                                      (input_quantiles[2], input_quantiles[3])]):
        mask = (df['input_tokens'] >= low) & (df['input_tokens'] < high) if i < 2 else (df['input_tokens'] >= low)
        plt.hist(ttft[mask], bins=30, alpha=0.5, color=colors[i], 
                edgecolor='black', label=bucket_names[i])
    plt.xlabel('TTFT (ms)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('2. Latency by Input Length', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 3. Latency vs Input Tokens (scatter)
    plt.subplot(3, 4, 3)
    plt.scatter(df['input_tokens'], df['ttft'], alpha=0.1, s=5, c='blue')
    
    # Add percentile lines
    input_bins = np.linspace(df['input_tokens'].min(), df['input_tokens'].max(), 20)
    medians = []
    p95s = []
    bin_centers = []
    for i in range(len(input_bins)-1):
        mask = (df['input_tokens'] >= input_bins[i]) & (df['input_tokens'] < input_bins[i+1])
        if mask.sum() > 10:
            medians.append(df[mask]['ttft'].median())
            p95s.append(df[mask]['ttft'].quantile(0.95))
            bin_centers.append((input_bins[i] + input_bins[i+1]) / 2)
    
    plt.plot(bin_centers, medians, 'r-', linewidth=3, label='Median', alpha=0.8)
    plt.plot(bin_centers, p95s, 'orange', linewidth=3, linestyle='--', label='P95', alpha=0.8)
    
    plt.xlabel('Input Tokens', fontsize=12)
    plt.ylabel('TTFT (ms)', fontsize=12)
    plt.title('3. Latency vs Input Length', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Log-scale latency distribution
    plt.subplot(3, 4, 4)
    plt.hist(np.log10(ttft + 1), bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Log10(TTFT + 1)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('4. Latency Dist (Log Scale)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5-8. Reward distributions for each function
    for idx, (name, rewards) in enumerate(reward_functions.items(), start=5):
        plt.subplot(3, 4, idx)
        
        # Overall distribution
        plt.hist(rewards, bins=50, alpha=0.3, color='blue', 
                histtype='stepfilled', label='All', zorder=1)
        
        # Stratified
        for i, (low, high) in enumerate([(input_quantiles[0], input_quantiles[1]), 
                                          (input_quantiles[1], input_quantiles[2]), 
                                          (input_quantiles[2], input_quantiles[3])]):
            mask = (df['input_tokens'] >= low) & (df['input_tokens'] < high) if i < 2 else (df['input_tokens'] >= low)
            bucket_rewards = rewards[mask]
            plt.hist(bucket_rewards, bins=30, alpha=0.5, color=colors[i], 
                    edgecolor='black', label=bucket_names[i].split('(')[0].strip(), zorder=2)
        
        plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        plt.xlabel('Reward', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'{idx}. {name}', fontsize=11, fontweight='bold')
        plt.legend(fontsize=8, loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add stats box
        span = rewards.max() - rewards.min()
        plt.text(0.98, 0.98, f'Range: {span:.2f}\nStd: {rewards.std():.2f}',
                transform=plt.gca().transAxes, verticalalignment='top', 
                horizontalalignment='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 9. Reward spread comparison
    plt.subplot(3, 4, 9)
    spreads_by_function = []
    function_names_short = ['Log', 'Reciprocal', 'Linear', 'Squared']
    
    for rewards in reward_functions.values():
        span = rewards.max() - rewards.min()
        spreads_by_function.append(span)
    
    bars = plt.bar(range(len(spreads_by_function)), spreads_by_function, 
                   color=['blue', 'green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    plt.xticks(range(len(function_names_short)), function_names_short, rotation=15)
    plt.ylabel('Reward Range (max - min)', fontsize=12)
    plt.title('9. Reward Range Comparison', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, spreads_by_function):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(spreads_by_function)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 10. Reward discrimination (good vs bad latency)
    plt.subplot(3, 4, 10)
    p50_lat = np.percentile(ttft, 50)
    p90_lat = np.percentile(ttft, 90)
    
    good_mask = ttft < p50_lat
    bad_mask = ttft >= p90_lat
    
    discrimination_scores = []
    for rewards in reward_functions.values():
        avg_good = rewards[good_mask].mean()
        avg_bad = rewards[bad_mask].mean()
        discrimination_scores.append(avg_good - avg_bad)
    
    bars = plt.bar(range(len(discrimination_scores)), discrimination_scores, 
                   color=['blue', 'green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    plt.xticks(range(len(function_names_short)), function_names_short, rotation=15)
    plt.ylabel('Reward Spread (P50 - P90)', fontsize=12)
    plt.title('10. Discrimination Power', fontsize=13, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, discrimination_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(discrimination_scores)*0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 11. Reward sensitivity to latency changes
    plt.subplot(3, 4, 11)
    # Sample latencies across the range
    sample_lats = np.linspace(ttft.min(), ttft.max(), 100)
    
    for name, color in zip(function_names_short, ['blue', 'green', 'orange', 'red']):
        if name == 'Log':
            sample_rewards = -np.log(sample_lats + 1.0)
        elif name == 'Reciprocal':
            sample_rewards = -1000.0 / np.maximum(sample_lats, 1.0)
        elif name == 'Linear':
            sample_rewards = -sample_lats / 1000.0
        else:  # Squared
            sample_rewards = -np.square(sample_lats / 1000.0)
        
        plt.plot(sample_lats, sample_rewards, linewidth=2, label=name, color=color, alpha=0.8)
    
    plt.xlabel('Latency (ms)', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('11. Reward Function Shapes', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Correlation with latency
    plt.subplot(3, 4, 12)
    correlations = []
    for rewards in reward_functions.values():
        corr = np.corrcoef(ttft, rewards)[0, 1]
        correlations.append(corr)
    
    bars = plt.bar(range(len(correlations)), correlations, 
                   color=['blue', 'green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    plt.xticks(range(len(function_names_short)), function_names_short, rotation=15)
    plt.ylabel('Correlation', fontsize=12)
    plt.title('12. Correlation with Latency', fontsize=13, fontweight='bold')
    plt.axhline(y=-1.0, color='green', linestyle='--', linewidth=1, label='Perfect: -1.0')
    plt.ylim(-1.05, -0.85)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, correlations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02,
                f'{val:.4f}', ha='center', va='top', fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(csv_path).parent / 'latency_reward_analysis.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.close()
    
    # Summary and recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    print("\nBased on the analysis:")
    print("\n1. NEGATIVE_RECIPROCAL (-1000/latency):")
    print("   ✅ Best balance: good range and discrimination")
    print("   ✅ More sensitive in low-latency range (100-1000ms)")
    print("   ✅ Natural diminishing returns for tail latencies")
    
    print("\n2. NEGATIVE_LINEAR (-latency/1000):")
    print("   ✅ Simplest and most interpretable")
    print("   ✅ Good discrimination across entire range")
    print("   ⚠️  Large range might need gradient clipping")
    
    print("\n3. NEGATIVE_SQUARED (-(latency/1000)^2):")
    print("   ✅ Excellent discrimination (largest range)")
    print("   ✅ Heavily penalizes tail latencies")
    print("   ⚠️  Very large range - may need clipping or cause training instability")
    
    print("\n4. SIMPLE_LATENCY_MIN (-log(latency)):")
    print("   ❌ Too compressed - poor differentiation")
    print("   ❌ Overlapping distributions by input length")
    print("   ❌ Not recommended")
    
    print("\n🎯 FINAL RECOMMENDATION: Start with NEGATIVE_RECIPROCAL")
    print("   - Best balance for RL training")
    print("   - If training is stable, try NEGATIVE_SQUARED for stronger tail-latency aversion")
    print("   - If you need simplicity and interpretability, use NEGATIVE_LINEAR")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = '/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/data-processed.csv'
    
    analyze_latency_distribution(csv_path)






