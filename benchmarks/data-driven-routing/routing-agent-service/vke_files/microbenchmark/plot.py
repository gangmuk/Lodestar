#!/usr/bin/env python3
"""
Professional plotting script for microbenchmark TTFT results.
Generates publication-quality PDF plots showing the impact of input length 
and prefix cache hit ratio on Time To First Token (TTFT).
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Set professional plot style
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 12

def parse_result_file(filename='result.csv'):
    """Parse the microbenchmark result file and extract TTFT data."""
    data = defaultdict(lambda: defaultdict(list))
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Parse by sections based on prefix hit ratio
    prefix_sections = re.split(r'={30,}\nPrefix Hit Ratio: ([\d.]+)\n={30,}', content)
    
    for i in range(1, len(prefix_sections), 2):
        prefix_ratio = float(prefix_sections[i])
        section = prefix_sections[i + 1]
        
        # Find all input length sections
        input_sections = re.split(r'\*{30,}\nInput length: (\d+)\n\*{30,}', section)
        
        for j in range(1, len(input_sections), 2):
            input_length = int(input_sections[j])
            subsection = input_sections[j + 1]
            
            # Extract all TTFT values
            ttft_matches = re.findall(r'x-timing-ttft-ms: (\d+)', subsection)
            ttft_values = [int(x) for x in ttft_matches]
            
            if ttft_values:
                data[prefix_ratio][input_length] = ttft_values
    
    return data

def plot_ttft_vs_input_length(data, output_file='ttft_vs_input_length.pdf'):
    """Plot TTFT vs Input Length for different prefix hit ratios."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Define colors and markers for different prefix ratios
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    prefix_ratios = sorted(data.keys())
    
    for idx, prefix_ratio in enumerate(prefix_ratios):
        input_lengths = sorted(data[prefix_ratio].keys())
        mean_ttfts = []
        std_ttfts = []
        valid_lengths = []
        
        for length in input_lengths:
            ttfts = data[prefix_ratio][length]
            if ttfts:
                mean_ttfts.append(np.mean(ttfts))
                std_ttfts.append(np.std(ttfts))
                valid_lengths.append(length)
        
        if mean_ttfts:
            label = f'{int(prefix_ratio*100)}%'
            ax.plot(valid_lengths, mean_ttfts, 
                   marker=markers[idx], 
                   color=colors[idx],
                   linewidth=2, 
                   markersize=8,
                   label=label,
                   alpha=0.9)
            
            # Add error bars
            ax.errorbar(valid_lengths, mean_ttfts, yerr=std_ttfts,
                       color=colors[idx], alpha=0.3, fmt='none', capsize=3)
    
    # ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=-0.5)
    ax.set_xlabel('Input Length (tokens)')
    ax.set_ylabel('TTFT (ms)')
    ax.legend(title='Prefix Hit Ratio', title_fontsize=11, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks([2000, 4000, 8000, 16000])
    ax.set_xticklabels(['2K', '4K', '8K', '16K'])
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()
    
def main():
    """Main function to generate all plots."""
    print("Parsing result file...")
    import sys
    input = sys.argv[1]
    data = parse_result_file(input)
    plot_ttft_vs_input_length(data, 'ttft_vs_input_length.pdf')

if __name__ == '__main__':
    main()

