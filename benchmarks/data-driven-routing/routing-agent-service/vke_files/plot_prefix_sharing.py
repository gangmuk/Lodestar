#!/usr/bin/env python3
"""Visualize temporal prefix sharing: original vs shuffled workload side-by-side."""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_workload(path):
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            records.append({
                'timestamp_ms': r['timestamp'],
                'hash_ids': r['requests'][0]['hash_ids'],
                'prefix_group': r['requests'][0]['prefix_group'],
            })
    return records


def find_lcp(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def compute_all(records):
    """Compute all metrics for a workload. Returns a dict of plottable data."""
    data = {}

    # Per-minute prefix sharing ratio
    by_minute = defaultdict(list)
    for r in records:
        minute = int(r['timestamp_ms'] / 1000 / 60)
        by_minute[minute].append(r)

    minute_keys, minute_ratios = [], []
    for m in sorted(by_minute.keys()):
        reqs = by_minute[m]
        if len(reqs) <= 1:
            minute_keys.append(m)
            minute_ratios.append(0.0)
            continue
        ratios = []
        for i in range(len(reqs)):
            hids = reqs[i]['hash_ids']
            if not hids:
                continue
            best = 0
            for j in range(i):
                best = max(best, find_lcp(hids, reqs[j]['hash_ids']))
            ratios.append(best / len(hids))
        minute_keys.append(m)
        minute_ratios.append(np.mean(ratios) if ratios else 0.0)
    data['minute_keys'] = minute_keys
    data['minute_ratios'] = minute_ratios

    # Per-second prefix sharing ratio
    by_second = defaultdict(list)
    for r in records:
        sec = int(r['timestamp_ms'] / 1000)
        by_second[sec].append(r)

    sec_keys, sec_ratios = [], []
    for s in sorted(by_second.keys()):
        reqs = by_second[s]
        if len(reqs) <= 1:
            sec_keys.append(s)
            sec_ratios.append(0.0)
            continue
        ratios = []
        for i in range(len(reqs)):
            hids = reqs[i]['hash_ids']
            if not hids:
                continue
            best = 0
            for j in range(i):
                best = max(best, find_lcp(hids, reqs[j]['hash_ids']))
            ratios.append(best / len(hids))
        sec_keys.append(s)
        sec_ratios.append(np.mean(ratios) if ratios else 0.0)
    data['sec_keys'] = sec_keys
    data['sec_ratios'] = sec_ratios

    # Per-request: best LCP length & time gap (100-req window)
    time_gaps, lcp_lengths, req_timestamps = [], [], []
    for i in range(1, len(records)):
        hids = records[i]['hash_ids']
        best_lcp, best_j = 0, -1
        for j in range(max(0, i - 100), i):
            lcp = find_lcp(hids, records[j]['hash_ids'])
            if lcp > best_lcp:
                best_lcp = lcp
                best_j = j
        if best_lcp > 0:
            gap = (records[i]['timestamp_ms'] - records[best_j]['timestamp_ms']) / 1000
            time_gaps.append(gap)
            lcp_lengths.append(best_lcp)
            req_timestamps.append(records[i]['timestamp_ms'] / 1000)
    data['time_gaps'] = time_gaps
    data['lcp_lengths'] = lcp_lengths
    data['req_timestamps'] = req_timestamps

    # Inter-arrival times within same prefix group
    group_timestamps = defaultdict(list)
    for r in records:
        group_timestamps[r['prefix_group']].append(r['timestamp_ms'] / 1000)
    inter_arrivals = []
    for ts_list in group_timestamps.values():
        ts_list.sort()
        for i in range(1, len(ts_list)):
            inter_arrivals.append(ts_list[i] - ts_list[i - 1])
    data['inter_arrivals'] = inter_arrivals
    data['group_timestamps'] = group_timestamps

    # Top prefix groups for activity timeline
    top_groups = sorted(group_timestamps.keys(), key=lambda g: -len(group_timestamps[g]))
    top_groups = [g for g in top_groups if len(group_timestamps[g]) > 5][:30]
    data['top_groups'] = top_groups

    # Root prefix composition over time (10s bins)
    bin_size = 10
    max_sec = int(records[-1]['timestamp_ms'] / 1000) + 1
    first_hid_set = sorted(set(r['hash_ids'][0] for r in records if r['hash_ids']))
    hid_to_idx = {h: i for i, h in enumerate(first_hid_set)}
    bins = range(0, max_sec + bin_size, bin_size)
    stacked = np.zeros((len(first_hid_set), len(bins)))
    for r in records:
        if not r['hash_ids']:
            continue
        b = int(r['timestamp_ms'] / 1000) // bin_size
        if b < len(bins):
            stacked[hid_to_idx[r['hash_ids'][0]], b] += 1
    data['bin_size'] = bin_size
    data['bins'] = bins
    data['stacked'] = stacked
    data['first_hid_set'] = first_hid_set
    data['max_sec'] = max_sec

    return data


def plot_column(fig, axes, col, data, title_suffix, color_accent):
    """Plot one column (original or shuffled) of the 6-row x 2-col figure."""
    max_sec = data['max_sec']

    # Row 0: Prefix sharing ratio over time
    ax = axes[0, col]
    ax.plot(data['sec_keys'], data['sec_ratios'], linewidth=0.6, alpha=0.4,
            color=color_accent, label='Per-second')
    minute_keys_sec = [m * 60 + 30 for m in data['minute_keys']]
    ax.plot(minute_keys_sec, data['minute_ratios'], linewidth=2.5,
            color='navy' if col == 0 else 'darkred', marker='o', markersize=5,
            label='Per-minute avg', zorder=5)
    ax.set_ylabel('Prefix Sharing Ratio')
    ax.set_title(f'Prefix Sharing Ratio — {title_suffix}')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(range(0, max_sec + 1, 60)))
    ax.set_xticklabels([f"{m}m" for m in range(len(range(0, max_sec + 1, 60)))], fontsize=7)
    # Add overall average text
    if data['minute_ratios']:
        avg = np.mean(data['minute_ratios'])
        ax.axhline(avg, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(max_sec * 0.98, avg + 0.03, f'avg={avg:.3f}', fontsize=8,
                ha='right', color='gray')

    # Row 1: Temporal distance to best prefix match
    ax = axes[1, col]
    if data['time_gaps']:
        sc = ax.scatter(data['req_timestamps'], data['time_gaps'],
                        c=data['lcp_lengths'], cmap='viridis',
                        s=3, alpha=0.3, rasterized=True)
        fig.colorbar(sc, ax=ax, label='Best LCP (blocks)', shrink=0.8)
    ax.set_ylabel('Time Gap to Best Match (s)')
    ax.set_title(f'Temporal Distance to Best Match — {title_suffix}')
    ax.set_xlabel('Request Time (s)')
    ax.grid(True, alpha=0.3)

    # Row 2: Inter-arrival time CDF
    ax = axes[2, col]
    if data['inter_arrivals']:
        ia_sorted = np.sort(data['inter_arrivals'])
        cdf = np.arange(1, len(ia_sorted) + 1) / len(ia_sorted)
        ax.plot(ia_sorted, cdf, linewidth=1.5, color='darkorange')
        ax.set_xscale('log')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        for thresh, label in [(1, '1s'), (10, '10s'), (60, '60s')]:
            ax.axvline(thresh, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
            ax.text(thresh, 0.02, label, fontsize=7, color='red', ha='center')
    ax.set_ylabel('CDF')
    ax.set_title(f'Inter-arrival CDF (Same Prefix Group) — {title_suffix}')
    ax.set_xlabel('Inter-arrival Time (s)')
    ax.grid(True, alpha=0.3)

    # Row 3: Prefix group activity timeline
    ax = axes[3, col]
    for idx, pg in enumerate(data['top_groups']):
        ts = data['group_timestamps'][pg]
        ax.scatter(ts, [idx] * len(ts), s=2, alpha=0.6, color=color_accent)
        ax.plot([min(ts), max(ts)], [idx, idx], linewidth=0.5, alpha=0.3, color='gray')
    ax.set_ylabel('Prefix Group (top 30)')
    ax.set_title(f'Prefix Group Activity — {title_suffix}')
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')

    # Row 4: Root prefix composition
    ax = axes[4, col]
    bin_centers = [b * data['bin_size'] + data['bin_size'] / 2 for b in range(len(data['bins']))]
    labels = [f"hid={h}" for h in data['first_hid_set']]
    ax.stackplot(bin_centers, data['stacked'], labels=labels, alpha=0.8)
    ax.set_ylabel(f"Reqs per {data['bin_size']}s bin")
    ax.set_title(f'Root Prefix Composition — {title_suffix}')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 5: LCP length histogram
    ax = axes[5, col]
    if data['lcp_lengths']:
        ax.hist(data['lcp_lengths'], bins=range(0, max(data['lcp_lengths']) + 2),
                alpha=0.7, color='teal' if col == 0 else 'coral',
                edgecolor='black', linewidth=0.3)
        mean_lcp = np.mean(data['lcp_lengths'])
        ax.axvline(mean_lcp, color='red', linestyle='--', linewidth=1.2,
                   label=f'Mean={mean_lcp:.1f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Best LCP Length (hash_id blocks)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Best Prefix Match Length — {title_suffix}')
    ax.grid(True, alpha=0.3)


def shuffle_workload(records):
    """Shuffle request contents while preserving the original timestamp distribution.

    This randomly reassigns which request content lands at which timestamp,
    destroying temporal locality (e.g., requests from the same conversation
    no longer arrive close together in time).
    """
    import random
    rng = random.Random(42)
    timestamps = [r['timestamp_ms'] for r in records]
    contents = [{'hash_ids': r['hash_ids'], 'prefix_group': r['prefix_group']} for r in records]
    rng.shuffle(contents)
    shuffled = []
    for ts, c in zip(timestamps, contents):
        shuffled.append({'timestamp_ms': ts, **c})
    return shuffled


def main():
    input_file = sys.argv[1]
    workload_name = input_file.rsplit('/', 1)[-1].rsplit('.', 1)[0]

    print("Loading original workload...")
    orig = load_workload(input_file)
    print(f"  {len(orig)} requests")

    print("Creating shuffled workload (timestamps preserved, content reassigned)...")
    shuf = shuffle_workload(orig)
    print(f"  {len(shuf)} requests")

    print("Computing metrics for original...")
    data_orig = compute_all(orig)
    print("Computing metrics for shuffled...")
    data_shuf = compute_all(shuf)

    # ── Side-by-side plot: 6 rows x 2 cols ──
    fig, axes = plt.subplots(6, 2, figsize=(16, 24))
    fig.suptitle(f'Temporal Prefix Sharing: Original vs Shuffled ({workload_name})',
                 fontsize=17, fontweight='bold', y=0.995)

    plot_column(fig, axes, 0, data_orig, 'Original', 'steelblue')
    plot_column(fig, axes, 1, data_shuf, 'Shuffled', 'indianred')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out = 'plot_prefix_sharing_comparison.pdf'
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close()


if __name__ == '__main__':
    main()
