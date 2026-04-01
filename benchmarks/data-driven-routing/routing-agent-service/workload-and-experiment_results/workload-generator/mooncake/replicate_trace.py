#!/usr/bin/env python3
"""
Trace Replicator (Option C: group-lifecycle interleaving)

Replicates a Mooncake trace N times with remapped hash_ids, interleaving
at the prefix-group level so that all workload characteristics are preserved:
  - Requests per prefix group (identical)
  - Intra-group request ordering and spacing
  - New prefix group arrival density relative to total requests
  - Block count and output length distributions

Each copy's hash_ids are shifted into a separate ID space so there is no
cross-copy prefix sharing.  The interleaving places group lifecycles from
all copies in the order they originally appeared, preserving the local mix
of active groups.

Timestamps are NOT assigned — the output preserves request order only.
The caller is responsible for assigning timestamps (RPS shaping, stretching, etc.).

Usage:
    python replicate_trace.py --trace Mooncake_synthetic_trace.jsonl \
                              --copies 3 \
                              --output-dir replicated_synthetic_3x
"""

import json
import argparse
import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from realistic_workload_generator import load_vocab_csv, generate_workload


def load_trace(path):
    """Load trace JSONL, sort by timestamp, normalize to start at 0."""
    entries = []
    with open(path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    entries.sort(key=lambda e: e['timestamp'])
    t0 = entries[0]['timestamp']
    for e in entries:
        e['timestamp'] -= t0
    return entries


def extract_groups(entries):
    """Extract prefix groups and their ordered request sequences.

    A prefix group is defined by the first 3 hash_ids (matching the
    convention in generate_workload / the Mooncake trace).

    Returns:
        group_arrivals: list of (first_seen_index, group_key, [entry_indices])
            sorted by first_seen_index.  This captures the order in which
            new groups appear and each group's request sequence.
    """
    # Map group_key -> list of (original_index, entry)
    group_members = defaultdict(list)
    group_first_seen = {}

    for i, e in enumerate(entries):
        gk = tuple(e['hash_ids'][:3])
        group_members[gk].append(i)
        if gk not in group_first_seen:
            group_first_seen[gk] = i

    # Sort groups by their first appearance
    group_arrivals = []
    for gk, indices in group_members.items():
        group_arrivals.append((group_first_seen[gk], gk, indices))
    group_arrivals.sort(key=lambda x: x[0])

    return group_arrivals


def remap_entry(entry, hid_offset):
    """Create a copy of an entry with all hash_ids shifted by hid_offset."""
    return {
        'timestamp': entry['timestamp'],
        'input_length': entry['input_length'],
        'output_length': entry['output_length'],
        'hash_ids': [h + hid_offset for h in entry['hash_ids']],
    }


def replicate_trace(entries, n_copies):
    """Replicate trace N times with group-lifecycle interleaving.

    For each prefix group in the original trace, we create N versions
    (one per copy, each with remapped hash_ids).  Groups are interleaved
    in the order they first appear in the original, and within each group,
    requests maintain their original relative ordering.

    Each copy is offset by R/N positions so that copies are staggered evenly
    across the trace, then merged by their effective position.

    Example with 3 copies, R=9 original requests:
        Copy 0 starts at offset 0: positions 0,1,2,3,4,5,6,7,8
        Copy 1 starts at offset 3: positions 3,4,5,6,7,8,0,1,2  (wraps around)
        Copy 2 starts at offset 6: positions 6,7,8,0,1,2,3,4,5  (wraps around)

    Merged by position, each original position has requests from different
    copies arriving at different phases of their lifecycle.  This means:
      - Intra-group distance within each copy: preserved (D unchanged)
      - New groups arrive smoothly (no batch boundaries)
      - All distributions preserved
    """
    # Find max hash_id for offset calculation
    max_hid = 0
    for e in entries:
        for h in e['hash_ids']:
            if h > max_hid:
                max_hid = h
    hid_space = max_hid + 1

    group_arrivals = extract_groups(entries)

    print(f"Original trace: {len(entries)} requests, {len(group_arrivals)} prefix groups")
    print(f"Replicating {n_copies}x -> {len(entries) * n_copies} requests, "
          f"{len(group_arrivals) * n_copies} prefix groups")

    R = len(entries)
    # Each copy is offset by R/N positions (staggered start)
    # Build list of (effective_position, entry) then sort
    output_with_pos = []

    for copy_idx in range(n_copies):
        hid_offset = copy_idx * hid_space
        phase_offset = copy_idx * R / n_copies  # stagger copies evenly
        for i, entry in enumerate(entries):
            # Effective position: original position + phase offset, wrapped
            eff_pos = (i + phase_offset) % R
            output_with_pos.append((eff_pos, i, copy_idx, remap_entry(entry, hid_offset)))

    # Sort by effective position, break ties by original index, then copy
    output_with_pos.sort(key=lambda x: (x[0], x[1], x[2]))
    output = [item[3] for item in output_with_pos]

    # Sanity checks
    assert len(output) == len(entries) * n_copies, \
        f"Expected {len(entries) * n_copies} entries, got {len(output)}"

    return output


def verify_characteristics(entries, output, n_copies):
    """Print verification stats comparing original and replicated traces."""
    # Original group stats
    orig_groups = defaultdict(list)
    for i, e in enumerate(entries):
        gk = tuple(e['hash_ids'][:3])
        orig_groups[gk].append(i)
    orig_group_sizes = [len(v) for v in orig_groups.values()]

    # Replicated group stats
    rep_groups = defaultdict(list)
    for i, e in enumerate(output):
        gk = tuple(e['hash_ids'][:3])
        rep_groups[gk].append(i)
    rep_group_sizes = [len(v) for v in rep_groups.values()]

    print(f"\nVerification:")
    print(f"  Original: {len(entries)} requests, {len(orig_groups)} groups")
    print(f"  Replicated: {len(output)} requests, {len(rep_groups)} groups")
    print(f"  Expected groups: {len(orig_groups)} x {n_copies} = {len(orig_groups) * n_copies}, "
          f"got {len(rep_groups)}")
    print(f"  Original group sizes:   min={min(orig_group_sizes)}, max={max(orig_group_sizes)}, "
          f"mean={sum(orig_group_sizes)/len(orig_group_sizes):.1f}")
    print(f"  Replicated group sizes: min={min(rep_group_sizes)}, max={max(rep_group_sizes)}, "
          f"mean={sum(rep_group_sizes)/len(rep_group_sizes):.1f}")

    # Check intra-group distances (in number of requests between consecutive same-group requests)
    def intra_group_distances(grouped):
        dists = []
        for indices in grouped.values():
            for i in range(1, len(indices)):
                dists.append(indices[i] - indices[i - 1])
        return dists

    orig_dists = intra_group_distances(orig_groups)
    rep_dists = intra_group_distances(rep_groups)

    if orig_dists:
        import numpy as np
        print(f"  Original intra-group distance:   "
              f"median={np.median(orig_dists):.0f}, mean={np.mean(orig_dists):.1f}")
        print(f"  Replicated intra-group distance:  "
              f"median={np.median(rep_dists):.0f}, mean={np.mean(rep_dists):.1f}")
        print(f"  Expected ratio (Nx): {n_copies}x, "
              f"actual: {np.median(rep_dists)/np.median(orig_dists):.1f}x (median), "
              f"{np.mean(rep_dists)/np.mean(orig_dists):.1f}x (mean)")


def plot_comparison(entries_orig, entries_rep, n_copies, trace_name, save_path,
                    num_tokens_per_hash_id=50):
    """Plot 4x2 comparison: original (left) vs replicated (right).

    Uses the same trie-based prefix hit ratio as mooncake_plot.py.
    """
    NUM_TOKEN_PER_HASH_ID = num_tokens_per_hash_id

    def build_df(entries):
        df = pd.DataFrame(entries)
        df['timestamp_seconds'] = df['timestamp'] / 1000
        df['num_token_blocks'] = df['hash_ids'].apply(len)
        df['input_length_from_blocks'] = df['num_token_blocks'] * NUM_TOKEN_PER_HASH_ID
        return df

    def compute_prefix_hit_ratios_trie(df_sorted):
        class _TrieNode:
            __slots__ = ['children']
            def __init__(self):
                self.children = {}
        root = _TrieNode()
        hit_ratios = []
        positions = []
        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            hids = row['hash_ids']
            total_blocks = len(hids)
            node = root
            matched = 0
            for h in hids:
                if h in node.children:
                    matched += 1
                    node = node.children[h]
                else:
                    break
            hit_ratios.append(matched / total_blocks if total_blocks > 0 else 0.0)
            positions.append(idx)
            node = root
            for h in hids:
                if h not in node.children:
                    node.children[h] = _TrieNode()
                node = node.children[h]
        return positions, hit_ratios

    def plot_one(df, axes_col, label, use_position=False):
        """Plot 4 rows for one dataset into a column of axes."""
        if use_position:
            # x-axis = request index (position in trace)
            x_label = 'Request Index'
            x_max = len(df)
        else:
            x_label = 'Time (seconds)'
            x_max = int(df['timestamp_seconds'].max()) + 1

        # Row 0: Input token length distribution
        axes_col[0].hist(df['input_length_from_blocks'], bins=50, alpha=0.7,
                         color='lightgreen', edgecolor='black')
        axes_col[0].set_xlabel('Input Token Length (from blocks)')
        axes_col[0].set_ylabel('Frequency')
        axes_col[0].set_title(f'{label}: Input Token Length Dist')
        axes_col[0].grid(True, alpha=0.3)

        # Row 1: Output token length distribution
        axes_col[1].hist(df['output_length'], bins=50, alpha=0.7,
                         color='lightcoral', edgecolor='black')
        axes_col[1].set_xlabel('Output Token Length')
        axes_col[1].set_ylabel('Frequency')
        axes_col[1].set_title(f'{label}: Output Token Length Dist')
        axes_col[1].grid(True, alpha=0.3)

        # Row 2: Token blocks per request distribution
        axes_col[2].hist(df['num_token_blocks'], bins=30, alpha=0.7,
                         color='orange', edgecolor='black')
        axes_col[2].set_xlabel('Number of Token Blocks')
        axes_col[2].set_ylabel('Frequency')
        axes_col[2].set_title(f'{label}: Token Blocks Dist')
        axes_col[2].grid(True, alpha=0.3)

        # Row 3: Per-request prefix cache hit ratio (trie-based, by position)
        df_sorted = df.sort_values('timestamp_seconds').reset_index(drop=True)
        positions, hit_ratios = compute_prefix_hit_ratios_trie(df_sorted)

        axes_col[3].scatter(positions, hit_ratios, alpha=0.15, s=8,
                            color='purple', label='per-request')

        # 10s sliding window average (by position count matching ~10s worth of requests)
        pos_arr = np.array(positions)
        hr_arr = np.array(hit_ratios)
        # Window size: number of requests in 10 seconds
        if use_position:
            duration_s = (df['timestamp_seconds'].max() - df['timestamp_seconds'].min())
            if duration_s > 0:
                rps = len(df) / duration_s
                win = max(1, int(rps * 10))
            else:
                win = 10
        else:
            win = 10  # not used in position mode but keep as fallback

        # Rolling average over request positions
        if len(hr_arr) > 0:
            kernel = max(1, int(len(hr_arr) / (x_max / 10)) if not use_position else win)
            if use_position:
                kernel = win
            cumsum = np.cumsum(np.insert(hr_arr, 0, 0))
            rolling = (cumsum[kernel:] - cumsum[:-kernel]) / kernel
            rolling_x = pos_arr[kernel - 1:] if use_position else np.arange(kernel - 1, len(hr_arr))
            axes_col[3].plot(rolling_x, rolling, color='red', linewidth=1.5,
                             alpha=0.9, label=f'{kernel}-req rolling avg')

        axes_col[3].set_xlabel(x_label)
        axes_col[3].set_ylabel('Prefix Cache Hit Ratio')
        axes_col[3].set_title(f'{label}: Prefix Hit Ratio')
        axes_col[3].grid(True, alpha=0.3)
        axes_col[3].set_xlim(0, x_max)
        axes_col[3].set_ylim(0, 1)
        axes_col[3].legend(fontsize=8)

    df_orig = build_df(entries_orig)
    df_rep = build_df(entries_rep)

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f'{trace_name} — Original vs {n_copies}x Replicated '
                 f'({len(entries_orig)} -> {len(entries_rep)} requests)', fontsize=14)

    plot_one(df_orig, [axes[r, 0] for r in range(4)],
             f'Original ({len(entries_orig)} req)', use_position=True)
    plot_one(df_rep, [axes[r, 1] for r in range(4)],
             f'{n_copies}x Replicated ({len(entries_rep)} req)', use_position=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Replicate a Mooncake trace N times with group-lifecycle interleaving."
    )
    parser.add_argument("--trace", required=True, help="Path to input trace JSONL")
    parser.add_argument("--copies", type=int, required=True, help="Number of copies (N)")
    parser.add_argument("--output-dir", required=True, help="Output directory (will contain workload.jsonl and plot PDF)")
    parser.add_argument("--num-tokens-per-hash-id", type=int, default=100,
                        help="Tokens per hash_id block (default: 100)")
    parser.add_argument("--output-length-scale", type=float, default=1.0,
                        help="Scale output lengths (default: 1.0)")
    parser.add_argument("--vocab-csv", default="vocab.csv",
                        help="Path to vocab.csv (default: vocab.csv)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    if args.copies < 1:
        parser.error("--copies must be >= 1")

    entries = load_trace(args.trace)
    output = replicate_trace(entries, args.copies)
    verify_characteristics(entries, output, args.copies)

    # Convert to workload format with actual prompt text
    vocab = load_vocab_csv(args.vocab_csv)
    print(f"\nGenerating workload (num_tokens_per_hash_id={args.num_tokens_per_hash_id}, "
          f"output_length_scale={args.output_length_scale})")
    records = generate_workload(output, rps_scale=1.0,
                                num_tokens_per_hash_id=args.num_tokens_per_hash_id,
                                output_length_scale=args.output_length_scale,
                                vocab=vocab)

    os.makedirs(args.output_dir, exist_ok=True)

    jsonl_path = os.path.join(args.output_dir, 'workload.jsonl')
    with open(jsonl_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    print(f"Saved {len(records)} records to {jsonl_path}")

    if not args.no_plot:
        trace_name = os.path.splitext(os.path.basename(args.trace))[0]
        pdf_path = os.path.join(args.output_dir, f'{trace_name}_{args.copies}x_comparison.pdf')
        plot_comparison(entries, output, args.copies, trace_name, pdf_path,
                        num_tokens_per_hash_id=args.num_tokens_per_hash_id)


if __name__ == "__main__":
    main()
