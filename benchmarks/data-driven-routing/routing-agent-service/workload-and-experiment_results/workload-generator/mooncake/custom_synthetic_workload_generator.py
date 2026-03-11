#!/usr/bin/env python3
"""
Custom Stretched Synthetic Workload Generator

Stretches both the head (low-prefix-sharing) and tail (high-prefix-sharing)
regions of the Mooncake trace independently, while maintaining RPS by filling
stretched time with synthesized requests that follow the same prefix sharing
+ input/output length pattern.

Params beyond realistic_workload_generator.py:
  --stretch-after-pct:    fraction of trace where head/tail split occurs (default: 0.7)
  --head-stretch-factor:  how much to stretch the head in time (default: 1.0)
  --tail-stretch-factor:  how much to stretch the tail in time (default: 3.0)
"""

import json
import os
import argparse
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from realistic_workload_generator import load_vocab_csv, load_trace, generate_workload, compute_stats
from plot import plot_timeseries_analysis


def build_session_pool(entries):
    """Build a session pool from entries.

    A session is identified by the first hash_id. Returns:
      sessions: dict mapping first_hid -> {template, count}
    """
    sessions = defaultdict(lambda: {"template": None, "count": 0})
    for e in entries:
        hids = e['hash_ids']
        if not hids:
            continue
        key = hids[0]
        sessions[key]["count"] += 1
        if sessions[key]["template"] is None or len(hids) > len(sessions[key]["template"]):
            sessions[key]["template"] = list(hids)
    return sessions


def _synthesize_fill(region_entries, all_entries_for_hid, stretch_factor,
                     region_start_ts, region_label, rng, np_rng, next_unique_id):
    """Stretch a region and synthesize fill requests to maintain RPS.

    Returns (stretched_entries, new_entries, next_unique_id).
    """
    stretched = [dict(e) for e in region_entries]

    if not stretched or stretch_factor <= 1.0:
        return stretched, [], next_unique_id

    original_end_ts = stretched[-1]['timestamp']
    original_duration = original_end_ts - region_start_ts

    if original_duration <= 0:
        return stretched, [], next_unique_id

    # Stretch timestamps
    for e in stretched:
        e['timestamp'] = region_start_ts + (e['timestamp'] - region_start_ts) * stretch_factor

    stretched_duration = original_duration * stretch_factor
    original_rps = len(stretched) / (original_duration / 1000.0)
    extra_time_ms = stretched_duration - original_duration
    num_new = int(original_rps * (extra_time_ms / 1000.0))

    print(f"  {region_label}:")
    print(f"    Entries: {len(stretched)}, Original duration: {original_duration/1000:.1f}s")
    print(f"    Stretched duration: {stretched_duration/1000:.1f}s, Original RPS: {original_rps:.2f}")
    print(f"    Extra time: {extra_time_ms/1000:.1f}s, New requests: {num_new}")

    if num_new <= 0:
        return stretched, [], next_unique_id

    # Build session pool from this region
    sessions = build_session_pool(region_entries)
    session_keys = list(sessions.keys())
    session_weights = [sessions[k]["count"] for k in session_keys]
    total_weight = sum(session_weights)
    session_weights = [w / total_weight for w in session_weights]

    # Output length distribution from this region
    output_lengths = [e['output_length'] for e in region_entries]

    # Assign requests to sessions, then cluster
    assigned = rng.choices(session_keys, weights=session_weights, k=num_new)
    session_groups = defaultdict(list)
    for idx, key in enumerate(assigned):
        session_groups[key].append(idx)

    cluster_window_ms = 30_000
    stretched_start = region_start_ts
    stretched_end = region_start_ts + stretched_duration

    new_entries = []
    for session_key, indices in session_groups.items():
        template = sessions[session_key]["template"]
        anchor = np_rng.uniform(stretched_start, stretched_end)

        for idx in indices:
            offset = np_rng.uniform(-cluster_window_ms / 2, cluster_window_ms / 2)
            ts = max(stretched_start, min(stretched_end, anchor + offset))

            if len(template) > 1:
                new_hash_ids = template[:-1] + [next_unique_id]
            else:
                new_hash_ids = [next_unique_id]
            next_unique_id += 1

            new_entries.append({
                "timestamp": float(ts),
                "hash_ids": new_hash_ids,
                "output_length": rng.choice(output_lengths),
            })

    print(f"    Synthesized {len(new_entries)} requests ({len(session_groups)} session clusters)")
    return stretched, new_entries, next_unique_id


def stretch_and_fill(entries, stretch_after_pct, head_stretch_factor, tail_stretch_factor):
    """Stretch both head and tail of the trace, filling with synthesized requests.

    Returns a new list of entries (dicts with 'timestamp', 'hash_ids', 'output_length').
    """
    n = len(entries)
    cut_index = int(n * stretch_after_pct)
    head_raw = entries[:cut_index]
    tail_raw = entries[cut_index:]

    # Find max hash_id for unique ID generation
    max_hid = 0
    for e in entries:
        for hid in e['hash_ids']:
            if isinstance(hid, int) and hid > max_hid:
                max_hid = hid
    next_unique_id = max_hid + 1

    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    print(f"Stretch info (split at {stretch_after_pct*100:.0f}% = entry {cut_index}):")

    # Stretch head
    head_start_ts = head_raw[0]['timestamp'] if head_raw else 0
    head_stretched, head_new, next_unique_id = _synthesize_fill(
        head_raw, entries, head_stretch_factor,
        head_start_ts, "Head", rng, np_rng, next_unique_id
    )

    if not tail_raw:
        all_entries = head_stretched + head_new
        all_entries.sort(key=lambda e: e['timestamp'])
        return all_entries

    # Compute where the tail should start after head stretching
    original_tail_start = tail_raw[0]['timestamp']
    if head_stretched:
        # Shift = how much the head end moved
        original_head_end = head_raw[-1]['timestamp'] if head_raw else 0
        head_time_shift = (original_head_end - head_start_ts) * (head_stretch_factor - 1)
    else:
        head_time_shift = 0

    # Shift tail timestamps by the head expansion before stretching the tail
    tail_shifted = [dict(e) for e in tail_raw]
    for e in tail_shifted:
        e['timestamp'] += head_time_shift

    new_tail_start_ts = original_tail_start + head_time_shift

    # Stretch tail
    tail_stretched, tail_new, next_unique_id = _synthesize_fill(
        tail_shifted, entries, tail_stretch_factor,
        new_tail_start_ts, "Tail", rng, np_rng, next_unique_id
    )

    # Merge all and sort
    all_entries = head_stretched + head_new + tail_stretched + tail_new
    all_entries.sort(key=lambda e: e['timestamp'])
    return all_entries


def main():
    parser = argparse.ArgumentParser(
        description="Custom stretched synthetic workload generator — stretches head and tail independently."
    )
    # Required
    parser.add_argument("--mooncake-trace", required=True, help="Path to Mooncake trace JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    # Workload-affecting (same as realistic)
    parser.add_argument("--rps-scale", type=float, default=1.0,
                        help="Scale RPS by compressing timestamps (2.0 = 2x faster, 0.5 = 2x slower)")
    parser.add_argument("--num-tokens-per-hash-id", type=int, default=100,
                        help="Number of tokens generated per hash_id")
    parser.add_argument("--output-length-scale", type=float, default=1.0,
                        help="Scale output lengths")
    # Operational
    parser.add_argument("--vocab-csv", required=True,
                        help="Path to vocab.csv generated by build_vocab.py")
    parser.add_argument("--duration", type=float, default=None,
                        help="Truncate workload to this many seconds (after rps scaling)")
    # Stretch parameters
    parser.add_argument("--stretch-after-pct", type=float, default=0.7,
                        help="Fraction of trace (0-1) where head/tail split occurs (default: 0.7)")
    parser.add_argument("--head-stretch-factor", type=float, default=1.0,
                        help="How much to stretch the head (low-sharing) region (>=1.0, default: 1.0)")
    parser.add_argument("--tail-stretch-factor", type=float, default=3.0,
                        help="How much to stretch the tail (high-sharing) region (>=1.0, default: 3.0)")

    args = parser.parse_args()

    if not 0.0 <= args.stretch_after_pct <= 1.0:
        parser.error("--stretch-after-pct must be between 0 and 1")
    if args.head_stretch_factor < 1.0:
        parser.error("--head-stretch-factor must be >= 1.0")
    if args.tail_stretch_factor < 1.0:
        parser.error("--tail-stretch-factor must be >= 1.0")

    # Load trace
    print(f"Loading trace from {args.mooncake_trace}")
    entries = load_trace(args.mooncake_trace)
    print(f"Loaded {len(entries)} requests")

    # Stretch and fill
    print(f"\nApplying stretch (split={args.stretch_after_pct}, "
          f"head={args.head_stretch_factor}x, tail={args.tail_stretch_factor}x)")
    stretched_entries = stretch_and_fill(
        entries, args.stretch_after_pct,
        args.head_stretch_factor, args.tail_stretch_factor
    )
    print(f"Total entries after stretch+fill: {len(stretched_entries)}")

    # Load vocab CSV for token-exact prompts
    vocab = load_vocab_csv(args.vocab_csv)

    # Generate workload using the existing function
    print(f"\nGenerating workload (rps_scale={args.rps_scale}, "
          f"num_tokens_per_hash_id={args.num_tokens_per_hash_id}, "
          f"output_length_scale={args.output_length_scale})")
    records = generate_workload(
        stretched_entries, args.rps_scale, args.num_tokens_per_hash_id,
        args.output_length_scale, vocab
    )

    # Truncate to duration if specified
    if args.duration is not None:
        cutoff_ms = int(args.duration * 1000)
        before = len(records)
        records = [r for r in records if r["timestamp"] <= cutoff_ms]
        print(f"Truncated to {args.duration}s: {before} -> {len(records)} requests")

    # Save workload.jsonl
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "workload.jsonl")
    with open(jsonl_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    print(f"Saved {len(records)} records to {jsonl_path}")

    # Save stats.json
    stats = compute_stats(stretched_entries, records, args.rps_scale,
                          args.num_tokens_per_hash_id, args.output_length_scale)
    stats["stretch_config"] = {
        "stretch_after_pct": args.stretch_after_pct,
        "head_stretch_factor": args.head_stretch_factor,
        "tail_stretch_factor": args.tail_stretch_factor,
        "original_entries": len(entries),
        "stretched_entries": len(stretched_entries),
        "synthesized_entries": len(stretched_entries) - len(entries),
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    # Generate plot
    trace_name = os.path.splitext(os.path.basename(args.mooncake_trace))[0]
    stretch_label = f"head{args.head_stretch_factor}x_tail{args.tail_stretch_factor}x_after{args.stretch_after_pct}"
    df = pd.DataFrame({
        'timestamp': [r['timestamp'] for r in records],
        'input_length': [r['requests'][0]['Prompt Length'] for r in records],
        'output_length': [r['requests'][0]['Output Length'] for r in records],
        'hash_ids': [r['requests'][0]['hash_ids'] for r in records],
    })
    save_path = os.path.join(args.output_dir, f"plot_{trace_name}_{stretch_label}.pdf")
    plot_timeseries_analysis(df, f"{trace_name} ({stretch_label})",
                             save_path=save_path,
                             num_tokens_per_hash_id=args.num_tokens_per_hash_id)

    print("Done.")


if __name__ == "__main__":
    main()
