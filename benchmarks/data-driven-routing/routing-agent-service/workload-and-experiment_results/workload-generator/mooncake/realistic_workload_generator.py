#!/usr/bin/env python3
"""
Simple Mooncake Workload Generator

Faithfully reproduces the original trace with only 3 workload-affecting parameters:
  --rps-scale: Scale RPS by compressing/expanding timestamps
  --num-tokens-per-hash-id: Tokens per hash_id block
  --output-length-scale: Scale output lengths
"""

import json
import os
import csv
import argparse
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from plot import plot_timeseries_analysis


def load_vocab_csv(path):
    """Load vocab CSV (token_id,text) into a dict {token_id: text}.

    Each entry represents a single safe BPE token that round-trips
    through encode/decode, so concatenating N entries = exactly N tokens.
    """
    vocab = {}
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ['token_id', 'text'], f"Unexpected CSV header: {header}"
        for row in reader:
            tid = int(row[0])
            text = row[1]
            vocab[tid] = text
    print(f"Loaded {len(vocab)} safe tokens from {path}")
    return vocab


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


def generate_workload(entries, rps_scale, num_tokens_per_hash_id, output_length_scale, vocab):
    """Transform trace entries into workload records with token-exact prompts.

    Each hash_id deterministically selects num_tokens_per_hash_id safe token IDs
    via a seeded RNG. Since each vocab entry = exactly 1 BPE token, the reported
    Prompt Length (len(hash_ids) * num_tokens_per_hash_id) is token-exact.
    """
    safe_ids = list(vocab.keys())
    block_cache = {}  # hid -> block_text (cache for prefix sharing perf)
    records = []

    for e in entries:
        ts_ms = int(e['timestamp'] / rps_scale)
        hash_ids = e['hash_ids']

        # Build prompt: each hash_id -> deterministic block of N safe tokens
        blocks = []
        for hid in hash_ids:
            if hid not in block_cache:
                rng = random.Random(hid)
                chosen = [rng.choice(safe_ids) for _ in range(num_tokens_per_hash_id)]
                block_cache[hid] = ''.join(vocab[tid] for tid in chosen)
            blocks.append(block_cache[hid])
        prompt = ''.join(blocks)

        output_length = max(1, int(e['output_length'] * output_length_scale))
        prefix_group = "_".join(str(h) for h in hash_ids[:3])

        records.append({
            "timestamp": ts_ms,
            "requests": [{
                "Prompt Length": len(hash_ids) * num_tokens_per_hash_id,
                "Output Length": output_length,
                "prefix_group": prefix_group,
                "hash_ids": hash_ids,
                "prompt": prompt,
            }]
        })
    return records


def compute_stats(entries, records, rps_scale, num_tokens_per_hash_id, output_length_scale):
    """Compute summary statistics."""
    prompt_lengths = [r["requests"][0]["Prompt Length"] for r in records]
    output_lengths = [r["requests"][0]["Output Length"] for r in records]
    timestamps_sec = [r["timestamp"] / 1000.0 for r in records]
    duration_sec = max(timestamps_sec) - min(timestamps_sec) if len(timestamps_sec) > 1 else 0

    # RPS per second
    rps_counts = defaultdict(int)
    for ts in timestamps_sec:
        rps_counts[int(ts)] += 1
    rps_values = list(rps_counts.values()) if rps_counts else [0]

    orig_output_lengths = [e['output_length'] for e in entries]
    orig_timestamps_sec = [e['timestamp'] / 1000.0 for e in entries]
    orig_duration = max(orig_timestamps_sec) - min(orig_timestamps_sec) if len(orig_timestamps_sec) > 1 else 0

    return {
        "config": {
            "rps_scale": rps_scale,
            "num_tokens_per_hash_id": num_tokens_per_hash_id,
            "output_length_scale": output_length_scale,
        },
        "original_trace": {
            "num_requests": len(entries),
            "duration_seconds": round(orig_duration, 2),
            "output_length_min": int(min(orig_output_lengths)),
            "output_length_max": int(max(orig_output_lengths)),
            "output_length_mean": round(float(np.mean(orig_output_lengths)), 2),
        },
        "generated_workload": {
            "num_requests": len(records),
            "duration_seconds": round(duration_sec, 2),
            "prompt_length_min": int(min(prompt_lengths)),
            "prompt_length_max": int(max(prompt_lengths)),
            "prompt_length_mean": round(float(np.mean(prompt_lengths)), 2),
            "output_length_min": int(min(output_lengths)),
            "output_length_max": int(max(output_lengths)),
            "output_length_mean": round(float(np.mean(output_lengths)), 2),
            "rps_min": int(min(rps_values)),
            "rps_max": int(max(rps_values)),
            "rps_mean": round(float(np.mean(rps_values)), 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simple Mooncake workload generator — faithfully reproduces trace with minimal knobs."
    )
    # Required
    parser.add_argument("--mooncake-trace", required=True, help="Path to Mooncake trace JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    # Workload-affecting
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

    args = parser.parse_args()

    # Load trace
    print(f"Loading trace from {args.mooncake_trace}")
    entries = load_trace(args.mooncake_trace)
    print(f"Loaded {len(entries)} requests")

    # Load vocab CSV for token-exact prompts
    vocab = load_vocab_csv(args.vocab_csv)

    # Generate workload
    print(f"Generating workload (rps_scale={args.rps_scale}, "
          f"num_tokens_per_hash_id={args.num_tokens_per_hash_id}, "
          f"output_length_scale={args.output_length_scale})")
    records = generate_workload(
        entries, args.rps_scale, args.num_tokens_per_hash_id,
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
    stats = compute_stats(entries, records, args.rps_scale, args.num_tokens_per_hash_id, args.output_length_scale)
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    # Generate plot
    trace_name = os.path.splitext(os.path.basename(args.mooncake_trace))[0]
    df = pd.DataFrame({
        'timestamp': [r['timestamp'] for r in records],
        'input_length': [r['requests'][0]['Prompt Length'] for r in records],
        'output_length': [r['requests'][0]['Output Length'] for r in records],
        'hash_ids': [r['requests'][0]['hash_ids'] for r in records],
    })
    save_path = os.path.join(args.output_dir, f"plot_{trace_name}.pdf")
    plot_timeseries_analysis(df, trace_name, save_path=save_path, num_tokens_per_hash_id=args.num_tokens_per_hash_id)

    print("Done.")


if __name__ == "__main__":
    main()
