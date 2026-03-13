#!/usr/bin/env python3
"""
Stretch timestamps in high-load windows of a workload to reduce load.

Uses total input tokens per second (tok/s) as the load metric.
Only windows where tok/s exceeds the head-portion average get stretched.
Windows below the threshold are left untouched.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-sec", type=float, default=50,
                        help="Window size in seconds for load computation")
    parser.add_argument("--head-end-sec", type=float, default=1200,
                        help="End of the head portion (seconds). "
                             "Only windows after this are candidates for stretching.")
    parser.add_argument("--target-tokps", type=float, default=0,
                        help="Target input tok/s for high-load windows. "
                             "0 = auto-compute from head average.")
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, "workload.jsonl")
    window_ms = args.window_sec * 1000
    head_end_ms = args.head_end_sec * 1000

    # --- Pass 1: read all entries, compute per-window stats ---
    entries = []
    with open(input_path) as f:
        for line in f:
            entries.append(json.loads(line))

    # Bucket by window: request count and total input tokens
    window_counts = defaultdict(int)
    window_input_tokens = defaultdict(int)
    for e in entries:
        ts = e["timestamp"]
        bucket = int(ts / window_ms)
        for r in e["requests"]:
            window_counts[bucket] += 1
            window_input_tokens[bucket] += r["Prompt Length"]

    # Compute head average tok/s
    all_buckets = sorted(window_counts.keys())
    head_buckets = [b for b in all_buckets if b * window_ms < head_end_ms]
    head_total_tokens = sum(window_input_tokens[b] for b in head_buckets)
    head_duration = len(head_buckets) * args.window_sec

    if args.target_tokps > 0:
        target_tokps = args.target_tokps
    else:
        target_tokps = head_total_tokens / head_duration if head_duration > 0 else 10000

    head_avg_rps = sum(window_counts[b] for b in head_buckets) / head_duration

    print(f"Head avg input tok/s: {target_tokps:.0f}")
    print(f"Head avg RPS: {head_avg_rps:.2f}")
    print(f"Window size: {args.window_sec}s")
    print(f"Head ends at: {args.head_end_sec}s")
    print()

    # --- Compute stretch factor per window based on tok/s ---
    stretch_factors = {}
    for b in all_buckets:
        tokps = window_input_tokens[b] / args.window_sec
        rps = window_counts[b] / args.window_sec
        if b * window_ms >= head_end_ms and tokps > target_tokps:
            stretch_factors[b] = tokps / target_tokps
        else:
            stretch_factors[b] = 1.0

    # Print all windows
    for b in all_buckets:
        tokps = window_input_tokens[b] / args.window_sec
        rps = window_counts[b] / args.window_sec
        sf = stretch_factors[b]
        t0 = b * args.window_sec
        t1 = t0 + args.window_sec
        marker = f" ** stretch {sf:.2f}x" if sf > 1.0 else ""
        print(f"  {t0:>6.0f}-{t1:>6.0f}s: {rps:.1f} rps, {tokps:>8.0f} tok/s{marker}")

    # --- Pass 2: remap timestamps ---
    window_new_start = {}
    cumulative = 0.0
    for b in all_buckets:
        window_new_start[b] = cumulative
        cumulative += window_ms * stretch_factors[b]

    new_entries = []
    for e in entries:
        ts = e["timestamp"]
        b = int(ts / window_ms)
        old_window_start = b * window_ms
        offset_in_window = ts - old_window_start
        new_ts = window_new_start[b] + offset_in_window * stretch_factors[b]
        new_e = dict(e)
        new_e["timestamp"] = int(round(new_ts))
        new_entries.append(new_e)

    new_duration = new_entries[-1]["timestamp"] / 1000
    old_duration = entries[-1]["timestamp"] / 1000
    print(f"\nOriginal duration: {old_duration:.1f}s")
    print(f"New duration:      {new_duration:.1f}s")
    print(f"Total requests:    {len(new_entries)}")

    # --- Write output ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "workload.jsonl")
    with open(output_path, "w") as f:
        for e in new_entries:
            f.write(json.dumps(e) + "\n")

    # Copy other files from input dir (except workload.jsonl and PDFs)
    for fname in os.listdir(args.input_dir):
        if fname == "workload.jsonl" or fname.endswith(".pdf"):
            continue
        src = os.path.join(args.input_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    print(f"\nWritten to: {output_path}")

    # --- Generate plot ---
    import pandas as pd
    from plot import plot_timeseries_analysis

    df = pd.DataFrame({
        'timestamp': [e['timestamp'] for e in new_entries],
        'input_length': [e['requests'][0]['Prompt Length'] for e in new_entries],
        'output_length': [e['requests'][0]['Output Length'] for e in new_entries],
        'hash_ids': [e['requests'][0]['hash_ids'] for e in new_entries],
    })
    plot_name = os.path.basename(args.output_dir)
    save_path = os.path.join(args.output_dir, f"plot_{plot_name}.pdf")
    plot_timeseries_analysis(df, plot_name, save_path=save_path, num_tokens_per_hash_id=100)


if __name__ == "__main__":
    main()
