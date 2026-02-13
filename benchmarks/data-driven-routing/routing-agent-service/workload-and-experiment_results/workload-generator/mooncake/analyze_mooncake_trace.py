#!/usr/bin/env python3
"""
Analyze Mooncake trace hash_id patterns and prefix sharing.

Outputs a JSON summary plus a human-readable console summary.
"""

import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np


def _safe_percentiles(values, percentiles):
    if not values:
        return {f"p{p}": 0 for p in percentiles}
    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def _basic_stats(values):
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "std": 0, "p50": 0, "p90": 0, "p99": 0}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def _format_prefix(prefix_tuple):
    if not prefix_tuple:
        return "none"
    return "_".join(str(x) for x in prefix_tuple)


def _longest_common_prefix_length(hash_id_lists):
    if not hash_id_lists:
        return 0
    min_len = min(len(ids) for ids in hash_id_lists)
    if min_len == 0:
        return 0
    common_len = 0
    for i in range(min_len):
        value = hash_id_lists[0][i]
        if all(ids[i] == value for ids in hash_id_lists[1:]):
            common_len = i + 1
        else:
            break
    return common_len


def analyze_trace(trace_path, num_tokens_per_hash_id=500, top_k=20, max_prefix_depth=3):
    requests = []
    with open(trace_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            requests.append(json.loads(line))

    total_requests = len(requests)
    hash_lengths = []
    input_lengths = []
    output_lengths = []
    tokens_per_hash = []
    pattern_counts = Counter()

    prefix_counts = {d: Counter() for d in range(1, max_prefix_depth + 1)}

    for req in requests:
        hash_ids = req.get("hash_ids", [])
        hash_lengths.append(len(hash_ids))
        input_lengths.append(req.get("input_length", 0))
        output_lengths.append(req.get("output_length", 0))

        if hash_ids:
            tokens_per_hash.append(req.get("input_length", 0) / len(hash_ids))

        pattern = tuple(hash_ids)
        pattern_counts[pattern] += 1

        for depth in range(1, max_prefix_depth + 1):
            prefix = tuple(hash_ids[:depth])
            prefix_counts[depth][_format_prefix(prefix)] += 1

    # Sharing ratio based on generator logic (group by first hash_id)
    prefix_groups = defaultdict(list)
    for req in requests:
        hash_ids = req.get("hash_ids", [])
        first_hash = hash_ids[0] if hash_ids else "none"
        prefix_groups[first_hash].append(hash_ids)

    total_tokens = sum(len(req.get("hash_ids", [])) * num_tokens_per_hash_id for req in requests)
    tokens_with_sharing = 0
    for group_hashes in prefix_groups.values():
        if not group_hashes:
            continue
        common_prefix_len = _longest_common_prefix_length(group_hashes)
        shared_tokens = common_prefix_len * num_tokens_per_hash_id
        unique_tokens = sum(max(0, len(ids) * num_tokens_per_hash_id - shared_tokens) for ids in group_hashes)
        tokens_with_sharing += shared_tokens + unique_tokens

    sharing_ratio = 0.0
    if total_tokens > 0:
        sharing_ratio = (total_tokens - tokens_with_sharing) / total_tokens
        sharing_ratio = max(0.0, sharing_ratio)

    prefix_group_stats = {}
    for depth, counts in prefix_counts.items():
        group_sizes = Counter(counts.values())
        prefix_group_stats[f"depth_{depth}"] = {
            "total_groups": len(counts),
            "top_prefixes": dict(counts.most_common(top_k)),
            "group_size_distribution": dict(group_sizes.most_common(top_k)),
        }

    analysis = {
        "trace_file": trace_path,
        "total_requests": total_requests,
        "num_tokens_per_hash_id": num_tokens_per_hash_id,
        "hash_ids_length": _basic_stats(hash_lengths),
        "input_length": _basic_stats(input_lengths),
        "output_length": _basic_stats(output_lengths),
        "approx_tokens_per_hash_id": _basic_stats(tokens_per_hash),
        "unique_hash_patterns": len(pattern_counts),
        "most_common_patterns": {
            str(pattern): count for pattern, count in pattern_counts.most_common(top_k)
        },
        "prefix_group_stats": prefix_group_stats,
        "sharing_ratio": sharing_ratio,
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze Mooncake hash_id trace patterns")
    parser.add_argument(
        "--trace-file",
        required=True,
        help="Path to Mooncake trace JSONL file",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output JSON file for analysis summary",
    )
    parser.add_argument(
        "--num-tokens-per-hash-id",
        type=int,
        default=500,
        help="Tokens represented by each hash_id (default: 500)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k patterns/prefixes to include in output",
    )
    parser.add_argument(
        "--max-prefix-depth",
        type=int,
        default=3,
        help="Max prefix depth to analyze for prefix groups",
    )
    args = parser.parse_args()

    analysis = analyze_trace(
        trace_path=args.trace_file,
        num_tokens_per_hash_id=args.num_tokens_per_hash_id,
        top_k=args.top_k,
        max_prefix_depth=args.max_prefix_depth,
    )

    with open(args.output_file, "w") as f:
        json.dump(analysis, f, indent=2)

    print("Mooncake Trace Analysis")
    print("=" * 80)
    print(f"Trace file: {analysis['trace_file']}")
    print(f"Total requests: {analysis['total_requests']}")
    print(f"Unique hash patterns: {analysis['unique_hash_patterns']}")
    print(f"Sharing ratio (by first hash_id): {analysis['sharing_ratio']:.4f}")
    print("-" * 80)
    print(f"Hash IDs per request: {analysis['hash_ids_length']}")
    print(f"Input length (tokens): {analysis['input_length']}")
    print(f"Output length (tokens): {analysis['output_length']}")
    print(f"Approx tokens per hash_id: {analysis['approx_tokens_per_hash_id']}")
    print("=" * 80)
    print(f"Saved analysis to: {args.output_file}")


if __name__ == "__main__":
    main()


















