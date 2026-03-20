#!/usr/bin/env python3
"""
Generate an extended workload trace that preserves the statistical properties
of an existing workload.jsonl.

Usage:
    python3 extend_workload.py <input_workload.jsonl> <output_workload.jsonl> [--multiplier 4] [--seed 42]

The generator:
1. Analyzes the input workload to extract all relevant distributions
2. Generates new prefix groups with fresh hash_ids over the extended timeline
3. Preserves: group size distribution, intra-group IAT (conditioned on group size),
   conversation growth pattern, output token distribution, hash_id hierarchy,
   global inter-arrival shape, and co-arrival mixing.
"""

import json
import argparse
import os
import sys
import math
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Analyze the original workload
# ---------------------------------------------------------------------------

def analyze_workload(entries: List[Dict]) -> Dict[str, Any]:
    """Extract all distributions from the original workload."""

    stats = {}

    # --- Basic ---
    timestamps = sorted(set(e["timestamp_ms"] for e in entries))
    stats["duration_ms"] = max(timestamps) - min(timestamps)
    stats["num_requests"] = len(entries)
    stats["tokens_per_block"] = entries[0]["input_tokens"] / len(entries[0]["hash_ids"])

    # --- Prefix groups ---
    pg_members = defaultdict(list)
    for e in entries:
        pg_members[e["prefix_group"]].append(e)
    for pg in pg_members:
        pg_members[pg].sort(key=lambda e: e["timestamp_ms"])

    stats["num_groups"] = len(pg_members)

    # Group size distribution (empirical)
    pg_sizes = [len(v) for v in pg_members.values()]
    stats["group_size_dist"] = pg_sizes

    # --- New group creation rate per window ---
    first_seen = {}
    for e in entries:
        pg = e["prefix_group"]
        if pg not in first_seen:
            first_seen[pg] = e["timestamp_ms"]

    window_ms = 60_000
    creation_counts = defaultdict(int)
    for pg, ts in first_seen.items():
        creation_counts[int(ts // window_ms)] += 1
    num_windows = int(stats["duration_ms"] / window_ms) + 1
    stats["creation_rate_per_window"] = [creation_counts.get(w, 0) for w in range(num_windows)]
    stats["window_ms"] = window_ms

    # --- Intra-group IAT conditioned on group size ---
    # Bucket: small (2-3), medium (4-10), large (11+)
    iat_by_size = {"small": [], "medium": [], "large": []}
    for pg, members in pg_members.items():
        if len(members) < 2:
            continue
        size_bucket = _size_bucket(len(members))
        for i in range(len(members) - 1):
            iat = members[i + 1]["timestamp_ms"] - members[i]["timestamp_ms"]
            iat_by_size[size_bucket].append(max(1, iat))
    # Fallback: if a bucket is empty, use all IATs
    all_iats = []
    for v in iat_by_size.values():
        all_iats.extend(v)
    for k in iat_by_size:
        if not iat_by_size[k]:
            iat_by_size[k] = all_iats
    stats["iat_by_size"] = iat_by_size

    # --- Position-in-group -> block count changes ---
    # Record block DELTAS (can be negative for fluctuating patterns like toolagent)
    block_deltas = defaultdict(list)  # position -> list of delta block counts
    first_block_counts = defaultdict(list)  # size_bucket -> first request block count
    # Also record absolute block counts per position for persistent groups
    block_counts_by_pos = defaultdict(list)
    for pg, members in pg_members.items():
        size_bucket = _size_bucket(len(members))
        first_block_counts[size_bucket].append(len(members[0]["hash_ids"]))
        prev_blocks = len(members[0]["hash_ids"])
        for pos in range(1, len(members)):
            cur_blocks = len(members[pos]["hash_ids"])
            delta = cur_blocks - prev_blocks
            block_deltas[pos].append(delta)
            block_counts_by_pos[pos].append(cur_blocks)
            prev_blocks = cur_blocks
    all_deltas = []
    for v in block_deltas.values():
        all_deltas.extend(v)
    stats["block_deltas_by_pos"] = dict(block_deltas)
    stats["all_deltas"] = all_deltas if all_deltas else [0]
    stats["block_counts_by_pos"] = dict(block_counts_by_pos)
    stats["first_block_counts_by_size"] = dict(first_block_counts)
    all_first_blocks = []
    for v in first_block_counts.values():
        all_first_blocks.extend(v)
    stats["all_first_blocks"] = all_first_blocks

    # --- Persistent groups: groups that span most of the trace ---
    # These need special handling — they should be scaled to span the extended trace
    persistent_threshold = stats["duration_ms"] * 0.5  # groups spanning >50% of trace
    persistent_groups = []
    normal_group_sizes = []
    for pg, members in pg_members.items():
        ms = sorted(members, key=lambda e: e["timestamp_ms"])
        span = ms[-1]["timestamp_ms"] - ms[0]["timestamp_ms"]
        if span > persistent_threshold and len(members) > 20:
            # Compute the DEEP shared prefix: minimum consecutive shared blocks
            # across all adjacent pairs in the group
            deep_shared = len(ms[0]["hash_ids"])
            for i in range(len(ms) - 1):
                shared = 0
                for a, b in zip(ms[i]["hash_ids"], ms[i+1]["hash_ids"]):
                    if a == b:
                        shared += 1
                    else:
                        break
                deep_shared = min(deep_shared, shared)
            # The deep shared prefix hash_ids (these are constant across all requests)
            deep_prefix_hids = ms[0]["hash_ids"][:deep_shared]

            # Extension beyond deep prefix: how many extra blocks each request has
            extension_counts = [len(m["hash_ids"]) - deep_shared for m in ms]

            persistent_groups.append({
                "size": len(members),
                "span_ms": span,
                "iat_dist": [ms[i+1]["timestamp_ms"] - ms[i]["timestamp_ms"]
                             for i in range(len(ms)-1)],
                "block_counts": [len(m["hash_ids"]) for m in ms],
                "output_tokens": [m["output_tokens"] for m in ms],
                "input_tokens": [m["input_tokens"] for m in ms],
                "deep_shared_len": deep_shared,
                "extension_counts": extension_counts,
            })
        else:
            normal_group_sizes.append(len(members))
    stats["persistent_groups"] = persistent_groups
    stats["normal_group_sizes"] = normal_group_sizes if normal_group_sizes else pg_sizes
    print(f"  Persistent groups (span > 50% of trace, size > 20): {len(persistent_groups)}")
    for i, pg_info in enumerate(persistent_groups):
        print(f"    [{i}] size={pg_info['size']}, deep_shared={pg_info['deep_shared_len']} blocks, "
              f"extension range=[{min(pg_info['extension_counts'])}, {max(pg_info['extension_counts'])}]")

    # --- Output tokens conditioned on input tokens ---
    output_by_input_bin = defaultdict(list)
    for e in entries:
        bin_key = _input_bin(e["input_tokens"])
        output_by_input_bin[bin_key].append(e["output_tokens"])
    stats["output_by_input_bin"] = dict(output_by_input_bin)
    stats["all_outputs"] = [e["output_tokens"] for e in entries]

    # --- Within-group output token correlation ---
    # For position > 0, compute delta from group mean
    within_group_output_offsets = []
    for pg, members in pg_members.items():
        if len(members) < 2:
            continue
        group_mean_out = np.mean([m["output_tokens"] for m in members])
        for m in members:
            within_group_output_offsets.append(m["output_tokens"] - group_mean_out)
    stats["within_group_output_std"] = float(np.std(within_group_output_offsets)) if within_group_output_offsets else 60.0

    # --- Hash ID hierarchy: fraction of groups that branch from existing ---
    depth2_groups = defaultdict(list)
    for pg in pg_members:
        parts = [int(x) for x in pg.split("_")]
        if len(parts) >= 2:
            depth2_groups[tuple(parts[:2])].append(pg)
    num_branching = sum(1 for v in depth2_groups.values() if len(v) > 1)
    stats["branch_fraction"] = num_branching / max(len(depth2_groups), 1)

    # --- Global inter-arrival time distribution ---
    global_times = sorted(e["timestamp_ms"] for e in entries)
    global_iats = [global_times[i + 1] - global_times[i] for i in range(len(global_times) - 1)]
    stats["global_iat_dist"] = global_iats

    # --- Prompt text: collect word pool ---
    word_pool = set()
    for e in entries[:200]:  # sample from first 200 for efficiency
        if isinstance(e.get("prompt"), str):
            word_pool.update(e["prompt"].split()[:50])
    if not word_pool:
        word_pool = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                     "golf", "hotel", "india", "juliet", "kilo", "lima"}
    stats["word_pool"] = list(word_pool)

    print(f"  Duration: {stats['duration_ms']/1000:.0f}s, Requests: {stats['num_requests']}, "
          f"Groups: {stats['num_groups']}, Tokens/block: {stats['tokens_per_block']:.0f}")
    print(f"  Group sizes: mean={np.mean(pg_sizes):.1f}, median={np.median(pg_sizes):.0f}, max={max(pg_sizes)}")
    print(f"  Creation rate/window: mean={np.mean(stats['creation_rate_per_window']):.0f}")
    print(f"  Branch fraction: {stats['branch_fraction']:.3f}")

    return stats


def _size_bucket(size: int) -> str:
    if size <= 3:
        return "small"
    elif size <= 10:
        return "medium"
    else:
        return "large"


def _input_bin(input_tokens: int) -> int:
    """Bin input tokens into 500-token buckets."""
    return (input_tokens // 500) * 500


# ---------------------------------------------------------------------------
# Step 2: Generate extended workload
# ---------------------------------------------------------------------------

def generate_workload(stats: Dict[str, Any], multiplier: float, rng: np.random.RandomState) -> List[Dict]:
    """Generate an extended workload preserving all distributions."""

    tokens_per_block = int(stats["tokens_per_block"])
    target_duration_ms = int(stats["duration_ms"] * multiplier)
    target_num_windows = int(target_duration_ms / stats["window_ms"]) + 1

    # Global hash_id counter (ensures uniqueness)
    next_hash_id = 1  # 0 is reserved as root

    # --- Step 2a: Generate persistent groups first ---
    # These span the entire trace and need to be scaled to the extended duration
    all_requests = []
    persistent_groups = stats["persistent_groups"]

    if persistent_groups:
        print(f"  Generating {len(persistent_groups)} persistent groups...")
        for pg_info in persistent_groups:
            # Build a deep shared prefix with fresh unique hash_ids
            deep_len = pg_info["deep_shared_len"]
            deep_prefix = []
            for _ in range(deep_len):
                deep_prefix.append(next_hash_id)
                next_hash_id += 1

            # The prefix_group name uses the first 3 hash_ids (matches original convention)
            prefix_group = "_".join(str(h) for h in deep_prefix[:3])

            # Scale the number of requests by the multiplier
            orig_size = pg_info["size"]
            target_size = int(orig_size * multiplier)
            iat_dist = pg_info["iat_dist"]
            ext_dist = pg_info["extension_counts"]
            output_dist = pg_info["output_tokens"]

            current_time_ms = rng.randint(0, 1000)  # start near beginning

            for pos in range(target_size):
                iat = rng.choice(iat_dist)
                if pos > 0:
                    current_time_ms += iat
                if current_time_ms > target_duration_ms:
                    break

                # Sample extension beyond deep prefix
                ext_count = max(0, int(rng.choice(ext_dist)))

                # Build hash_ids: deep shared prefix (constant) + unique extension
                hash_ids = list(deep_prefix)
                for _ in range(ext_count):
                    hash_ids.append(next_hash_id)
                    next_hash_id += 1

                input_tokens = len(hash_ids) * tokens_per_block
                output_tokens = max(1, int(rng.choice(output_dist)))

                all_requests.append({
                    "timestamp_ms": int(current_time_ms),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "hash_ids": list(hash_ids),
                    "prefix_group": prefix_group,
                })

        print(f"  Persistent groups generated {len(all_requests)} requests")

    # --- Step 2b: Generate the global arrival timeline first ---
    # This preserves the bursty IAT structure. Then assign groups to these slots.
    print(f"  Generating global arrival timeline...")

    orig_duration_ms = stats["duration_ms"]
    orig_iats = stats["global_iat_dist"]

    # Build the timeline via block resampling of the original IAT sequence.
    # Randomly sampling contiguous blocks (not cycling sequentially) preserves
    # local burst structure without the periodicity-smoothing that sequential
    # cycling causes when the trace is repeated (law-of-large-numbers averaging).
    BLOCK_SIZE = max(10, len(orig_iats) // 20)  # ~5% of original trace per block
    arrival_times = [0]
    while arrival_times[-1] < target_duration_ms:
        block_start = rng.randint(0, max(1, len(orig_iats) - BLOCK_SIZE))
        for iat in orig_iats[block_start: block_start + BLOCK_SIZE]:
            arrival_times.append(arrival_times[-1] + max(0, iat))
            if arrival_times[-1] >= target_duration_ms:
                break
    # Trim to target duration
    arrival_times = [t for t in arrival_times if t <= target_duration_ms]
    print(f"  Timeline: {len(arrival_times)} arrival slots over {arrival_times[-1]/1000:.0f}s "
          f"(block_size={BLOCK_SIZE})")

    # --- Step 2c: Determine which arrival slots start new groups ---
    # In the original, fraction of requests that are "first in group" = num_groups / num_requests
    # Singleton groups are always first-in-group. Multi-request groups have 1 first + N-1 follow-ups.
    orig_rates = stats["creation_rate_per_window"]
    window_ms = stats["window_ms"]

    # For each arrival slot, decide: is this a new group's first request, or a follow-up?
    # We mark slots as "new group" based on block-resampled creation rates per window.
    new_group_slots = []  # indices into arrival_times
    followup_slots = []   # indices for follow-up requests (will be replaced by group-internal timing)

    # Block-resample the creation rate sequence to preserve its full variability
    # (mean, std, spikes, lulls) without AR(1) mean-reversion smoothing.
    RATE_BLOCK_SIZE = max(2, len(orig_rates) // 10)
    extended_rates = []
    while len(extended_rates) < target_num_windows:
        block_start = rng.randint(0, max(1, len(orig_rates) - RATE_BLOCK_SIZE))
        extended_rates.extend(orig_rates[block_start: block_start + RATE_BLOCK_SIZE])
    extended_rates = extended_rates[:target_num_windows]

    for window_idx in range(target_num_windows):
        rate = int(extended_rates[window_idx])
        w_start = window_idx * window_ms
        w_end = w_start + window_ms
        # Find arrival slots in this window
        window_slot_indices = [i for i, t in enumerate(arrival_times) if w_start <= t < w_end]
        # Assign the first `rate` slots as new group starts
        # Shuffle so new groups aren't always at the window start
        rng.shuffle(window_slot_indices)
        for j, slot_idx in enumerate(window_slot_indices):
            if j < rate:
                new_group_slots.append(slot_idx)
            else:
                followup_slots.append(slot_idx)

    new_group_slots.sort()
    print(f"  New group slots: {len(new_group_slots)}, follow-up slots: {len(followup_slots)}")

    # --- Step 2d: Create groups and generate their requests ---
    print(f"  Generating prefix groups and requests...")

    groups = []
    active_depth2_prefixes = []

    for slot_idx in new_group_slots:
        first_time_ms = arrival_times[slot_idx]

        group_size = rng.choice(stats["normal_group_sizes"])

        branches_from = None
        if active_depth2_prefixes and rng.random() < stats["branch_fraction"]:
            branches_from = active_depth2_prefixes[rng.randint(0, len(active_depth2_prefixes))]

        if branches_from is not None:
            shared_prefix = [0, branches_from[1], next_hash_id]
            next_hash_id += 1
        else:
            shared_prefix = [0, next_hash_id, next_hash_id + 1]
            next_hash_id += 2

        active_depth2_prefixes.append((shared_prefix[0], shared_prefix[1]))
        if len(active_depth2_prefixes) > 10000:
            active_depth2_prefixes = active_depth2_prefixes[-5000:]

        size_bucket = _size_bucket(group_size)
        first_blocks_pool = stats["first_block_counts_by_size"].get(
            size_bucket, stats["all_first_blocks"])
        first_block_count = rng.choice(first_blocks_pool)
        first_block_count = max(first_block_count, len(shared_prefix))

        iat_pool = stats["iat_by_size"].get(size_bucket, stats["iat_by_size"]["small"])

        # Generate all requests for this group
        current_blocks = first_block_count
        current_time_ms = first_time_ms

        hash_ids = list(shared_prefix)
        while len(hash_ids) < current_blocks:
            hash_ids.append(next_hash_id)
            next_hash_id += 1

        for pos in range(group_size):
            if pos > 0:
                iat = rng.choice(iat_pool)
                current_time_ms += iat

                delta_pool = stats["block_deltas_by_pos"].get(pos, None)
                if delta_pool is None or len(delta_pool) == 0:
                    delta_pool = stats["all_deltas"]
                delta = rng.choice(delta_pool)

                if delta > 0:
                    for _ in range(delta):
                        hash_ids.append(next_hash_id)
                        next_hash_id += 1
                elif delta < 0:
                    new_len = max(len(shared_prefix), len(hash_ids) + delta)
                    hash_ids = hash_ids[:new_len]
                    # Replace last non-shared block with a fresh ID to avoid
                    # matching a previously-cached trie path (prevents spurious
                    # 100% KV hits from truncation back to an earlier state).
                    if len(hash_ids) > len(shared_prefix):
                        hash_ids[-1] = next_hash_id
                        next_hash_id += 1
                else:
                    # delta == 0: same block count but content changed (new
                    # conversation turn). Refresh the last block so hash_ids
                    # differ from the previous request, matching original
                    # workload behavior where delta=0 rarely causes a full hit.
                    if len(hash_ids) > len(shared_prefix):
                        hash_ids[-1] = next_hash_id
                        next_hash_id += 1

                current_blocks = len(hash_ids)

            if current_time_ms > target_duration_ms:
                break

            input_tokens = current_blocks * tokens_per_block
            output_tokens = max(1, int(_sample_output(input_tokens, stats, rng)))
            prefix_group = "_".join(str(h) for h in shared_prefix)

            all_requests.append({
                "timestamp_ms": int(current_time_ms),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "hash_ids": list(hash_ids),
                "prefix_group": prefix_group,
            })

    all_requests.sort(key=lambda r: r["timestamp_ms"])
    print(f"  Generated {len(all_requests)} total requests over "
          f"{(all_requests[-1]['timestamp_ms'] - all_requests[0]['timestamp_ms'])/1000:.0f}s")

    return all_requests


def _sample_output(input_tokens: int, stats: Dict, rng: np.random.RandomState) -> float:
    """Sample an output token value conditioned on input token bin."""
    bin_key = _input_bin(input_tokens)
    pool = stats["output_by_input_bin"].get(bin_key, None)
    if pool is None or len(pool) == 0:
        pool = stats["all_outputs"]
    return float(rng.choice(pool))


def _get_conditional_output_mean(input_tokens: int, stats: Dict) -> float:
    """Get mean output tokens for a given input token bin."""
    bin_key = _input_bin(input_tokens)
    pool = stats["output_by_input_bin"].get(bin_key, None)
    if pool is None or len(pool) == 0:
        pool = stats["all_outputs"]
    return float(np.mean(pool))


def _time_warp(requests: List[Dict], stats: Dict, rng: np.random.RandomState) -> List[Dict]:
    """
    Adjust global inter-arrival times to match the empirical IAT distribution.

    Strategy: replace the IATs between consecutive requests with samples from
    the empirical IAT distribution, while preserving the relative ordering
    of requests and the intra-group temporal structure (requests from the same
    group maintain their order and minimum spacing).
    """
    if len(requests) < 2:
        return requests

    orig_iats = stats["global_iat_dist"]
    if not orig_iats:
        return requests

    # Sample new IATs from empirical distribution
    n = len(requests) - 1
    new_iats = rng.choice(orig_iats, size=n, replace=True)

    # Rebuild timestamps
    base_time = 0
    requests[0]["timestamp_ms"] = base_time
    for i in range(1, len(requests)):
        base_time += max(0, int(new_iats[i - 1]))
        requests[i]["timestamp_ms"] = base_time

    return requests


# ---------------------------------------------------------------------------
# Step 3: Generate prompt text
# ---------------------------------------------------------------------------

def generate_prompt(input_tokens: int, tokens_per_block: int, word_pool: List[str],
                    rng: np.random.RandomState) -> str:
    """Generate random-word prompt text with approximately input_tokens tokens.
    Approximation: 1 word ≈ 1 token * (tokens_per_block / words_per_block).
    The original uses ~50 tokens per block with random words, roughly 1.33 tokens/word.
    """
    target_words = int(input_tokens)
    words = rng.choice(word_pool, size=max(1, target_words), replace=True)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Step 4: Write output
# ---------------------------------------------------------------------------

def write_workload(requests: List[Dict], output_path: str, word_pool: List[str],
                   tokens_per_block: int, rng: np.random.RandomState):
    """Write requests to workload.jsonl format."""
    print(f"  Writing {len(requests)} requests to {output_path}...")

    with open(output_path, "w") as f:
        for req in requests:
            prompt_text = generate_prompt(req["input_tokens"], tokens_per_block,
                                          word_pool, rng)
            entry = {
                "timestamp": int(req["timestamp_ms"]),
                "requests": [{
                    "Prompt Length": int(req["input_tokens"]),
                    "Output Length": int(req["output_tokens"]),
                    "prefix_group": req["prefix_group"],
                    "hash_ids": [int(h) for h in req["hash_ids"]],
                    "prompt": prompt_text,
                }]
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  Done. Output: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scale_rps(requests: List[Dict], target_rps: float) -> List[Dict]:
    """
    Scale timestamps uniformly so average RPS equals target_rps.
    The relative burstiness pattern is preserved (shape unchanged);
    only the vertical scale (RPS level) changes.

    Strategy: stretch or compress time around t0 so that
        len(requests) / new_duration_s == target_rps
    i.e., new_duration_ms = len(requests) / target_rps * 1000
    """
    if not requests:
        return requests
    t0 = requests[0]["timestamp_ms"]
    t_end = requests[-1]["timestamp_ms"]
    orig_duration_ms = t_end - t0
    if orig_duration_ms <= 0:
        return requests

    orig_rps = len(requests) / (orig_duration_ms / 1000)
    new_duration_ms = len(requests) / target_rps * 1000
    scale = new_duration_ms / orig_duration_ms  # >1 stretches, <1 compresses

    print(f"  RPS scaling: {orig_rps:.2f} → {target_rps:.2f} rps  "
          f"(duration: {orig_duration_ms/1000:.0f}s → {new_duration_ms/1000:.0f}s, "
          f"scale={scale:.3f})")

    for r in requests:
        r["timestamp_ms"] = int(t0 + (r["timestamp_ms"] - t0) * scale)
    return requests


def main():
    parser = argparse.ArgumentParser(
        description="Generate extended workload trace preserving statistical properties")
    parser.add_argument("input", help="Path to input workload.jsonl")
    parser.add_argument("output", help="Path to output workload-extended.jsonl")
    parser.add_argument("--multiplier", type=float, default=4.0,
                        help="Duration multiplier (default: 4.0)")
    parser.add_argument("--target-rps", type=float, default=None,
                        help="Scale timestamps so average RPS equals this value. "
                             "Preserves the burstiness pattern (shape), only "
                             "moves the RPS level vertically.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)

    # Load input workload
    print(f"Loading {args.input}...")
    entries = []
    with open(args.input) as f:
        for line in f:
            obj = json.loads(line)
            ts = obj["timestamp"]
            for req in obj["requests"]:
                entries.append({
                    "timestamp_ms": ts,
                    "input_tokens": req["Prompt Length"],
                    "output_tokens": req["Output Length"],
                    "hash_ids": req.get("hash_ids", []),
                    "prefix_group": req.get("prefix_group", ""),
                    "prompt": req.get("prompt", ""),
                })
    entries.sort(key=lambda e: e["timestamp_ms"])
    print(f"  Loaded {len(entries)} requests")

    # Analyze
    print("Analyzing distributions...")
    stats = analyze_workload(entries)

    # Generate
    print(f"Generating {args.multiplier}x extended workload...")
    new_requests = generate_workload(stats, args.multiplier, rng)

    # Scale RPS if requested
    if args.target_rps is not None:
        print(f"Scaling to target RPS={args.target_rps}...")
        new_requests = scale_rps(new_requests, args.target_rps)

    # Validate
    print("Validating...")
    _validate(entries, new_requests, stats)

    # Write
    write_workload(new_requests, args.output, stats["word_pool"],
                   int(stats["tokens_per_block"]), rng)


def _validate(original: List[Dict], generated: List[Dict], stats: Dict):
    """Print comparison stats between original and generated workloads."""
    from collections import defaultdict

    def _compute_stats(entries):
        pg_members = defaultdict(list)
        for e in entries:
            pg_members[e["prefix_group"]].append(e)

        pg_sizes = [len(v) for v in pg_members.values()]
        inputs = [e["input_tokens"] for e in entries]
        outputs = [e["output_tokens"] for e in entries]
        times = sorted(e["timestamp_ms"] for e in entries)
        duration_s = (times[-1] - times[0]) / 1000
        iats = [times[i+1] - times[i] for i in range(len(times)-1)]

        # Prefix hit ratio (using trie)
        class TrieNode:
            __slots__ = ['children']
            def __init__(self):
                self.children = {}

        root = TrieNode()
        hit_ratios = []
        for e in entries:
            node = root
            matched = 0
            for h in e["hash_ids"]:
                if h in node.children:
                    matched += 1
                    node = node.children[h]
                else:
                    break
            hr = (matched * int(stats["tokens_per_block"])) / e["input_tokens"] if e["input_tokens"] > 0 else 0
            hit_ratios.append(hr)
            node = root
            for h in e["hash_ids"]:
                if h not in node.children:
                    node.children[h] = TrieNode()
                node = node.children[h]

        return {
            "num_requests": len(entries),
            "num_groups": len(pg_members),
            "duration_s": duration_s,
            "rps": len(entries) / duration_s if duration_s > 0 else 0,
            "group_size_mean": np.mean(pg_sizes),
            "group_size_median": np.median(pg_sizes),
            "input_mean": np.mean(inputs),
            "input_median": np.median(inputs),
            "output_mean": np.mean(outputs),
            "output_median": np.median(outputs),
            "iat_mean": np.mean(iats) if iats else 0,
            "iat_median": np.median(iats) if iats else 0,
            "prefix_hit_mean": np.mean(hit_ratios),
            "prefix_hit_median": np.median(hit_ratios),
        }

    orig_stats = _compute_stats(original)
    gen_stats = _compute_stats(generated)

    fmt = "  {:30s} | {:>12s} | {:>12s}"
    print(fmt.format("", "Original", "Generated"))
    print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*12}")
    for key in orig_stats:
        ov = orig_stats[key]
        gv = gen_stats[key]
        if isinstance(ov, float):
            print(fmt.format(key, f"{ov:.1f}", f"{gv:.1f}"))
        else:
            print(fmt.format(key, str(ov), str(gv)))


if __name__ == "__main__":
    main()
