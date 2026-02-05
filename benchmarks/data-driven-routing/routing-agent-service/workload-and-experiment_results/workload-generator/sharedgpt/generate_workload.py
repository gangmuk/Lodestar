#!/usr/bin/env python3
"""
Workload Generator for LLM Inference Routing Experiments

Reads SharedGPT workload statistics and generates JSONL workload files.

Multi-turn conversation constraints:
- Each conversation has multiple turns (requests)
- Turn K+1 must come after Turn K in the same conversation
- Turn K's request includes accumulated context from turns 1 to K
- Different conversations can be interleaved

Output format (JSONL):
{"timestamp": 99, "requests": [{"Prompt Length": 3997, "Output Length": 100, "prompt": "...", ...}]}
"""

import json
import argparse
import random
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_workload_stats(stats_file: str) -> List[Dict]:
    """Load workload statistics from JSON file."""
    print(f"Loading stats from {stats_file}...")
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    print(f"Loaded {len(stats):,} turns from {len(set(s['conversation_id'] for s in stats)):,} conversations")
    return stats


def group_by_conversation(stats: List[Dict]) -> Dict[str, List[Dict]]:
    """Group turns by conversation ID, sorted by turn number."""
    conversations = defaultdict(list)
    for turn in stats:
        conversations[turn['conversation_id']].append(turn)

    # Sort each conversation by turn number
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x['turn_number'])

    return dict(conversations)


def load_trace_rps(trace_file: str, duration_seconds: float, scale_rps: Optional[float] = None) -> Tuple[List[float], List[float]]:
    """
    Load RPS trace from CSV file and return time and RPS arrays.

    Args:
        trace_file: Path to CSV file with columns: timestamp_sec, rps
        duration_seconds: Maximum duration to use from trace (trims if longer)
        scale_rps: If provided, scale the trace so average RPS matches this value

    Returns:
        Tuple of (times, rps_values) where times are relative seconds from start
    """
    times = []
    rps_values = []

    with open(trace_file, 'r') as f:
        reader = csv.DictReader(f)
        first_timestamp = None
        for row in reader:
            rps_val = float(row['rps'])

            # Parse timestamp - could be datetime string or numeric
            ts_str = row['timestamp_sec']
            if first_timestamp is None:
                first_timestamp = ts_str
                relative_time = 0.0
            else:
                # Assume 1-second intervals in the trace
                relative_time = len(times)

            # Trim to duration
            if relative_time > duration_seconds:
                break

            times.append(relative_time)
            rps_values.append(rps_val)

    times = np.array(times)
    rps_values = np.array(rps_values)

    # Scale RPS if requested
    if scale_rps is not None and len(rps_values) > 0:
        current_avg = np.mean(rps_values[rps_values > 0]) if np.any(rps_values > 0) else 1.0
        if current_avg > 0:
            scale_factor = scale_rps / current_avg
            rps_values = rps_values * scale_factor

    return times.tolist(), rps_values.tolist()


def generate_timestamps(num_requests: int, duration_seconds: float,
                       arrival_pattern: str = "poisson", rps: float = None,
                       trace_file: Optional[str] = None) -> List[float]:
    """
    Generate timestamps for requests.

    Args:
        num_requests: Total number of requests
        duration_seconds: Total duration of the workload
        arrival_pattern: "poisson", "uniform", "bursty", "mixed", or "trace"
        rps: Requests per second (if None, calculated from num_requests/duration)
             For "mixed" pattern, this is treated as the MAX rps
             For "trace" pattern, if provided, scales the trace to match this average
        trace_file: Path to CSV file with RPS trace (required for "trace" pattern)
    """
    if rps is None:
        rps = num_requests / duration_seconds

    if arrival_pattern == "poisson":
        # Poisson arrival (exponential inter-arrival times)
        inter_arrivals = np.random.exponential(1.0 / rps, num_requests)
        timestamps = np.cumsum(inter_arrivals)
    elif arrival_pattern == "uniform":
        # Uniform distribution
        timestamps = np.linspace(0, duration_seconds, num_requests)
    elif arrival_pattern == "bursty":
        # Bursty pattern: alternating high and low rates
        timestamps = []
        t = 0
        high_rps = rps * 2
        low_rps = rps * 0.5
        burst_duration = 10  # seconds
        is_burst = True
        while len(timestamps) < num_requests:
            current_rps = high_rps if is_burst else low_rps
            inter_arrival = np.random.exponential(1.0 / current_rps)
            t += inter_arrival
            timestamps.append(t)
            if t % burst_duration < inter_arrival:
                is_burst = not is_burst
        timestamps = np.array(timestamps[:num_requests])
    elif arrival_pattern == "mixed":
        # Mixed pattern: combines ramp-up, steady, bursty, and ramp-down phases
        # rps argument is treated as MAX rps
        max_rps = rps
        timestamps = []
        t = 0

        # Define phases as (duration_fraction, phase_type, params)
        # Phase types: "ramp_up", "ramp_down", "steady", "bursty"
        # Fractions are relative RPS levels (will be scaled by max_rps)
        phases = [
            (0.15, "ramp_up", {"start_frac": 0.1, "end_frac": 0.6}),
            (0.15, "steady", {"frac": 0.6}),
            (0.20, "ramp_up", {"start_frac": 0.6, "end_frac": 1.0}),
            (0.15, "steady", {"frac": 1.0}),
            (0.20, "bursty", {"high_frac": 1.0, "low_frac": 0.3, "burst_period": 5}),
            (0.15, "ramp_down", {"start_frac": 0.7, "end_frac": 0.2}),
        ]

        # Calculate expected average RPS fraction to scale properly
        # This ensures we generate approximately num_requests in duration_seconds
        avg_frac = 0
        for dur_frac, ptype, params in phases:
            if ptype == "ramp_up":
                phase_avg = (params["start_frac"] + params["end_frac"]) / 2
            elif ptype == "ramp_down":
                phase_avg = (params["start_frac"] + params["end_frac"]) / 2
            elif ptype == "steady":
                phase_avg = params["frac"]
            elif ptype == "bursty":
                phase_avg = (params["high_frac"] + params["low_frac"]) / 2
            avg_frac += dur_frac * phase_avg

        # Scale max_rps so that average RPS matches target
        # target_avg_rps = num_requests / duration_seconds
        # avg_frac * scaled_max_rps = target_avg_rps
        target_avg_rps = num_requests / duration_seconds
        scaled_max_rps = target_avg_rps / avg_frac

        # Calculate phase boundaries
        phase_boundaries = []
        cumulative = 0
        for dur_frac, ptype, params in phases:
            start_time = cumulative * duration_seconds
            cumulative += dur_frac
            end_time = cumulative * duration_seconds
            # Convert fractions to actual RPS values
            scaled_params = {}
            for k, v in params.items():
                if k.endswith("_frac"):
                    new_key = k.replace("_frac", "_rps")
                    scaled_params[new_key] = v * scaled_max_rps
                elif k == "frac":
                    scaled_params["rps"] = v * scaled_max_rps
                else:
                    scaled_params[k] = v
            phase_boundaries.append((start_time, end_time, ptype, scaled_params))

        def get_current_rps(current_time: float) -> float:
            """Get the target RPS at a given time based on phase."""
            for start_t, end_t, ptype, params in phase_boundaries:
                if start_t <= current_time < end_t:
                    phase_progress = (current_time - start_t) / (end_t - start_t)

                    if ptype == "ramp_up":
                        return params["start_rps"] + phase_progress * (params["end_rps"] - params["start_rps"])
                    elif ptype == "ramp_down":
                        return params["start_rps"] - phase_progress * (params["start_rps"] - params["end_rps"])
                    elif ptype == "steady":
                        return params["rps"]
                    elif ptype == "bursty":
                        # Alternate between high and low within the phase
                        period = params["burst_period"]
                        cycle_pos = (current_time - start_t) % (2 * period)
                        if cycle_pos < period:
                            return params["high_rps"]
                        else:
                            return params["low_rps"]
            # Default fallback (after all phases)
            return scaled_max_rps * 0.5

        # Generate timestamps using time-varying RPS
        # Generate extra to ensure we have enough, then trim
        while len(timestamps) < int(num_requests * 1.3) and t < duration_seconds * 2:
            current_rps = max(0.1, get_current_rps(t))  # Avoid division by zero
            inter_arrival = np.random.exponential(1.0 / current_rps)
            t += inter_arrival
            timestamps.append(t)

        # Scale timestamps to fit within duration and trim to num_requests
        timestamps = np.array(timestamps)
        if len(timestamps) >= num_requests:
            timestamps = timestamps[:num_requests]
            # Scale to fit within duration
            if timestamps[-1] > duration_seconds:
                timestamps = timestamps * (duration_seconds / timestamps[-1])
        else:
            # If we don't have enough, scale what we have
            timestamps = timestamps * (duration_seconds / timestamps[-1]) if len(timestamps) > 0 else timestamps

        # Print phase info
        actual_max_rps = scaled_max_rps
        print(f"[MIXED PATTERN] target_avg_rps={target_avg_rps:.1f}, peak_rps≈{actual_max_rps:.1f}")
        print(f"  Phases: ramp_up(10%→60%) → steady(60%) → ramp_up(60%→100%) → steady(100%) → bursty → ramp_down")
    elif arrival_pattern == "trace":
        # Trace-based pattern: follows RPS from a trace file
        if trace_file is None:
            raise ValueError("trace_file is required for 'trace' arrival pattern")

        # Load trace and optionally scale to target RPS
        trace_times, trace_rps = load_trace_rps(trace_file, duration_seconds, scale_rps=rps)

        if len(trace_times) == 0:
            raise ValueError(f"No data loaded from trace file: {trace_file}")

        # Calculate expected total requests from trace
        trace_total_requests = sum(trace_rps)
        trace_duration = len(trace_times)
        trace_avg_rps = trace_total_requests / trace_duration if trace_duration > 0 else 1.0

        print(f"[TRACE PATTERN] Loaded {trace_duration}s of trace from {os.path.basename(trace_file)}")
        print(f"  Trace avg RPS: {trace_avg_rps:.2f}, max RPS: {max(trace_rps):.1f}, min RPS: {min(trace_rps):.1f}")
        print(f"  Trace total requests: {trace_total_requests:.0f}, target: {num_requests}")

        # Generate timestamps following the trace
        # For each second in the trace, generate Poisson arrivals at that second's RPS
        timestamps = []
        for sec_idx, sec_rps in enumerate(trace_rps):
            if sec_rps <= 0:
                continue

            # Generate arrivals for this second using Poisson process
            t = float(sec_idx)
            while t < sec_idx + 1:
                inter_arrival = np.random.exponential(1.0 / sec_rps)
                t += inter_arrival
                if t < sec_idx + 1:
                    timestamps.append(t)

        timestamps = np.array(timestamps)

        # Adjust to match target num_requests
        if len(timestamps) > num_requests:
            # Downsample: randomly select num_requests timestamps
            indices = np.sort(np.random.choice(len(timestamps), num_requests, replace=False))
            timestamps = timestamps[indices]
        elif len(timestamps) < num_requests:
            # Upsample: add more requests proportionally
            # Scale timestamps to generate more
            shortage = num_requests - len(timestamps)
            if len(timestamps) > 0:
                # Add random timestamps within the duration, following the trace distribution
                extra_timestamps = []
                while len(extra_timestamps) < shortage:
                    # Pick a random second weighted by RPS
                    total_rps = sum(trace_rps)
                    if total_rps > 0:
                        probs = [r / total_rps for r in trace_rps]
                        sec_idx = np.random.choice(len(trace_rps), p=probs)
                        # Random time within that second
                        extra_timestamps.append(sec_idx + np.random.random())
                timestamps = np.sort(np.concatenate([timestamps, np.array(extra_timestamps)]))

        # Scale timestamps to fit exactly within duration_seconds
        if len(timestamps) > 0 and timestamps[-1] > 0:
            scale_factor = min(duration_seconds, len(trace_times)) / max(timestamps[-1], 1)
            timestamps = timestamps * scale_factor

        print(f"  Generated {len(timestamps)} timestamps over {timestamps[-1] if len(timestamps) > 0 else 0:.1f}s")
    else:
        raise ValueError(f"Unknown arrival pattern: {arrival_pattern}")

    return timestamps.tolist()


def interleave_conversations(conversations: Dict[str, List[Dict]],
                            timestamps: List[float],
                            max_concurrent_convs: int = 100) -> List[Tuple[float, Dict]]:
    """
    Interleave requests from different conversations while maintaining order within each.

    Args:
        conversations: Dict of conversation_id -> list of turns (sorted by turn_number)
        timestamps: Pre-generated timestamps for all requests
        max_concurrent_convs: Maximum number of concurrent conversations

    Returns:
        List of (timestamp, turn_data) tuples, sorted by timestamp
    """
    # Create a queue for each conversation with remaining turns
    conv_queues = {conv_id: list(turns) for conv_id, turns in conversations.items()}
    conv_ids = list(conv_queues.keys())
    random.shuffle(conv_ids)

    # Track which conversations are "active" (have had at least one turn sent)
    active_convs = set()
    pending_convs = set(conv_ids)

    result = []
    timestamp_idx = 0

    # Simple round-robin with constraints
    while conv_queues and timestamp_idx < len(timestamps):
        # Add new conversations if we have room
        while len(active_convs) < max_concurrent_convs and pending_convs:
            new_conv = pending_convs.pop()
            active_convs.add(new_conv)

        # Find conversations that have pending turns
        ready_convs = [c for c in active_convs if conv_queues.get(c)]

        if not ready_convs:
            # All active conversations are done, add more
            if pending_convs:
                new_conv = pending_convs.pop()
                active_convs.add(new_conv)
                ready_convs = [new_conv]
            else:
                break

        # Pick a random ready conversation
        chosen_conv = random.choice(ready_convs)
        turn = conv_queues[chosen_conv].pop(0)

        result.append((timestamps[timestamp_idx], turn))
        timestamp_idx += 1

        # If conversation is done, remove it from active set
        if not conv_queues[chosen_conv]:
            active_convs.discard(chosen_conv)
            del conv_queues[chosen_conv]

    # Sort by timestamp (should already be mostly sorted)
    result.sort(key=lambda x: x[0])

    return result


def format_request(turn: Dict, include_prompt_text: bool = True,
                   input_scale: float = 1.0, output_scale: float = 1.0) -> Dict:
    """Format a turn into the output request format with optional scaling."""
    # Apply scaling to token counts
    scaled_prompt_tokens = max(1, int(turn['prompt_tokens'] * input_scale))
    scaled_output_tokens = max(1, int(turn['output_tokens'] * output_scale))
    scaled_prefix_tokens = max(0, int(turn['prefix_tokens'] * input_scale))
    scaled_new_input_tokens = max(1, int(turn['new_input_tokens'] * input_scale))

    request = {
        "Prompt Length": scaled_prompt_tokens,
        "Output Length": scaled_output_tokens,
        "conversation_id": turn['conversation_id'],
        "turn_number": turn['turn_number'],
        "num_turns_in_conv": turn['num_turns_in_conv'],
        "prefix_tokens": scaled_prefix_tokens,
        "prefix_ratio": turn['prefix_ratio'],  # ratio stays the same
        "new_input_tokens": scaled_new_input_tokens
    }

    if include_prompt_text and 'accumulated_prompt' in turn:
        # Optionally truncate prompt to match scaled length
        prompt = turn['accumulated_prompt']
        if input_scale < 1.0:
            # Rough truncation based on character ratio (4 chars per token approx)
            target_chars = int(len(prompt) * input_scale)
            prompt = prompt[:target_chars]
        request["prompt"] = prompt

    return request


def generate_workload(stats: List[Dict],
                     output_file: str,
                     num_requests: int = None,
                     duration_seconds: float = 300,
                     rps: float = 10,
                     arrival_pattern: str = "poisson",
                     max_concurrent_convs: int = 100,
                     include_prompt_text: bool = True,
                     filter_min_turns: int = 1,
                     filter_max_prompt_tokens: int = None,
                     input_scale: float = 1.0,
                     output_scale: float = 1.0,
                     trace_file: Optional[str] = None,
                     seed: int = 42):
    """
    Generate a workload JSONL file.

    Args:
        stats: List of turn statistics
        num_requests: Number of requests to generate (None = calculated from duration * rps)
        duration_seconds: Duration of workload in seconds
        rps: Target requests per second
        arrival_pattern: "poisson", "uniform", "bursty", "mixed", or "trace"
        max_concurrent_convs: Max concurrent conversations
        include_prompt_text: Whether to include full prompt text
        filter_min_turns: Only include conversations with at least this many turns
        filter_max_prompt_tokens: Filter out prompts longer than this
        input_scale: Scale factor for input/prompt tokens (0-1 to reduce, >1 to increase)
        output_scale: Scale factor for output tokens (0-1 to reduce, >1 to increase)
        trace_file: Path to RPS trace CSV file (required for "trace" arrival pattern)
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group by conversation
    conversations = group_by_conversation(stats)

    # Filter conversations
    if filter_min_turns > 1:
        conversations = {k: v for k, v in conversations.items()
                        if len(v) >= filter_min_turns}
        print(f"After filtering (min_turns={filter_min_turns}): {len(conversations):,} conversations")

    if filter_max_prompt_tokens:
        for conv_id in list(conversations.keys()):
            conversations[conv_id] = [t for t in conversations[conv_id]
                                      if t['prompt_tokens'] <= filter_max_prompt_tokens]
            if not conversations[conv_id]:
                del conversations[conv_id]
        print(f"After filtering (max_prompt_tokens={filter_max_prompt_tokens}): {len(conversations):,} conversations")

    # Count total turns
    total_turns = sum(len(turns) for turns in conversations.values())
    print(f"Total available turns: {total_turns:,}")

    # Calculate num_requests from duration * rps if not specified
    if num_requests is None:
        num_requests = int(duration_seconds * rps)
        print(f"Calculated num_requests from duration*rps: {num_requests:,}")

    # Cap at available turns
    num_requests = min(num_requests, total_turns)

    print(f"Generating {num_requests:,} requests over {duration_seconds}s (target RPS: {rps})")
    if input_scale != 1.0 or output_scale != 1.0:
        print(f"Scaling: input={input_scale}, output={output_scale}")

    # Generate timestamps
    timestamps = generate_timestamps(num_requests, duration_seconds, arrival_pattern, rps, trace_file)

    # Interleave conversations
    print("Interleaving conversations...")
    scheduled_requests = interleave_conversations(conversations, timestamps, max_concurrent_convs)

    # Write to JSONL
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for timestamp, turn in tqdm(scheduled_requests):
            request = format_request(turn, include_prompt_text, input_scale, output_scale)
            line = {
                "timestamp": round(timestamp, 3),
                "requests": [request]
            }
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f"\nGenerated {len(scheduled_requests):,} requests")

    # Print summary statistics
    print("\n[WORKLOAD SUMMARY]")
    turn_counts = defaultdict(int)
    for _, turn in scheduled_requests:
        turn_counts[turn['turn_number']] += 1

    for turn_num in sorted(turn_counts.keys())[:10]:
        print(f"  Turn {turn_num}: {turn_counts[turn_num]:,} requests")

    # Verify ordering constraints
    print("\n[VERIFICATION]")
    conv_last_turn = {}
    violations = 0
    for _, turn in scheduled_requests:
        conv_id = turn['conversation_id']
        turn_num = turn['turn_number']
        if conv_id in conv_last_turn:
            if turn_num <= conv_last_turn[conv_id]:
                violations += 1
        conv_last_turn[conv_id] = turn_num

    print(f"  Turn ordering violations: {violations}")
    if violations == 0:
        print("  ✓ All conversations maintain correct turn order")

    return scheduled_requests


def plot_workload(scheduled_requests: List[Tuple[float, Dict]],
                  output_file: str,
                  input_scale: float = 1.0,
                  output_scale: float = 1.0):
    """
    Generate publication-quality plots for workload characterization.

    Creates a comprehensive figure with multiple subplots showing:
    - Request arrival pattern over time
    - Input/output token distributions
    - Turn number distribution and multi-turn analysis
    - Prefix sharing characteristics
    - Conversation-level statistics
    """
    # Extract data
    timestamps = [t for t, _ in scheduled_requests]
    prompt_lengths = [turn['prompt_tokens'] * input_scale for _, turn in scheduled_requests]
    output_lengths = [turn['output_tokens'] * output_scale for _, turn in scheduled_requests]
    turn_numbers = [turn['turn_number'] for _, turn in scheduled_requests]
    prefix_ratios = [turn['prefix_ratio'] for _, turn in scheduled_requests]
    prefix_tokens = [turn['prefix_tokens'] * input_scale for _, turn in scheduled_requests]
    new_input_tokens = [turn['new_input_tokens'] * input_scale for _, turn in scheduled_requests]
    conv_ids = [turn['conversation_id'] for _, turn in scheduled_requests]

    duration = max(timestamps) if timestamps else 0
    num_requests = len(scheduled_requests)
    num_conversations = len(set(conv_ids))

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Color palette
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'tertiary': '#F18F01',
        'quaternary': '#C73E1D',
        'accent': '#3B1F2B'
    }

    # =========================================================================
    # Plot 1: Request Arrival Timeline (top-left, spans 2 columns)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    # Calculate RPS over time windows
    window_size = max(1, duration / 50)  # 50 bins
    time_bins = np.arange(0, duration + window_size, window_size)
    rps_values, _ = np.histogram(timestamps, bins=time_bins)
    rps_values = rps_values / window_size  # Convert to RPS
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

    ax1.fill_between(bin_centers, rps_values, alpha=0.3, color=colors['primary'])
    ax1.plot(bin_centers, rps_values, color=colors['primary'], linewidth=1.5)
    ax1.axhline(y=np.mean(rps_values), color=colors['quaternary'], linestyle='--',
                linewidth=1.5, label=f'Mean: {np.mean(rps_values):.1f} RPS')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Requests per Second')
    ax1.set_title('Request Arrival Rate Over Time')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, duration)
    ax1.set_ylim(bottom=0)

    # =========================================================================
    # Plot 2: Summary Statistics Box (top-right)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    # Calculate statistics
    avg_prompt = np.mean(prompt_lengths)
    avg_output = np.mean(output_lengths)
    avg_prefix_ratio = np.mean([r for r in prefix_ratios if r > 0]) if any(r > 0 for r in prefix_ratios) else 0
    turn_1_count = sum(1 for t in turn_numbers if t == 1)
    multi_turn_count = num_requests - turn_1_count

    stats_text = f"""
    Workload Summary
    {'─' * 28}
    Total Requests:     {num_requests:,}
    Duration:           {duration:.1f}s
    Avg RPS:            {num_requests/duration:.1f}

    Conversations:      {num_conversations:,}
    Avg Turns/Conv:     {num_requests/num_conversations:.1f}

    Turn 1 Requests:    {turn_1_count:,} ({turn_1_count/num_requests*100:.1f}%)
    Turn 2+ Requests:   {multi_turn_count:,} ({multi_turn_count/num_requests*100:.1f}%)

    Avg Input Tokens:   {avg_prompt:.0f}
    Avg Output Tokens:  {avg_output:.0f}
    Avg Prefix Ratio:   {avg_prefix_ratio:.1%}
    """
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # Plot 3: Input Token Distribution (middle-left)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Clip for visualization
    prompt_clipped = np.clip(prompt_lengths, 0, np.percentile(prompt_lengths, 99))
    ax3.hist(prompt_clipped, bins=50, color=colors['primary'], alpha=0.7, edgecolor='white')
    ax3.axvline(np.median(prompt_lengths), color=colors['quaternary'], linestyle='--',
                linewidth=2, label=f'Median: {np.median(prompt_lengths):.0f}')
    ax3.axvline(np.mean(prompt_lengths), color=colors['tertiary'], linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(prompt_lengths):.0f}')
    ax3.set_xlabel('Input Tokens (Prompt Length)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Input Token Distribution')
    ax3.legend(loc='upper right')

    # =========================================================================
    # Plot 4: Output Token Distribution (middle-center)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    output_clipped = np.clip(output_lengths, 0, np.percentile(output_lengths, 99))
    ax4.hist(output_clipped, bins=50, color=colors['secondary'], alpha=0.7, edgecolor='white')
    ax4.axvline(np.median(output_lengths), color=colors['quaternary'], linestyle='--',
                linewidth=2, label=f'Median: {np.median(output_lengths):.0f}')
    ax4.axvline(np.mean(output_lengths), color=colors['tertiary'], linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(output_lengths):.0f}')
    ax4.set_xlabel('Output Tokens')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Output Token Distribution')
    ax4.legend(loc='upper right')

    # =========================================================================
    # Plot 5: Turn Number Distribution (middle-right)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 2])

    turn_counts = defaultdict(int)
    for t in turn_numbers:
        turn_counts[t] += 1

    max_turn_display = min(10, max(turn_counts.keys()))
    turns = list(range(1, max_turn_display + 1))
    counts = [turn_counts.get(t, 0) for t in turns]
    other_count = sum(turn_counts[t] for t in turn_counts if t > max_turn_display)

    bar_colors = [colors['primary'] if t == 1 else colors['secondary'] for t in turns]
    bars = ax5.bar(turns, counts, color=bar_colors, alpha=0.7, edgecolor='white')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)

    ax5.set_xlabel('Turn Number')
    ax5.set_ylabel('Request Count')
    ax5.set_title('Requests by Turn Position')
    ax5.set_xticks(turns)
    if other_count > 0:
        ax5.annotate(f'+{other_count:,} more\n(turns {max_turn_display+1}+)',
                    xy=(max_turn_display, 0), fontsize=8, ha='center')

    # =========================================================================
    # Plot 6: Prefix Ratio by Turn (bottom-left)
    # =========================================================================
    ax6 = fig.add_subplot(gs[2, 0])

    prefix_by_turn = defaultdict(list)
    for i, (_, turn) in enumerate(scheduled_requests):
        if turn['turn_number'] <= 10:
            prefix_by_turn[turn['turn_number']].append(turn['prefix_ratio'])

    turns_with_data = sorted([t for t in prefix_by_turn.keys() if t > 1])[:9]
    if turns_with_data:
        means = [np.mean(prefix_by_turn[t]) for t in turns_with_data]
        stds = [np.std(prefix_by_turn[t]) for t in turns_with_data]

        ax6.bar(turns_with_data, means, yerr=stds, capsize=3,
                color=colors['tertiary'], alpha=0.7, edgecolor='white')
        for t, m in zip(turns_with_data, means):
            ax6.text(t, m + 0.03, f'{m:.2f}', ha='center', fontsize=8)

    ax6.set_xlabel('Turn Number')
    ax6.set_ylabel('Prefix Hit Ratio')
    ax6.set_title('KV Cache Prefix Reuse by Turn')
    ax6.set_ylim(0, 1.1)
    ax6.set_xticks(turns_with_data if turns_with_data else [2, 3, 4, 5])

    # =========================================================================
    # Plot 7: Token Breakdown by Turn (bottom-center)
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 1])

    prompt_by_turn = defaultdict(list)
    new_input_by_turn = defaultdict(list)
    prefix_tokens_by_turn = defaultdict(list)

    for _, turn in scheduled_requests:
        t = turn['turn_number']
        if t <= 8:
            prompt_by_turn[t].append(turn['prompt_tokens'] * input_scale)
            new_input_by_turn[t].append(turn['new_input_tokens'] * input_scale)
            prefix_tokens_by_turn[t].append(turn['prefix_tokens'] * input_scale)

    turns_display = sorted(prompt_by_turn.keys())[:8]
    if turns_display:
        x = np.arange(len(turns_display))
        width = 0.35

        prefix_means = [np.mean(prefix_tokens_by_turn[t]) for t in turns_display]
        new_means = [np.mean(new_input_by_turn[t]) for t in turns_display]

        ax7.bar(x, prefix_means, width, label='Prefix (cached)', color=colors['primary'], alpha=0.7)
        ax7.bar(x, new_means, width, bottom=prefix_means, label='New input', color=colors['tertiary'], alpha=0.7)

        ax7.set_xlabel('Turn Number')
        ax7.set_ylabel('Tokens')
        ax7.set_title('Input Composition by Turn')
        ax7.set_xticks(x)
        ax7.set_xticklabels(turns_display)
        ax7.legend(loc='upper left')

    # =========================================================================
    # Plot 8: Input vs Output Scatter (bottom-right)
    # =========================================================================
    ax8 = fig.add_subplot(gs[2, 2])

    # Sample for performance if too many points
    if len(prompt_lengths) > 5000:
        indices = np.random.choice(len(prompt_lengths), 5000, replace=False)
        sample_prompts = [prompt_lengths[i] for i in indices]
        sample_outputs = [output_lengths[i] for i in indices]
        sample_turns = [turn_numbers[i] for i in indices]
    else:
        sample_prompts = prompt_lengths
        sample_outputs = output_lengths
        sample_turns = turn_numbers

    # Color by turn number
    turn_colors = ['#2E86AB' if t == 1 else '#F18F01' if t == 2 else '#A23B72' for t in sample_turns]

    scatter = ax8.scatter(np.clip(sample_prompts, 0, np.percentile(prompt_lengths, 98)),
                         np.clip(sample_outputs, 0, np.percentile(output_lengths, 98)),
                         c=sample_turns, cmap='viridis', alpha=0.4, s=15)

    ax8.set_xlabel('Input Tokens')
    ax8.set_ylabel('Output Tokens')
    ax8.set_title('Input vs Output Tokens')
    cbar = plt.colorbar(scatter, ax=ax8, label='Turn #')

    # =========================================================================
    # Save figure
    # =========================================================================
    plot_file = output_file.replace('.jsonl', '_analysis.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

    print(f"\n[PLOTS SAVED]")
    print(f"  PDF (publication): {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate LLM inference workload from SharedGPT stats")
    parser.add_argument("--stats-file", type=str, default="sharedgpt_workload_stats.json",
                       help="Input statistics JSON file")
    parser.add_argument("--num-requests", type=int, default=None,
                       help="Number of requests (default: all available)")
    parser.add_argument("--duration", type=float, default=300,
                       help="Workload duration in seconds (default: 300)")
    parser.add_argument("--rps", type=float, default=10,
                       help="Target requests per second (default: 10)")
    parser.add_argument("--arrival", type=str, default="poisson",
                       choices=["poisson", "uniform", "bursty", "mixed", "trace"],
                       help="Arrival pattern (default: poisson). 'mixed' combines ramp-up, steady, bursty phases. 'trace' follows RPS from --trace-file")
    parser.add_argument("--trace-file", type=str, default=None,
                       help="Path to RPS trace CSV file (required for --arrival trace). Expected format: timestamp_sec,rps")
    parser.add_argument("--max-concurrent", type=int, default=100,
                       help="Max concurrent conversations (default: 100)")
    parser.add_argument("--no-prompt-text", action="store_true",
                       help="Don't include full prompt text (smaller output)")
    parser.add_argument("--min-turns", type=int, default=1,
                       help="Only include conversations with at least this many turns")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                       help="Filter out prompts longer than this")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--input-scale", type=float, default=1.0,
                       help="Scale factor for input/prompt tokens (0-1 to reduce, >1 to increase)")
    parser.add_argument("--output-scale", type=float, default=1.0,
                       help="Scale factor for output tokens (0-1 to reduce, >1 to increase)")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip generating workload analysis plots")

    args = parser.parse_args()

    # Load statistics
    stats = load_workload_stats(args.stats_file)

    # Validate trace file if using trace pattern
    if args.arrival == "trace" and args.trace_file is None:
        parser.error("--trace-file is required when using --arrival trace")

    # Generate workload
    output_file = f"workload_{args.arrival}.jsonl"
    scheduled_requests = generate_workload(
        output_file=output_file,
        stats=stats,
        num_requests=args.num_requests,
        duration_seconds=args.duration,
        rps=args.rps,
        arrival_pattern=args.arrival,
        max_concurrent_convs=args.max_concurrent,
        include_prompt_text=not args.no_prompt_text,
        filter_min_turns=args.min_turns,
        filter_max_prompt_tokens=args.max_prompt_tokens,
        input_scale=args.input_scale,
        output_scale=args.output_scale,
        trace_file=args.trace_file,
        seed=args.seed
    )

    # Generate plots
    if not args.no_plot:
        print("\nGenerating workload analysis plots...")
        plot_workload(
            scheduled_requests=scheduled_requests,
            output_file=output_file,
            input_scale=args.input_scale,
            output_scale=args.output_scale
        )


if __name__ == "__main__":
    main()
