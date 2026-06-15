#!/usr/bin/env python3
"""
Finite-capacity KV-cache simulator for workload.jsonl traces.

Replays each workload's `hash_ids` in timestamp order against a block-level
cache with longest-common-prefix-from-root match semantics — i.e., a request
with blocks [b1, b2, b3, b4] earns hits for the longest prefix b1..bm that is
currently in the cache, exactly matching how a real prefix-cache (vLLM, etc.)
sees the workload. After the access, *all* blocks of the request are inserted
(evicting under the chosen policy if full), because in a real system the
suffix blocks are computed fresh and made available for future reuse.

Three policies in parallel:
  - LRU            (standard online policy)
  - FIFO           (recency-blind baseline)
  - Belady's MIN   (offline optimal — upper bound for any online policy)

Outputs (per workload):
  - Miss-Rate Curve / Hit-Rate Curve: aggregate hit ratio vs cache capacity,
    swept on a log scale. Three curves (LRU, FIFO, Belady).
  - Hit-rate over time at the user-specified capacity (100-req sliding mean).
    Skipped if no capacity is given.

Usage:
  python3 simulate_kvcache.py \
      --cache-capacity-tokens 500000 \
      --tokens-per-hash-id 128 \
      -o cache_sim.pdf \
      workload1.jsonl [workload2.jsonl ...]
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
from collections import OrderedDict, deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 26,
        "axes.titlesize": 30,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        "figure.titlesize": 32,
        "axes.linewidth": 1.4,
        "lines.linewidth": 2.6,
        "lines.markersize": 9,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.4",
        "savefig.bbox": "tight",
    }
)

# fp16 is the reference; smaller dtypes scale the KV-per-token cost down.
DTYPE_MULTIPLIER = {"fp16": 1.0, "bf16": 1.0, "fp8": 0.5, "int8": 0.5, "int4": 0.25}

# Binary KiB / MiB / GiB (matches what nvidia-smi reports for GPU memory).
KB_PER_GB = 1024 * 1024


# ---------------------------------------------------------------------------
# Workload loading
# ---------------------------------------------------------------------------

def load_workload(path: str) -> tuple[list[list[int]], int]:
    """Load workload.jsonl in file order. Returns (hash_id_sequences,
    inferred_tokens_per_block).

    Timestamps are intentionally ignored: LRU/FIFO/Belady cache state depends
    only on the SEQUENCE of accesses, not on the wall-clock interval between
    them (we don't model TTL/idle-based eviction). The workload generators
    always write requests in timestamp order, so file order == access order.
    """
    sequences: list[list[int]] = []
    inferred_tpb = None
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            for req in obj["requests"]:
                hids = req.get("hash_ids") or []
                sequences.append([int(h) for h in hids])
                if inferred_tpb is None and hids:
                    pl = req.get("Prompt Length") or 0
                    if pl > 0:
                        inferred_tpb = max(1, int(round(pl / len(hids))))
    return sequences, (inferred_tpb or 128)


def workload_label(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    return parent or os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# Cache simulators
# ---------------------------------------------------------------------------

def _prefix_match(cache_contains, hids: list[int]) -> int:
    """Longest-prefix-from-root match count."""
    m = 0
    for b in hids:
        if cache_contains(b):
            m += 1
        else:
            break
    return m


def simulate_lru(sequences: list[list[int]], capacity: int) -> list[float]:
    cache: OrderedDict[int, None] = OrderedDict()
    hit_ratios: list[float] = []
    for hids in sequences:
        if not hids:
            hit_ratios.append(0.0)
            continue
        m = _prefix_match(cache.__contains__, hids)
        hit_ratios.append(m / len(hids))
        for b in hids:
            if b in cache:
                cache.move_to_end(b)
            else:
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                cache[b] = None
    return hit_ratios


def simulate_fifo(sequences: list[list[int]], capacity: int) -> list[float]:
    cache_set: set[int] = set()
    cache_q: deque[int] = deque()
    hit_ratios: list[float] = []
    for hids in sequences:
        if not hids:
            hit_ratios.append(0.0)
            continue
        m = _prefix_match(cache_set.__contains__, hids)
        hit_ratios.append(m / len(hids))
        for b in hids:
            if b not in cache_set:
                if len(cache_set) >= capacity:
                    evicted = cache_q.popleft()
                    cache_set.discard(evicted)
                cache_set.add(b)
                cache_q.append(b)
    return hit_ratios


def simulate_belady(sequences: list[list[int]], capacity: int) -> list[float]:
    """Belady's MIN: evict the cached block whose NEXT access is farthest in
    the future. Offline-optimal upper bound."""
    # Flatten access sequence; precompute next-use index for each access.
    flat: list[int] = []
    boundaries: list[int] = [0]  # cumulative access count per request
    for hids in sequences:
        flat.extend(hids)
        boundaries.append(len(flat))
    n = len(flat)
    next_use = [n] * n
    last_seen: dict[int, int] = {}
    for p in range(n - 1, -1, -1):
        b = flat[p]
        next_use[p] = last_seen.get(b, n)
        last_seen[b] = p

    cache_nu: dict[int, int] = {}  # block -> current next-use value
    # Max-heap simulated via min-heap on -next_use; entries are lazy.
    heap: list[tuple[int, int]] = []

    hit_ratios: list[float] = []
    p = 0
    for ri, hids in enumerate(sequences):
        if not hids:
            hit_ratios.append(0.0)
            continue
        m = _prefix_match(cache_nu.__contains__, hids)
        hit_ratios.append(m / len(hids))
        for b in hids:
            nu = next_use[p]
            if b in cache_nu:
                cache_nu[b] = nu
                heapq.heappush(heap, (-nu, b))
            else:
                if len(cache_nu) >= capacity:
                    # Pop until we find a heap entry that's still valid.
                    while heap:
                        neg_nu, cand = heapq.heappop(heap)
                        if cand in cache_nu and cache_nu[cand] == -neg_nu:
                            del cache_nu[cand]
                            break
                cache_nu[b] = nu
                heapq.heappush(heap, (-nu, b))
            p += 1
    return hit_ratios


POLICY_FNS = {
    "lru": simulate_lru,
    "fifo": simulate_fifo,
    "belady": simulate_belady,
}
POLICY_COLOURS = {"lru": "C0", "fifo": "C1", "belady": "C2"}


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_hit_ratio(sequences: list[list[int]], hit_ratios: list[float]) -> float:
    """Token-weighted aggregate hit ratio = total_hits / total_blocks."""
    total_blocks = 0
    total_hits = 0
    for hids, hr in zip(sequences, hit_ratios):
        if not hids:
            continue
        total_blocks += len(hids)
        total_hits += int(round(hr * len(hids)))
    return total_hits / total_blocks if total_blocks > 0 else 0.0


def sliding_mean(values: list[float], window: int = 100) -> list[float]:
    if not values or len(values) < window:
        return []
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid").tolist()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(analyses: list[dict], out_pdf: str, ts_capacities_blocks: list[int],
                 mrc_capacities_blocks: list[int], tokens_per_block: int,
                 kv_kb_per_token: float, dtype_mult: float) -> None:
    """One row per workload; columns = [MRC, ts@cap_1, ts@cap_2, ...]."""
    n = len(analyses)
    n_ts = len(ts_capacities_blocks)
    ncol = 1 + n_ts
    row_h = 7.5
    fig_w = 10.0 * ncol
    fig_h = max(1, n) * row_h + 1.2
    fig, axes = plt.subplots(n, ncol, figsize=(fig_w, fig_h), squeeze=False)

    def blocks_to_gb(b: int) -> float:
        return b * tokens_per_block * kv_kb_per_token * dtype_mult / KB_PER_GB

    mrc_gb_x = [blocks_to_gb(c) for c in mrc_capacities_blocks]
    ts_gb_for_cap = [blocks_to_gb(c) for c in ts_capacities_blocks]

    for i, data in enumerate(analyses):
        row = axes[i]
        label = data["label"]
        unique_blocks = data["unique_blocks"]
        unique_gb = blocks_to_gb(unique_blocks)

        # --- MRC: aggregate hit ratio vs cache capacity (GiB on x-axis) ---
        ax = row[0]
        for policy, ratios in data["mrc"].items():
            ax.plot(
                mrc_gb_x,
                ratios,
                marker="o",
                color=POLICY_COLOURS[policy],
                label=policy.upper(),
            )
        ax.set_xscale("log")
        ax.set_xlabel("Cache capacity (GiB)")
        ax.set_ylabel("Aggregate hit ratio")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(
            f"{label}\nMRC ({tokens_per_block} tok/block, "
            f"|unique|={unique_blocks} ≈ {unique_gb:.2f} GiB)"
        )
        # Vertical guides at each time-series capacity (so MRC and time-series are
        # easy to cross-reference).
        for tg in ts_gb_for_cap:
            ax.axvline(tg, color="grey", linestyle=":", linewidth=1.2, alpha=0.55)
        ax.legend(loc="best")

        # Secondary x-axis in tokens (so block-level intuition is still available).
        sec = ax.secondary_xaxis(
            "top",
            functions=(
                lambda g: g * KB_PER_GB / (kv_kb_per_token * dtype_mult),
                lambda t: t * (kv_kb_per_token * dtype_mult) / KB_PER_GB,
            ),
        )
        sec.set_xlabel("Cache capacity (tokens)")

        # --- Hit ratio over request order, one panel per capacity ---
        # x-axis = request index. Cache state depends only on access order, not
        # on wall-clock time — see load_workload.
        n_requests = len(data["sequences"])
        for j, cap_b in enumerate(ts_capacities_blocks):
            ax = row[1 + j]
            cap_gb = ts_gb_for_cap[j]
            per_policy = data["timeseries"][cap_b]
            for policy, ratios in per_policy.items():
                sm = sliding_mean(ratios, window=100)
                if not sm:
                    ax.text(0.5, 0.5, "trace too short", transform=ax.transAxes,
                            ha="center", va="center")
                    continue
                # Each sliding-window value at position k corresponds to the
                # mean of per-request hit ratios for requests [k-99 .. k].
                xs = list(range(100 - 1, 100 - 1 + len(sm)))
                agg = aggregate_hit_ratio(data["sequences"], ratios)
                ax.plot(
                    xs,
                    sm,
                    color=POLICY_COLOURS[policy],
                    label=f"{policy.upper()} ({agg:.3f})",
                )
            ax.set_xlabel("Request index")
            ax.set_ylabel("Hit ratio (100-req SW)")
            ax.set_xlim(0, n_requests)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{label}\n{cap_gb:.0f} GiB ({cap_b} blocks)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def gb_to_blocks(gb: float, kv_kb_per_token: float, dtype_mult: float,
                 tokens_per_block: int) -> int:
    tokens = (gb * KB_PER_GB) / (kv_kb_per_token * dtype_mult)
    return max(1, int(tokens // tokens_per_block))


def auto_mrc_capacities_blocks(
    low_gb: float, high_gb: float, n_points: int,
    kv_kb_per_token: float, dtype_mult: float, tokens_per_block: int,
) -> list[int]:
    """Log-spaced sweep in GiB, converted to unique block counts."""
    gbs = np.logspace(np.log10(low_gb), np.log10(high_gb), n_points)
    blocks = [gb_to_blocks(g, kv_kb_per_token, dtype_mult, tokens_per_block) for g in gbs]
    return sorted(set(blocks))


def analyze_workload(path: str, mrc_capacities_blocks: list[int],
                     ts_capacities_blocks: list[int], policies: list[str]) -> dict:
    sequences, inferred_tpb = load_workload(path)
    label = workload_label(path)
    unique_blocks = len({b for hids in sequences for b in hids})
    print(f"  {label}: {len(sequences)} requests, "
          f"{sum(len(s) for s in sequences)} block accesses, "
          f"{unique_blocks} unique blocks (inferred {inferred_tpb} tok/block)")

    # MRC: aggregate hit ratio at each swept block-capacity.
    mrc = {p: [] for p in policies}
    for cap in mrc_capacities_blocks:
        for p in policies:
            ratios = POLICY_FNS[p](sequences, cap)
            mrc[p].append(aggregate_hit_ratio(sequences, ratios))

    # Time-series: per-request hit ratios at each of the chosen capacities.
    timeseries: dict[int, dict[str, list[float]]] = {}
    for cap in ts_capacities_blocks:
        timeseries[cap] = {p: POLICY_FNS[p](sequences, cap) for p in policies}

    return {
        "label": label,
        "path": path,
        "sequences": sequences,
        "unique_blocks": unique_blocks,
        "mrc_capacities": mrc_capacities_blocks,
        "mrc": mrc,
        "timeseries": timeseries,
        "inferred_tokens_per_block": inferred_tpb,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workloads", nargs="+", help="One or more workload.jsonl paths")
    p.add_argument("-o", "--output", default=None,
                   help="Output PDF path. Default: kvcache_simulation.pdf in the same directory as the first input workload.")
    p.add_argument("--tokens-per-hash-id", type=int, default=None,
                   help="Tokens per block. If omitted, inferred from the first workload's Prompt Length / len(hash_ids).")
    p.add_argument("--cache-capacity-gb", type=float, nargs="+",
                   default=[16.0, 32.0, 64.0, 128.0],
                   help="Time-series capacities in GiB. Default: 16 32 64 128. "
                        "Each value becomes one time-series panel.")
    p.add_argument("--cache-capacity-tokens", type=int, nargs="+", default=None,
                   help="Time-series capacities in tokens. Overrides --cache-capacity-gb.")
    p.add_argument("--cache-capacity-blocks", type=int, nargs="+", default=None,
                   help="Time-series capacities in blocks. Overrides the other capacity flags.")
    p.add_argument("--kv-kb-per-token", type=float, default=128.0,
                   help="KV cache cost per token in KiB at fp16 (default 128 = Llama-3 8B fp16). "
                        "Other examples: Llama-3 70B=320, Qwen2.5-7B=56, Llama-2 7B=512.")
    p.add_argument("--kv-dtype", choices=list(DTYPE_MULTIPLIER.keys()),
                   default="fp16",
                   help="KV dtype. Scales --kv-kb-per-token: fp16/bf16=1.0, fp8/int8=0.5, int4=0.25.")
    p.add_argument("--mrc-gb-range", type=float, nargs=2, default=[16.0, 128.0],
                   metavar=("LOW", "HIGH"),
                   help="MRC sweep range in GiB. Default: 16 128.")
    p.add_argument("--mrc-points", type=int, default=12,
                   help="Number of log-spaced MRC sweep points within --mrc-gb-range. Default: 12.")
    p.add_argument("--mrc-capacities", type=int, nargs="+", default=None,
                   help="Explicit MRC sweep points in BLOCKS (overrides --mrc-gb-range and --mrc-points).")
    p.add_argument("--policies", nargs="+", default=["lru", "fifo", "belady"],
                   choices=["lru", "fifo", "belady"],
                   help="Subset of policies to simulate (default: all three).")
    args = p.parse_args()

    paths = [os.path.abspath(x) for x in args.workloads]
    for path in paths:
        if not os.path.isfile(path):
            print(f"Error: not a file: {path}", file=sys.stderr)
            return 1

    tokens_per_block = args.tokens_per_hash_id
    if tokens_per_block is None:
        _, inferred = load_workload(paths[0])
        tokens_per_block = inferred
        print(f"Inferred tokens-per-block = {tokens_per_block} (from {paths[0]})")

    dtype_mult = DTYPE_MULTIPLIER[args.kv_dtype]
    kv_kb_per_token = args.kv_kb_per_token

    # Resolve time-series capacities into block counts.
    # Precedence: blocks > tokens > GB.
    if args.cache_capacity_blocks:
        ts_capacities_blocks = sorted(set(args.cache_capacity_blocks))
        ts_source = "blocks (user-specified)"
    elif args.cache_capacity_tokens:
        ts_capacities_blocks = sorted(set(
            max(1, t // tokens_per_block) for t in args.cache_capacity_tokens
        ))
        ts_source = "tokens (user-specified)"
    else:
        ts_capacities_blocks = sorted(set(
            gb_to_blocks(g, kv_kb_per_token, dtype_mult, tokens_per_block)
            for g in args.cache_capacity_gb
        ))
        ts_source = f"GB ({args.cache_capacity_gb})"

    print(f"\nTime-series capacities from {ts_source}:")
    for cb in ts_capacities_blocks:
        gb = cb * tokens_per_block * kv_kb_per_token * dtype_mult / KB_PER_GB
        print(f"  {cb} blocks ≈ {cb * tokens_per_block} tokens ≈ {gb:.2f} GiB "
              f"(at {kv_kb_per_token} KiB/tok × {dtype_mult} for {args.kv_dtype})")

    # Resolve MRC sweep.
    if args.mrc_capacities:
        mrc_capacities_blocks = sorted(set(args.mrc_capacities))
    else:
        mrc_capacities_blocks = auto_mrc_capacities_blocks(
            args.mrc_gb_range[0], args.mrc_gb_range[1], args.mrc_points,
            kv_kb_per_token, dtype_mult, tokens_per_block,
        )
    print(f"\nMRC sweep: {len(mrc_capacities_blocks)} points in blocks: "
          f"{mrc_capacities_blocks}")

    print("\nSimulating per-workload cache behaviour...")
    analyses = [
        analyze_workload(path, mrc_capacities_blocks, ts_capacities_blocks, args.policies)
        for path in paths
    ]

    print("\nAggregate hit-ratio at each swept capacity:")
    for a in analyses:
        print(f"  {a['label']}:")
        header = "    capacity(GiB)  capacity(blocks) " + "  ".join(f"{p.upper():>8s}" for p in args.policies)
        print(header)
        for ci, cap in enumerate(a["mrc_capacities"]):
            gb = cap * tokens_per_block * kv_kb_per_token * dtype_mult / KB_PER_GB
            row_vals = "  ".join(f"{a['mrc'][p][ci]:8.3f}" for p in args.policies)
            print(f"    {gb:13.2f}  {cap:16d}  {row_vals}")

    # Default output: same dir as first workload.
    if args.output is None:
        first_dir = os.path.dirname(paths[0]) or "."
        out_pdf = os.path.join(first_dir, "kvcache_simulation.pdf")
    else:
        out_pdf = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    plot_results(
        analyses, out_pdf, ts_capacities_blocks, mrc_capacities_blocks,
        tokens_per_block, kv_kb_per_token, dtype_mult,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
