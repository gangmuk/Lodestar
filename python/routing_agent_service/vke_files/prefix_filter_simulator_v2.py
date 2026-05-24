"""
Prefix filter placement simulator v2 — faithful to real system dynamics.

Models TWO cache layers (matching real infra):
  1. Gateway prefix tree (GLOBAL): reports same hit ratio for all pods.
     Used for filter activation (benefit threshold) and the prefix_hash grouping.
  2. Per-pod vLLM KV cache (LOCAL): each pod caches prefixes of requests it served.
     Determines ACTUAL prefill time reduction. This is why routing matters.

Other key features:
  - CB routing model: approximates contextual bandit as EWMA-based pod selection
  - Condition-based filter activation (cluster GPU KV + benefit threshold)
  - Filter overrides CB only when CB picks outside candidate set
  - Models the feedback loop: filter concentrates traffic → queue buildup →
    CB tries to avoid → filter overrides → stuck

Usage:
    python prefix_filter_simulator_v2.py <experiment_dir> [options]

Calibrated from NVIDIA-A30 + llama-3-8b-instruct real experiment data (E1):
  - Real TTFT: mean=1409, P50=424, P90=3793, P99=10998
  - Prefill throughput: ~3500 tokens/sec per pod (uncached tokens)
  - TPOT: ~35ms/token
  - Per-pod KV cache capacity: ~100K tokens
"""

import re
import json
import sys
import os
import numpy as np
from collections import defaultdict, OrderedDict
import argparse
import heapq


# ============================================================
# Configuration (calibrated from real A30 experiment data)
# ============================================================

# Prefill throughput: tokens/sec per pod for uncached tokens.
# vLLM uses chunked prefill + continuous batching, so effective throughput
# is higher than simple sequential prefill.
#
# Calibration from real A30 data (bimodal TTFT pattern):
#   - Cache-hit requests (uncached < 500 tokens): very fast, dominated by scheduling overhead
#     → modeled with high throughput (12000 tok/s) + 30ms min
#   - Cache-miss requests (uncached > 1000 tokens): slower, CPU-bound prefill
#     → modeled with lower throughput (3500 tok/s) reflecting decode interference
#
# We use a throughput that depends on uncached token count:
#   effective_tps = PREFILL_TPS_BASE if uncached > PREFILL_BIMODAL_THRESHOLD else PREFILL_TPS_FAST
# Effective prefill throughput accounting for vLLM continuous batching.
# Real vLLM interleaves prefill chunks with decode, achieving higher throughput
# than sequential prefill. These values are calibrated to match real A30 data:
#   Real E1: mean=1329, P50=341, P90=3702, P99=10932
PREFILL_TPS_BASE = 6000      # tokens/sec for uncached (large prefill, cache miss)
PREFILL_TPS_FAST = 20000     # tokens/sec for partially cached (cache hit, small prefill)
PREFILL_BIMODAL_THRESHOLD = 500  # uncached tokens threshold

# Decode throughput: ms per output token
TPOT_MS = 35

# Per-pod KV cache capacity in tokens (LRU eviction).
# A30 has 24GB VRAM. vLLM reserves most for KV cache blocks.
# From real data: high KV usage (~85%) with ~10 concurrent requests.
# Prefix caching uses idle KV blocks → effective capacity ~100K tokens.
DEFAULT_CACHE_CAPACITY_TOKENS = 100_000

# Minimum prefill time (vLLM scheduling + memory allocation overhead)
MIN_PREFILL_TIME_SEC = 0.030  # 30ms

# Max concurrent prefills per pod
MAX_CONCURRENT_PREFILLS = 1

# Default filter thresholds (matching real experiment config)
DEFAULT_GPU_KV_SATURATION_THRESHOLD = 0.5
DEFAULT_PREFIX_BENEFIT_THRESHOLD = 100  # tokens


# ============================================================
# Per-Pod Local KV Cache (LRU, token-based)
# ============================================================

class PodKVCache:
    """Simulates per-pod local KV cache with LRU eviction.

    Each pod independently caches prefix tokens from requests it serves.
    This is the ACTUAL prefill savings — unlike the gateway's global prefix tree.
    """

    def __init__(self, capacity_tokens):
        self.capacity = capacity_tokens
        self.used_tokens = 0
        self.cache = OrderedDict()  # prefix_hash → num_tokens (LRU order)

    def lookup(self, prefix_hash):
        """Check cache. Returns cached token count (0 if miss)."""
        if prefix_hash in self.cache:
            self.cache.move_to_end(prefix_hash)
            return self.cache[prefix_hash]
        return 0

    def insert(self, prefix_hash, num_tokens):
        """Insert prefix after serving request. Evicts LRU if needed."""
        if prefix_hash in self.cache:
            self.cache.move_to_end(prefix_hash)
            return
        while self.used_tokens + num_tokens > self.capacity and self.cache:
            _, evicted_tokens = self.cache.popitem(last=False)
            self.used_tokens -= evicted_tokens
        if num_tokens <= self.capacity:
            self.cache[prefix_hash] = num_tokens
            self.used_tokens += num_tokens

    def utilization(self):
        return self.used_tokens / self.capacity if self.capacity > 0 else 0


# ============================================================
# Per-Pod Queueing Model
# ============================================================

class PodQueue:
    """Simulates vLLM's prefill queue and decode pipeline."""

    def __init__(self):
        self.prefill_free_at = 0.0
        self.decode_heap = []  # min-heap of finish times
        self.total_requests = 0
        # EWMA TTFT for CB learning
        self.ewma_ttft = 200.0
        self.ewma_alpha = 0.05

    def num_active_decodes(self, current_time):
        while self.decode_heap and self.decode_heap[0] <= current_time:
            heapq.heappop(self.decode_heap)
        return len(self.decode_heap)

    def num_inflight_prefill(self, current_time):
        """Estimate inflight prefill requests (queue depth)."""
        if current_time >= self.prefill_free_at:
            return 0
        return max(1, int((self.prefill_free_at - current_time) / 0.3))

    def enqueue(self, current_time, uncached_tokens, output_tokens):
        """Add request. Returns TTFT in ms."""
        # Bimodal throughput: small prefills (cache hits) are fast, large prefills are slower
        tps = PREFILL_TPS_FAST if uncached_tokens < PREFILL_BIMODAL_THRESHOLD else PREFILL_TPS_BASE
        prefill_time = max(uncached_tokens / tps, MIN_PREFILL_TIME_SEC)

        prefill_start = max(current_time, self.prefill_free_at)
        prefill_end = prefill_start + prefill_time
        self.prefill_free_at = prefill_end

        ttft_sec = prefill_end - current_time
        ttft_ms = ttft_sec * 1000

        decode_duration = output_tokens * TPOT_MS / 1000.0
        decode_finish = prefill_end + decode_duration
        heapq.heappush(self.decode_heap, decode_finish)

        # Update EWMA for CB learning
        self.ewma_ttft = self.ewma_alpha * ttft_ms + (1 - self.ewma_alpha) * self.ewma_ttft

        self.total_requests += 1
        return ttft_ms

    def estimated_gpu_kv(self, current_time, avg_tokens_per_request=2500):
        """Estimate GPU KV usage from active requests."""
        n_active = self.num_active_decodes(current_time) + self.num_inflight_prefill(current_time)
        # Each active request holds ~2500 tokens in KV cache
        # A30 capacity estimate: ~60K tokens → usage = n * 2500 / 60000
        return min(n_active * avg_tokens_per_request / 60000, 0.99)


# ============================================================
# CB Routing Model
# ============================================================

class CBRouter:
    """Approximates the contextual bandit routing behavior.

    - Before warmup: round-robin (untrained model)
    - After warmup: pick pod with best EWMA TTFT (learned)
    - Exploration: random pod selection at configured rate
    """

    def __init__(self, num_pods, warmup_requests=2000, exploration_rate=0.1):
        self.num_pods = num_pods
        self.warmup_requests = warmup_requests
        self.exploration_rate = exploration_rate
        self.rng = np.random.RandomState(42)
        self.rr_idx = 0

    def pick_pod(self, request_idx, pod_queues, current_time):
        """CB's initial pod choice (before filter override)."""
        if request_idx < self.warmup_requests:
            pod = self.rr_idx % self.num_pods
            self.rr_idx += 1
            return pod, 'untrained'

        if self.rng.random() < self.exploration_rate:
            return self.rng.randint(self.num_pods), 'exploration'

        # Exploitation: lowest EWMA TTFT
        best = min(range(self.num_pods), key=lambda p: pod_queues[p].ewma_ttft)
        return best, 'exploitation'


# ============================================================
# Placement Algorithms
# ============================================================

class StaticHashPlacement:
    """Static K with salted hash-based pod assignment."""

    def __init__(self, num_pods, k):
        self.num_pods = num_pods
        self.k = k
        self.name = f"static_K{k}"

    def get_candidates(self, prefix_hash, request_idx):
        n = self.num_pods
        k = min(self.k, n)
        selected = set()
        candidates = []
        for i in range(k):
            h = hash((i, prefix_hash)) % n
            while h in selected:
                h = (h + 1) % n
            selected.add(h)
            candidates.append(h)
        return candidates

    def record_request(self, prefix_hash):
        pass


class DynamicPlacement:
    """Dynamic collision-aware placement (matches real PrefixGroupPlacement)."""

    def __init__(self, num_pods, recompute_interval=1000, fallback_k=2):
        self.num_pods = num_pods
        self.recompute_interval = recompute_interval
        self.fallback_k = fallback_k
        self.name = f"dynamic_ri{recompute_interval}"
        self.window_counts = defaultdict(int)
        self.window_total = 0
        self.requests_since_recompute = 0
        self.group_candidates = {}
        self.group_k = {}

    def record_request(self, prefix_hash):
        if prefix_hash == 0:
            return
        self.window_counts[prefix_hash] += 1
        self.window_total += 1
        self.requests_since_recompute += 1
        if self.requests_since_recompute >= self.recompute_interval:
            self._recompute()

    def _recompute(self):
        N = self.num_pods
        if self.window_total == 0:
            return
        fair_share = self.window_total / N
        pod_loads = [0.0] * N
        new_candidates = {}
        new_k = {}
        sorted_groups = sorted(self.window_counts.items(), key=lambda x: -x[1])
        for prefix_hash, count in sorted_groups:
            k = max(1, int(np.ceil(count / fair_share)))
            k = min(k, N)
            new_k[prefix_hash] = k
            if k >= N:
                new_candidates[prefix_hash] = list(range(N))
            elif count > fair_share * 0.1:
                pods_by_load = sorted(range(N), key=lambda p: (pod_loads[p], p))
                candidates = pods_by_load[:k]
                new_candidates[prefix_hash] = candidates
                for p in candidates:
                    pod_loads[p] += count / k
        self.group_candidates = new_candidates
        self.group_k = new_k
        self.window_counts = defaultdict(int)
        self.window_total = 0
        self.requests_since_recompute = 0

    def get_candidates(self, prefix_hash, request_idx):
        if prefix_hash in self.group_candidates:
            return self.group_candidates[prefix_hash]
        n = self.num_pods
        k = min(self.fallback_k, n)
        selected = set()
        candidates = []
        for i in range(k):
            h = hash((i, prefix_hash)) % n
            while h in selected:
                h = (h + 1) % n
            selected.add(h)
            candidates.append(h)
        return candidates


class AdaptiveDominantPlacement:
    """Adaptive K: exempt dominant groups (matches Solution 2 in analysis).

    Groups exceeding dominant_threshold requests use K=all (no restriction).
    Small groups use fallback_k with hash-based selection.
    Counter runs on EVERY request (not just filtered).
    """

    def __init__(self, num_pods, dominant_threshold=500, fallback_k=2):
        self.num_pods = num_pods
        self.dominant_threshold = dominant_threshold
        self.fallback_k = fallback_k
        self.name = f"adaptive_dom{dominant_threshold}_K{fallback_k}"
        self.group_counts = defaultdict(int)

    def record_request(self, prefix_hash):
        if prefix_hash == 0:
            return
        self.group_counts[prefix_hash] += 1

    def get_candidates(self, prefix_hash, request_idx):
        if self.group_counts.get(prefix_hash, 0) >= self.dominant_threshold:
            return list(range(self.num_pods))
        n = self.num_pods
        k = min(self.fallback_k, n)
        selected = set()
        candidates = []
        for i in range(k):
            h = hash((i, prefix_hash)) % n
            while h in selected:
                h = (h + 1) % n
            selected.add(h)
            candidates.append(h)
        return candidates


class NoFilterPlacement:
    """Prefix filter disabled. CB routes freely."""

    def __init__(self, num_pods):
        self.num_pods = num_pods
        self.name = "no_filter"

    def get_candidates(self, prefix_hash, request_idx):
        return list(range(self.num_pods))

    def record_request(self, prefix_hash):
        pass


class LoadAwareFilterPlacement:
    """Load-aware filter: skips override if candidate pods are overloaded."""

    def __init__(self, num_pods, k=2, load_threshold=1.5):
        self.num_pods = num_pods
        self.k = k
        self.load_threshold = load_threshold
        self.name = f"load_aware_K{k}_t{load_threshold}"
        self.skip_count = 0

    def record_request(self, prefix_hash):
        pass

    def get_candidates(self, prefix_hash, request_idx):
        n = self.num_pods
        k = min(self.k, n)
        selected = set()
        candidates = []
        for i in range(k):
            h = hash((i, prefix_hash)) % n
            while h in selected:
                h = (h + 1) % n
            selected.add(h)
            candidates.append(h)
        return candidates

    def should_skip_filter(self, candidates, pod_queues, current_time):
        all_inflight = [pod_queues[p].num_inflight_prefill(current_time)
                        for p in range(self.num_pods)]
        avg = np.mean(all_inflight) if all_inflight else 0
        candidate_max = max(pod_queues[p].num_inflight_prefill(current_time)
                            for p in candidates)
        if candidate_max > max(avg * self.load_threshold, 3):
            self.skip_count += 1
            return True
        return False


# ============================================================
# Workload Trace Extraction
# ============================================================

def extract_trace(experiment_dir):
    """Extract workload trace from gateway log."""
    log_path = os.path.join(experiment_dir, "filtered-aibrix-gateway-plugins.log.csv")

    requests = []
    with open(log_path) as f:
        for line in f:
            m = re.match(r'I\d+ (\d+):(\d+):(\d+)\.(\d+)', line)
            if not m:
                continue
            h, mi, s, us = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            t = h * 3600 + mi * 60 + s + us / 1e6

            fields = line.strip().split('@')
            d = {}
            for i in range(0, len(fields) - 1, 2):
                d[fields[i].strip()] = fields[i + 1].strip()

            try:
                inp = int(d.get('numInputTokens', 0))
                out = int(d.get('numOutputTokens', 0))
                prefix_hash = int(d.get('hashOfMatchedPrefix', 0))
                ttft_real = int(d.get('ttft', 0))

                # Gateway's global hit ratio (same for all pods)
                hit_str = d.get('allPodsKvCacheHitRatios', '{}')
                hit = json.loads(hit_str)
                hit_values = list(hit.values())
                gateway_hit_ratio = hit_values[0] / 100.0 if hit_values else 0.0

                # Real GPU KV usage from trace (used for filter activation)
                gpu_kv_str = d.get('vllmGPUKVCacheUsage', '{}')
                gpu_kv = json.loads(gpu_kv_str)
                gpu_kv_values = [v for v in gpu_kv.values() if isinstance(v, (int, float))]
                real_cluster_gpu_kv = np.mean(gpu_kv_values) if gpu_kv_values else 0.0

                # Estimate prefix tokens from hit ratio
                prefix_tokens = int(inp * gateway_hit_ratio) if gateway_hit_ratio > 0 else 0

                requests.append({
                    'arrival': t,
                    'hash': prefix_hash,
                    'input_tokens': inp,
                    'output_tokens': max(out, 1),
                    'prefix_tokens': prefix_tokens,
                    'gateway_hit_ratio': gateway_hit_ratio,
                    'real_cluster_gpu_kv': real_cluster_gpu_kv,
                    'ttft_real': ttft_real,
                })
            except (ValueError, KeyError):
                continue

    # Handle day wrap and normalize
    for i in range(1, len(requests)):
        if requests[i]['arrival'] < requests[i - 1]['arrival'] - 3600:
            requests[i]['arrival'] += 86400
    t0 = requests[0]['arrival']
    for r in requests:
        r['arrival'] -= t0

    # Build prefix group info (average prefix tokens per group)
    group_prefix = defaultdict(list)
    for r in requests:
        if r['prefix_tokens'] > 0:
            group_prefix[r['hash']].append(r['prefix_tokens'])
    group_avg_prefix = {}
    for h, lengths in group_prefix.items():
        group_avg_prefix[h] = int(np.mean(lengths))

    return requests, group_avg_prefix


# ============================================================
# Simulation Runner
# ============================================================

def run_simulation(requests, group_avg_prefix, num_pods, placement,
                   cache_capacity=DEFAULT_CACHE_CAPACITY_TOKENS,
                   gpu_kv_threshold=DEFAULT_GPU_KV_SATURATION_THRESHOLD,
                   benefit_threshold=DEFAULT_PREFIX_BENEFIT_THRESHOLD,
                   exploration_rate=0.1,
                   cb_warmup=2000):
    """Run simulation with two-layer cache model + CB routing + prefix filter.

    Two cache layers:
    1. Gateway prefix tree (global): determines filter activation (benefit calc).
       We use the real trace's gateway_hit_ratio for this.
    2. Per-pod vLLM KV cache (local): determines actual prefill time savings.
       Simulated with LRU eviction per pod.
    """

    # Per-pod state
    caches = [PodKVCache(cache_capacity) for _ in range(num_pods)]
    queues = [PodQueue() for _ in range(num_pods)]
    cb = CBRouter(num_pods, warmup_requests=cb_warmup, exploration_rate=exploration_rate)

    results = []
    filter_override_count = 0
    filter_active_count = 0

    for idx, req in enumerate(requests):
        t = req['arrival']
        prefix_hash = req['hash']
        input_tokens = req['input_tokens']
        output_tokens = req['output_tokens']
        prefix_tokens = req.get('prefix_tokens', 0)
        if prefix_hash in group_avg_prefix:
            prefix_tokens = group_avg_prefix[prefix_hash]

        # Record for placement learning (ALWAYS)
        placement.record_request(prefix_hash)

        # ---- Filter activation check (uses GATEWAY's global hit ratio) ----
        # The gateway reports same hit ratio for all pods; this determines
        # whether the filter considers this request "cacheable"
        gateway_hit = req.get('gateway_hit_ratio', 0)
        gateway_benefit = gateway_hit * input_tokens  # tokens of potential cache benefit

        # Use real cluster GPU KV from trace (workload property, not routing-dependent).
        # This is critical: the simulated GPU KV model underestimates because it
        # doesn't account for prefix cache blocks stored in KV memory.
        cluster_gpu_kv = req.get('real_cluster_gpu_kv', 0)
        if cluster_gpu_kv == 0:
            # Fallback to simulated estimate if trace data unavailable
            cluster_gpu_kv = np.mean([queues[p].estimated_gpu_kv(t) for p in range(num_pods)])

        # ---- Step 1: CB picks initial pod ----
        cb_pod, cb_reason = cb.pick_pod(idx, queues, t)

        # ---- Step 2: Prefix filter override ----
        filter_active = False
        filter_overridden = False
        final_pod = cb_pod

        is_load_aware = isinstance(placement, LoadAwareFilterPlacement)
        is_no_filter = isinstance(placement, NoFilterPlacement)

        if not is_no_filter and prefix_hash != 0 and cb_reason != 'exploration':
            if cluster_gpu_kv > gpu_kv_threshold and gateway_benefit > benefit_threshold:
                filter_active = True
                filter_active_count += 1

                candidates = placement.get_candidates(prefix_hash, idx)

                # Load-aware: check if candidates are overloaded
                if is_load_aware and placement.should_skip_filter(candidates, queues, t):
                    final_pod = cb_pod
                elif cb_pod not in candidates:
                    # Override: pick candidate with best EWMA TTFT (like CB would)
                    best_candidate = min(candidates, key=lambda p: queues[p].ewma_ttft)
                    final_pod = best_candidate
                    filter_overridden = True
                    filter_override_count += 1

        pod_id = final_pod

        # ---- Step 3: Compute actual TTFT using per-pod local KV cache ----
        # The per-pod cache determines ACTUAL uncached tokens (may differ from gateway's global ratio)
        pod_cached_tokens = caches[pod_id].lookup(prefix_hash) if prefix_hash != 0 else 0
        actual_uncached = max(input_tokens - pod_cached_tokens, 0)

        ttft_ms = queues[pod_id].enqueue(t, actual_uncached, output_tokens)

        # Update per-pod cache: prefix is cached after prefill completes (not at arrival).
        # We approximate by inserting at arrival time since we process requests sequentially.
        # The slight inaccuracy: a request arriving during this request's prefill might
        # see a hit that doesn't exist yet. For high-traffic groups this is negligible
        # (previous request already cached it). For correctness we'd need deferred inserts,
        # but that adds complexity for minimal accuracy gain.
        if prefix_hash != 0 and prefix_tokens > 0:
            caches[pod_id].insert(prefix_hash, prefix_tokens)

        results.append({
            'arrival': t,
            'ttft': ttft_ms,
            'pod': pod_id,
            'hash': prefix_hash,
            'cached_tokens': pod_cached_tokens,
            'uncached_tokens': actual_uncached,
            'cache_hit': pod_cached_tokens > 0,
            'filter_active': filter_active,
            'filter_overridden': filter_overridden,
            'cb_pod': cb_pod,
            'cb_reason': cb_reason,
            'ttft_real': req.get('ttft_real', 0),
        })

    return _compute_metrics(results, num_pods, caches, queues,
                            filter_override_count, filter_active_count)


def _compute_metrics(results, num_pods, caches, queues,
                     filter_override_count, filter_active_count):
    """Compute metrics with real vs simulated comparison."""
    if not results:
        return {}

    max_time = results[-1]['arrival']
    bucket_size = 120
    n_buckets = max(1, int(max_time / bucket_size) + 1)
    bucket_counts = [0] * n_buckets
    for r in results:
        b = min(int(r['arrival'] / bucket_size), n_buckets - 1)
        bucket_counts[b] += 1
    bucket_rps = [c / bucket_size for c in bucket_counts]

    median_rps = np.median(bucket_rps)
    high_rps_threshold = median_rps * 1.3
    high_rps_start = None
    for i, rps in enumerate(bucket_rps):
        if rps > high_rps_threshold:
            high_rps_start = i * bucket_size
            break
    if high_rps_start is None:
        high_rps_start = max_time * 0.6

    all_ttft = [r['ttft'] for r in results]
    post = [r['ttft'] for r in results if r['arrival'] >= high_rps_start]

    real_ttft = [r['ttft_real'] for r in results if r['ttft_real'] > 0]
    post_real = [r['ttft_real'] for r in results
                 if r['arrival'] >= high_rps_start and r['ttft_real'] > 0]

    def percentiles(vals):
        if not vals:
            return {'count': 0, 'mean': 0, 'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
        s = sorted(vals)
        n = len(s)
        return {
            'count': n, 'mean': np.mean(s),
            'p50': s[n // 2], 'p90': s[int(n * 0.9)],
            'p95': s[int(n * 0.95)], 'p99': s[int(n * 0.99)],
        }

    post_results = [r for r in results if r['arrival'] >= high_rps_start]
    pod_counts = defaultdict(int)
    for r in post_results:
        pod_counts[r['pod']] += 1
    total_post = len(post_results)
    ideal = total_post / num_pods if num_pods > 0 else 1
    max_load = max(pod_counts.values()) if pod_counts else 0
    imbalance = max_load / ideal if ideal > 0 else 0

    post_filter_active = sum(1 for r in post_results if r['filter_active'])
    post_filter_overridden = sum(1 for r in post_results if r['filter_overridden'])

    post_hits = sum(1 for r in post_results if r['cache_hit'])
    post_hit_rate = post_hits / len(post_results) if post_results else 0

    total_cached = sum(r['cached_tokens'] for r in post_results)
    total_input = sum(r['cached_tokens'] + r['uncached_tokens'] for r in post_results)
    cache_token_rate = total_cached / total_input if total_input > 0 else 0

    avg_cache_util = np.mean([c.utilization() for c in caches])

    pod_ttfts = defaultdict(list)
    for r in post_results:
        pod_ttfts[r['pod']].append(r['ttft'])
    pod_avg_ttft = {p: np.mean(v) for p, v in pod_ttfts.items()}

    return {
        'all': percentiles(all_ttft),
        'post_high_rps': percentiles(post),
        'real_all': percentiles(real_ttft),
        'real_post': percentiles(post_real),
        'imbalance': imbalance,
        'pod_distribution': dict(pod_counts),
        'pod_avg_ttft': dict(pod_avg_ttft),
        'filter_overrides': filter_override_count,
        'filter_active': filter_active_count,
        'post_filter_active': post_filter_active,
        'post_filter_overridden': post_filter_overridden,
        'cache_hit_rate': post_hit_rate,
        'cache_token_savings': cache_token_rate,
        'cache_utilization': avg_cache_util,
        'high_rps_start': high_rps_start,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prefix filter simulator v2 (faithful)")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--num-pods", type=int, default=7)
    parser.add_argument("--algorithms", type=str,
                        default="no_filter,static_K2,static_K3,dynamic,adaptive_dom500")
    parser.add_argument("--gpu-kv-threshold", type=float, default=DEFAULT_GPU_KV_SATURATION_THRESHOLD)
    parser.add_argument("--benefit-threshold", type=float, default=DEFAULT_PREFIX_BENEFIT_THRESHOLD)
    parser.add_argument("--cache-capacity", type=int, default=DEFAULT_CACHE_CAPACITY_TOKENS)
    parser.add_argument("--exploration-rate", type=float, default=0.1)
    parser.add_argument("--cb-warmup", type=int, default=2000)
    parser.add_argument("--recompute-interval", type=int, default=1000)
    parser.add_argument("--num-trials", type=int, default=1,
                        help="Number of trials with randomized hash remapping")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    print(f"Extracting trace from {args.experiment_dir}...")
    requests, group_avg_prefix = extract_trace(args.experiment_dir)
    print(f"  {len(requests)} requests, duration={requests[-1]['arrival']:.0f}s")
    print(f"  {len(group_avg_prefix)} groups with known prefix lengths")

    group_counts = defaultdict(int)
    for r in requests:
        if r['hash'] != 0:
            group_counts[r['hash']] += 1
    sorted_groups = sorted(group_counts.items(), key=lambda x: -x[1])
    print(f"  Top groups: {[(c, f'{c/len(requests)*100:.1f}%') for _, c in sorted_groups[:3]]}")

    real_ttft = [r['ttft_real'] for r in requests if r['ttft_real'] > 0]
    if real_ttft:
        s = sorted(real_ttft)
        n = len(s)
        print(f"  Real TTFT: mean={np.mean(s):.0f}, P50={s[n//2]}, P90={s[int(n*0.9)]}, P99={s[int(n*0.99)]}")

    N = args.num_pods

    print(f"\n{'Algorithm':<28s} {'Mean':>6s} {'P50':>6s} {'P90':>7s} {'P95':>7s} {'P99':>7s} "
          f"{'Imbal':>6s} {'HitR':>5s} {'TokSv':>5s} {'FiltOvr':>7s} "
          f"| {'PostMean':>8s} {'PostP90':>7s}")
    print("-" * 130)

    algo_names = args.algorithms.split(",")

    for trial in range(args.num_trials):
        if args.num_trials > 1:
            rng = np.random.RandomState(trial * 12345 + 42)
            hash_remap = {}
            for r in requests:
                if r['hash'] not in hash_remap:
                    hash_remap[r['hash']] = rng.randint(0, 2**62)
            trial_requests = []
            for r in requests:
                r2 = dict(r)
                r2['hash'] = hash_remap[r['hash']]
                trial_requests.append(r2)
            trial_gap = {hash_remap.get(h, h): v for h, v in group_avg_prefix.items()}
            print(f"\n--- Trial {trial + 1}/{args.num_trials} ---")
        else:
            trial_requests = requests
            trial_gap = group_avg_prefix

        for algo_name in algo_names:
            algo_name = algo_name.strip()
            if algo_name == "no_filter":
                algo = NoFilterPlacement(N)
            elif algo_name.startswith("static_K"):
                k = int(algo_name.split("K")[1])
                algo = StaticHashPlacement(N, k)
            elif algo_name == "dynamic":
                algo = DynamicPlacement(N, recompute_interval=args.recompute_interval)
            elif algo_name.startswith("adaptive_dom"):
                parts = algo_name.replace("adaptive_dom", "").split("_K")
                threshold = int(parts[0]) if parts[0] else 500
                fallback_k = int(parts[1]) if len(parts) > 1 else 2
                algo = AdaptiveDominantPlacement(N, dominant_threshold=threshold,
                                                 fallback_k=fallback_k)
            elif algo_name.startswith("load_aware"):
                parts = algo_name.replace("load_aware_K", "")
                k = int(parts) if parts else 2
                algo = LoadAwareFilterPlacement(N, k=k)
            else:
                print(f"Unknown algorithm: {algo_name}")
                continue

            metrics = run_simulation(trial_requests, trial_gap, N, algo,
                                     cache_capacity=args.cache_capacity,
                                     gpu_kv_threshold=args.gpu_kv_threshold,
                                     benefit_threshold=args.benefit_threshold,
                                     exploration_rate=args.exploration_rate,
                                     cb_warmup=args.cb_warmup)

            a = metrics['all']
            p = metrics['post_high_rps']

            print(f"{algo.name:<28s} {a['mean']:>6.0f} {a['p50']:>6.0f} {a['p90']:>7.0f} "
                  f"{a['p95']:>7.0f} {a['p99']:>7.0f} "
                  f"{metrics['imbalance']:>5.2f}x {metrics['cache_hit_rate']:>4.1%} "
                  f"{metrics['cache_token_savings']:>4.1%} {metrics['filter_overrides']:>7d} "
                  f"| {p['mean']:>8.0f} {p['p90']:>7.0f}")

            if args.verbose:
                pod_dist = metrics['pod_distribution']
                top2 = sorted(pod_dist.values(), reverse=True)[:2]
                total = sum(pod_dist.values()) or 1
                print(f"  Top-2 pod share: {sum(top2)/total:.1%}")
                print(f"  Pod avg TTFT: {', '.join(f'p{k}:{v:.0f}' for k, v in sorted(metrics['pod_avg_ttft'].items()))}")
                print(f"  Pod counts:   {', '.join(f'p{k}:{v}' for k, v in sorted(pod_dist.items()))}")
                print(f"  Filter: active={metrics['post_filter_active']}, overridden={metrics['post_filter_overridden']} during high-RPS")
                print(f"  Cache: hit_rate={metrics['cache_hit_rate']:.1%}, token_savings={metrics['cache_token_savings']:.1%}, utilization={metrics['cache_utilization']:.1%}")
                print(f"  High-RPS starts at t={metrics['high_rps_start']:.0f}s")
                if hasattr(algo, 'skip_count'):
                    print(f"  Load-aware skips: {algo.skip_count}")
                r = metrics.get('real_all', {})
                if r.get('count', 0) > 0:
                    print(f"  Real baseline: mean={r['mean']:.0f}, P50={r['p50']:.0f}, P90={r['p90']:.0f}")


if __name__ == "__main__":
    main()
