"""
Regression test for rewards.py.

Runs all 18 reward functions on deterministic synthetic inputs and checks their
outputs against a golden snapshot. The first time you run this with --refresh
it writes the snapshot; subsequent runs compare against it byte-for-byte.

Usage:
    python tests/test_rewards.py            # compare against existing golden
    python tests/test_rewards.py --refresh  # write a fresh golden snapshot
"""
import sys
import os
import json
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rewards


# Deterministic synthetic inputs. Fixed seed so this never drifts.
_rng = np.random.default_rng(42)
N = 100
TTFT = _rng.uniform(10.0, 8000.0, N)
TPOT = _rng.uniform(5.0, 250.0, N)
E2E = TTFT + TPOT * _rng.uniform(50, 200, N)
INPUT_TOKENS = _rng.integers(50, 5000, N).astype(np.float64)
OUTPUT_TOKENS = _rng.integers(1, 500, N).astype(np.float64)
KV_HIT = _rng.uniform(0.0, 1.0, N)

TTFT_SLO = 1000.0
TPOT_SLO = 50.0
TTFT_WEIGHT = 0.7
TTFT_P99 = float(np.percentile(TTFT, 99))
TPOT_P99 = float(np.percentile(TPOT, 99))


def _digest(result):
    """Hash a reward function's return dict for compact byte-comparison."""
    if not isinstance(result, dict):
        raise AssertionError(f"reward fn returned non-dict: {type(result)}")
    out = {}
    for k, v in sorted(result.items()):
        arr = np.asarray(v, dtype=np.float64)
        # Round to 9 decimal places: enough to detect any real change,
        # tight enough to survive normal float-rounding identity transforms.
        rounded = np.round(arr, 9)
        out[k] = {
            "shape": list(arr.shape),
            "sample": rounded[:5].tolist() if arr.ndim else float(arr),
            "sum":   float(rounded.sum()),
            "min":   float(rounded.min()),
            "max":   float(rounded.max()),
            "sha":   hashlib.sha256(rounded.tobytes()).hexdigest()[:16],
        }
    return out


# Each tuple: (name, callable, kwargs)
CASES = [
    ("calculate_rewards_simple",
        rewards.calculate_rewards_simple,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_slo=TTFT_SLO,
             avg_tpot_slo=TPOT_SLO, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_simple_extended",
        rewards.calculate_rewards_simple_extended,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_slo=TTFT_SLO,
             avg_tpot_slo=TPOT_SLO, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_piecewise_linear_steeper_gradient",
        rewards.calculate_rewards_piecewise_linear_steeper_gradient,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_slo=TTFT_SLO,
             avg_tpot_slo=TPOT_SLO, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_inverse_latency",
        rewards.calculate_rewards_inverse_latency,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_slo=TTFT_SLO,
             avg_tpot_slo=TPOT_SLO, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_latency_optimization",
        rewards.calculate_rewards_latency_optimization,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_slo=TTFT_SLO,
             avg_tpot_slo=TPOT_SLO, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_simple_latency_minimization",
        rewards.calculate_rewards_simple_latency_minimization,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_negative_reciprocal",
        rewards.calculate_rewards_negative_reciprocal,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_negative_linear",
        rewards.calculate_rewards_negative_linear,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_negative_squared",
        rewards.calculate_rewards_negative_squared,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_e2e_negative_linear",
        rewards.calculate_rewards_e2e,
        dict(e2e_values=E2E, reward_function="negative_linear")),
    ("calculate_rewards_e2e_log_normalized",
        rewards.calculate_rewards_e2e,
        dict(e2e_values=E2E, reward_function="log_normalized")),
    ("calculate_rewards_e2e_quantile_based",
        rewards.calculate_rewards_e2e,
        dict(e2e_values=E2E, reward_function="quantile_based",
             input_tokens=INPUT_TOKENS)),
    ("calculate_rewards_e2e_quantile_advantage",
        rewards.calculate_rewards_e2e,
        dict(e2e_values=E2E, reward_function="quantile_advantage",
             input_tokens=INPUT_TOKENS, num_buckets=5)),
    ("calculate_rewards_quantile_based",
        rewards.calculate_rewards_quantile_based,
        dict(ttft_values=TTFT, tpot_values=TPOT,
             input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS,
             ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_quantile_advantage",
        rewards.calculate_rewards_quantile_advantage,
        dict(latency_values=TTFT, input_tokens=INPUT_TOKENS, num_buckets=5)),
    ("calculate_rewards_absolute_latency",
        rewards.calculate_rewards_absolute_latency,
        dict(ttft_values=TTFT, tpot_values=TPOT,
             ttft_slo=TTFT_SLO, tpot_slo=TPOT_SLO,
             ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_throughput_based",
        rewards.calculate_rewards_throughput_based,
        dict(ttft_values=TTFT, tpot_values=TPOT,
             input_tokens=INPUT_TOKENS, ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_log_normalized",
        rewards.calculate_rewards_log_normalized,
        dict(ttft_values=TTFT, tpot_values=TPOT,
             ttft_p99=TTFT_P99, tpot_p99=TPOT_P99,
             ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_context_aware",
        rewards.calculate_rewards_context_aware,
        dict(ttft_values=TTFT, tpot_values=TPOT,
             input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS,
             kv_cache_hit_ratios=KV_HIT,
             base_ttft_slo=TTFT_SLO, avg_tpot_slo=TPOT_SLO,
             ttft_reward_weight=TTFT_WEIGHT)),
    ("calculate_rewards_negative_linear_and_prefix_locality",
        rewards.calculate_rewards_negative_linear_and_prefix_locality,
        dict(ttft_values=TTFT, tpot_values=TPOT, ttft_reward_weight=TTFT_WEIGHT,
             selected_pods=np.array(["pod_0001"]*N),
             base_data={"pod_0001-kv_differential": _rng.uniform(-50, 50, N)},
             hyperparameters={"PREFIX_LOCALITY_WEIGHT": 0.5})),
    ("calculate_combined_rewards",
        rewards.calculate_combined_rewards,
        dict(ttft_rewards=TTFT * -0.001, tpot_rewards=TPOT * -0.001,
             ttft_reward_weight=TTFT_WEIGHT)),
]


GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_rewards.json")


def run():
    refresh = "--refresh" in sys.argv

    results = {}
    for name, fn, kwargs in CASES:
        try:
            out = fn(**kwargs)
        except Exception as e:
            print(f"  FAIL  {name}: raised {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        if isinstance(out, np.ndarray):
            out = {"return": out}
        results[name] = _digest(out)

    if refresh:
        with open(GOLDEN_PATH, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"Wrote golden: {GOLDEN_PATH} ({len(results)} cases)")
        return 0

    if not os.path.exists(GOLDEN_PATH):
        print(f"No golden file at {GOLDEN_PATH}. Run with --refresh first.")
        return 2

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    n_pass = 0
    n_fail = 0
    for name in sorted(set(results) | set(golden)):
        cur = results.get(name)
        ref = golden.get(name)
        if cur == ref:
            n_pass += 1
        else:
            n_fail += 1
            print(f"  FAIL  {name}")
            if isinstance(cur, dict) and isinstance(ref, dict):
                for key in sorted(set(cur) | set(ref)):
                    cv = cur.get(key)
                    rv = ref.get(key)
                    if cv != rv:
                        print(f"    {key}: ref={rv}  cur={cv}")

    print(f"\n{n_pass} pass / {n_fail} fail / {len(CASES)} total")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
