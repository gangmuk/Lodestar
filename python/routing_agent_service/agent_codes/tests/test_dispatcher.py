"""
Dispatcher-equivalence test.

For every reward function name in rewards.KNOWN_REWARD_FUNCTIONS, asserts that
calling rewards.compute_rewards(name, df, hp) returns the same output that the
original elif chain in data_normalizer.py would have produced. Includes the
fallback paths (e.g. quantile_based -> latency_optimized when input_tokens
column missing).

This is the regression net for the elif-chain -> registry refactor.
"""
import sys
import os
import json
import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rewards


def _df(seed=0, n=100, with_tokens=True):
    """Build a synthetic processed-CSV-shaped DataFrame."""
    rng = np.random.default_rng(seed)
    cols = {
        'ttft':     rng.uniform(50.0, 8000.0, n),
        'avg_tpot': rng.uniform(5.0, 250.0, n),
    }
    if with_tokens:
        cols['input_tokens']  = rng.integers(50, 5000, n).astype(np.float64)
        cols['output_tokens'] = rng.integers(1, 500, n).astype(np.float64)
    return pd.DataFrame(cols)


def _hp(**extra):
    h = {
        'TTFT_SLO':            1000.0,
        'AVG_TPOT_SLO':        50.0,
        'TTFT_REWARD_WEIGHT':  0.7,
    }
    h.update(extra)
    return h


def _digest(result):
    """Hash a reward dict for compact byte-comparison."""
    out = {}
    for k, v in sorted(result.items()):
        arr = np.asarray(v, dtype=np.float64)
        rounded = np.round(arr, 9)
        out[k] = {
            "shape": list(arr.shape),
            "sum":   float(rounded.sum()),
            "sha":   hashlib.sha256(rounded.tobytes()).hexdigest()[:16],
        }
    return out


def _check(name, dispatcher_kwargs, direct_fn, direct_kwargs):
    """Run dispatcher; run direct function; assert identical results."""
    via_dispatch = rewards.compute_rewards(**dispatcher_kwargs)
    via_direct = direct_fn(**direct_kwargs)
    d1 = _digest(via_dispatch)
    d2 = _digest(via_direct)
    # Compare only the shared keys (some functions return diagnostic extras).
    shared = set(d1) & set(d2)
    if not shared:
        return False, f"no shared keys between dispatch ({list(d1)}) and direct ({list(d2)})"
    for k in shared:
        if d1[k] != d2[k]:
            return False, f"mismatch on '{k}': dispatch={d1[k]} direct={d2[k]}"
    return True, "ok"


# All 14 known reward function names with their direct-call equivalents.
# Each entry: (name, df_factory, hp_factory, direct_fn, direct_kwargs_factory)
CASES = []

def case(name, direct_fn, direct_kwargs_fn, df_factory=lambda: _df(), hp_factory=lambda: _hp()):
    CASES.append((name, df_factory, hp_factory, direct_fn, direct_kwargs_fn))


# SLO-based (5)
case('linear_simple',
     rewards.calculate_rewards_simple,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_slo=hp['TTFT_SLO'], avg_tpot_slo=hp['AVG_TPOT_SLO'],
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('linear_simple_extended',
     rewards.calculate_rewards_simple_extended,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_slo=hp['TTFT_SLO'], avg_tpot_slo=hp['AVG_TPOT_SLO'],
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('piecewise_linear_steeper_gradient',
     rewards.calculate_rewards_piecewise_linear_steeper_gradient,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_slo=hp['TTFT_SLO'], avg_tpot_slo=hp['AVG_TPOT_SLO'],
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('inverse_latency',
     rewards.calculate_rewards_inverse_latency,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_slo=hp['TTFT_SLO'], avg_tpot_slo=hp['AVG_TPOT_SLO'],
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('latency_optimized',
     rewards.calculate_rewards_latency_optimization,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_slo=hp['TTFT_SLO'], avg_tpot_slo=hp['AVG_TPOT_SLO'],
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))

# Latency-only (4)
case('simple_latency_minimization',
     rewards.calculate_rewards_simple_latency_minimization,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('negative_reciprocal',
     rewards.calculate_rewards_negative_reciprocal,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('negative_linear',
     rewards.calculate_rewards_negative_linear,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('negative_squared',
     rewards.calculate_rewards_negative_squared,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))

# Special cases (5)
case('quantile_based',
     rewards.calculate_rewards_quantile_based,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         input_tokens=df['input_tokens'].values, output_tokens=df['output_tokens'].values,
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('throughput_based',
     rewards.calculate_rewards_throughput_based,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         input_tokens=df['input_tokens'].values,
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('absolute_latency',
     rewards.calculate_rewards_absolute_latency,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_slo=hp.get('TTFT_SLO', 15000), tpot_slo=hp.get('AVG_TPOT_SLO', 100),
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
case('quantile_advantage',
     rewards.calculate_rewards_quantile_advantage,
     lambda df, hp: dict(latency_values=df['ttft'].values, input_tokens=df['input_tokens'].values,
                         num_buckets=5))
case('log_normalized',
     rewards.calculate_rewards_log_normalized,
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_p99=float(df['ttft'].quantile(0.99)),
                         tpot_p99=float(df['avg_tpot'].quantile(0.99)),
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
# context_aware: only fallback path is exercised (needs kv_cache_hit_ratios that's
# not in the dispatcher's df). The fallback target is latency_optimization.
case('context_aware',
     rewards.calculate_rewards_latency_optimization,  # fallback target
     lambda df, hp: dict(ttft_values=df['ttft'].values, tpot_values=df['avg_tpot'].values,
                         ttft_slo=hp['TTFT_SLO'], avg_tpot_slo=hp['AVG_TPOT_SLO'],
                         ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))


def run():
    n_pass = 0
    n_fail = 0
    failures = []

    # === Part A: normal call paths (df has all expected columns) ===
    for name, df_f, hp_f, direct_fn, direct_kwargs_fn in CASES:
        df = df_f()
        hp = hp_f()
        dk = direct_kwargs_fn(df, hp)
        ok, msg = _check(name,
                         dispatcher_kwargs=dict(reward_function=name, df=df, hyperparameters=hp),
                         direct_fn=direct_fn, direct_kwargs=dk)
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((name, msg))
            print(f"  FAIL {name}: {msg}")

    # === Part B: fallback paths (df missing optional columns) ===
    # quantile_based -> latency_optimized when input_tokens/output_tokens missing
    df_no_tokens = _df(with_tokens=False)
    hp = _hp()
    ok, msg = _check('quantile_based [fallback]',
        dispatcher_kwargs=dict(reward_function='quantile_based', df=df_no_tokens, hyperparameters=hp),
        direct_fn=rewards.calculate_rewards_latency_optimization,
        direct_kwargs=dict(ttft_values=df_no_tokens['ttft'].values,
                           tpot_values=df_no_tokens['avg_tpot'].values,
                           ttft_slo=hp['TTFT_SLO'], avg_tpot_slo=hp['AVG_TPOT_SLO'],
                           ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
    if ok: n_pass += 1
    else:  n_fail += 1; failures.append(('quantile_based-fallback', msg)); print(f"  FAIL quantile_based-fallback: {msg}")

    # throughput_based -> simple_latency_minimization when input_tokens missing
    ok, msg = _check('throughput_based [fallback]',
        dispatcher_kwargs=dict(reward_function='throughput_based', df=df_no_tokens, hyperparameters=hp),
        direct_fn=rewards.calculate_rewards_simple_latency_minimization,
        direct_kwargs=dict(ttft_values=df_no_tokens['ttft'].values,
                           tpot_values=df_no_tokens['avg_tpot'].values,
                           ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
    if ok: n_pass += 1
    else:  n_fail += 1; failures.append(('throughput_based-fallback', msg)); print(f"  FAIL throughput_based-fallback: {msg}")

    # quantile_advantage -> negative_linear when input_tokens missing
    ok, msg = _check('quantile_advantage [fallback]',
        dispatcher_kwargs=dict(reward_function='quantile_advantage', df=df_no_tokens, hyperparameters=hp),
        direct_fn=rewards.calculate_rewards_negative_linear,
        direct_kwargs=dict(ttft_values=df_no_tokens['ttft'].values,
                           tpot_values=df_no_tokens['avg_tpot'].values,
                           ttft_reward_weight=hp['TTFT_REWARD_WEIGHT']))
    if ok: n_pass += 1
    else:  n_fail += 1; failures.append(('quantile_advantage-fallback', msg)); print(f"  FAIL quantile_advantage-fallback: {msg}")

    # === Part C: unknown name raises ===
    try:
        rewards.compute_rewards('definitely_not_a_real_reward_function', _df(), _hp())
        n_fail += 1
        print("  FAIL unknown-name-should-raise: did not raise")
    except ValueError:
        n_pass += 1

    # === Part D: KNOWN_REWARD_FUNCTIONS sanity ===
    expected_names = {n for n, _, _, _, _ in CASES}
    if expected_names != rewards.KNOWN_REWARD_FUNCTIONS:
        missing = expected_names - rewards.KNOWN_REWARD_FUNCTIONS
        extra = rewards.KNOWN_REWARD_FUNCTIONS - expected_names
        print(f"  FAIL KNOWN_REWARD_FUNCTIONS mismatch — missing={missing}, extra={extra}")
        n_fail += 1
    else:
        n_pass += 1

    print(f"\n{n_pass} pass / {n_fail} fail")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
