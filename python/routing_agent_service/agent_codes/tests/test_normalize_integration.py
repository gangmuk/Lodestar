"""
Integration test for the normalize -> rewards-dispatch path.

Calls data_normalizer.normalize_processed_data on a fixture CSV that came out
of a real production run (the rps11-benchmark logs). Captures hashes of the
output reward columns and compares against a golden snapshot.

This exercises:
  - data_normalizer.normalize_processed_data (the full pipeline)
  - rewards.compute_rewards (the new dispatcher)
  - Statistics computation + feature normalization
  - The actual reward function used in production (negative_linear)

The fixture CSV is just 5 rows so the test is fast.

Usage:
    python tests/test_normalize_integration.py            # compare
    python tests/test_normalize_integration.py --refresh  # write golden
"""
import sys
import os
import json
import hashlib
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE_CSV = os.path.join(os.path.dirname(__file__), "fixtures", "processed_csv_first5rows.csv")
FIXTURE_HP = os.path.join(os.path.dirname(__file__), "fixtures", "model_config.json")
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_normalize.json")


def _col_digest(series):
    arr = np.asarray(series, dtype=np.float64)
    rounded = np.round(arr, 6)  # ~1e-6 tolerance for float rounding identity
    return {
        "n": len(arr),
        "sum": round(float(rounded.sum()), 6),
        "min": round(float(rounded.min()), 6) if len(arr) else None,
        "max": round(float(rounded.max()), 6) if len(arr) else None,
        "sha": hashlib.sha256(rounded.tobytes()).hexdigest()[:16],
    }


def run():
    refresh = "--refresh" in sys.argv

    # Import LATE so test failures from data_normalizer surface clearly.
    import data_normalizer
    import rewards

    with open(FIXTURE_HP) as f:
        hyperparameters = json.load(f)

    # Sanity-check fixtures.
    assert os.path.exists(FIXTURE_CSV), f"missing fixture: {FIXTURE_CSV}"
    df_in = pd.read_csv(FIXTURE_CSV)
    print(f"Loaded fixture: {len(df_in)} rows, {len(df_in.columns)} cols")
    print(f"Hyperparameters: REWARD_FUNCTION={hyperparameters.get('REWARD_FUNCTION')}, "
          f"TTFT_SLO={hyperparameters.get('TTFT_SLO')}, "
          f"AVG_TPOT_SLO={hyperparameters.get('AVG_TPOT_SLO')}")

    reward_function = hyperparameters.get('REWARD_FUNCTION', 'negative_linear')

    # Run the actual production code path.
    with tempfile.TemporaryDirectory() as tmp:
        stats_file = os.path.join(tmp, "stats.pkl")
        normalized_df, stats_instance, summary = data_normalizer.normalize_processed_data(
            processed_csv_file=FIXTURE_CSV,
            output_csv_file=os.path.join(tmp, "normalized.csv"),
            reward_function=reward_function,
            stats_file=stats_file,
            hyperparameters=hyperparameters,
        )

    print(f"Normalized DataFrame: {len(normalized_df)} rows, {len(normalized_df.columns)} cols")

    # Capture hashes of the key reward columns + a few normalized columns to
    # detect changes in either reward dispatch or normalization stats.
    columns_to_hash = ['ttft_reward', 'tpot_reward', 'reward']
    # Also hash a couple of normalized columns if present (per-pod, all pods)
    for col in normalized_df.columns:
        if col.endswith('-kv_hit_ratio_normalized') or col.endswith('-inflight_requests_normalized'):
            columns_to_hash.append(col)
            if len(columns_to_hash) > 10:
                break

    digests = {}
    for col in columns_to_hash:
        if col not in normalized_df.columns:
            digests[col] = "MISSING"
        else:
            digests[col] = _col_digest(normalized_df[col])

    digests["_meta"] = {
        "input_rows": len(df_in),
        "output_rows": len(normalized_df),
        "output_cols": len(normalized_df.columns),
        "reward_function": reward_function,
    }

    if refresh:
        with open(GOLDEN_PATH, "w") as f:
            json.dump(digests, f, indent=2, sort_keys=True)
        print(f"\nWrote golden: {GOLDEN_PATH}")
        for c in sorted(digests):
            print(f"  {c}: {digests[c]}")
        return 0

    if not os.path.exists(GOLDEN_PATH):
        print(f"\nNo golden at {GOLDEN_PATH}. Run with --refresh.")
        return 2

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    n_pass = 0
    n_fail = 0
    for col in sorted(set(digests) | set(golden)):
        cur = digests.get(col)
        ref = golden.get(col)
        if cur == ref:
            n_pass += 1
        else:
            n_fail += 1
            print(f"  FAIL {col}: ref={ref}  cur={cur}")
    print(f"\n{n_pass} pass / {n_fail} fail")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
