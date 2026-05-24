"""
Regression test for preprocess.main on the TRAINING path (input_file mode).

This is the path that `handle_flush` exercises:
    preprocess.main(input_file=path, log_message="", hyperparameters=...)

That code path goes through `preprocess_data_unified`, which contains an elif
chain that dispatches to `calculate_rewards_*` functions by name. When the
reward functions were extracted to rewards.py during the pre-release cleanup,
those bare-name calls became dangling references — a NameError surfaced only
when /flush received its first batch in production.

This test exercises the training path explicitly so that bug class can't sneak
past again.

Usage:
    python tests/test_preprocess_training.py
    python tests/test_preprocess_training.py --refresh
"""
import sys
import os
import json
import hashlib
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_latency_metrics_5lines.txt")
HP_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "model_config.json")
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_preprocess_training.json")


def _col_digest(values):
    """Stable hash of a column of values (list, np.array, or scalar)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    rounded = np.round(arr, 6)
    return {
        "n": int(arr.size),
        "sum": round(float(rounded.sum()), 6),
        "min": round(float(rounded.min()), 6),
        "max": round(float(rounded.max()), 6),
        "sha": hashlib.sha256(rounded.tobytes()).hexdigest()[:16],
    }


def run():
    refresh = "--refresh" in sys.argv

    import preprocess

    with open(HP_FIXTURE) as f:
        hyperparameters = json.load(f)

    # parse_log_file expects lines containing "latency_metrics" — the fixture
    # already has 5 klog-prefixed `**@latency_metrics@...` lines.
    with open(LOG_FIXTURE) as f:
        log_content = f.read()

    with tempfile.TemporaryDirectory() as tmp:
        input_file = os.path.join(tmp, "training_input.csv")
        with open(input_file, "w") as out:
            out.write(log_content)

        # TRAINING PATH: input_file is provided, log_message is "".
        # This is what handle_flush calls — and what was crashing.
        result, sorted_pods, overhead = preprocess.main(
            input_file=input_file,
            log_message="",
            hyperparameters=hyperparameters,
            pod_ip_mapping=None,
        )

    # In training mode the result is a DataFrame, not a dict.
    print(f"Training path returned: type={type(result).__name__}, rows={len(result)}")
    print(f"Reward function used: {hyperparameters.get('REWARD_FUNCTION')}")
    print(f"Sorted pods: {sorted(sorted_pods)}")

    # Critical: the elif chain in preprocess_data_unified must have produced
    # reward columns. If the NameError reappears, we won't get this far.
    required_columns = ['ttft_reward', 'tpot_reward', 'reward',
                        'avg_tpot_slo_satisfied', 'avg_ttft_slo_satisfied']
    missing = [c for c in required_columns if c not in result.columns]
    if missing:
        raise AssertionError(f"Training path output missing required columns: {missing}. "
                             f"Got columns: {sorted(result.columns)[:10]}...")

    digests = {col: _col_digest(result[col]) for col in required_columns}
    digests["_meta"] = {
        "n_rows": int(len(result)),
        "n_cols": int(len(result.columns)),
        "reward_function": hyperparameters.get('REWARD_FUNCTION'),
        "sorted_pods": sorted(sorted_pods),
    }

    if refresh:
        with open(GOLDEN_PATH, "w") as f:
            json.dump(digests, f, indent=2, sort_keys=True)
        print(f"\nWrote golden: {GOLDEN_PATH}")
        for k in sorted(digests):
            print(f"  {k}: {digests[k]}")
        return 0

    if not os.path.exists(GOLDEN_PATH):
        print(f"No golden at {GOLDEN_PATH}. Run with --refresh.")
        return 2

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    n_pass = 0
    n_fail = 0
    for key in sorted(set(digests) | set(golden)):
        cur = digests.get(key)
        ref = golden.get(key)
        if cur == ref:
            n_pass += 1
        else:
            n_fail += 1
            print(f"  FAIL {key}: ref={ref}  cur={cur}")
    print(f"\n{n_pass} pass / {n_fail} fail")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
