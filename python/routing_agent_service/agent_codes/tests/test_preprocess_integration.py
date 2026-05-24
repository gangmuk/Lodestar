"""
Integration test for preprocess.main on real log lines.

Loads 5 real `**@latency_metrics@...` log lines (captured from the
rps11-benchmark run) and runs them through preprocess.main() via the
single-row inference path. Captures hashes of the output dict and compares
against a golden snapshot.

Exercises:
  - parse_log_message / _parse_log_to_dict (log string -> dict)
  - preprocess_inference_fast (dict -> processed dict)
  - preprocess.main routing (inference path)

This is the regression net for the "move preprocess.main to top" refactor and
for any future refactors of preprocess.py.

Usage:
    python tests/test_preprocess_integration.py
    python tests/test_preprocess_integration.py --refresh
"""
import sys
import os
import json
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_latency_metrics_5lines.txt")
HP_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "model_config.json")
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_preprocess.json")


def _digest_value(v):
    """Stable string-ish representation for hashing."""
    if isinstance(v, dict):
        return {k: _digest_value(vv) for k, vv in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_digest_value(x) for x in v]
    if isinstance(v, float):
        return round(v, 6)
    if hasattr(v, "tolist"):  # numpy array
        return [round(float(x), 6) for x in v.tolist()]
    return v


def _digest_dict(d):
    """Compact hash of an output dict, robust to float jitter."""
    canonical = json.dumps(_digest_value(d), sort_keys=True, default=str)
    return {
        "sha": hashlib.sha256(canonical.encode()).hexdigest()[:16],
        "n_keys": len(d) if isinstance(d, dict) else None,
    }


def run():
    refresh = "--refresh" in sys.argv

    # Late imports so we get clear errors if preprocess.py is broken.
    import preprocess

    with open(HP_FIXTURE) as f:
        hyperparameters = json.load(f)

    # The log lines have a klog prefix (`I0401 00:02:33.440092 ...`) before the
    # `**@latency_metrics@...` payload. preprocess.main expects just the payload
    # starting at "latency_metrics@" or with the "**@" marker. Extract that.
    with open(LOG_FIXTURE) as f:
        log_lines = [line.strip() for line in f if line.strip()]

    # Strip the klog prefix; keep from "**@" onward.
    messages = []
    for line in log_lines:
        idx = line.find("**@")
        if idx == -1:
            raise AssertionError(f"line missing '**@' marker: {line[:120]}")
        messages.append(line[idx:])

    print(f"Loaded {len(messages)} real log messages")
    print(f"Sample message length: {len(messages[0])} chars")

    digests = {}
    for i, msg in enumerate(messages):
        try:
            result, sorted_pods, overhead = preprocess.main(
                input_file=None,
                log_message=msg,
                hyperparameters=hyperparameters,
                pod_ip_mapping=None,
            )
        except Exception as e:
            print(f"  FAIL message {i}: raised {type(e).__name__}: {e}")
            digests[f"msg_{i}"] = {"error": f"{type(e).__name__}: {e}"}
            continue

        # Result is a dict (inference fast-path returns dict, not DataFrame)
        digests[f"msg_{i}_result"] = _digest_dict(result) if isinstance(result, dict) else {"type": type(result).__name__}
        digests[f"msg_{i}_pods"] = sorted(sorted_pods) if isinstance(sorted_pods, list) else str(sorted_pods)

    digests["_meta"] = {
        "n_messages": len(messages),
        "hyperparameters_keys": sorted(hyperparameters.keys()),
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
