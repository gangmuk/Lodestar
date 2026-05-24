#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict

FEATURE_KEYS = [
    "allPodsKvCacheHitRatios",
    "numInflightRequestsAllPods",
    "vllmGPUKVCacheUsage",
    "vllmCPUKVCacheUsage",
    "vllmNumRequestsRunning",
    "vllmNumRequestsWaiting",
    "numPrefillTokensForAllPods",
    "numDecodeTokensForAllPods",
]

def parse_line(line: str):
    if "latency_metrics@" not in line:
        return {}
    # Keep only the telemetry segment
    parts = line.split("latency_metrics@")[-1].strip().split("@")
    row = {}
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        val = parts[i + 1]
        # JSON maps
        if val and val[0] == "{" and val[-1] == "}":
            try:
                if '\\"' in val:
                    val = val.replace('\\"', '"')
                row[key] = json.loads(val)
            except Exception:
                row[key] = {}
        else:
            # Try int -> float -> string
            try:
                row[key] = int(val)
            except ValueError:
                try:
                    row[key] = float(val)
                except ValueError:
                    row[key] = val
        i += 2
    return row

def analyze(log_path: str, missing_threshold_pct: float = 20.0, top_k: int = 10):
    pods_first_seen = {}  # pod_ip -> first line index
    pods_seen_set = set()
    present_counts = {k: defaultdict(int) for k in FEATURE_KEYS}  # feature -> pod -> count
    empty_dict_counts = {k: 0 for k in FEATURE_KEYS}
    total_lines = 0

    with open(log_path, "r") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = parse_line(line)
            if not row:
                continue
            total_lines += 1

            # Collect per-line pod IDs from all feature dicts
            pods_in_line = set()
            for feat in FEATURE_KEYS:
                val = row.get(feat, {})
                if isinstance(val, dict):
                    if len(val) == 0:
                        empty_dict_counts[feat] += 1
                    for pod_ip in val.keys():
                        pods_in_line.add(pod_ip)
                        present_counts[feat][pod_ip] += 1

            # Include selectedpod if present (ensures we consider it part of the cluster)
            sel_pod = row.get("selectedpod")
            if isinstance(sel_pod, str) and sel_pod:
                pods_in_line.add(sel_pod)

            # Track first-seen index for pods
            for pod_ip in pods_in_line:
                pods_seen_set.add(pod_ip)
                if pod_ip not in pods_first_seen:
                    pods_first_seen[pod_ip] = idx

    if total_lines == 0:
        print("No telemetry lines found.")
        return

    print(f"Total telemetry lines: {total_lines}")
    print(f"Total unique pods seen: {len(pods_seen_set)}")

    # Estimate missing ratio if concatenating historical data without backfilling
    # For a pod that appears at line L, older rows (1..L-1) would have NaN in new columns after concat
    risky_pods = []
    for pod_ip in pods_seen_set:
        first_seen = pods_first_seen.get(pod_ip, total_lines + 1)
        missing_ratio = (first_seen - 1) / total_lines
        missing_pct = missing_ratio * 100.0
        risky_pods.append((missing_pct, pod_ip, first_seen))
    risky_pods.sort(reverse=True)

    print("\nPods with highest estimated missing% if concatenated without fill (top N):")
    for missing_pct, pod_ip, first_seen in risky_pods[:top_k]:
        flag = " <-- exceeds threshold" if missing_pct >= missing_threshold_pct else ""
        print(f"  {pod_ip}: first_seen_line={first_seen}, est_missing%={missing_pct:.1f}{flag}")

    # Feature-level presence rates (how often each pod appears in each map)
    print("\nPer-feature presence rates (pod appears in feature dict on X% of lines):")
    for feat in FEATURE_KEYS:
        rates = []
        counts = present_counts[feat]
        for pod_ip in pods_seen_set:
            c = counts.get(pod_ip, 0)
            rate = (c / total_lines) * 100.0
            rates.append((rate, pod_ip))
        rates.sort()  # ascending -> lowest presence first
        print(f"  {feat}:")
        for rate, pod_ip in rates[:min(top_k, len(rates))]:
            print(f"    {pod_ip}: {rate:.1f}%")

    # Empty-dict diagnostics
    print("\nEmpty-dict frequency by feature (lines where the JSON map was {}):")
    for feat in FEATURE_KEYS:
        pct = (empty_dict_counts[feat] / total_lines) * 100.0
        print(f"  {feat}: {empty_dict_counts[feat]} lines ({pct:.1f}%)")

    # Summary recommendation
    print("\nRecommendation:")
    print("- If any pod shows high est_missing%, its per-pod columns will be mostly NaN in older rows after concat.")
    print("- Either fill numeric NaNs to 0 before encoding, or drop columns whose missing% exceeds your threshold.")
    print("- Presence rates highlight sparse features that may rarely report per-pod values (still fine if backfilled).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="Path to latency_metrics log file")
    ap.add_argument("--threshold", type=float, default=20.0, help="Missing% threshold for flagging pods")
    ap.add_argument("--topk", type=int, default=10, help="Top-N entries to display")
    args = ap.parse_args()
    analyze(args.log_path, missing_threshold_pct=args.threshold, top_k=args.topk)