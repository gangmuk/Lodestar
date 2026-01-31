#!/usr/bin/env python3
import argparse
import csv
import pathlib
import re
import statistics
from typing import Dict, Iterable, List, Tuple


OVERHEAD_RE = re.compile(r'overhead_log: oh, (.*)')


def parse_overhead_entries(text: str) -> List[Dict[str, float]]:
    entries: List[Dict[str, float]] = []
    for match in OVERHEAD_RE.finditer(text):
        line = match.group(1)
        parts = [p.strip() for p in line.split(',')]
        values: Dict[str, float] = {}
        for part in parts:
            if ': ' not in part:
                continue
            key, value = part.split(': ', 1)
            if value.endswith('ms'):
                try:
                    values[key] = float(value[:-2])
                except ValueError:
                    continue
        if values:
            entries.append(values)
    return entries


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float('nan')
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def summarize(entries: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    all_keys = sorted({k for e in entries for k in e.keys()})
    summary: Dict[str, Dict[str, float]] = {}
    for key in all_keys:
        vals = [e[key] for e in entries if key in e]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        summary[key] = {
            "count": float(len(vals)),
            "mean": statistics.mean(vals),
            "p50": statistics.median(vals_sorted),
            "p90": percentile(vals_sorted, 0.90),
            "p99": percentile(vals_sorted, 0.99),
            "max": max(vals_sorted),
        }
    return summary


def write_csv(summary: Dict[str, Dict[str, float]], output_path: pathlib.Path) -> None:
    fieldnames = ["metric", "count", "mean", "p50", "p90", "p99", "max"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, stats in summary.items():
            row = {"metric": key}
            row.update({k: stats[k] for k in fieldnames[1:]})
            writer.writerow(row)


def print_summary(summary: Dict[str, Dict[str, float]], focus: List[str]) -> None:
    print("Average metrics (ms):")
    for key in focus:
        if key in summary:
            s = summary[key]
            print(
                f"- {key}: mean={s['mean']:.1f} "
                f"p50={s['p50']:.0f} p90={s['p90']:.0f} "
                f"p99={s['p99']:.0f} max={s['max']:.0f}"
            )

    flat: List[Tuple[str, Dict[str, float]]] = list(summary.items())
    flat.sort(key=lambda item: item[1]["mean"], reverse=True)
    print("\nTop 10 by mean:")
    for key, s in flat[:10]:
        print(f"- {key}: mean={s['mean']:.1f} p99={s['p99']:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse overhead_log entries and summarize metrics.")
    parser.add_argument("log_path", type=pathlib.Path, help="Path to routing-agent service log file")
    parser.add_argument("--csv", type=pathlib.Path, default=None, help="Optional CSV output path")
    parser.add_argument(
        "--focus",
        type=str,
        default=(
            "handle_infer_end_to_end,"
            "handle_infer_preprocess_overhead,"
            "handle_infer_normalize,"
            "handle_infer_encode,"
            "handle_infer_contextual_bandit_write_lock,"
            "handle_infer_contextual_bandit_create,"
            "handle_infer_contextual_bandit_infer,"
            "handle_infer_calling_infer_from_tensor,"
            "encode_prepare_for_encoding,"
            "infer_from_tensor_tensor_transfer,"
            "infer_from_tensor_inference,"
            "infer_from_tensor_result_formatting,"
            "infer_from_tensor_total_inference"
        ),
        help="Comma-separated list of metrics to show first",
    )
    args = parser.parse_args()

    text = args.log_path.read_text(errors="ignore")
    entries = parse_overhead_entries(text)
    print(f"requests_with_overhead: {len(entries)}")
    summary = summarize(entries)

    focus = [f.strip() for f in args.focus.split(",") if f.strip()]
    print_summary(summary, focus)

    if args.csv:
        write_csv(summary, args.csv)
        print(f"\nWrote CSV to: {args.csv}")


if __name__ == "__main__":
    main()



