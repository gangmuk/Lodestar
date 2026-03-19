#!/usr/bin/env python3
"""
Replay a workload through the updated Go prefix indexer and save:
  - CSV (per-request hit ratios)
  - PNG (time-series plot)
  - PDF (time-series plot)

Usage:
  python3 run_prefix_hit_ratio_replay.py /path/to/workload.jsonl
  python3 run_prefix_hit_ratio_replay.py /path/to/workload.jsonl --output-dir /tmp/out
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay workload prefix hit ratio with updated indexer."
    )
    parser.add_argument(
        "workload_jsonl",
        help="Path to workload.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs (default: workload file directory)",
    )
    parser.add_argument(
        "--model-key",
        default="replay-model",
        help="Model key passed to prefix indexer replay (default: replay-model)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=100,
        help="Rolling window size for trend line (default: 100)",
    )
    return parser.parse_args()


def run_go_replay(repo_root: Path, workload_path: Path, out_csv: Path, model_key: str) -> None:
    go_script = repo_root / "benchmarks/data-driven-routing/routing-agent-service/vke_files/replay_prefix_hit_ratio.go"
    cmd = [
        "go",
        "run",
        str(go_script),
        "--workload",
        str(workload_path),
        "--out_csv",
        str(out_csv),
        "--model",
        model_key,
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def plot_timeseries(df: pd.DataFrame, out_png: Path, out_pdf: Path, rolling_window: int, title_suffix: str) -> None:
    x = df["timestamp_ms"] / 1000.0
    y = df["selected_pod_hit_ratio_before_add"]
    rolling = y.rolling(window=max(1, rolling_window), min_periods=1).mean()

    plt.figure(figsize=(16, 6))
    plt.scatter(x, y, s=5, alpha=0.25, label="Per-request ratio")
    plt.plot(x, rolling, color="red", linewidth=2, label=f"Rolling mean (window={rolling_window})")
    plt.ylim(-1, 101)
    plt.xlabel("Workload timestamp (s)")
    plt.ylabel("Prefix hit ratio before add (%)")
    plt.title(f"Updated Prefix Indexer Replay ({title_suffix})")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right")

    stats = (
        f"mean={y.mean():.2f}%, p50={y.quantile(0.5):.1f}%, p90={y.quantile(0.9):.1f}%, "
        f"p99={y.quantile(0.99):.1f}%, pct100={(y.eq(100).mean() * 100):.3f}%"
    )
    plt.text(
        0.01,
        0.02,
        stats,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()

    workload_path = Path(args.workload_jsonl).expanduser().resolve()
    if not workload_path.exists():
        print(f"Error: workload file not found: {workload_path}", file=sys.stderr)
        return 1

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]

    output_dir = Path(args.output_dir).resolve() if args.output_dir else workload_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    workload_stem = workload_path.stem
    base_name = f"prefix_hit_ratio_replay_updated_singlepod-{workload_stem}"

    out_csv = output_dir / f"{base_name}.csv"
    out_png = output_dir / f"{base_name}_timeseries.png"
    out_pdf = output_dir / f"{base_name}_timeseries.pdf"

    run_go_replay(repo_root=repo_root, workload_path=workload_path, out_csv=out_csv, model_key=args.model_key)

    df = pd.read_csv(out_csv)
    if df.empty:
        print(f"Error: replay output CSV is empty: {out_csv}", file=sys.stderr)
        return 1

    title_suffix = workload_path.name
    plot_timeseries(
        df=df,
        out_png=out_png,
        out_pdf=out_pdf,
        rolling_window=args.rolling_window,
        title_suffix=title_suffix,
    )

    print(f"Saved CSV: {out_csv}")
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

