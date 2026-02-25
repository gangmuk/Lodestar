#!/usr/bin/env python3
"""
Copy specific files from each final_model-* directory on the routing-agent-service pod
to the local filesystem, preserving the directory tree structure.

Uses two copy strategies:
  1. kubectl cp (fast, but fails on larger files over flaky connections)
  2. kubectl exec cat > local (streaming fallback, more reliable for large files)

After each copy, verifies file size matches the remote.

Usage:
    python copy_final_model_files.py
    python copy_final_model_files.py --pod <pod-name> --namespace <ns>
    python copy_final_model_files.py --dry-run
    python copy_final_model_files.py --retry-failed   # only re-copy files that failed or are missing/wrong-size
"""

import argparse
import json
import os
import subprocess
import sys
import time

REMOTE_BASE = "/app/NVIDIA-A30"

# All final_model dirs to copy, relative to REMOTE_BASE
FINAL_MODEL_DIRS = [
    # llama-3-8b-instruct / maxTokens_100-maxTokensStd_10
    # "llama-3-8b-instruct/maxTokens_100-maxTokensStd_10/final_model-latency_predictor",
    # llama-3-8b-instruct / maxTokens_1-maxTokensStd_0 (top-level)
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear",
    # llama-3-8b-instruct / maxTokens_1-maxTokensStd_0 / azure
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear",
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/azure/azure_code_poisson/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear",
    # llama-3-8b-instruct / maxTokens_1-maxTokensStd_0 / gangmuk-prefix
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear",
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_SharingRatio47%_30000",
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_SharingRatio47%_15000",
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_SharingRatio9%",
    # llama-3-8b-instruct / maxTokens_1-maxTokensStd_0 / mooncake
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear",
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_toolagent_2",
    "llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_conversation_2",
    # llama-3-8b-instruct / use_given_output_length / azure
    "llama-3-8b-instruct/use_given_output_length/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_without_output_tokens",
    # llama-3-8b-instruct / use_given_output_length / gangmuk-prefix
    "llama-3-8b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear",
    "llama-3-8b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear",
    "llama-3-8b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_without_output_tokens",
    # llama-3-8b-instruct / use_given_output_length / mooncake
    "llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear",
    "llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear",
    "llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_conversation_2",
    "llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2",
    "llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_toolagent_2",
    "llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2",
    "llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_without_output_tokens",
    # qwen3-4b-instruct / use_given_output_length / gangmuk-prefix
    "qwen3-4b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear",
    "qwen3-4b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear",
    # qwen3-4b-instruct / use_given_output_length / mooncake
    "qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_conversation_2",
    "qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2",
    "qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_toolagent_2",
    "qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2",
]

# Files to copy from each final_model dir
FILES_TO_COPY = [
    "model_config.json",
    "data_processor_command.txt",
    "data_processor.log.txt",
    "dataset_analyzer.log.txt",
    "python_command.txt",
    "feature_normalization_statistics.csv",
    "feature_distribution_statistics.csv",
    "reward_net.pth",
    "optimizer.pth",
    "metadata.pkl",
    "training_metrics-False.csv",
    "comprehensive_neural_cb_metrics-False.pdf",
    "output.txt",
    "offline_routing_agent.log.txt",
    "full_path.txt",
    "data.csv",
    "data-processed_summary.json",
    "data-processed.csv",
    "data-processed-sampled.csv",
    "dataset_analysis.pdf",
]

COPY_TIMEOUT = 600  # 10 minutes per file


def get_pod_name(namespace, app_label):
    """Auto-discover the pod name from the app label."""
    cmd = [
        "kubectl", "-n", namespace,
        "get", "pods", "-l", f"app={app_label}",
        "-o", "jsonpath={.items[*].metadata.name}",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except subprocess.CalledProcessError as e:
        print(f"Failed to get pods: {e}")
        sys.exit(1)

    pods = [p for p in out.split() if p]
    if not pods:
        print("No running pods found.")
        sys.exit(1)
    if len(pods) > 1:
        print(f"Multiple pods found: {pods}. Use --pod to specify one.")
        sys.exit(1)
    return pods[0]


def get_remote_size(namespace, pod, remote_path):
    """Get file size on the pod. Returns size in bytes or -1 on failure."""
    cmd = [
        "kubectl", "-n", namespace, "exec", pod, "--",
        "stat", "-c", "%s", remote_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30).strip()
        return int(out)
    except Exception:
        return -1


def get_remote_sizes_batch(namespace, pod, remote_dir, filenames):
    """Get sizes of multiple files in one kubectl exec call. Returns {filename: size}."""
    # Build a stat command for all files at once
    stat_args = " ".join(f'"{f}"' for f in filenames)
    script = f'cd "{remote_dir}" && for f in {stat_args}; do echo "$f $(stat -c %s "$f" 2>/dev/null || echo -1)"; done'
    cmd = ["kubectl", "-n", namespace, "exec", pod, "--", "sh", "-c", script]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=60).strip()
        sizes = {}
        for line in out.splitlines():
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                sizes[parts[0]] = int(parts[1])
        return sizes
    except Exception:
        return {}


def copy_via_kubectl_cp(namespace, pod, remote_path, local_path):
    """Strategy 1: kubectl cp (uses tar internally)."""
    cmd = ["kubectl", "-n", namespace, "cp", f"{pod}:{remote_path}", local_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=COPY_TIMEOUT)


def copy_via_exec_cat(namespace, pod, remote_path, local_path):
    """Strategy 2: kubectl exec cat > file (direct streaming, bypasses tar)."""
    cmd = [
        "kubectl", "-n", namespace, "exec", pod, "--",
        "cat", remote_path,
    ]
    with open(local_path, "wb") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=COPY_TIMEOUT)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=proc.stderr)


CHUNK_SIZE = 1024 * 1024  # 1MB chunks


def copy_via_chunked(namespace, pod, remote_path, local_path, expected_size=-1):
    """Strategy 3: read file in chunks via dd, verify each chunk, concatenate locally.
    Resilient to network drops that truncate large streams."""
    if expected_size <= 0:
        # Need to know the size for chunking
        cmd = [
            "kubectl", "-n", namespace, "exec", pod, "--",
            "stat", "-c", "%s", remote_path,
        ]
        out = subprocess.check_output(cmd, text=True, timeout=30).strip()
        expected_size = int(out)

    total_chunks = (expected_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    with open(local_path, "wb") as out_f:
        for chunk_idx in range(total_chunks):
            offset = chunk_idx * CHUNK_SIZE
            remaining = expected_size - offset
            this_chunk = min(CHUNK_SIZE, remaining)

            # dd with bs=CHUNK_SIZE, skip=chunk_idx (in blocks), count=1.
            # Last chunk naturally returns fewer bytes — dd stops at EOF.
            dd_cmd = f"dd if='{remote_path}' bs={CHUNK_SIZE} skip={chunk_idx} count=1 2>/dev/null"
            cmd = [
                "kubectl", "-n", namespace, "exec", pod, "--",
                "sh", "-c", dd_cmd,
            ]

            chunk_ok = False
            for retry in range(5):
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                if proc.returncode != 0:
                    continue
                data = proc.stdout
                if len(data) == this_chunk:
                    out_f.write(data)
                    chunk_ok = True
                    pct = (chunk_idx + 1) * 100 // total_chunks
                    print(f"\r    [chunked dd] {chunk_idx + 1}/{total_chunks} chunks ({pct}%)", end="", flush=True)
                    break
                time.sleep(1)

            if not chunk_ok:
                print()  # newline after progress
                raise ValueError(
                    f"chunk {chunk_idx + 1}/{total_chunks} failed (offset={offset}, expected={this_chunk})"
                )

    print()  # newline after progress
    # Final size check
    local_size = os.path.getsize(local_path)
    if local_size != expected_size:
        raise ValueError(f"final size mismatch: local={local_size} remote={expected_size}")


def copy_file(namespace, pod, remote_path, local_path, expected_size, max_retries=3):
    """Try to copy a file using multiple strategies with retries and size verification."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    strategies = [
        ("chunked dd", lambda: copy_via_chunked(namespace, pod, remote_path, local_path, expected_size)),
        ("kubectl cp", lambda: copy_via_kubectl_cp(namespace, pod, remote_path, local_path)),
        ("exec cat",   lambda: copy_via_exec_cat(namespace, pod, remote_path, local_path)),
    ]

    for strat_name, strat_fn in strategies:
        for attempt in range(1, max_retries + 1):
            try:
                strat_fn()

                # Verify size (chunked does its own check, but double-check here too)
                if expected_size > 0:
                    local_size = os.path.getsize(local_path)
                    if local_size != expected_size:
                        raise ValueError(
                            f"size mismatch: local={local_size} remote={expected_size}"
                        )
                return True

            except Exception as e:
                err_msg = str(e).split("\n")[0][:120]
                print(f"    [{strat_name} attempt {attempt}/{max_retries}] {err_msg}")
                # Remove partial file
                if os.path.exists(local_path):
                    os.remove(local_path)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

    return False


def needs_copy(local_path, expected_size):
    """Check if a file needs to be (re)copied."""
    if not os.path.exists(local_path):
        return True
    if expected_size > 0 and os.path.getsize(local_path) != expected_size:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Copy final_model files from pod to local")
    parser.add_argument("--pod", type=str, default=None, help="Pod name (auto-detected if not set)")
    parser.add_argument("--namespace", type=str, default="default")
    parser.add_argument("--app-label", type=str, default="routing-agent-service")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without copying")
    parser.add_argument("--retry-failed", action="store_true", default=True,
                        help="Only copy files that are missing or have wrong size (default: True)")
    args = parser.parse_args()

    pod = args.pod or get_pod_name(args.namespace, args.app_label)
    local_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NVIDIA-A30")

    total_files = len(FINAL_MODEL_DIRS) * len(FILES_TO_COPY)
    print(f"Pod: {pod}")
    print(f"Remote base: {REMOTE_BASE}")
    print(f"Local base: {local_base}")
    print(f"Directories: {len(FINAL_MODEL_DIRS)}")
    print(f"Files per directory: {len(FILES_TO_COPY)}")
    print(f"Total files: {total_files}")
    if args.retry_failed:
        print("Mode: retry-failed (skipping files that already exist with correct size)")
    print("=" * 80)

    if args.dry_run:
        for model_dir in FINAL_MODEL_DIRS:
            for fname in FILES_TO_COPY:
                remote = f"{REMOTE_BASE}/{model_dir}/{fname}"
                local = os.path.join(local_base, model_dir, fname)
                print(f"  {remote} -> {local}")
        print(f"\nDry run complete. {total_files} files would be copied.")
        return

    success = 0
    fail = 0
    skipped = 0
    failed_files = []

    for i, model_dir in enumerate(FINAL_MODEL_DIRS, 1):
        print(f"\n[{i}/{len(FINAL_MODEL_DIRS)}] {model_dir}")

        # Batch-fetch remote sizes for this directory
        remote_dir = f"{REMOTE_BASE}/{model_dir}"
        remote_sizes = get_remote_sizes_batch(args.namespace, pod, remote_dir, FILES_TO_COPY)

        for fname in FILES_TO_COPY:
            remote = f"{REMOTE_BASE}/{model_dir}/{fname}"
            local = os.path.join(local_base, model_dir, fname)
            expected_size = remote_sizes.get(fname, -1)

            # In retry-failed mode, skip files that are already OK
            if args.retry_failed and not needs_copy(local, expected_size):
                skipped += 1
                continue

            size_str = f" ({expected_size / 1024 / 1024:.1f}MB)" if expected_size > 1024 * 1024 else ""
            print(f"  {fname}{size_str} ... ", end="", flush=True)

            if copy_file(args.namespace, pod, remote, local, expected_size):
                print("OK")
                success += 1
            else:
                print("FAILED")
                fail += 1
                failed_files.append(remote)
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print(f"Done. {success} succeeded, {fail} failed, {skipped} skipped.")
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
