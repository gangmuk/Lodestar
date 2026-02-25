#!/usr/bin/env python3
"""
Copy workload files from the client-service pod back to the local filesystem,
matching the paths expected by the Dockerfile for rebuilding the image.

Uses the same copy strategies as copy_final_model_files.py:
  1. kubectl cp (fast, tar-based)
  2. kubectl exec cat > file (streaming fallback)

After each copy, verifies file size matches the remote.

Usage:
    python copy_workload_files.py
    python copy_workload_files.py --pod client-service-76d4c58cfd-c8f24
    python copy_workload_files.py --dry-run
    python copy_workload_files.py --retry-failed
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

REMOTE_BASE = "/app/workload"

# Local base is workload-and-experiment_results/ relative to the routing-agent-service dir.
# The Dockerfile COPYs from e.g. workload-and-experiment_results/gangmuk-prefix/SharingRatio71%/workload.jsonl
# to /app/workload/gangmuk-prefix/SharingRatio71%/workload.jsonl
# So we map: pod /app/workload/<path> -> local ../workload-and-experiment_results/<path>

# Workload entries: (remote_subdir, list_of_files)
# For single-file dirs we list the file; for directory copies (azure) we discover files dynamically.
WORKLOAD_ENTRIES = [
    # gangmuk-prefix workloads (single file each)
    ("gangmuk-prefix/SharingRatio71%", ["workload.jsonl"]),
    ("gangmuk-prefix/SharingRatio47%", ["workload.jsonl"]),
    ("gangmuk-prefix/SharingRatio28%", ["workload.jsonl"]),
    ("gangmuk-prefix/SharingRatio9%", ["workload.jsonl"]),
    ("gangmuk-prefix/MixedSharingRatio10_30_50_70%", ["workload.jsonl"]),
    # mooncake workloads (single file each)
    ("mooncake/conversation-1", ["workload.jsonl"]),
    ("mooncake/conversation-2", ["workload.jsonl"]),
    ("mooncake/toolagent-1", ["workload.jsonl"]),
    ("mooncake/toolagent-2", ["workload.jsonl"]),
    # azure workloads (entire directories — files discovered dynamically)
    ("azure/azure_code_poisson", None),
    ("azure/azure_code", None),
    ("azure/azure_conv-singleturn", None),
    ("azure/azure_conv-multiturn", None),
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


def discover_remote_files(namespace, pod, container, remote_dir):
    """List all files under a remote directory on the pod."""
    cmd = [
        "kubectl", "-n", namespace, "exec", pod, "-c", container, "--",
        "find", remote_dir, "-type", "f",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30).strip()
        if not out:
            return []
        # Return paths relative to remote_dir
        files = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(remote_dir):
                rel = line[len(remote_dir):].lstrip("/")
                if rel:
                    files.append(rel)
        return files
    except Exception as e:
        print(f"  Warning: failed to list files in {remote_dir}: {e}")
        return []


def get_remote_size(namespace, pod, container, remote_path):
    """Get file size on the pod. Returns size in bytes or -1 on failure."""
    cmd = [
        "kubectl", "-n", namespace, "exec", pod, "-c", container, "--",
        "stat", "-c", "%s", remote_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30).strip()
        return int(out)
    except Exception:
        return -1


def get_remote_sizes_batch(namespace, pod, container, remote_dir, filenames):
    """Get sizes of multiple files in one kubectl exec call. Returns {filename: size}."""
    stat_args = " ".join(f'"{f}"' for f in filenames)
    script = f'cd "{remote_dir}" && for f in {stat_args}; do echo "$f $(stat -c %s "$f" 2>/dev/null || echo -1)"; done'
    cmd = ["kubectl", "-n", namespace, "exec", pod, "-c", container, "--", "sh", "-c", script]
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


def get_remote_md5s_batch(namespace, pod, container, remote_dir, filenames):
    """Get MD5 checksums of multiple files in one kubectl exec call. Returns {filename: md5hex}."""
    md5_args = " ".join(f'"{f}"' for f in filenames)
    script = f'cd "{remote_dir}" && for f in {md5_args}; do md5sum "$f" 2>/dev/null || echo "FAIL $f"; done'
    cmd = ["kubectl", "-n", namespace, "exec", pod, "-c", container, "--", "sh", "-c", script]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=120).strip()
        md5s = {}
        for line in out.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0] != "FAIL":
                md5s[parts[1]] = parts[0]
        return md5s
    except Exception:
        return {}


def local_md5(filepath):
    """Compute MD5 checksum of a local file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_via_kubectl_cp(namespace, pod, container, remote_path, local_path):
    """Strategy 1: kubectl cp (uses tar internally)."""
    cmd = ["kubectl", "-n", namespace, "cp", f"{pod}:{remote_path}", local_path, "-c", container]
    subprocess.run(cmd, check=True, capture_output=True, timeout=COPY_TIMEOUT)


def copy_via_exec_cat(namespace, pod, container, remote_path, local_path):
    """Strategy 2: kubectl exec cat > file (direct streaming, bypasses tar)."""
    cmd = [
        "kubectl", "-n", namespace, "exec", pod, "-c", container, "--",
        "cat", remote_path,
    ]
    with open(local_path, "wb") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=COPY_TIMEOUT)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=proc.stderr)


CHUNK_SIZE = 1024 * 1024  # 1MB chunks


def copy_via_chunked(namespace, pod, container, remote_path, local_path, expected_size=-1, display_local_path=None):
    """Strategy 3: read file in chunks via dd, verify each chunk, concatenate locally.
    Resilient to network drops that truncate large streams."""
    if expected_size <= 0:
        cmd = [
            "kubectl", "-n", namespace, "exec", pod, "-c", container, "--",
            "stat", "-c", "%s", remote_path,
        ]
        out = subprocess.check_output(cmd, text=True, timeout=30).strip()
        expected_size = int(out)

    total_chunks = (expected_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunked_start = time.time()

    with open(local_path, "wb") as out_f:
        for chunk_idx in range(total_chunks):
            offset = chunk_idx * CHUNK_SIZE
            remaining = expected_size - offset
            this_chunk = min(CHUNK_SIZE, remaining)

            dd_cmd = f"dd if='{remote_path}' bs={CHUNK_SIZE} skip={chunk_idx} count=1 2>/dev/null"
            cmd = [
                "kubectl", "-n", namespace, "exec", pod, "-c", container, "--",
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
                    elapsed = time.time() - chunked_start
                    dst = display_local_path or local_path
                    print(f"\r    [chunked dd] {chunk_idx + 1}/{total_chunks} chunks ({pct}%) [{elapsed:.1f}s] -> {dst}", end="", flush=True)
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


def copy_file(namespace, pod, container, remote_path, local_path, expected_size,
              expected_md5=None, max_retries=3):
    """Try to copy a file using multiple strategies with retries, size + MD5 verification."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    strategies = [
        ("chunked dd", lambda: copy_via_chunked(namespace, pod, container, remote_path, local_path, expected_size, display_local_path=local_path)),
        ("kubectl cp", lambda: copy_via_kubectl_cp(namespace, pod, container, remote_path, local_path)),
        ("exec cat",   lambda: copy_via_exec_cat(namespace, pod, container, remote_path, local_path)),
    ]

    for strat_name, strat_fn in strategies:
        for attempt in range(1, max_retries + 1):
            try:
                strat_fn()

                # Verify size
                if expected_size > 0:
                    local_size = os.path.getsize(local_path)
                    if local_size != expected_size:
                        raise ValueError(
                            f"size mismatch: local={local_size} remote={expected_size}"
                        )
                # Verify MD5
                if expected_md5:
                    actual_md5 = local_md5(local_path)
                    if actual_md5 != expected_md5:
                        raise ValueError(
                            f"md5 mismatch: local={actual_md5} remote={expected_md5}"
                        )
                return True

            except Exception as e:
                err_msg = str(e).split("\n")[0][:120]
                print(f"    [{strat_name} attempt {attempt}/{max_retries}] {err_msg}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

    return False


def needs_copy(local_path, expected_size, expected_md5=None):
    """Check if a file needs to be (re)copied based on existence, size, and MD5."""
    if not os.path.exists(local_path):
        return True
    if expected_size > 0 and os.path.getsize(local_path) != expected_size:
        return True
    if expected_md5 and local_md5(local_path) != expected_md5:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Copy workload files from client-service pod to local")
    parser.add_argument("--pod", type=str, default=None, help="Pod name (auto-detected if not set)")
    parser.add_argument("--namespace", type=str, default="default")
    parser.add_argument("--app-label", type=str, default="client-service")
    parser.add_argument("--container", type=str, default="client")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without copying")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only copy files that are missing or have wrong size")
    args = parser.parse_args()

    pod = args.pod or get_pod_name(args.namespace, args.app_label)

    # Local base: ../workload-and-experiment_results/ relative to this script (recovery/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_base = os.path.join(script_dir, "..", "workload-and-experiment_results")
    local_base = os.path.normpath(local_base)

    print(f"Pod: {pod}")
    print(f"Container: {args.container}")
    print(f"Remote base: {REMOTE_BASE}")
    print(f"Local base: {local_base}")
    if args.retry_failed:
        print("Mode: retry-failed (skipping files that already match size + MD5)")
    print("=" * 80)

    # Build the full file list: resolve directory entries by discovering files on the pod
    file_tasks = []  # list of (remote_subdir, filename)
    for subdir, files in WORKLOAD_ENTRIES:
        if files is not None:
            for f in files:
                file_tasks.append((subdir, f))
        else:
            # Discover files in this directory on the pod
            remote_dir = f"{REMOTE_BASE}/{subdir}"
            discovered = discover_remote_files(args.namespace, pod, args.container, remote_dir)
            if discovered:
                print(f"Discovered {len(discovered)} file(s) in {remote_dir}")
                for f in discovered:
                    file_tasks.append((subdir, f))
            else:
                print(f"Warning: no files found in {remote_dir}")

    print(f"Total files to copy: {len(file_tasks)}")
    print("=" * 80)

    if args.dry_run:
        for subdir, fname in file_tasks:
            remote = f"{REMOTE_BASE}/{subdir}/{fname}"
            local = os.path.join(local_base, subdir, fname)
            print(f"  {remote} -> {local}")
        print(f"\nDry run complete. {len(file_tasks)} files would be copied.")
        return

    success = 0
    fail = 0
    skipped = 0
    failed_files = []

    # Group by subdir for batch size fetching
    from collections import defaultdict
    by_subdir = defaultdict(list)
    for subdir, fname in file_tasks:
        by_subdir[subdir].append(fname)

    for subdir, fnames in by_subdir.items():
        remote_dir = f"{REMOTE_BASE}/{subdir}"
        print(f"\n[{subdir}] ({len(fnames)} file(s))")

        # Batch-fetch remote sizes and MD5s
        remote_sizes = get_remote_sizes_batch(args.namespace, pod, args.container, remote_dir, fnames)
        remote_md5s = get_remote_md5s_batch(args.namespace, pod, args.container, remote_dir, fnames)

        for fname in fnames:
            remote = f"{REMOTE_BASE}/{subdir}/{fname}"
            local = os.path.join(local_base, subdir, fname)
            expected_size = remote_sizes.get(fname, -1)
            expected_md5 = remote_md5s.get(fname, None)

            if args.retry_failed and not needs_copy(local, expected_size, expected_md5):
                skipped += 1
                continue

            size_str = f" ({expected_size / 1024 / 1024:.1f}MB)" if expected_size > 1024 * 1024 else ""
            print(f"  {fname}{size_str} ... ", end="", flush=True)

            if copy_file(args.namespace, pod, args.container, remote, local, expected_size, expected_md5):
                print("OK")
                success += 1
            else:
                print("FAILED")
                fail += 1
                failed_files.append(remote)

    print("\n" + "=" * 80)
    print(f"Done. {success} succeeded, {fail} failed, {skipped} skipped.")
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
