#!/usr/bin/env python3
"""
Copy all *.py files from the routing-agent-service pod back to the host.

Based on ship_all.py:
- Reuses the same pod selection logic (namespace + app label)
- Uses `kubectl cp` to copy from pod -> host
- Filters to only *.py files on the host side
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


def wait_for_single_pod(namespace: str, app_label: str) -> str:
    """
    Get the single running pod name using `kubectl`, without relying on Python kube config.
    """
    cmd = [
        "kubectl",
        "-n",
        namespace,
        "get",
        "pods",
        "-l",
        f"app={app_label}",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get pods via kubectl: {e}")
        sys.exit(1)

    pods = [p for p in out.split() if p]
    if not pods:
        print("❌ No running pods found. Please ensure the routing-agent-service is deployed.")
        sys.exit(1)
    if len(pods) > 1:
        print(f"❌ Multiple pods found: {pods}. Please scale to a single pod or choose one manually.")
        sys.exit(1)

    pod_name = pods[0]
    print(f"Found 1 running pod: {pod_name}")
    return pod_name


def copy_py_files_from_pod(namespace: str, pod_name: str, remote_dir: str, local_target_dir: str) -> None:
    """
    Use `kubectl exec` + `find` to list *.py files under `remote_dir`, then
    `kubectl cp` each file individually into `local_target_dir`, preserving
    relative paths. This avoids one huge stream that can be fragile.
    """
    local_target_path = Path(local_target_dir).resolve()
    local_target_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Local target directory for *.py files: {local_target_path}")

    # 1) List all *.py files in the remote_dir using kubectl exec + find
    find_cmd = [
        "kubectl",
        "-n",
        namespace,
        "exec",
        pod_name,
        "--",
        "sh",
        "-c",
        f"cd {remote_dir} && find . -type f -name '*.py'",
    ]
    print(f"Listing *.py files with: {' '.join(find_cmd)}")
    try:
        out = subprocess.check_output(find_cmd, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to list *.py files via kubectl exec: {e}")
        sys.exit(1)

    rel_files = [line.strip() for line in out.splitlines() if line.strip()]
    if not rel_files:
        print("⚠️ No *.py files found under remote directory.")
        return

    print(f"Found {len(rel_files)} *.py files to copy.")

    # 2) Copy each file individually with kubectl cp
    copied_count = 0
    for rel in rel_files:
        # rel is like "./routing_agent_service.py" or "./subdir/file.py"
        # Normalize and drop leading "./"
        rel_clean = rel.lstrip("./")
        remote_file = f"{remote_dir.rstrip('/')}/{rel_clean}"
        dest_path = local_target_path / rel_clean
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        cp_cmd = [
            "kubectl",
            "-n",
            namespace,
            "cp",
            f"{pod_name}:{remote_file}",
            str(dest_path),
        ]
        print(f"Copying {remote_file} -> {dest_path}")
        try:
            subprocess.check_call(cp_cmd)
            copied_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to copy {remote_file}: exit code {e.returncode}")
            # Continue with others rather than aborting all
            continue

    if copied_count == 0:
        print("⚠️ No *.py files were successfully copied.")
    else:
        print(f"🎉 Finished copying {copied_count} *.py files from pod to {local_target_path}")


def copy_directories_from_pod(
    namespace: str,
    pod_name: str,
    remote_base_dir: str,
    local_base_dir: str,
    dir_names: list[str],
) -> None:
    """
    Copy entire directories from the pod to the host using `kubectl cp`.
    Each name in `dir_names` is treated as a subdirectory under `remote_base_dir`.
    """
    if not dir_names:
        return

    local_base_path = Path(local_base_dir).resolve()
    local_base_path.mkdir(parents=True, exist_ok=True)

    print(f"📁 Copying full directories into: {local_base_path}")

    for d in dir_names:
        d = d.strip()
        if not d:
            continue

        remote_path = f"{remote_base_dir.rstrip('/')}/{d}"
        local_path = local_base_path / d
        local_path_parent = local_path.parent
        local_path_parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "kubectl",
            "-n",
            namespace,
            "cp",
            f"{pod_name}:{remote_path}",
            str(local_path),
        ]
        print(f"Copying directory {remote_path} -> {local_path}")
        try:
            subprocess.check_call(cmd)
            print(f"✅ Copied directory {remote_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to copy directory {remote_path}: exit code {e.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy all *.py files from routing-agent-service pod to host directory"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace where routing-agent-service is running",
    )
    parser.add_argument(
        "--app_label",
        type=str,
        default="routing-agent-service",
        help="Value of the 'app' label used to select the routing-agent-service pod",
    )
    parser.add_argument(
        "--remote_dir",
        type=str,
        default="/app",
        help="Remote directory inside the pod to copy from (will be filtered to *.py unless --copy_dirs is used)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="../agent_codes",
        help="Local directory to copy *.py files into (routing-agent-service dir on host)",
    )
    parser.add_argument(
        "--copy_dirs",
        type=str,
        default="",
        help=(
            "Comma-separated list of directory names under --remote_dir to copy in full "
            "(e.g. 'NVIDIA-A10,NVIDIA-A30,NVIDIA-L20,hetero'). "
            "They will be created under --local_dir."
        ),
    )

    args = parser.parse_args()

    print(
        f"Preparing to copy *.py files from pod (namespace={args.namespace}, "
        f"app={args.app_label}) remote_dir={args.remote_dir} to local_dir={args.local_dir}"
    )

    pod_name = wait_for_single_pod(args.namespace, args.app_label)
    # Always copy *.py files
    copy_py_files_from_pod(
        namespace=args.namespace,
        pod_name=pod_name,
        remote_dir=args.remote_dir,
        local_target_dir=args.local_dir,
    )

    # Optionally copy entire directories
    if args.copy_dirs:
        dir_names = [d.strip() for d in args.copy_dirs.split(",") if d.strip()]
        if dir_names:
            copy_directories_from_pod(
                namespace=args.namespace,
                pod_name=pod_name,
                remote_base_dir=args.remote_dir,
                local_base_dir=args.local_dir,
                dir_names=dir_names,
            )


if __name__ == "__main__":
    main()


