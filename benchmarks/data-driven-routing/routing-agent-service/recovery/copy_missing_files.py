#!/usr/bin/env python3
"""
Copy missing and size-mismatched files from the pod to local.
Based on the comparison results.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


POD = "routing-agent-service-5d6d4b9b5-q9hnz"
NAMESPACE = "default"
REMOTE_BASE = "/app"
LOCAL_BASE = "/Users/gangmuk2/Downloads/recovery/routing-agent-service"

# Timeout per file copy attempt (in seconds)
# Increased for slow network connections - 60 minutes should be enough for large files
COPY_TIMEOUT = 60 * 60  # 60 minutes


# Files to copy - only the 3 that failed (large files that timed out)
# The other 13 files were successfully copied
FILES_TO_COPY = [
    # NVIDIA-A10: Large tensor dataset file (26.32 MB) - timed out
    ("NVIDIA-A10", "./PrefillOnly/final_model/contextual_bandit_linear_simple_extended_clustermodel/encoded_data/batch_1/tensor_dataset.pt"),
    
    # NVIDIA-A30: Large CSV file (31.88 MB) - timed out
    ("NVIDIA-A30", "./Aggregated/offline_training_data.csv"),
    
    # NVIDIA-L20: Large CSV file (46.45 MB) - timed out
    ("NVIDIA-L20", "./Aggregated/offline_training_data.csv"),
]


def copy_file(dir_name, rel_path, max_retries=3):
    """Copy a single file from pod to local with retry logic."""
    # Remove leading ./ from rel_path
    rel_path_clean = rel_path.lstrip("./")
    
    remote_path = f"{REMOTE_BASE}/{dir_name}/{rel_path_clean}"
    local_dir = f"{LOCAL_BASE}/{dir_name}"
    local_file = os.path.join(local_dir, rel_path_clean)
    
    # Create parent directory if needed
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    
    # Use kubectl cp with retry logic
    cmd = [
        "kubectl", "-n", NAMESPACE,
        "cp",
        f"{POD}:{remote_path}",
        local_file
    ]
    
    print(f"Copying: {remote_path} -> {local_file}")
    
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=COPY_TIMEOUT  # Configurable timeout (default: 60 minutes)
            )
            print(f"  ✅ Success (attempt {attempt})")
            return True
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  Timeout on attempt {attempt}/{max_retries}")
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                print(f"  ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            # Check if it's a timeout error
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print(f"  ⏱️  Timeout on attempt {attempt}/{max_retries}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                    print(f"  ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Failed after {max_retries} attempts: {error_msg}")
                    return False
            else:
                # Non-timeout error, don't retry
                print(f"  ❌ Failed: {error_msg}")
                return False
    
    print(f"  ❌ Failed after {max_retries} attempts")
    return False


def main():
    print("="*80)
    print("Copying missing and size-mismatched files")
    print(f"Pod: {POD}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Total files to copy: {len(FILES_TO_COPY)}")
    print(f"Timeout per file: {COPY_TIMEOUT // 60} minutes")
    print("="*80)
    
    success_count = 0
    fail_count = 0
    
    for dir_name, rel_path in FILES_TO_COPY:
        if copy_file(dir_name, rel_path):
            success_count += 1
        else:
            fail_count += 1
        # Small delay between files to avoid overwhelming the connection
        time.sleep(1)
        print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Successfully copied: {success_count}")
    print(f"❌ Failed: {fail_count}")
    
    if fail_count == 0:
        print("\n🎉 All files copied successfully!")
    else:
        print(f"\n⚠️  {fail_count} file(s) failed to copy. Check errors above.")


if __name__ == "__main__":
    main()

