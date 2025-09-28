#!/bin/bash

# Script to rename directories containing "prefix-cache-" to "prefix_cache_"
# Run this script from the experiment_results directory

echo "Starting directory rename process..."

# Find all directories containing "prefix-cache-" and rename them
find . -type d -name "*prefix-cache-*" | while read -r dir; do
    # Get the directory name without the path
    dirname=$(basename "$dir")

    # Replace "prefix-cache-" with "prefix_cache_"
    new_dirname=$(echo "$dirname" | sed 's/prefix-cache-/prefix_cache_/g')

    # Get the parent directory path
    parent_dir=$(dirname "$dir")

    # Construct new path
    new_path="$parent_dir/$new_dirname"

    echo "Renaming: $dir -> $new_path"

    # Rename the directory
    mv "$dir" "$new_path"
done

echo "Directory rename process completed!"
