#!/bin/bash

# Script to merge all filtered-aibrix-gateway-plugins.log.csv files into data.csv

# Check if root directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <root_directory> [output_dir_name] [filter_word...]"
    echo "  root_directory: Directory to search for CSV files"
    echo "  output_dir_name: (optional) Output directory name or full path"
    echo "  filter_word: (optional) One or more words - files in directories containing any of these words will be processed"
    exit 1
fi

ROOT_DIR="$1"
OUTPUT_DIR_NAME="$2"
# Collect all filter words from $3 onwards
shift 2
FILTER_WORDS=("$@")

# Validate that the directory exists
if [ ! -d "${ROOT_DIR}" ]; then
    echo "Error: Directory '${ROOT_DIR}' does not exist"
    exit 1
fi

if [ -z "${OUTPUT_DIR_NAME}" ]; then
    OUTPUT_FILE="${ROOT_DIR}/data.csv"
elif [[ "${OUTPUT_DIR_NAME}" == /* ]] || [[ "${OUTPUT_DIR_NAME}" == *.csv ]]; then
    OUTPUT_FILE="${OUTPUT_DIR_NAME}"
else
    OUTPUT_FILE="${ROOT_DIR}/${OUTPUT_DIR_NAME}/data.csv"
fi

# Ensure parent directory exists
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory '$OUTPUT_DIR' does not exist"
    exit 1
fi

# Remove output file if it exists
[ -f "$OUTPUT_FILE" ] && rm "$OUTPUT_FILE"

# Find all filtered-aibrix-gateway-plugins.log.csv files recursively
all_files=$(find "${ROOT_DIR}" -type f -name "filtered-aibrix-gateway-plugins.log.csv" | sort)

# Check if any files exist at all
if [ -z "$all_files" ]; then
    echo "No filtered-aibrix-gateway-plugins.log.csv files found"
    exit 1
fi

# Filter files by directory name if filter words are provided
if [ ${#FILTER_WORDS[@]} -gt 0 ]; then
    filtered_files=""
    while IFS= read -r file; do
        # Check if the file path contains any of the filter words
        match_found=false
        for filter_word in "${FILTER_WORDS[@]}"; do
            if [[ "$file" == *"$filter_word"* ]]; then
                match_found=true
                break
            fi
        done
        if [ "$match_found" = true ]; then
            filtered_files="${filtered_files}${file}"$'\n'
        fi
    done <<< "$all_files"
    filtered_files=$(echo "$filtered_files" | sed '/^$/d' | sort)
    
    # If filter matched no files, fall back to processing all files
    if [ -z "$filtered_files" ]; then
        echo "Filtering directories containing any of: ${FILTER_WORDS[*]}"
        echo "No files matched the filter - processing all files instead"
        files="$all_files"
    else
        files="$filtered_files"
        echo "Filtering directories containing any of: ${FILTER_WORDS[*]}"
    fi
else
    files="$all_files"
    echo "No filter specified - processing all files"
fi

echo "Found $(echo "$files" | wc -l) file(s)"

# Flag to track if header has been written
header_written=false

# Process each file
while IFS= read -r file; do
    echo "Processing: $file"
    
    if [ "$header_written" = false ]; then
        # For the first file, copy everything including header
        cat "$file" >> "$OUTPUT_FILE"
        header_written=true
    else
        # For subsequent files, skip the header (first line)
        tail -n +2 "$file" >> "$OUTPUT_FILE"
    fi
done <<< "$files"

echo "Merge complete! Output saved to: $OUTPUT_FILE"
echo "Total lines in output: $(wc -l < "$OUTPUT_FILE")"

