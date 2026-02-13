#!/bin/bash

# Script to merge all filtered-aibrix-gateway-plugins.log.csv files into data.csv

# Check if root directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <root_directory> [filter_word]"
    echo "  root_directory: Directory to search for CSV files"
    echo "  filter_word: (optional) Only process files in directories containing this word"
    exit 1
fi

ROOT_DIR="$1"
OUTPUT_DIR_NAME="$2"
FILTER_WORD="$3"

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
files=$(find "${ROOT_DIR}" -type f -name "filtered-aibrix-gateway-plugins.log.csv" | sort)

# Filter files by directory name if filter word is provided
if [ -n "$FILTER_WORD" ]; then
    filtered_files=""
    while IFS= read -r file; do
        # Check if the file path contains the filter word
        if [[ "$file" == *"$FILTER_WORD"* ]]; then
            filtered_files="${filtered_files}${file}"$'\n'
        fi
    done <<< "$files"
    files=$(echo "$filtered_files" | sed '/^$/d' | sort)
    echo "Filtering directories containing: $FILTER_WORD"
else
    echo "No filter specified - processing all files"
fi

# Check if any files were found
if [ -z "$files" ]; then
    echo "No filtered-aibrix-gateway-plugins.log.csv files found"
    exit 1
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

