#!/bin/bash

# Script to merge all filtered-aibrix-gateway-plugins.log.csv files into data.csv

# Check if parent directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <parent_directory>"
    echo "Example: $0 /path/to/logs"
    exit 1
fi

PARENT_DIR="$1"

# Check if the provided directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Directory '$PARENT_DIR' does not exist"
    exit 1
fi

OUTPUT_FILE="${PARENT_DIR}/data.csv"

# Remove output file if it exists
[ -f "$OUTPUT_FILE" ] && rm "$OUTPUT_FILE"

# Find all filtered-aibrix-gateway-plugins.log.csv files recursively
files=$(find "$PARENT_DIR" -type f -name "filtered-aibrix-gateway-plugins.log.csv" | sort)

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

