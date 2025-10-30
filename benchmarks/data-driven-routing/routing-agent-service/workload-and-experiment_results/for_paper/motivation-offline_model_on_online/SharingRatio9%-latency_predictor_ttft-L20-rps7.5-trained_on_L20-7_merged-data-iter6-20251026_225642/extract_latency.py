import csv
import re

def parse_log_line(line):
    """
    Parse a log line to extract ttft and chosenPodPredictedLatency.
    Returns (actual_ttft, predicted_ttft) or None if not found.
    """
    # Skip empty lines
    if not line.strip():
        return None
    
    # Check if line contains latency metrics
    if '@latency_metrics@' not in line:
        return None
    
    # Extract ttft value
    ttft_match = re.search(r'@ttft@(\d+(?:\.\d+)?)', line)
    # Extract chosenPodPredictedLatency value
    predicted_match = re.search(r'@chosenPodPredictedLatency@(\d+(?:\.\d+)?)', line)
    
    if ttft_match and predicted_match:
        actual_ttft = float(ttft_match.group(1))
        predicted_ttft = float(predicted_match.group(1))
        return (actual_ttft, predicted_ttft)
    
    return None

def extract_latencies(input_file, output_file):
    """
    Extract TTFT and predicted latencies from log file and save to CSV.
    """
    results = []
    
    # Read and parse the log file
    with open(input_file, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                results.append(parsed)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['actual_ttft', 'predicted_ttft'])
        writer.writerows(results)
    
    print(f"Extracted {len(results)} records")
    print(f"Saved to: {output_file}")
    
    # Print some statistics
    if results:
        actual_ttfts = [r[0] for r in results]
        predicted_ttfts = [r[1] for r in results]
        print(f"\nStatistics:")
        print(f"Actual TTFT - Min: {min(actual_ttfts):.2f}, Max: {max(actual_ttfts):.2f}, Avg: {sum(actual_ttfts)/len(actual_ttfts):.2f}")
        print(f"Predicted TTFT - Min: {min(predicted_ttfts):.2f}, Max: {max(predicted_ttfts):.2f}, Avg: {sum(predicted_ttfts)/len(predicted_ttfts):.2f}")

if __name__ == "__main__":
    input_file = "filtered-aibrix-gateway-plugins.log.csv"
    output_file = "latency_comparison.csv"
    
    extract_latencies(input_file, output_file)

