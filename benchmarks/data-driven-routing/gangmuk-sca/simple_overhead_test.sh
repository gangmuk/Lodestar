#!/bin/bash

# Simple Overhead Test Script
# Usage: ./simple_overhead_test.sh [RPS] [DURATION]

RPS=${1:-5}
DURATION=${2:-10}
SERVICE_URL="http://localhost:8080/infer"

echo "🚀 Testing $RPS RPS for $DURATION seconds..."
echo "Target URL: $SERVICE_URL"

# Sample request data with real IP addresses
REQUEST_DATA='{"test_req_1": "**@latency_metrics@requestID@test_req_1@request_start_time@1000@request_end_time@2000@selectedpod@10.0.0.39@ttft@100@avg_tpot@50@total_decode_time@500@e2e@1000@numInputTokens@100@numOutputTokens@50@numTotalTokens@150@allPodsKvCacheHitRatios@{\"10.0.0.39\":0.8,\"10.0.1.119\":0.6,\"10.0.0.142\":0.7}@numInflightRequestsAllPods@{\"10.0.0.39\":2,\"10.0.1.119\":1,\"10.0.0.142\":1}@vllmGPUKVCacheUsage@{\"10.0.0.39\":0.7,\"10.0.1.119\":0.5,\"10.0.0.142\":0.6}@vllmCPUKVCacheUsage@{}@vllmNumRequestsRunning@{\"10.0.0.39\":1,\"10.0.1.119\":1,\"10.0.0.142\":0}@vllmNumRequestsWaiting@{\"10.0.0.39\":0,\"10.0.1.119\":0,\"10.0.0.142\":0}@podMetricsLastSecond@{}@numPrefillTokensForAllPods@{\"10.0.0.39\":100,\"10.0.1.119\":50,\"10.0.0.142\":75}@numDecodeTokensForAllPods@{\"10.0.0.39\":200,\"10.0.1.119\":150,\"10.0.0.142\":175}@subAlgorithm@latency_predictor@prev_reward0.0"}'

# Calculate interval between requests
INTERVAL=$(echo "scale=3; 1/$RPS" | bc)

REQUEST_COUNT=0
SUCCESS_COUNT=0
TOTAL_OVERHEAD_MS=0
MIN_OVERHEAD_MS=999999999
MAX_OVERHEAD_MS=0

# Initialize overhead component tracking
declare -A TOTAL_COMPONENT_OVERHEAD
declare -A MIN_COMPONENT_OVERHEAD
declare -A MAX_COMPONENT_OVERHEAD
declare -A COMPONENT_COUNT

START_TIME=$(date +%s.%N)

while true; do
    CURRENT_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$CURRENT_TIME - $START_TIME" | bc)
    
    # Break if duration exceeded
    if [ $(echo "$ELAPSED >= $DURATION" | bc -l) -eq 1 ]; then
        echo "⏰ Duration limit reached. Stopping..."
        break
    fi
    
    REQUEST_COUNT=$((REQUEST_COUNT + 1))
    
    # Safety check: prevent infinite loops
    if [ "$REQUEST_COUNT" -gt 1000 ]; then
        echo "⚠️  Safety limit reached (1000 requests). Stopping..."
        break
    fi
    
    # Send request in background to achieve target RPS
    (
        RESPONSE=$(curl -s -X POST "$SERVICE_URL" \
            -H "Content-Type: application/json" \
            --data-binary "@test_request.json")
        
        # Check if response contains overhead_log (successful response)
        if echo "$RESPONSE" | grep -q '"overhead_log"'; then
            # Extract overhead data
            OVERHEAD_LOG=$(echo "$RESPONSE" | jq -r '.overhead_log // empty')
            if [ -n "$OVERHEAD_LOG" ] && [ "$OVERHEAD_LOG" != "null" ]; then
                # Extract end_to_end overhead
                END_TO_END_MS=$(echo "$OVERHEAD_LOG" | grep -oP 'handle_infer_end_to_end: \K[0-9.]+' | head -1)
                if [ -n "$END_TO_END_MS" ]; then
                    echo "✅ Request $REQUEST_COUNT: Overhead: ${END_TO_END_MS}ms" >> /tmp/overhead_results_$$
                    echo "$END_TO_END_MS" >> /tmp/overhead_values_$$
                    
                    # Extract all overhead components
                    echo "$OVERHEAD_LOG" >> /tmp/overhead_components_$$
                else
                    echo "❌ Request $REQUEST_COUNT: No overhead data found" >> /tmp/overhead_results_$$
                fi
            else
                echo "❌ Request $REQUEST_COUNT: No overhead_log in response" >> /tmp/overhead_results_$$
            fi
        else
            echo "❌ Request $REQUEST_COUNT: Error in response" >> /tmp/overhead_results_$$
        fi
    ) &
    
    # Wait for next request
    NEXT_REQUEST_TIME=$(echo "$START_TIME + ($REQUEST_COUNT * $INTERVAL)" | bc)
    SLEEP_TIME=$(echo "$NEXT_REQUEST_TIME - $(date +%s.%N)" | bc)
    
    if [ $(echo "$SLEEP_TIME > 0" | bc -l) -eq 1 ]; then
        sleep "$SLEEP_TIME"
    fi
done

# Wait for all background processes to complete
wait

FINAL_TIME=$(date +%s.%N)
TOTAL_ELAPSED=$(echo "$FINAL_TIME - $START_TIME" | bc)

# Collect results from background processes
if [ -f "/tmp/overhead_results_$$" ]; then
    echo ""
    echo "📋 Request Results:"
    cat /tmp/overhead_results_$$
    
    SUCCESS_COUNT=$(wc -l < /tmp/overhead_values_$$)
    if [ "$SUCCESS_COUNT" -gt 0 ]; then
        TOTAL_OVERHEAD_MS=$(awk '{sum+=$1} END {print sum}' /tmp/overhead_values_$$)
        AVG_OVERHEAD_MS=$(echo "scale=2; $TOTAL_OVERHEAD_MS / $SUCCESS_COUNT" | bc)
        MIN_OVERHEAD_MS=$(sort -n /tmp/overhead_values_$$ | head -1)
        MAX_OVERHEAD_MS=$(sort -n /tmp/overhead_values_$$ | tail -1)
    fi
    
    # Analyze overhead components
    if [ -f "/tmp/overhead_components_$$" ]; then
        echo ""
        echo "📊 Overhead Component Analysis:"
        echo "=================================="
        
        # Extract all components to a temporary file
        > /tmp/component_data_$$
        
        while IFS= read -r line; do
            # Extract all component:value pairs
            echo "$line" | sed 's/oh, //' | tr ',' '\n' | while IFS=':' read -r component value; do
                # Clean up component name and value
                component=$(echo "$component" | sed 's/^ *//;s/ *$//')
                value=$(echo "$value" | sed 's/^ *//;s/ms *$//')
                
                # Skip empty or invalid entries
                if [ -n "$component" ] && [ -n "$value" ] && [[ "$value" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
                    # Skip negative values (they indicate unused components)
                    if [ $(echo "$value >= 0" | bc -l) -eq 1 ]; then
                        echo "$component:$value" >> /tmp/component_data_$$
                    fi
                fi
            done
        done < /tmp/overhead_components_$$
        
        # Process and display component analysis
        echo "Component                    | Avg (ms) | Min (ms) | Max (ms) | Count"
        echo "----------------------------|----------|----------|----------|-------"
        
        # Get unique components and calculate stats
        cut -d':' -f1 /tmp/component_data_$$ | sort -u | while read -r component; do
            values=$(grep "^$component:" /tmp/component_data_$$ | cut -d':' -f2)
            count=$(echo "$values" | wc -l)
            
            if [ "$count" -gt 0 ]; then
                total=$(echo "$values" | awk '{sum+=$1} END {print sum}')
                avg=$(echo "scale=2; $total / $count" | bc)
                min=$(echo "$values" | sort -n | head -1)
                max=$(echo "$values" | sort -n | tail -1)
                
                printf "%-28s | %8s | %8s | %8s | %5d\n" "$component" "$avg" "$min" "$max" "$count"
            fi
        done | sort -k2 -nr
        
        rm -f /tmp/component_data_$$
    fi
    
    # Clean up temp files
    rm -f /tmp/overhead_results_$$ /tmp/overhead_values_$$ /tmp/overhead_components_$$
fi

echo ""
echo "📈 RESULTS:"
echo "🎯 Target RPS: $RPS"
echo "📊 Actual RPS: $(echo "scale=2; $REQUEST_COUNT / $TOTAL_ELAPSED" | bc)"
echo "📋 Total Requests: $REQUEST_COUNT"
echo "✅ Successful: $SUCCESS_COUNT"
echo "❌ Failed: $((REQUEST_COUNT - SUCCESS_COUNT))"
echo "📈 Success Rate: $(echo "scale=1; $SUCCESS_COUNT * 100 / $REQUEST_COUNT" | bc)%"
echo "⏱️  Total Time: ${TOTAL_ELAPSED}s"

if [ "$SUCCESS_COUNT" -gt 0 ]; then
    echo "📊 Average Overhead: ${AVG_OVERHEAD_MS}ms"
    echo "📊 Min Overhead: ${MIN_OVERHEAD_MS}ms"
    echo "📊 Max Overhead: ${MAX_OVERHEAD_MS}ms"
fi
