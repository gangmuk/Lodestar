#!/bin/bash

# Simple RPS Test Script for Routing Agent Service
# Usage: ./simple_rps_test.sh <rps> <duration_seconds>

RPS=${1:-10}
DURATION=${2:-10}
SERVICE_URL="http://localhost:8080/infer"

echo "🚀 Testing $RPS RPS for $DURATION seconds..."

# Sample request data - using correct format with real IP addresses
REQUEST_DATA='{"test_req_1": "**@latency_metrics@requestID@test_req_1@request_start_time@1000@request_end_time@2000@selectedpod@10.0.0.39@ttft@100@avg_tpot@50@total_decode_time@500@e2e@1000@numInputTokens@100@numOutputTokens@50@numTotalTokens@150@allPodsKvCacheHitRatios@{\"10.0.0.39\":0.8,\"10.0.1.119\":0.6,\"10.0.0.142\":0.7}@numInflightRequestsAllPods@{\"10.0.0.39\":2,\"10.0.1.119\":1,\"10.0.0.142\":1}@vllmGPUKVCacheUsage@{\"10.0.0.39\":0.7,\"10.0.1.119\":0.5,\"10.0.0.142\":0.6}@vllmCPUKVCacheUsage@{}@vllmNumRequestsRunning@{\"10.0.0.39\":1,\"10.0.1.119\":1,\"10.0.0.142\":0}@vllmNumRequestsWaiting@{\"10.0.0.39\":0,\"10.0.1.119\":0,\"10.0.0.142\":0}@podMetricsLastSecond@{}@numPrefillTokensForAllPods@{\"10.0.0.39\":100,\"10.0.1.119\":50,\"10.0.0.142\":75}@numDecodeTokensForAllPods@{\"10.0.0.39\":200,\"10.0.1.119\":150,\"10.0.0.142\":175}@subAlgorithm@latency_predictor@prev_reward@0.0"}'

# Calculate interval between requests (in milliseconds)
INTERVAL_MS=$((1000 / RPS))

echo "📊 Interval between requests: ${INTERVAL_MS}ms"
echo "🎯 Target RPS: $RPS"
echo "⏱️  Duration: $DURATION seconds"
echo ""

# Track statistics
REQUEST_COUNT=0
SUCCESS_COUNT=0
START_TIME=$(date +%s)

while [ $REQUEST_COUNT -lt $((RPS * DURATION)) ]; do
    REQUEST_START=$(date +%s.%N)
    
    # Send request
    RESPONSE=$(curl -s -w "%{http_code},%{time_total}" -X POST \
        -H "Content-Type: application/json" \
        -d "$REQUEST_DATA" \
        "$SERVICE_URL" 2>/dev/null)
    
    REQUEST_END=$(date +%s.%N)
    REQUEST_COUNT=$((REQUEST_COUNT + 1))
    
    # Parse response - curl appends HTTP code and time at the end
    HTTP_CODE=$(echo "$RESPONSE" | grep -oP ',\d+,\d+\.\d+$' | cut -d',' -f2)
    TIME_TOTAL=$(echo "$RESPONSE" | grep -oP ',\d+,\d+\.\d+$' | cut -d',' -f3)
    
    if [ "$HTTP_CODE" = "200" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        # Extract overhead data from JSON response
        JSON_RESPONSE=$(echo "$RESPONSE" | sed 's/,[0-9]*,[0-9.]*$//')
        OVERHEAD_LOG=$(echo "$JSON_RESPONSE" | jq -r '.overhead_log // empty')
        if [ -n "$OVERHEAD_LOG" ]; then
            # Extract end_to_end overhead
            END_TO_END_MS=$(echo "$OVERHEAD_LOG" | grep -oP 'handle_infer_end_to_end: \K[0-9.]+' | head -1)
            if [ -n "$END_TO_END_MS" ]; then
                echo "✅ Request $REQUEST_COUNT: ${TIME_TOTAL}s, Overhead: ${END_TO_END_MS}ms"
            else
                echo "✅ Request $REQUEST_COUNT: ${TIME_TOTAL}s"
            fi
        else
            echo "✅ Request $REQUEST_COUNT: ${TIME_TOTAL}s"
        fi
    else
        echo "❌ Request $REQUEST_COUNT: HTTP $HTTP_CODE"
    fi
    
    # Wait for next request interval
    sleep 0.001  # Sleep 1ms, then calculate remaining time
    CURRENT_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$CURRENT_TIME - $START_TIME" | bc)
    
    # Check if we should stop
    if [ $(echo "$ELAPSED >= $DURATION" | bc -l) -eq 1 ]; then
        break
    fi
    
    # Calculate sleep time for next request
    NEXT_REQUEST_TIME=$(echo "$REQUEST_START + 0.001" | bc)
    SLEEP_TIME=$(echo "$NEXT_REQUEST_TIME - $(date +%s.%N)" | bc)
    
    if [ $(echo "$SLEEP_TIME > 0" | bc -l) -eq 1 ]; then
        sleep "$SLEEP_TIME"
    fi
done

# Calculate final statistics
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
ACTUAL_RPS=$((REQUEST_COUNT / TOTAL_TIME))
SUCCESS_RATE=$((SUCCESS_COUNT * 100 / REQUEST_COUNT))

echo ""
echo "📈 RESULTS:"
echo "🎯 Target RPS: $RPS"
echo "📊 Actual RPS: $ACTUAL_RPS"
echo "📋 Total Requests: $REQUEST_COUNT"
echo "✅ Successful: $SUCCESS_COUNT"
echo "❌ Failed: $((REQUEST_COUNT - SUCCESS_COUNT))"
echo "📈 Success Rate: $SUCCESS_RATE%"
echo "⏱️  Total Time: ${TOTAL_TIME}s"
