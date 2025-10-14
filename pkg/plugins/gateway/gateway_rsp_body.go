/*
Copyright 2024 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
	"k8s.io/klog/v2"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
)

// HTTP client for async RL agent completion notifications
var (
	httpClientForRLCompletion = &http.Client{
		Timeout: 1000 * time.Millisecond,
		Transport: &http.Transport{
			MaxIdleConns:        10,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     30 * time.Second,
		},
	}
)

// Latency metrics log file writer configuration
var (
	// LatencyMetricsLogPath is the file path where latency metrics will be written
	// Can be set via environment variable LATENCY_METRICS_LOG_PATH
	LatencyMetricsLogPath = getEnvOrDefault("LATENCY_METRICS_LOG_PATH", "/tmp/latency_metrics.log")

	// Channel buffer size for latency metrics log messages
	latencyMetricsLogBufferSize = 100000

	bufferFlushSize = 500

	// logMessageChan is the channel for async log writing
	logMessageChan chan string

	// flushInterval defines how often to flush the log file
	flushInterval = 5 * time.Second

	// logWriterOnce ensures the log writer is initialized only once
	logWriterOnce sync.Once
)

// getEnvOrDefault returns the environment variable value or default if not set
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// initLatencyMetricsLogWriter initializes the background log writer goroutine
func initLatencyMetricsLogWriter() {
	logWriterOnce.Do(func() {
		logMessageChan = make(chan string, latencyMetricsLogBufferSize)
		go latencyMetricsLogWriter()
		klog.Infof("Latency metrics log writer initialized, writing to: %s", LatencyMetricsLogPath)
	})
}

// latencyMetricsLogWriter runs in background and writes log messages to file
func latencyMetricsLogWriter() {
	var file *os.File
	var err error
	var buffer []string

	// Open/create log file
	file, err = os.OpenFile(LatencyMetricsLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		klog.Errorf("Failed to open latency metrics log file %s: %v", LatencyMetricsLogPath, err)
		return
	}
	defer file.Close()

	klog.Infof("latency metrics log writer ticker started, flush interval: %s, buffer flush size: %d, max buffer size: %d", flushInterval, bufferFlushSize, latencyMetricsLogBufferSize)
	ticker := time.NewTicker(flushInterval)
	defer ticker.Stop()

	for {
		select {
		case msg, ok := <-logMessageChan:
			if !ok {
				// Channel closed, flush remaining and exit
				if len(buffer) > 0 {
					klog.Errorf("Flush buffer, Channel closed, flush remaining logs size: %d", len(buffer))
					flushBuffer(file, buffer)
				}
				return
			}
			buffer = append(buffer, msg)

			// Flush if buffer gets too large (prevent memory buildup)
			if len(buffer) >= bufferFlushSize {
				klog.Infof("Flush buffer, buffer reached the flush size: %d", bufferFlushSize)
				flushBuffer(file, buffer)
				buffer = buffer[:0] // Reset buffer
			}

		case <-ticker.C:
			// Periodic flush
			if len(buffer) > 0 {
				klog.Infof("Flush buffer, flushInterval, buffer size: %d, flushInterval: %s", len(buffer), flushInterval)
				flushBuffer(file, buffer)
				buffer = buffer[:0] // Reset buffer
			}
		}
	}
}

// flushBuffer writes all buffered messages to file
func flushBuffer(file *os.File, buffer []string) {
	klog.Infof("Flushing buffer, size: %d", len(buffer))
	numLogWrites := 0
	for _, msg := range buffer {
		if _, err := file.WriteString(msg + "\n"); err != nil {
			klog.Errorf("Failed to write latency metrics log: %v, message: %s", err, msg)
		} else {
			klog.V(5).Infof("Wrote latency metrics log: %s", msg)
		}
		numLogWrites++
	}
	// Sync to ensure data is written to disk
	if err := file.Sync(); err != nil {
		klog.Errorf("Failed to sync latency metrics log file: %v", err)
	}
	klog.Infof("Flushed to the file (%s), %d logs", file.Name(), numLogWrites)
}

// writeLatencyMetricsLog writes a latency metrics log message asynchronously
func writeLatencyMetricsLog(message string) {
	// Initialize log writer on first use
	initLatencyMetricsLogWriter()

	// Non-blocking write to channel
	select {
	case logMessageChan <- message:
		// Successfully queued
	default:
		// Channel full, drop message to avoid blocking
		klog.ErrorS(nil, "Latency metrics log channel full, dropping message. max buffer size", "max_buffer_size", latencyMetricsLogBufferSize)
	}
}

// notifyRLAgentRequestComplete sends async completion notification for scalable RL agent
func notifyRLAgentRequestComplete(routerCtx *types.RoutingContext, ttftMs int64, avgTpotMs int64) {
	// Only notify for scalable_rl_agent
	if routerCtx == nil || routerCtx.SubAlgorithm != "scalable_rl_agent" {
		return
	}

	completionData := map[string]interface{}{
		"request_id":   routerCtx.RequestID,
		"ttft":         ttftMs,
		"tpot":         avgTpotMs,
		"selected_pod": routerCtx.TargetAddressWithoutPort(),
	}

	reqBody, err := json.Marshal(completionData)
	if err != nil {
		klog.Errorf("Failed to marshal completion data for request %s: %v", routerCtx.RequestID, err)
		return
	}

	// Send async (don't block response to client)
	go func() {
		// Use the same routing agent URL as in rl_routing.go
		url := "http://routing-agent-service.default.svc.cluster.local:8080/request_complete"
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
		if err != nil {
			klog.V(5).Infof("Failed to create completion request for %s: %v", routerCtx.RequestID, err)
			return
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := httpClientForRLCompletion.Do(req)
		if err != nil {
			klog.V(5).Infof("Failed to notify RL agent completion for %s: %v", routerCtx.RequestID, err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			klog.V(5).Infof("RL agent completion notification returned %d for %s", resp.StatusCode, routerCtx.RequestID)
		} else {
			klog.V(5).Infof("✅ Notified scalable RL agent of completion for %s (ttft=%dms, tpot=%dms)",
				routerCtx.RequestID, ttftMs, avgTpotMs)
		}
	}()
}

func (s *Server) handleStreamingResponse(requestID string, responseBody []byte) (openai.CompletionUsage, bool, *extProcPb.ProcessingResponse) {
	lines := strings.Split(string(responseBody), "\n")
	existingUsageRaw, _ := s.streamingUsageCache.LoadOrStore(requestID, openai.CompletionUsage{})
	existingUsage := existingUsageRaw.(openai.CompletionUsage)
	timingObj, exists := utils.RequestTimings.Load(requestID)
	if !exists {
		return existingUsage, false, nil
	}
	timing := timingObj.(*RequestTiming)
	prefill_token_count := int(timing.prefillTokenCount)
	currentTime := time.Now()
	routerCtxObj, exists := s.routingContexts.Load(requestID)
	if !exists {
		return existingUsage, false, nil
	}
	routerCtx := routerCtxObj.(*types.RoutingContext)
	selectedPodIP := routerCtx.TargetAddressWithoutPort()
	podIPWithoutPort := routerCtx.TargetAddressWithoutPort()
	t := &http.Response{
		Body: io.NopCloser(bytes.NewReader(responseBody)),
	}
	streaming := ssestream.NewStream[openai.ChatCompletionChunk](ssestream.NewDecoder(t), nil)
	for streaming.Next() {
		evt := streaming.Current()
		if len(evt.Choices) > 0 && evt.Choices[0].Delta.Content != "" {
			// First token response
			if timing.firstTokenTime.IsZero() {
				timing.IsPrefill = false
				timing.decodeTokenCount = 1 // one token is generated by stream mode for the first response
				timing.firstTokenTime = currentTime
				timing.lastTokenTime = currentTime
				ttftMs := currentTime.Sub(timing.startTime).Milliseconds()
				if ttftMs > 0 {
					klog.V(5).Infof("First token received, requestID: %s, podIP: %s, ttft_ms: %d, AddPodMetric", requestID, selectedPodIP, ttftMs)
					utils.MetricsTracker.AddPodMetric(selectedPodIP, utils.PodMetric{
						RequestID:       requestID,
						Timestamp:       currentTime,
						TTFT:            ttftMs,
						TPOT:            0,
						PrefillTokenNum: int64(prefill_token_count),
						DecodeTokenNum:  1,
					})
				} else {
					klog.Errorf("Negative ttft_ms: %d, requestID: %s, podIP: %s", ttftMs, requestID, selectedPodIP)
				}
				klog.V(5).InfoS("First token received", "requestID", requestID, "ttft_ms", ttftMs)

				ret := utils.DecrementNumPrefillTokensForPod(podIPWithoutPort, prefill_token_count)
				klog.V(5).Infof("DecrementNumPrefillTokensForPod(%s) by %d, %d", podIPWithoutPort, prefill_token_count, ret)

				ret = utils.IncrementNumDecodeTokensForPod(podIPWithoutPort, prefill_token_count+1)
				klog.V(5).Infof("IncrementNumDecodeTokensForPod(%s) by %d, %d", podIPWithoutPort, prefill_token_count+1, ret)

				ret = utils.IncrementNumDecodeTokensForRequest(requestID, prefill_token_count+1)
				klog.V(5).Infof("IncrementNumDecodeTokensForRequest(%s) by %d, %d", requestID, prefill_token_count+1, ret)
			} else { // Decode token response
				if timing.firstDecodeTokenTime.IsZero() {
					// First decode token
					timing.firstDecodeTokenTime = currentTime
					klog.V(5).Infof("First decode token received, requestID,%s, timing.prefillTokenCount.%d", requestID, timing.prefillTokenCount)
				}
				timing.decodeTokenCount++
				ret := utils.IncrementNumDecodeTokensForRequest(requestID, 1)
				klog.V(5).Infof("IncrementNumDecodeTokensForRequest(%s) by 1, %d", requestID, ret)

				ret = utils.IncrementNumDecodeTokensForPod(podIPWithoutPort, 1)
				klog.V(5).Infof("IncrementNumDecodeTokensForPod(%s) by 1, %d", podIPWithoutPort, ret)

				timeSincePrevToken := currentTime.Sub(timing.lastTokenTime).Milliseconds()
				if timeSincePrevToken > 0 {
					klog.V(5).Infof("Decoded token received, requestID: %s, podIP: %s, tpot_ms: %d, AddPodMetric", requestID, selectedPodIP, timeSincePrevToken)
					utils.MetricsTracker.AddPodMetric(selectedPodIP, utils.PodMetric{
						RequestID:       requestID,
						Timestamp:       currentTime,
						TTFT:            0,
						TPOT:            timeSincePrevToken,
						PrefillTokenNum: 0,
						DecodeTokenNum:  timing.decodeTokenCount,
					})
				}
			}
			klog.V(5).Infof("Token received, requestID: %s, timing.decodeTokenCount: %d, timing.prefillTokenCount: %d", requestID, timing.decodeTokenCount, timing.prefillTokenCount)
			timing.lastTokenTime = currentTime
			timing.totalTokenCount++
		}
	}

	if err := streaming.Err(); err != nil {
		klog.ErrorS(err, "error processing streaming response", "requestID", requestID)

		complete := true
		errorResponse := generateErrorResponse(
			envoyTypePb.StatusCode_InternalServerError,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorStreaming, RawValue: []byte("true"),
			}}},
			err.Error())

		return existingUsage, complete, errorResponse
	}

	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])

		if !strings.HasPrefix(line, "data:") || line == "data: [DONE]" {
			continue
		}

		cleanLine := strings.TrimPrefix(line, "data: ")

		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(cleanLine), &chunk); err != nil {
			continue
		}

		if usageMap, ok := chunk["usage"].(map[string]interface{}); ok {
			promptTokens := int64(usageMap["prompt_tokens"].(float64))
			completionTokens := int64(usageMap["completion_tokens"].(float64))
			totalTokens := int64(usageMap["total_tokens"].(float64))

			if promptTokens > 0 || completionTokens > 0 || totalTokens > 0 {
				newUsage := openai.CompletionUsage{
					PromptTokens:     promptTokens,
					CompletionTokens: completionTokens,
					TotalTokens:      totalTokens,
				}

				s.streamingUsageCache.Store(requestID, newUsage)

				return newUsage, false, nil
			}
		}
	}

	return existingUsage, false, nil
}

func (s *Server) HandleResponseBody(ctx context.Context, req *extProcPb.ProcessingRequest, user utils.User, rpm int64, model string, stream bool, traceTerm int64, hasCompleted bool) (*extProcPb.ProcessingResponse, bool) {
	b := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)
	var res openai.ChatCompletion
	var usage openai.CompletionUsage
	var promptTokens, completionTokens int64
	var headers []*configPb.HeaderValueOption
	complete := hasCompleted
	routerCtx, _ := ctx.(*types.RoutingContext)

	timingObj, exists := utils.RequestTimings.Load(routerCtx.RequestID)
	var timing *RequestTiming
	if exists {
		timing = timingObj.(*RequestTiming)
	}
	currentTime := time.Now()
	if timing != nil {
		if stream {
			usage_, complete, errorResponse := s.handleStreamingResponse(routerCtx.RequestID, b.ResponseBody.GetBody())
			usage = usage_
			if errorResponse != nil {
				return errorResponse, complete
			}
		} else {
			buf, _ := s.requestBuffers.LoadOrStore(routerCtx.RequestID, &bytes.Buffer{})
			buffer := buf.(*bytes.Buffer)
			buffer.Write(b.ResponseBody.Body)
			if timing.firstTokenTime.IsZero() && b.ResponseBody.EndOfStream {
				timing.firstTokenTime = currentTime
			}
			if !b.ResponseBody.EndOfStream {
				return &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{},
						},
					},
				}, complete
			}
			finalBody := buffer.Bytes()
			s.requestBuffers.Delete(routerCtx.RequestID)
			if err := json.Unmarshal(finalBody, &res); err != nil {
				klog.ErrorS(err, "error to unmarshal response", "requestID", routerCtx.RequestID)
				complete = true
				return generateErrorResponse(
					envoyTypePb.StatusCode_InternalServerError,
					[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
						Key: HeaderErrorResponseUnmarshal, RawValue: []byte("true"),
					}}},
					err.Error()), complete
			} else if len(res.Model) == 0 {
				msg := ErrorUnknownResponse.Error()
				responseBodyContent := string(finalBody)
				if len(responseBodyContent) != 0 {
					msg = responseBodyContent
				}
				klog.ErrorS(nil, "unexpected response", "requestID", routerCtx.RequestID)
				complete = true
				return generateErrorResponse(
					envoyTypePb.StatusCode_InternalServerError,
					[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
						Key: HeaderErrorResponseUnknown, RawValue: []byte("true"),
					}}},
					msg), complete
			} else {
				if len(res.Choices) > 0 && res.Choices[0].Message.Content != "" {
					klog.V(5).Infof("CONFIRMATION - RequestID: %s, Generated text (%d tokens): %s", routerCtx.RequestID, usage.CompletionTokens, res.Choices[0].Message.Content)
				}
			}
			usage = res.Usage
		}
		if b.ResponseBody.EndOfStream {
			// if routerCtx.Algorithm == "preble" {
			ret := utils.DecrementNumDecodeTokensForPod(routerCtx.TargetAddressWithoutPort(), int(timing.totalTokenCount))
			klog.V(5).Infof("DecrementNumDecodeTokensForPod(%s) by %d, %d", routerCtx.TargetAddressWithoutPort(), timing.totalTokenCount, ret)

			klog.Infof("numOutputTokens(usage.CompletionTokens): %d", usage.CompletionTokens)
			hash_of_prefixHashes, _ := utils.GetHashOfPrefixHashesForRequest(routerCtx.RequestID)
			utils.SetNumOutputTokensForHashOfPrefix(hash_of_prefixHashes, int(usage.CompletionTokens))

			timingHeaders, logMessage := s.calculateTimingMetrics(timing, currentTime, routerCtx, stream, usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
			// if utils.UseRealRequest == "true" {
			utils.AddRequestLogMessage(routerCtx.RequestID, logMessage)
			// }
			utils.CleanupKVCacheHitRatio(routerCtx.RequestID)
			utils.CleanupInflightRequests(routerCtx.RequestID)
			utils.CleanupvLLMGPUKVCacheUsage(routerCtx.RequestID)
			// utils.CleanupvLLMCPUKVCacheUsage(routerCtx.RequestID)
			utils.CleanupvLLMNumRequestsRunning(routerCtx.RequestID)
			utils.CleanupvLLMNumRequestsWaiting(routerCtx.RequestID)
			utils.CleanupNumPrefillTokensForRequest(routerCtx.RequestID)
			utils.CleanupNumDecodeTokensForRequest(routerCtx.RequestID)
			utils.CleanupRequestPodMetrics(routerCtx.RequestID)
			utils.CleanupRawMessageForRequest(routerCtx.RequestID)
			utils.CleanupByteArrayPrefillTokensForRequest(routerCtx.RequestID)
			utils.CleanupHashOfPrefixHashesForRequest(routerCtx.RequestID)
			utils.CleanupExploration(routerCtx.RequestID)
			utils.CleanupPredictedLatencies(routerCtx.RequestID)
			utils.CleanupChosenPodPredictedLatency(routerCtx.RequestID)
			headers = append(headers, timingHeaders...)
			utils.RequestTimings.Delete(routerCtx.RequestID)
			s.routingContexts.Delete(routerCtx.RequestID)
			s.requestHeaders.Delete(routerCtx.RequestID) // Clean up cached request headers
		}
	}
	if usage.TotalTokens > 0 {
		complete = true
		promptTokens = usage.PromptTokens
		completionTokens = usage.CompletionTokens
		if user.Name != "" {
			tpm, err := s.ratelimiter.Incr(ctx, fmt.Sprintf("%v_TPM_CURRENT", user), usage.TotalTokens)
			if err != nil {
				return generateErrorResponse(
					envoyTypePb.StatusCode_InternalServerError,
					[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
						Key: HeaderErrorIncrTPM, RawValue: []byte("true"),
					}}},
					err.Error()), complete
			}
			headers = append(headers,
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderUpdateRPM,
						RawValue: []byte(fmt.Sprintf("%d", rpm)),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderUpdateTPM,
						RawValue: []byte(fmt.Sprintf("%d", tpm)),
					},
				},
			)
		}
		if routerCtx != nil {
			headers = append(headers,
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderTargetPod,
						RawValue: []byte(routerCtx.TargetAddress()),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderRequestID,
						RawValue: []byte(routerCtx.RequestID),
					},
				},
			)
		}
	}

	defer func() {
		if !hasCompleted && complete {
			s.cache.DoneRequestTrace(routerCtx, routerCtx.RequestID, model, promptTokens, completionTokens, traceTerm)
			if routerCtx != nil {
				routerCtx.Delete()
			}
		}
	}()

	if stream {
		t := &http.Response{
			Body: io.NopCloser(bytes.NewReader(b.ResponseBody.GetBody())),
		}
		streaming := ssestream.NewStream[openai.ChatCompletionChunk](ssestream.NewDecoder(t), nil)
		defer func() {
			_ = streaming.Close()
		}()
		for streaming.Next() {
			evt := streaming.Current()
			if len(evt.Choices) == 0 {
				// Do not overwrite model, res can be empty.
				usage = evt.Usage
			}
		}
		if err := streaming.Err(); err != nil {
			klog.ErrorS(err, "error to unmarshal response", "requestID", routerCtx.RequestID, "responseBody", string(b.ResponseBody.GetBody()))
			complete = true
			return generateErrorResponse(
				envoyTypePb.StatusCode_InternalServerError,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorStreaming, RawValue: []byte("true"),
				}}},
				err.Error()), complete
		}
	} else {
		buf, _ := requestBuffers.LoadOrStore(routerCtx.RequestID, &bytes.Buffer{})
		buffer := buf.(*bytes.Buffer)
		buffer.Write(b.ResponseBody.Body)
		if !b.ResponseBody.EndOfStream {
			return &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseBody{
					ResponseBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{},
					},
				},
			}, complete
		}
		finalBody := buffer.Bytes()
		requestBuffers.Delete(routerCtx.RequestID)
		if err := json.Unmarshal(finalBody, &res); err != nil {
			klog.ErrorS(err, "error to unmarshal response", "requestID", routerCtx.RequestID, "responseBody", string(b.ResponseBody.GetBody()))
			complete = true
			return generateErrorResponse(
				envoyTypePb.StatusCode_InternalServerError,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorResponseUnmarshal, RawValue: []byte("true"),
				}}},
				err.Error()), complete
		} else if len(res.Model) == 0 {
			msg := ErrorUnknownResponse.Error()
			responseBodyContent := string(b.ResponseBody.GetBody())
			if len(responseBodyContent) != 0 {
				msg = responseBodyContent
			}
			klog.ErrorS(err, "unexpected response", "requestID", routerCtx.RequestID, "responseBody", responseBodyContent)
			complete = true
			return generateErrorResponse(
				envoyTypePb.StatusCode_InternalServerError,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorResponseUnknown, RawValue: []byte("true"),
				}}},
				msg), complete
		}
		// Do not overwrite model, res can be empty.
		usage = res.Usage
	}

	var requestEnd string
	if usage.TotalTokens != 0 {
		complete = true
		// Update promptTokens and completeTokens
		promptTokens = usage.PromptTokens
		completionTokens = usage.CompletionTokens
		// Count token per user.
		if user.Name != "" {
			tpm, err := s.ratelimiter.Incr(ctx, fmt.Sprintf("%v_TPM_CURRENT", user), res.Usage.TotalTokens)
			if err != nil {
				return generateErrorResponse(
					envoyTypePb.StatusCode_InternalServerError,
					[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
						Key: HeaderErrorIncrTPM, RawValue: []byte("true"),
					}}},
					err.Error()), complete
			}

			headers = append(headers,
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderUpdateRPM,
						RawValue: []byte(fmt.Sprintf("%d", rpm)),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderUpdateTPM,
						RawValue: []byte(fmt.Sprintf("%d", tpm)),
					},
				},
			)
			requestEnd = fmt.Sprintf(requestEnd+"rpm: %s, tpm: %s, ", rpm, tpm)
		}
		if routerCtx != nil {
			targetPodName := routerCtx.TargetName()
			targetPodIP := routerCtx.TargetAddress()
			headers = append(headers,
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderTargetPod,
						RawValue: []byte(targetPodIP),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderRequestID,
						RawValue: []byte(routerCtx.RequestID),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderTargetPodName,
						RawValue: []byte(targetPodName),
					},
				},
			)
			requestEnd = fmt.Sprintf(requestEnd+"targetPod: %s", targetPodIP)
		}

		klog.Infof("request end, requestID: %s - %s", routerCtx.RequestID, requestEnd)
	} else if b.ResponseBody.EndOfStream {
		complete = true
	}

	// klog.Infof("SetHeaders: %s", headers)

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: headers,
					},
				},
			},
		},
	}, complete
}

// MetricsData holds all the performance metrics for a request
type MetricsData struct {
	TTFT               int64              `json:"ttft_ms"`
	TPOT               int64              `json:"tpot_ms"`
	E2ELatency         int64              `json:"e2e_latency_ms"`
	KVCacheHitRatio    float64            `json:"kv_cache_hit_ratio"`
	AllPodsRatios      map[string]float64 `json:"all_pods_ratios,omitempty"`
	InflightRequests   map[string]int     `json:"inflight_requests,omitempty"`
	GPUKVCacheUsage    map[string]float64 `json:"gpu_kv_cache_usage,omitempty"`
	CPUKVCacheUsage    map[string]float64 `json:"cpu_kv_cache_usage,omitempty"`
	NumRequestsRunning map[string]float64 `json:"num_requests_running,omitempty"`
	NumRequestsWaiting map[string]float64 `json:"num_requests_waiting,omitempty"`
	InputTokens        int64              `json:"input_tokens"`
	OutputTokens       int64              `json:"output_tokens"`
	TotalTokens        int64              `json:"total_tokens"`
	SelectedPod        string             `json:"selected_pod"`
}

// Helper function to add a JSON metric to headers
func addMetricToHeaders(headers []*configPb.HeaderValueOption, key string, data interface{}, lock *sync.RWMutex) ([]*configPb.HeaderValueOption, string) {
	lock.RLock()
	defer lock.RUnlock()
	jsonData, err := json.Marshal(data)
	jsonStr := "{}"
	if err == nil {
		jsonStr = string(jsonData)
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      key,
				RawValue: jsonData,
			},
		})
	}
	return headers, jsonStr
}

func (s *Server) calculateTimingMetrics(timing *RequestTiming, currentTime time.Time, routerCtx *types.RoutingContext, stream bool, numInputTokens int64, numOutputTokens int64, numTotalTokens int64) ([]*configPb.HeaderValueOption, string) {
	klog.V(5).Infof("requestID: %s, timing.decodeTokenCount: %d, timing.prefillTokenCount: %d", routerCtx.RequestID, timing.decodeTokenCount, timing.prefillTokenCount)

	ttftMs := int64(0)
	if !timing.firstTokenTime.IsZero() {
		ttftMs = timing.firstTokenTime.Sub(timing.startTime).Milliseconds()
	}
	avgTpotMs := int64(0)
	totalGenerationTimeMs := int64(0)
	if !timing.firstTokenTime.IsZero() {
		totalGenerationTimeMs = currentTime.Sub(timing.firstTokenTime).Milliseconds()
		effectiveTokenCount := int64(0)
		if stream && timing.decodeTokenCount > 1 {
			effectiveTokenCount = int64(timing.decodeTokenCount - 1) // Exclude first token
		} else if numOutputTokens > 1 {
			effectiveTokenCount = numOutputTokens - 1
		}
		if effectiveTokenCount > 0 {
			avgTpotMs = totalGenerationTimeMs / effectiveTokenCount
			klog.V(5).Infof("ttftMS:%d, avgTpotMs: %d, totalGenerationTimeMs: %d, effectiveTokenCount: %d", ttftMs, avgTpotMs, totalGenerationTimeMs, effectiveTokenCount)
		}
	}

	end_to_end_latency_in_ms := time.Since(timing.startTime).Milliseconds()

	// Initialize headers with basic metrics
	headers := []*configPb.HeaderValueOption{
		{
			Header: &configPb.HeaderValue{
				Key:      HeaderTTFT,
				RawValue: []byte(fmt.Sprintf("%d", ttftMs)),
			},
		},
		{
			Header: &configPb.HeaderValue{
				Key:      HeaderTPOT,
				RawValue: []byte(fmt.Sprintf("%d", avgTpotMs)),
			},
		},
		{
			Header: &configPb.HeaderValue{
				Key:      HeaderE2ELatency,
				RawValue: []byte(fmt.Sprintf("%d", end_to_end_latency_in_ms)),
			},
		},
	}

	// Prepare for JSON strings to use in logging
	var jsonStrings = make(map[string]string)

	// 1. KV cache hit ratios
	allPodsKvCacheHitRatios := utils.GetAllPodsKVCacheHitRatios(routerCtx.RequestID)
	headers, jsonStrings["allPodsKvCacheHitRatios"] = addMetricToHeaders(headers, HeaderKVCacheHitRatioAllPods, allPodsKvCacheHitRatios, utils.GetrequestAllPodsKVCacheMutex())

	// 2. Inflight requests
	numInflightRequestsAllPods := utils.GetInflightRequestsForAllPods(routerCtx.RequestID)
	headers, jsonStrings["numInflightRequestsAllPods"] = addMetricToHeaders(headers, HeaderNumInflightRequestsAllPods, numInflightRequestsAllPods, utils.GetrequestInflightMutex())
	utils.DecrementNumInflightForPod(routerCtx.RequestID, routerCtx.TargetAddressWithoutPort())

	// 3. GPU KV cache usage
	vllmGPUKVCacheUsage, err := utils.GetvLLMGPUKVCacheUsageForAllPods(routerCtx.RequestID)
	if err == nil {
		headers, jsonStrings["vllmGPUKVCacheUsage"] = addMetricToHeaders(headers, HeadervLLMGPUKVCacheUsage, vllmGPUKVCacheUsage, utils.GetvllmGPUKVCacheUsageMutex())
	} else {
		jsonStrings["vllmGPUKVCacheUsage"] = "{}"
	}

	// // 4. CPU KV cache usage
	// vllmCPUKVCacheUsage, err := utils.GetvLLMCPUKVCacheUsageForTheRequestForAllPods(routerCtx.RequestID)
	// if err == nil {
	// 	headers, jsonStrings["vllmCPUKVCacheUsage"] = addMetricToHeaders(headers, HeadervLLMCPUKVCacheUsage, vllmCPUKVCacheUsage, utils.GetvllmCPUKVCacheUsageMutex())
	// } else {
	// 	klog.ErrorS(err, "error to get vllm cpu kv cache usage, fill vllmCPUKVCacheUsage with empty map {}", "requestID", routerCtx.RequestID)
	// 	jsonStrings["vllmCPUKVCacheUsage"] = "{}"
	// }
	jsonStrings["vllmCPUKVCacheUsage"] = "{}"

	// 5. Number of running requests
	vllmNumRequestsRunning, err := utils.GetvLLMNumRequestsRunningForAllPods(routerCtx.RequestID)
	if err == nil {
		headers, jsonStrings["vllmNumRequestsRunning"] = addMetricToHeaders(headers, HeadervLLMNumRunningRequests, vllmNumRequestsRunning, utils.GetvllmNumRequestsRunningMutex())
	} else {
		jsonStrings["vllmNumRequestsRunning"] = "{}"
	}

	// 6. Number of waiting requests
	vllmNumRequestWaiting, err := utils.GetvLLMNumRequestsWaitingForAllPods(routerCtx.RequestID)
	if err == nil {
		headers, jsonStrings["vllmNumRequestWaiting"] = addMetricToHeaders(headers, HeadervLLMNumwWaitingRequests, vllmNumRequestWaiting, utils.GetvllmNumRequestsWaitingMutex())
	} else {
		jsonStrings["vllmNumRequestWaiting"] = "{}"
	}

	numPrefillTokensForAllPods := utils.GetNumPrefillTokensForAllPods()
	headers, jsonStrings["numPrefillTokensForAllPods"] = addMetricToHeaders(headers, HeaderNumPrefillTokensForAllPods, numPrefillTokensForAllPods, utils.GetpodTotalPrefillTokensMutex())

	numDecodeTokensForAllPods := utils.GetNumDecodeTokensForAllPods()
	headers, jsonStrings["numDecodeTokensForAllPods"] = addMetricToHeaders(headers, HeaderNumDecodeTokensForAllPods, numDecodeTokensForAllPods, utils.GetpodTotalDecodeTokensMutex())

	// Get selected pod
	selectedPodIP := "unknown"
	if routerCtx != nil {
		selectedPodIP = routerCtx.TargetAddressWithoutPort()
	}

	ts := time.Now()
	podDetailedMetrics := utils.GetRequestPodMetrics(routerCtx.RequestID)
	klog.V(5).Infof("GetAndCleanupRequestPodMetrics took %d, %s, %s", time.Since(ts).Milliseconds(), routerCtx.RequestID, selectedPodIP)
	headers, jsonStrings["podMetricsLastSecond"] = addMetricToHeaders(headers, HeaderPodDetailedMetrics, podDetailedMetrics, utils.MetricsTracker.GetMutex())

	// log_window_end_time := time.Now()
	// log_window_start_time := time.Now().Add(utils.MetricsTracker.WindowSize * -1)
	if utils.FirstRequestStartTime == 0 {
		utils.FirstRequestStartTime = timing.startTime.UnixMicro()
	}
	normalized_request_start_time := timing.startTime.UnixMicro() - utils.FirstRequestStartTime
	normalized_request_end_time := currentTime.UnixMicro() - utils.FirstRequestStartTime
	exploration, explorationEnabled := utils.GetExploration(routerCtx.RequestID)

	// Get predicted latencies
	predictedLatencies := utils.GetPredictedLatencies(routerCtx.RequestID)
	headers, jsonStrings["predictedLatencies"] = addMetricToHeaders(headers, HeaderPredictedLatencies, predictedLatencies, utils.GetPredictedLatenciesMutex())

	logMessage := fmt.Sprintf("**@latency_metrics@requestID@%s@request_start_time@%d@request_end_time@%d@selectedpod@%s@ttft@%d@avg_tpot@%d@total_decode_time@%d@e2e@%d@numInputTokens@%d@numOutputTokens@%d@numTotalTokens@%d@allPodsKvCacheHitRatios@%s@numInflightRequestsAllPods@%s@vllmGPUKVCacheUsage@%s@vllmCPUKVCacheUsage@%s@vllmNumRequestsRunning@%s@vllmNumRequestsWaiting@%s@numPrefillTokensForAllPods@%s@numDecodeTokensForAllPods@%s@numTrains@%d@numFlush@%d@exploration@%d@explorationEnabled@%d@predictedLatencies@%s@chosenPodPredictedLatency@%f@iteration@%d@subAlgorithm@%s",
		routerCtx.RequestID,
		normalized_request_start_time,
		normalized_request_end_time,
		selectedPodIP,
		ttftMs,
		avgTpotMs,
		totalGenerationTimeMs,
		end_to_end_latency_in_ms,
		numInputTokens,
		numOutputTokens,
		numTotalTokens,
		jsonStrings["allPodsKvCacheHitRatios"],
		jsonStrings["numInflightRequestsAllPods"],
		jsonStrings["vllmGPUKVCacheUsage"],
		jsonStrings["vllmCPUKVCacheUsage"],
		jsonStrings["vllmNumRequestsRunning"],
		jsonStrings["vllmNumRequestWaiting"],
		jsonStrings["numPrefillTokensForAllPods"],
		jsonStrings["numDecodeTokensForAllPods"],
		utils.GetNumTrains(),
		utils.GetNumFlush(),
		exploration,
		explorationEnabled,
		jsonStrings["predictedLatencies"],
		utils.GetChosenPodPredictedLatency(routerCtx.RequestID),
		routerCtx.Iteration,
		routerCtx.SubAlgorithm,
	)

	// logMessage := fmt.Sprintf("**@latency_metrics@requestID@%s@request_start_time@%d@request_end_time@%d@selectedpod@%s@ttft@%d@avg_tpot@%d@total_decode_time@%d@e2e@%d@numInputTokens@%d@numOutputTokens@%d@numTotalTokens@%d@allPodsKvCacheHitRatios@%s@numInflightRequestsAllPods@%s@vllmGPUKVCacheUsage@%s@vllmCPUKVCacheUsage@%s@vllmNumRequestsRunning@%s@vllmNumRequestsWaiting@%s@podMetricsLastSecond@%s@numPrefillTokensForAllPods@%s@numDecodeTokensForAllPods@%s@numTrains@%d@numFlush@%d@exploration@%d@explorationEnabled@%d",
	// 	routerCtx.RequestID,
	// 	normalized_request_start_time,
	// 	normalized_request_end_time,
	// 	selectedPodIP,
	// 	ttftMs,
	// 	avgTpotMs,
	// 	totalGenerationTimeMs,
	// 	end_to_end_latency_in_ms,
	// 	numInputTokens,
	// 	numOutputTokens,
	// 	numTotalTokens,
	// 	jsonStrings["allPodsKvCacheHitRatios"],
	// 	jsonStrings["numInflightRequestsAllPods"],
	// 	jsonStrings["vllmGPUKVCacheUsage"],
	// 	jsonStrings["vllmCPUKVCacheUsage"],
	// 	jsonStrings["vllmNumRequestsRunning"],
	// 	jsonStrings["vllmNumRequestWaiting"],
	// 	jsonStrings["podMetricsLastSecond"],
	// 	jsonStrings["numPrefillTokensForAllPods"],
	// 	jsonStrings["numDecodeTokensForAllPods"],
	// 	utils.GetNumTrains(),
	// 	utils.GetNumFlush(),
	// 	exploration,
	// 	explorationEnabled,
	// )
	klog.Infof("%s", logMessage)

	// Write to file asynchronously (non-blocking)
	writeLatencyMetricsLog(logMessage)

	// Notify scalable RL agent of request completion (async, non-blocking)
	notifyRLAgentRequestComplete(routerCtx, ttftMs, avgTpotMs)

	return headers, logMessage
}

///////////////////////////////////////////////////////////////////////////

// Helper functions for metrics tracking
// IsMetricsEnabled returns whether metrics collection is enabled
func (s *Server) IsMetricsEnabled() bool {
	return utils.MetricsEnabled.Load()
}

// EnableMetrics enables metrics collection
func (s *Server) EnableMetrics() {
	utils.MetricsEnabled.Store(true)
	klog.Info("Metrics collection enabled")
}

// DisableMetrics disables metrics collection
func (s *Server) DisableMetrics() {
	utils.MetricsEnabled.Store(false)
	klog.Info("Metrics collection disabled")
}
