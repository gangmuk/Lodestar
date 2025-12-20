package routingalgorithms

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	"github.com/vllm-project/aibrix/pkg/utils/prefixcacheindexer"
	"github.com/vllm-project/aibrix/pkg/utils/tokenizer"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

var (
	enableFlush                = utils.LoadEnvInt("ENABLE_FLUSH", 0)
	flushPeriod                = time.Duration(utils.LoadEnvInt("FLUSH_PERIOD", 10)) * time.Second
	minNumLogMessagesToFlush   = utils.LoadEnvInt("MIN_NUM_LOG_MESSAGES_TO_FLUSH", 100)
	useRealRequest             = utils.LoadEnvInt("useRealRequest", 1)
	flushed                    = false
	received_the_first_request = false
	allPodIPs                  = []string{}
	fake_request_id            = 0
	numFlush                   = 0
)

var (
	httpClientForRLAgent = &http.Client{
		Timeout: 30000 * time.Millisecond,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     180 * time.Second,
			DisableCompression:  false,
			DialContext: (&net.Dialer{
				Timeout:   5000 * time.Millisecond,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   5000 * time.Millisecond,
			ForceAttemptHTTP2:     true, // Enable HTTP/2
			ResponseHeaderTimeout: 5 * time.Second,
		},
	}
	routingAgentURL = "http://routing-agent-service.default.svc.cluster.local:8080"
	inferEndpoint   = "/infer"
	flushEndpoint   = "/flush"
)

type rlOnlineRouter struct {
	prefixCacheIndexer *prefixcacheindexer.PrefixHashTable
	tokenizer          tokenizer.Tokenizer
	cache              cache.Cache
}

func NewRLOnlineRouter() (types.Router, error) {
	var tokenizerObj tokenizer.Tokenizer
	// tokenizerObj = tokenizer.NewTiktokenTokenizer()
	// tokenizerObj = tokenizer.NewCharacterTokenizer()
	tokenizerObj = tokenizer.NewWordTokenizer()

	c, err := cache.Get()
	if err != nil {
		klog.Error("fail to get cache store in prefix cache router")
		return nil, err
	}

	router := &rlOnlineRouter{
		cache:              c,
		tokenizer:          tokenizerObj,
		prefixCacheIndexer: prefixcacheindexer.NewPrefixHashTable(),
	}

	// Initialize GPU models for all pods in the background
	go initializeGPUModels(c)

	klog.InfoS("Created RL online router")
	return router, nil
}

//  pre-populates GPU model information for all pods
// This runs in the background to avoid blocking router initialization
func initializeGPUModels(c cache.Cache) {
	// Wait a bit for pods to be registered in cache
	klog.InfoS("Starting GPU model initialization for all pods")

	// Get all models
	models := c.ListModels()
	if len(models) == 0 {
		klog.Warningf("No models found during GPU initialization, will retry on first request")
		return
	}

	// For each model, get all pods
	podCount := 0
	for _, model := range models {
		pods, err := c.ListPodsByModel(model)
		if err != nil {
			klog.Warningf("Failed to list pods for model %s during GPU init: %v", model, err)
			continue
		}

		// Extract and cache GPU model for each pod
		for _, pod := range pods.All() {
			if pod.Status.PodIP == "" {
				continue
			}

			// Check if already cached
			if _, exists := utils.GetGPUModel(pod.Status.PodIP); exists {
				continue
			}

			// Extract GPU model from pod
			gpuModel := extractGPUModelFromPod(pod)
			utils.SetGPUModel(pod.Status.PodIP, gpuModel)
			podCount++
			klog.V(4).Infof("Initialized GPU model for pod %s: %s", pod.Status.PodIP, gpuModel)
		}
	}

	klog.InfoS("GPU model initialization completed", "podsInitialized", podCount)
}

// // flush real request log collected
func FlushLogMessageToRLAgent() {
	if useRealRequest == 1 {
		klog.Infof("flushing real request log to RL agent")
		done := make(chan struct{})
		go func() {
			ticker := time.NewTicker(flushPeriod)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					if !received_the_first_request {
						klog.Infof("The first request has not been received yet, skipping the flush. (needed to construct the running pod IPs)")
						continue // Skip this iteration and check again on the next tick
					}
					utils.RequestToLogMessageMutex.RLock()
					numMessages := len(utils.RequestToLogMessage)
					utils.RequestToLogMessageMutex.RUnlock()

					if numMessages > minNumLogMessagesToFlush {
						klog.Infof("Starting flushing %dth flush for %d number of log messages", utils.GetNumFlush(), numMessages)
						utils.RequestToLogMessageMutex.RLock()
						reqBody, err := json.Marshal(utils.RequestToLogMessage)
						utils.RequestToLogMessageMutex.RUnlock()
						if err != nil {
							klog.Errorf("Failed flush. failed marshal RequestToLogMessage: %v", err)
							utils.CleanupAllRequestLogMessage()
							continue
						}
						url := fmt.Sprintf("%s%s", routingAgentURL, flushEndpoint)
						req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
						if reqErr != nil {
							klog.Errorf("Failed flush. failed to create request: %v", reqErr)
							utils.CleanupAllRequestLogMessage()
							continue
						}
						req.Header.Set("Content-Type", "application/json")
						resp, sendErr := httpClientForRLAgent.Do(req) // flush request
						if sendErr != nil {
							klog.Errorf("Failed flush. failed to send request: %v", sendErr)
							utils.CleanupAllRequestLogMessage()
							continue
						}
						if resp.StatusCode != http.StatusOK {
							klog.Errorf("Received non-200 response: %s", resp.Status)
							utils.CleanupAllRequestLogMessage()
							klog.Errorf("Failed flush. Received non-200 response: %s", resp.Status)
							continue
						}
						body, readErr := ioutil.ReadAll(resp.Body)
						if readErr != nil {
							klog.Errorf("Failed to read response body: %v", readErr)
							utils.CleanupAllRequestLogMessage()
							klog.Errorf("Failed flush. Failed to read response body: %v", readErr)
							continue
						}
						resp.Body.Close()
						utils.CleanupAllRequestLogMessage()
						klog.Infof("Successfully flush, response: %s", string(body))
						flushed = true
						numFlush += 1
					} else {
						klog.Infof("Not enough log messages to flush: %d", len(utils.RequestToLogMessage))
					}
				case <-done:
					klog.Info("Flushing goroutine is shutting down")
					return
				}
			}
		}()
	} else { // useRealRequest == 0
		done := make(chan struct{})
		go func() {
			ticker := time.NewTicker(flushPeriod)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					if !received_the_first_request {
						klog.Infof("The first request has not been received yet, skipping the flush")
						continue // Skip this iteration and check again on the next tick
					}

					// if flushed {
					// 	// flush only once to simplify experiment
					// 	klog.Infof("Skip flushing. Configured to flush only once for Fake data, utils.UseRealRequest == false")
					// 	continue
					// }

					// If we got here, the first request has been received
					klog.Infof("Start flushing log messages to RL agent, %dth flush", numFlush)
					allPodIPs = utils.GetAllPodIPsFromRegistry()
					klog.Infof("All pod IPs: %v", allPodIPs)

					logs := utils.GenerateLogMessages(allPodIPs, minNumLogMessagesToFlush)
					start_request_id_of_this_batch := fake_request_id
					for _, log := range logs {
						utils.AddRequestLogMessage(fmt.Sprintf("%d", fake_request_id), log)
						fake_request_id += 1
					}
					end_request_id_of_this_batch := fake_request_id
					klog.Infof("Newly added request ids %d-%d", start_request_id_of_this_batch, end_request_id_of_this_batch)
					utils.RequestToLogMessageMutex.RLock()
					numLogs := len(utils.RequestToLogMessage)
					klog.Infof("Starting flushing process for %d logs ", numLogs)
					klog.V(5).Infof("logs: %v", logs)
					reqBody, err := json.Marshal(utils.RequestToLogMessage)
					utils.RequestToLogMessageMutex.RUnlock()
					if err != nil {
						klog.Errorf("Failed flush. failed to marshal RequestToLogMessage: %v", err)
						continue
					}

					url := fmt.Sprintf("%s%s", routingAgentURL, flushEndpoint)
					req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
					if reqErr != nil {
						klog.Errorf("Failed flush. failed to create request: %v", reqErr)
						continue
					}

					req.Header.Set("Content-Type", "application/json")
					resp, sendErr := httpClientForRLAgent.Do(req)
					if sendErr != nil {
						klog.Errorf("Failed flush. failed to send request: %v", sendErr)
						continue
					}

					// Ensure we have a valid response before proceeding
					if resp != nil {
						if resp.StatusCode != http.StatusOK {
							klog.Errorf("Received non-200 response: %s", resp.Status)
						}

						body, readErr := ioutil.ReadAll(resp.Body)
						if readErr != nil {
							klog.Errorf("Failed flush. failed to read response body: %v", readErr)
						} else {
							klog.Infof("Successfully sent RequestToLogMessage to RL agent: %s", string(body))
						}
						resp.Body.Close()
					}

					//// Delete when the RL agent is doing continuous learning.
					//// When the RL agent trains the model from scratch at every flush call, don't discard previous logs but flush all history every time.
					// utils.CleanupAllRequestLogMessage()
					// flushed = true
					numFlush += 1
				case <-done:
					return
				}
			}
		}()
	}
}

func init() {
	RegisterDelayedConstructor("rl-online-router", NewRLOnlineRouter)
	klog.Infof("enableFlush: %d", enableFlush)
	klog.Infof("flushPeriod: %d", flushPeriod)
	klog.Infof("minNumLogMessagesToFlush: %d", minNumLogMessagesToFlush)
	if enableFlush == 1 {
		FlushLogMessageToRLAgent()
	}
}

// RouteResponse is received from the routing agent
type RouteResponse struct {
	RequestID                 string             `json:"request_id"`
	SelectedPod               string             `json:"selected_pod"`
	SelectedPodGeneralPodId   string             `json:"selected_pod_generalpodid"`
	Confidence                float64            `json:"confidence"`
	NumTrains                 int                `json:"num_trains"`
	NumFlush                  int                `json:"num_flush"`
	Exploration               int                `json:"exploration"`
	ExplorationEnabled        int                `json:"exploration_enabled"`
	OverheadLog               string             `json:"overhead_log"`
	PredictedLatencies        map[string]float64 `json:"predicted_latencies"`
	PredictedRewards          map[string]float64 `json:"predicted_rewards"`
	ChosenPodPredictedLatency float64            `json:"chosen_pod_predicted_latency"`
	ChosenPodPredictedReward  float64            `json:"chosen_pod_predicted_reward"`
}

func jsonStringify(data interface{}, lock *sync.RWMutex) string {
	lock.RLock()
	defer lock.RUnlock()
	jsonData, err := json.Marshal(data)
	if err != nil {
		klog.Errorf("Error marshaling data to JSON: %v", err)
		return "{}"
	}
	return string(jsonData)
}

func GetPod(podIP string, pods []*v1.Pod) *v1.Pod {
	for _, pod := range pods {
		if pod.Status.PodIP == podIP {
			return pod
		}
	}
	return nil
}

// extracts GPU model information from pod metadata
// It checks multiple sources in order of preference:
// 1. Pod label "machine.cluster.vke.volcengine.com/gpu-name" (if inherited from node)
// 2. Fetch from node (using pod.Spec.NodeName)
// 3. Pod label "gpu-model" or "nvidia.com/gpu.product"
// 4. Pod annotation "gpu-model"
// 5. Falls back to "GPU-L3c" as default
func extractGPUModelFromPod(pod *v1.Pod) string {
	// First, try the VKE GPU label that may be on the pod itself
	if gpuModel, ok := pod.Labels["machine.cluster.vke.volcengine.com/gpu-name"]; ok && gpuModel != "" {
		klog.V(5).Infof("Found GPU model from pod VKE label for pod %s: %s", pod.Status.PodIP, gpuModel)
		return gpuModel
	}

	// Second, try to get GPU model from the node the pod is running on
	if pod.Spec.NodeName != "" {
		if gpuModel, err := utils.GetGPUModelFromNode(pod.Spec.NodeName); err == nil && gpuModel != "" {
			klog.V(5).Infof("Found GPU model from node %s for pod %s: %s", pod.Spec.NodeName, pod.Status.PodIP, gpuModel)
			return gpuModel
		} else if err != nil {
			klog.V(4).Infof("Failed to get GPU model from node %s for pod %s: %v", pod.Spec.NodeName, pod.Status.PodIP, err)
		}
	}

	// Try other common pod labels
	if gpuModel, ok := pod.Labels["gpu-model"]; ok && gpuModel != "" {
		return gpuModel
	}
	if gpuModel, ok := pod.Labels["nvidia.com/gpu.product"]; ok && gpuModel != "" {
		return gpuModel
	}

	// Try pod annotations
	if gpuModel, ok := pod.Annotations["gpu-model"]; ok && gpuModel != "" {
		return gpuModel
	}

	// Default fallback - since we can't determine GPU type, use most common
	klog.Errorf("No GPU model found for pod %s (node: %s), using default GPU-L3c", pod.Status.PodIP, pod.Spec.NodeName)
	return "GPU-L3c"
}

// Route selects the optimal pod based on latency predictions
func (r *rlOnlineRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	route_start_time := time.Now()
	// Get all ready pods
	readyPods := pods.All()
	var targetPod *v1.Pod = nil
	if !received_the_first_request {
		klog.Infof("This is the first request, using fallback routing and return right away. Give some time for the RL agent to warm up.")
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		received_the_first_request = true
		allPodIPs = utils.GetAllPodIPsFromRegistry()
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}

	readyPodsMap := map[string]struct{}{}
	for _, pod := range readyPods {
		readyPodsMap[pod.Status.PodIP] = struct{}{}

		// Set GPU model for this pod if not already cached
		// Check cache first to avoid unnecessary API calls
		if _, exists := utils.GetGPUModel(pod.Status.PodIP); !exists {
			// Only extract if not already cached
			gpuModel := extractGPUModelFromPod(pod)
			utils.SetGPUModel(pod.Status.PodIP, gpuModel)
		}
	}

	// input_tokens, err := r.tokenizer.TokenizeInputText(ctx.Message) // character tokenizer by four

	// input_tokens := utils.GetPrefillTokensForRequest(ctx.RequestID) // word tokenizer by default
	// utils.CleanupPrefillTokensForRequest(ctx.RequestID)
	input_tokens_in_bytearray := utils.GetByteArrayPrefillTokensForRequest(ctx.RequestID)
	// input_message, _ := utils.GetRawMessageForRequest(ctx.RequestID)

	var podIPsWithMatchingRatios map[string]int
	var allPrefixHashes []uint64     // allPrefixHashes is used to store all prefix hashes for the request
	var matchedPrefixHashes []uint64 // matchedPrefixHashes is used to store matched prefix hashes for the request

	// podIPsWithMatchingRatios, allPrefixHashes = r.prefixCacheIndexer.MatchPrefix(input_tokens_in_bytearray, ctx.Model, readyPodsMap)

	podIPsWithMatchingRatios, matchedPrefixHashes, allPrefixHashes = r.prefixCacheIndexer.MatchPrefix_returning_matchedprefixes(input_tokens_in_bytearray, ctx.Model, readyPodsMap)
	numInputTokens := utils.GetNumPrefillTokensForRequest(ctx.RequestID)

	// predict output!
	hash_of_matchedprefix := utils.HashPrefixHashes(matchedPrefixHashes)
	utils.SetHashOfPrefixHashesForRequest(ctx.RequestID, hash_of_matchedprefix)
	expectedNumOutputTokens, exist := utils.GetNumOutputTokensForPrefix(hash_of_matchedprefix)
	expectedNumOutputTokens = 100
	if !exist {
		klog.Infof("requestID: %s, No cached output token length found for hash_of_matchedprefix: %d. Using default value %d", ctx.RequestID, hash_of_matchedprefix, expectedNumOutputTokens)
	}

	numTotalTokens := numInputTokens + expectedNumOutputTokens

	for _, pod := range readyPods {
		if _, ok := podIPsWithMatchingRatios[pod.Status.PodIP]; !ok {
			// klog.Infof("requestID: %s, No found prefix matched pods. Filled all readypods with 0 kv cache hit ratio", ctx.RequestID)
			podIPsWithMatchingRatios[pod.Status.PodIP] = 0
			// podIPsWithMatchingRatios[pod.Name] = 0
		}
	}
	utils.StoreKVCacheHitRatio(ctx.RequestID, podIPsWithMatchingRatios)

	if len(readyPods) == 0 {
		klog.Errorf("requestID: %s, No ready pods available for routing", ctx.RequestID)
		return "", fmt.Errorf("no ready pods available")
	}

	if len(readyPods) == 1 {
		ctx.SetTargetPod(readyPods[0])
		klog.Warningf("requestID: %s, Only one ready pod available, using it as target pod: %s", ctx.RequestID, readyPods[0].Status.PodIP)
		return ctx.TargetAddress(), nil
	}

	var logMessage string
	log_construction_start_time := time.Now()
	if useRealRequest == 1 {
		// Prepare for JSON strings to use in logging
		var jsonStrings = make(map[string]string)

		// 1. KV cache hit ratios
		allPodsKvCacheHitRatios := utils.GetAllPodsKVCacheHitRatios(ctx.RequestID)
		jsonStrings["allPodsKvCacheHitRatios"] = jsonStringify(allPodsKvCacheHitRatios, utils.GetrequestAllPodsKVCacheMutex())
		klog.V(5).Infof("allPodsKvCacheHitRatios: %s", jsonStrings["allPodsKvCacheHitRatios"])

		// 2. Inflight requests
		numInflightRequestsAllPods := utils.GetInflightRequestsForAllPods(ctx.RequestID)
		jsonStrings["numInflightRequestsAllPods"] = jsonStringify(numInflightRequestsAllPods, utils.GetrequestInflightMutex())

		// 3. GPU KV cache usage
		vllmGPUKVCacheUsage, err := utils.GetvLLMGPUKVCacheUsageForAllPods(ctx.RequestID)
		if err == nil {
			jsonStrings["vllmGPUKVCacheUsage"] = jsonStringify(vllmGPUKVCacheUsage, utils.GetvllmGPUKVCacheUsageMutex())
		} else {
			jsonStrings["vllmGPUKVCacheUsage"] = "{}"
		}

		// // 4. CPU KV cache usage
		// vllmCPUKVCacheUsage, err := utils.GetvLLMCPUKVCacheUsageForTheRequestForAllPods(ctx.RequestID)
		// if err == nil {
		// 	jsonStrings["vllmCPUKVCacheUsage"] = jsonStringify(vllmCPUKVCacheUsage, utils.GetvllmCPUKVCacheUsageMutex())
		// } else {
		// 	klog.ErrorS(err, "error to get vllm cpu kv cache usage, fill vllmCPUKVCacheUsage with empty map {}", "requestID", ctx.RequestID)
		// 	// create 0 value map for all pods, with type map[string]float64
		// 	vllmCPUKVCacheUsage = make(map[string]float64)
		// 	for _, pod := range readyPods {
		// 		vllmCPUKVCacheUsage[pod.Status.PodIP] = 0.0
		// 	}
		// 	jsonStrings["vllmCPUKVCacheUsage"] = jsonStringify(vllmCPUKVCacheUsage, utils.GetvllmCPUKVCacheUsageMutex())
		// }
		jsonStrings["vllmCPUKVCacheUsage"] = "{}"

		// 5. Number of running requests
		vllmNumRequestsRunning, err := utils.GetvLLMNumRequestsRunningForAllPods(ctx.RequestID)
		if err == nil {
			jsonStrings["vllmNumRequestsRunning"] = jsonStringify(vllmNumRequestsRunning, utils.GetvllmNumRequestsRunningMutex())
		} else {
			jsonStrings["vllmNumRequestsRunning"] = "{}"
		}

		// 6. Number of waiting requests
		vllmNumRequestWaiting, err := utils.GetvLLMNumRequestsWaitingForAllPods(ctx.RequestID)
		if err == nil {
			jsonStrings["vllmNumRequestWaiting"] = jsonStringify(vllmNumRequestWaiting, utils.GetvllmNumRequestsWaitingMutex())
		} else {
			jsonStrings["vllmNumRequestWaiting"] = "{}"
		}

		numPrefillTokensForAllPods := utils.GetNumPrefillTokensForAllPods()
		jsonStrings["numPrefillTokensForAllPods"] = jsonStringify(numPrefillTokensForAllPods, utils.GetpodTotalPrefillTokensMutex())

		numDecodeTokensForAllPods := utils.GetNumDecodeTokensForAllPods()
		jsonStrings["numDecodeTokensForAllPods"] = jsonStringify(numDecodeTokensForAllPods, utils.GetpodTotalDecodeTokensMutex())

		// Create GPU model map for each pod (proper JSON format like other metrics)
		gpuModelMap := make(map[string]string)
		for _, pod := range readyPods {
			// Get GPU model for each pod, with fallback to default
			gpuModel, exists := utils.GetGPUModel(pod.Status.PodIP)
			if !exists {
				gpuModel = "GPU-L3c" // Default fallback
			}
			gpuModelMap[pod.Status.PodIP] = gpuModel
		}
		// Use a local mutex for this request-scoped data
		var gpuMapMutex sync.RWMutex
		jsonStrings["GPU"] = jsonStringify(gpuModelMap, &gpuMapMutex)
		klog.V(5).Infof("GPU model map JSON: %s", jsonStrings["GPU"])

		podDetailedMetrics := utils.GetRequestPodMetrics(ctx.RequestID)
		jsonStrings["podMetricsLastSecond"] = jsonStringify(podDetailedMetrics, utils.MetricsTracker.GetMutex())

		if utils.FirstRequestStartTime == 0 {
			utils.FirstRequestStartTime = time.Now().UnixMicro()
		}

		prev_reward := 0.0 // total latency in seconds of all live and completed requests
		if ctx.SubAlgorithm == "scalable_rl_agent" {
			cur_time_in_microseconds := time.Now().UnixMicro()
			klog.V(5).Infof("calculate_prev_reward, requestID: %s, cur_time_in_microseconds: %d", ctx.RequestID, cur_time_in_microseconds)

			// Copy live request IDs to avoid concurrent map iteration and write
			utils.LiveRequestsMutex.RLock()
			liveRequestIDs := make([]string, 0, len(utils.LiveRequests))
			for reqID := range utils.LiveRequests {
				liveRequestIDs = append(liveRequestIDs, reqID)
			}
			utils.LiveRequestsMutex.RUnlock()

			// Now safely iterate over the copy
			for _, live_request_id := range liveRequestIDs {
				live_request_last_time, exists := utils.GetLiveRequestLastTime(live_request_id)
				if !exists {
					klog.Errorf("calculate_prev_reward, requestID: %s, live_requestID: %s, not found in LiveRequests. Use default value %d seconds", ctx.RequestID, live_request_id, live_request_last_time)
					// continue
				}
				pass_time_in_second := float64(cur_time_in_microseconds-live_request_last_time) / 1000000.0 // microseconds to seconds
				prev_reward += pass_time_in_second
				klog.V(5).Infof("calculate_prev_reward, requestID: %s, live_requestID: %s, pass_time_in_second: %f, prev_reward: %f, cur_time_in_microseconds: %d, live_request_last_time: %d", ctx.RequestID, live_request_id, pass_time_in_second, prev_reward, cur_time_in_microseconds, live_request_last_time)
				utils.UpdateLiveRequestLastTime(live_request_id, cur_time_in_microseconds)
			}

			// Copy completed request IDs to avoid concurrent map iteration and write
			utils.RemainingLatencyMutex.RLock()
			completedRequestIDs := make([]string, 0, len(utils.RemainingLatencyInMicroseconds))
			for reqID := range utils.RemainingLatencyInMicroseconds {
				completedRequestIDs = append(completedRequestIDs, reqID)
			}
			utils.RemainingLatencyMutex.RUnlock()

			// Now safely iterate over the copy
			for _, completed_request_id := range completedRequestIDs {
				remaining_latency_in_microseconds, exists := utils.GetRemainingLatenyFromCompletedRequest(completed_request_id)
				if !exists {
					klog.Errorf("calculate_prev_reward, requestID: %s, completed_requestID: %s, not found in RemainingLatencyInMicroseconds", ctx.RequestID, completed_request_id)
					continue
				}
				pass_time_in_second := float64(remaining_latency_in_microseconds) / 1000000.0 // microseconds to seconds
				prev_reward += pass_time_in_second
				klog.V(5).Infof("calculate_prev_reward, requestID: %s, completed_requestID: %s, pass_time_in_second: %f, prev_reward: %f", ctx.RequestID, completed_request_id, pass_time_in_second, prev_reward)
				utils.RemoveRemainingLatencyFromCompletedRequest(completed_request_id)
			}
			utils.UpdateLiveRequestLastTime(ctx.RequestID, cur_time_in_microseconds)

			klog.V(5).Infof("calculate_prev_reward, requestID: %s, total_prev_reward: %f", ctx.RequestID, prev_reward)
			utils.SetPrevRewardForRequest(ctx.RequestID, prev_reward)
		}
		logFormat := `**@latency_metrics@requestID@%s@request_start_time@%d@request_end_time@-9999@selectedpod@-9999@ttft@-9999@avg_tpot@-9999@total_decode_time@-9999@e2e@-9999@numInputTokens@%d@numOutputTokens@%d@numTotalTokens@%d@allPodsKvCacheHitRatios@%s@numInflightRequestsAllPods@%s@vllmGPUKVCacheUsage@%s@vllmCPUKVCacheUsage@%s@vllmNumRequestsRunning@%s@vllmNumRequestsWaiting@%s@podMetricsLastSecond@%s@numPrefillTokensForAllPods@%s@numDecodeTokensForAllPods@%s@subAlgorithm@%s@prev_reward@%f@GPU@%s@selectedPodGPU@%s`
		logMessage = fmt.Sprintf(
			logFormat,
			ctx.RequestID,
			time.Now().UnixMicro(),
			numInputTokens,
			expectedNumOutputTokens,
			numTotalTokens,
			jsonStrings["allPodsKvCacheHitRatios"],
			jsonStrings["numInflightRequestsAllPods"],
			jsonStrings["vllmGPUKVCacheUsage"],
			jsonStrings["vllmCPUKVCacheUsage"],
			jsonStrings["vllmNumRequestsRunning"],
			jsonStrings["vllmNumRequestWaiting"],
			jsonStrings["podMetricsLastSecond"],
			jsonStrings["numPrefillTokensForAllPods"],
			jsonStrings["numDecodeTokensForAllPods"],
			ctx.SubAlgorithm,
			prev_reward,
			jsonStrings["GPU"],
			"NotDecidedYet",
		)
	} else { // useRealRequest == 0
		logMessage = utils.GenerateLogMessages(allPodIPs, 1)[0]
	}
	log_construction_overhead := time.Since(log_construction_start_time).Milliseconds()
	reqBody, err := json.Marshal(logMessage)
	if err != nil {
		klog.Errorf("Failed to marshal RequestToLogMessage: %v, requestID: %s", err, ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
	}
	request_prepare_overhead := time.Since(route_start_time).Milliseconds()
	url := fmt.Sprintf("%s%s", routingAgentURL, inferEndpoint)
	http_req_to_routing_agent, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
	if reqErr != nil {
		klog.Errorf("Failed to create request: %v, requestID: %s", reqErr, ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
	}
	http_req_to_routing_agent.Header.Set("Content-Type", "application/json")
	/////////////////////////////////////////////////
	// Send HTTP Request to routing-agent-service  //
	/////////////////////////////////////////////////
	klog.V(5).Infof("Sending request to routing-agent-service: %s, requestID: %s", url, ctx.RequestID)
	resp, sendErr := httpClientForRLAgent.Do(http_req_to_routing_agent)
	if sendErr != nil {
		klog.Errorf("Request failed!!: %v, requestID: %s", sendErr, ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
	} else {
		if resp.StatusCode != http.StatusOK {
			klog.Errorf("Failed, Received non-200 response: %s, requestID: %s", resp.Status, ctx.RequestID)
		}
		responseBody, readErr := ioutil.ReadAll(resp.Body) // body: {"confidence":0.4398832619190216,"request_id":"10","selected_pod":"10.0.1.30"}
		if readErr != nil {
			klog.Errorf("Failed to read response body: %v, requestID: %s", readErr, ctx.RequestID)
			targetPod, _ = r.fallbackRouting(ctx, readyPods)
		} else {
			var routeResponse RouteResponse
			unmarshal_start := time.Now()
			if err := json.Unmarshal(responseBody, &routeResponse); err != nil {
				klog.Errorf("Failed to unmarshal responseBody: %v, requestID: %s", err, ctx.RequestID)
				targetPod, _ = r.fallbackRouting(ctx, readyPods)
			}
			resp.Body.Close()
			unmarshal_overhead := time.Since(unmarshal_start).Milliseconds()
			if targetPod == nil { // This means that routing-agent-service infer http request was successful
				getpod_start := time.Now()
				targetPod = GetPod(routeResponse.SelectedPod, readyPods)
				getpod_overhead := time.Since(getpod_start).Milliseconds()
				klog.Infof("RL inference http success, requestID: %s, selectedPod: %s", ctx.RequestID, routeResponse.SelectedPod)
				if targetPod == nil {
					klog.Errorf("Failed, No suitable pod found for selected pod IP: %s, requestID: %s", routeResponse.SelectedPod, ctx.RequestID)
					targetPod, _ = r.fallbackRouting(ctx, readyPods)
				}

				//////////////////////////////////////////////////////////////////
				// Overwrite the targetPod if the subAlgorithm is heuristics!!! //
				//////////////////////////////////////////////////////////////////
				if ctx.SubAlgorithm == "random" {
					klog.Infof("subAlgorithm, random routing, request_id: %s", ctx.RequestID)
					targetPod, _ = selectRandomPod(readyPods, rand.Intn)
				} else if ctx.SubAlgorithm == "least_latency" {
					klog.Infof("subAlgorithm, least_latency routing, request_id: %s", ctx.RequestID)
					targetPod = selectTargetPodWithLeastLatency(r.cache, readyPods, ctx.Model)
					if targetPod == nil {
						klog.Errorf("least_latency routing, No suitable pod found for least latency routing, requestID: %s", ctx.RequestID)
					} else {
						klog.Infof("least_latency routing, request_id: %s, targetPod: %s", ctx.RequestID, targetPod.Status.PodIP)
					}
				} else if ctx.SubAlgorithm == "least_request" {
					klog.Infof("subAlgorithm, least_request routing, request_id: %s", ctx.RequestID)
					targetPod = selectTargetPodWithLeastRequestCount(r.cache, readyPods)
					if targetPod == nil {
						klog.Errorf("least_request routing, No suitable pod found for least request count routing, requestID: %s", ctx.RequestID)
					} else {
						klog.Infof("least_request routing, request_id: %s, targetPod: %s", ctx.RequestID, targetPod.Status.PodIP)
					}
				} else if ctx.SubAlgorithm == "least_kv_cache" {
					klog.Infof("subAlgorithm, least_kv_cache routing, request_id: %s", ctx.RequestID)
					targetPod = selectTargetPodWithLeastKVCache(r.cache, readyPods, ctx.Model)
					if targetPod == nil {
						klog.Errorf("least_kv_cache routing, No suitable pod found for least kv cache routing, requestID: %s", ctx.RequestID)
					} else {
						klog.Infof("least_kv_cache routing, request_id: %s, targetPod: %s", ctx.RequestID, targetPod.Status.PodIP)
					}
				} else if ctx.SubAlgorithm == "prefix_cache_1" {
					klog.Infof("subAlgorithm, prefix_cache_1 routing, request_id: %s", ctx.RequestID)
					targetPod = r.routeWithPrefixCache1(ctx, readyPods, podIPsWithMatchingRatios)
				} else if ctx.SubAlgorithm == "prefix_cache_2" {
					targetPod = routePrefixRatioAndLoad(r.cache, readyPods, podIPsWithMatchingRatios)
				}
				// Safety check: if targetPod is still nil after routing algorithm execution, use fallback routing
				if targetPod == nil {
					klog.Warningf("targetPod is nil after routing algorithm execution, using fallback routing for requestID: %s", ctx.RequestID)
					targetPod, _ = r.fallbackRouting(ctx, readyPods)
				}

				set_shared_var_start := time.Now()
				if routeResponse.NumTrains > utils.GetNumTrains() {
					utils.SetNumTrains(routeResponse.NumTrains)
				}
				if routeResponse.NumFlush > utils.GetNumFlush() {
					utils.SetNumFlush(routeResponse.NumFlush)
				}
				utils.SetExploration(routeResponse.Exploration, routeResponse.ExplorationEnabled, ctx.RequestID)
				utils.SetPredictedLatencies(routeResponse.PredictedLatencies, ctx.RequestID)
				utils.SetChosenPodPredictedLatency(routeResponse.ChosenPodPredictedLatency, ctx.RequestID)
				
				// Set predicted rewards if present (contextual_bandit provides this)
				if len(routeResponse.PredictedRewards) > 0 {
					utils.SetPredictedRewards(routeResponse.PredictedRewards, ctx.RequestID)
					utils.SetChosenPodPredictedReward(routeResponse.ChosenPodPredictedReward, ctx.RequestID)
					klog.V(5).Infof("Predicted rewards for requestID %s: %v, chosen pod reward: %f",
						ctx.RequestID, routeResponse.PredictedRewards, routeResponse.ChosenPodPredictedReward)
				}
				
				selectedPodGPU, _ := utils.GetGPUModel(routeResponse.SelectedPod)
				utils.SetSelectedPodGPU(selectedPodGPU, ctx.RequestID)
				set_shared_var_overhead := time.Since(set_shared_var_start).Milliseconds()
				end_to_end_overhead := time.Since(route_start_time).Milliseconds()
				utils.SetEndToEndOverheadForRequest(float64(end_to_end_overhead), ctx.RequestID)
				if targetPod != nil {
					klog.V(5).Infof("RL router, requestID: %s, SelectedPod: %s, SelectedPodGeneralPodId: %s, Route end_to_end_overhead: %dms, log_construction_overhead: %dms, request_prepare_overhead: %dms, unmarshal_overhead: %dms, getpod_overhead: %dms, set_shared_var_overhead: %dms, OverheadLog: %s", ctx.RequestID, targetPod.Status.PodIP, routeResponse.SelectedPodGeneralPodId, end_to_end_overhead, log_construction_overhead, request_prepare_overhead, unmarshal_overhead, getpod_overhead, set_shared_var_overhead, routeResponse.OverheadLog)
				} else {
					klog.V(5).Infof("RL router, requestID: %s, SelectedPod: nil, SelectedPodGeneralPodId: %s, Route end_to_end_overhead: %dms, log_construction_overhead: %dms, request_prepare_overhead: %dms, unmarshal_overhead: %dms, getpod_overhead: %dms, set_shared_var_overhead: %dms, OverheadLog: %s", ctx.RequestID, routeResponse.SelectedPodGeneralPodId, end_to_end_overhead, log_construction_overhead, request_prepare_overhead, unmarshal_overhead, getpod_overhead, set_shared_var_overhead, routeResponse.OverheadLog)
				}
			}
		}
	}

	if len(allPrefixHashes) > 0 && targetPod != nil {
		klog.V(5).Infof("Adding prefix hashes to cache. pod: %s", targetPod.Status.PodIP)
		r.prefixCacheIndexer.AddPrefix(allPrefixHashes, ctx.Model, targetPod.Status.PodIP)
	} else {
		if len(allPrefixHashes) == 0 {
			klog.Warningf("No prefix hashes found for requestID: %s, not adding to cache", ctx.RequestID)
		} else if targetPod == nil {
			klog.Errorf("targetPod is nil, cannot add prefix hashes to cache for requestID: %s", ctx.RequestID)
		}
	}

	if targetPod == nil {
		klog.Errorf("CRITICAL: targetPod is still nil before setting context, requestID: %s. Using fallback routing.", ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
	}

	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}

func formatJSONResponse(RequestID string, jsonBytes []byte) string {
	var data map[string]interface{}
	if err := json.Unmarshal(jsonBytes, &data); err != nil {
		return string(jsonBytes) // Return original if parsing fails
	}

	var result strings.Builder
	for key, value := range data {
		result.WriteString(fmt.Sprintf("requestID: %s, %s:%v\n", RequestID, key, value))
	}
	return strings.TrimSuffix(result.String(), "\n")
}

// routeWithPrefixCache1 implements the prefix_cache_1 routing algorithm
// 1. if the load is imbalanced, use routePrefixRatioAndLoad (prefix_cache.go) which has another load balance and prefix routing logic
// 2. if the load is balanced, use prefix routing
// 3. if there is no prefix matching pod, then use least request count routing
func (r *rlOnlineRouter) routeWithPrefixCache1(ctx *types.RoutingContext, readyPods []*v1.Pod, podIPsWithMatchingRatios map[string]int) *v1.Pod {
	var targetPod *v1.Pod
	var isLoadImbalanced bool
	targetPod, isLoadImbalanced = getTargetPodOnLoadImbalance(r.cache, readyPods)
	if !isLoadImbalanced {
		if len(podIPsWithMatchingRatios) > 0 {
			// routePrefixRatioAndLoad algorithm is...
			// 1. Sort the pods by prefix matching ratio and request count
			// 2. Select the pod with the highest prefix matching ratio as long as the pod is not overloaded
			targetPod = routePrefixRatioAndLoad(r.cache, readyPods, podIPsWithMatchingRatios)
			if targetPod == nil {
				klog.Errorf("prefix_cache_1, No pod found in podIPsWithMatchingRatios for prefix routing, requestID: %s", ctx.RequestID)
			} else {
				klog.Infof("prefix_cache_1, prefix routing, request_id: %s, targetPod: %s, podIPsWithMatchingRatios: %v", ctx.RequestID, targetPod.Status.PodIP, podIPsWithMatchingRatios)
			}
		}
	} else {
		klog.Infof("prefix_cache_1, load balancing (least_request), request_id: %s, targetPod: %s", ctx.RequestID, targetPod.Status.PodIP)
	}
	if len(podIPsWithMatchingRatios) == 0 || targetPod == nil {
		klog.Infof("prefix_cache_1, least request count routing, request_id: %s", ctx.RequestID)
		targetPod = selectTargetPodWithLeastRequestCount(r.cache, readyPods)
		if targetPod == nil {
			klog.Errorf("prefix_cache_1, No suitable pod found for least request count routing, requestID: %s", ctx.RequestID)
		}
	}
	return targetPod
}

func (r *rlOnlineRouter) fallbackRouting(ctx *types.RoutingContext, readyPods []*v1.Pod) (*v1.Pod, error) {
	klog.Infof("Using fallback routing (prefix_cache_1) for request %s", ctx.RequestID)
	
	// Try to calculate prefix matches for prefix_cache_1 routing
	var podIPsWithMatchingRatios map[string]int
	
	// Get input tokens if available
	input_tokens_in_bytearray := utils.GetByteArrayPrefillTokensForRequest(ctx.RequestID)
	
	if len(input_tokens_in_bytearray) > 0 {
		// Build ready pods map
		readyPodsMap := map[string]struct{}{}
		for _, pod := range readyPods {
			readyPodsMap[pod.Status.PodIP] = struct{}{}
		}
		
		// Calculate prefix matches (discard prefix hashes as we don't add to cache in fallback)
		podIPsWithMatchingRatios, _, _ = r.prefixCacheIndexer.MatchPrefix_returning_matchedprefixes(input_tokens_in_bytearray, ctx.Model, readyPodsMap)
		
		// Fill in pods without matches
		for _, pod := range readyPods {
			if _, ok := podIPsWithMatchingRatios[pod.Status.PodIP]; !ok {
				podIPsWithMatchingRatios[pod.Status.PodIP] = 0
			}
		}
		
		klog.V(4).Infof("Fallback routing calculated prefix matches for request %s: %v", ctx.RequestID, podIPsWithMatchingRatios)
	} else {
		klog.V(4).Infof("No input tokens available for fallback routing, will use least request count for request %s", ctx.RequestID)
		podIPsWithMatchingRatios = make(map[string]int)
	}
	
	// Use prefix_cache_1 routing logic
	targetPod := r.routeWithPrefixCache1(ctx, readyPods, podIPsWithMatchingRatios)
	
	if targetPod == nil {
		klog.Errorf("prefix_cache_1 fallback failed, using random routing for request %s", ctx.RequestID)
		var err error
		targetPod, err = selectRandomPod(readyPods, rand.Intn)
		if err != nil {
			klog.Errorf("Failed to select random pod: %v", err)
			return nil, err
		}
	}
	
	if targetPod == nil {
		klog.Errorf("No suitable pod found for fallback routing")
		return nil, fmt.Errorf("no suitable pod found for fallback routing")
	}
	
	return targetPod, nil
}


func (r *rlOnlineRouter) fallbackRouting_with_random(ctx *types.RoutingContext, readyPods []*v1.Pod) (*v1.Pod, error) {
	klog.Infof("Using fallback routing (random) for request %s", ctx.RequestID)
	var err error
	targetPod, err := selectRandomPod(readyPods, rand.Intn)
	if err != nil {
		klog.Errorf("Failed to select random pod: %v", err)
		return nil, err
	}
	if targetPod == nil {
		klog.Errorf("No suitable pod found for fallback routing")
		return nil, fmt.Errorf("no suitable pod found for fallback routing")
	}
	return targetPod, nil
}
