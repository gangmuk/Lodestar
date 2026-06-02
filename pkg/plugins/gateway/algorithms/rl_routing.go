package routingalgorithms

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
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
	flushed                    = false
	received_the_first_request = false
	allPodIPs                  = []string{}
	numFlush                   = 0
	// Depth threshold for prefix group identity hash.
	// Uses allPrefixHashes[depth] as the group fingerprint.
	// Should be larger than typical system prompt blocks (3-5) to distinguish conversations.
	prefixGroupDepthThreshold = utils.LoadEnvInt("PREFIX_GROUP_DEPTH_THRESHOLD", 10)

	// Background vLLM metric scraping interval in milliseconds.
	// The background scraper refreshes metrics for all pods at this interval,
	// so Route() reads cached values with zero HTTP overhead.
	bgScrapeIntervalMs = utils.LoadEnvInt("BG_SCRAPE_INTERVAL_MS", 100)
)

// bgMetricsCache holds the latest vLLM metrics snapshot from the background scraper.
// Written by the background goroutine, read by Route() on every request.
type bgMetricsSnapshot struct {
	GPUKVCacheUsage    map[string]float64 // podIP -> value
	NumRequestsRunning map[string]float64
	NumRequestsWaiting map[string]float64
	NumPreemptions     map[string]float64
}

var (
	bgMetricsMu    sync.RWMutex
	bgMetricsReady bool
	bgMetrics      bgMetricsSnapshot
)

// bgScraperOnce ensures the scraper goroutine is launched at most once per
// process, so both rl-online-router and preble can safely prime it from their
// constructors without racing or starting duplicate scrapers. A pointer is
// used so tests can swap in a fresh *sync.Once without tripping go vet's
// copylocks warning.
var bgScraperOnce = &sync.Once{}

// startBackgroundMetricsScraper launches a goroutine that scrapes vLLM metrics
// from all pods at a fixed interval. Route() reads the cached snapshot instead
// of scraping per-request, reducing overhead from ~50ms to ~0ms.
//
// Safe to call from multiple routers — bgScraperOnce guards the body.
func startBackgroundMetricsScraper(c cache.Cache) {
	bgScraperOnce.Do(func() { startBackgroundMetricsScraperImpl(c) })
}

func startBackgroundMetricsScraperImpl(c cache.Cache) {
	// Defense-in-depth: a nil cache would nil-deref on the first ticker tick
	// (c.ListModels in the goroutine body). Both production call sites guard
	// via cache.Get(), so this should never fire — but if it does, fail loud
	// without crashing the process.
	if c == nil {
		klog.Errorf("Background metrics scraper: nil cache, scraper NOT started")
		return
	}
	interval := time.Duration(bgScrapeIntervalMs) * time.Millisecond
	klog.Infof("Starting background vLLM metrics scraper with interval %v", interval)

	targetMetrics := []string{
		utils.MetricGPUCacheUsagePerc,
		utils.MetricNumRequestsRunning,
		utils.MetricNumRequestsWaiting,
		utils.MetricNumPreemptions,
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			// Get current pods from cache
			models := c.ListModels()
			var allPods []*v1.Pod
			for _, model := range models {
				pods, err := c.ListPodsByModel(model)
				if err != nil {
					continue
				}
				allPods = append(allPods, pods.All()...)
			}
			if len(allPods) == 0 {
				continue
			}

			// Scrape all pods in parallel using a temporary requestID
			scrapeID := fmt.Sprintf("bg_scrape_%d", time.Now().UnixMicro())
			var wg sync.WaitGroup
			for _, pod := range allPods {
				wg.Add(1)
				go func(pod *v1.Pod) {
					defer wg.Done()
					if err := utils.ReadAndStoreVLLMMetrics(scrapeID, pod, targetMetrics); err != nil {
						klog.V(5).Infof("Background scrape failed for pod %s: %v", pod.Status.PodIP, err)
					}
				}(pod)
			}
			wg.Wait()

			// Read the scraped values and build a snapshot
			snap := bgMetricsSnapshot{
				GPUKVCacheUsage:    make(map[string]float64),
				NumRequestsRunning: make(map[string]float64),
				NumRequestsWaiting: make(map[string]float64),
				NumPreemptions:     make(map[string]float64),
			}
			if gpuKV, err := utils.GetvLLMGPUKVCacheUsageForAllPods(scrapeID); err == nil {
				snap.GPUKVCacheUsage = gpuKV
			}
			if running, err := utils.GetvLLMNumRequestsRunningForAllPods(scrapeID); err == nil {
				snap.NumRequestsRunning = running
			}
			if waiting, err := utils.GetvLLMNumRequestsWaitingForAllPods(scrapeID); err == nil {
				snap.NumRequestsWaiting = waiting
			}
			if preemptions, err := utils.GetvLLMNumPreemptionsForAllPods(scrapeID); err == nil {
				snap.NumPreemptions = preemptions
			}

			// Publish snapshot
			bgMetricsMu.Lock()
			bgMetrics = snap
			bgMetricsReady = true
			bgMetricsMu.Unlock()

			// Clean up the temporary requestID storage
			utils.CleanupVLLMMetricsForRequest(scrapeID)
		}
	}()
}

// copyBgMetricsToRequest copies the latest background metrics snapshot into
// the per-request storage maps so downstream code (which reads by requestID)
// works unchanged.
func copyBgMetricsToRequest(requestID string) {
	bgMetricsMu.RLock()
	snap := bgMetrics
	bgMetricsMu.RUnlock()

	utils.StoreVLLMMetricsForRequest(requestID,
		snap.GPUKVCacheUsage,
		snap.NumRequestsRunning,
		snap.NumRequestsWaiting,
		snap.NumPreemptions,
	)
}

var (
	httpClientForRLAgent = &http.Client{
		Timeout: 500 * time.Millisecond,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     180 * time.Second,
			DisableCompression:  false,
			DialContext: (&net.Dialer{
				Timeout:   500 * time.Millisecond,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   500 * time.Millisecond,
			ForceAttemptHTTP2:     true, // Enable HTTP/2
			ResponseHeaderTimeout: 500 * time.Millisecond,
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

	// Start background vLLM metrics scraper
	startBackgroundMetricsScraper(c)

	klog.InfoS("Created RL online router")
	return router, nil
}

//	pre-populates GPU model information for all pods
//
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
			klog.Warningf("failure to list pods for model %s during GPU init: %v", model, err)
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
						klog.Errorf("failure flush. failure marshal RequestToLogMessage: %v", err)
						utils.CleanupAllRequestLogMessage()
						continue
					}
					url := fmt.Sprintf("%s%s", routingAgentURL, flushEndpoint)
					req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
					if reqErr != nil {
						klog.Errorf("failure flush. failure to create request: %v", reqErr)
						utils.CleanupAllRequestLogMessage()
						continue
					}
					req.Header.Set("Content-Type", "application/json")
					resp, sendErr := httpClientForRLAgent.Do(req) // flush request
					if sendErr != nil {
						klog.Errorf("failure flush. failure to send request: %v", sendErr)
						utils.CleanupAllRequestLogMessage()
						continue
					}
					if resp.StatusCode != http.StatusOK {
						klog.Errorf("Received non-200 response: %s", resp.Status)
						utils.CleanupAllRequestLogMessage()
						klog.Errorf("failure flush. Received non-200 response: %s", resp.Status)
						continue
					}
					body, readErr := ioutil.ReadAll(resp.Body)
					if readErr != nil {
						klog.Errorf("failure to read response body: %v", readErr)
						utils.CleanupAllRequestLogMessage()
						klog.Errorf("failure flush. failure to read response body: %v", readErr)
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
	RequestID               string `json:"request_id"`
	SelectedPod             string `json:"selected_pod"`
	SelectedPodGeneralPodId string `json:"selected_pod_generalpodid"`
	NumTrains               int    `json:"num_trains"`
	NumFlush                int    `json:"num_flush"`
	Exploration             int    `json:"exploration"`
	ExplorationEnabled      int    `json:"exploration_enabled"`
	// OverheadLog               string             `json:"overhead_log"`
	TensorTransferOverhead    float64            `json:"tensor_transfer_overhead"`
	InferOverhead             float64            `json:"infer_overhead"`
	OtherOverhead             float64            `json:"other_overhead"`
	EndToEndOverhead          float64            `json:"end_to_end_overhead"`
	PredictedLatencies        map[string]float64 `json:"predicted_latencies"`
	PredictedRewards          map[string]float64 `json:"predicted_rewards"`
	ChosenPodPredictedLatency float64            `json:"chosen_pod_predicted_latency"`
	ChosenPodPredictedReward  float64            `json:"chosen_pod_predicted_reward"`
	OODFallback               int                `json:"ood_fallback"`
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
	klog.Errorf("Getpod, No pod found for podIP: %s", podIP)
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
			klog.V(4).Infof("failure to get GPU model from node %s for pod %s: %v", pod.Spec.NodeName, pod.Status.PodIP, err)
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
	route_start_time := time.Now().UnixMilli()
	readyPods := pods.All()
	var targetPod *v1.Pod = nil
	if !received_the_first_request {
		klog.Infof("This is the first request, using fallback routing and return right away. Give some time for the RL agent to warm up.")
		targetPod = r.fallbackRouting_with_least_request(ctx, readyPods)
		received_the_first_request = true
		allPodIPs = utils.GetAllPodIPsFromRegistry()
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}

	// === Phase 1: vLLM metric scraping ===
	// Use background-scraped metrics (refreshed every BG_SCRAPE_INTERVAL_MS).
	// This avoids per-request HTTP round-trips to all pods (~50ms overhead).
	vllmScrapeStart := time.Now()
	if bgMetricsReady {
		// Fast path: copy cached background metrics into per-request storage
		copyBgMetricsToRequest(ctx.RequestID)
	} else {
		// Fallback: background scraper hasn't produced a snapshot yet (cold start).
		// Scrape synchronously on this request.
		klog.Infof("Background metrics not ready yet, falling back to per-request scraping for requestID: %s", ctx.RequestID)
		targetMetrics := []string{
			utils.MetricGPUCacheUsagePerc,
			utils.MetricNumRequestsRunning,
			utils.MetricNumRequestsWaiting,
			utils.MetricNumPreemptions,
		}
		var wg sync.WaitGroup
		for _, pod := range readyPods {
			wg.Add(1)
			go func(pod *v1.Pod) {
				defer wg.Done()
				if err := utils.ReadAndStoreVLLMMetrics(ctx.RequestID, pod, targetMetrics); err != nil {
					klog.Errorf("ReadAndStoreVLLMMetrics failed: %v", err)
				}
			}(pod)
		}
		wg.Wait()
	}
	detailedpodmetrics := utils.MetricsTracker.GetDetailedMetrics(time.Now().Add(-utils.MetricsTracker.WindowSize))
	utils.AddRequestPodMetrics(ctx.RequestID, detailedpodmetrics)
	vllmScrapeOverhead := time.Since(vllmScrapeStart).Milliseconds()
	utils.SetVLLMScrapingOverheadForRequest(vllmScrapeOverhead, ctx.RequestID)

	// === Phase 2: Feature preparation ===
	featurePrepStart := time.Now()

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

	// Build a prefix-cache-capable subset of readyPodsMap for prefix matching.
	// Pods that don't support prefix caching (e.g. V100) are excluded from matching
	// so they always get kv_hit_ratio=0 and don't pollute the prefix indexer.
	prefixCapablePodsMap := make(map[string]struct{}, len(readyPodsMap))
	for podIP := range readyPodsMap {
		if utils.IsPrefixCacheCapable(podIP) {
			prefixCapablePodsMap[podIP] = struct{}{}
		}
	}

	podIPsWithMatchingRatios, matchedPrefixHashes, allPrefixHashes = r.prefixCacheIndexer.MatchPrefix_returning_matchedprefixes(input_tokens_in_bytearray, ctx.Model, prefixCapablePodsMap)
	numInputTokens := utils.GetNumPrefillTokensForRequest(ctx.RequestID)

	// predict output!
	hash_of_matchedprefix := utils.HashPrefixHashes(matchedPrefixHashes)
	utils.SetHashOfPrefixHashesForRequest(ctx.RequestID, hash_of_matchedprefix)

	// Prefix group identity: use the chained hash at a specific depth in allPrefixHashes.
	// At depth D, allPrefixHashes[D] encodes the first D blocks (deterministic, conversation-specific).
	// This distinguishes conversations that share only a short system prompt.
	var prefixGroupHash uint64
	if len(allPrefixHashes) > prefixGroupDepthThreshold {
		prefixGroupHash = allPrefixHashes[prefixGroupDepthThreshold]
	} else if len(allPrefixHashes) > 0 {
		prefixGroupHash = allPrefixHashes[len(allPrefixHashes)-1]
	} else {
		prefixGroupHash = 0
	}
	expectedNumOutputTokens, exist := utils.GetNumOutputTokensForPrefix(hash_of_matchedprefix)
	expectedNumOutputTokens = 100
	// expectedNumOutputTokens = 1
	if !exist {
		klog.V(5).Infof("requestID: %s, No cached output token length found for hash_of_matchedprefix: %d. Using default value %d", ctx.RequestID, hash_of_matchedprefix, expectedNumOutputTokens)
	}

	numTotalTokens := numInputTokens + expectedNumOutputTokens

	for _, pod := range readyPods {
		if _, ok := podIPsWithMatchingRatios[pod.Status.PodIP]; !ok {
			// klog.Infof("requestID: %s, No found prefix matched pods. Filled all readypods with 0 kv cache hit ratio", ctx.RequestID)
			podIPsWithMatchingRatios[pod.Status.PodIP] = 0
			// podIPsWithMatchingRatios[pod.Name] = 0
		}
	}
	utils.SetSnapShotForKVCacheHitRatio(ctx.RequestID, podIPsWithMatchingRatios)
	utils.SetPrefixGroupHashForRequest(ctx.RequestID, prefixGroupHash)

	// Extract per-pod prefix last-access timestamps for time-weighted KV hit ratio.
	// Uses a fresh readyPods set since matchPods may have mutated the original.
	readyPodsMapForLastAccess := map[string]struct{}{}
	for _, pod := range readyPods {
		readyPodsMapForLastAccess[pod.Status.PodIP] = struct{}{}
	}
	podIPsWithLastAccessAbsolute := r.prefixCacheIndexer.GetPodPrefixLastAccess(allPrefixHashes, ctx.Model, readyPodsMapForLastAccess)
	// Convert absolute unix-millis timestamps to relative microseconds (same
	// reference frame as request_start_time = UnixMicro - FirstRequestStartTime).
	// preprocess.py computes age as (request_start_time - last_access) and both
	// must use the same epoch for the subtraction to be meaningful.
	podIPsWithLastAccess := make(map[string]int64, len(podIPsWithLastAccessAbsolute))
	for pod, absMillis := range podIPsWithLastAccessAbsolute {
		if absMillis > 0 && utils.FirstRequestStartTime > 0 {
			// absMillis is unix millis, FirstRequestStartTime is unix micros
			podIPsWithLastAccess[pod] = absMillis*1000 - utils.FirstRequestStartTime
		} else {
			podIPsWithLastAccess[pod] = 0
		}
	}
	utils.SetSnapShotForKVCacheLastAccess(ctx.RequestID, podIPsWithLastAccess)

	if len(readyPods) == 0 {
		klog.Errorf("requestID: %s, No ready pods available for routing", ctx.RequestID)
		return "", fmt.Errorf("no ready pods available")
	}

	if len(readyPods) == 1 {
		ctx.SetTargetPod(readyPods[0])
		klog.Warningf("requestID: %s, Only one ready pod available, using it as target pod: %s", ctx.RequestID, readyPods[0].Status.PodIP)
		return ctx.TargetAddress(), nil
	}

	// LMETRIC pre-compute: score + counter increment must happen atomically
	// BEFORE the HTTP round-trip so concurrent requests see this contribution.
	// See lmetric.go for details.
	var lmetricSelectedPodIP string
	if ctx.SubAlgorithm == "lmetric" {
		lmetricSelectedPodIP = lmetricPreCompute(ctx, readyPods, podIPsWithMatchingRatios, numInputTokens)
	}

	// === MOONCAKE: min_expected_latency pre-compute (before HTTP call for atomicity) ===
	var mooncakeSelectedPodIP string
	if ctx.SubAlgorithm == "mooncake" {
		vllmRunning, runErr := utils.GetvLLMNumRequestsRunningForAllPods(ctx.RequestID)
		vllmWaiting, waitErr := utils.GetvLLMNumRequestsWaitingForAllPods(ctx.RequestID)
		podBatchSizes := make(map[string]int, len(readyPods))
		podIPs := make([]string, 0, len(readyPods))
		for _, pod := range readyPods {
			ip := pod.Status.PodIP
			podIPs = append(podIPs, ip)
			bs := 0
			if runErr == nil {
				bs += int(vllmRunning[ip])
			}
			if waitErr == nil {
				bs += int(vllmWaiting[ip])
			}
			podBatchSizes[ip] = bs
		}
		podSpeeds := utils.GetMooncakeSpeedsForPods(podIPs)
		selectedIP, ok := utils.MooncakeScorePods(
			podBatchSizes,
			podSpeeds,
			ctx.RequestID,
		)
		if ok {
			mooncakeSelectedPodIP = selectedIP
		}
		klog.Infof("MOONCAKE pre-compute: requestID=%s, selectedPod=%s", ctx.RequestID, mooncakeSelectedPodIP)
	}

	var logMessage string
	var jsonStrings = make(map[string]string)
	// 1. KV cache hit ratios
	allPodsKvCacheHitRatios := utils.GetAllPodsKVCacheHitRatios(ctx.RequestID)
	jsonStrings["allPodsKvCacheHitRatios"] = jsonStringify(allPodsKvCacheHitRatios, utils.GetrequestAllPodsKVCacheMutex())
	klog.V(5).Infof("allPodsKvCacheHitRatios: %s", jsonStrings["allPodsKvCacheHitRatios"])

	// 1b. KV cache last-access timestamps (for time-weighted freshness in RL routing)
	allPodsKvCacheLastAccess := utils.GetAllPodsKVCacheLastAccess(ctx.RequestID)
	jsonStrings["allPodsKvCacheLastAccess"] = jsonStringify(allPodsKvCacheLastAccess, utils.GetKVCacheLastAccessMutex())

	// 2. Inflight requests
	numInflightRequestsAllPods := utils.GetInflightRequestsForAllPods(ctx.RequestID)
	jsonStrings["numInflightRequestsAllPods"] = jsonStringify(numInflightRequestsAllPods, utils.GetrequestInflightMutex())

	// 2a. Inflight prefill requests
	numInflightPrefillRequestsAllPods := utils.GetSnapshotNumInflightPrefillRequestsForRequest(ctx.RequestID)
	jsonStrings["numInflightPrefillRequestsAllPods"] = jsonStringify(numInflightPrefillRequestsAllPods, utils.GetrequestInflightMutex())

	// 2b. Inflight decode requests
	numInflightDecodeRequestsAllPods := utils.GetSnapshotNumInflightDecodeRequestsForRequest(ctx.RequestID)
	jsonStrings["numInflightDecodeRequestsAllPods"] = jsonStringify(numInflightDecodeRequestsAllPods, utils.GetrequestInflightMutex())

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

	// 7. Number of preemptions
	vllmNumPreemptions, err := utils.GetvLLMNumPreemptionsForAllPods(ctx.RequestID)
	if err == nil {
		jsonStrings["vllmNumPreemptions"] = jsonStringify(vllmNumPreemptions, utils.GetvllmNumPreemptionsMutex())
	} else {
		jsonStrings["vllmNumPreemptions"] = "{}"
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
	// exploration, explorationEnabled := utils.GetExploration(ctx.RequestID)
	normalized_request_start_time := time.Now().UnixMicro() - utils.FirstRequestStartTime
	logFormat := `**@latency_metrics@requestID@%s@request_start_time@%d@request_end_time@-9999@selectedpod@-9999@ttft@-9999@avg_tpot@-9999@total_decode_time@-9999@e2e@-9999@numInputTokens@%d@numOutputTokens@%d@numTotalTokens@%d@allPodsKvCacheHitRatios@%s@allPodsKvCacheLastAccess@%s@hashOfMatchedPrefix@%d@numInflightRequestsAllPods@%s@numInflightPrefillRequestsAllPods@%s@numInflightDecodeRequestsAllPods@%s@vllmGPUKVCacheUsage@%s@vllmCPUKVCacheUsage@%s@vllmNumRequestsRunning@%s@vllmNumRequestsWaiting@%s@vllmNumPreemptions@%s@numPrefillTokensForAllPods@%s@numDecodeTokensForAllPods@%s@subAlgorithm@%s@prev_reward@%f@GPU@%s`
	logMessage = fmt.Sprintf(
		logFormat,
		ctx.RequestID,
		normalized_request_start_time,
		numInputTokens,
		expectedNumOutputTokens,
		numTotalTokens,
		jsonStrings["allPodsKvCacheHitRatios"],
		jsonStrings["allPodsKvCacheLastAccess"],
		prefixGroupHash,
		jsonStrings["numInflightRequestsAllPods"],
		jsonStrings["numInflightPrefillRequestsAllPods"],
		jsonStrings["numInflightDecodeRequestsAllPods"],
		jsonStrings["vllmGPUKVCacheUsage"],
		jsonStrings["vllmCPUKVCacheUsage"],
		jsonStrings["vllmNumRequestsRunning"],
		jsonStrings["vllmNumRequestWaiting"],
		jsonStrings["vllmNumPreemptions"],
		jsonStrings["numPrefillTokensForAllPods"],
		jsonStrings["numDecodeTokensForAllPods"],
		ctx.SubAlgorithm,
		prev_reward,
		jsonStrings["GPU"],
	)

	featurePrepOverhead := time.Since(featurePrepStart).Milliseconds()
	utils.SetFeaturePrepOverheadForRequest(featurePrepOverhead, ctx.RequestID)

	// === Phase 3: routing decision ===
	// Contextual-bandit subAlgorithms consult the routing-agent-service over
	// HTTP. All other subAlgorithms (random, least_request, prefix_cache_*,
	// least_kv_cache, least_prefill_tokens, lmetric, mooncake, ...) dispatch
	// the heuristic locally and skip the round-trip entirely.
	httpRoundTripStart := time.Now()
	if ctx.SubAlgorithm != "lodestar" {
		targetPod = r.selectHeuristicTargetPod(ctx, readyPods, podIPsWithMatchingRatios, lmetricSelectedPodIP, mooncakeSelectedPodIP)
		utils.SetFailureFallbackForRequest(0, ctx.RequestID)
	} else {
		routing_agent_failed := 0
		reqBody, err := json.Marshal(logMessage)
		if err != nil {
			klog.Errorf("failure to marshal RequestToLogMessage: %v, requestID: %s", err, ctx.RequestID)
			targetPod = r.fallbackRouting(ctx, readyPods, ctx.SubAlgorithm)
			if targetPod == nil {
				return "", fmt.Errorf("fallback routing failed after marshal error, requestID: %s", ctx.RequestID)
			}
			ctx.SetTargetPod(targetPod)
			return ctx.TargetAddress(), nil
		}
		url := fmt.Sprintf("%s%s", routingAgentURL, inferEndpoint)
		http_req_to_routing_agent, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
		if reqErr != nil {
			klog.Errorf("failure to create request: %v, requestID: %s", reqErr, ctx.RequestID)
			targetPod = r.fallbackRouting(ctx, readyPods, ctx.SubAlgorithm)
			if targetPod == nil {
				return "", fmt.Errorf("fallback routing failed after request creation error, requestID: %s", ctx.RequestID)
			}
			ctx.SetTargetPod(targetPod)
			return ctx.TargetAddress(), nil
		}
		http_req_to_routing_agent.Header.Set("Content-Type", "application/json")
		/////////////////////////////////////////////////
		// Send HTTP Request to routing-agent-service  //
		/////////////////////////////////////////////////
		klog.V(5).Infof("Sending request to routing-agent-service: %s, requestID: %s", url, ctx.RequestID)
		// lmetricFallback prefers the LMETRIC or MOONCAKE pre-computed pod if available,
		// otherwise uses the standard fallback — so those decisions still stand when
		// the routing agent is unreachable.
		lmetricFallback := func() *v1.Pod {
			if pod := lmetricFallbackPod(ctx, readyPods, lmetricSelectedPodIP); pod != nil {
				return pod
			}
			if mooncakeSelectedPodIP != "" {
				if pod := GetPod(mooncakeSelectedPodIP, readyPods); pod != nil {
					klog.Infof("MOONCAKE: using pre-computed selection %s (routing agent unavailable), requestID=%s", mooncakeSelectedPodIP, ctx.RequestID)
					return pod
				}
			}
			return r.fallbackRouting(ctx, readyPods, ctx.SubAlgorithm)
		}

		resp, sendErr := httpClientForRLAgent.Do(http_req_to_routing_agent)
		if sendErr != nil {
			klog.Errorf("Request failure!!: %v, requestID: %s", sendErr, ctx.RequestID)
			targetPod = lmetricFallback()
			routing_agent_failed = 1
		} else {
			if resp.StatusCode != http.StatusOK {
				klog.Errorf("failure, Received non-200 response: %s, requestID: %s", resp.Status, ctx.RequestID)
				routing_agent_failed = 1
			}
			responseBody, readErr := ioutil.ReadAll(resp.Body) // body: {"request_id":"10","selected_pod":"10.0.1.30"}
			if readErr != nil {
				klog.Errorf("failure to read response body: %v, requestID: %s", readErr, ctx.RequestID)
				targetPod = lmetricFallback()
				routing_agent_failed = 1
			} else {
				var routeResponse RouteResponse
				if err := json.Unmarshal(responseBody, &routeResponse); err != nil {
					klog.Errorf("failure to unmarshal responseBody: %v, requestID: %s", err, ctx.RequestID)
					if ctx.SubAlgorithm == "lodestar" {
						targetPod = r.fallbackRouting(ctx, readyPods, ctx.SubAlgorithm)
						routing_agent_failed = 1
					}
				}
				resp.Body.Close()
				if targetPod == nil { // This means that routing-agent-service infer http request was successful
					targetPod = GetPod(routeResponse.SelectedPod, readyPods)
					klog.Infof("RL inference http success, requestID: %s, selectedPod: %s", ctx.RequestID, routeResponse.SelectedPod)
					if targetPod == nil {
						klog.Errorf("failure, No suitable pod found for selected pod IP: %s, requestID: %s", routeResponse.SelectedPod, ctx.RequestID)
						targetPod = lmetricFallback()
						routing_agent_failed = 1
					}
					utils.SetOODFallbackForRequest(routeResponse.OODFallback, ctx.RequestID)
					// ood_fallback: 0=normal, 1=OOD detected, 2=model not trained yet, 3=reward tiebreak (agent picked randomly), 4=PC1-biased exploration
					// ood_fallback=3: agent already picked randomly among near-best pods, trust the agent's choice
					if routeResponse.OODFallback >= 1 && routeResponse.OODFallback != 3 && ctx.SubAlgorithm == "lodestar" {
						klog.Infof("subAlgorithm, ood_fallback routing (ood_fallback=%d), request_id: %s", routeResponse.OODFallback, ctx.RequestID)
						targetPod = r.fallbackRouting(ctx, readyPods, ctx.SubAlgorithm)
						routing_agent_failed = 1
					}
					utils.SetFailureFallbackForRequest(routing_agent_failed, ctx.RequestID)

					// NOTE: do NOT call selectHeuristicTargetPod here. Inside this
					// branch ctx.SubAlgorithm is always "lodestar" (the outer
					// guard is `if ctx.SubAlgorithm != "lodestar" { ... } else
					// { /* HTTP block */ }`), so no heuristic branch in
					// selectHeuristicTargetPod can match. Calling it would hit
					// the safety net (routeWithPrefixCache1) and overwrite the
					// agent's selection (or the OOD-fallback selection set
					// above), which silently regressed lodestar to prefix_cache_1
					// for every request.
					if routeResponse.NumTrains > utils.GetNumTrains() {
						utils.SetNumTrains(routeResponse.NumTrains)
					}
					if routeResponse.NumFlush > utils.GetNumFlush() {
						utils.SetNumFlush(routeResponse.NumFlush)
					}
					utils.SetExploration(routeResponse.Exploration, routeResponse.ExplorationEnabled, ctx.RequestID)
					utils.SetPredictedLatencies(routeResponse.PredictedLatencies, ctx.RequestID)
					utils.SetChosenPodPredictedLatency(routeResponse.ChosenPodPredictedLatency, ctx.RequestID)
					if len(routeResponse.PredictedRewards) > 0 {
						utils.SetPredictedRewards(routeResponse.PredictedRewards, ctx.RequestID)
						klog.V(5).Infof("Predicted rewards for requestID %s: %v)", ctx.RequestID, routeResponse.PredictedRewards)
					}

					utils.SetChosenPodPredictedReward(routeResponse.ChosenPodPredictedReward, ctx.RequestID)
					klog.V(5).Infof("ChosenPodPredictedReward for requestID %s: %f", ctx.RequestID, routeResponse.ChosenPodPredictedReward)
					selectedPodGPU, _ := utils.GetGPUModel(routeResponse.SelectedPod)
					utils.SetSelectedPodGPU(selectedPodGPU, ctx.RequestID)
				} else {
					utils.SetOODFallbackForRequest(routeResponse.OODFallback, ctx.RequestID)
					// ood_fallback: 0=normal, 1=OOD detected, 2=model not trained yet, 3=reward tiebreak (agent picked randomly), 4=PC1-biased exploration
					// ood_fallback=3: agent already picked randomly among near-best pods, trust the agent's choice
					if routeResponse.OODFallback >= 1 && routeResponse.OODFallback != 3 && ctx.SubAlgorithm == "lodestar" {
						klog.Infof("subAlgorithm, ood_fallback routing (ood_fallback=%d), request_id: %s", routeResponse.OODFallback, ctx.RequestID)
						targetPod = r.fallbackRouting(ctx, readyPods, ctx.SubAlgorithm)
						routing_agent_failed = 1
					}
					utils.SetFailureFallbackForRequest(routing_agent_failed, ctx.RequestID)
				}
			}
		}
	}

	httpRoundTripOverhead := time.Since(httpRoundTripStart).Milliseconds()
	utils.SetHTTPRoundTripOverheadForRequest(httpRoundTripOverhead, ctx.RequestID)

	if len(allPrefixHashes) > 0 && targetPod != nil {
		// Only insert into prefix tracking for pods that support prefix caching.
		// V100 pods don't have prefix caching enabled in vLLM, so tracking them
		// would create false prefix hit signals for future requests.
		if utils.IsPrefixCacheCapable(targetPod.Status.PodIP) {
			r.prefixCacheIndexer.AddPrefix(allPrefixHashes, ctx.Model, targetPod.Status.PodIP)
		} else {
			klog.V(5).Infof("Skipping AddPrefix for non-prefix-cache-capable pod %s (requestID: %s)", targetPod.Status.PodIP, ctx.RequestID)
		}
	} else {
		if len(allPrefixHashes) == 0 {
			klog.Warningf("No prefix hashes found for requestID: %s, not adding to cache", ctx.RequestID)
		} else if targetPod == nil {
			klog.Errorf("targetPod is nil, cannot add prefix hashes to cache for requestID: %s", ctx.RequestID)
		}
	}

	if targetPod == nil {
		klog.Errorf("CRITICAL: targetPod is still nil before setting context, requestID: %s. Using fallback routing.", ctx.RequestID)
		targetPod = r.fallbackRouting_with_prefix_cache_1(ctx, readyPods)
	}

	if targetPod == nil {
		return "", fmt.Errorf("all routing attempts failed, no available pod for requestID: %s", ctx.RequestID)
	}

	lmetricRollbackIfMismatch(ctx, targetPod.Status.PodIP, lmetricSelectedPodIP)

	end_to_end_overhead := time.Now().UnixMilli() - route_start_time
	utils.SetEndToEndOverheadForRequest(end_to_end_overhead, ctx.RequestID)

	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}

// func formatJSONResponse(RequestID string, jsonBytes []byte) string {
// 	var data map[string]interface{}
// 	if err := json.Unmarshal(jsonBytes, &data); err != nil {
// 		return string(jsonBytes) // Return original if parsing fails
// 	}

// 	var result strings.Builder
// 	for key, value := range data {
// 		result.WriteString(fmt.Sprintf("requestID: %s, %s:%v\n", RequestID, key, value))
// 	}
// 	return strings.TrimSuffix(result.String(), "\n")
// }

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
		targetPod = selectTargetPodWithLeastRequestCount(ctx, r.cache, readyPods)
		if targetPod == nil {
			klog.Errorf("prefix_cache_1, No suitable pod found for least request count routing, requestID: %s", ctx.RequestID)
		}
	}
	return targetPod
}

// selectHeuristicTargetPod dispatches the routing decision to the named heuristic
// baseline (random, least_request, prefix_cache_1, lmetric, mooncake, ...).
//
// Called from two sites in Route():
//  1. For non-lodestar subAlgorithms — the routing-agent-service is not
//     consulted at all (no HTTP round-trip).
//  2. For the lodestar subAlgorithm — invoked as a final stage that can
//     overwrite the agent's choice if the subAlgorithm name happens to match
//     one of the branches here. "lodestar" itself does not match any branch,
//     so this call is a no-op for it; it is kept for symmetry with the
//     original control flow.
//
// Returns the chosen pod. Falls back to routeWithPrefixCache1 if no branch
// matched or the matched branch produced nil.
func (r *rlOnlineRouter) selectHeuristicTargetPod(
	ctx *types.RoutingContext,
	readyPods []*v1.Pod,
	podIPsWithMatchingRatios map[string]int,
	lmetricSelectedPodIP string,
	mooncakeSelectedPodIP string,
) *v1.Pod {
	var targetPod *v1.Pod
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
		targetPod = selectTargetPodWithLeastRequestCount(ctx, r.cache, readyPods)
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
	} else if ctx.SubAlgorithm == "least_prefill_tokens" {
		klog.Infof("subAlgorithm, least_prefill_tokens routing, request_id: %s", ctx.RequestID)
		targetPod = selectTargetPodWithLeastPrefillTokens(ctx, readyPods)
		if targetPod == nil {
			klog.Errorf("least_prefill_tokens routing, No suitable pod found for least prefill tokens routing, requestID: %s", ctx.RequestID)
		} else {
			klog.Infof("least_prefill_tokens routing, request_id: %s, targetPod: %s", ctx.RequestID, targetPod.Status.PodIP)
		}
	} else if ctx.SubAlgorithm == "prefix_hit_threshold_or_least_request" {
		klog.Infof("subAlgorithm, prefix_hit_threshold_or_least_request routing (threshold-based), request_id: %s", ctx.RequestID)
		targetPod = routePrefixHitThresholdOrLeastRequest(ctx, r.cache, readyPods, podIPsWithMatchingRatios)
	} else if ctx.SubAlgorithm == "lmetric" {
		targetPod = selectLMetricPod(ctx, readyPods, lmetricSelectedPodIP)
		if targetPod == nil {
			targetPod = r.fallbackRouting_with_prefix_cache_1(ctx, readyPods)
		}
	} else if ctx.SubAlgorithm == "mooncake" {
		// MOONCAKE: use the pre-computed pod selection (scored before HTTP call for atomicity)
		if mooncakeSelectedPodIP != "" {
			targetPod = GetPod(mooncakeSelectedPodIP, readyPods)
		}
		if targetPod == nil {
			klog.Errorf("MOONCAKE: pre-computed pod %s not in readyPods, falling back. requestID=%s", mooncakeSelectedPodIP, ctx.RequestID)
			utils.RollbackMooncakeForRequest(ctx.RequestID)
			targetPod = r.fallbackRouting_with_prefix_cache_1(ctx, readyPods)
		} else {
			klog.Infof("MOONCAKE: requestID=%s, selectedPod=%s", ctx.RequestID, targetPod.Status.PodIP)
		}
	}
	// Safety net: if no branch produced a pod, use prefix_cache_1 as the
	// universal fallback. This catches both unknown subAlgorithm names and
	// the case where a matched branch returned nil.
	if targetPod == nil {
		klog.Warningf("targetPod is nil after heuristic dispatch, using fallback routing for requestID: %s", ctx.RequestID)
		targetPod = r.routeWithPrefixCache1(ctx, readyPods, podIPsWithMatchingRatios)
	}
	return targetPod
}

func (r *rlOnlineRouter) fallbackRouting_with_prefix_cache_1(ctx *types.RoutingContext, readyPods []*v1.Pod) *v1.Pod {
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
		klog.Errorf("prefix_cache_1 fallback failure, using random routing for request %s", ctx.RequestID)
		var err error
		targetPod, err = selectRandomPod(readyPods, rand.Intn)
		if err != nil {
			klog.Errorf("failure to select random pod: %v", err)
			return nil
		}
	}

	// if targetPod == nil {
	// 	klog.Errorf("No suitable pod found for fallback routing")
	// 	return nil, fmt.Errorf("no suitable pod found for fallback routing")
	// }

	return targetPod
}

func (r *rlOnlineRouter) fallbackRouting_with_least_request(ctx *types.RoutingContext, readyPods []*v1.Pod) *v1.Pod {
	klog.Infof("Using fallback routing (least_request) for request %s", ctx.RequestID)
	targetPod := selectTargetPodWithLeastRequestCount(ctx, r.cache, readyPods)
	// if targetPod == nil {
	// 	klog.Errorf("least_request fallback failure, using random routing for request %s", ctx.RequestID)
	// 	var err error
	// 	targetPod, err = selectRandomPod(readyPods, rand.Intn)
	// 	if err != nil {
	// 		klog.Errorf("failure to select random pod: %v", err)
	// 		return nil, err
	// 	}
	// }

	// if targetPod == nil {
	// 	klog.Errorf("No suitable pod found for fallback routing")
	// 	return nil, fmt.Errorf("no suitable pod found for fallback routing")
	// }

	return targetPod
}

func (r *rlOnlineRouter) fallbackRouting_with_least_prefill_tokens(ctx *types.RoutingContext, readyPods []*v1.Pod) *v1.Pod {
	klog.Infof("Using fallback routing (least_prefill_tokens) for request %s", ctx.RequestID)
	targetPod := selectTargetPodWithLeastPrefillTokens(ctx, readyPods)
	return targetPod
}

func (r *rlOnlineRouter) fallbackRouting_with_least_kv_cache(ctx *types.RoutingContext, readyPods []*v1.Pod) *v1.Pod {
	klog.Infof("Using fallback routing (least_kv_cache) for request %s", ctx.RequestID)
	targetPod := selectTargetPodWithLeastKVCache(r.cache, readyPods, ctx.Model)
	// if targetPod == nil {
	// 	klog.Errorf("least_kv_cache fallback failure, using random routing for request %s", ctx.RequestID)
	// 	var err error
	// 	targetPod, err = selectRandomPod(readyPods, rand.Intn)
	// 	if err != nil {
	// 		klog.Errorf("failure to select random pod: %v", err)
	// 		return nil, err
	// 	}
	// }

	// if targetPod == nil {
	// 	klog.Errorf("No suitable pod found for fallback routing")
	// 	return nil, fmt.Errorf("no suitable pod found for fallback routing")
	// }
	return targetPod
}

// func (r *rlOnlineRouter) fallbackRouting_with_least_latency(ctx *types.RoutingContext, readyPods []*v1.Pod) (*v1.Pod, error) {
// 	klog.Infof("Using fallback routing (least_latency) for request %s", ctx.RequestID)

// 	targetPod := selectTargetPodWithLeastLatency(r.cache, readyPods, ctx.Model)
// 	if targetPod == nil {
// 		klog.Errorf("least_latency fallback failure, using random routing for request %s", ctx.RequestID)
// 		var err error
// 		targetPod, err = selectRandomPod(readyPods, rand.Intn)
// 		if err != nil {
// 			klog.Errorf("failure to select random pod: %v", err)
// 			return nil, err
// 		}
// 	}

// 	if targetPod == nil {
// 		klog.Errorf("No suitable pod found for fallback routing")
// 		return nil, fmt.Errorf("no suitable pod found for fallback routing")
// 	}
// 	return targetPod, nil
// }

func (r *rlOnlineRouter) fallbackRouting_with_random(ctx *types.RoutingContext, readyPods []*v1.Pod) *v1.Pod {
	klog.Infof("Using fallback routing (random) for request %s", ctx.RequestID)
	targetPod, _ := selectRandomPod(readyPods, rand.Intn)
	// if err != nil {
	// 	klog.Errorf("failure to select random pod: %v", err)
	// 	return nil, err
	// }
	// if targetPod == nil {
	// 	klog.Errorf("No suitable pod found for fallback routing")
	// 	return nil, fmt.Errorf("no suitable pod found for fallback routing")
	// }
	return targetPod
}

func (r *rlOnlineRouter) fallbackRouting(ctx *types.RoutingContext, readyPods []*v1.Pod, subAlgorithm string) *v1.Pod {
	if subAlgorithm == "random" {
		targetPod := r.fallbackRouting_with_random(ctx, readyPods)
		return targetPod
	} else if subAlgorithm == "least_request" {
		targetPod := r.fallbackRouting_with_least_request(ctx, readyPods)
		return targetPod
	} else if subAlgorithm == "least_kv_cache" {
		targetPod := r.fallbackRouting_with_least_kv_cache(ctx, readyPods)
		return targetPod
	} else if subAlgorithm == "prefix_cache_1" || subAlgorithm == "lodestar" {
		targetPod := r.fallbackRouting_with_prefix_cache_1(ctx, readyPods)
		return targetPod
	} else if subAlgorithm == "least_prefill_tokens" {
		targetPod := r.fallbackRouting_with_least_prefill_tokens(ctx, readyPods)
		return targetPod
	} else if subAlgorithm == "lmetric" {
		// LMETRIC fallback: use prefix_cache_1 (closest heuristic — considers both KV hits and load)
		targetPod := r.fallbackRouting_with_prefix_cache_1(ctx, readyPods)
		return targetPod
	} else {
		return nil
	}
}
