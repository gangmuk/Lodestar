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

package routingalgorithms

import (
	"math"
	"math/rand"
	"sort"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	"github.com/vllm-project/aibrix/pkg/utils/prefixcacheindexer"
	"github.com/vllm-project/aibrix/pkg/utils/tokenizer"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

const (
	defaultTokenizerType                      = "character"
	defaultPodRunningRequestImbalanceAbsCount = 8
	defaultStandardDeviationFactor            = 1
)

var (
	RouterPrefixCache                  types.RoutingAlgorithm = "prefix-cache"
	tokenizerType                                             = utils.LoadEnv("AIBRIX_PREFIX_CACHE_TOKENIZER_TYPE", "character")
	podRunningRequestImbalanceAbsCount int                    = utils.LoadEnvInt("AIBRIX_PREFIX_CACHE_POD_RUNNING_REQUEST_IMBALANCE_ABS_COUNT", defaultPodRunningRequestImbalanceAbsCount)
	standardDeviationFactor            int                    = utils.LoadEnvInt("AIBRIX_PREFIX_CACHE_STANDARD_DEVIATION_FACTOR", defaultStandardDeviationFactor)
)

func init() {
	RegisterDelayedConstructor(RouterPrefixCache, NewPrefixCacheRouter)
}

type prefixCacheRouter struct {
	cache              cache.Cache
	tokenizer          tokenizer.Tokenizer
	prefixCacheIndexer *prefixcacheindexer.PrefixHashTable
}

func NewPrefixCacheRouter() (types.Router, error) {
	var tokenizerObj tokenizer.Tokenizer
	// TODO: refactor initilization
	// supported tokenizers: ["character", "tiktoken"]
	if tokenizerType == "tiktoken" {
		tokenizerObj = tokenizer.NewTiktokenTokenizer()
	} else {
		tokenizerObj = tokenizer.NewCharacterTokenizer()
	}

	c, err := cache.Get()
	if err != nil {
		klog.Error("fail to get cache store in prefix cache router")
		return nil, err
	}

	klog.InfoS("prefix_cache_configurations",
		"tokenizer_type", tokenizerType,
		"pod_running_request_imbalance_abs_count", podRunningRequestImbalanceAbsCount,
		"matched_pods_running_requests_standard_deviation_factor", standardDeviationFactor)

	return prefixCacheRouter{
		cache:              c,
		tokenizer:          tokenizerObj,
		prefixCacheIndexer: prefixcacheindexer.NewPrefixHashTable(),
	}, nil
}

func (p prefixCacheRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	var prefixHashes []uint64
	var matchedPods map[string]int
	var targetPod *v1.Pod

	tokens, err := p.tokenizer.TokenizeInputText(ctx.Message)
	if err != nil {
		return "", err
	}

	readyPods := pods.All()
	readyPodsMap := map[string]struct{}{}
	for _, pod := range readyPods {
		readyPodsMap[pod.Name] = struct{}{}
	}

	var isLoadImbalanced bool
	targetPod, isLoadImbalanced = getTargetPodOnLoadImbalance(p.cache, readyPods)
	if isLoadImbalanced {
		prefixHashes = p.prefixCacheIndexer.GetPrefixHashes(tokens)
		klog.InfoS("prefix_cache_load_imbalanced",
			"request_id", ctx.RequestID,
			"target_pod", targetPod.Name,
			"target_pod_ip", targetPod.Status.PodIP,
			"pod_request_count", getRequestCounts(p.cache, readyPods))
	} else {
		matchedPods, prefixHashes = p.prefixCacheIndexer.MatchPrefix(tokens, ctx.Model, readyPodsMap)
		klog.InfoS("prefix_hashes", "request_id", ctx.RequestID, "prefix_hashes", prefixHashes)

		if len(matchedPods) > 0 {
			targetPod = getTargetPodFromMatchedPods(p.cache, readyPods, matchedPods)
			if targetPod != nil {
				klog.InfoS("prefix_cache_matched_pods",
					"request_id", ctx.RequestID,
					"target_pod", targetPod.Name,
					"target_pod_ip", targetPod.Status.PodIP,
					"matched_pods", matchedPods,
					"pod_request_count", getRequestCounts(p.cache, readyPods))
			} else {
				klog.InfoS("prefix_cache_skip_matched_pods",
					"request_id", ctx.RequestID,
					"matched_pods", matchedPods,
					"pod_request_count", getRequestCounts(p.cache, readyPods))
			}
		}
	}

	// no pod with prefix match, as a fallback select pod with least request count
	if len(matchedPods) == 0 || targetPod == nil {
		targetPod = selectTargetPodWithLeastRequestCount(p.cache, readyPods)
		klog.InfoS("prefix_cache_fallback_least_request_count",
			"request_id", ctx.RequestID,
			"target_pod", targetPod.Name,
			"target_pod_ip", targetPod.Status.PodIP,
			"matched_pods", matchedPods,
			"pod_request_count", getRequestCounts(p.cache, readyPods))
	}

	if len(prefixHashes) > 0 {
		p.prefixCacheIndexer.AddPrefix(prefixHashes, ctx.Model, targetPod.Name)
	}

	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}

func getTargetPodFromMatchedPods(cache cache.Cache, readyPods []*v1.Pod, matchedPods map[string]int) *v1.Pod {
	var targetPodName string
	requestCount := []float64{}

	podRequestCount := getRequestCounts(cache, readyPods)
	for _, cnt := range podRequestCount {
		requestCount = append(requestCount, float64(cnt))
	}
	meanRequestCount := mean(requestCount)
	stdDevRequestCount := standardDeviation(requestCount)

	podnames := []string{}
	for podname := range matchedPods {
		podnames = append(podnames, podname)
	}
	rand.Shuffle(len(podnames), func(i, j int) {
		podnames[i], podnames[j] = podnames[j], podnames[i]
	})

	// sort pods with decreasing %perfixmatch AND for same %prefixmatch sort by increasing request count
	sort.SliceStable(podnames, func(i, j int) bool {
		if matchedPods[podnames[i]] == matchedPods[podnames[j]] {
			return podRequestCount[podnames[i]] < podRequestCount[podnames[j]]
		}
		return matchedPods[podnames[i]] > matchedPods[podnames[j]]
	})

	// select targetpod with highest %prefixmatch and request_count within stddev
	for _, podname := range podnames {
		reqCnt := float64(podRequestCount[podname])
		if reqCnt <= meanRequestCount+float64(standardDeviationFactor)*stdDevRequestCount {
			targetPodName = podname
			break
		}
	}
	// klog.Infof("getTargetPodFromMatchedPods, matched_pods: %v, mean: %f, stddev: %f, target_pod: %s, targetPodName: %s", matchedPods, meanRequestCount, stdDevRequestCount, targetPodName, podnames)
	// for _, pod := range readyPods {
	// 	klog.Infof("readyPods, pod: %s, request_count: %d", pod.Name, podRequestCount[pod.Name])
	// }
	targetPod, _ := utils.FilterPodByName(targetPodName, readyPods)
	if targetPod == nil {
		klog.Warningf("No target pod found for matched pods: %v", matchedPods)
	}
	return targetPod
}

// func getTargetPodFromMatchedPodsByPodIP(cache cache.Cache, readyPods []*v1.Pod, matchedPods map[string]int) *v1.Pod
func routePrefixRatioAndLoad(cache cache.Cache, readyPods []*v1.Pod, matchedPods map[string]int) *v1.Pod {
	var targetPodIP string
	requestCount := []float64{}

	podRequestCount := getRequestCounts(cache, readyPods)
	for _, cnt := range podRequestCount {
		requestCount = append(requestCount, float64(cnt))
	}
	meanRequestCount := mean(requestCount)
	stdDevRequestCount := standardDeviation(requestCount)

	podIPs := []string{}
	for podIP := range matchedPods {
		podIPs = append(podIPs, podIP)
	}
	rand.Shuffle(len(podIPs), func(i, j int) {
		podIPs[i], podIPs[j] = podIPs[j], podIPs[i]
	})

	// sort pods with decreasing %perfixmatch AND for same %prefixmatch sort by increasing request count
	sort.SliceStable(podIPs, func(i, j int) bool {
		if matchedPods[podIPs[i]] == matchedPods[podIPs[j]] {
			return podRequestCount[podIPs[i]] < podRequestCount[podIPs[j]]
		}
		return matchedPods[podIPs[i]] > matchedPods[podIPs[j]]
	})

	// pods: [pod_1, pod_2, pod_3, pod_4]
	// number of requests: [7, 4, 25, 15]
	// prefix matching ratio: [0.2, 0.4, 0.8, 0.6]
	// sorted pods: [pod_3, pod_4, pod_2, pod_1]
	// Mean = (7 + 4 + 25 + 15) / 4 = 12.75
	// Std Dev = 6.1
	// Threshold = 12.75 + 6.1 = 18.85
	// pod_3 request count: 25 > 18.85
	// skip pod_3 even if it has the highest prefix matching ratio
	// pod_2 request count: 4 <= 18.85
	// select pod_2 as target pod
	var idx int
	idx = 0
	for _, podIP := range podIPs {
		reqCnt := float64(podRequestCount[podIP])
		if reqCnt <= meanRequestCount+float64(standardDeviationFactor)*stdDevRequestCount {
			targetPodIP = podIP
			break
		}
		klog.Infof("routePrefixRatioAndLoad, podIP(%s) has high prefix matching ratio but too overloaded. prefix matching ratio: %f, request_count: %d, avg_request_count: %d, std_request_count: %f", podIP, matchedPods[podIP], reqCnt, meanRequestCount, stdDevRequestCount)
		idx++
	}
	klog.Infof("routePrefixRatioAndLoad, finally selected pod: %s (%dth highest prefix match), prefix matching ratio: %f, request_count: %d", targetPodIP, idx, matchedPods[targetPodIP], podRequestCount[targetPodIP])
	// iterate readyPods and log
	targetPod, _ := utils.FilterPodByIP(targetPodIP, readyPods)
	if targetPod == nil {
		klog.Warningf("No target pod found for matched pods: %v", matchedPods)
	}
	return targetPod
}

func routePrefixRatio(cache cache.Cache, readyPods []*v1.Pod, matchedPods map[string]int) *v1.Pod {
	var targetPodIP string
	requestCount := []float64{}

	podRequestCount := getRequestCounts(cache, readyPods)
	for _, cnt := range podRequestCount {
		requestCount = append(requestCount, float64(cnt))
	}

	podIPs := []string{}
	for podIP := range matchedPods {
		podIPs = append(podIPs, podIP)
	}
	rand.Shuffle(len(podIPs), func(i, j int) {
		podIPs[i], podIPs[j] = podIPs[j], podIPs[i]
	})

	// sort pods with decreasing %perfixmatch AND for same %prefixmatch sort by increasing request count
	sort.SliceStable(podIPs, func(i, j int) bool {
		if matchedPods[podIPs[i]] == matchedPods[podIPs[j]] {
			return podRequestCount[podIPs[i]] < podRequestCount[podIPs[j]]
		}
		return matchedPods[podIPs[i]] > matchedPods[podIPs[j]]
	})

	targetPodIP = podIPs[0]

	klog.Infof("routePrefixRatio, finally selected pod: %s (%dth highest prefix match), prefix matching ratio: %f, request_count: %d", targetPodIP, 0, matchedPods[targetPodIP], podRequestCount[targetPodIP])
	// iterate readyPods and log
	targetPod, _ := utils.FilterPodByIP(targetPodIP, readyPods)
	if targetPod == nil {
		klog.Warningf("No target pod found for matched pods: %v", matchedPods)
	}
	return targetPod
}

// getTargetPodOnLoadImbalance evaluates if the load is imbalanced based on the abs difference between
// pods with min and max outstanding request counts
func getTargetPodOnLoadImbalance(cache cache.Cache, readyPods []*v1.Pod) (*v1.Pod, bool) {
	var imbalance bool
	var targetPod *v1.Pod
	targetPods := []string{}
	minValue := math.MaxInt32
	maxValue := math.MinInt32

	podRequestCount := getRequestCounts(cache, readyPods)
	for _, value := range podRequestCount {
		if value <= minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
	}
	klog.Infof("pod_request_count: %v, min: %d, max: %d", podRequestCount, minValue, maxValue)
	for podname, value := range podRequestCount {
		if minValue == value {
			targetPods = append(targetPods, podname)
		}
	}

	if maxValue-minValue > podRunningRequestImbalanceAbsCount {
		targetPod, _ = utils.FilterPodByName(targetPods[rand.Intn(len(targetPods))], readyPods)
		imbalance = true
	}

	return targetPod, imbalance
}
