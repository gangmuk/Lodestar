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
	"fmt"
	"math"
	"math/rand"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

var (
	RouterLeastPrefillTokens types.RoutingAlgorithm = "least-prefill-tokens"
)

func init() {
	RegisterDelayedConstructor(RouterLeastPrefillTokens, NewLeastPrefillTokensRouter)
}

type leastPrefillTokensRouter struct {
	cache cache.Cache
}

func NewLeastPrefillTokensRouter() (types.Router, error) {
	c, err := cache.Get()
	if err != nil {
		return nil, err
	}

	return leastPrefillTokensRouter{
		cache: c,
	}, nil
}

// Routes request to the pod with the least inflight prefill tokens
func (r leastPrefillTokensRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	targetPod := selectTargetPodWithLeastPrefillTokens(ctx, pods.All())
	if targetPod == nil {
		klog.Warning("no pods with valid prefill token metrics found for least-prefill-tokens routing; selecting a pod randomly as fallback",
			"requestID", ctx.RequestID)
		var err error
		targetPod, err = selectRandomPod(pods.All(), rand.Intn)
		if err != nil {
			return "", err
		}
	}
	if targetPod == nil {
		return "", fmt.Errorf("no pods to forward request")
	}

	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}

func (r *leastPrefillTokensRouter) SubscribedMetrics() []string {
	return []string{}
}

func selectTargetPodWithLeastPrefillTokens(ctx *types.RoutingContext, readyPods []*v1.Pod) *v1.Pod {
	var targetPod *v1.Pod
	targetPods := []string{}

	// Get prefill tokens for all pods (keyed by podIP)
	podPrefillTokens := utils.GetNumPrefillTokensForAllPods()

	// Build a map of podIP -> token count for ready pods only, defaulting to 0
	readyPodTokens := make(map[string]int, len(readyPods))
	for _, pod := range readyPods {
		if tokens, ok := podPrefillTokens[pod.Status.PodIP]; ok {
			readyPodTokens[pod.Status.PodIP] = tokens
		} else {
			readyPodTokens[pod.Status.PodIP] = 0
		}
	}

	// Find the minimum prefill token count
	minTokens := math.MaxInt32
	for _, tokens := range readyPodTokens {
		if tokens <= minTokens {
			minTokens = tokens
		}
	}

	// Collect all pods with the minimum token count
	for podIP, tokens := range readyPodTokens {
		if tokens == minTokens {
			targetPods = append(targetPods, podIP)
		}
	}

	// Randomly pick among tied pods
	if len(targetPods) > 0 {
		targetPod, _ = utils.FilterPodByIP(targetPods[rand.Intn(len(targetPods))], readyPods)
	}

	targetPodName, targetPodIP := "", ""
	if targetPod != nil {
		targetPodName = targetPod.Name
		targetPodIP = targetPod.Status.PodIP
	}
	klog.InfoS("routing_decision", "algorithm", "least_prefill_tokens", "request_id", ctx.RequestID, "target_pod", targetPodName, "target_pod_ip", targetPodIP, "pod_prefill_tokens", readyPodTokens)

	return targetPod
}
