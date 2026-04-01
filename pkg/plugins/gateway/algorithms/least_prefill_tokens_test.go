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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/vllm-project/aibrix/pkg/utils"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func makePod(name, ip string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: v1.PodStatus{
			PodIP: ip,
			Conditions: []v1.PodCondition{
				{Type: v1.PodReady, Status: v1.ConditionTrue},
			},
		},
	}
}

// resetPrefillTokens zeroes out the prefill token counts for the given pod IPs
// by decrementing whatever is currently there.
func resetPrefillTokens(podIPs []string) {
	for _, ip := range podIPs {
		if tokens, ok := utils.GetNumPrefillTokensForPod(ip); ok && tokens > 0 {
			utils.DecrementNumPrefillTokensForPod(ip, tokens)
		}
	}
}

func TestLeastPrefillTokens_BasicSelection(t *testing.T) {
	pods := []*v1.Pod{
		makePod("p1", "10.0.0.1"),
		makePod("p2", "10.0.0.2"),
		makePod("p3", "10.0.0.3"),
	}
	ips := []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"}
	resetPrefillTokens(ips)

	// Set different prefill token counts: p1=500, p2=100, p3=300
	utils.IncrementNumPrefillTokensForPod("10.0.0.1", 500)
	utils.IncrementNumPrefillTokensForPod("10.0.0.2", 100)
	utils.IncrementNumPrefillTokensForPod("10.0.0.3", 300)
	defer resetPrefillTokens(ips)

	ctx := requestContext("")
	target := selectTargetPodWithLeastPrefillTokens(ctx, pods)

	assert.NotNil(t, target, "should select a pod")
	assert.Equal(t, "10.0.0.2", target.Status.PodIP, "should select pod with least prefill tokens (p2=100)")
}

func TestLeastPrefillTokens_TieBreaking(t *testing.T) {
	pods := []*v1.Pod{
		makePod("p1", "10.0.0.1"),
		makePod("p2", "10.0.0.2"),
		makePod("p3", "10.0.0.3"),
	}
	ips := []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"}
	resetPrefillTokens(ips)

	// p1=200, p2=200, p3=500 — tie between p1 and p2
	utils.IncrementNumPrefillTokensForPod("10.0.0.1", 200)
	utils.IncrementNumPrefillTokensForPod("10.0.0.2", 200)
	utils.IncrementNumPrefillTokensForPod("10.0.0.3", 500)
	defer resetPrefillTokens(ips)

	ctx := requestContext("")
	selected := map[string]int{}
	for i := 0; i < 100; i++ {
		target := selectTargetPodWithLeastPrefillTokens(ctx, pods)
		assert.NotNil(t, target)
		selected[target.Status.PodIP]++
	}

	// Both p1 and p2 should be selected, p3 should never be selected
	assert.Greater(t, selected["10.0.0.1"], 0, "p1 should be selected in tie-breaking")
	assert.Greater(t, selected["10.0.0.2"], 0, "p2 should be selected in tie-breaking")
	assert.Equal(t, 0, selected["10.0.0.3"], "p3 should never be selected (highest tokens)")
}

func TestLeastPrefillTokens_AllZero(t *testing.T) {
	pods := []*v1.Pod{
		makePod("p1", "10.0.0.1"),
		makePod("p2", "10.0.0.2"),
	}
	ips := []string{"10.0.0.1", "10.0.0.2"}
	resetPrefillTokens(ips)
	// Don't increment anything — all pods have 0 prefill tokens

	ctx := requestContext("")
	target := selectTargetPodWithLeastPrefillTokens(ctx, pods)

	assert.NotNil(t, target, "should select a pod even when all have 0 tokens")
	assert.Contains(t, []string{"10.0.0.1", "10.0.0.2"}, target.Status.PodIP)
}

func TestLeastPrefillTokens_UnknownPodDefaultsToZero(t *testing.T) {
	// p1 has tokens tracked, p2 is unknown to the prefill tracker
	pods := []*v1.Pod{
		makePod("p1", "10.0.0.1"),
		makePod("p2", "10.0.0.99"), // never incremented
	}
	ips := []string{"10.0.0.1", "10.0.0.99"}
	resetPrefillTokens(ips)

	utils.IncrementNumPrefillTokensForPod("10.0.0.1", 1000)
	defer resetPrefillTokens(ips)

	ctx := requestContext("")
	target := selectTargetPodWithLeastPrefillTokens(ctx, pods)

	assert.NotNil(t, target)
	assert.Equal(t, "10.0.0.99", target.Status.PodIP, "unknown pod should default to 0 tokens and be selected")
}

func TestLeastPrefillTokens_SinglePod(t *testing.T) {
	pods := []*v1.Pod{
		makePod("p1", "10.0.0.1"),
	}
	ips := []string{"10.0.0.1"}
	resetPrefillTokens(ips)

	utils.IncrementNumPrefillTokensForPod("10.0.0.1", 999)
	defer resetPrefillTokens(ips)

	ctx := requestContext("")
	target := selectTargetPodWithLeastPrefillTokens(ctx, pods)

	assert.NotNil(t, target)
	assert.Equal(t, "10.0.0.1", target.Status.PodIP, "single pod should always be selected")
}

func TestLeastPrefillTokens_NoPods(t *testing.T) {
	ctx := requestContext("")
	target := selectTargetPodWithLeastPrefillTokens(ctx, []*v1.Pod{})
	assert.Nil(t, target, "should return nil when no pods available")
}
