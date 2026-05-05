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
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	"github.com/vllm-project/aibrix/pkg/utils/prefixcacheindexer"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

// decodeByteArrayToTokenIDs reverses utils.TokenizeInputTextToByteArray, which
// packs token IDs as little-endian uint32 (4 bytes per token). Trailing bytes
// that don't form a full 4-byte chunk are ignored.
func decodeByteArrayToTokenIDs(b []byte) []int {
	n := len(b) / 4
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = int(binary.LittleEndian.Uint32(b[i*4 : (i+1)*4]))
	}
	return out
}

// isLargeNode: reference global_scheduler_with_time.py:258.
//
//	def is_large_node(self, node):
//	    return node.num_tokens > node.context_so_far
//
// where context_so_far = context_length - num_tokens. A node is "large" when
// its own chunk outweighs the cached prefix leading up to it. The sliding
// window histogram is keyed on the deepest large ancestor of a leaf (the
// "important node"), so many leaves that share a common important ancestor
// collapse into one histogram entry and node_to_count becomes meaningful.
func isLargeNode(node *prefixcacheindexer.TreeNode) bool {
	if node == nil {
		return false
	}
	numTokens := node.NumTokens()
	contextSoFar := node.ContextLength() - numTokens
	return numTokens > contextSoFar
}

// getImportantNode: reference global_scheduler_with_time.py:261.
// Walks from the leaf up the tree and returns the deepest "large" ancestor.
// Python crashes with AttributeError if no large ancestor exists; we prefer
// to fall back to the input node so the histogram still gets an update for
// pathological cases (e.g. a fully-cached prompt with zero new tokens).
func getImportantNode(node *prefixcacheindexer.TreeNode) *prefixcacheindexer.TreeNode {
	cur := node
	for cur != nil {
		if isLargeNode(cur) {
			return cur
		}
		cur = cur.GetParent()
	}
	return node
}

var (
	RouterPrefixCacheAndLoad types.RoutingAlgorithm = "preble"
	// targetGPU                                       = utils.LoadEnv("TARGET_GPU", "L20")
	prefixRoutingThreshold = utils.LoadEnvInt("PREFIX_ROUTING_THRESHOLD", 50)
)

func init() {
	RegisterDelayedConstructor(RouterPrefixCacheAndLoad, NewPrefixCacheAndLoadRouter)
}

// prebleOutstandingRequest carries the minimum state needed to undo a Route's
// ref_counter increments when the request finishes: the leaf node (so we can
// walk up to root) and the selected pod IP (the key to decrement).
type prebleOutstandingRequest struct {
	leaf  *prefixcacheindexer.TreeNode
	podIP string
}

// prebleOutstandingRequests maps requestID → *prebleOutstandingRequest for every
// request routed by Preble that hasn't finished yet. LoadAndDelete on completion
// makes PrebleOnRequestComplete idempotent — repeat calls (e.g., from multiple
// cleanup paths) are no-ops.
var prebleOutstandingRequests sync.Map

// PrebleOnRequestComplete decrements the per-pod ref_counter on every ancestor
// of the request's stashed leaf node. No-op when no state was stashed — i.e.,
// when the request wasn't routed by Preble or was already cleaned up — so the
// caller can invoke it unconditionally from the gateway response path without
// needing to check the routing algorithm.
//
// Reference: global_scheduler_with_time.py:367 where finish_request calls
// self.cache.remove_completed_input_ids(input_ids, runtime_id), which in turn
// walks leaf→root decrementing ref_counter[runtime_id] (global_lru_cache.py:221).
func PrebleOnRequestComplete(requestID string) {
	raw, ok := prebleOutstandingRequests.LoadAndDelete(requestID)
	if !ok {
		return
	}
	state := raw.(*prebleOutstandingRequest)
	for cur := state.leaf; cur != nil; cur = cur.GetParent() {
		cur.DecrementRefCounterForPod(state.podIP)
	}
}

const (
	// prefixRoutingThreshold = 50                      // 50%
	defaultDecodingLength = 45                      // FIXME: decode length is hardcoded. Preble as well.
	slidingWindowPeriod   = 3 * time.Minute         // NOTE: hardcoded
	evictionLoopInterval  = 1000 * time.Millisecond // NOTE: hardcoded

	// Rebalancing constants — from reference Preble
	// (global_scheduler_with_time.py:207-214, 378, 393). handle_important_node_stealing is called
	// from runtime_selector on every request when enable_rebalancing is true (the default).
	// rebalancingMinRequestsPerPod: skip rebalancing until sum(per_gpu_load) >= this * num_pods.
	// Warmup gate — costs aren't meaningful before the histogram has seen traffic.
	rebalancingMinRequestsPerPod = 50
	// rebalancingMinNodeCount: only consider histogram nodes with node_to_count > this value
	// (reference uses strictly > 1, i.e. >= 2, for both the filtered cost fn and the stealable set).
	rebalancingMinNodeCount = 1
	// rebalancingChainLength: skip rebalancing when (leaf.depth - important.depth) >= this.
	// Reference comment: "Ignore longer chains for Infercept optimizations".
	rebalancingChainLength = 3
	// rebalancingHighLoadThreshold: rebalance only when the most-loaded pod's cost exceeds
	// this multiplier of the least-loaded pod's cost. 1.5x matches the Python HIGH_LOAD_THRESHOLD.
	rebalancingHighLoadThreshold = 1.5
)

type SlidingWindowHistogram struct {
	mu                         sync.RWMutex
	windowDuration             time.Duration
	histogram                  map[*prefixcacheindexer.TreeNode]int
	nodeToCount                map[*prefixcacheindexer.TreeNode]int
	hitTokens                  map[*prefixcacheindexer.TreeNode]int
	promptTokens               map[*prefixcacheindexer.TreeNode]int
	decodingSize               map[*prefixcacheindexer.TreeNode]int
	timestamps                 []histogramEntry
	numPods                    int
	podAllocations             map[*prefixcacheindexer.TreeNode]map[int]bool
	currentDecodeLengthsPerPod map[string]int       // pod name -> total decode length
	avgTimePerTokenPerPod      map[string][]float64 // pod name -> list of time/token measurements
	perNodeTotalDecodeLengths  map[*prefixcacheindexer.TreeNode]int
	// currentPrefillCostPerPod   map[string]float64 // pod name -> prefill cost
	// perNodePrefillCost         map[*prefixcacheindexer.TreeNode]float64
}

type histogramEntry struct {
	timestamp time.Time
	node      *prefixcacheindexer.TreeNode
	leafNode  *prefixcacheindexer.TreeNode
}

type prefixCacheAndLoadRouter struct {
	cache          *prefixcacheindexer.LPRadixCache
	histogram      *SlidingWindowHistogram
	numPods        int
	podAllocations map[*prefixcacheindexer.TreeNode]map[int]bool
	// cacheMu        sync.RWMutex // Protects cache operations
	// histogramMu sync.RWMutex // Protects histogram operations
	podsMu sync.RWMutex // Protects pod-related data

	// Monotonic total request counter for the rebalancing warmup gate.
	// Matches reference sum(self.per_gpu_load.values()) which is also monotonic
	// (per_gpu_load is incremented on every runtime_selector call in Python).
	totalRequests atomic.Int64
}

// Find all prefix matches with their depths
type prefixMatch struct {
	node        *prefixcacheindexer.TreeNode
	pods        []*v1.Pod
	matchLength int
	depth       int
}

type PrefillTimeParams struct {
	NumRequests      int
	NumBatchedTokens int
	TotalContext     int
	InputIDLens      []int
	NumUniqueKV      int
	SeqLens          []int
}

func mistral7BA6000LinearTime(numBatchedTokens int) float64 {
	if numBatchedTokens >= 384 {
		return (0.10842571*float64(numBatchedTokens) + 4.209777054806409) / 1000.0
	} else if numBatchedTokens >= 192 {
		return (-118 + 1.25*float64(numBatchedTokens) - 2.56e-3*math.Pow(float64(numBatchedTokens), 2)) / 1000.0
	}
	return 22.0 / 1000.0
}

func mistral7BA6000AttentionTime(numReqs, totalContext, numUniqueKV int) float64 {
	if numUniqueKV == 0 {
		numUniqueKV = totalContext
	}

	var forwardTime float64
	if totalContext <= 1024 {
		forwardTime = 0.32
	} else {
		forwardTime = 1.86e-4*float64(totalContext) + 0.159
		if float64(numUniqueKV)/float64(numReqs) <= 1024 && numReqs*numUniqueKV <= 32*256*2048 {
			forwardTime /= 2
		}
	}
	return forwardTime / 1000.0
}

// Adjusted for V100 characteristics
func mistral7BV100LinearTime(numBatchedTokens int) float64 {
	if numBatchedTokens >= 384 {
		// Increased coefficient due to lower compute power
		// ~2.5x increase for linear component due to slower tensor cores
		return (0.27106428*float64(numBatchedTokens) + 10.52444263) / 1000.0
	} else if numBatchedTokens >= 192 {
		// Adjusted quadratic coefficient to reflect V100's architecture
		return (-295 + 3.125*float64(numBatchedTokens) - 6.4e-3*math.Pow(float64(numBatchedTokens), 2)) / 1000.0
	}
	// Base latency increased
	return 55.0 / 1000.0
}

func mistral7BV100AttentionTime(numReqs, totalContext, numUniqueKV int) float64 {
	if numUniqueKV == 0 {
		numUniqueKV = totalContext
	}

	var forwardTime float64
	if totalContext <= 1024 {
		// Increased base attention time for shorter sequences
		forwardTime = 0.80
	} else {
		// Increased linear coefficient and base time for attention
		// Memory bandwidth is better but compute is slower
		forwardTime = 4.65e-4*float64(totalContext) + 0.398
		if float64(numUniqueKV)/float64(numReqs) <= 1024 && numReqs*numUniqueKV <= 32*256*2048 {
			forwardTime /= 2
		}
	}
	return forwardTime / 1000.0
}

func calculateAttnQuadA6000(numTokens int, seqLen *int) float64 {
	var attnQuad float64
	if seqLen == nil {
		// Case 1: No sequence length provided
		if numTokens >= 4096 {
			attnQuad += -7.37 + 3.86e-3*float64(numTokens) + 2.16e-6*math.Pow(float64(numTokens), 2)
		}
	} else {
		// Case 2: Sequence length provided
		if numTokens*(*seqLen) > 1024*1024 {
			attnQuad += 1.13e-3*float64(numTokens) +
				1.75e-3*float64(*seqLen) +
				2.19e-6*float64(numTokens)*float64(*seqLen)
		}
	}
	return attnQuad / 1000.0
}

func calculateAttnQuadV100(numTokens int, seqLen *int) float64 {
	var attnQuad float64
	if seqLen == nil {
		// Case 1: No sequence length provided
		if numTokens >= 4096 {
			// ~2.5x slower for quadratic costs due to older tensor cores and memory architecture
			attnQuad += -18.425 + // from -7.37
				9.65e-3*float64(numTokens) + // from 3.86e-3
				5.4e-6*math.Pow(float64(numTokens), 2) // from 2.16e-6
		}
	} else {
		// Case 2: Sequence length provided
		if numTokens*(*seqLen) > 1024*1024 {
			attnQuad += 2.825e-3*float64(numTokens) + // from 1.13e-3
				4.375e-3*float64(*seqLen) + // from 1.75e-3
				5.475e-6*float64(numTokens)*float64(*seqLen) // from 2.19e-6
		}
	}
	return attnQuad / 1000.0
}

// Performance models for NVIDIA A30 + Mistral 7B.
//
// A30 is the GA100 die (same silicon as A100) with 56/108 SMs enabled — so
// it inherits A100's memory hierarchy (40 MB L2, HBM2) but roughly half the
// compute. Key ratios vs the A6000 (GA102) baseline:
//
//   | metric                | A6000   | A30    | ratio (A30/A6000) |
//   |-----------------------|---------|--------|-------------------|
//   | FP16 tensor dense     | 154.8 T | 82.6 T | 0.534  (1.87x slower) |
//   | HBM bandwidth         | 768     | 933    | 1.21   (A30 faster) |
//   | L2 cache              | 6 MB    | 40 MB  | 6.7    (A30 much bigger) |
//   | SMs                   | 84      | 56     | 0.667  |
//
// Scaling strategy (relative to the A6000 Mistral-7B baseline above):
//   - LinearTime: ~1.85x. MLP + QKV projections are pure tensor-core matmul,
//     gated by dense FP16 TFLOPS. Tracks compute ratio almost exactly.
//   - AttentionTime: ~1.4x short-context, ~1.3x long-context. At <~300 tokens
//     of Mistral-7B KV, state fits in A30's 40 MB L2 but not A6000's 6 MB,
//     so A30's cache partially offsets its compute deficit. At longer context
//     (>1024 tokens, ~134+ MB of KV) both cards go to HBM — A30's 1.21x BW
//     advantage narrows the gap further.
//   - AttnQuad: ~1.85x. The O(n^2) attention matmul is compute-bound.
//
// Cross-check: V100 uses a flat 2.5x scaling. A30 at 1.85x means A30 is
// ~1.35x faster than V100 in this cost model. Real LLM serving numbers
// (vLLM, Llama-2-7B) put A30 at ~2x V100 throughput — so 1.85x is a
// conservative, defensible first cut that won't over-reward A30 pods.
// Recalibrate from measured latency once you have production traces.

func mistral7BA30LinearTime(numBatchedTokens int) float64 {
	// 1.85x A6000: scales with dense FP16 tensor-core throughput.
	// 0.10842571 * 1.85 = 0.20058756; 4.209777054806409 * 1.85 = 7.7880876.
	if numBatchedTokens >= 384 {
		return (0.20058756*float64(numBatchedTokens) + 7.7880876) / 1000.0
	} else if numBatchedTokens >= 192 {
		// -118 * 1.85 = -218.3; 1.25 * 1.85 = 2.3125; 2.56e-3 * 1.85 = 4.736e-3.
		return (-218.3 + 2.3125*float64(numBatchedTokens) - 4.736e-3*math.Pow(float64(numBatchedTokens), 2)) / 1000.0
	}
	// Base latency: 22.0 * 1.85 = 40.7.
	return 40.7 / 1000.0
}

func mistral7BA30AttentionTime(numReqs, totalContext, numUniqueKV int) float64 {
	if numUniqueKV == 0 {
		numUniqueKV = totalContext
	}

	var forwardTime float64
	if totalContext <= 1024 {
		// 1.4x A6000. A30's 40 MB L2 covers <~300 tokens of Mistral-7B KV
		// where A6000 already spills to HBM, partially offsetting A30's
		// compute deficit at the short end. 0.32 * 1.4 = 0.448.
		forwardTime = 0.448
	} else {
		// 1.3x on the linear coef (memory-bound long context; A30's HBM BW
		// partly compensates for its slower tensor cores) and 1.35x on the
		// constant term (kernel-launch + softmax overhead that still favors
		// A6000's higher CUDA-core count).
		// 1.86e-4 * 1.3 = 2.418e-4; 0.159 * 1.35 = 0.21465.
		forwardTime = 2.418e-4*float64(totalContext) + 0.21465
		if float64(numUniqueKV)/float64(numReqs) <= 1024 && numReqs*numUniqueKV <= 32*256*2048 {
			forwardTime /= 2
		}
	}
	return forwardTime / 1000.0
}

func calculateAttnQuadA30(numTokens int, seqLen *int) float64 {
	var attnQuad float64
	// 1.85x A6000: O(n^2) attention matmul is compute-bound and tracks the
	// FP16 dense tensor-core throughput ratio.
	if seqLen == nil {
		// Case 1: No sequence length provided.
		if numTokens >= 4096 {
			attnQuad += -13.6345 + // from A6000: -7.37
				7.141e-3*float64(numTokens) + // from A6000: 3.86e-3
				3.996e-6*math.Pow(float64(numTokens), 2) // from A6000: 2.16e-6
		}
	} else {
		// Case 2: Sequence length provided.
		if numTokens*(*seqLen) > 1024*1024 {
			attnQuad += 2.0905e-3*float64(numTokens) + // from A6000: 1.13e-3
				3.2375e-3*float64(*seqLen) + // from A6000: 1.75e-3
				4.0515e-6*float64(numTokens)*float64(*seqLen) // from A6000: 2.19e-6
		}
	}
	return attnQuad / 1000.0
}

// Performance models for NVIDIA A30 + Llama-3 8B Instruct.
//
// Hardware factor (A30 / A6000) is inherited from the mistral7BA30* block:
// LinearTime 1.85x, AttentionTime 1.4x short / 1.3x-coef 1.35x-const long,
// AttnQuad 1.85x. See the A30 comment block above mistral7BA30LinearTime for
// the derivation.
//
// Model factor (Llama-3-8B / Mistral-7B): architectures are near-identical
// (both hidden=4096, layers=32, heads=32, KV-heads=8 GQA 4:1, head_dim=128,
// intermediate=14336). The only divergence is vocab: 128256 vs 32000, i.e.
// Llama-3's LM head is ~4x bigger. For pure prefill the LM head fires only
// on the last token, so its FLOP contribution amortized over an N-token
// prefix is <~5%. Attention and AttnQuad are transformer-body only and
// scale with (hidden, heads, head_dim) — model factor 1.0x.
//
//   | component   | model factor | rationale                               |
//   |-------------|--------------|-----------------------------------------|
//   | LinearTime  | 1.10x        | ~1.05x FLOP floor + small empirical     |
//   |             |              | margin for kernel/tokenizer overhead.   |
//   |             |              | Below L20's 1.15x, which appears to     |
//   |             |              | conflate prefill + decode.              |
//   | AttentionTime | 1.00x      | identical heads/hidden/GQA config.      |
//   | AttnQuad    | 1.00x        | identical attention compute shape.      |
//
// Combined A30 x Llama-3 factors vs A6000-Mistral baseline:
//   LinearTime:     1.85 * 1.10 = 2.035x
//   AttentionTime:  1.4x / 1.3x-coef / 1.35x-const (model factor = 1)
//   AttnQuad:       1.85x (model factor = 1)
//
// Sanity check @ n=500: A6000-Mistral LinearTime=58.4 ms, A30-Llama-3=118.9 ms
// (2.035x), A30-Mistral=108.1 ms (Llama-3 bump = 1.10x, as intended).
// Recalibrate from measured latency once A30+Llama-3 production traces exist.

func llama3_8B_A30_LinearTime(numBatchedTokens int) float64 {
	// 2.035x A6000-Mistral: 1.85 hardware * 1.10 model.
	// 0.10842571 * 2.035 = 0.22064632; 4.209777054806409 * 2.035 = 8.56689631.
	if numBatchedTokens >= 384 {
		return (0.22064632*float64(numBatchedTokens) + 8.56689631) / 1000.0
	} else if numBatchedTokens >= 192 {
		// -118 * 2.035 = -240.13; 1.25 * 2.035 = 2.54375; 2.56e-3 * 2.035 = 5.2096e-3.
		return (-240.13 + 2.54375*float64(numBatchedTokens) - 5.2096e-3*math.Pow(float64(numBatchedTokens), 2)) / 1000.0
	}
	// Base latency: 22.0 * 2.035 = 44.77.
	return 44.77 / 1000.0
}

func llama3_8B_A30_AttentionTime(numReqs, totalContext, numUniqueKV int) float64 {
	if numUniqueKV == 0 {
		numUniqueKV = totalContext
	}

	var forwardTime float64
	if totalContext <= 1024 {
		// 1.4x A6000 (hardware-only; model factor = 1 since attention compute
		// is identical between Llama-3-8B and Mistral-7B). 0.32 * 1.4 = 0.448.
		forwardTime = 0.448
	} else {
		// 1.3x linear coef (memory-bound long context) and 1.35x constant term
		// (kernel-launch / softmax overhead). Model factor = 1.
		// 1.86e-4 * 1.3 = 2.418e-4; 0.159 * 1.35 = 0.21465.
		forwardTime = 2.418e-4*float64(totalContext) + 0.21465
		if float64(numUniqueKV)/float64(numReqs) <= 1024 && numReqs*numUniqueKV <= 32*256*2048 {
			forwardTime /= 2
		}
	}
	return forwardTime / 1000.0
}

func calculateAttnQuadA30Llama3(numTokens int, seqLen *int) float64 {
	var attnQuad float64
	// 1.85x A6000 (hardware-only; model factor = 1 since quadratic attention
	// compute is identical between Llama-3-8B and Mistral-7B). Coefficients
	// match calculateAttnQuadA30 — they are kept separate so future
	// Llama-3-specific recalibration can diverge from the Mistral calibration.
	if seqLen == nil {
		// Case 1: No sequence length provided.
		if numTokens >= 4096 {
			attnQuad += -13.6345 + // from A6000: -7.37
				7.141e-3*float64(numTokens) + // from A6000: 3.86e-3
				3.996e-6*math.Pow(float64(numTokens), 2) // from A6000: 2.16e-6
		}
	} else {
		// Case 2: Sequence length provided.
		if numTokens*(*seqLen) > 1024*1024 {
			attnQuad += 2.0905e-3*float64(numTokens) + // from A6000: 1.13e-3
				3.2375e-3*float64(*seqLen) + // from A6000: 1.75e-3
				4.0515e-6*float64(numTokens)*float64(*seqLen) // from A6000: 2.19e-6
		}
	}
	return attnQuad / 1000.0
}

// Performance models for NVIDIA L20 + Llama-3 8B Instruct
// Based on scaling from A6000 Mistral 7B baseline

func llama3_8B_L20_LinearTime(numBatchedTokens int) float64 {
	// L20 has ~30% less compute than A6000, Llama-3 8B has ~15% more compute than Mistral 7B
	// Combined scaling factor: 1.3 * 1.15 = 1.495 ≈ 1.5x
	if numBatchedTokens >= 384 {
		return (0.16264*float64(numBatchedTokens) + 6.314665) / 1000.0
	} else if numBatchedTokens >= 192 {
		return (-177 + 1.875*float64(numBatchedTokens) - 3.84e-3*math.Pow(float64(numBatchedTokens), 2)) / 1000.0
	}
	return 33.0 / 1000.0
}

func llama3_8B_L20_AttentionTime(numReqs, totalContext, numUniqueKV int) float64 {
	if numUniqueKV == 0 {
		numUniqueKV = totalContext
	}

	var forwardTime float64
	if totalContext <= 1024 {
		// L20 has better memory architecture than V100 but less compute than A6000
		// Scaling factor: ~1.2x from A6000 baseline
		forwardTime = 0.384
	} else {
		// Linear scaling with context length, adjusted for L20 characteristics
		forwardTime = 2.232e-4*float64(totalContext) + 0.191
		if float64(numUniqueKV)/float64(numReqs) <= 1024 && numReqs*numUniqueKV <= 32*256*2048 {
			forwardTime /= 2
		}
	}
	return forwardTime / 1000.0
}

func calculateAttnQuadL20(numTokens int, seqLen *int) float64 {
	var attnQuad float64
	// L20 quadratic attention costs - scaled between A6000 and V100
	// Using ~1.8x scaling factor from A6000 baseline
	if seqLen == nil {
		// Case 1: No sequence length provided
		if numTokens >= 4096 {
			attnQuad += -13.266 + // from A6000: -7.37
				6.948e-3*float64(numTokens) + // from A6000: 3.86e-3
				3.888e-6*math.Pow(float64(numTokens), 2) // from A6000: 2.16e-6
		}
	} else {
		// Case 2: Sequence length provided
		if numTokens*(*seqLen) > 1024*1024 {
			attnQuad += 2.034e-3*float64(numTokens) + // from A6000: 1.13e-3
				3.15e-3*float64(*seqLen) + // from A6000: 1.75e-3
				3.942e-6*float64(numTokens)*float64(*seqLen) // from A6000: 2.19e-6
		}
	}
	return attnQuad / 1000.0
}

func (h *SlidingWindowHistogram) getPrefillCost(targetGPU string, node *prefixcacheindexer.TreeNode) float64 {
	missRate := 1.0
	if h.promptTokens[node] > 0 {
		missRate = 1.0 - (float64(h.hitTokens[node]) / float64(h.promptTokens[node])) // ????
	}
	numTokens := node.NumTokens()
	contextLength := node.ContextLength()
	baseTime := 0.0
	// GPU strings match the K8s labels set by utils.GetGPUModel / extractGPUModelFromPod
	// (e.g. "NVIDIA-A30", "Tesla-V100") so the same targetGPU value flowing through
	// from the gateway header dispatches cleanly here without any normalization.
	if targetGPU == "A6000" {
		baseTime = mistral7BA6000LinearTime(numTokens) + mistral7BA6000AttentionTime(1, contextLength, numTokens)
	} else if targetGPU == "Tesla-V100" {
		baseTime = mistral7BV100LinearTime(numTokens) + mistral7BV100AttentionTime(1, contextLength, numTokens)
	} else if targetGPU == "NVIDIA-A30" {
		baseTime = llama3_8B_A30_LinearTime(numTokens) + llama3_8B_A30_AttentionTime(1, contextLength, numTokens)
	} else if targetGPU == "NVIDIA-L20" {
		baseTime = llama3_8B_L20_LinearTime(numTokens) + llama3_8B_L20_AttentionTime(1, contextLength, numTokens)
	} else {
		klog.V(5).Infof("Unknown target GPU: %s. Assume Tesla-V100 as default", targetGPU)
		baseTime = mistral7BV100LinearTime(numTokens) + mistral7BV100AttentionTime(1, contextLength, numTokens)
	}

	attnQuad := 0.0
	if targetGPU == "A6000" {
		attnQuad = calculateAttnQuadA6000(numTokens, nil)
	} else if targetGPU == "Tesla-V100" {
		attnQuad = calculateAttnQuadV100(numTokens, nil)
	} else if targetGPU == "NVIDIA-A30" {
		attnQuad = calculateAttnQuadA30Llama3(numTokens, nil)
	} else if targetGPU == "NVIDIA-L20" {
		attnQuad = calculateAttnQuadL20(numTokens, nil)
	} else {
		klog.V(5).Infof("Unknown target GPU: %s. Assume Tesla-V100 as default", targetGPU)
		attnQuad = calculateAttnQuadV100(numTokens, nil)
	}
	prefillTime := (baseTime + attnQuad) / 0.9
	numPods := node.GetModelToPodCount() // You might need to adjust this based on your actual GPU allocation tracking
	klog.V(5).Infof("numTokens: %d, contextLength: %d, targetGPU: %s", numTokens, contextLength, targetGPU)
	klog.V(5).Infof("prefillTime: %.2f = (Base time(%.2f) + attnQuad(%.2f)) / 0.9", prefillTime, baseTime, attnQuad)
	totalPrefillCost := missRate * float64(h.nodeToCount[node]) * prefillTime / float64(numPods)
	klog.V(5).Infof("totalPrefillCost: %.2f = miss rate(%.2f) * nodeToCount(%d) * prefillTime(%.2f) / numPods(%d)", totalPrefillCost, missRate, h.nodeToCount[node], prefillTime, numPods)
	return totalPrefillCost
}

// TODO: It needs to read the running pods accordingly.
// Also, the radix tree cache does not support varying number of pods.
// The tree data structure should be updated in real time with varying number of pods.
// Especially when a pod is removed, the corresponding TreeNode should be removed from the RadixTree and from the related data structures in SlidingWindowHistogram.
func NewPrefixCacheAndLoadRouter() (types.Router, error) {
	numPods := 0 // NOTE: it will be initialized in Route function. This number can change dynamically due to scaling or failure.
	histogram := &SlidingWindowHistogram{
		windowDuration:             slidingWindowPeriod,
		histogram:                  make(map[*prefixcacheindexer.TreeNode]int),
		nodeToCount:                make(map[*prefixcacheindexer.TreeNode]int),
		hitTokens:                  make(map[*prefixcacheindexer.TreeNode]int),
		promptTokens:               make(map[*prefixcacheindexer.TreeNode]int),
		decodingSize:               make(map[*prefixcacheindexer.TreeNode]int),
		numPods:                    numPods,
		podAllocations:             make(map[*prefixcacheindexer.TreeNode]map[int]bool),
		currentDecodeLengthsPerPod: make(map[string]int),
		perNodeTotalDecodeLengths:  make(map[*prefixcacheindexer.TreeNode]int),
		avgTimePerTokenPerPod:      make(map[string][]float64),
		// currentPrefillCostPerPod:   make(map[string]float64),
		// perNodePrefillCost:         make(map[*prefixcacheindexer.TreeNode]float64),
	}

	router := &prefixCacheAndLoadRouter{
		cache:          prefixcacheindexer.NewLPRadixCache(numPods),
		histogram:      histogram,
		numPods:        numPods,
		podAllocations: make(map[*prefixcacheindexer.TreeNode]map[int]bool),
	}

	// Start eviction ticker
	go router.evictionLoop()

	// Prime the shared background vLLM metrics scraper so preble requests see
	// the same vllm{GPUKVCacheUsage,NumRequestsRunning,NumRequestsWaiting,
	// NumPreemptions} fields in the latency_metrics log as rl-online-router
	// requests. Guarded by bgScraperOnce in rl_routing.go — idempotent.
	if c, err := cache.Get(); err == nil {
		startBackgroundMetricsScraper(c)
	} else {
		klog.Errorf("preble router: failed to get cache store; background vLLM metrics will be unavailable: %v", err)
	}

	return router, nil
}

func (h *SlidingWindowHistogram) removeEvictedNodes(nodes []*prefixcacheindexer.TreeNode) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Create a map for faster lookup
	nodeMap := make(map[*prefixcacheindexer.TreeNode]bool)
	for _, node := range nodes {
		nodeMap[node] = true
	}

	// Filter timestamps
	newTimestamps := make([]histogramEntry, 0)
	for _, entry := range h.timestamps {
		if !nodeMap[entry.node] {
			newTimestamps = append(newTimestamps, entry)
		}
	}
	h.timestamps = newTimestamps

	// Remove from all maps
	for node := range nodeMap {
		delete(h.histogram, node)
		delete(h.nodeToCount, node)
		delete(h.hitTokens, node)
		delete(h.promptTokens, node)
		delete(h.decodingSize, node)
		delete(h.podAllocations, node)
	}
}

func (h *SlidingWindowHistogram) removeOldEntries(currentTime time.Time) {
	h.mu.Lock()
	defer h.mu.Unlock()
	windowStart := currentTime.Add(-h.windowDuration)
	newTimestamps := make([]histogramEntry, 0)
	for _, entry := range h.timestamps {
		if entry.timestamp.After(windowStart) {
			newTimestamps = append(newTimestamps, entry)
		} else {
			node := entry.node
			leafNode := entry.leafNode
			h.histogram[node] -= leafNode.ContextLength()
			h.nodeToCount[node]--
			h.hitTokens[node] -= leafNode.ContextLength() - leafNode.NumTokens()
			h.promptTokens[node] -= leafNode.ContextLength()
			if h.histogram[node] <= 0 {
				delete(h.histogram, node)
				delete(h.nodeToCount, node)
				delete(h.hitTokens, node)
				delete(h.promptTokens, node)
				delete(h.decodingSize, node)
				delete(h.podAllocations, node)
			}
		}
	}
	h.timestamps = newTimestamps
}

// rebalancePodCost is the pair passed through rebalanceRecursive: a pod IP
// plus its current aggregate allocation cost. Kept at package scope so the
// entry point and recursive helper agree on the element type.
type rebalancePodCost struct {
	podIP string
	cost  float64
}

// handleImportantNodeStealing is the entry point for Preble's rebalancing.
// Reference: global_scheduler_with_time.py:377 handle_important_node_stealing.
// Called from Route after the histogram update, on every request, and gated
// by the warmup threshold (total monotonic request count) and — at the call
// site — by the leaf-to-important chain length.
//
// The rebalancing scans the histogram for "large" nodes that are currently
// allocated to the most-loaded pod and migrates a subset of them to the
// least-loaded pod until the cost imbalance drops below 1.5x. The migration
// changes the node's ModelToPods[model] allocation — all future requests
// whose prefix matches this node (or its descendants) via the prefix-aware
// branch will be routed to the new pod, draining load from the hot pod.
//
// Skipped optimizations relative to reference:
//   - TTFT overload detector (the single-stealable-node case). The reference
//     code only consults it in a narrow branch; we skip that branch entirely.
//   - Eviction-aware cost (enable_eviction path in the reference).
func (p *prefixCacheAndLoadRouter) handleImportantNodeStealing(routingCtx *types.RoutingContext, readyPods []*v1.Pod) {
	numPods := len(readyPods)
	if numPods < 2 {
		return
	}
	if p.totalRequests.Load() < int64(rebalancingMinRequestsPerPod*numPods) {
		return
	}

	// Cost per pod, filtered to histogram nodes with node_to_count > 1
	// (matches Python's current_allocation_per_gpu_with_atleast_min_load(2)).
	costs := p.histogram.getFilteredAllocationCostPerPodForModel(
		routingCtx.TargetGPU, routingCtx.Model, rebalancingMinNodeCount+1)

	allocs := make([]rebalancePodCost, 0, numPods)
	for _, pod := range readyPods {
		allocs = append(allocs, rebalancePodCost{pod.Status.PodIP, costs[pod.Status.PodIP]})
	}
	// Descending by cost so allocs[0] is largest, allocs[len-1] is smallest.
	sort.Slice(allocs, func(i, j int) bool { return allocs[i].cost > allocs[j].cost })

	p.rebalanceRecursive(routingCtx.Model, routingCtx.TargetGPU, allocs)
}

// rebalanceRecursive: reference handle_important_node_stealing_recursive
// (global_scheduler_with_time.py:387). Takes largest and smallest pods from
// the current slice; if imbalance > 1.5x, migrates large-nodes from largest
// to smallest until the imbalance clears or moving more would invert it;
// then recurses on the slice minus the largest (keeping the updated
// smallest in place — matching reference behavior, which does not re-sort).
func (p *prefixCacheAndLoadRouter) rebalanceRecursive(model, targetGPU string, allocs []rebalancePodCost) {
	if len(allocs) <= 1 {
		return
	}
	largerIdx := 0
	smallerIdx := len(allocs) - 1
	largerPod := allocs[largerIdx].podIP
	largerCost := allocs[largerIdx].cost
	smallerPod := allocs[smallerIdx].podIP
	smallerCost := allocs[smallerIdx].cost

	if largerCost < rebalancingHighLoadThreshold*smallerCost {
		return // already balanced enough
	}

	// Collect stealable nodes: node_to_count > 1, is_large, larger pod holds it.
	type nodeCost struct {
		cost float64
		node *prefixcacheindexer.TreeNode
	}
	var candidates []nodeCost
	p.histogram.mu.RLock()
	for node := range p.histogram.histogram {
		if p.histogram.nodeToCount[node] <= rebalancingMinNodeCount {
			continue
		}
		if !isLargeNode(node) {
			continue
		}
		pods := node.GetPodsForModel(model)
		if pods == nil {
			continue
		}
		if _, held := pods[largerPod]; !held {
			continue
		}
		candidates = append(candidates, nodeCost{
			cost: p.histogram.getNodeCost(targetGPU, node, largerPod),
			node: node,
		})
	}
	p.histogram.mu.RUnlock()

	// Ascending cost: steal smallest-impact nodes first, matching Python heapq min-heap.
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].cost < candidates[j].cost })

	// len == 0: nothing to steal; just recurse.
	// len == 1: reference consults TTFT overload detector, which we intentionally skip.
	// len >= 2: main migration path.
	if len(candidates) >= 2 {
		stealN := 0
		now := time.Now()
		for _, c := range candidates {
			if largerCost-c.cost < smallerCost+c.cost {
				break // moving this would invert the balance — stop
			}
			largerCost -= c.cost
			smallerCost += c.cost
			c.node.SetOnlyPodForModel(model, smallerPod, now)
			p.reallocateSubtreeToPod(c.node, model, smallerPod, now)
			stealN++
			if largerCost < rebalancingHighLoadThreshold*smallerCost {
				break
			}
		}
		if stealN > 0 {
			klog.V(5).Infof("preble rebalance: stole %d node(s) from %s to %s (model %s, cost %.2f → %.2f vs %.2f → %.2f)",
				stealN, largerPod, smallerPod, model,
				allocs[largerIdx].cost, largerCost, allocs[smallerIdx].cost, smallerCost)
		}
	}

	allocs[largerIdx].cost = largerCost
	allocs[smallerIdx].cost = smallerCost
	p.rebalanceRecursive(model, targetGPU, allocs[1:])
}

// reallocateSubtreeToPod mirrors reference update_children
// (global_scheduler_with_time.py:459): after migrating an important node to
// a new pod, all descendants are also re-homed to that pod for this model.
// This keeps the prefix-aware branch (which walks up from a future leaf to
// find an ancestor with an allocation) seeing a consistent view.
func (p *prefixCacheAndLoadRouter) reallocateSubtreeToPod(node *prefixcacheindexer.TreeNode, model, podIP string, timestamp time.Time) {
	for _, child := range node.GetChildren() {
		child.SetOnlyPodForModel(model, podIP, timestamp)
		p.reallocateSubtreeToPod(child, model, podIP, timestamp)
	}
}

func (p *prefixCacheAndLoadRouter) evictionLoop() {
	ticker := time.NewTicker(evictionLoopInterval)
	for range ticker.C {
		evictedNodes := p.cache.Evict(time.Now())
		if len(evictedNodes) > 0 {
			p.histogram.removeEvictedNodes(evictedNodes)
		}
		p.histogram.removeOldEntries(time.Now())
	}
}

func (h *SlidingWindowHistogram) getSimplePrefillCost(node *prefixcacheindexer.TreeNode) float64 {
	missRate := 1.0
	if h.promptTokens[node] > 0 {
		missRate = 1.0 - (float64(h.hitTokens[node]) / float64(h.promptTokens[node]))
	}
	// Simplified prefill time calculation - you may want to use a more sophisticated model
	prefillTime := float64(node.NumTokens()) * float64(node.ContextLength()) * 0.001
	return missRate * float64(h.nodeToCount[node]) * prefillTime
}

func (h *SlidingWindowHistogram) getNodeCost(targetGPU string, node *prefixcacheindexer.TreeNode, podIP string) float64 {
	// prefillCost := h.getSimplePrefillCost(node)
	prefillCost := h.getPrefillCost(targetGPU, node)
	// Get median time per token for the pod
	timePerToken := 0.15 // default value
	if times, ok := h.avgTimePerTokenPerPod[podIP]; ok && len(times) > 0 {
		sort.Float64s(times)
		timePerToken = times[len(times)/2] // median
	}
	// Reference get_node_cost (global_scheduler_with_time.py:152):
	//   decode_cost = active_requests * output_len * tpot
	// active_requests = node.ref_counter[gpu] — the live-request count on this
	// pod touching this node's cache. Without it, decode cost is a constant
	// per-node weight that doesn't rank pods against each other; with it, the
	// cost spikes on pods that have accumulated live requests through this
	// prefix, which is the only instantaneous load signal in this function.
	activeRequests := node.GetRefCounterForPod(podIP)
	outputLen := h.decodingSize[node]
	decodeCost := float64(activeRequests) * float64(outputLen) * timePerToken
	return prefillCost + decodeCost
}

func (h *SlidingWindowHistogram) getCurrentAllocationCostPerPod(targetGPU string) map[string]float64 {
	// Must hold the lock while iterating h.histogram and reading the per-node
	// maps (promptTokens/hitTokens/nodeToCount/decodingSize/avgTimePerTokenPerPod)
	// that getNodeCost consults — otherwise concurrent update/remove* writers
	// trigger "concurrent map iteration and map write".
	h.mu.RLock()
	defer h.mu.RUnlock()
	costs := make(map[string]float64)
	for node := range h.histogram {
		for _, modelPods := range node.GetModelToPods() {
			for podIP := range modelPods {
				costs[podIP] += h.getNodeCost(targetGPU, node, podIP)
			}
		}
	}
	return costs
}

// getFilteredAllocationCostPerPodForModel matches reference
// SlidingWindowHistogram.current_allocation_per_gpu_with_atleast_min_load
// (global_scheduler_with_time.py:135), used exclusively by rebalancing to
// ignore histogram entries with tiny node_to_count. Model-scoped because
// rebalancing migrates per-model pod allocations.
func (h *SlidingWindowHistogram) getFilteredAllocationCostPerPodForModel(targetGPU, model string, minLoad int) map[string]float64 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	costs := make(map[string]float64)
	for node := range h.histogram {
		if h.nodeToCount[node] < minLoad {
			continue
		}
		pods := node.GetPodsForModel(model)
		if pods == nil {
			continue
		}
		for podIP := range pods {
			costs[podIP] += h.getNodeCost(targetGPU, node, podIP)
		}
	}
	return costs
}

func (p *prefixCacheAndLoadRouter) updatePodSet(readyPods []*v1.Pod) {
	currentPodSet := make(map[string]bool)
	for _, pod := range readyPods {
		currentPodSet[pod.Status.PodIP] = true
	}
	allNodes := p.cache.GetAllNodes()
	podsChanged := false
	// Update cache structures
	for _, node := range allNodes {
		// 1. Update ModelToPods
		if node.RemovePodsNotInSet(currentPodSet) {
			podsChanged = true
		}
		// 2. Update node's pod-specific data structures
		node.ResetEvictedPods()                  // Reset as pod IDs might change
		node.ResetCachedPods()                   // Reset as pod IDs might change
		node.ResetRefCounter(len(currentPodSet)) // Resize for new pod count
	}

	// Update router and histogram if pods changed
	if podsChanged || len(currentPodSet) != p.numPods {
		if podsChanged {
			klog.V(5).InfoS("Pod set changed")
		}
		if len(currentPodSet) != p.numPods {
			klog.V(5).InfoS("the number of pods updated", "old_count", p.numPods, "new_count", len(currentPodSet))
		}
		// Update router structures
		p.numPods = len(currentPodSet)
		p.podAllocations = make(map[*prefixcacheindexer.TreeNode]map[int]bool)

		// Update histogram structures
		h := p.histogram
		h.mu.Lock()
		defer h.mu.Unlock()

		h.numPods = len(currentPodSet)

		// Clean up pod-specific maps
		for podIP := range h.currentDecodeLengthsPerPod {
			if !currentPodSet[podIP] {
				delete(h.currentDecodeLengthsPerPod, podIP)
				delete(h.avgTimePerTokenPerPod, podIP)
			}
		}

		// Reset pod allocation maps
		h.podAllocations = make(map[*prefixcacheindexer.TreeNode]map[int]bool)

		// No need to clean up these as they're node-based, not pod-based:
		// - histogram
		// - nodeToCount
		// - hitTokens
		// - promptTokens
		// - decodingSize
		// - perNodeTotalDecodeLengths

		// Filter timestamps entries for nodes that still have valid pods
		newTimestamps := make([]histogramEntry, 0)
		for _, entry := range h.timestamps {
			if entry.node == nil {
				continue
			}
			if entry.node.HasValidPods(currentPodSet) {
				newTimestamps = append(newTimestamps, entry)
			}
		}
		h.timestamps = newTimestamps
	}
}

// Compute the load in a pod fo a specific model based on the sliding window histogram
func (h *SlidingWindowHistogram) getPodLoad(pod *v1.Pod) int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	load := 0
	for node, count := range h.nodeToCount {
		for _, podMap := range node.GetModelToPods() {
			if _, exists := podMap[pod.Status.PodIP]; exists {
				load += count
				break // Found this pod in this node, no need to check other models
			}
		}
	}
	return load
}

// Update histogram to use pod name instead of pod ID
func (h *SlidingWindowHistogram) update(timestamp time.Time, node, leafNode *prefixcacheindexer.TreeNode, podIP string, decodingLength int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.timestamps = append(h.timestamps, histogramEntry{
		timestamp: timestamp,
		node:      node,
		leafNode:  leafNode,
	})

	h.histogram[node] += leafNode.ContextLength()
	h.nodeToCount[node]++
	h.decodingSize[node] = decodingLength
	h.hitTokens[node] += leafNode.ContextLength() - leafNode.NumTokens()
	h.promptTokens[node] += leafNode.ContextLength()

	// // Update costs
	// oldCost := h.perNodePrefillCost[node]
	// newCost := h.getPrefillCost(node)
	// h.currentPrefillCostPerPod[podIP] -= oldCost
	// h.currentPrefillCostPerPod[podIP] += newCost
	// h.perNodePrefillCost[node] = newCost

	h.currentDecodeLengthsPerPod[podIP] += decodingLength
	h.perNodeTotalDecodeLengths[node] += decodingLength
}

// Modified Route function that preserves original routing logic but adds metrics logging
func (p *prefixCacheAndLoadRouter) Route(routingCtx *types.RoutingContext, pods types.PodList) (string, error) {
	startTime := time.Now()
	klog.V(5).InfoS("Entering Route", "requestID", routingCtx.RequestID, "routingCtx", routingCtx.Err() != nil)
	defer func() {
		if routingCtx.Err() != nil {
			klog.ErrorS(routingCtx.Err(), "Error in Route", "requestID", routingCtx.RequestID)
		} else {
			klog.V(5).Infof("Exiting Route successfully, requestID: %s", routingCtx.RequestID)
		}
	}()
	readyPods := utils.FilterRoutablePods(pods.All())

	// Populate per-request telemetry that calculateTimingMetrics (in
	// gateway_rsp_body.go) later reads to emit the latency_metrics log line.
	// This mirrors what rl_routing.go does at the top of its Route — but
	// without the synchronous fallback, since Preble doesn't consume these
	// fields for routing decisions. If bgMetricsReady is false (first ~bg
	// scrape interval after process start) the vllm* log fields will be
	// empty for those early requests, and populate automatically thereafter.
	if bgMetricsReady {
		copyBgMetricsToRequest(routingCtx.RequestID)
	}
	detailedpodmetrics := utils.MetricsTracker.GetDetailedMetrics(time.Now().Add(-utils.MetricsTracker.WindowSize))
	utils.AddRequestPodMetrics(routingCtx.RequestID, detailedpodmetrics)

	if len(readyPods) == 0 {
		klog.Errorf("no pods ready to forward request, requestID: %s", routingCtx.RequestID)
		return "", fmt.Errorf("no pods to forward request")
	}
	if len(readyPods) == 1 {
		for _, pod := range readyPods {
			routingCtx.SetTargetPod(pod)
			klog.V(5).Infof("Only one pod is ready. requestID: %s, Route to this pod: %s", routingCtx.RequestID, pod.Status.PodIP)
			return routingCtx.TargetAddress(), nil
		}
	}

	var podUpdateNeeded bool
	func() {
		p.podsMu.RLock()
		defer p.podsMu.RUnlock()
		podUpdateNeeded = len(readyPods) != p.numPods
	}()

	if podUpdateNeeded {
		klog.V(5).Infof("requestID: %s, num pods in data structure: %d", routingCtx.RequestID, p.numPods)
		klog.V(5).Infof("requestID: %s, current actual ready pods: %d", routingCtx.RequestID, len(readyPods))
		p.podsMu.Lock()
		p.updatePodSet(readyPods) // Move update pod logic to separate function
		p.podsMu.Unlock()
		klog.V(5).Infof("requestID: %s, num pods in data structure after updatePodSet: %d", routingCtx.RequestID, p.numPods)
	}

	// Tokens are populated upstream by the gateway as a packed byte array
	// (4 bytes per token, little-endian uint32) — see utils.TokenizeInputTextToByteArray.
	// Decode back to []int for the radix tree's native signature.
	prefill_tokens_bytes := utils.GetByteArrayPrefillTokensForRequest(routingCtx.RequestID)
	prefill_tokens := decodeByteArrayToTokenIDs(prefill_tokens_bytes)

	node, matchedTokens, _ := p.cache.AddPrefix(prefill_tokens, routingCtx.Model, "")
	var matchedPods []*v1.Pod
	var matchedPodsNames []string
	if modelPods, ok := node.GetModelToPods()[routingCtx.Model]; ok {
		readyPodsMap := make(map[string]*v1.Pod)
		for _, pod := range readyPods {
			readyPodsMap[pod.Status.PodIP] = pod
		}
		for podIP := range modelPods {
			if pod, exists := readyPodsMap[podIP]; exists {
				klog.V(5).Infof("requestID: %s, Matched pod: %s", routingCtx.RequestID, podIP)
				matchedPods = append(matchedPods, pod)
				matchedPodsNames = append(matchedPodsNames, podIP)
			}
		}
	}

	var targetPod *v1.Pod
	if prefill_tokens == nil || len(prefill_tokens) == 0 {
		// fallback to random routing
		targetPod = readyPods[rand.Intn(len(readyPods))]
		klog.V(5).Infof("requestID: %s, No prefill tokens found, fallback to random routing. selected pod: %s", routingCtx.RequestID, targetPod.Status.PodIP)
		return routingCtx.TargetAddress(), nil
	}
	matchPercentage := len(matchedTokens) * 100 / len(prefill_tokens)
	klog.V(5).Infof("requestID: %s, Matched tokens/Total tokens: %d/%d, Matching ratio: %d, len(matchedPodsNames): %d, matchedPodsNames: %v", routingCtx.RequestID, len(matchedTokens), len(prefill_tokens), matchPercentage, len(matchedPods), matchedPodsNames)

	if matchPercentage > prefixRoutingThreshold {
		klog.V(5).Infof("requestID: %s, Do prefix-aware routing! (matching ratio: %d > %d)", routingCtx.RequestID, matchPercentage, prefixRoutingThreshold)
		var prefixMatches []prefixMatch

		currentNode := node
		for currentNode != nil {
			if modelPods, ok := currentNode.GetModelToPods()[routingCtx.Model]; ok {
				var nodePods []*v1.Pod
				for podIP := range modelPods {
					for _, pod := range readyPods {
						if pod.Status.PodIP == podIP {
							nodePods = append(nodePods, pod)
						}
					}
				}
				if len(nodePods) > 0 {
					prefixMatches = append(prefixMatches, prefixMatch{
						node:        currentNode,
						pods:        nodePods,
						depth:       currentNode.GetDepth(),
						matchLength: currentNode.ContextLength(),
					})
					klog.V(5).Infof("Found matching pod(s) in node: %v, total match length: %d",
						currentNode.GetID(), currentNode.ContextLength())
				}
			}
			currentNode = currentNode.GetParent()
		}

		sort.Slice(prefixMatches, func(i, j int) bool {
			return prefixMatches[i].matchLength > prefixMatches[j].matchLength
		})

		if len(prefixMatches) > 0 {
			longestMatch := prefixMatches[0]
			minLoad := -1
			for _, pod := range longestMatch.pods {
				load := p.histogram.getPodLoad(pod)
				if minLoad == -1 || load < minLoad {
					minLoad = load
					targetPod = pod
				}
			}
			klog.V(5).Infof("requestID: %s, Selected pod %s from longest matching node with match length %d", routingCtx.RequestID, targetPod.Status.PodIP, longestMatch.matchLength)
		} else {
			tokenInString, err := utils.DetokenizeText(prefill_tokens)
			matchedTokensInString, _ := utils.DetokenizeText(matchedTokens)
			if err != nil {
				klog.Errorf("requestID: %s, DetokenizeTexts failed: %s, tokens: '%v', matchedTokens: '%v', model: %s", routingCtx.RequestID, err, tokenInString, matchedTokensInString, routingCtx.Model)
			} else {
				klog.V(5).Infof("requestID: %s, No matched pods found for tokens: '%v', matchedTokens: '%v', model: %s", routingCtx.RequestID, tokenInString, matchedTokensInString, routingCtx.Model)
				klog.V(5).Infof("requestID: %s, Go to cost model based routing!", routingCtx.RequestID)
			}
		}
	}

	if targetPod == nil {
		klog.V(5).Infof("requestID: %s, Do cost model based routing! (matching ratio: %d%%, len(matchedPods): %d)", routingCtx.RequestID, matchPercentage, len(matchedPods))
		// ts = time.Now()
		podCosts := p.histogram.getCurrentAllocationCostPerPod(routingCtx.TargetGPU)
		minCost := math.MaxFloat64
		for _, pod := range readyPods {
			cost := podCosts[pod.Status.PodIP]
			klog.V(5).Infof("Pod: %s, Cost: %.2f", pod.Status.PodIP, cost)
			if cost < minCost {
				minCost = cost
				targetPod = pod
			}
		}
		if targetPod == nil {
			klog.Errorf("requestID: %s, After all logic, no suitable pod found. readyPods: %v", routingCtx.RequestID, readyPods)
			return "", fmt.Errorf("no suitable pod found")
		}
		klog.V(5).Infof("requestID: %s, Lowest cost pod: %s", routingCtx.RequestID, targetPod.Status.PodIP)
	}

	allPodCacheHitRatio := map[string]int{}
	for _, pod := range readyPods {
		podHitRatio := p.cache.GetCacheHitRatioForTargetPod(prefill_tokens, routingCtx.Model, pod.Status.PodIP)
		allPodCacheHitRatio[pod.Status.PodIP] = podHitRatio
	}
	utils.SetSnapShotForKVCacheHitRatio(routingCtx.RequestID, allPodCacheHitRatio)

	// Walk leaf → root: attach the selected pod to every ancestor's model
	// allocation AND bump that pod's live-request counter on every ancestor.
	// The ref_counter increment matches reference update_allocated_size
	// (global_lru_cache.py:312) — ancestors are what feed getNodeCost's decode
	// term, so the walk needs to cover the whole chain up to root.
	now := time.Now()
	targetPodIP := targetPod.Status.PodIP
	for currentNode := node; currentNode != nil; currentNode = currentNode.GetParent() {
		currentNode.AddOrUpdatePodForModel(routingCtx.Model, targetPodIP, now)
		currentNode.IncrementRefCounterForPod(targetPodIP)
	}
	// Stash the leaf pointer and pod so PrebleOnRequestComplete can undo the
	// increment when this request finishes. Stored after the walk so that any
	// mid-walk crash leaves nothing to decrement; LoadAndDelete semantics make
	// the completion call idempotent across multiple cleanup paths.
	prebleOutstandingRequests.Store(routingCtx.RequestID, &prebleOutstandingRequest{
		leaf:  node,
		podIP: targetPodIP,
	})

	// Histogram is keyed on the deepest "large" ancestor of the leaf (reference
	// global_scheduler_with_time.py:322,339). Without this step every leaf is its
	// own histogram entry and node_to_count stays ≈1, which makes the cost-model
	// branch numerically degenerate — many callers collapsing into one entry is
	// what makes node_to_count a load signal.
	importantNode := getImportantNode(node)
	if importantNode == nil {
		importantNode = node
	}
	p.histogram.update(now, importantNode, node, targetPodIP, defaultDecodingLength)

	// Rebalancing (reference global_scheduler_with_time.py:345). Gated by leaf-
	// to-important chain length — the reference comment notes "Ignore longer
	// chains for Infercept optimizations". The warmup + imbalance gates are
	// enforced inside handleImportantNodeStealing.
	p.totalRequests.Add(1)
	if node.GetDepth()-importantNode.GetDepth() < rebalancingChainLength {
		p.handleImportantNodeStealing(routingCtx, readyPods)
	}

	routingCtx.SetTargetPod(targetPod)
	klog.V(5).Infof("requestID: %s, entire Route overhead: %d ms, Routing complete for request. target pod: %s", routingCtx.RequestID, time.Since(startTime).Milliseconds(), targetPod.Status.PodIP)
	return routingCtx.TargetAddress(), nil
}
