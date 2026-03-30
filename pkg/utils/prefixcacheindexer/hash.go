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

package prefixcacheindexer

import (
	"encoding/binary"
	"math/rand"
	"sync"
	"time"

	"github.com/cespare/xxhash/v2"
	"github.com/vllm-project/aibrix/pkg/utils"
	lrustore "github.com/vllm-project/aibrix/pkg/utils/lrustore"
	"k8s.io/klog/v2"
)

const (
	defaultPrefixCacheBlockNumber           = 200000
	defaultPrefixCacheBlockSize             = 4
	defaultPrefixCacheEvictionInternalInSec = 1 // 1 second
	// defaultPrefixCacheEvictionDurationInSec = 300 // 60 seconds
	defaultPrefixCacheEvictionDurationInMins = 20 // 10 minutes
)

var (
	prefixCacheBlockNumber      = utils.LoadEnvInt("AIBRIX_PREFIX_CACHE_BLOCK_NUMBER", defaultPrefixCacheBlockNumber)
	prefixCacheBlockSize        = utils.LoadEnvInt("AIBRIX_PREFIX_CACHE_BLOCK_SIZE", defaultPrefixCacheBlockSize)
	prefixCacheEvictionInterval = time.Duration(utils.LoadEnvInt("AIBRIX_PREFIX_CACHE_EVICTION_INTERVAL_SECONDS", defaultPrefixCacheEvictionInternalInSec)) * time.Second
	// prefixCacheEvictionDuration = time.Duration(utils.LoadEnvInt("AIBRIX_PREFIX_CACHE_EVICTION_DURATION_SECONDS", defaultPrefixCacheEvictionDurationInSec)) * time.Second
	prefixCacheEvictionDuration = time.Duration(utils.LoadEnvInt("AIBRIX_PREFIX_CACHE_EVICTION_DURATION_MINS", defaultPrefixCacheEvictionDurationInMins)) * time.Minute
)

type PrefixHashTable struct {
	mu    sync.RWMutex
	seed  uint64
	store lrustore.Store[uint64, Block]
}

type Block struct {
	modelToPods map[string]map[string]time.Time // model_name: map[pod_name]pod_last_access_time
}

func NewPrefixHashTable() *PrefixHashTable {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	seed := r.Uint64()
	klog.InfoS("prefix_cache_hash_table_configurations",
		"prefix_cache_block_number", prefixCacheBlockNumber,
		"prefix_cache_block_size", prefixCacheBlockSize,
		"prefix_cache_block_eviction_interval_seconds", prefixCacheEvictionInterval,
		"prefix_cache_block_eviction_duration_seconds", prefixCacheEvictionDuration)
	instance := &PrefixHashTable{
		seed: seed,
		store: lrustore.NewLRUStore[uint64, Block](prefixCacheBlockNumber,
			prefixCacheEvictionDuration,
			prefixCacheEvictionInterval,
			func() time.Time { return time.Now() }),
	}
	return instance
}

// // original version
// MatchPrefix matches the input token prefix's if already cached
// returns map[podname]%prefixmatch along with all prefix hashes
func (c *PrefixHashTable) MatchPrefix(tokens []byte, model string, readyPods map[string]struct{}) (map[string]int, []uint64) {
	prefixHashes := getPrefixHashes(c.seed, tokens)
	return c.seqSearchPrefix(prefixHashes, model, readyPods)
}

// // gangmuk version returning matchedHashes
func (c *PrefixHashTable) MatchPrefix_returning_matchedprefixes(tokens []byte, model string, readyPods map[string]struct{}) (map[string]int, []uint64, []uint64) {
	allPrefixHashes := getPrefixHashes(c.seed, tokens)
	podMatches, matchedPrefixHashes := c.seqSearchPrefix_returning_matchedprefixes(allPrefixHashes, model, readyPods)
	return podMatches, matchedPrefixHashes, allPrefixHashes
}

// Original version
func (c *PrefixHashTable) seqSearchPrefix(prefixHashes []uint64, model string, readyPods map[string]struct{}) (map[string]int, []uint64) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// podname -> %prefixmatch
	prefixMatchPods := map[string]int{}
	for i := 0; i < len(prefixHashes); i++ {
		prefixHash := prefixHashes[i]
		prefixMatchPercent := (i + 1) * 100 / len(prefixHashes)

		block, ok := c.store.Get(prefixHash)
		if !ok || len(block.modelToPods[model]) == 0 ||
			!matchPods(block.modelToPods[model], readyPods, prefixMatchPods, prefixMatchPercent) {
			break
		}
	}
	klog.V(5).Infof("prefixMatchPods: %v, prefixHashes: %v", prefixMatchPods, prefixHashes)
	return prefixMatchPods, prefixHashes
}

// Gangmuk version:
func (c *PrefixHashTable) seqSearchPrefix_returning_matchedprefixes(prefixHashes []uint64, model string, readyPods map[string]struct{}) (map[string]int, []uint64) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	prefixMatchPods := map[string]int{}
	matchedHashes := []uint64{}

	for i := 0; i < len(prefixHashes); i++ {
		prefixHash := prefixHashes[i]
		prefixMatchPercent := (i + 1) * 100 / len(prefixHashes)

		block, ok := c.store.Get(prefixHash)
		if !ok || len(block.modelToPods[model]) == 0 ||
			!matchPods(block.modelToPods[model], readyPods, prefixMatchPods, prefixMatchPercent) {
			break
		}
		matchedHashes = append(matchedHashes, prefixHash)
	}

	return prefixMatchPods, matchedHashes
}

// AddPrefix add prefix hashes for input tokens
func (c *PrefixHashTable) AddPrefix(prefixHashes []uint64, model, pod string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := 0; i < len(prefixHashes); i++ {
		prefixHash := prefixHashes[i]

		block, ok := c.store.Get(prefixHash)
		if !ok {
			block = Block{
				modelToPods: map[string]map[string]time.Time{
					model: {
						pod: time.Now(),
					},
				},
			}
		} else {
			blockPods, ok := block.modelToPods[model]
			if !ok {
				blockPods = map[string]time.Time{}
			}
			blockPods[pod] = time.Now()
			block.modelToPods[model] = blockPods
		}

		c.store.Put(prefixHash, block)
	}
	// klog.InfoS("prefix_cache_add_prefix",
	// 	"prefix_hashes", prefixHashes,
	// 	"model", model,
	// 	"pod", pod,
	// 	"pod_last_access_time", time.Now())
}

// matchPods returns ready pods that intersect with pods on which prefix tokens are catched.
func matchPods(blockPods map[string]time.Time, readyPods map[string]struct{}, prefixMatchPods map[string]int, prefixMatchPercent int) bool {
	var isMatch bool
	for pod := range readyPods {
		if _, ok := blockPods[pod]; ok {
			prefixMatchPods[pod] = prefixMatchPercent
			isMatch = true
		} else {
			delete(readyPods, pod)
		}
	}
	return isMatch
}

// GetPodPrefixLastAccess returns the last-access timestamp (unix millis) of each
// pod's DEEPEST matched prefix block. Read-only — does not modify any shared state.
//
// The deepest block is the most conversation-specific one. Its timestamp reflects
// when THIS pod last handled a request from the same conversation prefix, which
// is the best proxy for whether vLLM still has those KV cache blocks.
//
// The shallow blocks (system prompt) are shared by all conversations and accessed
// frequently — their timestamps are uninformative for routing decisions.
func (c *PrefixHashTable) GetPodPrefixLastAccess(prefixHashes []uint64, model string, readyPods map[string]struct{}) map[string]int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Track the deepest matched block's timestamp per pod.
	// We walk blocks in order (0, 1, 2, ...) and overwrite on each match,
	// so the final value is the deepest block's timestamp.
	podLastAccess := make(map[string]int64)

	for i := 0; i < len(prefixHashes); i++ {
		block, ok := c.store.Get(prefixHashes[i])
		if !ok {
			break
		}
		modelPods, ok := block.modelToPods[model]
		if !ok || len(modelPods) == 0 {
			break
		}

		anyMatch := false
		for pod := range readyPods {
			if lastAccess, exists := modelPods[pod]; exists {
				// Overwrite (not max) — last write is the deepest block
				podLastAccess[pod] = lastAccess.UnixMilli()
				anyMatch = true
			}
		}
		if !anyMatch {
			break
		}
	}

	// Fill non-matching pods with 0
	for pod := range readyPods {
		if _, exists := podLastAccess[pod]; !exists {
			podLastAccess[pod] = 0
		}
	}

	return podLastAccess
}

func getPrefixHashes(seed uint64, tokens []byte) []uint64 {
	numBlocks := len(tokens) / prefixCacheBlockSize
	prefixHashes := make([]uint64, 0, numBlocks)
	digest := xxhash.NewWithSeed(seed)

	// Chain block hashes so each prefix hash depends on all preceding blocks.
	// This preserves true prefix semantics and avoids independent block matches.
	parentHash := seed
	var parentHashBytes [8]byte

	for i := 0; i < len(tokens); i += prefixCacheBlockSize {
		end := i + prefixCacheBlockSize
		if end > len(tokens) {
			// Don't hash incomplete blocks
			break
		}

		digest.ResetWithSeed(seed)

		// Write parent hash first (as 8 bytes), then current block.
		binary.LittleEndian.PutUint64(parentHashBytes[:], parentHash)
		_, _ = digest.Write(parentHashBytes[:])
		_, _ = digest.Write(tokens[i:end])

		currentHash := digest.Sum64()
		prefixHashes = append(prefixHashes, currentHash)
		parentHash = currentHash
	}
	return prefixHashes
}

func (c *PrefixHashTable) GetPrefixHashes(tokens []byte) []uint64 {
	return getPrefixHashes(c.seed, tokens)
}
