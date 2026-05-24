package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/vllm-project/aibrix/pkg/utils/prefixcacheindexer"
	"github.com/vllm-project/aibrix/pkg/utils/tokenizer"
	"k8s.io/klog/v2"
)

type workloadLine struct {
	Timestamp int               `json:"timestamp"`
	Requests  []json.RawMessage `json:"requests"`
}

type requestPayload struct {
	Prompt interface{} `json:"prompt"`
}

func promptToString(p interface{}) (string, error) {
	switch v := p.(type) {
	case string:
		return v, nil
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
}

func maxMatchedRatio(m map[string]int) int {
	maxV := 0
	for _, v := range m {
		if v > maxV {
			maxV = v
		}
	}
	return maxV
}

func main() {
	// Suppress noisy klog INFO output from prefix indexer internals.
	klog.SetOutput(io.Discard)

	workloadPath := flag.String("workload", "", "Path to workload.jsonl")
	outCSV := flag.String("out_csv", "", "Output CSV path")
	model := flag.String("model", "replay-model", "Model key for prefix indexer")
	flag.Parse()

	if *workloadPath == "" {
		fmt.Fprintln(os.Stderr, "missing --workload")
		os.Exit(1)
	}
	if *outCSV == "" {
		base := filepath.Dir(*workloadPath)
		*outCSV = filepath.Join(base, "prefix_hit_ratio_replay_updated_singlepod.csv")
	}

	inFile, err := os.Open(*workloadPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed opening workload: %v\n", err)
		os.Exit(1)
	}
	defer inFile.Close()

	if err := os.MkdirAll(filepath.Dir(*outCSV), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "failed creating output directory: %v\n", err)
		os.Exit(1)
	}
	outFile, err := os.Create(*outCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed creating csv: %v\n", err)
		os.Exit(1)
	}
	defer outFile.Close()

	w := csv.NewWriter(outFile)
	if err := w.Write([]string{
		"request_idx",
		"timestamp_ms",
		"selected_pod_hit_ratio_before_add",
		"best_pod_hit_ratio_before_add",
		"num_blocks",
		"prompt_bytes",
	}); err != nil {
		fmt.Fprintf(os.Stderr, "failed writing csv header: %v\n", err)
		os.Exit(1)
	}

	scanner := bufio.NewScanner(inFile)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024*64)

	tok := tokenizer.NewCharacterTokenizer()
	cache := prefixcacheindexer.NewPrefixHashTable()
	selectedPod := "pod-0"
	readyPods := map[string]struct{}{selectedPod: {}}

	reqIdx := 0
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var wl workloadLine
		if err := json.Unmarshal(line, &wl); err != nil {
			fmt.Fprintf(os.Stderr, "invalid workload line: %v\n", err)
			os.Exit(1)
		}

		for _, rawReq := range wl.Requests {
			var req requestPayload
			if err := json.Unmarshal(rawReq, &req); err != nil {
				fmt.Fprintf(os.Stderr, "invalid request payload: %v\n", err)
				os.Exit(1)
			}

			prompt, err := promptToString(req.Prompt)
			if err != nil {
				fmt.Fprintf(os.Stderr, "failed converting prompt: %v\n", err)
				os.Exit(1)
			}

			tokens, err := tok.TokenizeInputText(prompt)
			if err != nil {
				fmt.Fprintf(os.Stderr, "failed tokenizing prompt: %v\n", err)
				os.Exit(1)
			}

			matched, prefixHashes := cache.MatchPrefix(tokens, *model, readyPods)
			selectedRatio := matched[selectedPod]
			bestRatio := maxMatchedRatio(matched)
			numBlocks := len(prefixHashes)

			if err := w.Write([]string{
				strconv.Itoa(reqIdx),
				strconv.Itoa(wl.Timestamp),
				strconv.Itoa(selectedRatio),
				strconv.Itoa(bestRatio),
				strconv.Itoa(numBlocks),
				strconv.Itoa(len(tokens)),
			}); err != nil {
				fmt.Fprintf(os.Stderr, "failed writing csv row: %v\n", err)
				os.Exit(1)
			}

			cache.AddPrefix(prefixHashes, *model, selectedPod)
			reqIdx++
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "scan error: %v\n", err)
		os.Exit(1)
	}
	w.Flush()
	if err := w.Error(); err != nil {
		fmt.Fprintf(os.Stderr, "csv flush error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Wrote %d rows to %s\n", reqIdx, *outCSV)
}
