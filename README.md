# Lodestar

An online-learning request router for large language model inference clusters.

Lodestar replaces the heuristic routing policies that LLM gateways usually rely on — least-loaded, prefix-aware, round-robin — with a small neural network that **predicts the reward** of routing a request to each candidate instance, and picks the instance with the highest predicted reward. The reward is `−TTFT` (negative time-to-first-token, in seconds), so maximizing predicted reward minimizes expected TTFT. The predictor trains **online** from the cluster's own measurements while it serves requests; no offline labeling, no manual rules, no hand-tuned thresholds.

Online learning is the central design choice, not an afterthought. A reward predictor trained offline does not generalize to its deployment distribution, because the routing policy itself reshapes the load profile it has to predict (the "circular dependency" the paper documents in §3.3). Online retraining closes that loop: each retraining round narrows the gap between the predictor's training data and the distribution its successor will actually face.

Lodestar is built on top of [AIBrix](https://github.com/vllm-project/aibrix). It reuses AIBrix's Envoy `ext_proc` gateway, pod registry, and vLLM metric scraping. On top of those it adds:

- the learned reward predictor (a per-instance MLP) served as a sibling Python service
- an online training loop driven by the gateway's own request traffic, with a recency + diversity training-buffer design
- a set of heuristic baselines (random, least-request, prefix-cache, LMETRIC, MOONCAKE) wired through the same gateway, so research comparisons are apples-to-apples

## Architecture

```
                                 Kubernetes cluster
   ┌─────────────────────────────────────────────────────────────────────┐
   │                                                                     │
   │     +-------------------+                                           │
   │     |  Stateful         |        /infer        +------------------+ │
   │     |  Gateway          | ───── per-request ─► |  Routing         | │
   │     |       +           | ◄──── target pod ─── |  Service         | │
   │     |  Lodestar         |                      |  (Python/Flask)  | │
   │     |  ext_proc plugin  | ===== /flush =====►  |                  | │
   │     |   (Go)            |   batch of finished  |  • Reward        | │
   │     |                   |   request logs       |    predictor MLP | │
   │     |                   |                      |  • Online        | │
   │     +-------+-----------+                      |    training      | │
   │             |                                  +------------------+ │
   │             | forwards request                                      │
   │             ▼                                                       │
   │     +-------------------+   +-------------------+                   │
   │     |  vLLM instance 0  |   |  vLLM instance N  |                   │
   │     |  (GPU)            |   |  (GPU)            |                   │
   │     +-------------------+   +-------------------+                   │
   │                                                                     │
   └─────────────────────────────────────────────────────────────────────┘
                  ▲
                  │  OpenAI-compatible request
                  │
              client
```

Two components:

**Stateful Gateway** — Go, runs in the Envoy data plane as an `ext_proc` filter.
Code lives in `pkg/plugins/gateway/`. The plugin intercepts every chat-completion
request, takes a cluster snapshot (per-instance KV-cache hit ratio against the
prompt prefix, inflight prefill/decode tokens, queue depth, GPU memory
utilization, GPU model), asks the routing service which instance should serve
it, forwards the request, streams the response, and records measured TTFT/TPOT
latencies. As a safety net, the Stateful Gateway pre-computes a heuristic
selection before issuing the RPC, so any timeout or error in the Routing
Service path falls through to that pre-computed choice with no added latency.

**Routing Service** — Python, runs as its own pod.
Code lives in `python/routing_agent_service/`. Holds the reward predictor and
the training loop. Exposes two HTTP endpoints:

| Endpoint  | Caller         | Purpose                                                                                  |
| --------- | -------------- | ---------------------------------------------------------------------------------------- |
| `/infer`  | Gateway, per request | Score every instance for one request, return the choice + diagnostic predictions. |
| `/flush`  | Gateway, periodic    | Receive a batch of completed-request logs, label them with reward = −TTFT, run an online training step. |

### The reward predictor

A small multilayer perceptron — 3 hidden layers × 128 units, ReLU, dropout 0.1, scalar output. Crucially, **the same parameters are shared across every instance**, and instance identity is never an input. This has two consequences:

- The architecture is **instance-count independent** — instances can be added or removed without retraining, which matters for elastic cloud deployments.
- The architecture is **instance-index independent** — the model can't memorize that "pod 3 is fast" and herd traffic onto it; every routing decision is purely from the per-request features.

For each request, the gateway sends `N` (request_features, instance_features) pairs (one per active instance). The Routing Service does a single batched forward pass of shape `[N, d]`, returns the argmax, and the gateway routes there.

When cluster KV-cache utilization is saturated (>80% by default), a consistent-hashing filter narrows the candidate set to `k=2` instances before the argmax. This prevents greedy routing from evicting widely shared prefix-cache state for marginal latency gains — a global vs local trade-off the paper discusses in §4.1.

### Online learning loop

The predictor retrains every θ=1000 newly accumulated samples. Training data lives in a **two-pool buffer**:

- A FIFO buffer of 5000 most-recent samples (recency).
- A replay buffer of 5000 samples promoted via gradient-coreset selection from FIFO evictions — i.e., samples whose last-hidden-layer activations, weighted by prediction residual, are most diverse from those already kept (diversity).

Each round trains on the union F ∪ R. Total storage is bounded.

Three conditions trigger a fallback to the heuristic instead of the model: (1) cold start — no model trained yet; (2) out-of-distribution input — feature ranges outside the training buffer; (3) RPC timeout/error.

## Request flow

For a single chat completion:

1. **Client → Stateful Gateway.** Envoy receives the request and hands it to the Lodestar `ext_proc` plugin.
2. **Cluster snapshot.** The plugin parses the request, computes the per-instance prefix-cache hit ratio against the prompt, and reads the latest vLLM-metrics snapshot from its background scraper (refreshed every 100 ms — no per-request HTTP fan-out).
3. **Routing decision.** The plugin POSTs the feature vector to the Routing Service's `/infer` endpoint. The service runs one forward pass of the reward predictor, returns the argmax instance plus diagnostics (predicted reward per instance, exploration flag, OOD-fallback flag). Meanwhile the gateway has already pre-computed a heuristic backup choice in case the RPC times out.
4. **Forwarding.** The plugin forwards the request to the chosen instance and streams the response back to the client.
5. **Measurement.** The plugin captures TTFT on the first response token, TPOT on every subsequent token, and E2E latency on stream close. On end-of-stream it emits a structured `@latency_metrics@…` log line carrying the full feature vector and the measured latencies.
6. **Training.** A flush goroutine in the gateway batches these log lines (100 per batch by default) and POSTs them to `/flush`. The Routing Service labels each sample with reward = −TTFT in seconds, drops or promotes samples per the two-pool design, and runs an online training step every θ samples. Future `/infer` calls use the updated model.

## Configuration

Both components are configured via environment variables.

- **Stateful Gateway** reads env from the `aibrix-gateway-plugins` pod's deployment spec. Update the manifest or use `kubectl set env deployment/aibrix-gateway-plugins KEY=VALUE`.
- **Routing Service** reads env from `python/routing_agent_service/k8s/routing-agent/routing-agent-service.yaml`. For quick experiments, `kubectl set env deployment/routing-agent-service KEY=VALUE` is the fastest path.

The knobs that materially change behavior:

| Side    | Variable                          | Default              | What it does                                                                                                                  |
| ------- | --------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Gateway | `ENABLE_FLUSH`                    | `0`                  | When `1`, the gateway batches completed-request logs and POSTs them to `/flush`. Must be on for online learning.              |
| Gateway | `FLUSH_PERIOD`                    | `10` (sec)           | How often the flush goroutine wakes up to check whether a batch is ready.                                                     |
| Gateway | `MIN_NUM_LOG_MESSAGES_TO_FLUSH`   | `100`                | Minimum batch size before the gateway actually POSTs. Avoids tiny chatty flushes.                                             |
| Gateway | `BG_SCRAPE_INTERVAL_MS`           | `100`                | Background vLLM-metrics scrape period. Lower = fresher load signal, more `/metrics` traffic.                                  |
| Service | `ROUTING_STRATEGY`                | `latency_predictor`  | Which policy is active. Set to `lodestar` for the data-driven router; or one of the baselines (`random`, `least_request`, `prefix_cache_1`, `lmetric`, `mooncake`, …) for comparison runs. |
| Service | `ENABLE_ONLINE_LEARNING`          | `0`                  | When `1`, the predictor retrains on flushed batches in the background. When `0`, the model is frozen (inference-only).        |
| Service | `MIN_NUM_TRAINING_DATA`           | `5000`               | Samples that must accumulate before the first training round fires.                                                           |
| Service | `MIN_NUM_UPDATE_DATA`             | `500`                | Samples added since the previous round before the next round fires.                                                           |
| Service | `ONLINE_TRAIN_FROM_SCRATCH`       | `1`                  | When `1`, each round starts from random weights using the current window. When `0`, the model continues from the previous round's weights. |
| Service | `FIFO_SIZE`                       | `5000` (paper)       | Size of the recency buffer.                                                                                                   |
| Service | `REPLAY_SIZE`                     | `5000` (paper)       | Size of the gradient-coreset replay buffer.                                                                                   |
| Service | `EXPLORATION_ENABLED`             | `0`                  | When `1`, the policy occasionally picks a non-argmax instance to gather counterfactual data. `EXPLORATION_RATE` controls the rate. |
| Service | `MAX_NUM_ONLINE_TRAINS`           | `-1`                 | Cap on the number of online training rounds. `-1` = unlimited.                                                                |

A typical bring-up: deploy with `ENABLE_FLUSH=0`, `ENABLE_ONLINE_LEARNING=0` to validate the inference path, then turn both on to enable the training loop. Convergence to a useful policy takes roughly five minutes of traffic on the workloads in the paper.

## Headline results

From the paper's evaluation on a public-cloud Kubernetes cluster, comparing Lodestar against `Prefix-cache-and-load-aware` (the state-of-the-art LLM-aware heuristic in AIBrix):

| Setup                                | Mean TTFT improvement | P99 TTFT improvement |
| ------------------------------------ | --------------------- | -------------------- |
| Overall                              | 1.41×                 | 1.47×                |
| Homogeneous (8× NVIDIA-A30)          | 1.02× – 2.15×         | 1.07× – 1.86×        |
| Heterogeneous (A30 + V100)           | 1.25× – 4.38×         | 1.32× – 4.42×        |

Online learning converges in roughly **5 minutes** of traffic.

## Project layout

```
pkg/plugins/gateway/                Go ext_proc plugin (Stateful Gateway)
    gateway_req_headers.go          Phase 1: header extraction
    gateway_req_body.go             Phase 2: body parsing + instance selection
    gateway_rsp_headers.go          Phase 3: response header handling
    gateway_rsp_body.go             Phase 4: streaming + measurement
    algorithms/rl_routing.go        Entry point that talks to the Routing Service
    algorithms/                     Heuristic baselines (prefix-cache, LMETRIC, …)

pkg/utils/                          Per-request feature store + helpers

python/routing_agent_service/       The Python Routing Service
    Dockerfile, start.sh, k8s/      Container build + deployment manifests
    agent_codes/
        routing_agent_service.py    Flask app, /infer + /flush handlers
        reward_predictor.py         Reward MLP (3×128, shared θ across instances)
        preprocess.py               Log-line parser, per-instance feature extraction
        rewards.py                  Reward functions (default: negative TTFT) + dispatcher
        data_normalizer.py          Running statistics + z-score normalization
        encoding.py                 DataFrame → PyTorch tensor encoding
        replay_buffer.py            Gradient-coreset replay selection
        distribution_shift_detector.py  Online OOD detection (used by fallback)
        offline_routing_agent.py    Offline-training driver (cold start)
        tests/                      Regression + integration tests

benchmarks/                         Workload generators, replay tools, evaluation
```

## Quick start

A working deployment needs a Kubernetes cluster with vLLM pods and the AIBrix
gateway components. High-level steps:

```shell
git clone https://github.com/<your-org>/Lodestar.git
cd Lodestar

# 1. AIBrix dependencies (gateway, controller, autoscaler, ...)
kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-dependency-v0.4.1.yaml
kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-core-v0.4.1.yaml

# 2. Lodestar Routing Service
kubectl apply -f python/routing_agent_service/k8s/routing-agent/routing-agent-service.yaml

# 3. Build and install the Lodestar gateway plugin
bash build-gateway.sh
```

A full walkthrough, configuration reference, and reward-function guide live in
[`docs/`](docs/).

## Relation to AIBrix

Lodestar is a research fork of [AIBrix](https://github.com/vllm-project/aibrix)
(Apache 2.0). The gateway plumbing, controllers, and pod registry come from
upstream and remain structurally unchanged; Lodestar's contribution is the
routing brain — the reward predictor and online-training loop in
`pkg/plugins/gateway/algorithms/` and `python/routing_agent_service/`. Upstream
package paths and copyright headers are preserved so changes can flow in either
direction.

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening one. New reward functions, new routing baselines, and workload
traces for evaluation are particularly useful contributions.

## License

Apache 2.0 — see [LICENSE](LICENSE). All upstream AIBrix code retains its
original copyright headers.

## Citation

If you use Lodestar in your research, please cite:

> Gangmuk Lim, Wanyu Zhao, Brighten Godfrey, Jiaxin Shan, Le Xu, and Liguang Xie. *Lodestar: An Online-Learning LLM Inference Router.* arXiv preprint [2606.00946](https://arxiv.org/abs/2606.00946), 2026.

```bibtex
@article{lodestar2026,
  title         = {Lodestar: An Online-Learning LLM Inference Router},
  author        = {Lim, Gangmuk and Zhao, Wanyu and Godfrey, Brighten and Shan, Jiaxin and Xu, Le and Xie, Liguang},
  year          = {2026},
  eprint        = {2606.00946},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2606.00946}
}
```
