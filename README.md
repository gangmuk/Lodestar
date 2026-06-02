# Lodestar

A data-driven request router for large language model inference clusters.

Lodestar replaces the heuristic routing policies that LLM gateways usually rely on — least-loaded, prefix-aware, round-robin — with a learned model that predicts which inference pod will serve each request fastest. The model is a neural contextual bandit. It trains online from the cluster's own measurements: no offline labeling, no human-tuned weights, no static rules.

Lodestar is built on top of [AIBrix](https://github.com/vllm-project/aibrix). It reuses AIBrix's Envoy `ext_proc` gateway, pod registry, and vLLM metric scraping. On top of those it adds:

- a learned routing policy, served as a sibling Python service
- an online training loop driven by the gateway's own request traffic
- a set of heuristic baselines (random, least-request, prefix-cache, LMETRIC, MOONCAKE) so research comparisons are apples-to-apples

## Architecture

```
                                 Kubernetes cluster
   ┌─────────────────────────────────────────────────────────────────────┐
   │                                                                     │
   │     +-------------------+                                           │
   │     |   Envoy gateway   |        /infer        +------------------+ │
   │     |       +           | ───── per-request ─► |  Routing Agent   | │
   │     |  Lodestar         | ◄──── target pod ─── |  Service         | │
   │     |  ext_proc plugin  |                      |  (Python/Flask)  | │
   │     |   (Go)            | ===== /flush =====►  |                  | │
   │     |                   |   batch of finished  |  • CB model      | │
   │     |                   |   request logs       |  • Online train  | │
   │     +-------+-----------+                      +------------------+ │
   │             |                                                       │
   │             | forwards request                                      │
   │             ▼                                                       │
   │     +-------------------+   +-------------------+                   │
   │     |  vLLM pod 0       |   |  vLLM pod N       |                   │
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

**Gateway plugin** — Go, runs in the Envoy data plane as an `ext_proc` filter.
Code lives in `pkg/plugins/gateway/`. The plugin intercepts every chat-completion
request, collects per-pod features (KV-cache hit ratio against the prompt prefix,
inflight prefill/decode counts, the latest vLLM `/metrics` snapshot, GPU model),
asks the routing agent which pod should serve the request, forwards the request,
streams the response, and records measured TTFT/TPOT/E2E latencies.

**Routing Agent Service** — Python, runs as its own pod.
Code lives in `python/routing_agent_service/`. The service holds the routing
model and the training loop, and exposes two HTTP endpoints:

| Endpoint  | Caller         | Purpose                                              |
| --------- | -------------- | ---------------------------------------------------- |
| `/infer`  | Gateway, per request | Score every pod for one request, return the choice. |
| `/flush`  | Gateway, periodic    | Receive a batch of completed-request logs, label them with rewards, run an online training step. |

## Request flow

For a single chat completion:

1. **Client → gateway.** Envoy receives the request and hands it to the Lodestar `ext_proc` plugin.
2. **Feature collection.** The plugin parses the request, computes the per-pod prefix-cache hit ratio against the prompt, and reads the latest vLLM-metrics snapshot from its background scraper (refreshed every 100 ms — no per-request HTTP fan-out).
3. **Routing decision.** The plugin POSTs the assembled feature vector to the agent's `/infer` endpoint. The agent runs a forward pass through the neural contextual bandit, scores every pod, and returns the chosen pod ID plus diagnostics (predicted reward, exploration flag, OOD-fallback flag).
4. **Forwarding.** The plugin forwards the request to the chosen vLLM pod and streams the response back to the client.
5. **Measurement.** The plugin captures TTFT on the first response token, TPOT on every subsequent token, and E2E latency on stream close. On end-of-stream it emits a structured `@latency_metrics@…` log line carrying the full feature vector and the measured latencies.
6. **Training.** A flush goroutine in the gateway batches these log lines and POSTs them to the agent's `/flush` endpoint. The agent computes reward labels from the measured latencies (the reward function is configurable; `negative_linear` is the default), then takes one or more online training steps. Future `/infer` calls use the updated model.

The result is a routing policy that adapts to the cluster it's running on, in real time, without needing the operator to hand-tune weights for each workload.

## Project layout

```
pkg/plugins/gateway/                Go ext_proc plugin
    gateway_req_headers.go          Phase 1: header extraction
    gateway_req_body.go             Phase 2: body parsing + pod selection
    gateway_rsp_headers.go          Phase 3: response header handling
    gateway_rsp_body.go             Phase 4: streaming + measurement
    algorithms/rl_routing.go        Entry point that talks to the agent service
    algorithms/                     Heuristic baselines (prefix-cache, LMETRIC, …)

pkg/utils/                          Per-request feature store + helpers

python/routing_agent_service/
    Dockerfile, start.sh, k8s/      Container build + deployment manifests
    agent_codes/
        routing_agent_service.py    Flask app, /infer + /flush handlers
        reward_predictor.py         Neural contextual bandit (per-pod scorer)
        preprocess.py               Log-line parser, per-pod feature extraction
        rewards.py                  Reward functions + dispatcher
        data_normalizer.py          Running statistics + z-score normalization
        encoding.py                 DataFrame → PyTorch tensor encoding
        replay_buffer.py            Gradient-based experience selection
        distribution_shift_detector.py  Online OOD detection
        offline_routing_agent.py    Offline-training driver (cold start)
        tests/                      Regression + integration tests

benchmarks/                         Workload generators, replay tools, evaluation
```

## Quick start

A working deployment needs a Kubernetes cluster with vLLM pods and the AIBrix
gateway components. The high-level steps:

```shell
git clone https://github.com/<your-org>/Lodestar.git
cd Lodestar

# 1. AIBrix dependencies (gateway, controller, autoscaler, ...)
kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-dependency-v0.4.1.yaml
kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-core-v0.4.1.yaml

# 2. Lodestar routing agent
kubectl apply -f python/routing_agent_service/k8s/routing-agent/routing-agent-service.yaml

# 3. Build and install the Lodestar gateway plugin
bash build-gateway.sh
```

A full walkthrough, configuration reference, and reward-function guide live in
[`docs/`](docs/).

## Configuration

Both components are configured via environment variables.

- **Gateway plugin** reads its env from the gateway pod's spec — the `aibrix-gateway-plugins` deployment shipped by AIBrix. Update the deployment manifest or use `kubectl set env deployment/aibrix-gateway-plugins KEY=VALUE`.
- **Routing Agent Service** reads its env from `python/routing_agent_service/k8s/routing-agent/routing-agent-service.yaml`. For quick experiments, `kubectl set env deployment/routing-agent-service KEY=VALUE` is the fastest path.

The full list of knobs is large; the ones that materially change behaviour:

| Side    | Variable                          | Default              | What it does                                                                                                                  |
| ------- | --------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Gateway | `ENABLE_FLUSH`                    | `0`                  | When `1`, the gateway batches completed-request logs and POSTs them to the agent's `/flush` endpoint. Must be on for online learning. |
| Gateway | `FLUSH_PERIOD`                    | `10` (sec)           | How often the flush goroutine wakes up to check whether a batch is ready.                                                     |
| Gateway | `MIN_NUM_LOG_MESSAGES_TO_FLUSH`   | `100`                | Minimum batch size before the gateway actually POSTs. Avoids tiny chatty flushes.                                             |
| Gateway | `BG_SCRAPE_INTERVAL_MS`           | `100`                | Background vLLM-metrics scrape period. Lower = fresher load signal, more `/metrics` traffic.                                  |
| Agent   | `ROUTING_STRATEGY`                | `latency_predictor`  | Which policy is active. Set to `lodestar` for the data-driven router; or one of the baselines (`random`, `least_request`, `prefix_cache_1`, `lmetric`, `mooncake`, …) for comparison runs. |
| Agent   | `ENABLE_ONLINE_LEARNING`          | `0`                  | When `1`, the agent trains on flushed batches in the background. When `0`, the model is frozen (inference-only).              |
| Agent   | `MIN_NUM_TRAINING_DATA`           | `5000`               | Samples that must accumulate before the first training round fires.                                                           |
| Agent   | `MIN_NUM_UPDATE_DATA`             | `500`                | Samples added since the previous round before the next training round fires.                                                  |
| Agent   | `ONLINE_TRAIN_FROM_SCRATCH`       | `1`                  | When `1`, each round starts from random weights using the current data window. When `0`, the model continues from the previous round's weights. |
| Agent   | `EXPLORATION_ENABLED`             | `0`                  | When `1`, the policy occasionally picks a non-argmax pod to gather counterfactual data. `EXPLORATION_RATE` controls the rate. |
| Agent   | `MAX_NUM_ONLINE_TRAINS`           | `-1`                 | Cap on the number of online training rounds. `-1` = unlimited.                                                                |

Note: the routing decision is per-request, but online learning is opt-in. A
common pattern is to deploy with `ENABLE_FLUSH=0` and `ENABLE_ONLINE_LEARNING=0`
first to validate inference, then turn both on to enable the training loop.

## Relation to AIBrix

Lodestar is a research fork of [AIBrix](https://github.com/vllm-project/aibrix)
(Apache 2.0). The gateway plumbing, controllers, and pod registry come from
upstream and remain structurally unchanged; Lodestar's contribution is the
routing brain — the `rl-online-router` algorithm in
`pkg/plugins/gateway/algorithms/` and the Python agent service in
`python/routing_agent_service/`. Upstream package paths and copyright headers
are preserved so changes can flow in either direction.

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening one. New reward functions, new routing baselines, and workload
traces for evaluation are particularly useful contributions.

## License

Apache 2.0 — see [LICENSE](LICENSE). All upstream AIBrix code retains its
original copyright headers.

## Citation

If you use Lodestar in your research, please cite:

> *Lodestar: A Data-Driven Request Router for LLM Inference Clusters.* arXiv preprint [2606.00946](https://arxiv.org/abs/2606.00946), 2026.

```bibtex
@article{lodestar2026,
  title         = {Lodestar: A Data-Driven Request Router for LLM Inference Clusters},
  author        = {TODO: author list},
  year          = {2026},
  eprint        = {2606.00946},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2606.00946}
}
```
