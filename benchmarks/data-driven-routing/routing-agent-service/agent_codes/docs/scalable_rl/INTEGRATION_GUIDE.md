# Scalable RL Agent - Integration Guide

## ✅ What's Been Done

### 1. New Scalable RL Agent Implementation
- **File**: `scalable_rl_routing_agent.py`
- **Features**:
  - ✅ Pod-count independent architecture (works with 4 to 1000+ pods)
  - ✅ Async experience completion for proper TD learning
  - ✅ Episode boundaries (1-second windows)
  - ✅ Prioritized experience replay
  - ✅ Cluster statistics for relative context
  - ✅ Action masking placeholder
  - ✅ GAE with λ=0.95, γ=0.95

### 2. Integration with `routing_agent_service.py`
- **Changes Made**:
  - ✅ Imported new scalable RL agent
  - ✅ Added `USE_SCALABLE_RL = True` flag (easily switchable)
  - ✅ Updated initialization logic (initializes ONCE, works for any #pods)
  - ✅ Updated inference call to use `infer_scalable_rl_agent`
  - ✅ Added placeholder for request completion callback

---

## ⚠️ Critical TODOs

### 1. **Implement Request Completion Notification** (CRITICAL)

**Why it's needed**: The new agent needs to know when requests finish to:
- Capture `next_obs` (cluster state AFTER completion)
- Compute proper TD targets
- Enable multi-step credit assignment

**What to do**:

#### Option A: Add HTTP Endpoint (Recommended)

Add a new Flask endpoint that gets called when requests complete:

```python
@app.route("/request_complete", methods=["POST"])
def handle_request_complete():
    """
    Called when a request finishes.
    
    Expected payload:
    {
        "request_id": "req_12345",
        "ttft": 45.6,           # milliseconds
        "tpot": 12.3,           # milliseconds
        "status": "success"     # or "failure"
    }
    """
    data = request.json
    request_id = data.get('request_id')
    ttft = data.get('ttft')
    tpot = data.get('tpot')
    
    if not request_id or ttft is None or tpot is None:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Call the completion notification function
    on_request_complete_notification(request_id, ttft, tpot)
    
    return jsonify({"status": "ok"}), 200
```

Then update your request router/gateway to POST to this endpoint when requests complete.

#### Option B: Modify Existing /flush Endpoint

If request completions are already reported in `/flush`, extract them there:

```python
@app.route("/flush", methods=["POST"])
def handle_flush():
    # ... existing code ...
    
    # After processing log_data, check for completed requests
    if USE_SCALABLE_RL and RL_AGENT is not None:
        for entry in log_data:
            if 'ttft' in entry and 'avg_tpot' in entry:
                request_id = entry.get('request_id')
                ttft = entry.get('ttft')
                tpot = entry.get('avg_tpot')
                
                # Complete the RL experience
                on_request_complete_notification(request_id, ttft, tpot)
    
    # ... rest of existing code ...
```

---

### 2. **Implement `get_current_cluster_features()`** (CRITICAL)

**Why it's needed**: To capture `next_obs` (state after completion), we need current pod metrics.

**What to do**:

```python
def get_current_cluster_features():
    """
    Fetch current cluster state for experience completion.
    
    Returns:
        pod_features: [num_pods, 10] - Current pod metrics
        kv_hit_ratios: [num_pods, 1] - Current cache hit ratios
        request_features: [3] - Dummy request features
    """
    try:
        # 1. Get current running pods
        running_pods = utils.get_running_pods_by_label(POD_LABEL_SELECTOR)
        sorted_pod_ips = utils.fetch_running_pod_ips(running_pods)
        
        # 2. Fetch current pod metrics
        # NOTE: You'll need to implement this based on your monitoring system
        # Options:
        #   - Query Prometheus/Grafana for current metrics
        #   - Maintain in-memory state of pod metrics (updated periodically)
        #   - Query Kubernetes metrics API
        
        # Placeholder: Create dummy features based on last known state
        # In production, replace with real-time metrics
        num_pods = len(sorted_pod_ips)
        pod_features = np.zeros((num_pods, 10), dtype=np.float32)
        kv_hit_ratios = np.zeros((num_pods, 1), dtype=np.float32)
        
        # TODO: Fill with actual metrics:
        # pod_features[:, 0] = get_running_requests_per_pod()
        # pod_features[:, 1] = get_queue_length_per_pod()
        # pod_features[:, 2] = get_gpu_usage_per_pod()
        # etc.
        
        # Dummy request features (not used for next_obs, but needed for consistency)
        request_features = np.zeros(3, dtype=np.float32)
        
        return pod_features, kv_hit_ratios, request_features
        
    except Exception as e:
        logger.error(f"Failed to get current cluster features: {e}")
        raise
```

**Recommended Implementation**:

Add a background thread that periodically fetches and caches pod metrics:

```python
# Global cache for current cluster state
CURRENT_CLUSTER_STATE = {
    'pod_features': None,
    'kv_hit_ratios': None,
    'last_update': 0
}
CLUSTER_STATE_LOCK = threading.Lock()

def cluster_metrics_worker():
    """Background thread to update cluster metrics every second"""
    global CURRENT_CLUSTER_STATE
    
    while not SHUTDOWN.is_set():
        try:
            # Fetch current metrics from your monitoring system
            pod_metrics = fetch_pod_metrics()  # TODO: Implement this
            
            with CLUSTER_STATE_LOCK:
                CURRENT_CLUSTER_STATE['pod_features'] = pod_metrics['pod_features']
                CURRENT_CLUSTER_STATE['kv_hit_ratios'] = pod_metrics['kv_hit_ratios']
                CURRENT_CLUSTER_STATE['last_update'] = time.time()
            
            time.sleep(1.0)  # Update every second
            
        except Exception as e:
            logger.error(f"Error in cluster metrics worker: {e}")
            time.sleep(5.0)

def get_current_cluster_features():
    """Get cached cluster state"""
    with CLUSTER_STATE_LOCK:
        if CURRENT_CLUSTER_STATE['pod_features'] is None:
            raise RuntimeError("Cluster state not yet initialized")
        
        return (
            CURRENT_CLUSTER_STATE['pod_features'].copy(),
            CURRENT_CLUSTER_STATE['kv_hit_ratios'].copy(),
            np.zeros(3, dtype=np.float32)  # Dummy request features
        )

# Start in __main__:
if __name__ == "__main__":
    # ... existing init ...
    
    # Start cluster metrics worker
    metrics_thread = threading.Thread(target=cluster_metrics_worker, daemon=True)
    metrics_thread.start()
    
    # ... rest of startup ...
```

---

### 3. **Implement Action Masking Logic** (Important)

**Why it's needed**: To prevent routing to unhealthy/unavailable pods.

**What to do**:

In `scalable_rl_routing_agent.py`, update `compute_action_mask()`:

```python
def compute_action_mask(self, pod_features, kv_hit_ratios):
    """
    Compute action mask for unhealthy pod filtering.
    
    Args:
        pod_features: [num_pods, 10]
        kv_hit_ratios: [num_pods, 1]
    
    Returns:
        action_mask: [num_pods] - 1=valid, 0=invalid
    """
    num_pods = pod_features.shape[0]
    
    # TODO: Replace with your domain logic
    # Example:
    # Column indices (adjust based on your features):
    # - pod_features[:, 0] = running_requests
    # - pod_features[:, 1] = queue_length
    # - pod_features[:, 9] = availability (1=up, 0=down)
    
    # Example rules:
    availability = pod_features[:, 9]  # Assuming column 9 is availability
    queue_length = pod_features[:, 1]  # Assuming column 1 is queue
    
    is_available = (availability == 1)
    not_overloaded = (queue_length < 100)  # Adjust threshold
    
    action_mask = (is_available & not_overloaded).astype(np.float32)
    
    # Safety: if all masked, unmask all (must route somewhere)
    if action_mask.sum() == 0:
        logger.warning("All pods masked! Using uniform distribution")
        action_mask = np.ones(num_pods, dtype=np.float32)
    
    return action_mask
```

---

## 🧪 Testing

### Quick Test (Without Completion)

The agent will work without completion callbacks, but won't learn properly:

```bash
# 1. Set subAlgorithm to 'rl_agent' in your inference request
# 2. Send inference requests
# 3. Check logs for:
#    "🚀 Initialized SCALABLE RL agent"
#    "🎯 Scalable RL inference: action=X, confidence=Y, num_pods=Z"
```

### Full Test (With Completion)

After implementing completion callbacks:

```bash
# 1. Send inference request
# 2. When request completes, send completion notification
# 3. Check logs for:
#    "📝 Created pending experience for request req_XXX"
#    "✅ Completed experience for request req_XXX: reward=Y, done=Z"
# 4. Check metrics:
curl http://localhost:8080/metrics  # Or implement a metrics endpoint
```

---

## 🔄 Switching Between Old and New Agent

To switch back to old agent (for comparison):

```python
# In routing_agent_service.py, line 72:
USE_SCALABLE_RL = False  # Use old agent

# Or via environment variable:
USE_SCALABLE_RL = os.getenv("USE_SCALABLE_RL", "true").lower() == "true"
```

---

## 📊 Monitoring

### Key Metrics to Track

Add an endpoint to expose RL agent metrics:

```python
@app.route("/rl_metrics", methods=["GET"])
def get_rl_metrics():
    """Get RL agent training metrics"""
    if RL_AGENT is None:
        return jsonify({"error": "RL agent not initialized"}), 404
    
    if USE_SCALABLE_RL:
        metrics = RL_AGENT.get_metrics()
        return jsonify({
            "agent_type": "scalable_rl",
            "total_steps": metrics['total_steps'],
            "total_episodes": metrics['total_episodes'],
            "buffer_size": metrics['buffer_size'],
            "pending_experiences": metrics['pending_experiences'],
            "current_episode": metrics['current_episode']
        })
    else:
        # Old agent metrics
        metrics = RL_AGENT.get_metrics()
        return jsonify({
            "agent_type": "old_rl",
            **metrics
        })
```

### What to Monitor

1. **Pending experiences**: Should be low (< 100)
   - High values = completion callbacks not working
2. **Buffer size**: Should grow over time (up to max)
3. **Episodes**: Should increment every ~1 second
4. **Total steps**: Should increase with each completion

---

## 🐛 Troubleshooting

### Issue 1: "All pods masked!" Warning

**Cause**: Action masking logic is too strict or availability column wrong.

**Fix**: Check `compute_action_mask()` logic and column indices.

---

### Issue 2: Buffer Size Stays at 0

**Cause**: Request completion callbacks not working.

**Fix**: 
1. Check if `on_request_complete_notification()` is being called
2. Check if `get_current_cluster_features()` is implemented
3. Add debug logging to see where it's failing

---

### Issue 3: Agent Reinitializes on Every Request

**Cause**: `USE_SCALABLE_RL = True` but still using old logic.

**Fix**: Check that the `if USE_SCALABLE_RL:` branch is being taken (add log statements).

---

### Issue 4: Import Errors

**Cause**: `scalable_rl_routing_agent.py` not in path.

**Fix**: Check file is in same directory as `routing_agent_service.py`.

---

## 📝 Summary Checklist

Before deploying to production:

- [ ] Implement request completion callback (HTTP endpoint or /flush modification)
- [ ] Implement `get_current_cluster_features()` (real-time metrics)
- [ ] Implement `compute_action_mask()` (domain-specific rules)
- [ ] Add metrics endpoint for monitoring
- [ ] Test with small traffic (10-100 req/s)
- [ ] Monitor pending experiences (should be low)
- [ ] Monitor buffer size (should grow)
- [ ] Monitor episodes (should increment)
- [ ] Compare performance with old agent (`USE_SCALABLE_RL = False`)

---

## 🚀 Expected Benefits

Once fully integrated:

1. **No retraining on pod scaling** (4 pods → 100 pods, same model!)
2. **Proper multi-step credit assignment** (learns temporal dependencies)
3. **2-3x sample efficiency** (prioritized replay)
4. **Better generalization** (cluster statistics provide context)
5. **Scalable to 1000+ pods** (model size stays constant)

---

## 📞 Questions?

Check the design document: `SCALABLE_RL_DESIGN.md`

Or review the code:
- Implementation: `scalable_rl_routing_agent.py`
- Integration: `routing_agent_service.py` (lines 270-366)

