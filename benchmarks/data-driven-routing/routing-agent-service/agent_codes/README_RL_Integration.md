# RL Routing Agent Integration Guide

This document explains how to integrate the new Reinforcement Learning (RL) based routing agent with the existing system.

## Overview

The RL implementation includes:
- **Custom reward formulation**: `reward_t = π(a_t|s_t) * point_reward_t`
- **Online learning**: Agent learns from experience during runtime
- **Temporal credit assignment**: Uses discounted cumulative rewards
- **Mini-batch updates**: Updates policy every N requests or time interval

## Key Files

1. **`rl_routing_agent.py`**: Core RL algorithm implementation
2. **`rl_contextual_bandit.py`**: Integration wrapper for existing service
3. **`routing_agent_service.py`**: Main service (needs modification to use RL)

## Integration Steps

### 1. Modify `routing_agent_service.py`

Replace the contextual bandit import and usage:

```python
# OLD:
import simpler_contextual_bandit

# NEW: 
import rl_contextual_bandit as simpler_contextual_bandit  # Drop-in replacement
```

Or for explicit RL mode:

```python
# At the top of routing_agent_service.py
import rl_contextual_bandit

# In the inference logic (around line 234):
if model_type == 'rl_contextual_bandit':
    logger.info(f"Using RL contextual bandit model for inference (request_id: {request_id})")
    result, infer_from_tensor_overhead_summary = rl_contextual_bandit.infer_from_tensor(
        tensor_data, request_id, MODEL_UPDATED, RL_MODEL_HYPERPARAMETERS, final_model_dir
    )
```

### 2. Add RL Hyperparameters

Add these to your model configuration JSON:

```json
{
    "MODEL_TYPE": "rl_contextual_bandit",
    "learning_rate": 0.0003,
    "reward_decay_factor": 0.95,
    "baseline_decay": 0.95,
    "mini_batch_size": 30,
    "episode_length_minutes": 3,
    "update_frequency_seconds": 60,
    "entropy_coeff": 0.01,
    "hidden_dim": 64
}
```

### 3. Enable Online Learning

To enable online learning, you need to provide feedback to the RL agent. Add a feedback endpoint:

```python
@app.route("/feedback", methods=["POST"])
def handle_feedback():
    """Provide feedback for RL online learning"""
    feedback_data = request.json
    
    # Extract feedback information
    request_id = feedback_data['request_id']
    latency = feedback_data['latency']
    slo_met = feedback_data['slo_met']
    
    # Store in global cache for experience replay
    # Implementation depends on your specific setup
    
    return jsonify({"status": "feedback_received"})
```

### 4. Experience Storage

For proper online learning, store experiences when feedback arrives:

```python
# When feedback is received about a previous request:
tensor_data = get_cached_tensor_data(request_id)  # You need to implement caching
action = get_cached_action(request_id)  # You need to implement caching
reward_info = {'latency': latency, 'slo_met': slo_met}

rl_contextual_bandit.store_experience(tensor_data, action, reward_info, request_id)
```

## RL vs Contextual Bandit Differences

| Aspect | Contextual Bandit | RL Implementation |
|--------|------------------|-------------------|
| **Learning** | Offline batch training | Online learning from experience |
| **Reward** | Direct point rewards | `π(a\|s) * point_reward` with temporal credit |
| **Updates** | Epoch-based | Mini-batch based (every N requests) |
| **Memory** | Stateless | Maintains experience buffer |
| **Exploration** | Epsilon-greedy/UCB | Policy gradient with entropy bonus |

## Monitoring & Debugging

### Check Agent Status

```python
# Get current RL agent metrics
metrics = rl_contextual_bandit.global_rl_agent.get_metrics()
print(f"Agent metrics: {metrics}")
```

### Key Metrics to Monitor

- **baseline**: Moving average of rewards (should increase over time)
- **buffer_size**: Number of experiences in buffer
- **recent_avg_reward**: Recent performance
- **total_steps**: Total experiences processed

### Expected Behavior

1. **Initial Phase**: Random-like behavior, low confidence
2. **Learning Phase**: Baseline increases, better pod selection
3. **Converged Phase**: Stable performance, high confidence for good pods

## Troubleshooting

### Common Issues

1. **Agent not learning**: Check if `store_experience()` is being called
2. **Poor performance**: Adjust `reward_decay_factor` and `learning_rate`
3. **Instability**: Reduce `learning_rate` or increase `mini_batch_size`

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('rl_contextual_bandit').setLevel(logging.DEBUG)
```

## Performance Considerations

- **Memory**: Experience buffer size is limited (default: 1000 experiences)
- **Computation**: Policy updates happen every `update_frequency_seconds`
- **Network**: Same as contextual bandit (no additional network overhead)

## Configuration Examples

### Conservative (Stable Learning)
```json
{
    "learning_rate": 0.0001,
    "reward_decay_factor": 0.98,
    "mini_batch_size": 50,
    "update_frequency_seconds": 120
}
```

### Aggressive (Fast Learning)
```json
{
    "learning_rate": 0.001,
    "reward_decay_factor": 0.9,
    "mini_batch_size": 20,
    "update_frequency_seconds": 30
}
```

## Next Steps

1. **Test in staging**: Deploy with existing contextual bandit as fallback
2. **Monitor metrics**: Track baseline and recent_avg_reward
3. **Tune hyperparameters**: Adjust based on performance
4. **Add advanced features**: Multi-objective rewards, priority replay, etc.

## Support

For questions or issues:
1. Check logs for RL-specific messages
2. Monitor agent metrics via `/metrics` endpoint (if implemented)
3. Compare performance with baseline contextual bandit