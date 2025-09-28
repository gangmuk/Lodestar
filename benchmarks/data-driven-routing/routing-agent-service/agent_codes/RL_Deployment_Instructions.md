# SB3 RL Routing Agent - Deployment Instructions

## Overview

The SB3-based RL routing agent is now **fully integrated** with your existing pipeline. This guide shows how to deploy it end-to-end.

---

## 1. Prerequisites

### ✅ Updated Files
- ✅ `requirements.txt` - Added SB3 dependencies
- ✅ `routing_agent_service.py` - Added RL model type support
- ✅ `offline_routing_agent.py` - Added RL training support
- ✅ `rl_contextual_bandit_sb3.py` - SB3 integration wrapper
- ✅ `rl_routing_agent_sb3.py` - Core SB3 RL implementation

### 📋 Pipeline Components (Unchanged)
- ✅ `preprocess.py` - Works as-is
- ✅ `data_normalizer.py` - Works as-is
- ✅ `encoding.py` - Works as-is
- ✅ `Dockerfile` - Works as-is (uses updated requirements.txt)

---

## 2. Configuration

### Step 1: Create RL Model Config
```bash
# Use the example configuration
cp agent_codes/rl_model_config_example.json final_model/model_config.json

# Or modify existing config to add:
{
    "MODEL_TYPE": "rl_contextual_bandit_sb3",
    "learning_rate": 0.0003,
    "reward_decay_factor": 0.95,
    "use_custom_reward": true,
    "n_steps": 2048,
    "batch_size": 64
}
```

### Step 2: Key Parameters Explained
```json
{
    "MODEL_TYPE": "rl_contextual_bandit_sb3",  // Enables SB3 RL
    "use_custom_reward": true,                  // Uses π(a|s) * reward formulation
    "learning_rate": 0.0003,                   // PPO learning rate
    "reward_decay_factor": 0.95,               // Gamma (discount factor)
    "n_steps": 2048,                           // SB3 rollout steps
    "batch_size": 64,                          // SB3 batch size
    "online_update_threshold": 50              // Trigger online learning
}
```

---

## 3. Training (Offline)

### Step 1: Prepare Training Data
```bash
# Your existing pipeline works unchanged:
cd /path/to/routing-agent-service

# Preprocess data
python agent_codes/preprocess.py --input raw_data/ --output processed_data/

# Normalize features  
python agent_codes/data_normalizer.py --input processed_data/

# Encode for training
python agent_codes/encoding.py --input processed_data/ --output encoded_data/
```

### Step 2: Train RL Model
```bash
# Run offline training (with RL config)
python agent_codes/offline_routing_agent.py \
    --encoded_data_dir encoded_data/ \
    --final_model_dir final_model/ \
    --hyperparameters_file rl_model_config_example.json
```

### Step 3: Training Output
```
final_model/
├── rl_agent_sb3.zip              # SB3 model file
├── rl_agent_sb3_additional.pkl   # Additional state
├── rl_sb3_hyperparameters.pkl    # Hyperparameters
└── model_config.json             # Configuration
```

---

## 4. Deployment

### Step 1: Build Docker Image
```bash
# Your existing build process works unchanged:
./build-and-push.sh

# Or manually:
docker build -t routing-agent-rl:latest .
```

### Step 2: Deploy Container
```bash
# Deploy using your existing deployment scripts
# The container will automatically use RL if MODEL_TYPE is set correctly

kubectl apply -f k8s/
# or your existing deployment method
```

### Step 3: Verify RL is Active
```bash
# Check logs for RL activation
kubectl logs -f deployment/routing-agent

# Look for:
# "Using SB3 RL contextual bandit model for inference"
# "SB3 RL agent created successfully"
```

---

## 5. Runtime Behavior

### Inference Flow
```
Request → routing_agent_service.py 
       → preprocess.py 
       → data_normalizer.py 
       → encoding.py 
       → rl_contextual_bandit_sb3.infer_from_tensor()
       → SB3 PPO model prediction
       → Pod selection
```

### Online Learning Flow
```
Request feedback → store_experience()
                → experience buffer
                → [After 50 experiences]
                → SB3 PPO online update
                → Model improvement
```

---

## 6. Monitoring & Debugging

### Key Metrics to Monitor
```python
# Available via agent.get_metrics():
{
    'total_steps': 1500,           # Total experiences processed
    'buffer_size': 25,             # Current experience buffer size  
    'model_num_timesteps': 2048    # SB3 internal timesteps
}
```

### Debug Logs
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Key log messages:
# "SB3 RL agent created successfully"
# "SB3 RL inference for request {id}"  
# "Online learning update completed: {steps} steps"
# "SB3 RL agent loaded from {path}"
```

### Performance Comparison
```bash
# A/B test vs contextual bandit:
# Set 50% traffic to MODEL_TYPE="contextual_bandit"
# Set 50% traffic to MODEL_TYPE="rl_contextual_bandit_sb3"
# Compare latency and SLO metrics
```

---

## 7. Troubleshooting

### Common Issues

#### 1. SB3 Dependencies Missing
```bash
# Symptom: "No module named 'stable_baselines3'"
# Solution: Check requirements.txt includes SB3

# Fix:
pip install stable-baselines3[extra] gymnasium
```

#### 2. Model Type Not Recognized
```bash
# Symptom: Falls back to contextual bandit
# Solution: Check model_config.json

# Fix:
{
    "MODEL_TYPE": "rl_contextual_bandit_sb3"  // Exact string
}
```

#### 3. Online Learning Not Triggering
```bash
# Symptom: No learning updates in logs
# Solution: Check feedback mechanism

# Fix: Ensure store_experience() is called with:
store_experience(tensor_data, action, reward_info, request_id)
```

#### 4. Memory Issues with SB3
```bash
# Symptom: OOM errors
# Solution: Reduce SB3 parameters

# Fix in config:
{
    "n_steps": 1024,      // Reduce from 2048
    "batch_size": 32      // Reduce from 64
}
```

---

## 8. Migration Path

### From Contextual Bandit to RL

#### Step 1: Parallel Deployment
```bash
# Keep existing contextual bandit running
# Deploy RL version alongside
# Route small percentage to RL initially
```

#### Step 2: Gradual Rollout
```bash
# Week 1: 10% traffic to RL
# Week 2: 25% traffic to RL  
# Week 3: 50% traffic to RL
# Week 4: 100% traffic to RL (if metrics are good)
```

#### Step 3: Rollback Plan
```bash
# If RL performs poorly:
# Change MODEL_TYPE back to "contextual_bandit"
# Redeploy immediately
# No data loss - both models use same preprocessing
```

---

## 9. Advanced Configuration

### Conservative RL (Stable Learning)
```json
{
    "MODEL_TYPE": "rl_contextual_bandit_sb3",
    "learning_rate": 0.0001,        // Lower learning rate
    "reward_decay_factor": 0.98,    // Higher gamma
    "online_update_threshold": 100, // Less frequent updates
    "use_custom_reward": true
}
```

### Aggressive RL (Fast Learning)  
```json
{
    "MODEL_TYPE": "rl_contextual_bandit_sb3",
    "learning_rate": 0.001,         // Higher learning rate
    "reward_decay_factor": 0.9,     // Lower gamma
    "online_update_threshold": 25,  // More frequent updates
    "use_custom_reward": true
}
```

### Standard RL (No Custom Reward)
```json
{
    "MODEL_TYPE": "rl_contextual_bandit_sb3",
    "use_custom_reward": false,     // Standard RL reward
    // ... other parameters
}
```

---

## 10. Performance Expectations

### Compared to Contextual Bandit
- **Initialization**: RL may start slightly worse (needs exploration)
- **Learning**: RL should improve faster with online adaptation
- **Steady State**: RL should perform better with dynamic adaptation
- **Overhead**: ~10-20% more computational overhead

### Metrics to Track
- **Latency percentiles** (p50, p95, p99)
- **SLO compliance rate**
- **Pod load distribution**
- **Learning convergence speed**

---

## Conclusion

The SB3 RL integration is **production-ready** and **fully integrated** with your existing pipeline. 

**Key Benefits**:
- ✅ **Drop-in replacement** for contextual bandit
- ✅ **Online learning** for continuous improvement  
- ✅ **Professional RL infrastructure** (SB3)
- ✅ **Same preprocessing pipeline**
- ✅ **Easy rollback** if needed

**Next Steps**:
1. **Deploy with small traffic percentage**
2. **Monitor metrics vs contextual bandit**
3. **Gradually increase traffic if performance is good**
4. **Tune hyperparameters based on results**