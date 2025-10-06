# Reinforcement Learning for LLM Inference Request Routing: A Technical Analysis

**A comprehensive study of credit assignment, temporal dependencies, and implementation challenges in production routing systems**

---
## Abstract

We analyze the application of Reinforcement Learning (RL) to the problem of routing LLM inference requests across a heterogeneous GPU cluster. Through detailed examination of sequential decision-making dynamics, we identify critical challenges in credit assignment when routing decisions exhibit both direct and indirect temporal dependencies. We compare learning paradigms from contextual bandits to full RL approaches, uncovering fundamental issues with naive multi-step return computation that can lead to catastrophically incorrect learning. Our analysis reveals three primary failure modes: attribution dilution, self-interference (the multi-agent problem), and confounding from request characteristics. We conclude with a phased implementation strategy that prioritizes robustness over theoretical optimality.

---

## 1. Problem Formulation

### 1.1 System Architecture

**Infrastructure:**
- A cluster of GPU pods, where each pod \( p_i \) is a GPU server capable of processing LLM inference requests
- Pods may have heterogeneous hardware (different GPU types, memory capacities)
- Dynamic scaling: number of pods \( N \) varies with load (cluster autoscaling)

**Request Characteristics:**
- Each request \( r_t \) arrives at time \( t \) with features:
  - `input_tokens`: Input sequence length
  - `output_tokens`: Requested output length (known for some workloads)
  - `total_tokens`: Sum of input and output tokens
- Requests are LLM inference workloads (e.g., text generation, completion)
- Processing time is dominated by:
  - **Time to First Token (TTFT)**: Time until first output token
  - **Time Per Output Token (TPOT)**: Average generation latency per token

**Pod State Features:**
- For each pod \( p_i \):
  - `running_requests`: Number of concurrent requests being processed
  - `queue_size`: Number of queued requests awaiting processing
  - `gpu_type`: Hardware identifier (e.g., A100, H100)
  - `memory_utilization`: GPU memory usage
  - `kv_cache_hit_ratio`: Probability of KV cache hit for similar requests

**Routing Objective:**
Minimize per-request latency while satisfying Service Level Objectives (SLOs):
- TTFT SLO: \( T_{ttft} \) (e.g., 100ms)
- Average TPOT SLO: \( T_{tpot} \) (e.g., 10ms)

### 1.2 Why Reinforcement Learning?

Traditional load-balancing approaches (round-robin, least-loaded, hash-based) fail to capture:

1. **KV Cache Affinity:** Routing similar requests to the same pod increases cache hit rates
2. **Request-Pod Matching:** Different GPU types perform differently for various request sizes
3. **Adaptive Learning:** Optimal routing strategies change with workload patterns and cluster composition
4. **Delayed Feedback:** Performance metrics (TTFT, TPOT) are only observed after request completion

These characteristics suggest a sequential decision-making problem where an agent learns optimal routing policies from experience.

---

## 2. MDP Formulation for Routing

Before discussing learning algorithms, we establish the theoretical foundation of routing as a Markov Decision Process.

### 2.1 State, Action, Reward Definition

See Section 1.1 for full details. In summary:
- **State:** Pod features (load, GPU type) + KV hit ratios + Request features (tokens)
- **Action:** Select pod ∈ {1, ..., N}
- **Reward:** Function of TTFT, TPOT vs. SLOs

### 2.1 Markov Property

**Definition:** A state is Markov if:
```
P(s_{t+1}, r_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0)
  = P(s_{t+1}, r_{t+1} | s_t, a_t)
```

**Intuition:** The current state contains all information needed to predict the future. History beyond the current state is irrelevant.

### 2.3 State Transitions & Temporal Dependencies

**Is Cluster State Markov?** Is Cluster State Markov?

**Our State Representation:**
```python
state_t = {
    'pod_features': {
        'running_requests': [...],  # Per pod
        'queue_size': [...],
        'gpu_type': [...],
        'memory_util': [...]
    },
    'kv_hit_ratios': [...],
    'request_features': {
        'input_tokens': scalar,
        'output_tokens': scalar,
        'total_tokens': scalar
    }
}
```

**Markov Analysis:**

**What State Captures:**
- ✓ Current cluster load (running_requests, queue_size)
- ✓ Hardware configuration (gpu_type)
- ✓ Cache state proxy (kv_hit_ratios)
- ✓ Incoming request characteristics (input/output tokens)

**What State Misses:**
- ✗ Completion time distribution of running requests (some about to finish, some just started)
- ✗ Historical routing patterns (cache warming effects)
- ✗ Workload trends (peak hours vs. off-peak)
- ✗ Specific request content (semantic similarity for cache hits)

**Is This "Markov Enough"?**

**Argument YES:**
- Running requests is a sufficient statistic for near-future load
- KV hit ratios approximate historical patterns
- Request features dominate latency more than cluster history
- Unmeasured factors average out over many samples

**Argument NO:**
- Completion time distribution matters (5 requests finishing in 10ms vs. 5 requests with 200ms left)
- Subtle cache patterns not captured by aggregate hit ratios
- Non-stationarity: time-of-day effects, seasonal patterns

**Practical Implication:** Our state is **approximately Markov** but not perfectly Markov. This means:
- Multi-step returns will have some irreducible noise
- Value function V(s) will have bounded accuracy
- Short-horizon planning (n=3-5) preferred over long-horizon

### 2.4 Why Routing Has Sequential Structure (see also Section 3.5 for challenges) The Asymmetry in Multi-Step Returns

**User's Question:** "If Request-1 affects Request-2, why does return_1 include reward_2 but return_2 doesn't include reward_1?"

**Answer:** The state acts as an **information bottleneck**.

```
Decision-1 at t=0: Route Request-1 to Pod-A
  ↓
State_2 at t=5: Pod-A shows as loaded (running_requests=3)
  ↓
Decision-2: Agent sees loaded Pod-A, chooses Pod-B
  ↓
Reward_2: -120ms

Return computation:
  return_1 = reward_1 + γ·reward_2
  return_2 = reward_2 + γ·reward_3  ← Does NOT include reward_1
```

**Why this is correct:**

1. **State_2 already captures Decision-1's effect**
   - Pod-A's load (running_requests=3) is the consequence of Decision-1
   - When computing Q(state_2, action_2), the state already includes Decision-1's impact
   - Including reward_1 would be **double-counting**

2. **Markov property in action**
   - Future (reward_2, state_3, ...) depends only on (state_2, action_2)
   - NOT on (state_1, action_1, reward_1)
   - The effect of Decision-1 flows through state_2, not directly through reward_1

3. **Counterfactual reasoning**
   ```
   If we included reward_1 in return_2:
   
   Scenario A: Request-1 slow (reward_1 = -500ms)
     State_2: Pod-A loaded
     Decision_2: Pod-B, reward_2 = -120ms
     return_2 = -120 + (-500) = -620
   
   Scenario B: Request-1 fast (reward_1 = -50ms)
     State_2: SAME (Pod-A loaded)  ← Same state!
     Decision_2: Pod-B, reward_2 = -120ms
     return_2 = -120 + (-50) = -170
   
   Problem: Same (state_2, action_2) pair has different returns
            depending on past history → Breaks Markov property!
   ```

**The correct formulation:**
- **return_1 includes reward_2**: Captures "Decision-1 affected future through state transitions"
- **return_2 excludes reward_1**: Evaluates "Decision-2 given state_2" (which already reflects Decision-1)


---

## 3. Reinforcement Learning Approaches

Having established the MDP structure, we now survey learning algorithms and their suitability for routing.

### 3.1 Contextual Bandits

**Formulation:**
Each routing decision is treated as an independent multi-armed bandit problem with context.

**Key Assumption:** Decisions are **independent**—the choice at time t does not affect future states or rewards.

**Advantages:**
- ✓ Simple, well-understood algorithms
- ✓ Fast convergence (no temporal credit assignment)
- ✓ Robust to non-stationarity
- ✓ Low variance in learning signal

**Limitations:**
- ✗ Ignores sequential dependencies (Section 2.3)
- ✗ Cannot model how current routing affects future cluster state
- ✗ Misses multi-step strategic planning opportunities

**When Appropriate:**
- Temporal correlation < 0.2
- Cluster state changes rapidly independent of routing
- Requests short-lived relative to cluster dynamics

### 3.2 Markov Decision Process Framework

**Extension of Bandits:**
Routing as a sequential decision problem where actions affect future states.

**Key Difference:** The Markov property (Section 2.2) enables planning over future consequences.

**Advantages:**
- ✓ Models temporal dependencies (Section 2.3)
- ✓ Enables strategic planning
- ✓ Can learn complex policies

**Challenges:**
- ✗ Complex credit assignment (which past actions caused current reward?)
- ✗ High variance
- ✗ Requires more data

**Critical Question:** How do we compute returns to assign credit? → Section 3.3


### 3.3 Return Computation & Credit Assignment

**Only relevant for multi-step methods** (contextual bandits skip this entirely).

#### 3.3.1 The Credit Assignment Problem

**Central Question:** When we observe reward \( r_t \), which past actions deserve credit (or blame)?

**Example Timeline:**
```
t=0:   Request-1 arrives, routed to Pod-A
t=5:   Request-2 arrives, routed to Pod-B
t=10:  Request-3 arrives, routed to Pod-C
t=50:  Request-1 completes → reward_1 = -50ms
t=120: Request-2 completes → reward_2 = -120ms
t=200: Request-3 completes → reward_3 = -200ms
```

**Question:** How much of `reward_2` should we attribute to `Decision_1`?

#### 3.3.2 Return Computation Methods

##### 3.3.2.1 Monte Carlo (Full Episode Return)

**Definition:**
```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^T·r_T
```

**Intuition:** Sum all future rewards, discounted by temporal distance.

**For routing:**
```
return_1 = reward_1 + γ·reward_2 + γ²·reward_3 + ... + γ^n·reward_n
```
where \( n \) is the last request completed before Request-1 finishes or buffer limit.

**Advantages:**
- ✓ Unbiased: Captures actual cumulative outcome
- ✓ No bootstrapping needed (no value function required)

**Critical Problems in Routing:**

**Problem 1: Attribution Dilution**
```
Request-1: 100 tokens, routes to Pod-A, completes in 50ms
Request-2: 10,000 tokens, routes to Pod-B, completes in 500ms (slow due to size!)
Request-3: 50,000 tokens, routes to Pod-C, completes in 2000ms (very slow due to size!)

Full episode return:
return_1 = -50 + 0.95·(-500) + 0.95²·(-2000)
         = -50 - 475 - 1805
         = -2330  ← Disaster!

Agent learns: "Routing to Pod-A in state_1 is terrible (-2330)"

Actual truth: "Routing to Pod-A in state_1 was good (-50ms)"
```

**Root Cause:** Requests 2 and 3 were slow because of their **intrinsic characteristics** (large size), not because of Decision-1. This is **spurious correlation**.

**Problem 2: Causal Confounding**
```
What determines reward_2?
  - Decision_1 (5% impact if different pod)
  - Decision_2 (30% impact - pod choice)
  - Request_2 size (60% impact - intrinsic slowness)
  - Random factors (5% - network jitter, cache state)

Signal-to-noise ratio: 5% / 95% ≈ 0.05  ← Terrible!
```

Including `reward_2` in `return_1` attributes 100% of its value to Decision-1, but only 5% is causally related.

**Problem 3: Multi-Agent Problem (Self-Interference)**
```
Timeline:
  t=0:  Decision-1: Route Request-1 to Pod-A
        → Pod-A loads from 2 → 3 running requests
  
  t=5:  Decision-2: Route Request-2
        Observes: Pod-A loaded (3 requests), Pod-B idle (2 requests)
        Chooses: Pod-B (because Pod-A is loaded)
        Result: reward_2 = -120ms
  
  t=10: Decision-3: Route Request-3
        Observes: Pod-A still loaded, Pod-C idle
        Chooses: Pod-C
        Result: reward_3 = -200ms (slow because Request-3 is huge)
```

**The Problem:** 
- Decision-1 affected Decision-2's state (Pod-A loaded)
- Decision-2 affected Decision-3's state (Pod-B loaded)
- We're treating a **single agent's sequential decisions** as if they're independent

**This creates circular credit assignment:**
```
return_1 includes reward_2, which depends on Decision-2, which depends on state_2,
which was created by Decision-1.

Attribution chain:
  Decision-1 → state_2 → Decision-2 → reward_2 → return_1 → learn about Decision-1

But Decision-2 was made by the SAME agent (not an external actor).
```

This is fundamentally different from traditional multi-agent RL where other agents are external. Here, **the agent interferes with itself** through sequential decisions.

##### 3.3.2.2 Temporal Difference (TD) Learning

**Definition:**
```
G_t^TD = r_t + γ·V(s_{t+1})
```

**Intuition:** Immediate reward plus estimated future value (bootstrap from critic).

**TD Error:**
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
```

**Advantages:**
- ✓ Low variance (only one-step actual reward)
- ✓ Can learn online (don't need full episode)
- ✓ Critic V(s) can learn to factor out confounding variables

**How TD Helps with Confounding:**

**Example:**
```
Request-1: 100 tokens → reward_1 = -50ms
Request-2: 10,000 tokens arriving next

State_2 = {Pod-A: loaded, request_features: [10000 tokens, ...]}

If critic is well-trained:
  V(state_2) ≈ -500  ← "A 10K token request typically takes 500ms"

TD return:
  return_1^TD = -50 + 0.95·(-500) = -525

Compare to Monte Carlo:
  return_1^MC = -50 + 0.95·(-500) + 0.95²·(-2000) = -2330

The critic V(state_2) has learned:
  "When I see a state with a 10K token request, expect ~-500"

So the TD target doesn't penalize Decision-1 for Request-2's intrinsic slowness
as much, because V(state_2) already accounts for the large request.
```

**Requirements:**
- ✗ Needs `next_obs` (next state observation)
- ✗ Requires well-trained value function V(s)
- ✗ Biased estimate (depends on critic quality)

**Critical Implementation Issue:**
Current implementation stores:
```python
experience = {
    'obs': obs,
    'action': action,
    'reward': reward,
    'timestamp': timestamp
    # ❌ Missing: 'next_obs', 'done'
}
```

**Without `next_obs`**, TD learning cannot compute \( V(s_{t+1}) \), forcing fallback to Monte Carlo returns → suffers from all the confounding problems!

##### 3.3.2.3 n-Step Returns

**Definition:**
```
G_t^(n) = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1} + γ^n·V(s_{t+n})
```

**Intuition:** Balance between MC (high variance, unbiased) and TD (low variance, biased).

**Bias-Variance Tradeoff:**

| n-Step | Bias | Variance | Credit Assignment |
|--------|------|----------|-------------------|
| n=1 (TD) | High (if V(s) is poor) | Low | Local, immediate |
| n=3 | Medium | Medium | Good balance |
| n=10 | Low | High | Diluted signal |
| n=∞ (MC) | Zero | Extreme | Noisy, confounded |

**Empirical Guidance:** \( n = 3-5 \) works well for most domains.

**For Request Routing:**

**Temporal Overlap Analysis:**
```
Request-1: t=0 → t=100ms (Pod-A)
Request-2: t=30 → t=130ms (Pod-B, 70ms overlap with Request-1)
Request-3: t=80 → t=180ms (Pod-C, 20ms overlap with Request-1)

Causal strength:
  Request-1 → Request-2: Strong (70ms overlap, state_2 heavily affected)
  Request-1 → Request-3: Weak (20ms overlap, state_3 barely affected)

Standard 3-step:
  return_1 = reward_1 + 0.95·reward_2 + 0.90·reward_3
  Weights: 100%, 95%, 90%  ← Uniform decay

Overlap-weighted (better):
  return_1 = reward_1 + 0.70·reward_2 + 0.20·reward_3
  Weights: 100%, 70%, 20%  ← Proportional to overlap
```

**Problem:** Standard n-step uses **time-based** discount, but routing has **variable overlap**. Requests arriving 50ms apart might have 0% overlap (short requests) or 90% overlap (long requests).

##### 3.3.2.4 Generalized Advantage Estimation (GAE)

**Definition:**
```
A_t^GAE = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...

where δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
```

**Intuition:** Exponentially-weighted average of n-step advantages.

**Hyperparameter λ:**
- λ=0: Pure TD (1-step, high bias, low variance)
- λ=1: Pure MC (full episode, zero bias, high variance)
- λ=0.95: Common choice (balance)

**Why GAE for Routing:**
```
Advantage_t = "How much better was action_t compared to average?"

With GAE:
  - TD errors δ_t capture immediate surprise
  - Exponential weighting (γλ) emphasizes recent effects
  - Critic V(s) learns to predict based on request size AND cluster state
  - Advantages automatically normalize for confounding factors

Example:
  State: {Pod-A loaded, 10K token request}
  V(state) ≈ -500  ← Critic learned: "10K tokens → slow"
  
  Action: Route to Pod-B (good choice!)
  Actual reward: -400ms (better than expected!)
  
  Advantage = -400 - (-500) = +100  ← Positive! Good action!

Contrast without critic:
  Reward: -400ms  ← Looks bad in isolation
  But it's actually GOOD given the large request size
```

**Requirements:**
1. ✓ Request features in state (we have this)
2. ✗ `next_obs` for TD errors (we don't have this)
3. ✗ Well-trained critic V(s) (uncertain in online setting)
4. ✗ Sufficient diverse data (1000-sample buffer may be too small)

---


### 3.4 Policy Gradient Methods

Now we examine algorithms that **use** the returns computed above to learn policies.

#### 3.4.1 REINFORCE (Vanilla Policy Gradient)

**Gradient Estimator:**
```
∇J(θ) = E_τ[Σ_t ∇log π_θ(a_t|s_t) · G_t]

where G_t is the return (MC, TD, n-step, or GAE from Section 3.3)
```

**Intuition:** Increase probability of actions that led to high returns.

**Problem:** High variance - small changes in policy can drastically change returns.

#### 3.4.2 Actor-Critic Architecture

**Components:**

1. **Actor (Policy):** \( \pi_\theta(a|s) \) — probability distribution over actions given state
2. **Critic (Value Function):** \( V_\phi(s) \) — expected cumulative reward from state \( s \)

**How They Interact:**
```
Actor: "In state s, I think action a is good (probability π(a|s))"
Critic: "State s typically leads to value V(s)"
Learning: Actor adjusts based on advantage A(s,a) = Q(s,a) - V(s)
```

**Advantage Estimation:**
The advantage \( A(s,a) \) measures "how much better is action \( a \) compared to average":
```
A(s,a) = R(s,a) - V(s)
where:
  R(s,a) = actual return from taking action a in state s
  V(s) = expected return from state s under current policy
```

**Why This Helps:**
- Reduces variance: Subtracting baseline \( V(s) \) removes state-dependent factors
- Focuses learning on action quality, not absolute reward magnitude

**Challenge:** Requires good value function \( V(s) \)—if critic is poorly trained, advantages are noisy.



#### 3.4.3 Proximal Policy Optimization (PPO)

**PPO is an advanced Actor-Critic method** that stabilizes policy updates through clipped objective.

**Key Innovation:** Stabilizes policy updates through clipped objective.

**Loss Function:**
```
L_PPO = E_t[
  min(
    r_t(θ) · A_t,
    clip(r_t(θ), 1-ε, 1+ε) · A_t
  )
  + c_1 · L_value - c_2 · H(π_θ)
]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  [probability ratio]
  A_t = advantage estimate
  L_value = (V_θ(s_t) - V_target)²  [critic loss]
  H(π_θ) = entropy bonus for exploration
```

**Clipping Mechanism:**
Prevents destructively large policy updates by limiting how much the new policy can deviate from the old policy.

**Advantages for Routing:**
- ✓ Stable online learning (critical for production systems)
- ✓ Sample efficient (reuses data through multiple epochs)
- ✓ Robust to hyperparameter choices
- ✓ Industry-proven (used in robotics, game AI, etc.)

**Implementation Considerations:**
- Requires rollout buffer with complete transitions: \( (s_t, a_t, r_t, s_{t+1}, \text{done}) \)
- Needs Generalized Advantage Estimation (GAE) for low-variance advantages
- Critic must be well-trained for stable learning



**Relationship to Other Methods:**
```
Contextual Bandit → MDP Framework → Return Computation (MC/TD/GAE)
                                          ↓
                      Policy Gradient (REINFORCE) → Actor-Critic → PPO
                                                                      ↑
                                                        Most advanced for routing
```


### 3.5 Challenges Specific to Request Routing

Having established learning methods, we now identify domain-specific failure modes.

#### 3.5.1 Missing next_obs

**Current Code (rl_routing_agent_sb3.py, line 358-364):**
```python
experience = {
    'obs': obs,
    'action': action,
    'reward': custom_reward,
    'point_reward': point_reward,
    'timestamp': time.time()
    # ❌ Missing: 'next_obs', 'done'
}
```

**Impact:**
Without `next_obs`, PPO cannot compute:
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
              ↑
        Requires next_obs!
```

**Consequence:** PPO falls back to Monte Carlo returns, suffering from:
- Attribution dilution
- Confounding from request characteristics
- Multi-agent problem

**Fix Required:**
```python
def remember_experience(self, ...):
    # Store current experience as "pending"
    if hasattr(self, '_pending_exp') and self._pending_exp is not None:
        # Complete previous experience with current obs as next_obs
        self._pending_exp['next_obs'] = current_obs
        self._pending_exp['done'] = False
        self.experience_buffer.append(self._pending_exp)
    
    # Save current as pending (will be completed by next request)
    self._pending_exp = {'obs': current_obs, 'action': action, 'reward': reward}
```

#### 3.5.2 Confounding from Request Characteristics

**The Problem:**
```
Request-1: 100 tokens, routes to Pod-A
  reward_1 = -50ms (fast because small)

Request-2: 10,000 tokens, routes to Pod-B
  reward_2 = -500ms (slow because LARGE, not because of Decision-1!)

Naive multi-step return:
  return_1 = -50 + 0.95·(-500) = -525

Agent learns: "Routing to Pod-A in state_1 is bad (-525)"
Correct lesson: "Routing to Pod-A in state_1 is good (-50)"
```

**Why This Happens:**
Request size determines 60-80% of latency variance, but multi-step returns attribute all future rewards to past actions.

**Potential Solutions:**

**Option A: Single-Step Returns (Simplest)**
```python
return_t = reward_t  # Ignore future
```
- ✓ No confounding
- ✗ Ignores temporal dependencies

**Option B: Reward Normalization**
```python
normalized_reward = actual_reward - expected_reward_for_size(input_tokens)
return_t = normalized_reward_t + γ·normalized_reward_{t+1} + ...
```
- ✓ Removes size confound
- ✗ Requires good baseline model

**Option C: Trust the Critic (GAE)**
```python
V(state with 10K tokens) ≈ -500  ← Critic learns this
Advantage = actual_reward - V(state)
```
- ✓ Critic factors out expected slowness
- ✗ Requires well-trained critic (hard with 1000 samples, non-stationarity)

#### 3.5.3 Pod-Conditional Credit Assignment

**Observation:** Causal link is stronger when consecutive requests go to the same pod.

**Proposed Solution:**
```python
def compute_returns_pod_conditional(experiences, n=3):
    for i, exp in enumerate(experiences):
        ret = exp['reward']
        current_pod = exp['action']
        
        for j in range(1, min(n+1, len(experiences)-i)):
            next_exp = experiences[i+j]
            next_pod = next_exp['action']
            
            if next_pod == current_pod:
                # Strong causal link: same pod affected
                ret += (gamma ** j) * next_exp['reward']
            else:
                # Weak causal link: different pod, only cluster-level effect
                # Could include with reduced weight, or stop here
                break
        
        returns.append(ret)
```

**Rationale:**
- Same pod: Direct resource contention (queue, GPU memory, cache)
- Different pod: Only indirect effect through cluster-state rebalancing (weaker)

---


#### 3.5.4 Non-Stationarity & Value Function Instability

**Definition:** Environment dynamics change over time, making learned value functions stale.

**Sources in Routing:**

1. **Infrastructure Changes:** Sources of Non-Stationarity

**Infrastructure Changes:**
- Pod autoscaling: Number of pods \( N \) changes dynamically
- Hardware updates: GPU types, memory configurations evolve
- Network conditions: Inter-pod communication latency varies

**Workload Distribution Shift:**
- Time-of-day effects: Peak vs. off-peak traffic
- Request size distribution: Batch jobs vs. interactive queries
- Seasonal patterns: Holiday traffic, special events

**Policy-Induced Non-Stationarity:**
- As the agent learns, its routing policy changes
- This changes the cluster state distribution the agent observes
- Classic RL problem: "chasing a moving target"



**Impact on Multi-Step Methods:** Impact on Learning

**Value Function Instability:**
```
Week 1: Most requests are small (100-500 tokens)
  V({Pod-A loaded, 300 tokens}) ≈ -60ms
  Agent learns this mapping

Week 2: Workload shifts to large requests (5K-10K tokens)
  V({Pod-A loaded, 7K tokens}) should be ≈ -600ms
  But critic was trained on old distribution!
  
Result: Stale value estimates → Poor advantages → Bad policy updates
```

**Policy Lag:**
- Agent learns optimal policy for current cluster size (e.g., 4 pods)
- Cluster scales to 8 pods
- Policy is now suboptimal (doesn't leverage new capacity)



**Mitigation Strategies:** Mitigation Strategies

**Online Learning:**
- Continuous policy updates from recent data
- Current implementation uses this (experience buffer + update_online)

**Recency Weighting:**
- Prioritize recent experiences in buffer
- Current: FIFO buffer (deque with maxlen=1000)
- Better: Prioritized replay with recency factor

**Adaptive Learning Rate:**
- Increase learning rate during distribution shift detection
- Decrease during stability

**State Augmentation:**
- Add time-of-day features
- Add workload statistics (e.g., recent average request size)

---



**Why This Favors Simple Methods:**
- GAE requires stable, well-trained V(s) → fragile under non-stationarity
- Single-step (γ ≈ 0) relies less on V(s) → more robust
- Contextual bandits ignore future entirely → most robust


---

## 4. Deep Dive: Routing-Specific MDP Dynamics

Having covered general RL methods, we now examine subtle properties unique to request routing.

### 4.1 Cross-Pod Influence and Partial Causality

**Subtle Point:** Even when requests go to different pods, Decision-1 can affect Decision-2.

```
t=0: Request-1 arrives
  State_0: {Pod-A: 2 running, Pod-B: 2 running}  ← Equal load
  Decision-1: Route to Pod-A
  
t=5: Request-2 arrives
  State_2: {Pod-A: 3 running, Pod-B: 2 running}  ← Imbalanced!
  Decision-2: Sees Pod-A loaded → Chooses Pod-B
  
  Counterfactual: If Decision-1 had chosen Pod-B:
    State_2': {Pod-A: 2 running, Pod-B: 3 running}
    Decision-2': Would choose Pod-A instead!
```

**Causal Effect:** Decision-1 influenced Decision-2 through **cluster state rebalancing**, even though Request-2 went to a different pod.

**How much influence?**
- Strong: If Request-1 is long-running (overlaps with Request-2 decision)
- Weak: If Request-1 completes before Request-2 arrives
- Medium: If partial overlap

**This motivates overlap-weighted returns:**
```python
overlap_ratio = overlap_time(Request-1, Request-2) / duration(Request-1)
weight = γ * overlap_ratio

return_1 = reward_1 + weight * reward_2
```



### 4.2 The Asymmetry in Multi-Step Returns

**User's Question:** "If Request-1 affects Request-2, why does return_1 include reward_2 but return_2 doesn't include reward_1?"

**Answer:** The state acts as an **information bottleneck**.

```
Decision-1 at t=0: Route Request-1 to Pod-A
  ↓
State_2 at t=5: Pod-A shows as loaded (running_requests=3)
  ↓
Decision-2: Agent sees loaded Pod-A, chooses Pod-B
  ↓
Reward_2: -120ms

Return computation:
  return_1 = reward_1 + γ·reward_2
  return_2 = reward_2 + γ·reward_3  ← Does NOT include reward_1
```

**Why this is correct:**

1. **State_2 already captures Decision-1's effect**
   - Pod-A's load (running_requests=3) is the consequence of Decision-1
   - When computing Q(state_2, action_2), the state already includes Decision-1's impact
   - Including reward_1 would be **double-counting**

2. **Markov property in action**
   - Future (reward_2, state_3, ...) depends only on (state_2, action_2)
   - NOT on (state_1, action_1, reward_1)
   - The effect of Decision-1 flows through state_2, not directly through reward_1

3. **Counterfactual reasoning**
   ```
   If we included reward_1 in return_2:
   
   Scenario A: Request-1 slow (reward_1 = -500ms)
     State_2: Pod-A loaded
     Decision_2: Pod-B, reward_2 = -120ms
     return_2 = -120 + (-500) = -620
   
   Scenario B: Request-1 fast (reward_1 = -50ms)
     State_2: SAME (Pod-A loaded)  ← Same state!
     Decision_2: Pod-B, reward_2 = -120ms
     return_2 = -120 + (-50) = -170
   
   Problem: Same (state_2, action_2) pair has different returns
            depending on past history → Breaks Markov property!
   ```

**The correct formulation:**
- **return_1 includes reward_2**: Captures "Decision-1 affected future through state transitions"
- **return_2 excludes reward_1**: Evaluates "Decision-2 given state_2" (which already reflects Decision-1)



### 4.3 Partial Causality & Overlap-Weighted Returns

**Observation:** Causal strength varies with temporal overlap.

**Standard Discount Factor:**
```
return_1 = r_1 + γ·r_2 + γ²·r_3 + ...
```
Assumes uniform temporal steps, but requests have variable durations.

**Overlap-Weighted Alternative:**
```python
overlap_ratio = overlap_time(Request-1, Request-2) / duration(Request-1)
weight = γ * overlap_ratio

return_1 = reward_1 + weight * reward_2
```

**Example:**
```
Request-1: t=0 → t=100ms (Pod-A)
Request-2: t=30 → t=130ms (Pod-B, 70ms overlap)
Request-3: t=80 → t=180ms (Pod-C, 20ms overlap)

Standard (γ=0.95):
  return_1 = r_1 + 0.95·r_2 + 0.90·r_3

Overlap-weighted:
  return_1 = r_1 + 0.665·r_2 + 0.18·r_3
  (weights proportional to overlap)
```

**Advantage:** Better captures actual causal influence.

---

## 5. Comparison of Approaches

### 5.1 Summary Table

| Approach | Temporal Deps | Credit Assignment | Variance | Bias | Implementation Complexity | Robustness |
|----------|---------------|-------------------|----------|------|---------------------------|------------|
| **Contextual Bandit** | None | N/A (independent) | Low | High (ignores future) | Simple | High |
| **1-Step TD** | Weak | Clean, immediate | Low | Medium (depends on V) | Simple | High |
| **3-Step Returns** | Medium | Diluted | Medium | Medium | Medium | Medium |
| **GAE (proper)** | Strong | Principled | Medium | Low (if V is good) | Complex | Low (depends on V) |
| **Monte Carlo** | Full | Severely diluted | Extreme | Zero | Simple | Very Low |

### 5.2 When to Use Each

**Contextual Bandit:**
- Temporal correlation < 0.2
- Cluster state changes rapidly (external factors dominate)
- Need fast convergence and robustness

**1-Step TD:**
- Temporal correlation 0.2-0.4
- Cannot tolerate confounding from request characteristics
- Value function training is difficult (limited data, non-stationarity)

**3-Step Returns (with pod-conditional weighting):**
- Temporal correlation > 0.4
- Same-pod routing frequency > 30%
- Can invest in careful implementation

**GAE (Full RL with proper next_obs):**
- Temporal correlation > 0.6
- Large experience buffer (5000+ samples)
- Stable environment (or well-handled non-stationarity)
- Can invest significant engineering effort

---


---

## 6. Implementation Strategy

### 6.1 Critical Implementation Issues

Before implementing any phase, we must address existing bugs.

#### 6.1.1 Missing next_obs

(See content from old Section 5.1)

#### 6.1.2 Confounding Mitigation Strategies

(See content from old Section 5.2)

#### 6.1.3 Pod-Conditional Credit Assignment

(See content from old Section 5.3)

### 6.2 Phase 1: Single-Step Baseline (Week 1-2)

**Objective:** Establish robust baseline that avoids confounding.

**Implementation:**
```python
# In hyperparameters:
RL_MODEL_HYPERPARAMETERS = {
    'gamma': 0.0,  # or 0.1 for very weak temporal dependency
    'n_steps': 64,
    'batch_size': 64,
    ...
}

# This makes returns purely immediate:
return_t ≈ reward_t
```

**Metrics to Track:**
- P50, P95, P99 latency
- SLO violation rate
- Policy entropy (exploration level)
- Reward distribution statistics

**Success Criteria:**
- Performance ≥ contextual bandit baseline
- Stable learning (no catastrophic forgetting)
- Low variance in latency metrics

### 6.3 Phase 2: Temporal Correlation Analysis (Week 3)

**Objective:** Measure whether multi-step returns would help.

**Metrics to Compute:**
```python
def analyze_temporal_correlation(buffer):
    metrics = {
        'reward_autocorrelation': [],
        'same_pod_routing_rate': 0.0,
        'average_time_gap': 0.0,
        'overlap_distribution': []
    }
    
    for i in range(len(buffer) - 1):
        exp_t = buffer[i]
        exp_t1 = buffer[i+1]
        
        # Reward correlation
        corr = correlation(exp_t['reward'], exp_t1['reward'])
        metrics['reward_autocorrelation'].append(corr)
        
        # Same-pod routing
        if exp_t['action'] == exp_t1['action']:
            metrics['same_pod_routing_rate'] += 1
        
        # Time gap
        gap = exp_t1['timestamp'] - exp_t['timestamp']
        metrics['average_time_gap'] += gap
        
        # Overlap (if request durations available)
        overlap = compute_overlap(exp_t, exp_t1)
        metrics['overlap_distribution'].append(overlap)
    
    return metrics
```

**Decision Rule:**
```
If avg(reward_autocorrelation) > 0.4 AND same_pod_routing_rate > 25%:
    → Proceed to Phase 3 (multi-step might help)
Else:
    → Stay with single-step (temporal dependency is weak)
```

### 6.4 Phase 3: Multi-Step with next_obs (Week 4-6)

**Only proceed if Phase 2 shows strong temporal correlation.**

**Implementation Changes:**

**1. Add next_obs to Experience Buffer:**
```python
class RLRoutingAgentSB3:
    def __init__(self, ...):
        self._pending_experience = None  # Store incomplete experience
    
    def remember_experience(self, pod_features, kv_hit_ratios, request_features,
                          action, point_reward, lock=None):
        current_obs = self.env._flatten_state(pod_features, kv_hit_ratios, request_features)
        
        # Complete previous experience with current obs as next_obs
        if self._pending_experience is not None:
            self._pending_experience['next_obs'] = current_obs
            self._pending_experience['done'] = False
            
            if lock is not None:
                with lock.write():
                    self.experience_buffer.append(self._pending_experience)
            else:
                self.experience_buffer.append(self._pending_experience)
        
        # Store current as pending (will be completed next call)
        self._pending_experience = {
            'obs': current_obs,
            'action': action,
            'reward': custom_reward,
            # next_obs will be filled by next remember_experience call
        }
```

**2. Use SB3's RolloutBuffer:**
```python
from stable_baselines3.common.buffers import RolloutBuffer

self.rollout_buffer = RolloutBuffer(
    buffer_size=5000,  # Increased from 1000
    observation_space=self.env.observation_space,
    action_space=self.env.action_space,
    device=device,
    gae_lambda=0.95,
    gamma=0.95,
    n_envs=1
)
```

**3. Increase Buffer Size:**
```
1000 samples → 5000 samples
Rationale: Critic needs more diverse data to learn V(s) well
```

**4. Monitor Critic Training:**
```python
def update_online(self, n_steps):
    # ... PPO update ...
    
    # Log critic loss
    critic_loss = compute_value_loss()
    logger.info(f"Critic loss: {critic_loss:.4f}")
    
    if critic_loss > THRESHOLD:
        logger.warning("Critic not converging - GAE may be unreliable")
```

**5. A/B Test:**
```
Split traffic 50/50:
  - Group A: Single-step (gamma=0.1)
  - Group B: Multi-step GAE (gamma=0.95, gae_lambda=0.95)

Compare:
  - P99 latency
  - SLO violation rate
  - Learning stability
  
Run for 1 week, then choose winner
```

### 6.5 Alternative: Pod-Conditional Returns (Week 4-5)

**If you want multi-step but can't invest in next_obs infrastructure:**

```python
def compute_returns_pod_conditional(experiences, n=3, gamma=0.95):
    """
    Only accumulate future rewards if:
    1. Same pod (strong causal link), OR
    2. High temporal overlap (cross-pod influence)
    """
    returns = []
    
    for i, exp in enumerate(experiences):
        ret = exp['reward']
        current_pod = exp['action']
        current_time = exp['timestamp']
        
        for j in range(1, min(n+1, len(experiences)-i)):
            next_exp = experiences[i+j]
            next_pod = next_exp['action']
            next_time = next_exp['timestamp']
            
            time_gap = next_time - current_time
            
            # Discount based on time and causality
            if next_pod == current_pod:
                # Same pod: strong direct effect
                weight = gamma ** j
            elif time_gap < 0.1:  # 100ms threshold
                # Different pod but close in time: cluster-level effect
                weight = (gamma ** j) * 0.5  # Reduced weight
            else:
                # Different pod, distant in time: negligible effect
                break
            
            ret += weight * next_exp['reward']
        
        returns.append(ret)
    
    return returns
```

**Advantages:**
- ✓ Captures same-pod direct effects
- ✓ Weights cross-pod effects by temporal proximity
- ✓ Works with current implementation (no next_obs needed)

**Disadvantages:**
- ✗ Heuristic-based (not theoretically principled)
- ✗ Still suffers from some confounding
- ✗ Hyperparameters (time threshold, cross-pod weight) need tuning

---


---

## 7. Open Research Questions

### 7.1 Counterfactual Credit Assignment

**Challenge:** Distinguish between causal effect and correlation.

**Ideal formulation:**
```
Causal return_1 = E[reward_2 | do(action_1=Pod-A)] 
                - E[reward_2 | do(action_1=Pod-B)]
```

This requires estimating counterfactual outcomes ("what would have happened if...?"), which standard RL doesn't address.

**Potential Approaches:**
- Causal inference methods (propensity scores, instrumental variables)
- Model-based RL (learn transition model, simulate counterfactuals)
- Multi-objective optimization (separate immediate vs. future impact)

### 7.2 Time-Aware Discount Factors

**Current:** \( \gamma^n \) assumes uniform time steps.

**Better:** \( \gamma^{\Delta t} \) where \( \Delta t \) is actual elapsed time.

```python
# Instead of:
return_t = r_t + γ¹·r_{t+1} + γ²·r_{t+2}

# Use:
return_t = r_t + γ^(Δt₁)·r_{t+1} + γ^(Δt₁+Δt₂)·r_{t+2}

where Δt_i = actual milliseconds between decisions
```

**Advantage:** Properly weights rapid consecutive decisions vs. sparse decisions.

### 7.3 Hierarchical Credit Assignment

**Observation:** Different timescales of credit assignment:
- Immediate: Did this pod choice minimize latency for this request?
- Short-term: Did this choice improve cache hit rates for next 10 requests?
- Long-term: Did this choice prevent load imbalance over next minute?

**Hierarchical RL:** Separate policies for different timescales.

### 7.4 Multi-Agent Formulation

**Reframe:** Treat each routing decision as a separate agent, all sharing a policy.

**This makes the multi-agent problem explicit:**
- Agent_1 (request t=0): Routes to Pod-A
- Agent_2 (request t=5): Routes given Agent_1's effect on cluster

**Cooperative MARL methods** (e.g., QMIX, MADDPG) might handle self-interference better.

---


---

## 8. Conclusions

### 8.1 Key Findings

1. **Request routing is a sequential decision problem** with temporal dependencies, not a contextual bandit. Current routing decisions affect future cluster state and thus future decisions.

2. **Naive multi-step returns are catastrophically wrong** due to:
   - **Attribution dilution:** Long-horizon returns accumulate noise from unrelated events
   - **Confounding:** Request characteristics (size) determine 60-80% of latency, but multi-step returns attribute this to past routing decisions
   - **Self-interference:** The same agent makes sequential decisions, creating circular credit assignment

3. **The current implementation lacks next_obs**, forcing PPO to fall back to Monte Carlo returns that suffer from all the above problems.

4. **Single-step returns are surprisingly strong** for this domain because:
   - They avoid confounding entirely
   - Temporal dependencies may be weaker than initially assumed (need Phase 2 analysis)
   - Cross-pod influence is indirect and noisy

5. **GAE can theoretically solve confounding** if the critic \( V(s) \) learns to predict based on both cluster state AND request characteristics, but this requires:
   - Implementing next_obs (non-trivial engineering)
   - Large diverse dataset (5000+ samples)
   - Stable environment or sophisticated non-stationarity handling

### 8.2 Recommended Approach

**Phase 1 (Immediate):** Deploy single-step TD (\( \gamma = 0-0.1 \))
- Robust, avoids confounding, works with current codebase
- Expected to perform well given weak temporal correlations

**Phase 2 (Week 3):** Empirically measure temporal correlation
- If correlation < 0.3 → Stay with single-step
- If correlation > 0.4 → Consider multi-step

**Phase 3 (If needed):** Implement GAE with next_obs
- Only if Phase 2 shows strong temporal dependency
- A/B test against single-step baseline
- Revert if no significant improvement

**Alternative (If limited engineering resources):** Pod-conditional returns
- Pragmatic middle ground
- Captures same-pod effects, downweights cross-pod noise

### 8.3 Broader Implications

This analysis highlights a general challenge in applying RL to production systems with:
- High-dimensional confounding variables
- Self-interfering sequential decisions
- Non-stationary dynamics
- Limited data for critic training

**The lesson:** Theoretically optimal solutions (GAE, full RL) may be practically inferior to simpler approaches (single-step, contextual bandits) when implementation constraints and environment characteristics make sophisticated methods fragile.

**The recommendation:** Start simple, measure empirically, increase complexity only when justified by data.


---

## References

*(Omitted for brevity, but would include citations to:)*
- Sutton & Barto: Reinforcement Learning: An Introduction
- Schulman et al.: Proximal Policy Optimization
- Schulman et al.: High-Dimensional Continuous Control Using Generalized Advantage Estimation
- Mnih et al.: Asynchronous Methods for Deep Reinforcement Learning
- Pearl: Causality (for counterfactual reasoning)
- Domain-specific: LLM serving systems, KV cache optimization, etc.

---

## Appendix: Code Snippets

### A.1 Single-Step Implementation (Phase 1)
```python
# In hyperparameters config:
RL_MODEL_HYPERPARAMETERS = {
    'gamma': 0.1,  # Nearly single-step
    'n_steps': 64,
    'batch_size': 64,
    'gae_lambda': 0.95,  # Unused when gamma ≈ 0
    # ... other params
}
```

### A.2 Temporal Correlation Analysis (Phase 2)
```python
import numpy as np
from scipy.stats import pearsonr

def analyze_temporal_structure(experience_buffer):
    rewards = [exp['reward'] for exp in experience_buffer]
    actions = [exp['action'] for exp in experience_buffer]
    timestamps = [exp['timestamp'] for exp in experience_buffer]
    
    # Reward autocorrelation
    if len(rewards) > 1:
        autocorr_1 = pearsonr(rewards[:-1], rewards[1:])[0]
    else:
        autocorr_1 = 0.0
    
    # Same-pod routing rate
    same_pod_count = sum(1 for i in range(len(actions)-1) 
                        if actions[i] == actions[i+1])
    same_pod_rate = same_pod_count / max(1, len(actions) - 1)
    
    # Average time gap
    time_gaps = [timestamps[i+1] - timestamps[i] 
                for i in range(len(timestamps)-1)]
    avg_gap = np.mean(time_gaps) if time_gaps else 0.0
    
    return {
        'reward_autocorr_lag1': autocorr_1,
        'same_pod_rate': same_pod_rate,
        'avg_time_gap_ms': avg_gap * 1000,
        'recommendation': 'multi-step' if autocorr_1 > 0.4 else 'single-step'
    }
```

### A.3 next_obs Implementation (Phase 3)
```python
def remember_experience(self, pod_features, kv_hit_ratios, request_features,
                      action: int, point_reward: float, lock=None):
    current_obs = self.env._flatten_state(pod_features, kv_hit_ratios, request_features)
    
    # Calculate custom reward
    if self.custom_reward_callback is not None:
        obs_tensor = obs_as_tensor(current_obs, self.model.device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            action_prob = distribution.distribution.probs[0, action].item()
        custom_reward = action_prob * point_reward
    else:
        custom_reward = point_reward
    
    # Complete previous experience
    if hasattr(self, '_pending_experience') and self._pending_experience is not None:
        self._pending_experience['next_obs'] = current_obs
        self._pending_experience['done'] = False
        
        if lock is not None:
            with lock.write():
                self.experience_buffer.append(self._pending_experience)
                self.total_steps += 1
        else:
            self.experience_buffer.append(self._pending_experience)
            self.total_steps += 1
    
    # Store current as pending
    self._pending_experience = {
        'obs': current_obs,
        'action': action,
        'reward': custom_reward,
        'point_reward': point_reward,
        'timestamp': time.time()
        # next_obs will be filled by next call
    }
```

---


---

**Document Version:** 2.0  
**Last Updated:** 2025-10-01  
**Status:** Technical Analysis Complete, Reorganized Structure (Option A), Ready for Implementation

**Key Changes from v1.0:**
- Section 2 (MDP Foundations) moved before Section 3 (Learning Methods)
- Credit Assignment integrated into Section 3.3 (not standalone section)
- PPO shown as advanced Actor-Critic under Section 3.4
- All routing challenges grouped in Section 3.5
- Improved pedagogical flow: Problem → MDP Theory → Learning → Advanced Details → Implementation


---

other things to remember and revisit
- priortized experience buffer. currently experience buffer samples uniformly
- reward normalization (OpenGym already does it. batch normalization.)
- significance of feature normalization since features are not in the same range unlike tokens in language or pixels in image. and the normalization will affect the neural network weights.
- 