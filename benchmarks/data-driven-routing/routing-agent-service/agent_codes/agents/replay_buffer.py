import numpy as np
import threading
from collections import deque

# ============================================================================
# Prioritized Experience Replay Buffer
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay for sample-efficient RL learning.
    
    Samples experiences proportional to their TD error (learning value):
    - High TD error = surprising outcome = learn more from it
    - Rare events (failures) naturally get high priority
    - 2-3x better sample efficiency than uniform sampling
    
    Based on: "Prioritized Experience Replay" (Schaul et al., 2015)
    """
    def __init__(self, maxlen=1000, alpha=0.6, beta=0.4):
        """
        Args:
            maxlen: Maximum buffer size
            alpha: Prioritization strength (0=uniform, 1=full prioritization)
            beta: Importance sampling correction (reduces bias)
        """
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.lock = threading.Lock()  # Thread-safe operations
        
    def add(self, experience):
        """Add experience with maximum priority (explore new experiences first)"""
        with self.lock:
            self.buffer.append(experience)
            # New experiences get max priority
            self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        """
        Sample batch with probability ∝ priority^alpha
        
        Returns:
            batch: List of experiences
            indices: Indices in buffer (for priority updates)
            weights: Importance sampling weights (for unbiasing)
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                return [], [], []
            
            # Convert priorities to sampling probabilities
            priorities = np.array(self.priorities, dtype=np.float32)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            # Sample indices
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            
            # Importance sampling weights: (N * P(i))^(-beta)
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize for stability
            
            batch = [self.buffer[i] for i in indices]
            
            return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD error magnitude
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD errors (target - prediction)
        """
        with self.lock:
            for idx, td_error in zip(indices, td_errors):
                if idx < len(self.priorities):
                    # Priority = |TD error| + small constant
                    priority = abs(td_error) + 1e-6
                    self.priorities[idx] = priority
                    self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)
