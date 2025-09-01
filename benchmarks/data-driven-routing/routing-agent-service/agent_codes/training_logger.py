#!/usr/bin/env python3
"""
CSV Training Logger for RL Models
Logs detailed training metrics at every iteration for analysis
"""

import csv
import os
import torch
import time
from pathlib import Path

class TrainingLogger:
    def __init__(self, log_dir, filename="training_metrics.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        
        self.fieldnames = [
            # Iteration info
            'timestamp', 'epoch', 'global_batch_idx', 'learn_call', 'local_batch_idx',
            
            # Loss components
            'policy_loss', 'entropy', 'entropy_bonus', 'total_loss',
            
            # Batch statistics
            'batch_size', 'avg_reward', 'reward_std', 'reward_min', 'reward_max',
            
            # Policy metrics
            'avg_confidence', 'max_confidence', 'min_confidence',
            'action_entropy', 'policy_sharpness',
            
            # Gradient metrics
            'max_param_change', 'avg_grad_norm', 'max_grad_norm',
            
            # Action distribution
            'action_0_prob', 'action_1_prob', 'action_2_prob', 'action_3_prob',
            'action_4_prob', 'action_5_prob', 'action_6_prob',
            
            # Learning dynamics
            'learning_rate', 'optimizer_step',
            
            # Evaluation metrics (when available)
            'eval_accuracy', 'eval_confidence'
        ]
        
        # Initialize CSV file with headers
        self._init_csv_file()
        self.current_row = {}
        
    def _init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def start_iteration(self, epoch, global_batch_idx, learn_call, local_batch_idx):
        """Start logging for a new iteration"""
        self.current_row = {
            'timestamp': time.time(),
            'epoch': epoch,
            'global_batch_idx': global_batch_idx,
            'learn_call': learn_call,
            'local_batch_idx': local_batch_idx
        }
    
    def log_losses(self, policy_loss, entropy, entropy_bonus, total_loss):
        """Log loss components"""
        self.current_row.update({
            'policy_loss': float(policy_loss),
            'entropy': float(entropy),
            'entropy_bonus': float(entropy_bonus),
            'total_loss': float(total_loss)
        })
    
    def log_batch_stats(self, batch_rewards, batch_size):
        """Log batch statistics"""
        rewards = batch_rewards.cpu().numpy() if hasattr(batch_rewards, 'cpu') else batch_rewards
        self.current_row.update({
            'batch_size': batch_size,
            'avg_reward': float(rewards.mean()),
            'reward_std': float(rewards.std()),
            'reward_min': float(rewards.min()),
            'reward_max': float(rewards.max())
        })
    
    def log_policy_metrics(self, action_probs):
        """Log policy-related metrics"""
        probs = action_probs.detach().cpu()
        
        # Confidence metrics (max probability per sample)
        confidences = probs.max(dim=1)[0]
        
        # Action distribution (average probability for each action)
        avg_action_probs = probs.mean(dim=0)
        
        # Policy sharpness (how peaked are the distributions)
        entropy_per_sample = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        avg_entropy = entropy_per_sample.mean()
        
        self.current_row.update({
            'avg_confidence': float(confidences.mean()),
            'max_confidence': float(confidences.max()),
            'min_confidence': float(confidences.min()),
            'action_entropy': float(avg_entropy),
            'policy_sharpness': float(1.0 / (avg_entropy + 1e-8))  # Inverse of entropy
        })
        
        # Log individual action probabilities
        for i, prob in enumerate(avg_action_probs[:7]):  # Assume 7 actions max
            self.current_row[f'action_{i}_prob'] = float(prob)
    
    def log_gradients(self, model, max_param_change):
        """Log gradient statistics"""
        total_norm = 0.0
        max_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_norm = max(max_norm, param_norm.item())
                param_count += 1
        
        avg_grad_norm = (total_norm / param_count) ** 0.5 if param_count > 0 else 0.0
        
        self.current_row.update({
            'max_param_change': float(max_param_change),
            'avg_grad_norm': float(avg_grad_norm),
            'max_grad_norm': float(max_norm)
        })
    
    def log_optimizer_info(self, optimizer, step_count):
        """Log optimizer state"""
        # Get learning rate from optimizer
        lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
        
        self.current_row.update({
            'learning_rate': float(lr),
            'optimizer_step': int(step_count)
        })
    
    def log_evaluation(self, accuracy=None, confidence=None):
        """Log evaluation metrics"""
        if accuracy is not None:
            self.current_row['eval_accuracy'] = float(accuracy)
        if confidence is not None:
            self.current_row['eval_confidence'] = float(confidence)
    
    def save_iteration(self):
        """Save the current iteration to CSV"""
        # Fill missing fields with None
        complete_row = {field: self.current_row.get(field, None) for field in self.fieldnames}
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(complete_row)
    
    def get_recent_metrics(self, n_iterations=10):
        """Get recent training metrics for analysis"""
        try:
            import pandas as pd
            df = pd.read_csv(self.log_file)
            return df.tail(n_iterations)
        except ImportError:
            print("pandas not available for metric analysis")
            return None
        except Exception as e:
            print(f"Error reading training log: {e}")
            return None

def example_usage():
    """Example of how to integrate with training loop"""
    logger = TrainingLogger("/tmp/test_model", "training_log.csv")
    
    # In training loop:
    for epoch in range(2):
        for batch_idx in range(3):
            # Start iteration
            logger.start_iteration(epoch, batch_idx, 1, 0)
            
            # Simulate training step
            import torch
            policy_loss = torch.tensor(0.5)
            entropy = torch.tensor(1.2)
            entropy_bonus = torch.tensor(0.01)
            total_loss = policy_loss - entropy_bonus
            
            # Log metrics
            logger.log_losses(policy_loss, entropy, entropy_bonus, total_loss)
            
            # Simulate batch data
            batch_rewards = torch.randn(32)
            logger.log_batch_stats(batch_rewards, 32)
            
            # Simulate action probabilities
            action_probs = torch.softmax(torch.randn(32, 7), dim=1)
            logger.log_policy_metrics(action_probs)
            
            # Save iteration
            logger.save_iteration()
            
            print(f"Logged epoch {epoch}, batch {batch_idx}")

if __name__ == "__main__":
    example_usage()


