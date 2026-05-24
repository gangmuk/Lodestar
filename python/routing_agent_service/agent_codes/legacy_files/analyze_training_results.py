#!/usr/bin/env python3
"""
Analyze the training results to see if the model was trained well.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_training_results():
    """
    Analyze the training results to assess if the model was trained well.
    """
    
    print("="*80)
    print("TRAINING RESULTS ANALYSIS: Was the model trained well?")
    print("="*80)
    
    # Load the training metrics
    metrics_path = "/users/gangmuk/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/training_data/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80-half/all/final_model-data-processed/training_metrics.csv"
    df = pd.read_csv(metrics_path)
    
    print(f"Training data: {len(df)} training steps")
    print(f"Training epochs: {df['epoch'].max() + 1}")
    print(f"Total batches: {df['global_batch_idx'].max()}")
    
    # 1. REWARD IMPROVEMENT ANALYSIS
    print(f"\n" + "="*60)
    print("1. REWARD IMPROVEMENT ANALYSIS")
    print("="*60)
    
    # Calculate moving averages
    window_size = 50
    df['avg_reward_ma'] = df['avg_reward'].rolling(window=window_size, min_periods=1).mean()
    df['reward_std_ma'] = df['reward_std'].rolling(window=window_size, min_periods=1).mean()
    
    initial_reward = df['avg_reward_ma'].iloc[:100].mean()
    final_reward = df['avg_reward_ma'].iloc[-100:].mean()
    improvement = final_reward - initial_reward
    
    print(f"Initial average reward (first 100 steps): {initial_reward:.3f}")
    print(f"Final average reward (last 100 steps): {final_reward:.3f}")
    print(f"Improvement: {improvement:.3f}")
    
    if improvement > 0.05:
        print("✅ REWARD IMPROVEMENT: Significant improvement during training")
    elif improvement > 0.01:
        print("⚠️  REWARD IMPROVEMENT: Moderate improvement during training")
    else:
        print("❌ REWARD IMPROVEMENT: Little to no improvement during training")
    
    # 2. POLICY CONVERGENCE ANALYSIS
    print(f"\n" + "="*60)
    print("2. POLICY CONVERGENCE ANALYSIS")
    print("="*60)
    
    # Calculate action probability statistics
    action_cols = [col for col in df.columns if col.startswith('action_') and col.endswith('_prob')]
    
    initial_entropy = df['action_entropy'].iloc[:100].mean()
    final_entropy = df['action_entropy'].iloc[-100:].mean()
    entropy_reduction = initial_entropy - final_entropy
    
    print(f"Initial action entropy: {initial_entropy:.3f}")
    print(f"Final action entropy: {final_entropy:.3f}")
    print(f"Entropy reduction: {entropy_reduction:.3f}")
    
    if entropy_reduction > 0.5:
        print("✅ POLICY CONVERGENCE: Strong convergence (low entropy)")
    elif entropy_reduction > 0.2:
        print("⚠️  POLICY CONVERGENCE: Moderate convergence")
    else:
        print("❌ POLICY CONVERGENCE: Weak convergence (high entropy)")
    
    # Check action probability distribution
    final_action_probs = df[action_cols].iloc[-100:].mean()
    print(f"\nFinal action probabilities:")
    for i, prob in enumerate(final_action_probs):
        print(f"  Action {i}: {prob:.3f}")
    
    # Check if policy is too deterministic
    max_prob = final_action_probs.max()
    if max_prob > 0.8:
        print("⚠️  WARNING: Policy is very deterministic (may be overfitting)")
    elif max_prob > 0.6:
        print("✅ Policy has good balance between exploration and exploitation")
    else:
        print("⚠️  WARNING: Policy is still very exploratory")
    
    # 3. LOSS CONVERGENCE ANALYSIS
    print(f"\n" + "="*60)
    print("3. LOSS CONVERGENCE ANALYSIS")
    print("="*60)
    
    # Calculate moving averages for losses
    df['policy_loss_ma'] = df['policy_loss'].rolling(window=window_size, min_periods=1).mean()
    df['total_loss_ma'] = df['total_loss'].rolling(window=window_size, min_periods=1).mean()
    
    initial_policy_loss = df['policy_loss_ma'].iloc[:100].mean()
    final_policy_loss = df['policy_loss_ma'].iloc[-100:].mean()
    loss_improvement = initial_policy_loss - final_policy_loss  # Negative loss is better
    
    print(f"Initial policy loss: {initial_policy_loss:.3f}")
    print(f"Final policy loss: {final_policy_loss:.3f}")
    print(f"Loss improvement: {loss_improvement:.3f}")
    
    if loss_improvement > 0.5:
        print("✅ LOSS CONVERGENCE: Strong loss reduction")
    elif loss_improvement > 0.2:
        print("⚠️  LOSS CONVERGENCE: Moderate loss reduction")
    else:
        print("❌ LOSS CONVERGENCE: Weak loss reduction")
    
    # 4. GRADIENT ANALYSIS
    print(f"\n" + "="*60)
    print("4. GRADIENT ANALYSIS")
    print("="*60)
    
    initial_grad_norm = df['avg_grad_norm'].iloc[:100].mean()
    final_grad_norm = df['avg_grad_norm'].iloc[-100:].mean()
    
    print(f"Initial gradient norm: {initial_grad_norm:.3f}")
    print(f"Final gradient norm: {final_grad_norm:.3f}")
    
    if final_grad_norm < 0.1:
        print("✅ GRADIENT: Small gradients (good convergence)")
    elif final_grad_norm < 0.5:
        print("⚠️  GRADIENT: Moderate gradients")
    else:
        print("❌ GRADIENT: Large gradients (may need more training)")
    
    # 5. CONFIDENCE ANALYSIS
    print(f"\n" + "="*60)
    print("5. CONFIDENCE ANALYSIS")
    print("="*60)
    
    initial_confidence = df['avg_confidence'].iloc[:100].mean()
    final_confidence = df['avg_confidence'].iloc[-100:].mean()
    confidence_improvement = final_confidence - initial_confidence
    
    print(f"Initial confidence: {initial_confidence:.3f}")
    print(f"Final confidence: {final_confidence:.3f}")
    print(f"Confidence improvement: {confidence_improvement:.3f}")
    
    if confidence_improvement > 0.3:
        print("✅ CONFIDENCE: Strong confidence improvement")
    elif confidence_improvement > 0.1:
        print("⚠️  CONFIDENCE: Moderate confidence improvement")
    else:
        print("❌ CONFIDENCE: Weak confidence improvement")
    
    # 6. EVALUATION METRICS (if available)
    print(f"\n" + "="*60)
    print("6. EVALUATION METRICS")
    print("="*60)
    
    eval_accuracy = df['eval_accuracy'].dropna()
    eval_confidence = df['eval_confidence'].dropna()
    
    if len(eval_accuracy) > 0:
        final_eval_accuracy = eval_accuracy.iloc[-1]
        final_eval_confidence = eval_confidence.iloc[-1]
        
        print(f"Final evaluation accuracy: {final_eval_accuracy:.3f}")
        print(f"Final evaluation confidence: {final_eval_confidence:.3f}")
        
        if final_eval_accuracy > 0.8:
            print("✅ EVALUATION: High accuracy")
        elif final_eval_accuracy > 0.6:
            print("⚠️  EVALUATION: Moderate accuracy")
        else:
            print("❌ EVALUATION: Low accuracy")
    else:
        print("No evaluation metrics available")
    
    # 7. OVERALL TRAINING ASSESSMENT
    print(f"\n" + "="*60)
    print("7. OVERALL TRAINING ASSESSMENT")
    print("="*60)
    
    # Score the training quality
    score = 0
    max_score = 6
    
    # Reward improvement (2 points)
    if improvement > 0.05:
        score += 2
        print("✅ Reward improvement: 2/2 points")
    elif improvement > 0.01:
        score += 1
        print("⚠️  Reward improvement: 1/2 points")
    else:
        print("❌ Reward improvement: 0/2 points")
    
    # Policy convergence (1 point)
    if entropy_reduction > 0.5:
        score += 1
        print("✅ Policy convergence: 1/1 point")
    else:
        print("❌ Policy convergence: 0/1 point")
    
    # Loss convergence (1 point)
    if loss_improvement > 0.5:
        score += 1
        print("✅ Loss convergence: 1/1 point")
    else:
        print("❌ Loss convergence: 0/1 point")
    
    # Gradient stability (1 point)
    if final_grad_norm < 0.5:
        score += 1
        print("✅ Gradient stability: 1/1 point")
    else:
        print("❌ Gradient stability: 0/1 point")
    
    # Confidence improvement (1 point)
    if confidence_improvement > 0.1:
        score += 1
        print("✅ Confidence improvement: 1/1 point")
    else:
        print("❌ Confidence improvement: 0/1 point")
    
    print(f"\nOVERALL TRAINING SCORE: {score}/{max_score} points")
    
    if score >= 5:
        print("🎯 VERDICT: Model was trained VERY WELL")
    elif score >= 3:
        print("⚠️  VERDICT: Model was trained MODERATELY WELL")
    else:
        print("❌ VERDICT: Model was trained POORLY")
    
    return {
        'score': score,
        'max_score': max_score,
        'reward_improvement': improvement,
        'entropy_reduction': entropy_reduction,
        'loss_improvement': loss_improvement,
        'final_grad_norm': final_grad_norm,
        'confidence_improvement': confidence_improvement,
        'final_eval_accuracy': final_eval_accuracy if len(eval_accuracy) > 0 else None,
        'final_eval_confidence': final_eval_confidence if len(eval_confidence) > 0 else None
    }

def create_training_visualization():
    """
    Create visualization of the training progress.
    """
    
    # Load data
    metrics_path = "/users/gangmuk/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/training_data/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80-half/all/final_model-data-processed/training_metrics.csv"
    df = pd.read_csv(metrics_path)
    
    # Calculate moving averages
    window_size = 50
    df['avg_reward_ma'] = df['avg_reward'].rolling(window=window_size, min_periods=1).mean()
    df['policy_loss_ma'] = df['policy_loss'].rolling(window=window_size, min_periods=1).mean()
    df['action_entropy_ma'] = df['action_entropy'].rolling(window=window_size, min_periods=1).mean()
    df['avg_confidence_ma'] = df['avg_confidence'].rolling(window=window_size, min_periods=1).mean()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Reward progression
    axes[0,0].plot(df['global_batch_idx'], df['avg_reward'], alpha=0.3, color='lightblue', label='Raw')
    axes[0,0].plot(df['global_batch_idx'], df['avg_reward_ma'], color='blue', linewidth=2, label='Moving Average')
    axes[0,0].set_xlabel('Batch Index')
    axes[0,0].set_ylabel('Average Reward')
    axes[0,0].set_title('Reward Progression')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Policy loss progression
    axes[0,1].plot(df['global_batch_idx'], df['policy_loss'], alpha=0.3, color='lightcoral', label='Raw')
    axes[0,1].plot(df['global_batch_idx'], df['policy_loss_ma'], color='red', linewidth=2, label='Moving Average')
    axes[0,1].set_xlabel('Batch Index')
    axes[0,1].set_ylabel('Policy Loss')
    axes[0,1].set_title('Policy Loss Progression')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Action entropy progression
    axes[1,0].plot(df['global_batch_idx'], df['action_entropy'], alpha=0.3, color='lightgreen', label='Raw')
    axes[1,0].plot(df['global_batch_idx'], df['action_entropy_ma'], color='green', linewidth=2, label='Moving Average')
    axes[1,0].set_xlabel('Batch Index')
    axes[1,0].set_ylabel('Action Entropy')
    axes[1,0].set_title('Action Entropy Progression')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Confidence progression
    axes[1,1].plot(df['global_batch_idx'], df['avg_confidence'], alpha=0.3, color='lightyellow', label='Raw')
    axes[1,1].plot(df['global_batch_idx'], df['avg_confidence_ma'], color='orange', linewidth=2, label='Moving Average')
    axes[1,1].set_xlabel('Batch Index')
    axes[1,1].set_ylabel('Average Confidence')
    axes[1,1].set_title('Confidence Progression')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Training visualization saved as 'training_progress_analysis.png'")
    
    return fig

if __name__ == "__main__":
    results = analyze_training_results()
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    eval_acc_str = f"{results['final_eval_accuracy']:.3f}" if results['final_eval_accuracy'] is not None else 'N/A'
    eval_conf_str = f"{results['final_eval_confidence']:.3f}" if results['final_eval_confidence'] is not None else 'N/A'
    
    print(f"""
TRAINING RESULTS SUMMARY:

Score: {results['score']}/{results['max_score']} points

Key Metrics:
- Reward improvement: {results['reward_improvement']:.3f}
- Entropy reduction: {results['entropy_reduction']:.3f}
- Loss improvement: {results['loss_improvement']:.3f}
- Final gradient norm: {results['final_grad_norm']:.3f}
- Confidence improvement: {results['confidence_improvement']:.3f}
- Final eval accuracy: {eval_acc_str}
- Final eval confidence: {eval_conf_str}

CONCLUSION: The model was trained {'VERY WELL' if results['score'] >= 5 else 'MODERATELY WELL' if results['score'] >= 3 else 'POORLY'}.
""")
    
    # Create visualization
    try:
        create_training_visualization()
    except Exception as e:
        print(f"Could not create visualization: {e}")
