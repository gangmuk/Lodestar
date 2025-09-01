#!/usr/bin/env python3
"""
CSV Training Data Analyzer
Comprehensive analysis of detailed training metrics for RL models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

class CSVTrainingAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = Path(csv_file)
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load CSV training data"""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        self.df = pd.read_csv(self.csv_file)
        print(f"📊 Loaded {len(self.df)} training iterations from {self.csv_file}")
        
    def analyze_convergence(self):
        """Analyze training convergence patterns"""
        print("\n🔬 CONVERGENCE ANALYSIS")
        print("=" * 50)
        
        if len(self.df) < 10:
            print("❌ Not enough data for convergence analysis")
            return
        
        # Loss convergence
        loss_trend = self.df['total_loss'].rolling(window=20).mean()
        loss_stability = self.df['total_loss'].rolling(window=20).std()
        
        print(f"📉 Loss Trends:")
        print(f"   Initial loss (first 10): {self.df['total_loss'][:10].mean():.4f}")
        print(f"   Final loss (last 10): {self.df['total_loss'][-10:].mean():.4f}")
        print(f"   Improvement: {((self.df['total_loss'][:10].mean() - self.df['total_loss'][-10:].mean()) / self.df['total_loss'][:10].mean() * 100):.1f}%")
        
        # Parameter change convergence
        param_changes = self.df['max_param_change'].dropna()
        if len(param_changes) > 10:
            print(f"   Initial param changes: {param_changes[:10].mean():.6f}")
            print(f"   Final param changes: {param_changes[-10:].mean():.6f}")
            
        # Entropy evolution (exploration)
        print(f"📊 Exploration (Entropy):")
        print(f"   Initial entropy: {self.df['entropy'][:10].mean():.4f}")
        print(f"   Final entropy: {self.df['entropy'][-10:].mean():.4f}")
        
        # Confidence evolution
        confidence_data = self.df['avg_confidence'].dropna()
        if len(confidence_data) > 10:
            print(f"🎯 Confidence Evolution:")
            print(f"   Initial confidence: {confidence_data[:10].mean():.4f}")
            print(f"   Final confidence: {confidence_data[-10:].mean():.4f}")
    
    def analyze_learning_dynamics(self):
        """Analyze learning dynamics and stability"""
        print("\n⚡ LEARNING DYNAMICS")
        print("=" * 50)
        
        # Gradient analysis
        grad_norms = self.df['avg_grad_norm'].dropna()
        if len(grad_norms) > 0:
            print(f"🔀 Gradient Statistics:")
            print(f"   Average grad norm: {grad_norms.mean():.6f}")
            print(f"   Max grad norm: {grad_norms.max():.6f}")
            print(f"   Grad norm stability (std): {grad_norms.std():.6f}")
        
        # Policy sharpness evolution
        sharpness = self.df['policy_sharpness'].dropna()
        if len(sharpness) > 10:
            print(f"🎯 Policy Sharpness:")
            print(f"   Initial sharpness: {sharpness[:10].mean():.4f}")
            print(f"   Final sharpness: {sharpness[-10:].mean():.4f}")
            print(f"   (Higher = more decisive, Lower = more uniform)")
            
        # Reward statistics evolution
        print(f"💰 Batch Reward Evolution:")
        print(f"   Average reward: {self.df['avg_reward'].mean():.4f}")
        print(f"   Reward variance: {self.df['reward_std'].mean():.4f}")
        
    def analyze_action_distribution(self):
        """Analyze action selection patterns"""
        print("\n🎲 ACTION DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Get action probability columns
        action_cols = [col for col in self.df.columns if col.startswith('action_') and col.endswith('_prob')]
        
        if not action_cols:
            print("❌ No action probability data found")
            return
            
        # Calculate average action probabilities
        action_probs = self.df[action_cols].mean()
        action_entropy = -(action_probs * np.log(action_probs + 1e-8)).sum()
        
        print(f"📊 Average Action Probabilities:")
        for i, col in enumerate(action_cols):
            prob = action_probs[col]
            print(f"   Action {i}: {prob:.4f} ({prob*100:.1f}%)")
            
        print(f"📈 Action Distribution Metrics:")
        print(f"   Entropy: {action_entropy:.4f} (max={np.log(len(action_cols)):.4f})")
        print(f"   Balance score: {action_entropy/np.log(len(action_cols)):.4f} (1.0=perfect balance)")
        
        # Check for action selection drift over time
        if len(self.df) > 50:
            early_probs = self.df[action_cols][:25].mean()
            late_probs = self.df[action_cols][-25:].mean()
            drift = np.abs(late_probs - early_probs).sum()
            print(f"   Policy drift: {drift:.4f} (lower=more stable)")
    
    def detect_training_issues(self):
        """Detect potential training problems"""
        print("\n⚠️  TRAINING ISSUE DETECTION")
        print("=" * 50)
        
        issues = []
        
        # Check for exploding/vanishing gradients
        grad_norms = self.df['avg_grad_norm'].dropna()
        if len(grad_norms) > 0:
            if grad_norms.max() > 1.0:
                issues.append(f"🔴 Large gradients detected (max: {grad_norms.max():.4f})")
            if grad_norms.mean() < 1e-6:
                issues.append(f"🔴 Very small gradients detected (avg: {grad_norms.mean():.8f})")
        
        # Check for loss explosion/stagnation
        losses = self.df['total_loss'].dropna()
        if len(losses) > 20:
            recent_loss_std = losses[-20:].std()
            if recent_loss_std < 1e-6:
                issues.append("🔴 Loss appears to have stagnated")
            if losses.max() - losses.min() > 10:
                issues.append("🔴 Large loss variations detected")
        
        # Check for overly confident/uniform policies
        confidence = self.df['avg_confidence'].dropna()
        if len(confidence) > 10:
            final_confidence = confidence[-10:].mean()
            if final_confidence > 0.95:
                issues.append(f"🟡 Very high confidence (avg: {final_confidence:.3f}) - possible overconfidence")
            elif final_confidence < 0.2:
                issues.append(f"🟡 Very low confidence (avg: {final_confidence:.3f}) - model may be too uncertain")
        
        # Check for parameter update health
        param_changes = self.df['max_param_change'].dropna()
        if len(param_changes) > 10:
            final_changes = param_changes[-10:].mean()
            if final_changes > 0.1:
                issues.append(f"🟡 Large parameter changes (avg: {final_changes:.4f}) - training may be unstable")
            elif final_changes < 1e-8:
                issues.append(f"🟡 Very small parameter changes (avg: {final_changes:.8f}) - learning may have stopped")
        
        if issues:
            for issue in issues:
                print(f"   {issue}")
        else:
            print("   ✅ No obvious training issues detected")
    
    def generate_plots(self, output_dir=None):
        """Generate comprehensive training plots"""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Training Analysis: {self.csv_file.name}', fontsize=16)
        
        # 1. Loss curves
        axes[0,0].plot(self.df['total_loss'], label='Total Loss', alpha=0.7)
        axes[0,0].plot(self.df['policy_loss'], label='Policy Loss', alpha=0.7)
        axes[0,0].set_title('Loss Evolution')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 2. Entropy and confidence
        axes[0,1].plot(self.df['entropy'], label='Entropy', alpha=0.7)
        if 'avg_confidence' in self.df.columns:
            axes2 = axes[0,1].twinx()
            axes2.plot(self.df['avg_confidence'], 'r-', label='Avg Confidence', alpha=0.7)
            axes2.set_ylabel('Confidence')
            axes2.legend(loc='upper right')
        axes[0,1].set_title('Exploration vs Confidence')
        axes[0,1].legend(loc='upper left')
        axes[0,1].grid(True)
        
        # 3. Parameter changes
        if 'max_param_change' in self.df.columns:
            axes[0,2].semilogy(self.df['max_param_change'].dropna())
            axes[0,2].set_title('Max Parameter Changes (log scale)')
            axes[0,2].grid(True)
        
        # 4. Reward statistics
        axes[1,0].plot(self.df['avg_reward'], label='Avg Reward', alpha=0.7)
        axes[1,0].fill_between(range(len(self.df)), 
                               self.df['avg_reward'] - self.df['reward_std'],
                               self.df['avg_reward'] + self.df['reward_std'],
                               alpha=0.3, label='±1 std')
        axes[1,0].set_title('Batch Reward Evolution')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # 5. Gradient norms
        if 'avg_grad_norm' in self.df.columns:
            grad_data = self.df['avg_grad_norm'].dropna()
            if len(grad_data) > 0:
                axes[1,1].plot(grad_data)
                axes[1,1].set_title('Gradient Norms')
                axes[1,1].grid(True)
        
        # 6. Action probabilities
        action_cols = [col for col in self.df.columns if col.startswith('action_') and col.endswith('_prob')]
        if action_cols:
            for i, col in enumerate(action_cols):
                axes[1,2].plot(self.df[col], label=f'Action {i}', alpha=0.7)
            axes[1,2].set_title('Action Probability Evolution')
            axes[1,2].legend()
            axes[1,2].grid(True)
        
        # 7. Policy sharpness
        if 'policy_sharpness' in self.df.columns:
            axes[2,0].plot(self.df['policy_sharpness'].dropna())
            axes[2,0].set_title('Policy Sharpness (1/entropy)')
            axes[2,0].grid(True)
        
        # 8. Evaluation metrics
        eval_cols = [col for col in self.df.columns if col.startswith('eval_')]
        if eval_cols:
            for col in eval_cols:
                eval_data = self.df[col].dropna()
                if len(eval_data) > 0:
                    axes[2,1].plot(eval_data, 'o-', label=col.replace('eval_', ''), alpha=0.7)
            axes[2,1].set_title('Evaluation Metrics')
            axes[2,1].legend()
            axes[2,1].grid(True)
        
        # 9. Learning rate
        if 'learning_rate' in self.df.columns:
            lr_data = self.df['learning_rate'].dropna()
            if len(lr_data) > 0 and lr_data.nunique() > 1:
                axes[2,2].plot(lr_data)
                axes[2,2].set_title('Learning Rate Schedule')
                axes[2,2].grid(True)
        
        plt.tight_layout()
        
        if output_dir:
            plot_file = output_dir / 'training_analysis.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"📈 Plots saved to: {plot_file}")
        else:
            plt.show()
    
    def generate_report(self):
        """Generate comprehensive training report"""
        print(f"\n📋 COMPREHENSIVE TRAINING REPORT")
        print("=" * 60)
        print(f"Dataset: {self.csv_file}")
        print(f"Total iterations: {len(self.df)}")
        print(f"Epochs covered: {self.df['epoch'].nunique()}")
        
        self.analyze_convergence()
        self.analyze_learning_dynamics()
        self.analyze_action_distribution()
        self.detect_training_issues()

def main():
    parser = argparse.ArgumentParser(description='Analyze detailed CSV training logs')
    parser.add_argument('csv_file', help='Path to training_metrics.csv file')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    parser.add_argument('--output-dir', help='Output directory for plots')
    
    args = parser.parse_args()
    
    analyzer = CSVTrainingAnalyzer(args.csv_file)
    analyzer.generate_report()
    
    if args.plots:
        analyzer.generate_plots(args.output_dir)

if __name__ == "__main__":
    main()

