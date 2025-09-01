#!/usr/bin/env python3
"""
Learning Stability Analyzer for RL Models
Assess model training stability without real cluster deployment
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

class StabilityAnalyzer:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.output_file = self.model_dir / "output.txt"
        self.config_file = self.model_dir / "model_config.json"
        
    def extract_training_metrics(self):
        """Extract loss, entropy, param changes from training logs"""
        if not self.output_file.exists():
            print(f"❌ No output.txt found in {self.model_dir}")
            return None
            
        with open(self.output_file) as f:
            lines = f.readlines()
        
        metrics = {
            'losses': [], 'entropies': [], 'param_changes': [],
            'epochs': [], 'batches': [], 'confidences': []
        }
        
        for line in lines:
            # Training metrics
            if 'Loss=' in line and 'Entropy=' in line:
                epoch_match = re.search(r'Epoch\s+(\d+)', line)
                batch_match = re.search(r'Global Batch Idx\s+(\d+)', line)
                loss_match = re.search(r'Loss=([0-9.-]+)', line)
                entropy_match = re.search(r'Entropy=([0-9.-]+)', line)
                
                if all([epoch_match, batch_match, loss_match, entropy_match]):
                    metrics['epochs'].append(int(epoch_match.group(1)))
                    metrics['batches'].append(int(batch_match.group(1)))
                    metrics['losses'].append(float(loss_match.group(1)))
                    metrics['entropies'].append(float(entropy_match.group(1)))
            
            # Parameter changes
            elif 'Max param change:' in line:
                param_match = re.search(r'Max param change: ([0-9.-]+)', line)
                if param_match:
                    metrics['param_changes'].append(float(param_match.group(1)))
            
            # Evaluation confidence
            elif 'Evaluation - Confidence:' in line:
                conf_match = re.search(r'Confidence: ([0-9.-]+)', line)
                if conf_match:
                    metrics['confidences'].append(float(conf_match.group(1)))
        
        return {k: np.array(v) for k, v in metrics.items()}
    
    def compute_stability_scores(self, metrics):
        """Compute stability indicators"""
        scores = {}
        
        # Loss stability
        if len(metrics['losses']) > 10:
            losses = metrics['losses']
            scores['loss_variance'] = np.var(losses)
            scores['loss_trend'] = losses[-10:].mean() - losses[:10].mean()
            scores['loss_convergence'] = self._check_convergence(losses)
        
        # Entropy stability (policy exploration)
        if len(metrics['entropies']) > 10:
            entropies = metrics['entropies']
            scores['entropy_variance'] = np.var(entropies)
            scores['entropy_trend'] = entropies[-10:].mean() - entropies[:10].mean()
            scores['entropy_final'] = entropies[-10:].mean()
        
        # Parameter stability
        if len(metrics['param_changes']) > 10:
            param_changes = metrics['param_changes']
            scores['param_variance'] = np.var(param_changes)
            scores['param_trend'] = param_changes[-10:].mean() - param_changes[:10].mean()
            scores['param_convergence'] = self._check_convergence(param_changes)
        
        # Confidence stability
        if len(metrics['confidences']) > 5:
            confidences = metrics['confidences']
            scores['confidence_variance'] = np.var(confidences)
            scores['confidence_final'] = confidences[-5:].mean()
            scores['confidence_trend'] = confidences[-5:].mean() - confidences[:5].mean()
        
        return scores
    
    def _check_convergence(self, values, window=50):
        """Check if values have converged (lower variance in recent window)"""
        if len(values) < window * 2:
            return 0.0
        
        recent_std = np.std(values[-window:])
        early_std = np.std(values[:window])
        
        if early_std == 0:
            return 1.0 if recent_std == 0 else 0.0
        
        improvement = (early_std - recent_std) / early_std
        return max(0.0, improvement)  # 0-1 scale, higher = better convergence
    
    def assess_stability(self):
        """Main stability assessment"""
        print(f"🔬 STABILITY ANALYSIS: {self.model_dir.name}")
        print("=" * 60)
        
        # Extract metrics
        metrics = self.extract_training_metrics()
        if metrics is None:
            return None
        
        scores = self.compute_stability_scores(metrics)
        
        # Print results
        print(f"\n📊 TRAINING METRICS:")
        print(f"   Loss points: {len(metrics['losses'])}")
        print(f"   Confidence evaluations: {len(metrics['confidences'])}")
        
        print(f"\n📈 STABILITY SCORES:")
        if 'loss_convergence' in scores:
            print(f"   Loss convergence: {scores['loss_convergence']:.3f} (0=poor, 1=excellent)")
        if 'param_convergence' in scores:
            print(f"   Parameter convergence: {scores['param_convergence']:.3f}")
        if 'entropy_final' in scores:
            print(f"   Final entropy: {scores['entropy_final']:.3f} (exploration level)")
        if 'confidence_final' in scores:
            print(f"   Final confidence: {scores['confidence_final']:.3f}")
        
        # Overall stability assessment
        overall_score = self._compute_overall_stability(scores)
        print(f"\n🎯 OVERALL STABILITY: {overall_score:.3f}")
        
        if overall_score > 0.7:
            print("   ✅ STABLE - Safe for deployment")
        elif overall_score > 0.4:
            print("   ⚠️  MODERATE - Consider more training")
        else:
            print("   ❌ UNSTABLE - Do NOT deploy")
        
        return scores
    
    def _compute_overall_stability(self, scores):
        """Compute overall stability score 0-1"""
        stability_components = []
        
        if 'loss_convergence' in scores:
            stability_components.append(scores['loss_convergence'])
        if 'param_convergence' in scores:
            stability_components.append(scores['param_convergence'])
        
        # Entropy should be moderate (not too high, not too low)
        if 'entropy_final' in scores:
            entropy = scores['entropy_final']
            # Optimal entropy around 1.0-1.5 for 7 actions
            entropy_score = 1.0 - abs(entropy - 1.2) / 2.0
            stability_components.append(max(0.0, entropy_score))
        
        if stability_components:
            return np.mean(stability_components)
        else:
            return 0.0
    
    def compare_models(self, other_model_dirs):
        """Compare stability across multiple models"""
        print("🔍 COMPARATIVE STABILITY ANALYSIS")
        print("=" * 60)
        
        all_models = [self.model_dir] + [Path(d) for d in other_model_dirs]
        results = []
        
        for model_dir in all_models:
            analyzer = StabilityAnalyzer(model_dir)
            scores = analyzer.assess_stability()
            if scores:
                results.append({
                    'model': model_dir.name,
                    'overall_stability': analyzer._compute_overall_stability(scores),
                    **scores
                })
            print()
        
        if results:
            df = pd.DataFrame(results)
            print("📊 STABILITY COMPARISON:")
            print(df[['model', 'overall_stability', 'confidence_final']].to_string(index=False))
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Analyze RL model training stability')
    parser.add_argument('model_dir', help='Path to model directory with output.txt')
    parser.add_argument('--compare', nargs='*', help='Additional model directories to compare')
    
    args = parser.parse_args()
    
    analyzer = StabilityAnalyzer(args.model_dir)
    
    if args.compare:
        analyzer.compare_models(args.compare)
    else:
        analyzer.assess_stability()

if __name__ == "__main__":
    main()

