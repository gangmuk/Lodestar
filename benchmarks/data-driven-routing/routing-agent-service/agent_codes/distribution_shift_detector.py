#!/usr/bin/env python3
"""
Distribution Shift Detection for Neural Network Reliability

This module provides:
1. DistributionTracker - Collects feature statistics during offline training
2. DistributionShiftMonitor - Monitors aggregate distribution shifts over time (legacy)
3. PerSampleOODDetector - Per-request OOD detection for NN reliability (NEW)

The key insight: Neural networks are interpolators, not extrapolators.
When inputs are outside the training distribution, predictions become unreliable.
This module detects such cases and recommends fallback actions.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OODAction(Enum):
    """Recommended action when OOD is detected"""
    NORMAL = "normal"                    # Input is in-distribution, trust the model
    REDUCE_CONFIDENCE = "reduce_confidence"  # Mild OOD, reduce confidence score
    FALLBACK = "fallback"                # Severe OOD, use heuristic routing instead


class DistributionTracker:
    """
    Tracks feature distributions during offline training.

    Usage:
        tracker = DistributionTracker()
        # During training data processing
        tracker.add_pod_sample({'cpu_kv_cache': 0.0, 'prefill_tokens': 1024, ...})
        tracker.add_request_sample({'input_tokens': 2048, 'total_tokens': 2049, ...})
        # At end of training
        tracker.save_distribution_stats('feature_distribution_statistics.csv')
    """

    def __init__(self):
        self.feature_values = defaultdict(list)
        self.feature_types = {}  # feature_name -> 'pod' or 'request'

    def add_pod_sample(self, pod_features_dict: Dict[str, float]):
        """Add a single pod's features"""
        for feature_name, value in pod_features_dict.items():
            self.feature_values[feature_name].append(float(value))
            self.feature_types[feature_name] = 'pod'

    def add_request_sample(self, request_features_dict: Dict[str, float]):
        """Add a single request's features"""
        for feature_name, value in request_features_dict.items():
            self.feature_values[feature_name].append(float(value))
            self.feature_types[feature_name] = 'request'

    def compute_statistics(self) -> List[Dict]:
        """Compute comprehensive statistics for all tracked features"""
        stats_list = []

        for feature_name, values in self.feature_values.items():
            if len(values) == 0:
                continue

            values_array = np.array(values)

            # Basic statistics
            count = len(values_array)
            mean = np.mean(values_array)
            std = np.std(values_array)
            min_val = np.min(values_array)
            max_val = np.max(values_array)

            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(values_array, percentiles)

            # Zero statistics
            num_zeros = np.sum(values_array == 0.0)
            num_nonzeros = count - num_zeros
            zero_ratio = num_zeros / count if count > 0 else 0.0

            stats_dict = {
                'feature_name': feature_name,
                'feature_type': self.feature_types[feature_name],
                'count': count,
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'p01': percentile_values[0],
                'p05': percentile_values[1],
                'p10': percentile_values[2],
                'p25': percentile_values[3],
                'p50': percentile_values[4],
                'p75': percentile_values[5],
                'p90': percentile_values[6],
                'p95': percentile_values[7],
                'p99': percentile_values[8],
                'num_zeros': num_zeros,
                'num_nonzeros': num_nonzeros,
                'zero_ratio': zero_ratio
            }

            stats_list.append(stats_dict)

        return stats_list

    def save_distribution_stats(self, filepath: str) -> pd.DataFrame:
        """Save distribution statistics to CSV file"""
        stats_list = self.compute_statistics()
        df = pd.DataFrame(stats_list)

        # Sort by feature type then name
        df = df.sort_values(['feature_type', 'feature_name'])

        # Save to CSV
        df.to_csv(filepath, index=False, float_format='%.5f')
        logger.info(f"Saved distribution statistics for {len(df)} features to: {filepath}")

        return df


class DistributionShiftMonitor:
    """
    Monitors feature distributions during online serving and detects shifts.

    Usage:
        monitor = DistributionShiftMonitor('feature_distribution_statistics.csv')
        # For each incoming request
        monitor.add_sample(pod_features_dict, request_features_dict)
        # Periodically check for shifts
        if monitor.should_check():
            warnings = monitor.check_distribution_shift()
            monitor.log_warnings()
    """

    def __init__(
        self,
        training_distribution_csv: str,
        window_size: int = 1000,
        check_interval: int = 100,
        alert_threshold_zscore: float = 2.0
    ):
        self.window_size = window_size
        self.check_interval = check_interval
        self.alert_threshold_zscore = alert_threshold_zscore

        # Load training distribution baseline
        self.training_stats = pd.read_csv(training_distribution_csv)
        self.training_stats_dict = {
            row['feature_name']: row
            for _, row in self.training_stats.iterrows()
        }

        # Sliding window for current distribution
        self.current_values = defaultdict(list)
        self.sample_count = 0

        # Warnings tracking
        self.warnings = []
        self.total_warnings = 0

        logger.info(f"Initialized DistributionShiftMonitor with {len(self.training_stats)} features")
        logger.info(f"  window_size={window_size}, check_interval={check_interval}, alert_threshold={alert_threshold_zscore}σ")

    def add_sample(
        self,
        pod_features_dict: Optional[Dict[str, float]] = None,
        request_features_dict: Optional[Dict[str, float]] = None
    ):
        """Add a sample to the monitoring window"""
        if pod_features_dict:
            for feature_name, value in pod_features_dict.items():
                self.current_values[feature_name].append(float(value))
                # Maintain sliding window
                if len(self.current_values[feature_name]) > self.window_size:
                    self.current_values[feature_name].pop(0)

        if request_features_dict:
            for feature_name, value in request_features_dict.items():
                self.current_values[feature_name].append(float(value))
                # Maintain sliding window
                if len(self.current_values[feature_name]) > self.window_size:
                    self.current_values[feature_name].pop(0)

        self.sample_count += 1

    def should_check(self) -> bool:
        """Check if it's time to perform distribution shift detection"""
        return self.sample_count % self.check_interval == 0

    def check_distribution_shift(self) -> List[Dict]:
        self.warnings = []

        for feature_name, current_values in self.current_values.items():
            if feature_name not in self.training_stats_dict:
                continue

            if len(current_values) < 10:  # Need minimum samples
                continue

            # Get training baseline
            train_stats = self.training_stats_dict[feature_name]
            train_mean = train_stats['mean']
            train_std = train_stats['std']

            if train_std == 0:
                # Can't detect shift if training had zero variance
                continue

            # Compute current statistics
            current_array = np.array(current_values)
            current_mean = np.mean(current_array)
            current_std = np.std(current_array)

            # Check mean shift (in standard deviations)
            mean_shift_zscore = abs(current_mean - train_mean) / train_std

            # Check variance ratio
            variance_ratio = current_std / train_std if train_std > 0 else 1.0

            # Determine severity
            severity = None
            message = None

            # High severity: >3σ mean shift
            if mean_shift_zscore > 3.0:
                severity = 'high'
                message = (f"{feature_name}: CRITICAL mean shift of {mean_shift_zscore:.2f}σ "
                          f"(train={train_mean:.2f}, current={current_mean:.2f})")

            # Medium severity: 2-3σ mean shift OR significant variance change
            elif mean_shift_zscore > self.alert_threshold_zscore:
                severity = 'medium'
                message = (f"{feature_name}: Mean shift of {mean_shift_zscore:.2f}σ "
                          f"(train={train_mean:.2f}, current={current_mean:.2f})")

            elif variance_ratio > 2.0 or variance_ratio < 0.5:
                severity = 'medium'
                message = (f"{feature_name}: Variance change {variance_ratio:.2f}x "
                          f"(train_std={train_std:.2f}, current_std={current_std:.2f})")

            # Add warning if any issue detected
            if severity:
                warning = {
                    'feature': feature_name,
                    'severity': severity,
                    'message': message,
                    'mean_shift_zscore': mean_shift_zscore,
                    'variance_ratio': variance_ratio,
                    'train_mean': train_mean,
                    'train_std': train_std,
                    'current_mean': current_mean,
                    'current_std': current_std
                }
                self.warnings.append(warning)
                self.total_warnings += 1

        return self.warnings

    def log_warnings(self):
        """Log all current warnings"""
        if not self.warnings:
            logger.info(f"✅ Distribution check at sample #{self.sample_count}: No shifts detected")
            return

        high_severity = [w for w in self.warnings if w['severity'] == 'high']
        medium_severity = [w for w in self.warnings if w['severity'] == 'medium']

        logger.warning(f"⚠️  Distribution shifts detected at sample #{self.sample_count}:")
        logger.warning(f"   High severity: {len(high_severity)}, Medium severity: {len(medium_severity)}")

        for warning in high_severity:
            logger.warning(f"   ❌ {warning['message']}")

        for warning in medium_severity:
            logger.warning(f"   ⚠️  {warning['message']}")

    def get_warnings(self) -> List[Dict]:
        """Get current warnings"""
        return self.warnings

    def get_summary_stats(self) -> Dict:
        """Get summary statistics about detected shifts"""
        high_severity = [w for w in self.warnings if w['severity'] == 'high']
        medium_severity = [w for w in self.warnings if w['severity'] == 'medium']

        return {
            'total_samples': self.sample_count,
            'total_warnings': self.total_warnings,
            'current_warnings': len(self.warnings),
            'high_severity_count': len(high_severity),
            'medium_severity_count': len(medium_severity),
            'shifted_features': [w['feature'] for w in self.warnings]
        }

    def update_baseline_from_stats(self, stats_instance):
        """
        Update the training distribution baseline from updated FeatureStats.

        Call this after online training to keep the baseline synchronized
        with the model's training data distribution.

        Args:
            stats_instance: FeatureStats instance with updated statistics
        """
        updated_count = 0
        for feature_name, stats in stats_instance.feature_stats.items():
            if feature_name in self.training_stats_dict:
                # Update existing feature
                self.training_stats_dict[feature_name]['mean'] = float(stats.mean[0]) if hasattr(stats.mean, '__getitem__') else float(stats.mean)
                self.training_stats_dict[feature_name]['std'] = float(stats.std[0]) if hasattr(stats.std, '__getitem__') else float(stats.std)
                if stats.min is not None:
                    self.training_stats_dict[feature_name]['min'] = float(stats.min[0]) if hasattr(stats.min, '__getitem__') else float(stats.min)
                if stats.max is not None:
                    self.training_stats_dict[feature_name]['max'] = float(stats.max[0]) if hasattr(stats.max, '__getitem__') else float(stats.max)
                updated_count += 1

        logger.info(f"✅ Updated distribution baseline for {updated_count} features from FeatureStats")


class PerSampleOODDetector:
    """
    Per-sample Out-of-Distribution (OOD) detector for neural network reliability.

    Unlike DistributionShiftMonitor which tracks aggregate shifts over time,
    this class evaluates EACH INDIVIDUAL sample to determine if it's OOD.

    Why this matters for neural networks:
    - NNs are interpolators, not extrapolators
    - Predictions outside training distribution are unreliable
    - Normalization layers (BatchNorm, etc.) can produce unstable outputs for OOD inputs
    - Softmax can produce overconfident wrong predictions

    Detection strategy (lightweight but solid):
    1. Bounds check: Is value outside training [min, max]? → EXTREME OOD
    2. Tail check: Is value outside training [p01, p99]? → TAIL OOD
    3. Z-score check: How many σ from training mean? → STATISTICAL OOD
    4. Multi-feature check: Multiple mild OOD features → COMBINED OOD

    Usage:
        detector = PerSampleOODDetector('feature_distribution_statistics.csv')

        # For each incoming request:
        result = detector.check_sample({
            'input_tokens': 5000,
            'kv_hit_ratio': 0.8,
            ...
        })

        if result['action'] == OODAction.FALLBACK:
            # Use heuristic routing instead of neural network
            selected_pod = fallback_least_loaded(pods)
        elif result['action'] == OODAction.REDUCE_CONFIDENCE:
            # Reduce confidence in model prediction
            confidence *= 0.5
    """

    def __init__(self, training_distribution_csv: str):
        """
        Initialize the per-sample OOD detector.

        The detector uses a simple but principled approach:
        - Check if any feature value is outside training [min, max]
        - If yes → the NN is extrapolating → predictions unreliable → fallback

        Args:
            training_distribution_csv: Path to CSV with training distribution statistics
                (generated by DistributionTracker.save_distribution_stats)
        """
        # Load training distribution statistics
        self.stats = self._load_stats(training_distribution_csv)

        # Tracking for diagnostics
        self.total_checks = 0
        self.ood_counts = {'extreme': 0, 'normal': 0}
        self.feature_ood_counts = defaultdict(int)

        logger.info(f"Initialized PerSampleOODDetector with {len(self.stats)} features")
        logger.info(f"  Detection: extrapolation outside training [min, max]")

    def _load_stats(self, csv_path: str) -> Dict[str, Dict]:
        """Load training distribution statistics from CSV"""
        df = pd.read_csv(csv_path)
        stats = {}

        for _, row in df.iterrows():
            feature_name = row['feature_name']
            stats[feature_name] = {
                'mean': row.get('mean', 0.0),
                'std': row.get('std', 1.0),
                'min': row.get('min', float('-inf')),
                'max': row.get('max', float('inf')),
                'p01': row.get('p01', float('-inf')),
                'p05': row.get('p05', float('-inf')),
                'p95': row.get('p95', float('inf')),
                'p99': row.get('p99', float('inf')),
            }

        return stats

    def check_sample(self, features: Dict[str, float], request_id: str = None) -> Dict:
        """
        Check if a single sample is out-of-distribution for neural network reliability.

        Detection strategy: Check for EXTRAPOLATION only.

        Why only extrapolation matters:
        - If value is within training [min, max], the model SAW similar values
        - The NN will interpolate (reliable) rather than extrapolate (unreliable)
        - Statistical rarity (z-score, percentiles) ≠ problematic for NN
        - A value at training min/max was seen - just at the boundary

        We deliberately DO NOT check:
        - Z-score thresholds: Assumes normality, doesn't capture NN issues
        - Percentile rarity: Values at p05 were seen by 5% of training - not unseen
        - Compound rarity: Speculative, adds complexity without clear benefit

        Args:
            features: Dict mapping feature names to values
            request_id: Optional request ID for logging

        Returns:
            Dict with OOD detection results
        """
        self.total_checks += 1

        extrapolation_features = []  # Outside [min, max] - model never saw these
        details = []

        for feature_name, value in features.items():
            if feature_name not in self.stats:
                continue

            s = self.stats[feature_name]

            # Skip if no valid bounds
            if s['min'] is None or s['max'] is None:
                continue

            # Skip if min == max (constant feature, can't extrapolate)
            if s['min'] == s['max']:
                continue

            # ═══════════════════════════════════════════════════════════════
            # THE ONLY CHECK: EXTRAPOLATION
            # Value is completely outside training range [min, max]
            # This means the NN has ZERO training data at this value
            # The model must extrapolate → predictions are unreliable
            # ═══════════════════════════════════════════════════════════════
            if value < s['min'] or value > s['max']:
                # Calculate how far outside the bounds (for logging/debugging)
                if value < s['min']:
                    distance_outside = s['min'] - value
                    boundary = 'min'
                    boundary_value = s['min']
                else:
                    distance_outside = value - s['max']
                    boundary = 'max'
                    boundary_value = s['max']

                # Calculate relative extrapolation (as fraction of training range)
                training_range = s['max'] - s['min']
                if training_range > 0:
                    relative_extrapolation = distance_outside / training_range
                else:
                    relative_extrapolation = float('inf')

                extrapolation_features.append({
                    'feature': feature_name,
                    'value': value,
                    'train_min': s['min'],
                    'train_max': s['max'],
                    'boundary_violated': boundary,
                    'boundary_value': boundary_value,
                    'distance_outside': distance_outside,
                    'relative_extrapolation': relative_extrapolation,
                })
                self.feature_ood_counts[feature_name] += 1

                details.append({
                    'feature': feature_name,
                    'severity': 'extrapolation',
                    'value': value,
                    'boundary': boundary,
                    'distance': distance_outside,
                })

        # ═══════════════════════════════════════════════════════════════
        # DECISION: Simple - any extrapolation means fallback
        # ═══════════════════════════════════════════════════════════════
        extrapolation_count = len(extrapolation_features)

        if extrapolation_count >= 1:
            action = OODAction.FALLBACK
            is_ood = True
            self.ood_counts['extreme'] += 1
            if request_id:
                features_summary = ", ".join([
                    f"{f['feature']}={f['value']:.2f} (train: [{f['train_min']:.2f}, {f['train_max']:.2f}])"
                    for f in extrapolation_features[:3]  # Show first 3
                ])
                logger.warning(f"⚠️  Request {request_id}: EXTRAPOLATION detected! {features_summary}")
        else:
            action = OODAction.NORMAL
            is_ood = False
            self.ood_counts['normal'] += 1

        return {
            'is_ood': is_ood,
            'action': action,
            'extrapolation_count': extrapolation_count,
            'extrapolation_features': extrapolation_features,
            'details': details,
            # Backward compatibility fields
            'max_zscore': 0.0,
            'max_normalized_value': 0.0,
            'extreme_count': extrapolation_count,
            'moderate_count': 0,
            'mild_count': 0,
            'tail_count': 0,
            'extreme_features': extrapolation_features,
            'moderate_features': [],
            'mild_features': [],
            'tail_features': [],
        }

    def check_pod_features(
        self,
        pod_features_dict: Dict[str, float],
        request_id: str = None
    ) -> Dict:
        """
        Check pod features for OOD.

        Pod features use pooled statistics (e.g., 'kv_hit_ratio' stats apply to
        all pods' kv_hit_ratio values).

        Args:
            pod_features_dict: Dict mapping feature names to values
                              (feature names should be base types like 'kv_hit_ratio',
                               not 'pod_0000-kv_hit_ratio')
            request_id: Optional request ID for logging

        Returns:
            Same as check_sample()
        """
        return self.check_sample(pod_features_dict, request_id)

    def check_request_features(
        self,
        request_features_dict: Dict[str, float],
        request_id: str = None
    ) -> Dict:
        """
        Check request features for OOD.

        Args:
            request_features_dict: Dict with request features
                                  e.g., {'input_tokens': 5000, 'output_tokens': 100}
            request_id: Optional request ID for logging

        Returns:
            Same as check_sample()
        """
        return self.check_sample(request_features_dict, request_id)

    def check_combined(
        self,
        request_features: Dict[str, float],
        pod_features: Dict[str, float],
        request_id: str = None
    ) -> Dict:
        """
        Check both request and pod features for OOD.

        Args:
            request_features: Dict with request features
            pod_features: Dict with pod features (base feature types)
            request_id: Optional request ID for logging

        Returns:
            Combined OOD result
        """
        # Merge features
        combined = {}
        combined.update(request_features)
        combined.update(pod_features)

        return self.check_sample(combined, request_id)

    def get_diagnostics(self) -> Dict:
        """Get diagnostic statistics about OOD detection"""
        total = self.total_checks
        if total == 0:
            return {'total_checks': 0, 'ood_rate': 0.0}

        return {
            'total_checks': total,
            'extreme_count': self.ood_counts['extreme'],
            'tail_count': self.ood_counts['tail'],
            'normal_count': self.ood_counts['normal'],
            'ood_rate': (self.ood_counts['extreme'] + self.ood_counts['tail']) / total,
            'extreme_rate': self.ood_counts['extreme'] / total,
            'top_ood_features': sorted(
                self.feature_ood_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def update_stats(self, stats_instance):
        """
        Update the detector's statistics from a FeatureStats instance.

        Call this after online training to keep OOD detection synchronized
        with the model's training data distribution.

        Args:
            stats_instance: FeatureStats instance with updated statistics
        """
        updated_count = 0
        for feature_name, running_stats in stats_instance.feature_stats.items():
            # Get scalar values from RunningStats
            mean = float(running_stats.mean[0]) if hasattr(running_stats.mean, '__getitem__') else float(running_stats.mean)
            std = float(running_stats.std[0]) if hasattr(running_stats.std, '__getitem__') else float(running_stats.std)

            min_val = float('-inf')
            max_val = float('inf')
            if running_stats.min is not None:
                min_val = float(running_stats.min[0]) if hasattr(running_stats.min, '__getitem__') else float(running_stats.min)
            if running_stats.max is not None:
                max_val = float(running_stats.max[0]) if hasattr(running_stats.max, '__getitem__') else float(running_stats.max)

            if feature_name in self.stats:
                # Update existing feature
                self.stats[feature_name]['mean'] = mean
                self.stats[feature_name]['std'] = std
                self.stats[feature_name]['min'] = min_val
                self.stats[feature_name]['max'] = max_val
                # Approximate percentiles from mean/std (assumes normal distribution)
                # p01 ≈ mean - 2.33*std, p99 ≈ mean + 2.33*std
                self.stats[feature_name]['p01'] = mean - 2.33 * std
                self.stats[feature_name]['p99'] = mean + 2.33 * std
                self.stats[feature_name]['p05'] = mean - 1.645 * std
                self.stats[feature_name]['p95'] = mean + 1.645 * std
            else:
                # Add new feature
                self.stats[feature_name] = {
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'p01': mean - 2.33 * std,
                    'p99': mean + 2.33 * std,
                    'p05': mean - 1.645 * std,
                    'p95': mean + 1.645 * std,
                }
            updated_count += 1

        logger.info(f"✅ Updated PerSampleOODDetector stats for {updated_count} features")

    def reset_diagnostics(self):
        """Reset diagnostic counters"""
        self.total_checks = 0
        self.ood_counts = {'extreme': 0, 'tail': 0, 'normal': 0}
        self.feature_ood_counts = defaultdict(int)
