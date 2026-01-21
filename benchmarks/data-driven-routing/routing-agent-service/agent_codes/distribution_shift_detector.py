#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


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
            logger.error(f"   ❌ {warning['message']}")

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
