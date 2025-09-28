#!/usr/bin/env python3

import os
import json
import argparse


RL_MODEL_HYPERPARAMETERS = {
    'hidden_dim': 64, # 64, 128, 256
    'batch_size': 32,
    # 'offline_learning_rate': 0.001,
    'training_epochs': 15, # 5,
    'learning_every_x_iter': 5,
    'weight_decay': 0.0001,
    'max_updates_per_epoch': 1000, # 1000000000
    'exploration_rate': 0.1, # 0.1
    'explore': True,
    'weight_initialization': 'xavier', # 'kaiming', 'xavier', 'static'
    
    'eval_interval': 10,
    'entropy_bonus_factor': 0.02,
    'per_learn_reward_normalization': False,
    'normalization': {
        "SIGNAL_AMPLIFICATION_DEGREE": 1.0,  # 1.5
        "REWARD_AMPLIFICATION_DEGREE": 1.0,
        "REWARD_AMPLIFICATION_THRESHOLD": 0.5,
        "STD_THRESHOLD_FOR_REQ_FEAT_NORMALIZATION": 0.1,
        "STD_THRESHOLD_FOR_POD_FEAT_NORMALIZATION": 0.1,
        "FEATURES_NORMALIZED": set(),
        "NUM_FEATURES_NORMALIZED": 0,
        "FEATURE_AMPLIFICATION": False,
        "FEATURES_AMPLIFIED": set(),
        "NUM_FEATURES_AMPLIFIED": 0,
    },
    'dataset_analysis': None,
    'deterministic_training': True,
    'training_seed': 42,
    'REWARD_FUNCTION': 'linear_simple',
    'TTFT_SLO': 1000,  # Default TTFT SLO threshold (ms)
    'AVG_TPOT_SLO': 50,  # Default average TPOT SLO threshold (ms)
    'TTFT_REWARD_WEIGHT': 0.5,
    'OFFLINE_LEARNING_RATE': 0.001,
    'ONLINE_LEARNING_RATE': 0.0005,
    'EXCLUDED_POD_FEATURES': [],
    'NO_NORMALIZE_FEATURES': [],
    
    # Learning rate scheduling options
    'lr_scheduler_type': 'exponential',  # 'plateau', 'exponential', 'gradient_adaptive'
    'lr_scheduler_gamma': 0.95,  # For exponential scheduler
    
    # Model type selection
    'MODEL_TYPE': 'contextual_bandit',  # 'contextual_bandit', 'latency_predictor'
    'LATENCY_METRIC': 'ttft',  # 'ttft', 'avg_tpot', 'e2e_latency' (for latency_predictor)
}


def main():
    parser = argparse.ArgumentParser(description='Write hyperparameters JSON (single source of truth)')
    parser.add_argument('--output', required=True, help='Path to write model_config.json')
    parser.add_argument('--ttft_slo', type=float, default=None)
    parser.add_argument('--avg_tpot_slo', type=float, default=None)
    parser.add_argument('--ttft_reward_weight', type=float, default=None)
    parser.add_argument('--reward_function', type=str, default=None)
    parser.add_argument('--offline_learning_rate', type=float, default=None)
    parser.add_argument('--excluded_pod_features', type=str, default='', help='Comma-separated pod features to exclude')
    parser.add_argument('--no_normalize_features', type=str, default='', help='Comma-separated features to not normalize')
    parser.add_argument('--lr_scheduler_type', type=str, default=None, choices=['plateau', 'exponential', 'gradient_adaptive'], help='Learning rate scheduler type')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=None, help='Gamma for exponential scheduler')
    parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension')
    parser.add_argument('--reward_decay_factor', type=float, default=0.9, help='reward_decay_factor (lambda)')
    parser.add_argument('--model_type', type=str, default=None, choices=['contextual_bandit', 'latency_predictor'], help='Model type to use')
    parser.add_argument('--latency_metric', type=str, default=None, choices=['ttft', 'avg_tpot', 'e2e_latency'], help='Latency metric for latency_predictor model')
    
    args = parser.parse_args()

    excluded = [x.strip() for x in args.excluded_pod_features.split(',') if x.strip()]
    no_normalize_features = [x.strip() for x in args.no_normalize_features.split(',') if x.strip()]

    if args.ttft_slo:
        RL_MODEL_HYPERPARAMETERS['TTFT_SLO'] = float(args.ttft_slo)
    if args.avg_tpot_slo:
        RL_MODEL_HYPERPARAMETERS['AVG_TPOT_SLO'] = float(args.avg_tpot_slo)
    if args.hidden_dim:
        RL_MODEL_HYPERPARAMETERS['hidden_dim'] = int(args.hidden_dim)
    if args.ttft_reward_weight:
        RL_MODEL_HYPERPARAMETERS['TTFT_REWARD_WEIGHT'] = float(args.ttft_reward_weight)
    if args.reward_function:
        RL_MODEL_HYPERPARAMETERS['REWARD_FUNCTION'] = args.reward_function
    if args.offline_learning_rate:
        RL_MODEL_HYPERPARAMETERS['OFFLINE_LEARNING_RATE'] = float(args.offline_learning_rate)
    if args.excluded_pod_features:
        RL_MODEL_HYPERPARAMETERS['EXCLUDED_POD_FEATURES'] = excluded
    if args.lr_scheduler_type:
        RL_MODEL_HYPERPARAMETERS['lr_scheduler_type'] = args.lr_scheduler_type
    if args.lr_scheduler_gamma:
        RL_MODEL_HYPERPARAMETERS['lr_scheduler_gamma'] = float(args.lr_scheduler_gamma)
    if args.no_normalize_features:
        RL_MODEL_HYPERPARAMETERS['NO_NORMALIZE_FEATURES'] = no_normalize_features
    if args.model_type:
        RL_MODEL_HYPERPARAMETERS['MODEL_TYPE'] = args.model_type
    if args.latency_metric:
        RL_MODEL_HYPERPARAMETERS['LATENCY_METRIC'] = args.latency_metric
    if args.reward_decay_factor:
        RL_MODEL_HYPERPARAMETERS['reward_decay_factor'] = float(args.reward_decay_factor)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(RL_MODEL_HYPERPARAMETERS, f, indent=4, default=str)
    print(f"Saved hyperparameters to {args.output}")


if __name__ == '__main__':
    main()


