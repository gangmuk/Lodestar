#!/usr/bin/env python3

import os
import json
import argparse


RL_MODEL_HYPERPARAMETERS = {
    'hidden_dim': 64, # 256,
    'batch_size': 64,
    # 'offline_learning_rate': 0.001,
    'ONLINE_LEARNING_RATE': 0.0005,
    'training_epochs': 5, # 5,
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
    'EXCLUDED_POD_FEATURES': [],
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
    args = parser.parse_args()

    excluded = [x.strip() for x in args.excluded_pod_features.split(',') if x.strip()]

    if args.ttft_slo:
        RL_MODEL_HYPERPARAMETERS['TTFT_SLO'] = float(args.ttft_slo)
    if args.avg_tpot_slo:
        RL_MODEL_HYPERPARAMETERS['AVG_TPOT_SLO'] = float(args.avg_tpot_slo)
    if args.ttft_reward_weight:
        RL_MODEL_HYPERPARAMETERS['TTFT_REWARD_WEIGHT'] = float(args.ttft_reward_weight)
    if args.reward_function:
        RL_MODEL_HYPERPARAMETERS['REWARD_FUNCTION'] = args.reward_function
    if args.offline_learning_rate:
        RL_MODEL_HYPERPARAMETERS['OFFLINE_LEARNING_RATE'] = float(args.offline_learning_rate)
    if args.excluded_pod_features:
        RL_MODEL_HYPERPARAMETERS['EXCLUDED_POD_FEATURES'] = excluded

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(RL_MODEL_HYPERPARAMETERS, f, indent=4, default=str)
    print(f"Saved hyperparameters to {args.output}")


if __name__ == '__main__':
    main()


