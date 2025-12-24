#!/usr/bin/env python3

import os
import json
import argparse


# RL_MODEL_HYPERPARAMETERS = {
#     'eval_interval': 10,
#     'entropy_bonus_factor': 0.02,
#     'per_learn_reward_normalization': False,
#     'normalization': {
#         "SIGNAL_AMPLIFICATION_DEGREE": 1.0,  # 1.5
#         "REWARD_AMPLIFICATION_DEGREE": 1.0,
#         "REWARD_AMPLIFICATION_THRESHOLD": 0.5,
#         "STD_THRESHOLD_FOR_REQ_FEAT_NORMALIZATION": 0.1,
#         "STD_THRESHOLD_FOR_POD_FEAT_NORMALIZATION": 0.1,
#         "FEATURES_NORMALIZED": set(),
#         "NUM_FEATURES_NORMALIZED": 0,
#         "FEATURE_AMPLIFICATION": False,
#         "FEATURES_AMPLIFIED": set(),
#         "NUM_FEATURES_AMPLIFIED": 0,
#     },
#     'dataset_analysis': None,
#     'deterministic_training': True,
#     'training_seed': 42,
#     'REWARD_FUNCTION': 'linear_simple',
#     'TTFT_SLO': 1000,  # Default TTFT SLO threshold (ms)
#     'AVG_TPOT_SLO': 50,  # Default average TPOT SLO threshold (ms)
#     'TTFT_REWARD_WEIGHT': 0.5,
#     # 'lr_scheduler_type': 'exponential',
#     'lr_scheduler_type': 'constant',
#     'lr_scheduler_gamma': 0.95,
#     'OFFLINE_LEARNING_RATE': 0.001,
#     # 'ONLINE_LEARNING_RATE': 0.0005,
#     'EXCLUDED_POD_FEATURES': [],
#     'NO_NORMALIZE_FEATURES': [],
    
#     # Learning rate scheduling options
#     'lr_scheduler_gamma': 0.95,  # For exponential scheduler
    
#     # Model type selection
#     'MODEL_TYPE': 'contextual_bandit',  # 'contextual_bandit', 'latency_predictor'
#     'LATENCY_METRIC': 'ttft',  # 'ttft', 'avg_tpot', 'e2e_latency' (for latency_predictor)
# }


def main():
    parser = argparse.ArgumentParser(description='Write hyperparameters JSON (single source of truth)')
    parser.add_argument('--output', required=True, help='Path to write model_config.json')
    parser.add_argument('--ttft_slo', type=float, default=None)
    parser.add_argument('--avg_tpot_slo', type=float, default=None)
    parser.add_argument('--ttft_reward_weight', type=float, default=None)
    parser.add_argument('--reward_function', type=str, default=None)
    parser.add_argument('--offline_learning_rate', type=float, default=None)
    parser.add_argument('--excluded_pod_features', type=str, default='', help='Comma-separated pod features to exclude')
    parser.add_argument('--excluded_request_features', type=str, default='', help='Comma-separated request features to exclude')
    parser.add_argument('--no_normalize_features', type=str, default='', help='Comma-separated features to not normalize')
    parser.add_argument('--lr_scheduler_type', type=str, default=None, choices=['plateau', 'exponential', 'gradient_adaptive', 'constant'], help='Learning rate scheduler type')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=None, help='Gamma for exponential scheduler')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--reward_decay_factor', type=float, default=0.9, help='reward_decay_factor (lambda)')
    parser.add_argument('--model_type', type=str, default=None, help='Model type to use')
    parser.add_argument('--test_size_ratio', type=float, default=0.2, help='Test size ratio')
    # RL (SB3 PPO) specific hyperparameters (optional)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='PPO learning rate')
    parser.add_argument('--n_steps', type=int, default=32, help='PPO n_steps')
    parser.add_argument('--n_epochs', type=int, default=10, help='PPO n_epochs')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='PPO gae_lambda')
    parser.add_argument('--clip_range', type=float, default=0.2, help='PPO clip_range')
    parser.add_argument('--entropy_coeff', type=float, default=0.02, help='PPO entropy coefficient (ent_coef)')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='PPO value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='PPO max_grad_norm')
    parser.add_argument('--rl_checkpoint_path', type=str, default='', help='Path to PPO checkpoint to load at init')
    parser.add_argument('--freeze_transferred_weights', action='store_true', help='Freeze transferred contextual bandit weights')
    parser.add_argument('--latency_metric', type=str, default=None, choices=['ttft', 'avg_tpot', 'e2e_latency'], help='Latency metric for latency_predictor model')
    parser.add_argument('--weight_initialization', type=str, default='xavier', choices=['xavier', 'kaiming', 'static'], help='Weight initialization for latency_predictor model')
    parser.add_argument('--training_epochs', type=int, default=50, help='Training epochs for latency_predictor model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for latency_predictor model')
    parser.add_argument('--include_gpu_features', type=int, default=0, help='Include GPU features (0=False, 1=True)')
    parser.add_argument('--training_seed', type=int, default=42, help='Training seed')
    
    args = parser.parse_args()

    excluded = [x.strip() for x in args.excluded_pod_features.split(',') if x.strip()]
    excluded_request_features = [x.strip() for x in args.excluded_request_features.split(',') if x.strip()]
    no_normalize_features = [x.strip() for x in args.no_normalize_features.split(',') if x.strip()]

    RL_MODEL_HYPERPARAMETERS = {}
    RL_MODEL_HYPERPARAMETERS['test_size_ratio'] = float(args.test_size_ratio) if args.test_size_ratio is not None else 0.2
    RL_MODEL_HYPERPARAMETERS['weight_initialization'] = args.weight_initialization or 'xavier'
    RL_MODEL_HYPERPARAMETERS['training_seed'] = int(args.training_seed) if args.training_seed is not None else 42
    RL_MODEL_HYPERPARAMETERS['TTFT_SLO'] = float(args.ttft_slo) if args.ttft_slo is not None else 1000.0
    RL_MODEL_HYPERPARAMETERS['AVG_TPOT_SLO'] = float(args.avg_tpot_slo) if args.avg_tpot_slo is not None else 50.0
    RL_MODEL_HYPERPARAMETERS['hidden_dim'] = int(args.hidden_dim) if args.hidden_dim is not None else 64
    RL_MODEL_HYPERPARAMETERS['TTFT_REWARD_WEIGHT'] = float(args.ttft_reward_weight) if args.ttft_reward_weight is not None else 1.0
    RL_MODEL_HYPERPARAMETERS['REWARD_FUNCTION'] = args.reward_function or 'linear_simple'
    RL_MODEL_HYPERPARAMETERS['OFFLINE_LEARNING_RATE'] = float(args.offline_learning_rate) if args.offline_learning_rate is not None else 0.001
    RL_MODEL_HYPERPARAMETERS['EXCLUDED_POD_FEATURES'] = excluded
    RL_MODEL_HYPERPARAMETERS['EXCLUDED_REQUEST_FEATURES'] = excluded_request_features
    RL_MODEL_HYPERPARAMETERS['lr_scheduler_type'] = args.lr_scheduler_type or 'constant'
    RL_MODEL_HYPERPARAMETERS['lr_scheduler_gamma'] = float(args.lr_scheduler_gamma) if args.lr_scheduler_gamma is not None else 0.95
    RL_MODEL_HYPERPARAMETERS['NO_NORMALIZE_FEATURES'] = no_normalize_features
    RL_MODEL_HYPERPARAMETERS['MODEL_TYPE'] = args.model_type or 'latency_predictor'
    RL_MODEL_HYPERPARAMETERS['learning_rate'] = float(args.learning_rate)
    RL_MODEL_HYPERPARAMETERS['n_steps'] = int(args.n_steps)
    RL_MODEL_HYPERPARAMETERS['n_epochs'] = int(args.n_epochs)
    RL_MODEL_HYPERPARAMETERS['gae_lambda'] = float(args.gae_lambda)
    RL_MODEL_HYPERPARAMETERS['clip_range'] = float(args.clip_range)
    RL_MODEL_HYPERPARAMETERS['entropy_coeff'] = float(args.entropy_coeff)
    RL_MODEL_HYPERPARAMETERS['vf_coef'] = float(args.vf_coef)
    RL_MODEL_HYPERPARAMETERS['max_grad_norm'] = float(args.max_grad_norm)
    RL_MODEL_HYPERPARAMETERS['RL_CHECKPOINT_PATH'] = args.rl_checkpoint_path
    RL_MODEL_HYPERPARAMETERS['LATENCY_METRIC'] = args.latency_metric
    RL_MODEL_HYPERPARAMETERS['reward_decay_factor'] = float(args.reward_decay_factor)
    RL_MODEL_HYPERPARAMETERS['training_epochs'] = int(args.training_epochs)
    RL_MODEL_HYPERPARAMETERS['batch_size'] = int(args.batch_size)
    RL_MODEL_HYPERPARAMETERS['INCLUDE_GPU_FEATURES'] = int(args.include_gpu_features)
        
    print(f"args.output: {args.output}")
    print(f"RL_MODEL_HYPERPARAMETERS: {RL_MODEL_HYPERPARAMETERS}")
    print("="*50)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(RL_MODEL_HYPERPARAMETERS, f, indent=4, default=str)
    print(f"Saved hyperparameters to {args.output}")


if __name__ == '__main__':
    main()


