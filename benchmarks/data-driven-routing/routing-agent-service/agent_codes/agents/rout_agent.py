import pickle
import os

from logger import logger
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from policies import ActorCriticRoutingPolicy
from envs.rout_env import ScalableRoutingEnvironment 
from envs.wrappers import EpisodeLengthWrapper, EpisodeCounterWrapper
from envs.broker import RequestBroker
from envs.request_source_gateway import GatewayRequestSource

from .replay_buffer import PrioritizedReplayBuffer

BROKER = RequestBroker()
SOURCE = GatewayRequestSource(BROKER)

# ============================================================================
# Scalable RL Routing Agent
# ============================================================================

class ScalableRLRoutingAgent:

    def __init__(
        self, 
        num_pods: int,
        per_pod_dim: int = 11, 
        request_dim: int = 3, 
        max_pods: int = 100, 
        inference_mode: bool = False,
        rl: str = 'PPO',
        use_prioritized_replay: bool = False, 
        **hyperparameters
        ):
        """
        Args:
            per_pod_dim: Features per pod (pod_features + kv_hit_ratios)
            request_dim: Request feature dimensions
            max_pods: Maximum expected pods (for space allocation)
            hyperparameters: PPO and training hyperparameters
        """

        self.num_pods = num_pods
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.max_pods = max_pods # XXX: useless
        self.hyperparameters = hyperparameters
        
        # Create environment
        # self.env = ScalableRoutingEnvironment(per_pod_dim, request_dim, max_pods)
        self.env = self.make_env(hyperparameters.get('horizon', 1024))
        # self.env = ScalableRoutingEnvironment(num_pods, per_pod_dim, request_dim, max_pods)

        self.setup_model(rl, per_pod_dim, request_dim, hyperparameters)
        
        # === Prioritized Experience Replay ===
        if use_prioritized_replay:
            self.experience_buffer = PrioritizedReplayBuffer(
                maxlen=hyperparameters.get('buffer_size', 1000),
                alpha=hyperparameters.get('priority_alpha', 0.6),
                beta=hyperparameters.get('priority_beta', 0.4)
            )
        
        
        # === Training statistics ===
        self.total_steps = 0
        self.total_episodes = 0
    
        logger.info(f"ScalableRLRoutingAgent initialization complete")


    def make_env(self, horizon: int):
        env = ScalableRoutingEnvironment(
            num_pods=self.num_pods,
            num_requests=10_000_000_000_000, # effectively infinite; EpisodeLengthWrapper handles resets
            per_pod_dim=self.per_pod_dim,
            request_dim=self.request_dim,
            source = SOURCE,
        )
        env = Monitor(env)
        env = EpisodeLengthWrapper(env, horizon=horizon)
        env = EpisodeCounterWrapper(env)

        logger.info(f"Environment created with horizon {horizon}") ## XXX: change to steps to time

        return env
        

    def setup_model(self, rl: str, per_pod_dim: int, request_dim: int, hyperparameters: dict):
        # Extract hyperparameters
        learning_rate = hyperparameters.get('learning_rate', 3e-4)
        hidden_dim = hyperparameters.get('hidden_dim', 64)
        gamma = hyperparameters.get('reward_decay_factor', 1)
        gae_lambda = hyperparameters.get('gae_lambda', 0.95)

        if rl == 'PG':
            pass
        elif rl == 'PPO':
            # === Create PPO model with our scalable policy ===
            self.model = PPO(
                ActorCriticRoutingPolicy,
                self.env,
                learning_rate=learning_rate,
                n_steps=hyperparameters.get('n_steps', 256), # number of env steps per policy update, must less that horizon
                batch_size=hyperparameters.get('batch_size', 64),
                n_epochs=hyperparameters.get('n_epochs', 10),
                gamma=gamma,                    # Discount factor
                gae_lambda=gae_lambda,          # GAE lambda (short horizon)
                clip_range=hyperparameters.get('clip_range', 0.2),
                ent_coef=hyperparameters.get('entropy_coeff', 0.01),
                vf_coef=hyperparameters.get('vf_coef', 0.5),
                max_grad_norm=hyperparameters.get('max_grad_norm', 0.5),
                policy_kwargs={
                    'per_pod_dim': per_pod_dim,
                    'request_dim': request_dim,
                    'hidden_dim': hidden_dim,
                    'last_layer_dim_pi': hyperparameters.get('last_layer_dim_pi', 1),
                    'last_layer_dim_vf': hyperparameters.get('last_layer_dim_vf', 0),
                },
                verbose=1
            )
        else:
            raise ValueError(f"{rl} not supported")
    

    def train(self, total_timesteps: int, save_path: str):
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        self.total_steps = total_timesteps
        self.save(save_path) ## TODO: match


    def predict_sb3(self, pod_features, kv_hit_ratios, request_features):
        assert self.num_pods == pod_features.shape[0]
        
        # Build observation dict (this pads to max_pods internally)
        obs = build_observation(self.num_pods, pod_features, kv_hit_ratios, request_features)
        action, _ = self.model.predict(obs)

        return int(action)
    
    
    def save(self, path: str, save_buffer: bool = False):
        """
        Save model with comprehensive metadata for reproducibility and analysis.
        
        Args:
            path: Base path for checkpoint
            save_buffer: If True, also save experience buffer (large file)
        """
        import datetime
        
        # Save PPO model (weights, optimizer state, etc.)
        self.model.save(path)
        
        # Collect comprehensive metadata
        metadata = {
            # === Model Architecture ===
            'model_architecture': {
                'per_pod_dim': self.per_pod_dim,
                'request_dim': self.request_dim,
                'max_pods': self.max_pods,
                'hidden_dim': self.hyperparameters.get('hidden_dim', 64),
            },
            
            # === Training Hyperparameters ===
            'hyperparameters': self.hyperparameters,
            
            # # === Training Progress ===
            # 'training_progress': {
            #     'total_steps': self.total_steps,
            #     'total_episodes': self.total_episodes,
            #     'current_episode_id': self.episode_tracker.episode_id,
            #     'episode_request_count': self.episode_tracker.episode_request_count,
            # },
            
            # # === Buffer Statistics ===
            # 'buffer_stats': {
            #     'buffer_size': len(self.experience_buffer),
            #     'buffer_capacity': self.experience_buffer.buffer.maxlen,
            #     'pending_experiences': len(self.pending_experiences),
            #     'priority_alpha': self.experience_buffer.alpha,
            #     'priority_beta': self.experience_buffer.beta,
            #     'max_priority': self.experience_buffer.max_priority,
            # },
            
            # # === Episode Configuration ===
            # 'episode_config': {
            #     'episode_duration': self.episode_tracker.episode_duration,
            #     'episode_start_time': self.episode_tracker.episode_start_time,
            # },
            
            # # === Model Performance (if tracked) ===
            # 'performance_metrics': self.get_metrics(),
            
            # === Checkpoint Metadata ===
            'checkpoint_info': {
                'save_time': datetime.datetime.now().isoformat(),
                'save_path': path,
                'version': '1.0',
            },
            
            # === Environment Info ===
            'environment': {
                'observation_space': str(self.env.observation_space),
                'action_space': str(self.env.action_space),
            }
        }
        
        # Save metadata
        metadata_path = f"{path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save human-readable metadata (JSON)
        json_metadata_path = f"{path}_metadata.json"
        try:
            import json
            # Convert to JSON-serializable format
            json_metadata = {
                'model_architecture': metadata['model_architecture'],
                # 'training_progress': metadata['training_progress'],
                # 'buffer_stats': metadata['buffer_stats'],
                # 'episode_config': metadata['episode_config'],
                # 'performance_metrics': metadata['performance_metrics'],
                'checkpoint_info': metadata['checkpoint_info'],
            }
            with open(json_metadata_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
            logger.info(f"Saved human-readable metadata to {json_metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save JSON metadata: {e}")
        
        # # Optionally save experience buffer (can be large!)
        # if save_buffer and len(self.experience_buffer) > 0:
        #     buffer_path = f"{path}_buffer.pkl"
        #     try:
        #         with self.experience_buffer.lock:
        #             buffer_data = {
        #                 'experiences': list(self.experience_buffer.buffer),
        #                 'priorities': list(self.experience_buffer.priorities),
        #             }
        #         with open(buffer_path, 'wb') as f:
        #             pickle.dump(buffer_data, f)
        #         logger.info(f"Saved experience buffer to {buffer_path} ({len(buffer_data['experiences'])} experiences)")
        #     except Exception as e:
        #         logger.warning(f"Could not save buffer: {e}")
        
        logger.info(f"Model checkpoint saved to {path}")
        # logger.info(f"   Total steps: {self.total_steps}, Episodes: {self.total_episodes}")
        # logger.info(f"   Buffer size: {len(self.experience_buffer)}/{self.experience_buffer.buffer.maxlen}")
   
    
    def load(self, path: str, load_buffer: bool = False):
        """
        Load model with comprehensive metadata restoration.
        
        Args:
            path: Base path for checkpoint
            load_buffer: If True, also load experience buffer (if available)
        """
        try:
            # Load PPO model (weights, optimizer state)
            self.model = PPO.load(path, env=self.env)
            logger.info(f"✅ Loaded model from {path}")
            
            # Load metadata
            try:
                with open(f"{path}_metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                
                # # Restore training progress
                # training_progress = metadata.get('training_progress', {})
                # self.total_steps = training_progress.get('total_steps', 0)
                # self.total_episodes = training_progress.get('total_episodes', 0)
                
                # # Restore episode tracker state
                # episode_config = metadata.get('episode_config', {})
                # if 'episode_duration' in episode_config:
                #     self.episode_tracker.episode_duration = episode_config['episode_duration']
                
                # Store loaded metadata for inspection
                self.loaded_metadata = metadata
                
                # Log checkpoint info
                checkpoint_info = metadata.get('checkpoint_info', {})
                # buffer_stats = metadata.get('buffer_stats', {})
                
                logger.info(f"📊 Loaded checkpoint metadata:")
                logger.info(f"   - Created: {checkpoint_info.get('save_time', 'unknown')}")
                # logger.info(f"   - Total steps: {self.total_steps}")
                # logger.info(f"   - Total episodes: {self.total_episodes}")
                # logger.info(f"   - Buffer was at: {buffer_stats.get('buffer_size', 0)} experiences")
                
                # # Display performance metrics if available
                # perf_metrics = metadata.get('performance_metrics', {})
                # if perf_metrics:
                #     logger.info(f"   - Last avg reward: {perf_metrics.get('avg_reward_recent', 'N/A')}")
                #     logger.info(f"   - Success rate: {perf_metrics.get('success_rate', 'N/A')}")
                    
            except FileNotFoundError:
                logger.warning("Metadata file not found, using defaults")
                self.loaded_metadata = None
            
            # Optionally load experience buffer
            if load_buffer:
                buffer_path = f"{path}_buffer.pkl"
                if os.path.exists(buffer_path):
                    try:
                        with open(buffer_path, 'rb') as f:
                            buffer_data = pickle.load(f)
                        
                        # Restore buffer
                        with self.experience_buffer.lock:
                            for exp in buffer_data['experiences']:
                                self.experience_buffer.buffer.append(exp)
                            for priority in buffer_data['priorities']:
                                self.experience_buffer.priorities.append(priority)
                        
                        logger.info(f"Loaded {len(buffer_data['experiences'])} experiences from buffer")
                    except Exception as e:
                        logger.warning(f"Could not load buffer: {e}")
                else:
                    logger.info(f"No buffer file found (load_buffer=True but file missing)")
        
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    # def get_metrics(self):
    #     """
    #     Get comprehensive training and performance metrics.
        
    #     Returns:
    #         dict: Comprehensive metrics including training progress, performance, and model quality
    #     """
    #     import numpy as np
        
    #     # Basic training metrics
    #     metrics = {
    #         'total_steps': self.total_steps,
    #         'total_episodes': self.total_episodes,
    #         'buffer_size': len(self.experience_buffer),
    #         'pending_experiences': len(self.pending_experiences),
    #         'current_episode': self.episode_tracker.episode_id,
    #         'episode_request_count': self.episode_tracker.episode_request_count,
    #     }
        
    #     # Reward statistics (recent 100 and all)
    #     if len(self.reward_history) > 0:
    #         rewards = list(self.reward_history)
    #         recent_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            
    #         metrics['reward_stats'] = {
    #             'avg_reward_recent': float(np.mean(recent_100)),
    #             'std_reward_recent': float(np.std(recent_100)),
    #             'max_reward_recent': float(np.max(recent_100)),
    #             'min_reward_recent': float(np.min(recent_100)),
    #             'avg_reward_all': float(np.mean(rewards)),
    #             'num_samples': len(rewards),
    #         }
            
    #         # Success rate (reward > 0 means good routing decision)
    #         success_count = sum(1 for r in recent_100 if r > 0)
    #         metrics['success_rate'] = success_count / len(recent_100) if len(recent_100) > 0 else 0.0
    #     else:
    #         metrics['reward_stats'] = None
    #         metrics['success_rate'] = None
        
    #     # Decision quality metrics
    #     if len(self.recent_decisions) > 0:
    #         decisions = list(self.recent_decisions)
    #         confidences = [d['confidence'] for d in decisions]
    #         latencies = [d['latency_ms'] for d in decisions]
    #         rewards = [d['reward'] for d in decisions]
            
    #         metrics['decision_quality'] = {
    #             'avg_confidence': float(np.mean(confidences)),
    #             'avg_latency_ms': float(np.mean(latencies)),
    #             'p50_latency_ms': float(np.percentile(latencies, 50)),
    #             'p95_latency_ms': float(np.percentile(latencies, 95)),
    #             'p99_latency_ms': float(np.percentile(latencies, 99)),
    #             'high_confidence_success_rate': self._compute_high_confidence_success(decisions),
    #         }
    #     else:
    #         metrics['decision_quality'] = None
        
    #     # Learning progress (compare first 100 vs last 100 rewards)
    #     if len(self.reward_history) >= 200:
    #         rewards = list(self.reward_history)
    #         first_100 = rewards[:100]
    #         last_100 = rewards[-100:]
    #         improvement = np.mean(last_100) - np.mean(first_100)
    #         metrics['learning_progress'] = {
    #             'reward_improvement': float(improvement),
    #             'first_100_avg': float(np.mean(first_100)),
    #             'last_100_avg': float(np.mean(last_100)),
    #         }
    #     else:
    #         metrics['learning_progress'] = None
        
    #     return metrics
    
    def _compute_high_confidence_success(self, decisions, confidence_threshold=0.7):
        """
        Compute success rate for high-confidence decisions.
        This helps evaluate if the model is well-calibrated.
        """
        high_conf = [d for d in decisions if d['confidence'] >= confidence_threshold]
        if len(high_conf) == 0:
            return None
        success = sum(1 for d in high_conf if d['reward'] > 0)
        return success / len(high_conf)


