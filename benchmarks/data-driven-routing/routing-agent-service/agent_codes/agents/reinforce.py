
import torch.nn as nn

from typing import Type, Union
from gymnasium import spaces

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv, Schedule

## XXX: This solution is not used because it takes too long to train.
class Reinforce(OnPolicyAlgorithm):
    """
    Monte Carlo Policy Gradient (REINFORCE) using SB3's infrastructure.
    Uses ActorCriticPolicy for convenience but ignores the critic/value loss.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        vf_coef: float = 0.0,              # ignored
        max_grad_norm: float = 0.5,
        use_gae: bool = False,             # must be False to get Monte Carlo returns
        gae_lambda: float = 1.0,           # irrelevant when use_gae=False
        **kwargs,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            use_sde=False,
            sde_sample_freq=-1,
            ent_coef=ent_coef,
            vf_coef=0.0,                   # do not use value loss
            max_grad_norm=max_grad_norm,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            **kwargs,
        )

    def train(self) -> None:
        # Standard SB3 on-policy training loop, but with REINFORCE loss.
        self.policy.train()
        # We’ll treat buffer.advantages as G_t (returns) and ignore values
        advantages = self.rollout_buffer.advantages
        returns = advantages  # identical in our setup

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Mini-batch updates (SB3 handles shuffling)
        entropy_losses, pg_losses, clip_fractions = [], [], []
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Get distribution and log probs from current policy
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                distribution = self.policy.get_distribution(rollout_data.observations)
                log_prob = distribution.log_prob(actions)
                # Sum/mean for multidimensional actions
                if log_prob.dim() > 1:
                    log_prob = log_prob.sum(dim=1)

                # REINFORCE objective: maximize E[G_t * log pi(a|s)]
                # -> minimize -(G_t * log pi(a|s))
                advantage_mb = rollout_data.advantages
                policy_loss = -(advantage_mb * log_prob).mean()

                # Optional entropy bonus (unchanged)
                entropy_loss = -distribution.entropy().mean()

                loss = policy_loss + self.ent_coef * entropy_loss
                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                entropy_losses.append(-entropy_loss.item())
                pg_losses.append(policy_loss.item())

        self._n_updates += self.n_epochs
        self.logger.record("train/entropy_loss", sum(entropy_losses) / len(entropy_losses))
        self.logger.record("train/policy_gradient_loss", sum(pg_losses) / len(pg_losses))
        self.logger.record("train/approx_kl", 0.0)
        self.logger.record("train/clip_fraction", 0.0)
        self.logger.record("train/loss", sum(pg_losses) / len(pg_losses))

    def collect_rollouts(
        self,
        env: VecEnv,
        callback,
        rollout_buffer,
        n_rollout_steps: int
    ) -> bool:
        """
        We reuse SB3’s collector but:
          - disable GAE
          - set last_values = 0 so returns are pure Monte Carlo on episode ends
          - treat truncations as terminals (so no bootstrapping on time limit)
        """
        # Use parent implementation
        success = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if not success:
            return False

        # Overwrite values with zeros and make advantages = returns computed without bootstrapping.
        # Because we ran with use_gae=False, SB3 already computed returns = discounted sum of rewards
        # using last_value only when episode not done. We zero that out to get pure MC.
        rollout_buffer.values[:] = 0.0

        # Recompute advantages as returns (G_t)
        # SB3 stored returns in rollout_buffer.returns when use_gae=False
        # (but it also stores advantages = returns - values). Since we set values=0, advantages = returns.
        rollout_buffer.advantages = rollout_buffer.returns.copy()

        return True


# ---- Usage example ----
if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    model = Reinforce(
        policy=ActorCriticPolicy,   # we’ll ignore the critic head
        env=env,
        learning_rate=1e-3,
        gamma=0.99,
        ent_coef=0.01,              # try 0.0 if you want pure REINFORCE
        n_steps=2048,               # collect ~one or more full episodes per update
        batch_size=64,
        n_epochs=4,
        seed=0,
        verbose=1,
    )

    model.learn(total_timesteps=100_000)
    model.save("reinforce_cartpole")
