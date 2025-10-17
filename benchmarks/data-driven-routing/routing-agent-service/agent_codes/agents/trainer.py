
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
from tqdm.auto import tqdm


class Trainer:
    """
    A drop-in manual training loop for dynamic Gym environments.
    Keeps SB3 logging, callbacks, and progress bar.
    """

    def __init__(self, model, env, log_dir="./logs", eval_env=None):
        self.model = model
        self.env = env
        self.eval_env = eval_env
        self.logger = configure(log_dir, ["stdout", "tensorboard"])
        self.model.set_logger(self.logger)
        self.total_steps = 0

        # === Setup callbacks (use any you like) ===
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=f"{log_dir}/checkpoints",
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        callbacks = [checkpoint_callback]

        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"{log_dir}/best_model",
                log_path=f"{log_dir}/results",
                eval_freq=5_000,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        self.callback = CallbackList(callbacks)
        self.callback.init_callback(self.model)  # must initialize manually

    def train(self, total_timesteps: int):
        obs, _ = self.env.reset()  # <-- FIXED here
        done = False
        episode_reward = 0.0

        pbar = tqdm(total=total_timesteps, desc="Training", unit="steps")

        for step in range(total_timesteps):
            # === 1. Predict ===
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = self.env.step(action)

            # === 2. Record reward ===
            episode_reward += reward
            self.model.logger.record("rollout/reward", reward)

            # === 3. Handle episode end ===
            if done:
                self.model.logger.record("rollout/episode_reward", episode_reward)
                obs, _ = self.env.reset()
                episode_reward = 0.0

            # === 4. Callbacks & progress ===
            continue_training = self.callback.on_step()
            if not continue_training:
                print("Training stopped early by callback.")
                break

            if step % 1000 == 0:
                self.model.logger.dump(step=step)

            pbar.update(1)

        pbar.close()
        self.callback.on_training_end()
        print("Training complete ✅")


    def save(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")
