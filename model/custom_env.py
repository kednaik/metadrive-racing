"""
MetaDrive Custom Environment
==============================
Extends MetaDriveEnv with:
  - Custom reward shaping
  - Optional RGB camera observation (for CNN policy)
  - Comfort penalty (penalises jerky steering/acceleration)
  - Logging of extra episode metrics

Usage:
    from custom_env import CustomMetaDriveEnv
    env = CustomMetaDriveEnv({"use_image": False})
"""

import numpy as np
import gymnasium as gym
from metadrive.envs import MetaDriveEnv


class CustomMetaDriveEnv(MetaDriveEnv):
    """
    MetaDriveEnv with enhanced reward shaping and optional image observations.
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(dict(
            # ── Custom flags ──────────────────────────────────────────────────
            use_image=False,            # switch to RGB camera obs
            comfort_penalty_coef=0.1,  # penalise large steering/accel changes
            # ── Standard reward weights ───────────────────────────────────────
            driving_reward=1.0,
            speed_reward=0.1,
            success_reward=10.0,
            out_of_road_penalty=5.0,
            crash_vehicle_penalty=5.0,
            # ── Environment ───────────────────────────────────────────────────
            num_scenarios=1000,
            start_seed=0,
            traffic_density=0.1,
            horizon=1000,
            use_render=False,
            # ── Image obs ─────────────────────────────────────────────────────
            image_observation=False,   # set to True when use_image=True
            rgb_clip=True,
            norm_pixel=True,
        ))
        return config

    def __init__(self, config=None):
        cfg = config or {}
        if cfg.get("use_image", False):
            cfg["image_observation"] = True
        super().__init__(cfg)
        self._prev_action = np.zeros(2)
        self._episode_stats = {}

    # ── Reward ────────────────────────────────────────────────────────────────

    def reward_function(self, vehicle_id: str):
        """Add comfort penalty on top of base reward."""
        reward, reward_info = super().reward_function(vehicle_id)

        # Comfort penalty: punish jerky control
        vehicle = self.agents[vehicle_id]
        current_action = np.array([
            vehicle.steering,
            vehicle.throttle_brake,
        ])
        jerk = np.abs(current_action - self._prev_action).sum()
        comfort_penalty = self.config["comfort_penalty_coef"] * jerk
        self._prev_action = current_action

        reward -= comfort_penalty
        reward_info["comfort_penalty"] = -comfort_penalty
        return reward, reward_info

    # ── Episode bookkeeping ───────────────────────────────────────────────────

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._prev_action = np.zeros(2)
        self._episode_stats = {"steps": 0, "total_reward": 0.0}
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        self._episode_stats["steps"] += 1
        self._episode_stats["total_reward"] += reward

        if terminated or truncated:
            info.update({
                "episode_steps": self._episode_stats["steps"],
                "episode_reward": self._episode_stats["total_reward"],
            })
        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────────────
# CNN Training helper (image observations)
# ──────────────────────────────────────────────────────────────────────────────

def make_image_env():
    """Returns a MetaDrive env using RGB camera observations for a CNN policy."""
    return CustomMetaDriveEnv(dict(
        use_image=True,
        image_observation=True,
        num_scenarios=500,
        start_seed=0,
        traffic_density=0.1,
        use_render=False,
    ))


def train_cnn_policy():
    """
    Example: train PPO with CnnPolicy on image observations.
    Requires a GPU or more time than the MLP version.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    env = DummyVecEnv([make_image_env])
    env = VecTransposeImage(env)  # HWC → CHW for PyTorch

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/cnn",
    )

    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save("./models/ppo_metadrive_cnn")
    env.close()
    print("CNN model saved.")


if __name__ == "__main__":
    # Quick sanity-check: one episode with random actions
    env = CustomMetaDriveEnv()
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")

    total_r = 0.0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        if terminated or truncated:
            break

    env.close()
    print(f"Random episode reward: {total_r:.2f}")
    print(f"Arrive destination  : {info.get('arrive_dest', False)}")
