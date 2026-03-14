"""
MetaDrive Autonomous Driving - PPO Training Script
====================================================
Compatible with Python 3.7, MetaDrive, stable-baselines3==2.0.0

Install:
    pip install metadrive-simulator stable-baselines3==2.0.0 tensorboard==2.11.2

Usage:
    python train_metadrive_ppo.py                        # Train
    python train_metadrive_ppo.py --eval                 # Evaluate best model
    python train_metadrive_ppo.py --eval --render        # Watch agent drive
    tensorboard --logdir ./logs                          # Monitor training
"""

import argparse
import os
import numpy as np
from typing import Callable

from metadrive.envs import MetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_CONFIG = dict(
    num_scenarios=1000,
    start_seed=0,
    map="SSSSSSSS",          # 8 straight blocks – easy start
    traffic_density=0.1,
    use_render=False,
    success_reward=10.0,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,
    driving_reward=1.0,
    speed_reward=0.1,
    crash_vehicle_done=True,
    crash_object_done=False,
    out_of_road_done=True,
    horizon=1000,
)

EVAL_CONFIG = dict(TRAIN_CONFIG)
EVAL_CONFIG.update(
    num_scenarios=100,
    start_seed=1000,          # held-out scenarios
    use_render=False,
)

PPO_HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="./logs",
)

N_ENVS            = 4
TOTAL_TIMESTEPS   = 2_000_000
EVAL_FREQ         = 50_000
SAVE_FREQ         = 100_000
MODEL_PATH        = "./models/ppo_metadrive"
BEST_MODEL_PATH   = "./models/best_ppo_metadrive"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers – unwrap gym API differences
# ──────────────────────────────────────────────────────────────────────────────

def _reset(env):
    """Reset env, handling both old (obs) and new (obs, info) return styles."""
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _step(env, action):
    """Step env, normalising to (obs, reward, done, info)."""
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    return obs, reward, done, info


# ──────────────────────────────────────────────────────────────────────────────
# Environment factories
# ──────────────────────────────────────────────────────────────────────────────

def make_train_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        cfg = dict(TRAIN_CONFIG)
        cfg["start_seed"] = seed + rank * 100
        env = MetaDriveEnv(cfg)
        env = Monitor(env)
        return env
    return _init


def make_eval_env(render: bool = False) -> MetaDriveEnv:
    cfg = dict(EVAL_CONFIG)
    if render:
        cfg["use_render"] = True
    env = MetaDriveEnv(cfg)
    env = Monitor(env)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs",   exist_ok=True)

    print(f"\n{'='*60}")
    print("  MetaDrive PPO Training")
    print(f"{'='*60}")
    print(f"  Parallel envs  : {N_ENVS}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"{'='*60}\n")

    train_env = SubprocVecEnv([make_train_env(rank=i) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env)

    eval_env = make_eval_env(render=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_PATH,
        log_path="./logs/eval",
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // N_ENVS, 1),
        save_path="./models/checkpoints",
        name_prefix="ppo_metadrive",
        verbose=1,
    )

    model = PPO("MlpPolicy", train_env, **PPO_HYPERPARAMS)

    print("Starting training …\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([eval_callback, checkpoint_callback]),
        progress_bar=True,
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}.zip")

    train_env.close()
    eval_env.close()


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model_path: str = BEST_MODEL_PATH + "/best_model",
             n_episodes: int = 20,
             render: bool = False):
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)
    env   = make_eval_env(render=render)

    episode_rewards, episode_lengths = [], []
    success_count = 0

    for ep in range(n_episodes):
        obs   = _reset(env)
        done  = False
        total = 0.0
        steps = 0
        info  = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = _step(env, action)
            total += reward
            steps += 1

        episode_rewards.append(total)
        episode_lengths.append(steps)
        arrived = info.get("arrive_dest", False)
        if arrived:
            success_count += 1

        print(f"  Ep {ep+1:3d}/{n_episodes} | "
              f"Reward {total:7.2f} | Steps {steps:5d} | "
              f"{'✓ arrived' if arrived else '✗'}")

    env.close()
    print(f"\n{'─'*50}")
    print(f"  Mean reward  : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length  : {np.mean(episode_lengths):.0f} steps")
    print(f"  Success rate : {success_count}/{n_episodes} "
          f"({100*success_count/n_episodes:.1f}%)")
    print(f"{'─'*50}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",     action="store_true")
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--model",    type=str, default=BEST_MODEL_PATH + "/best_model")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    if args.eval or args.render:
        evaluate(model_path=args.model, n_episodes=args.episodes, render=args.render)
    else:
        train()