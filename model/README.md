# MetaDrive RL Training — Autonomous Driving Agent

Train a PPO agent to drive autonomously using [MetaDrive](https://github.com/metadriverse/metadrive), a fast, lightweight simulator that generates infinite procedural driving scenarios.

---

## 📁 Files

| File | Purpose |
|---|---|
| `train_metadrive_ppo.py` | **Main training script** – PPO with Lidar obs (MLP policy) |
| `custom_env.py` | Custom env with comfort reward shaping + CNN image policy |
| `requirements.txt` | All Python dependencies |

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install metadrive-simulator stable-baselines3[extra] tensorboard
```

> MetaDrive works on **Linux**, **Windows**, and **macOS** (Linux recommended for headless servers).  
> Runs **headlessly** (no display needed) during training.

### 2. Train the agent

```bash
python train_metadrive_ppo.py
```

Training runs for **2M timesteps** using **4 parallel environments**.  
Expect ~30–60 minutes on a modern CPU; much faster with more cores.

### 3. Monitor with TensorBoard

```bash
tensorboard --logdir ./logs
```

Open http://localhost:6006 to watch reward curves live.

### 4. Evaluate the best model

```bash
python train_metadrive_ppo.py --eval
```

### 5. Watch the agent drive (requires a display)

```bash
python train_metadrive_ppo.py --eval --render
```

---

## 🏗️ Architecture

```
MetaDriveEnv (procedural maps, lidar obs)
        │
        ▼
SubprocVecEnv  ×4 parallel workers
        │
        ▼
PPO (MlpPolicy, 64×64 hidden layers)
  ├── Actor  → [steering, throttle/brake]  (continuous)
  └── Critic → value estimate
```

**Observation**: 240-dim Lidar vector (distances to surrounding objects/road edges)  
**Action space**: `[steering ∈ [-1,1], throttle_brake ∈ [-1,1]]`

---

## 🔧 Key Configuration Knobs

Edit these in `train_metadrive_ppo.py`:

| Parameter | Default | Effect |
|---|---|---|
| `num_scenarios` | 1000 | More → better generalisation |
| `traffic_density` | 0.1 | 0 = empty road, 1 = heavy traffic |
| `map` | `"SSSSSSSS"` | Road layout (S=straight, C=curve, X=intersection, r=roundabout) |
| `N_ENVS` | 4 | Parallel workers (set to CPU count) |
| `TOTAL_TIMESTEPS` | 2M | Training budget |
| `horizon` | 1000 | Max steps per episode |

---

## 🗺️ Environment Types

MetaDrive ships four RL environment categories:

| Env | Class | Description |
|---|---|---|
| **Generalisation** | `MetaDriveEnv` | Single agent, diverse procedural maps |
| **Safe RL** | `SafeMetaDriveEnv` | Adds obstacle cost signal |
| **Multi-agent** | `MultiAgentRoundaboutEnv` etc. | Cooperative/competitive traffic |
| **Real-world** | `ScenarioEnv` | Replays Waymo / nuScenes logs |

To switch environment, change the import in `train_metadrive_ppo.py`:
```python
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
```

---

## 🖼️ Using Image Observations (CNN Policy)

```python
from custom_env import train_cnn_policy
train_cnn_policy()
```

This switches to RGB camera observations and trains with `CnnPolicy`.  
GPU strongly recommended.

---

## 📈 Expected Results

| Timesteps | Typical success rate |
|---|---|
| 200k | ~10–20% |
| 500k | ~30–50% |
| 1M | ~50–70% |
| 2M | ~70–85% |

*Success = agent reaches the destination without crashing or leaving the road.*

---

## 🧩 Tips

- **More envs = faster training**: set `N_ENVS` equal to your CPU core count.
- **Curriculum**: start with `map="SSSSSSSS"` (straight roads), then increase map complexity.
- **Reward shaping**: increase `success_reward` if the agent doesn't explore enough; decrease `crash_vehicle_penalty` if training is too slow.
- **Multi-agent**: once the single-agent policy converges, try `MultiAgentPGMAEnv` for emergent social driving behaviour.
