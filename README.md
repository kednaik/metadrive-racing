# MetaDrive Autonomous Driving - PPO Training

This project implements an autonomous driving agent using Reinforcement Learning (Proximal Policy Optimization - PPO) in the MetaDrive simulator.

## 🚀 Quick Start

### 1. Project Setup

Use the provided setup script to create a virtual environment and install all dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

To activate the virtual environment manually:
```bash
source venv/bin/activate
```

### 2. Training the Agent

To start training the PPO agent:

```bash
python model/train_metadrive_ppo.py
```

Training runs for 2M timesteps by default and saves models to `model/models/`.

### 3. Monitoring Training

You can monitor the training progress (rewards, loss, etc.) using TensorBoard:

```bash
tensorboard --logdir model/logs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

### 4. Evaluation

To evaluate the best trained model:

```bash
python model/train_metadrive_ppo.py --eval
```

To watch the agent drive (requires a graphical display):

```bash
python model/train_metadrive_ppo.py --eval --render
```

## 📁 Project Structure

```text
.
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── setup.sh             # Automated setup script
└── model/               # Model and training logic
    ├── train_metadrive_ppo.py  # Main training/eval script
    ├── custom_env.py           # Custom environment wrappers
    ├── logs/                   # Training logs for TensorBoard
    └── models/                 # Saved model checkpoints and final models
```

## 🛠️ Configuration

Key parameters like `num_scenarios`, `traffic_density`, and `TOTAL_TIMESTEPS` can be adjusted in `model/train_metadrive_ppo.py`.

## 📜 Requirements

- Python 3.7+
- MetaDrive Simulator
- Stable-Baselines3
- TensorBoard
