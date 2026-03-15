# Behavior Cloning Implementation for G1 Walking

This repository contains a **behavior cloning (BC)** baseline for **Unitree G1 walking** in **MuJoCo Playground**.

The main goal is to provide a **non-diffusion imitation learning baseline** using demonstrations generated from a pretrained **PPO walking policy**, so that it can later be compared against diffusion-based imitation methods.

---

## Overview

The current pipeline is:

1. Load a pretrained **PPO walking policy** for G1
2. Collect expert demonstrations as **observation-action pairs**
3. Train a **behavior cloning** policy using supervised learning
4. Evaluate the learned BC policy in the same environment
5. Compare **PPO vs BC** qualitatively and quantitatively

The current BC policy successfully completes the full **1000-step / 20-second** rollout horizon and is competitive with PPO on the tested walking command.

---

## Repository Structure

```text
behavior_cloning_implementation
├── README.md
├── requirements.txt
├── .gitattributes
├── checkpoints
│   ├── bc_g1
│   │   ├── bc_g1_best.pt
│   │   ├── train_config.json
│   │   └── train_history.json
│   └── ppo_g1
│       └── g1_walking_policy_v3
│           └── 000000000001
├── data
│   ├── demonstrations
│   │   └── g1_walking
│   │       ├── demos.npy
│   │       └── meta.npy
│   └── huggingface
│       └── G1_WB_Dex5_Pickup_Pillow.parquet
├── notebooks
│   ├── g1_data_exploration.ipynb
│   └── g1_policy_rollout.ipynb
├── results
│   └── ppo_vs_bc_metrics_v2.json
├── scripts
│   ├── collect_ppo_dataset.py
│   ├── train_bc.py
│   ├── eval_bc_viewer.py
│   ├── eval_ppo_video.py
│   ├── compare_rollouts.py
│   ├── compare_rollouts_v2.py
│   ├── parquet_player.py
│   └── policy_rollout_viewer.py
└── videos
    ├── bc_g1_rollout.mp4
    └── ppo_g1_rollout.mp4
````

---

## Method

### Expert policy

A pretrained **PPO locomotion policy** is used as the expert demonstrator in the `G1JoystickFlatTerrain` MuJoCo Playground environment.

### Dataset collection

The PPO expert is rolled out to collect:

* observation: `state.obs["state"]` with dimension **103**
* action: PPO control output `ctrl` with dimension **29**

These are saved as expert demonstrations in:

```text
data/demonstrations/g1_walking/
```

### Behavior cloning

The BC policy is trained offline as a supervised learning problem:

* **input:** observation
* **target:** expert action
* **loss:** mean squared error (MSE)

The current implementation uses a multilayer perceptron (MLP) policy.

### Evaluation

The learned BC policy is compared against PPO using:

* rollout survival
* forward progress
* lateral drift
* heading deviation
* torso-height stability
* action smoothness

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Git LFS

Large files such as checkpoints, `.npy` demonstration files, `.parquet` files, and `.mp4` videos should be tracked with Git LFS.

Install Git LFS:

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows
choco install git-lfs
```

Then initialize it:

```bash
git lfs install
```

### 4. Register the virtual environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name=venv
```

---

## Usage

### Run the PPO expert viewer

```bash
python scripts/policy_rollout_viewer.py \
  -p ./checkpoints/ppo_g1/g1_walking_policy_v3/000000000001
```

### Collect PPO expert demonstrations

```bash
python scripts/collect_ppo_dataset.py \
  --policy-path ./checkpoints/ppo_g1/g1_walking_policy_v3/000000000001 \
  --save-dir ./data/demonstrations/g1_walking \
  --n-episodes 200
```

### Train the BC policy

```bash
python scripts/train_bc.py \
  --demo-path ./data/demonstrations/g1_walking/demos.npy \
  --meta-path ./data/demonstrations/g1_walking/meta.npy \
  --save-dir ./checkpoints/bc_g1
```

### Evaluate the BC policy and save rollout video

```bash
python scripts/eval_bc_viewer.py \
  --ckpt-path ./checkpoints/bc_g1/bc_g1_best.pt \
  --save-video ./videos/bc_g1_rollout.mp4
```

### Evaluate the PPO policy and save rollout video

```bash
python scripts/eval_ppo_video.py \
  --policy-path ./checkpoints/ppo_g1/g1_walking_policy_v3/000000000001 \
  --save-video ./videos/ppo_g1_rollout.mp4
```

### Compare PPO and BC quantitatively

```bash
python scripts/compare_rollouts_v2.py \
  --ppo-path ./checkpoints/ppo_g1/g1_walking_policy_v3/000000000001 \
  --bc-path ./checkpoints/bc_g1/bc_g1_best.pt \
  --max-steps 1000 \
  --save-json ./results/ppo_vs_bc_metrics_v2.json
```

---

## Dataset and Training Summary

Current dataset:

* requested rollouts: **200**
* usable filtered rollouts: **199**
* training split: **179 rollouts**
* validation split: **20 rollouts**

Current dataset dimensions:

* observation dimension: **103**
* action dimension: **29**

Current evaluation horizon:

* **1000 steps**
* **20 seconds**

---

## Current Result

Under the current straight-walking command evaluation:

* **PPO** and **BC** both complete the full **1000-step / 20-second** rollout
* **BC** is highly competitive with PPO
* BC and PPO are very close in forward-progress and locomotion metrics
* depending on the rollout, BC may show:

  * slightly different lateral / heading behavior
  * slightly smoother actions than PPO
* PPO may retain a small advantage in some posture-stability metrics
* overall, BC serves as a strong **non-diffusion baseline**

The saved quantitative comparison is in:

```text
results/ppo_vs_bc_metrics_v2.json
```

The rollout videos are in:

```text
videos/ppo_g1_rollout.mp4
videos/bc_g1_rollout.mp4
```

---

## Notebooks

### `notebooks/g1_data_exploration.ipynb`

Used to inspect:

* dataset columns
* state/action dimensions
* G1 robot joint structure
* MuJoCo `qpos` interpretation

### `notebooks/g1_policy_rollout.ipynb`

Used to:

* load the PPO checkpoint
* run a rollout in notebook form
* render policy behavior as video

---

## Notes

* The current comparison is based on a **single fixed forward-walking command**
* A stronger future evaluation should test multiple commands, such as:

  * slower forward walking
  * diagonal walking
  * turning
* This repository currently focuses on the **behavior cloning baseline**
* Diffusion-based training is expected to be handled separately

---

## Running the notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Then select the `venv` kernel in the notebook interface.

---

## Acknowledgement

This implementation uses:

* **MuJoCo Playground** for simulation environments
* **Brax PPO checkpoint loading** for expert policy reuse
* **PyTorch** for behavior cloning training
