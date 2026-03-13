# Imitation Learning

This folder contains code and resources for training diffusion-based imitation learning policies for the G1 Unitree robot, using demonstrations generated from a pretrained PPO policy.

## Folder Structure

```
/imitation_learning
    /data
        /demonstrations/      # Rollouts generated from the PPO policy
        /huggingface/         # Datasets downloaded from HuggingFace
        README.md
    /checkpoints
        /ppo_g1/              # Pretrained PPO checkpoint
        README.md
    /notebooks
        g1_data_exploration.ipynb     # G1 robot joints and data walkthrough
        g1_policy_rollout.ipynb     # Running and visualising the PPO policy
    /scripts
        parquet_player.py     # Visualise demonstration data
        policy_rollout_viewer.py     # Visualise policy rollout
    README.md
```

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

Parquet files are tracked via Git LFS. Install it for your platform before pulling data:

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows
choco install git-lfs
```

Then initialise:

```bash
git lfs install
```

### 4. Register the virtual environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name=venv
```

## Running Notebooks

Make sure the virtual environment is active, then launch Jupyter:

```bash
jupyter notebook
```

When a notebook opens, select **venv** as the kernel via **Kernel → Change Kernel → venv**.