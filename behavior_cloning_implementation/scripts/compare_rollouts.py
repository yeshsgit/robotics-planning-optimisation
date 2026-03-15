import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import jax
import jax.numpy as jp
from mujoco_playground import registry
from brax.training.agents.ppo import checkpoint


ENV_NAME = "G1JoystickFlatTerrain"


class MLPPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantitatively compare PPO and BC on G1 walking.")
    parser.add_argument(
        "--ppo-path",
        type=str,
        default="./checkpoints/ppo_g1/g1_walking_policy_v3/000000000001",
    )
    parser.add_argument(
        "--bc-path",
        type=str,
        default="./checkpoints/bc_g1/bc_g1_best.pt",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="./results/ppo_vs_bc_metrics.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    return parser.parse_args()


def get_obs_vector(state) -> np.ndarray:
    if isinstance(state.obs, dict):
        if "state" in state.obs:
            return np.array(state.obs["state"], dtype=np.float32)
        raise KeyError(f"state.obs is dict but no 'state' key. Keys: {list(state.obs.keys())}")
    return np.array(state.obs, dtype=np.float32)


def build_bc_model(ckpt: dict, device: torch.device) -> nn.Module:
    model = MLPPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def bc_action(model, obs_vec, obs_mean, obs_std, act_mean, act_std, device):
    obs_norm = (obs_vec - obs_mean) / obs_std
    obs_tensor = torch.from_numpy(obs_norm.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        act_norm = model(obs_tensor).cpu().numpy()[0]
    act = act_norm * act_std + act_mean
    return act.astype(np.float32)


def get_yaw_from_xmat(xmat_row_major_9: np.ndarray) -> float:
    # xmat is flattened 3x3 rotation matrix in row-major
    # first column (world x-axis of body) can be recovered as [r00, r10, r20] if reshaped
    R = np.asarray(xmat_row_major_9).reshape(3, 3)
    x_axis = R[:, 0]
    yaw = -np.arctan2(x_axis[1], x_axis[0])
    return float(yaw)


def summarize(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        arr = np.asarray(v)
        if arr.ndim == 0:
            out[k] = float(arr)
        else:
            out[k] = arr.tolist()
    return out


def compute_rollout_metrics(name, positions, yaws, torso_heights, actions, done_step, dt):
    positions = np.asarray(positions, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64)
    torso_heights = np.asarray(torso_heights, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)

    if len(positions) == 0:
        return {
            "method": name,
            "survival_steps": 0,
            "survival_time_sec": 0.0,
            "distance_x": 0.0,
            "avg_forward_velocity": 0.0,
            "lateral_drift_abs_mean": 0.0,
            "yaw_abs_mean": 0.0,
            "torso_height_mean": 0.0,
            "torso_height_std": 0.0,
            "action_l2_mean": 0.0,
            "action_delta_l2_mean": 0.0,
        }

    survival_steps = done_step if done_step is not None else len(positions)
    survival_time_sec = survival_steps * dt

    x0 = positions[0, 0]
    xT = positions[-1, 0]
    distance_x = float(xT - x0)
    avg_forward_velocity = float(distance_x / max(survival_time_sec, 1e-8))

    lateral_drift_abs_mean = float(np.mean(np.abs(positions[:, 1])))
    yaw_abs_mean = float(np.mean(np.abs(yaws)))
    torso_height_mean = float(np.mean(torso_heights))
    torso_height_std = float(np.std(torso_heights))

    action_l2 = np.linalg.norm(actions, axis=1)
    action_l2_mean = float(np.mean(action_l2))

    if len(actions) >= 2:
        action_delta = actions[1:] - actions[:-1]
        action_delta_l2_mean = float(np.mean(np.linalg.norm(action_delta, axis=1)))
    else:
        action_delta_l2_mean = 0.0

    return {
        "method": name,
        "survival_steps": int(survival_steps),
        "survival_time_sec": survival_time_sec,
        "distance_x": distance_x,
        "avg_forward_velocity": avg_forward_velocity,
        "lateral_drift_abs_mean": lateral_drift_abs_mean,
        "yaw_abs_mean": yaw_abs_mean,
        "torso_height_mean": torso_height_mean,
        "torso_height_std": torso_height_std,
        "action_l2_mean": action_l2_mean,
        "action_delta_l2_mean": action_delta_l2_mean,
    }


def rollout_ppo(env, max_steps, ppo_path):
    policy_fn = checkpoint.load_policy(str(Path(ppo_path).resolve()))
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_fn)

    x_vel, y_vel, yaw_vel = 1.0, 0.0, 0.0
    command = jp.array([x_vel, y_vel, yaw_vel])
    phase_dt = 2 * jp.pi * env.dt * 1.5
    phase = jp.array([0, jp.pi])

    rng = jax.random.PRNGKey(1)
    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase
    state.info["command"] = command

    torso_id = env.mj_model.body("torso_link").id

    positions = []
    yaws = []
    torso_heights = []
    actions = []
    done_step = None

    for step_idx in range(max_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        action = np.array(ctrl, dtype=np.float32)

        state = jit_step(state, ctrl)
        state.info["command"] = command

        pos = np.array(state.data.xpos[torso_id], dtype=np.float32)
        yaw = get_yaw_from_xmat(np.array(state.data.xmat[torso_id], dtype=np.float32))

        positions.append(pos)
        yaws.append(yaw)
        torso_heights.append(float(pos[2]))
        actions.append(action)

        if bool(state.done):
            done_step = step_idx + 1
            break

    return compute_rollout_metrics("PPO", positions, yaws, torso_heights, actions, done_step, env.dt)


def rollout_bc(env, max_steps, bc_path, device):
    ckpt = torch.load(Path(bc_path).resolve(), map_location=device, weights_only=False)
    model = build_bc_model(ckpt, device)

    obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)
    act_mean = np.asarray(ckpt["act_mean"], dtype=np.float32)
    act_std = np.asarray(ckpt["act_std"], dtype=np.float32)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    x_vel, y_vel, yaw_vel = 1.0, 0.0, 0.0
    command = jp.array([x_vel, y_vel, yaw_vel])
    phase_dt = 2 * jp.pi * env.dt * 1.5
    phase = jp.array([0, jp.pi])

    rng = jax.random.PRNGKey(1)
    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase
    state.info["command"] = command

    torso_id = env.mj_model.body("torso_link").id

    positions = []
    yaws = []
    torso_heights = []
    actions = []
    done_step = None

    for step_idx in range(max_steps):
        obs_vec = get_obs_vector(state)
        action = bc_action(model, obs_vec, obs_mean, obs_std, act_mean, act_std, device)

        state = jit_step(state, jp.array(action))
        state.info["command"] = command

        pos = np.array(state.data.xpos[torso_id], dtype=np.float32)
        yaw = get_yaw_from_xmat(np.array(state.data.xmat[torso_id], dtype=np.float32))

        positions.append(pos)
        yaws.append(yaw)
        torso_heights.append(float(pos[2]))
        actions.append(action)

        if bool(state.done):
            done_step = step_idx + 1
            break

    return compute_rollout_metrics("BC", positions, yaws, torso_heights, actions, done_step, env.dt)


def print_comparison(ppo_metrics, bc_metrics):
    keys = [
        "survival_steps",
        "survival_time_sec",
        "distance_x",
        "avg_forward_velocity",
        "lateral_drift_abs_mean",
        "yaw_abs_mean",
        "torso_height_mean",
        "torso_height_std",
        "action_l2_mean",
        "action_delta_l2_mean",
    ]

    print("\n=== PPO vs BC quantitative comparison ===")
    print(f"{'Metric':30s} {'PPO':>14s} {'BC':>14s} {'BC-PPO':>14s}")
    for k in keys:
        p = float(ppo_metrics[k])
        b = float(bc_metrics[k])
        d = b - p
        print(f"{k:30s} {p:14.6f} {b:14.6f} {d:14.6f}")


def main():
    args = parse_args()

    save_json = Path(args.save_json)
    save_json.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = registry.get_default_config(ENV_NAME)
    env_cfg.push_config.enable = False
    env = registry.load(ENV_NAME, config=env_cfg)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ppo_metrics = rollout_ppo(env, args.max_steps, args.ppo_path)
    bc_metrics = rollout_bc(env, args.max_steps, args.bc_path, device)

    print_comparison(ppo_metrics, bc_metrics)

    out = {
        "env_name": ENV_NAME,
        "max_steps": args.max_steps,
        "ppo": ppo_metrics,
        "bc": bc_metrics,
    }

    with open(save_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved metrics to: {save_json}")


if __name__ == "__main__":
    main()