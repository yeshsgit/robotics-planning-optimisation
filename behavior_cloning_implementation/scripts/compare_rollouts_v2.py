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
    parser = argparse.ArgumentParser(description="Cleaner quantitative comparison for PPO vs BC on G1 walking.")
    parser.add_argument("--ppo-path", type=str, default="./checkpoints/ppo_g1/g1_walking_policy_v3/000000000001")
    parser.add_argument("--bc-path", type=str, default="./checkpoints/bc_g1/bc_g1_best.pt")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-json", type=str, default="./results/ppo_vs_bc_metrics_v2.json")
    parser.add_argument("--device", type=str, default="cpu")
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


def rotation_from_xmat(xmat_flat: np.ndarray) -> np.ndarray:
    return np.asarray(xmat_flat, dtype=np.float64).reshape(3, 3)


def rpy_from_R(R: np.ndarray):
    # ZYX yaw-pitch-roll convention
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return roll, pitch, yaw


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def compute_rollout_metrics(name, positions, rotations, actions, done_step, dt):
    positions = np.asarray(positions, dtype=np.float64)      # [T, 3]
    actions = np.asarray(actions, dtype=np.float64)          # [T, A]
    T = len(positions)

    if T == 0:
        return {
            "method": name,
            "survival_steps": 0,
            "survival_time_sec": 0.0,
        }

    survival_steps = int(done_step if done_step is not None else T)
    survival_time_sec = float(survival_steps * dt)

    # Basic position stats
    p0 = positions[0]
    pT = positions[-1]
    disp = pT - p0
    disp_xy = disp[:2]

    # Initial heading axis from torso orientation at first step
    R0 = rotations[0]
    fwd0 = R0[:, 0].copy()
    fwd0[2] = 0.0
    fwd0_norm = np.linalg.norm(fwd0)
    if fwd0_norm < 1e-8:
        fwd0 = np.array([1.0, 0.0, 0.0])
    else:
        fwd0 = fwd0 / fwd0_norm

    left0 = np.array([-fwd0[1], fwd0[0], 0.0])

    forward_progress_initial_heading = float(np.dot(disp, fwd0))
    lateral_offset_initial_heading = float(np.dot(disp, left0))

    # World-frame path metrics
    diffs = positions[1:, :2] - positions[:-1, :2]
    step_dists = np.linalg.norm(diffs, axis=1)
    path_length_xy = float(np.sum(step_dists))
    net_disp_xy = float(np.linalg.norm(disp_xy))
    straightness_ratio = float(net_disp_xy / max(path_length_xy, 1e-8))

    # Heading-aware velocity metrics using current body forward axis
    body_forward_vels = []
    body_lateral_vels = []
    rolls, pitches, yaws = [], [], []

    for t in range(T):
        R = rotations[t]
        roll, pitch, yaw = rpy_from_R(R)
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)

    rolls = np.asarray(rolls)
    pitches = np.asarray(pitches)
    yaws = np.asarray(yaws)

    for t in range(T - 1):
        dp = positions[t + 1] - positions[t]
        vel = dp / dt

        R = rotations[t]
        fwd = R[:, 0].copy()
        fwd[2] = 0.0
        nf = np.linalg.norm(fwd)
        if nf < 1e-8:
            fwd = np.array([1.0, 0.0, 0.0])
        else:
            fwd = fwd / nf

        left = np.array([-fwd[1], fwd[0], 0.0])

        body_forward_vels.append(np.dot(vel, fwd))
        body_lateral_vels.append(np.dot(vel, left))

    body_forward_vels = np.asarray(body_forward_vels)
    body_lateral_vels = np.asarray(body_lateral_vels)

    torso_height = positions[:, 2]

    action_l2 = np.linalg.norm(actions, axis=1)
    if len(actions) >= 2:
        action_delta = actions[1:] - actions[:-1]
        action_delta_l2 = np.linalg.norm(action_delta, axis=1)
        action_delta_l2_mean = float(np.mean(action_delta_l2))
    else:
        action_delta_l2_mean = 0.0

    yaw_rel = wrap_to_pi(yaws - yaws[0])

    return {
        "method": name,
        "survival_steps": survival_steps,
        "survival_time_sec": survival_time_sec,

        "initial_position_xyz": p0.tolist(),
        "final_position_xyz": pT.tolist(),
        "world_displacement_xyz": disp.tolist(),

        "world_x_displacement": float(disp[0]),
        "world_y_displacement": float(disp[1]),
        "forward_progress_initial_heading": forward_progress_initial_heading,
        "lateral_offset_initial_heading": lateral_offset_initial_heading,

        "path_length_xy": path_length_xy,
        "net_displacement_xy": net_disp_xy,
        "straightness_ratio": straightness_ratio,

        "body_forward_velocity_mean": float(np.mean(body_forward_vels)) if len(body_forward_vels) else 0.0,
        "body_forward_velocity_std": float(np.std(body_forward_vels)) if len(body_forward_vels) else 0.0,
        "body_lateral_velocity_abs_mean": float(np.mean(np.abs(body_lateral_vels))) if len(body_lateral_vels) else 0.0,

        "torso_height_mean": float(np.mean(torso_height)),
        "torso_height_std": float(np.std(torso_height)),

        "roll_abs_mean": float(np.mean(np.abs(rolls))),
        "pitch_abs_mean": float(np.mean(np.abs(pitches))),
        "yaw_rel_abs_mean": float(np.mean(np.abs(yaw_rel))),
        "yaw_rel_final": float(yaw_rel[-1]),

        "action_l2_mean": float(np.mean(action_l2)),
        "action_delta_l2_mean": action_delta_l2_mean,
    }


def rollout_ppo(env, max_steps, ppo_path):
    policy_fn = checkpoint.load_policy(str(Path(ppo_path).resolve()))
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_fn)

    command = jp.array([1.0, 0.0, 0.0])
    phase_dt = 2 * jp.pi * env.dt * 1.5
    phase = jp.array([0, jp.pi])

    rng = jax.random.PRNGKey(1)
    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase
    state.info["command"] = command

    torso_id = env.mj_model.body("torso_link").id

    positions, rotations, actions = [], [], []
    done_step = None

    for step_idx in range(max_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        action = np.array(ctrl, dtype=np.float32)

        state = jit_step(state, ctrl)
        state.info["command"] = command

        pos = np.array(state.data.xpos[torso_id], dtype=np.float32)
        R = rotation_from_xmat(np.array(state.data.xmat[torso_id], dtype=np.float32))

        positions.append(pos)
        rotations.append(R)
        actions.append(action)

        if bool(state.done):
            done_step = step_idx + 1
            break

    return compute_rollout_metrics("PPO", positions, rotations, actions, done_step, env.dt)


def rollout_bc(env, max_steps, bc_path, device):
    ckpt = torch.load(Path(bc_path).resolve(), map_location=device, weights_only=False)
    model = build_bc_model(ckpt, device)

    obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)
    act_mean = np.asarray(ckpt["act_mean"], dtype=np.float32)
    act_std = np.asarray(ckpt["act_std"], dtype=np.float32)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    command = jp.array([1.0, 0.0, 0.0])
    phase_dt = 2 * jp.pi * env.dt * 1.5
    phase = jp.array([0, jp.pi])

    rng = jax.random.PRNGKey(1)
    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase
    state.info["command"] = command

    torso_id = env.mj_model.body("torso_link").id

    positions, rotations, actions = [], [], []
    done_step = None

    for step_idx in range(max_steps):
        obs_vec = get_obs_vector(state)
        action = bc_action(model, obs_vec, obs_mean, obs_std, act_mean, act_std, device)

        state = jit_step(state, jp.array(action))
        state.info["command"] = command

        pos = np.array(state.data.xpos[torso_id], dtype=np.float32)
        R = rotation_from_xmat(np.array(state.data.xmat[torso_id], dtype=np.float32))

        positions.append(pos)
        rotations.append(R)
        actions.append(action)

        if bool(state.done):
            done_step = step_idx + 1
            break

    return compute_rollout_metrics("BC", positions, rotations, actions, done_step, env.dt)


def print_comparison(ppo_metrics, bc_metrics):
    keys = [
        "survival_steps",
        "survival_time_sec",
        "world_x_displacement",
        "world_y_displacement",
        "forward_progress_initial_heading",
        "lateral_offset_initial_heading",
        "path_length_xy",
        "straightness_ratio",
        "body_forward_velocity_mean",
        "body_lateral_velocity_abs_mean",
        "torso_height_std",
        "roll_abs_mean",
        "pitch_abs_mean",
        "yaw_rel_abs_mean",
        "action_delta_l2_mean",
    ]

    print("\n=== PPO vs BC cleaner comparison ===")
    print(f"{'Metric':34s} {'PPO':>14s} {'BC':>14s} {'BC-PPO':>14s}")
    for k in keys:
        p = float(ppo_metrics[k])
        b = float(bc_metrics[k])
        d = b - p
        print(f"{k:34s} {p:14.6f} {b:14.6f} {d:14.6f}")

    print("\nInitial/final torso positions")
    print("PPO initial:", ppo_metrics["initial_position_xyz"])
    print("PPO final  :", ppo_metrics["final_position_xyz"])
    print("BC  initial:", bc_metrics["initial_position_xyz"])
    print("BC  final  :", bc_metrics["final_position_xyz"])


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