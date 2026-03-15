import argparse
import time
from pathlib import Path

import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import jax
import jax.numpy as jp

from mujoco_playground import registry


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
    parser = argparse.ArgumentParser(description="Evaluate BC policy on G1 walking and save video.")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./checkpoints/bc_g1/bc_g1_best.pt",
        help="Path to trained BC checkpoint.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default="./videos/bc_g1_rollout.mp4",
        help="Path to save rendered rollout video.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max rollout steps. Defaults to env episode length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Show live MuJoCo viewer during rollout.",
    )
    return parser.parse_args()


def get_obs_vector(state) -> np.ndarray:
    # Matches your collector format
    if isinstance(state.obs, dict):
        if "state" in state.obs:
            return np.array(state.obs["state"], dtype=np.float32)
        raise KeyError(f"state.obs is dict but does not contain 'state'. Keys: {list(state.obs.keys())}")
    return np.array(state.obs, dtype=np.float32)


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> nn.Module:
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


def bc_action(
    model: nn.Module,
    obs_vec: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    act_mean: np.ndarray,
    act_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    obs_norm = (obs_vec - obs_mean) / obs_std
    obs_tensor = torch.from_numpy(obs_norm.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        act_norm = model(obs_tensor).cpu().numpy()[0]

    act = act_norm * act_std + act_mean
    return act.astype(np.float32)


def rollout_bc(args):
    ckpt_path = Path(args.ckpt_path).resolve()
    video_path = Path(args.save_video).resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Loading BC checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)
    act_mean = np.asarray(ckpt["act_mean"], dtype=np.float32)
    act_std = np.asarray(ckpt["act_std"], dtype=np.float32)

    model = build_model_from_ckpt(ckpt, device)

    print("Loading environment...")
    env_cfg = registry.get_default_config(ENV_NAME)
    env_cfg.push_config.enable = False
    env = registry.load(ENV_NAME, config=env_cfg)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Match expert data collection conditions
    x_vel = 1.0
    y_vel = 0.0
    yaw_vel = 0.0
    command = jp.array([x_vel, y_vel, yaw_vel])

    phase_dt = 2 * jp.pi * env.dt * 1.5
    phase = jp.array([0, jp.pi])

    rng = jax.random.PRNGKey(1)
    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase
    state.info["command"] = command

    max_steps = args.max_steps if args.max_steps is not None else env_cfg.episode_length
    print(f"Rollout steps: {max_steps}")

    traj = []
    done_step = None

    if args.show_viewer:
        data = mujoco.MjData(env.mj_model)
        with mujoco.viewer.launch_passive(env.mj_model, data) as viewer:
            for step_idx in range(max_steps):
                if not viewer.is_running():
                    print("Viewer closed by user.")
                    break

                obs_vec = get_obs_vector(state)
                action = bc_action(model, obs_vec, obs_mean, obs_std, act_mean, act_std, device)

                state = jit_step(state, jp.array(action))
                state.info["command"] = command

                traj.append(state)

                data.qpos[:] = np.array(state.data.qpos)
                data.qvel[:] = np.array(state.data.qvel)
                mujoco.mj_forward(env.mj_model, data)
                viewer.sync()
                time.sleep(env.dt)

                if bool(state.done):
                    done_step = step_idx + 1
                    print(f"Episode terminated at step {done_step}")
                    break
    else:
        for step_idx in range(max_steps):
            obs_vec = get_obs_vector(state)
            action = bc_action(model, obs_vec, obs_mean, obs_std, act_mean, act_std, device)

            state = jit_step(state, jp.array(action))
            state.info["command"] = command
            traj.append(state)

            if bool(state.done):
                done_step = step_idx + 1
                print(f"Episode terminated at step {done_step}")
                break

    print(f"Collected {len(traj)} states for rendering.")

    if len(traj) == 0:
        print("No trajectory collected; skipping video render.")
        return

    render_every = 1
    fps = 1.0 / env.dt / render_every

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    frames = env.render(
        traj[::render_every],
        camera="track",
        scene_option=scene_option,
        width=1280,
        height=480,
    )

    media.write_video(str(video_path), frames, fps=fps)
    print(f"Saved video to: {video_path}")
    print(f"Approx rollout duration: {len(traj) * env.dt:.2f} seconds")

    if done_step is None:
        print("BC policy survived full rollout horizon.")
    else:
        print(f"BC policy terminated early at step {done_step} ({done_step * env.dt:.2f} s).")


def main():
    args = parse_args()
    rollout_bc(args)


if __name__ == "__main__":
    main()