import argparse
import time
from pathlib import Path

import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np
import jax
import jax.numpy as jp

from mujoco_playground import registry
from brax.training.agents.ppo import checkpoint


ENV_NAME = "G1JoystickFlatTerrain"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on G1 walking and save video.")
    parser.add_argument(
        "--policy-path",
        type=str,
        default="./checkpoints/ppo_g1/g1_walking_policy_v3/000000000001",
        help="Path to PPO checkpoint directory.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default="./videos/ppo_g1_rollout.mp4",
        help="Path to save rendered rollout video.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max rollout steps. Defaults to env episode length.",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Show live MuJoCo viewer during rollout.",
    )
    return parser.parse_args()


def rollout_ppo(args):
    policy_path = Path(args.policy_path).resolve()
    video_path = Path(args.save_video).resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading PPO checkpoint from: {policy_path}")

    env_cfg = registry.get_default_config(ENV_NAME)
    env_cfg.push_config.enable = False
    env = registry.load(ENV_NAME, config=env_cfg)

    policy_fn = checkpoint.load_policy(str(policy_path))

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_fn)

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

                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)

                state = jit_step(state, ctrl)
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
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)

            state = jit_step(state, ctrl)
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
        print("PPO policy survived full rollout horizon.")
    else:
        print(f"PPO policy terminated early at step {done_step} ({done_step * env.dt:.2f} s).")


def main():
    args = parse_args()
    rollout_ppo(args)


if __name__ == "__main__":
    main()