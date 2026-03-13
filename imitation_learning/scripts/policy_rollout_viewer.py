import time
import argparse
from pathlib import Path

import mujoco
from mujoco_playground import registry
from brax.training.agents.ppo import checkpoint
import jax
import jax.numpy as jnp
import numpy as np

ENV_NAME = "G1JoystickFlatTerrain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout and play a policy.")
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to policy checkpoint."
    )
    return parser.parse_args()


def rollout_policy(path: str):
    env_cfg = registry.get_default_config(ENV_NAME)
    env_cfg.push_config.enable = False  # Disable random pushes
    env = registry.load(ENV_NAME, config=env_cfg)

    checkpoint_path = str(Path(path).absolute())
    policy_fn = checkpoint.load_policy(checkpoint_path)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_fn)

    # desired walking command
    x_vel = 1.0
    y_vel = 0.0
    yaw_vel = 0.0
    command = jnp.array([x_vel, y_vel, yaw_vel])

    phase_dt = 2 * jnp.pi * env.dt * 1.5
    phase = jnp.array([0, jnp.pi])

    rng = jax.random.PRNGKey(1)
    state = jit_reset(rng)

    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase

    data = mujoco.MjData(env.mj_model)

    with mujoco.viewer.launch_passive(env.mj_model, data) as viewer:
        for i in range(env_cfg.episode_length):
            if not viewer.is_running():
                break

            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            if state.done:
                state = jit_reset(rng)
                state.info["phase_dt"] = phase_dt
                state.info["phase"] = phase
            state.info["command"] = command  # set after step

            data.qpos[:] = np.array(state.data.qpos)
            data.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(env.mj_model, data)
            viewer.sync()
            time.sleep(env.dt)


def main():
    args = parse_args()
    rollout_policy(args.path)


if __name__ == "__main__":
    main()
