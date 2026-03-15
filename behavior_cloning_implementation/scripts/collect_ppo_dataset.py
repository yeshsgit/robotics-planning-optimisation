import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jp
from mujoco_playground import registry
from brax.training.agents.ppo import checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Collect PPO expert demonstrations for G1 walking.")
    parser.add_argument(
        "--policy-path",
        type=str,
        default="./checkpoints/ppo_g1/g1_walking_policy_v3/000000000001",
        help="Path to PPO checkpoint directory.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./data/demonstrations/g1_walking",
        help="Directory to save demos.npy and meta.npy",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=200,
        help="Number of rollout episodes to collect.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Keep only episodes with at least this many steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for JAX PRNG.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env_name = "G1JoystickFlatTerrain"
    policy_path = Path(args.policy_path).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading environment...")
    env_cfg = registry.get_default_config(env_name)
    env_cfg.push_config.enable = False  # disable random pushes for clean demonstrations
    eval_env = registry.load(env_name, config=env_cfg)

    print("Loading PPO policy...")
    custom_policy_fn = checkpoint.load_policy(str(policy_path))

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(custom_policy_fn)

    rng = jax.random.PRNGKey(args.seed)

    # Desired walking command
    x_vel = 1.0
    y_vel = 0.0
    yaw_vel = 0.0
    command = jp.array([x_vel, y_vel, yaw_vel])

    # Gait phase info
    phase_dt = 2 * jp.pi * eval_env.dt * 1.5
    phase = jp.array([0, jp.pi])

    all_episodes = []

    print(f"Collecting {args.n_episodes} episodes...")
    for j in range(args.n_episodes):
        print(f"episode {j}")
        state = jit_reset(rng)
        state.info["phase_dt"] = phase_dt
        state.info["phase"] = phase
        state.info["command"] = command

        episode = {"obs": [], "action": []}

        for _ in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)

            # Expert observation and action
            obs = np.array(state.obs["state"], dtype=np.float32)   # expected shape (103,)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            action = np.array(ctrl, dtype=np.float32)              # expected shape (29,)

            episode["obs"].append(obs)
            episode["action"].append(action)

            state = jit_step(state, ctrl)
            state.info["command"] = command

            if state.done:
                break

        episode["obs"] = np.array(episode["obs"], dtype=np.float32)
        episode["action"] = np.array(episode["action"], dtype=np.float32)
        all_episodes.append(episode)

    # Filter too-short episodes
    filtered_episodes = [ep for ep in all_episodes if len(ep["obs"]) >= args.min_length]

    print("Episode length counts after filtering:")
    print(Counter(len(ep["obs"]) for ep in filtered_episodes))

    np.save(save_dir / "demos.npy", filtered_episodes, allow_pickle=True)

    meta = {
        "env_name": env_name,
        "obs_dim": int(filtered_episodes[0]["obs"].shape[1]) if filtered_episodes else None,
        "action_dim": int(filtered_episodes[0]["action"].shape[1]) if filtered_episodes else None,
        "dt": float(eval_env.dt),
        "hz": float(1.0 / eval_env.dt),
        "command": [float(x_vel), float(y_vel), float(yaw_vel)],
        "n_episodes_requested": args.n_episodes,
        "n_episodes_kept": len(filtered_episodes),
        "min_length": args.min_length,
    }
    np.save(save_dir / "meta.npy", meta, allow_pickle=True)

    # Reload sanity check
    demos = np.load(save_dir / "demos.npy", allow_pickle=True)
    print("Len demos:", len(demos))
    if len(demos) > 0:
        print(type(demos[0]))
        print(demos[0].keys())
        print(demos[0]["obs"].shape)
        print(demos[0]["action"].shape)

    print(f"Saved demonstrations to: {save_dir}")


if __name__ == "__main__":
    main()