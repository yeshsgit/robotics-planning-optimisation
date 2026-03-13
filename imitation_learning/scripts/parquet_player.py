import time
import argparse

import pandas as pd
import numpy as np

import mujoco
from mujoco_playground import registry


FPS = 30  # hf datasets for g1 are 30fps
ENV_NAME = "G1JoystickFlatTerrain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a motion parquet file.")
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to motion file."
    )
    return parser.parse_args()


def play_parquet_file(path: str):
    df = pd.read_parquet(path)

    episode = df[df["episode_index"] == 0]
    trajectory = episode["action.robot_q_desired"]
    trajectory_array = np.stack(trajectory.to_numpy())

    env = registry.load(ENV_NAME)
    model = env.mj_model
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for qpos in trajectory_array:
                if not viewer.is_running():
                    break
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1 / FPS)


def main():
    args = parse_args()
    play_parquet_file(args.path)


if __name__ == "__main__":
    main()
