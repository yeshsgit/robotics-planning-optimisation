import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_demos(demo_path: Path) -> List[dict]:
    demos = np.load(demo_path, allow_pickle=True)
    # np.save(list_of_dicts) usually comes back as an object ndarray
    if isinstance(demos, np.ndarray):
        demos = demos.tolist()
    if not isinstance(demos, list):
        raise TypeError(f"Expected list of episode dicts, got {type(demos)}")
    if len(demos) == 0:
        raise ValueError("No demonstrations found.")
    return demos


def split_by_episode(
    demos: List[dict], train_ratio: float = 0.9, seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    idxs = list(range(len(demos)))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    n_train = max(1, int(len(idxs) * train_ratio))
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:] if n_train < len(idxs) else idxs[-1:]

    train_eps = [demos[i] for i in train_idxs]
    val_eps = [demos[i] for i in val_idxs]
    return train_eps, val_eps


def flatten_episodes(demos: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    obs_list = []
    act_list = []

    for ep in demos:
        obs = np.asarray(ep["obs"], dtype=np.float32)
        act = np.asarray(ep["action"], dtype=np.float32)

        if obs.ndim != 2 or act.ndim != 2:
            raise ValueError(f"Episode obs/action must be 2D, got {obs.ndim}, {act.ndim}")
        if len(obs) != len(act):
            raise ValueError(f"Episode length mismatch: {len(obs)} vs {len(act)}")

        obs_list.append(obs)
        act_list.append(act)

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)
    return obs_all, act_all


class BCDataset(Dataset):
    def __init__(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        act_mean: np.ndarray,
        act_std: np.ndarray,
    ):
        self.obs = ((obs - obs_mean) / obs_std).astype(np.float32)
        self.act = ((act - act_mean) / act_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.obs[idx]), torch.from_numpy(self.act[idx])


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for obs, act in loader:
            obs = obs.to(device)
            act = act.to(device)

            pred = model(obs)
            loss = criterion(pred, act)

            bs = obs.shape[0]
            total_loss += loss.item() * bs
            total_count += bs

    return total_loss / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a behavior cloning policy on PPO demonstrations.")
    parser.add_argument("--demo-path", type=str, required=True, help="Path to demos.npy")
    parser.add_argument("--meta-path", type=str, default=None, help="Optional path to meta.npy")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/bc_g1")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)

    demo_path = Path(args.demo_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    demos = load_demos(demo_path)

    if args.meta_path is not None and Path(args.meta_path).exists():
        meta_raw = np.load(args.meta_path, allow_pickle=True).item()
        print("Loaded meta:", meta_raw)
    else:
        meta_raw = None

    train_eps, val_eps = split_by_episode(demos, train_ratio=args.train_ratio, seed=args.seed)
    print(f"Num episodes: total={len(demos)}, train={len(train_eps)}, val={len(val_eps)}")

    train_obs, train_act = flatten_episodes(train_eps)
    val_obs, val_act = flatten_episodes(val_eps)

    obs_dim = train_obs.shape[1]
    action_dim = train_act.shape[1]

    print(f"Train obs shape: {train_obs.shape}")
    print(f"Train act shape: {train_act.shape}")
    print(f"Val   obs shape: {val_obs.shape}")
    print(f"Val   act shape: {val_act.shape}")
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")

    obs_mean = train_obs.mean(axis=0)
    obs_std = train_obs.std(axis=0) + 1e-6
    act_mean = train_act.mean(axis=0)
    act_std = train_act.std(axis=0) + 1e-6

    train_ds = BCDataset(train_obs, train_act, obs_mean, obs_std, act_mean, act_std)
    val_ds = BCDataset(val_obs, val_act, obs_mean, obs_std, act_mean, act_std)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improve = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)

            optimizer.zero_grad()
            pred = model(obs)
            loss = criterion(pred, act)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            bs = obs.shape[0]
            running_loss += loss.item() * bs
            sample_count += bs

        train_loss = running_loss / max(sample_count, 1)
        val_loss = evaluate(model, val_loader, device, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improve = 0

            ckpt = {
                "model_state_dict": model.state_dict(),
                "obs_mean": obs_mean.astype(np.float32),
                "obs_std": obs_std.astype(np.float32),
                "act_mean": act_mean.astype(np.float32),
                "act_std": act_std.astype(np.float32),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "meta": meta_raw,
            }
            torch.save(ckpt, save_dir / "bc_g1_best.pt")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    with open(save_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(save_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training complete. Best val loss = {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Saved checkpoint to: {save_dir / 'bc_g1_best.pt'}")


if __name__ == "__main__":
    main()