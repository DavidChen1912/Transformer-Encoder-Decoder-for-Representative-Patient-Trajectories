"""Load one patient's complete path distribution as one sample."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    config = json.loads(Path(path).read_text())
    for key in ("data_dir", "output_dir"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else ROOT / value)
    return config


class PatientDataset:
    def __init__(self, directory, time_scale):
        self.directory = Path(directory)
        self.states = pd.read_csv(self.directory / "states.csv")["cluster_number"].tolist()
        if not self.states or len(set(self.states)) != len(self.states):
            raise ValueError("State vocabulary must be nonempty and unique")
        self.mapping = {state: i for i, state in enumerate(self.states)}
        self.files = sorted(self.directory.glob("*/paths.csv"))
        self.time_scale = float(time_scale)
        if not self.files or self.time_scale <= 0:
            raise ValueError("Need patient paths and a positive common time scale")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        frame = pd.read_csv(path)
        visits = sorted(
            [c.removeprefix("state_") for c in frame if re.fullmatch(r"state_day[0-9]+", c)],
            key=lambda v: int(v[3:]),
        )
        if len(visits) < 2:
            raise ValueError(f"Need at least two visits: {path}")
        ids = np.stack([frame[f"state_{v}"].map(self.mapping).to_numpy() for v in visits], axis=1)
        if pd.isna(ids).any():
            raise ValueError(f"Unknown state in {path}")
        ids = ids.astype(np.int64)
        if len(np.unique(ids, axis=0)) != len(ids):
            raise ValueError(f"Duplicate path sequences: {path}")
        days = np.array([int(v[3:]) for v in visits])
        if (np.diff(days) <= 0).any():
            raise ValueError(f"Visits must have unique increasing days: {path}")
        q = np.stack([
            frame[f"conditional_prob_{a}_to_{b}"].to_numpy()
            for a, b in zip(visits, visits[1:])
        ], axis=1)
        weights = frame["path_mass"].to_numpy()
        if (not np.isfinite(q).all() or (q < 0).any() or (q > 1 + 1e-6).any()
                or not np.isfinite(weights).all() or (weights < 0).any()
                or not np.isclose(weights.sum(), 1, atol=1e-6)):
            raise ValueError(f"Invalid Q or path distribution: {path}")
        # Zero-mass paths cannot affect the weighted set aggregation or coverage.
        active = weights > 0
        ids, q, weights = ids[active], q[active], weights[active]
        weights = weights / weights.sum()
        n, t = ids.shape
        transition = np.zeros((n, t, 3), dtype=np.float32)
        transition[:, :-1, 0] = q
        transition[:, :-1, 1] = np.diff(days)[None, :] / self.time_scale
        transition[:, :-1, 2] = 1
        k = len(self.states)
        offsets = np.arange(t - 1) * k * k
        edge_indices = offsets[None, :] + ids[:, :-1] * k + ids[:, 1:]
        target = np.zeros((t - 1) * k * k, dtype=np.float64)
        for step in range(t - 1):
            np.add.at(target, edge_indices[:, step], weights)
        # Recover original OT from repeated edge values without counting duplicates.
        for step, (a, b) in enumerate(zip(visits, visits[1:])):
            values = frame[f"mass_{a}_to_{b}"].to_numpy()
            all_ids = np.stack([frame[f"state_{a}"].map(self.mapping),
                                frame[f"state_{b}"].map(self.mapping)], axis=1)
            local = all_ids[:, 0] * k + all_ids[:, 1]
            expected = target[step * k * k:(step + 1) * k * k]
            np.testing.assert_allclose(expected[local], values, atol=1e-5)
        return {
            "patient": path.parent.name,
            "states": torch.as_tensor(ids, dtype=torch.long),
            "transition": torch.as_tensor(transition),
            "weights": torch.as_tensor(weights, dtype=torch.float32),
            "days": torch.as_tensor(days, dtype=torch.float32),
            "edge_indices": torch.as_tensor(edge_indices, dtype=torch.long),
            "target": torch.as_tensor(target.reshape(t - 1, k, k), dtype=torch.float32),
        }


def to_device(sample, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in sample.items()}
