"""Deterministic source-group split for self-supervised validation only."""

from __future__ import annotations

import numpy as np


def source_group_split(group_ids: list[str], seed: int, train_fraction: float = 0.8) -> dict[str, object]:
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ValueError("At least two source groups are needed for group-held-out validation.")
    if not (0.5 <= train_fraction < 1.0):
        raise ValueError("Training fraction must be in [0.5, 1).")
    order = np.random.default_rng(seed).permutation(unique_groups).tolist()
    validation_count = max(1, int(round((1.0 - train_fraction) * len(unique_groups))))
    validation_count = min(validation_count, len(unique_groups) - 1)
    validation = sorted(order[:validation_count])
    train = sorted(order[validation_count:])
    return {
        "seed": int(seed),
        "method": "deterministic source-folder group split; identity is not verified participant identity",
        "requested_train_fraction": float(train_fraction),
        "train_group_count": len(train),
        "validation_group_count": len(validation),
        "train_group_ids": train,
        "validation_group_ids": validation,
    }
