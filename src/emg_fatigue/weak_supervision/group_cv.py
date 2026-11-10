"""Deterministic nested group folds for development-only model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class GroupFold:
    fold_id: int
    train_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class FoldPlan:
    seed: int
    development_group_ids: tuple[str, ...]
    outer_folds: tuple[GroupFold, ...]
    inner_folds: dict[int, tuple[GroupFold, ...]]


def group_folds(
    group_ids: Sequence[str],
    outer_folds: int = 4,
    inner_folds: int = 3,
    seed: int = 20261113,
) -> FoldPlan:
    """Build shuffled GroupKFold splits using source groups, never windows as groups.

    ``group_ids`` contains one source-group ID per row. Supplying all primary
    window IDs lets GroupKFold balance fold sizes while preserving every group.
    Inner partitions are constructed only from each outer-training group set.
    """
    raw = np.asarray(group_ids, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("group_ids must be a non-empty one-dimensional sequence")
    if any(value is None or (isinstance(value, float) and np.isnan(value)) for value in raw):
        raise ValueError("group_ids cannot contain missing values")
    groups = np.asarray([str(value) for value in raw], dtype=object)
    unique_groups = tuple(sorted(set(groups.tolist())))
    outer_count = int(outer_folds)
    inner_count = int(inner_folds)
    if outer_count < 2 or inner_count < 2:
        raise ValueError("outer_folds and inner_folds must both be at least 2")
    if len(unique_groups) < outer_count:
        raise ValueError(f"Need at least {outer_count} distinct groups for {outer_count} outer folds")

    outer_splitter = GroupKFold(n_splits=outer_count, shuffle=True, random_state=int(seed))
    sample_indices = np.arange(groups.size)
    outer_plan: list[GroupFold] = []
    inner_plan: dict[int, tuple[GroupFold, ...]] = {}

    for outer_index, (outer_train, outer_valid) in enumerate(
        outer_splitter.split(sample_indices.reshape(-1, 1), groups=groups), start=1
    ):
        outer_train_groups = tuple(sorted(set(groups[outer_train].tolist())))
        outer_valid_groups = tuple(sorted(set(groups[outer_valid].tolist())))
        if len(outer_train_groups) < inner_count:
            raise ValueError(
                f"Each outer training partition must contain at least {inner_count} distinct groups"
            )
        outer_plan.append(
            GroupFold(
                fold_id=outer_index,
                train_group_ids=outer_train_groups,
                validation_group_ids=outer_valid_groups,
            )
        )

        inner_splitter = GroupKFold(
            n_splits=inner_count,
            shuffle=True,
            random_state=int(seed) + outer_index,
        )
        inner_folds_for_outer: list[GroupFold] = []
        outer_train_groups_per_row = groups[outer_train]
        outer_train_indices = np.arange(outer_train_groups_per_row.size)
        for inner_index, (inner_train, inner_valid) in enumerate(
            inner_splitter.split(
                outer_train_indices.reshape(-1, 1),
                groups=outer_train_groups_per_row,
            ),
            start=1,
        ):
            inner_folds_for_outer.append(
                GroupFold(
                    fold_id=inner_index,
                    train_group_ids=tuple(sorted(set(outer_train_groups_per_row[inner_train].tolist()))),
                    validation_group_ids=tuple(sorted(set(outer_train_groups_per_row[inner_valid].tolist()))),
                )
            )
        inner_plan[outer_index] = tuple(inner_folds_for_outer)

    return FoldPlan(
        seed=int(seed),
        development_group_ids=unique_groups,
        outer_folds=tuple(outer_plan),
        inner_folds=inner_plan,
    )
