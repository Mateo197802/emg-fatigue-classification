"""Deterministic nested group folds for development-only model selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
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


def group_folds_with_class_coverage(
    group_ids: Sequence[str],
    recording_ids: Sequence[str],
    class_ids: Sequence[int],
    outer_folds: int = 3,
    inner_folds: int = 2,
    seed: int = 20261113,
    min_groups_per_class: int = 2,
    min_trials_per_class: int = 2,
) -> FoldPlan:
    """Build nested folds balanced on class support by group and recording.

    Each group contributes at most one support unit per class to fold assignment,
    regardless of how many overlapping windows it contains. A trial is a
    ``(group_id, recording_id)`` pair. Both training and validation sides of
    every outer and inner fold must meet the requested independent-group and
    trial minima for both classes.
    """
    raw_groups = np.asarray(group_ids, dtype=object)
    raw_recordings = np.asarray(recording_ids, dtype=object)
    raw_classes = np.asarray(class_ids)
    if raw_groups.ndim != 1 or raw_groups.size == 0:
        raise ValueError("group_ids must be a non-empty one-dimensional sequence")
    if raw_recordings.shape != raw_groups.shape or raw_classes.shape != raw_groups.shape:
        raise ValueError("group_ids, recording_ids, and class_ids must have equal lengths")
    if any(value is None or (isinstance(value, float) and np.isnan(value)) for value in raw_groups):
        raise ValueError("group_ids cannot contain missing values")
    if any(value is None or (isinstance(value, float) and np.isnan(value)) for value in raw_recordings):
        raise ValueError("recording_ids cannot contain missing values")
    if not np.isin(raw_classes, (-1, 0, 1)).all():
        raise ValueError("class_ids must use only -1, 0, and 1")

    groups = tuple(sorted({str(value) for value in raw_groups.tolist()}))
    outer_count = int(outer_folds)
    inner_count = int(inner_folds)
    min_group_count = int(min_groups_per_class)
    min_trial_count = int(min_trials_per_class)
    if outer_count < 2 or inner_count < 2:
        raise ValueError("outer_folds and inner_folds must both be at least 2")
    if min_group_count < 1 or min_trial_count < 1:
        raise ValueError("coverage minima must be positive")
    if len(groups) < outer_count:
        raise ValueError(f"Need at least {outer_count} distinct groups for {outer_count} outer folds")
    if len(groups) % outer_count:
        raise ValueError("Coverage-aware fold assignment requires equal-sized outer group folds")
    outer_valid_size = len(groups) // outer_count
    outer_train_size = len(groups) - outer_valid_size
    if outer_train_size % inner_count:
        raise ValueError("Coverage-aware fold assignment requires equal-sized inner group folds")
    inner_valid_size = outer_train_size // inner_count

    trials_by_group: dict[str, dict[int, set[str]]] = {
        group: {0: set(), 1: set()} for group in groups
    }
    for group, recording, class_id in zip(raw_groups, raw_recordings, raw_classes, strict=True):
        label = int(class_id)
        if label in (0, 1):
            group_name = str(group)
            trials_by_group[group_name][label].add(str(recording))

    def support(group_set: Sequence[str]) -> dict[int, tuple[int, int]]:
        selected = tuple(sorted(set(str(value) for value in group_set)))
        return {
            class_id: (
                sum(bool(trials_by_group[group][class_id]) for group in selected),
                sum(len(trials_by_group[group][class_id]) for group in selected),
            )
            for class_id in (0, 1)
        }

    def has_coverage(group_set: Sequence[str]) -> bool:
        return all(
            group_count >= min_group_count and trial_count >= min_trial_count
            for group_count, trial_count in support(group_set).values()
        )

    def balanced_partitions(values: Sequence[str], fold_count: int, fold_size: int):
        ordered = tuple(sorted(values))
        if fold_count == 1:
            yield (ordered,)
            return
        first = ordered[0]
        for remainder in combinations(ordered[1:], fold_size - 1):
            first_fold = tuple(sorted((first, *remainder)))
            remaining = tuple(value for value in ordered if value not in first_fold)
            for rest in balanced_partitions(remaining, fold_count - 1, fold_size):
                yield (first_fold, *rest)

    def imbalance(folds: Sequence[Sequence[str]]) -> int:
        totals = [support(fold) for fold in folds]
        return sum(
            10 * (max(item[class_id][0] for item in totals) - min(item[class_id][0] for item in totals))
            + max(item[class_id][1] for item in totals) - min(item[class_id][1] for item in totals)
            for class_id in (0, 1)
        )

    def tie_key(value: object) -> str:
        return hashlib.sha256(f"{int(seed)}:{value!r}".encode("utf-8")).hexdigest()

    feasible_plans: list[
        tuple[int, str, tuple[tuple[str, ...], ...], tuple[tuple[tuple[str, ...], ...], ...]]
    ] = []
    for outer_validation in balanced_partitions(groups, outer_count, outer_valid_size):
        if any(
            not has_coverage(valid)
            or not has_coverage(tuple(group for group in groups if group not in valid))
            for valid in outer_validation
        ):
            continue

        inner_validation_by_outer: list[tuple[tuple[str, ...], ...]] = []
        feasible = True
        for outer_valid in outer_validation:
            outer_train = tuple(group for group in groups if group not in outer_valid)
            inner_candidates: list[tuple[int, str, tuple[tuple[str, ...], ...]]] = []
            for inner_validation in balanced_partitions(outer_train, inner_count, inner_valid_size):
                inner_train_by_fold = [
                    tuple(group for group in outer_train if group not in validation)
                    for validation in inner_validation
                ]
                if not all(
                    has_coverage(validation) and has_coverage(train)
                    for validation, train in zip(inner_validation, inner_train_by_fold, strict=True)
                ):
                    continue
                inner_candidates.append(
                    (
                        imbalance(inner_validation),
                        tie_key(inner_validation),
                        inner_validation,
                    )
                )
            if not inner_candidates:
                feasible = False
                break
            inner_candidates.sort(key=lambda item: (item[0], item[1]))
            inner_validation_by_outer.append(inner_candidates[0][2])
        if feasible:
            score = imbalance(outer_validation) + sum(
                imbalance(inner) for inner in inner_validation_by_outer
            )
            feasible_plans.append(
                (score, tie_key((outer_validation, inner_validation_by_outer)), outer_validation, tuple(inner_validation_by_outer))
            )

    if not feasible_plans:
        raise ValueError(
            f"Labels cannot support {outer_count} outer x {inner_count} inner group folds "
            f"with at least {min_group_count} groups and {min_trial_count} trials per class"
        )

    feasible_plans.sort(key=lambda item: (item[0], item[1]))
    _, _, selected_outer, selected_inner = feasible_plans[0]
    outer_plan: list[GroupFold] = []
    inner_plan: dict[int, tuple[GroupFold, ...]] = {}
    for outer_id, outer_valid in enumerate(selected_outer, start=1):
        outer_train = tuple(group for group in groups if group not in outer_valid)
        outer_plan.append(GroupFold(outer_id, outer_train, outer_valid))
        inner_folds_for_outer: list[GroupFold] = []
        for inner_id, inner_valid in enumerate(selected_inner[outer_id - 1], start=1):
            inner_train = tuple(group for group in outer_train if group not in inner_valid)
            inner_folds_for_outer.append(GroupFold(inner_id, inner_train, inner_valid))
        inner_plan[outer_id] = tuple(inner_folds_for_outer)

    return FoldPlan(
        seed=int(seed),
        development_group_ids=groups,
        outer_folds=tuple(outer_plan),
        inner_folds=inner_plan,
    )
