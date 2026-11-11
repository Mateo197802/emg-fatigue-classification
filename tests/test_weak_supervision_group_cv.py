from __future__ import annotations

import pytest

from emg_fatigue.weak_supervision.group_cv import (
    group_folds,
    group_folds_with_class_coverage,
)


def test_nested_group_folds_partition_groups_without_overlap_and_are_reproducible():
    groups = [f"G{index:02d}" for index in range(12) for _ in range(index + 1)]

    plan = group_folds(groups, outer_folds=4, inner_folds=3, seed=271)
    repeated = group_folds(groups, outer_folds=4, inner_folds=3, seed=271)

    outer_validation = [set(fold.validation_group_ids) for fold in plan.outer_folds]
    assert len(plan.outer_folds) == 4
    assert [fold.validation_group_ids for fold in plan.outer_folds] == [
        fold.validation_group_ids for fold in repeated.outer_folds
    ]
    assert set.union(*outer_validation) == set(groups)
    assert sum(len(values) for values in outer_validation) == len(set(groups))

    for fold in plan.outer_folds:
        assert set(fold.train_group_ids).isdisjoint(fold.validation_group_ids)
        inner = plan.inner_folds[fold.fold_id]
        inner_validation = [set(item.validation_group_ids) for item in inner]
        assert len(inner) == 3
        assert set.union(*inner_validation) == set(fold.train_group_ids)
        assert sum(len(values) for values in inner_validation) == len(fold.train_group_ids)
        for item in inner:
            assert set(item.train_group_ids).isdisjoint(item.validation_group_ids)
            assert set(item.train_group_ids).union(item.validation_group_ids) == set(fold.train_group_ids)


def test_group_folds_reject_too_few_distinct_groups():
    with pytest.raises(ValueError, match="at least 4 distinct groups"):
        group_folds(["G1", "G2", "G3"], outer_folds=4, inner_folds=2)


def _sparse_proxy_trial_rows():
    trials_per_class = {
        "G01": {0: 1, 1: 0},
        "G02": {0: 2, 1: 1},
        "G03": {0: 3, 1: 1},
        "G05": {0: 0, 1: 1},
        "G06": {0: 0, 1: 0},
        "G07": {0: 0, 1: 0},
        "G08": {0: 0, 1: 0},
        "G09": {0: 1, 1: 0},
        "G11": {0: 2, 1: 0},
        "G12": {0: 3, 1: 1},
        "G14": {0: 2, 1: 1},
        "G15": {0: 2, 1: 2},
    }
    groups = []
    recordings = []
    classes = []
    for group, support in trials_per_class.items():
        for class_id, trial_count in support.items():
            for trial_index in range(trial_count):
                groups.append(group)
                recordings.append(f"{group}_F{trial_index + 1:03d}")
                classes.append(class_id)
    # Unlabeled development groups remain in the plan, even with no assigned proxy labels.
    for group in ("G06", "G07", "G08"):
        groups.append(group)
        recordings.append(f"{group}_UNLABELED")
        classes.append(-1)
    return groups, recordings, classes


def _assert_minimum_class_coverage(groups, recordings, classes):
    for class_id in (0, 1):
        selected = [
            (group, recording)
            for group, recording, label in zip(groups, recordings, classes, strict=True)
            if label == class_id
        ]
        assert len({group for group, _ in selected}) >= 2
        assert len({(group, recording) for group, recording in selected}) >= 2


def test_coverage_aware_nested_folds_balance_independent_groups_not_window_counts():
    groups, recordings, classes = _sparse_proxy_trial_rows()
    plan = group_folds_with_class_coverage(
        groups,
        recordings,
        classes,
        outer_folds=3,
        inner_folds=2,
        seed=20261113,
    )

    repeated = group_folds_with_class_coverage(
        groups,
        recordings,
        classes,
        outer_folds=3,
        inner_folds=2,
        seed=20261113,
    )
    assert [fold.validation_group_ids for fold in plan.outer_folds] == [
        fold.validation_group_ids for fold in repeated.outer_folds
    ]
    assert len(plan.outer_folds) == 3

    outer_validation_groups = []
    for outer in plan.outer_folds:
        outer_validation_groups.extend(outer.validation_group_ids)
        _assert_minimum_class_coverage(
            [g for g, label in zip(groups, classes, strict=True) if label in (0, 1) and g in outer.validation_group_ids],
            [r for g, r, label in zip(groups, recordings, classes, strict=True) if label in (0, 1) and g in outer.validation_group_ids],
            [label for g, label in zip(groups, classes, strict=True) if label in (0, 1) and g in outer.validation_group_ids],
        )
        _assert_minimum_class_coverage(
            [g for g, label in zip(groups, classes, strict=True) if label in (0, 1) and g in outer.train_group_ids],
            [r for g, r, label in zip(groups, recordings, classes, strict=True) if label in (0, 1) and g in outer.train_group_ids],
            [label for g, label in zip(groups, classes, strict=True) if label in (0, 1) and g in outer.train_group_ids],
        )
        assert set(outer.train_group_ids).isdisjoint(outer.validation_group_ids)
        assert len(outer.validation_group_ids) == 4

        inner_validation_groups = []
        for inner in plan.inner_folds[outer.fold_id]:
            inner_validation_groups.extend(inner.validation_group_ids)
            for role_groups in (inner.train_group_ids, inner.validation_group_ids):
                role_rows = [
                    (g, r, label)
                    for g, r, label in zip(groups, recordings, classes, strict=True)
                    if g in role_groups and label in (0, 1)
                ]
                _assert_minimum_class_coverage(
                    [row[0] for row in role_rows],
                    [row[1] for row in role_rows],
                    [row[2] for row in role_rows],
                )
        assert set(inner_validation_groups) == set(outer.train_group_ids)
        assert len(inner_validation_groups) == len(set(inner_validation_groups))

    assert set(outer_validation_groups) == set(groups)
    assert len(outer_validation_groups) == len(set(outer_validation_groups))

    # Replicating overlapping windows within a trial must not alter group assignment.
    replicated_groups = groups + ["G15"] * 100
    replicated_recordings = recordings + ["G15_F002"] * 100
    replicated_classes = classes + [1] * 100
    replicated_plan = group_folds_with_class_coverage(
        replicated_groups,
        replicated_recordings,
        replicated_classes,
        outer_folds=3,
        inner_folds=2,
        seed=20261113,
    )
    assert [fold.validation_group_ids for fold in plan.outer_folds] == [
        fold.validation_group_ids for fold in replicated_plan.outer_folds
    ]


def test_coverage_aware_nested_folds_reject_insufficient_independent_positive_groups():
    groups, recordings, classes = _sparse_proxy_trial_rows()
    classes = [0 if class_id == 1 and group != "G15" else class_id for group, class_id in zip(groups, classes, strict=True)]

    with pytest.raises(ValueError, match="cannot support 3 outer x 2 inner group folds"):
        group_folds_with_class_coverage(
            groups,
            recordings,
            classes,
            outer_folds=3,
            inner_folds=2,
            seed=20261113,
        )
