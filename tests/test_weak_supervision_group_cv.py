from __future__ import annotations

import pytest

from emg_fatigue.weak_supervision.group_cv import group_folds


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
