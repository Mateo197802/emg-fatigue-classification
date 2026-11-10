import pandas as pd
import pytest


def _predictions():
    rows = []
    examples = {
        "group-a": ([0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, 1]),
        "group-b": ([0, 1], [1, 1], [0, 1]),
    }
    for group, (truth, baseline, candidate) in examples.items():
        for row_index, (actual, base_pred, candidate_pred) in enumerate(
            zip(truth, baseline, candidate, strict=True)
        ):
            for architecture, predicted in (
                ("direct", base_pred),
                ("cascade", candidate_pred),
            ):
                rows.append({
                    "architecture": architecture,
                    "group_id": group,
                    "row_index": f"{group}-{row_index}",
                    "y_true": actual,
                    "y_pred": predicted,
                    "score": float(predicted),
                })
    return pd.DataFrame(rows)


def test_group_summary_averages_groups_as_the_uncertainty_unit():
    from emg_fatigue.evaluation import summarize_grouped_predictions

    result = summarize_grouped_predictions(_predictions(), bootstrap_replicates=200, seed=2)
    direct_accuracy = result.query(
        "architecture == 'direct' and metric == 'accuracy'"
    ).iloc[0]

    assert direct_accuracy["mean"] == pytest.approx((0.75 + 0.5) / 2)
    assert direct_accuracy["pooled_support"] == 6
    assert direct_accuracy["n_groups"] == 2
    assert direct_accuracy["uncertainty_unit"] == "group_id"


def test_paired_group_differences_compare_models_on_identical_groups():
    from emg_fatigue.evaluation import paired_group_differences

    result = paired_group_differences(
        _predictions(),
        baseline="direct",
        candidates=["cascade"],
        metrics=("accuracy", "f1_macro"),
        bootstrap_replicates=200,
        seed=2,
    )
    accuracy = result.query("candidate == 'cascade' and metric == 'accuracy'").iloc[0]

    assert accuracy["mean_difference"] == pytest.approx((0.25 + 0.5) / 2)
    assert accuracy["n_paired_groups"] == 2
    assert accuracy["ci95_lower"] <= accuracy["mean_difference"] <= accuracy["ci95_upper"]


def test_paired_group_differences_reject_unmatched_group_support():
    from emg_fatigue.evaluation import paired_group_differences

    data = _predictions()
    data = data[~((data.architecture == "cascade") & (data.group_id == "group-b"))]
    with pytest.raises(ValueError, match="identical group support"):
        paired_group_differences(
            data,
            baseline="direct",
            candidates=["cascade"],
            metrics=("accuracy",),
            bootstrap_replicates=20,
        )


def test_confusion_matrix_export_contains_counts_denominators_and_row_rates():
    from emg_fatigue.evaluation import confusion_matrix_table

    result = confusion_matrix_table(_predictions())
    direct_class0 = result.query(
        "architecture == 'direct' and reference_class == 0 and predicted_class == 0"
    ).iloc[0]
    direct_class1 = result.query(
        "architecture == 'direct' and reference_class == 1 and predicted_class == 1"
    ).iloc[0]

    assert direct_class0["count"] == 2
    assert direct_class0["reference_support"] == 3
    assert direct_class0["row_fraction"] == pytest.approx(2 / 3)
    assert direct_class1["count"] == 2
    assert direct_class1["reference_support"] == 3
    assert direct_class1["row_fraction"] == pytest.approx(2 / 3)
