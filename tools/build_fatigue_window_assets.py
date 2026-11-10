"""Create figures and slide data for the reference-aligned fatigue-window study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reference_fatigue"
FIGURES = ROOT / "docs" / "figures"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#7f8d96",
        "axes.labelcolor": "#253746",
        "xtick.color": "#465966",
        "ytick.color": "#253746",
        "text.color": "#253746",
    })


def _metrics_frame() -> pd.DataFrame:
    metrics = pd.read_csv(RESULTS / "benchmark_metrics.csv")
    labels = {
        "hist_gradient_boosting": "Global HistGradientBoosting",
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "mlp_compact": "Compact MLP",
        "svm_rbf": "SVM RBF",
        "time_only_control": "Time only control",
        "majority": "Majority baseline",
    }
    metrics = metrics[metrics.architecture.isin(labels)].copy()
    metrics["display"] = metrics.architecture.map(labels)
    metrics["kind"] = "Direct fixed model"
    if (RESULTS / "prime_mover_specialist_metrics.csv").is_file():
        row = pd.read_csv(RESULTS / "prime_mover_specialist_metrics.csv")
        row = row[row.model == "specialist_by_prime_mover"].copy()
        if not row.empty:
            row = row.iloc[[0]].rename(columns={
                "model": "architecture",
                "accuracy_participant_macro": "accuracy_participant_macro",
                "accuracy_ci95_low": "accuracy_ci95_low",
                "accuracy_ci95_high": "accuracy_ci95_high",
                "f1_macro_participant_macro": "f1_macro_participant_macro",
                "f1_macro_ci95_low": "f1_macro_ci95_low",
                "f1_macro_ci95_high": "f1_macro_ci95_high",
            })
            row["display"] = "HGB experts by prime mover"
            row["kind"] = "Exploratory specialist"
            metrics = pd.concat([metrics, row], ignore_index=True, sort=False)
    nested_path = RESULTS / "nested_hgb_metrics.csv"
    if nested_path.is_file():
        row = pd.read_csv(nested_path)
        if not row.empty:
            row = row.iloc[[0]].copy()
            row["architecture"] = "nested_hist_gradient_boosting"
            row["display"] = "Nested tuned HGB"
            row["kind"] = "Nested tuning"
            metrics = pd.concat([metrics, row], ignore_index=True, sort=False)
    order = [
        "Time only control", "Global HistGradientBoosting", "Nested tuned HGB",
        "HGB experts by prime mover", "Random forest", "Logistic regression",
        "Compact MLP", "SVM RBF", "Majority baseline",
    ]
    metrics["_order"] = metrics.display.map({name: i for i, name in enumerate(order)})
    return metrics.sort_values("_order", kind="stable").drop(columns="_order")


def _plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    view = metrics.reset_index(drop=True)
    y = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    bars = (
        ("accuracy_participant_macro", "accuracy_ci95_low", "accuracy_ci95_high", "Accuracy", -0.16, "#276b8e"),
        ("f1_macro_participant_macro", "f1_macro_ci95_low", "f1_macro_ci95_high", "Macro-F1", 0.16, "#2b987c"),
    )
    for value_col, low_col, high_col, label, shift, color in bars:
        values = view[value_col].astype(float)
        low = np.maximum(0.0, values - view[low_col].astype(float))
        high = np.maximum(0.0, view[high_col].astype(float) - values)
        ax.barh(y + shift, values, height=0.29, color=color, label=label)
        ax.errorbar(values, y + shift, xerr=np.vstack((low, high)), fmt="none", ecolor="#334e5e", capsize=2, lw=0.9)
    ax.set_yticks(y, view.display)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Media por participante de métricas OOF · 13 participantes")
    ax.set_title(
        "Clasificación por ventana contra autoreporte estable\n"
        "Intervalos 95% bootstrap por participante",
        loc="left", pad=14, weight="bold", fontsize=13,
    )
    ax.axvline(0.5, color="#9aa6ad", ls="--", lw=1)
    ax.grid(axis="x", color="#e2e8ec", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_confusions(path: Path) -> None:
    files = [
        (RESULTS / "confusion_matrices.csv", "architecture", "hist_gradient_boosting", "HGB global"),
        (RESULTS / "nested_hgb_confusion.csv", "model", "nested_hist_gradient_boosting", "HGB con búsqueda anidada"),
        (RESULTS / "prime_mover_specialist_confusions.csv", "model", "specialist_by_prime_mover", "Expertos por músculo"),
        (RESULTS / "confusion_matrices.csv", "architecture", "time_only_control", "Control solo tiempo"),
    ]
    selected = []
    for csv_path, model_column, model, title in files:
        if not csv_path.is_file():
            continue
        frame = pd.read_csv(csv_path)
        rows = frame[frame[model_column] == model]
        if rows.empty:
            continue
        matrix = np.zeros((2, 2), dtype=int)
        for row in rows.itertuples(index=False):
            actual = int(getattr(row, "actual_label"))
            predicted = int(getattr(row, "predicted_label"))
            matrix[actual, predicted] = int(getattr(row, "count"))
        selected.append((title, matrix))
    cols = 2
    rows_n = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(10.3, 4.6 * rows_n), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (title, matrix) in zip(axes, selected):
        ax.imshow(matrix, cmap="Blues")
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals > 0)
        for i in range(2):
            for j in range(2):
                color = "white" if normalized[i, j] > 0.58 else "#182a36"
                ax.text(j, i, f"{matrix[i,j]:,}\n{normalized[i,j]:.1%}", ha="center", va="center", fontsize=11, color=color, weight="bold")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks([0, 1], ["Sin fatiga percibida", "Fatiga percibida"])
        ax.set_yticks([0, 1], ["Sin fatiga percibida", "Fatiga percibida"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Referencia")
        ax.tick_params(length=0, labelsize=8)
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle("Matrices OOF · conteo y tasa normalizada por fila", fontsize=14, weight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _box(ax, x, y, w, h, title, body, edge):
    shape = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.02", fc="#f1f5f7", ec=edge, lw=1.6)
    ax.add_patch(shape)
    ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=10, weight="bold", color="#182a36")
    ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=8.2, color="#3b4f5c")


def _plot_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, 0.02, 0.72, 0.20, 0.18, "sEMG + referencia", "4 canales · 1259.26 Hz\nlabels 0 / 1 / 2 a 50 Hz", "#527c96")
    _box(ax, 0.28, 0.72, 0.20, 0.18, "Etiquetado por ventana", "500 ms estable\n0=sin percibida; 2=percibida", "#b8762d")
    _box(ax, 0.55, 0.72, 0.18, 0.18, "Preprocesamiento", "20-450 Hz SOS\n50/60 Hz solo si procede", "#527c96")
    _box(ax, 0.80, 0.72, 0.18, 0.18, "60 features", "15 por canal\ntransiciones excluidas", "#2b987c")
    for a, b in ((0.22, 0.28), (0.48, 0.55), (0.73, 0.80)):
        ax.add_patch(FancyArrowPatch((a, 0.81), (b, 0.81), arrowstyle="->", mutation_scale=12, color="#527c96", lw=1.2))
    _box(ax, 0.14, 0.38, 0.30, 0.18, "HGB global", "Recibe features sEMG\nuna predicción por ventana", "#234f6b")
    _box(ax, 0.57, 0.38, 0.30, 0.18, "HGB especialista", "Enrutado por músculo principal\nno predice fase de movimiento", "#7b5aa6")
    ax.add_patch(FancyArrowPatch((0.89, 0.72), (0.89, 0.65), arrowstyle="-", mutation_scale=12, color="#6b7f8a", lw=1.2))
    ax.plot([0.29, 0.89], [0.65, 0.65], color="#6b7f8a", lw=1.2)
    ax.add_patch(FancyArrowPatch((0.29, 0.65), (0.29, 0.56), arrowstyle="->", mutation_scale=12, color="#6b7f8a", lw=1.2))
    ax.add_patch(FancyArrowPatch((0.72, 0.65), (0.72, 0.56), arrowstyle="->", mutation_scale=12, color="#6b7f8a", lw=1.2))
    _box(ax, 0.32, 0.08, 0.36, 0.17, "Evaluación", "13 folds leave-one-participant-out\n13 sujetos bootstrap para incertidumbre", "#234f6b")
    for x in (0.29, 0.72):
        ax.add_patch(FancyArrowPatch((x, 0.38), (0.5, 0.25), arrowstyle="->", mutation_scale=12, color="#6b7f8a", lw=1.2))
    ax.text(0.5, 0.015, "La salida representa autoreporte de fatiga percibida, no pérdida mecánica de fuerza.", ha="center", fontsize=9, color="#9e4141")
    fig.suptitle("Clasificador binario entrenado contra autoreporte sincronizado por ventana", x=0.03, y=0.98, ha="left", fontsize=14, weight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build() -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = _metrics_frame()
    paths = {
        "metrics": FIGURES / "perceived_fatigue_window_metrics.png",
        "confusions": FIGURES / "perceived_fatigue_window_confusions.png",
        "architecture": FIGURES / "perceived_fatigue_window_architecture.png",
    }
    _plot_metrics(metrics, paths["metrics"])
    _plot_confusions(paths["confusions"])
    _plot_architecture(paths["architecture"])
    confusion = pd.read_csv(RESULTS / "confusion_matrices.csv")
    specialist_confusion = pd.read_csv(RESULTS / "prime_mover_specialist_confusions.csv")
    matrix_summaries = {}
    for model, frame, column in (
        ("hist_gradient_boosting", confusion, "architecture"),
        ("time_only_control", confusion, "architecture"),
        ("specialist_by_prime_mover", specialist_confusion, "model"),
    ):
        subset = frame[frame[column] == model].sort_values(["actual_label", "predicted_label"])
        matrix_summaries[model] = [
            [int(subset[(subset.actual_label == actual) & (subset.predicted_label == predicted)].iloc[0]["count"]) for predicted in (0, 1)]
            for actual in (0, 1)
        ]
    if (RESULTS / "nested_hgb_confusion.csv").is_file():
        nested = pd.read_csv(RESULTS / "nested_hgb_confusion.csv").sort_values(["actual_label", "predicted_label"])
        matrix_summaries["nested_hist_gradient_boosting"] = [
            [int(nested[(nested.actual_label == actual) & (nested.predicted_label == predicted)].iloc[0]["count"]) for predicted in (0, 1)]
            for actual in (0, 1)
        ]
    metrics_out = []
    for row in metrics.itertuples(index=False):
        metrics_out.append({
            "architecture": str(row.architecture),
            "display": str(row.display),
            "accuracy": float(row.accuracy),
            "macro_f1": float(row.f1_macro),
            "accuracy_participant_macro": float(row.accuracy_participant_macro),
            "macro_f1_participant_macro": float(row.f1_macro_participant_macro),
            "accuracy_ci95": [float(row.accuracy_ci95_low), float(row.accuracy_ci95_high)],
            "macro_f1_ci95": [float(row.f1_macro_ci95_low), float(row.f1_macro_ci95_high)],
        })
    summary = {
        "dataset": {
            "doi": "10.5281/zenodo.14182446",
            "version": "v2",
            "participants": 13,
            "trials": 156,
            "channels": 4,
            "window_count": 125165,
            "labels": {"0": 64283, "1": 60882},
            "label_semantics": "stable self-report 0=no perceived fatigue; stable self-report 2=perceived fatigue; transition level 1 excluded",
            "sampling_rate_hz": 1259.26,
            "window_s": 0.5,
            "overlap": 0.5,
            "features": 60,
        },
        "models": metrics_out,
        "confusion_matrices": matrix_summaries,
        "specialist_comparison": {
            row.metric: {key: getattr(row, key) for key in ("mean_paired_difference", "ci95_low", "ci95_high")}
            for row in pd.read_csv(RESULTS / "prime_mover_specialist_paired_differences.csv").itertuples(index=False)
        },
        "nested_tuning_status": "completed" if (RESULTS / "nested_hgb_metrics.csv").is_file() else "running",
        "phase_cascade_status": "not evaluated: no synchronized extension/flexion annotations in this dataset",
        "scope_note": "Reference-aligned self-reported fatigue classification; not objective mechanical muscle fatigue",
    }
    summary_path = RESULTS / "perceived_fatigue_slides.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {
        "status": "built",
        "source_hashes": {
            name: _sha256(RESULTS / name)
            for name in (
                "benchmark_metrics.csv", "confusion_matrices.csv",
                "prime_mover_specialist_metrics.csv", "prime_mover_specialist_confusions.csv",
                "prime_mover_specialist_paired_differences.csv",
            )
        },
        "figure_hashes": {key: _sha256(path) for key, path in paths.items()},
        "summary_sha256": _sha256(summary_path),
    }
    (RESULTS / "perceived_fatigue_assets_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    _style()
    print(json.dumps(build(), indent=2))
