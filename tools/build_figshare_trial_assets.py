"""Build plots and slide-ready summaries from frozen trial-level benchmark outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reference_fatigue" / "figshare_task_level"
PROCESSED = ROOT / "private" / "figshare_task_failure_2021" / "processed"
FIGURES = ROOT / "docs" / "figures"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _set_style() -> None:
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


def plot_model_metrics(metrics: pd.DataFrame, path: Path) -> None:
    names = [
        "time_only_control", "lr_rms_mdf", "lr_all_features", "svm_rbf",
        "random_forest", "mlp_compact", "cnn1d_trial_mil", "dummy_prior",
        "hist_gradient_boosting",
    ]
    labels = {
        "time_only_control": "Control: tiempo y ráfagas",
        "lr_rms_mdf": "Logística: RMS + MDF",
        "lr_all_features": "Logística: 8 features EMG",
        "svm_rbf": "SVM RBF",
        "random_forest": "Random Forest",
        "mlp_compact": "MLP compacto",
        "cnn1d_trial_mil": "CNN-1D, pooling por ensayo",
        "dummy_prior": "Dummy mayoritario",
        "hist_gradient_boosting": "HistGradientBoosting",
    }
    view = metrics.set_index("model").loc[names].reset_index()
    y = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(10.4, 6.2), constrained_layout=True)
    acc_low = view["accuracy"] - view["accuracy_ci95_low"]
    acc_high = view["accuracy_ci95_high"] - view["accuracy"]
    f1_low = view["f1_macro"] - view["f1_macro_ci95_low"]
    f1_high = view["f1_macro_ci95_high"] - view["f1_macro"]
    ax.barh(y - 0.18, view["accuracy"], height=0.32, color="#276b8e", label="Accuracy")
    ax.errorbar(
        view["accuracy"], y - 0.18,
        xerr=np.vstack((acc_low, acc_high)), fmt="none", ecolor="#173f55", capsize=2, lw=1,
    )
    ax.barh(y + 0.18, view["f1_macro"], height=0.32, color="#2b987c", label="Macro-F1")
    ax.errorbar(
        view["f1_macro"], y + 0.18,
        xerr=np.vstack((f1_low, f1_high)), fmt="none", ecolor="#145a48", capsize=2, lw=1,
    )
    ax.set_yticks(y, [labels[name] for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("Métrica OOF, una predicción por ensayo (n = 28)")
    ax.set_title(
        "Resultado del ensayo, no estado instantáneo de fatiga\n"
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


def plot_confusion_matrices(confusions: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    models = ["lr_all_features", "time_only_control", "cnn1d_trial_mil"]
    titles = {
        "lr_all_features": "EMG: regresión logística",
        "time_only_control": "Control: solo estructura temporal",
        "cnn1d_trial_mil": "CNN-1D: pooling por ensayo",
    }
    metric_index = metrics.set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.2), constrained_layout=True)
    for ax, model in zip(axes, models):
        rows = confusions[confusions["model"] == model]
        matrix = np.zeros((2, 2), dtype=int)
        for row in rows.itertuples(index=False):
            matrix[int(row.true_label), int(row.predicted_label)] = int(row.count)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=14)
        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i, j] >= 9 else "#182a36"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=16, color=color, weight="bold")
        result = metric_index.loc[model]
        ax.set_title(f"{titles[model]}\nAcc {result.accuracy:.3f} | Macro-F1 {result.f1_macro:.3f}", fontsize=10, weight="bold")
        ax.set_xticks([0, 1], ["Ref.", "TLim"])
        ax.set_yticks([0, 1], ["Ref.", "TLim"])
        ax.set_xlabel("Predicción")
        if ax is axes[0]:
            ax.set_ylabel("Etiqueta real")
        ax.tick_params(length=0)
    fig.suptitle("Matrices de confusión OOF por ensayo (14 participantes retenidos)", fontsize=14, weight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_trial_structure(trials: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    conditions = ["30", "TLim"]
    colors = {"30": "#276b8e", "TLim": "#b8762d"}
    for subject, group in trials.groupby("subject_id", sort=True):
        ordered = group.set_index("condition").reindex(conditions)
        if ordered["active_duration_s"].notna().all():
            axes[0].plot([0, 1], ordered["active_duration_s"], color="#c6d1d7", lw=0.7, zorder=1)
    for i, condition in enumerate(conditions):
        group = trials[trials["condition"] == condition]
        x = np.full(len(group), i, dtype=float) + np.linspace(-0.06, 0.06, len(group))
        axes[0].scatter(x, group["active_duration_s"], color=colors[condition], s=30, zorder=2, label=condition)
        axes[1].scatter(x, group["active_bout_count"], color=colors[condition], s=30, zorder=2, label=condition)
    axes[0].set_title("Duración activa por ensayo")
    axes[0].set_ylabel("Segundos")
    axes[1].set_title("Número de ráfagas activas")
    axes[1].set_ylabel("Ráfagas")
    for ax in axes:
        ax.set_xticks([0, 1], ["30: referencia", "TLim: tarea al fallo"])
        ax.grid(axis="y", color="#e2e8ec")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("La estructura del protocolo es claramente diferente entre etiquetas", fontsize=14, weight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _box(ax, x, y, w, h, title, body, color):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.02", fc="#f1f5f7", ec=color, lw=1.4)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=10, weight="bold", color="#182a36")
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=8.1, color="#3b4f5c", wrap=True)


def plot_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Data and trial target
    _box(ax, 0.02, 0.72, 0.19, 0.19, "MAT por ensayo", "65 posiciones RF\nref_signal solo segmenta", "#527c96")
    _box(ax, 0.28, 0.72, 0.20, 0.19, "Preprocesamiento", "20-450 Hz SOS\n20% fuerza / 10 s mínimo", "#527c96")
    _box(ax, 0.55, 0.72, 0.18, 0.19, "Ventanas", "0.5 s = 1024 muestras\n50% solapamiento", "#527c96")
    _box(ax, 0.80, 0.72, 0.18, 0.19, "Etiqueta por ensayo", "30 = referencia\nTLim = tarea al fallo", "#b8762d")
    for x1, x2 in ((0.21, 0.28), (0.48, 0.55)):
        ax.add_patch(FancyArrowPatch((x1, 0.815), (x2, 0.815), arrowstyle="->", mutation_scale=13, color="#527c96", lw=1.3))
    ax.text(0.79, 0.64, "Una etiqueta por ensayo; no se replica por ventana", ha="center", fontsize=8.8, color="#9e4141", weight="bold")
    # Parallel model families
    _box(ax, 0.10, 0.39, 0.32, 0.18, "Modelos de características", "mediana + IQR + tendencia\npor ensayo: RMS/MDF y 8 features", "#2b987c")
    _box(ax, 0.57, 0.39, 0.32, 0.18, "CNN-1D MIL", "8 ventanas distribuidas\nlogits por ventana -> promedio", "#7b5aa6")
    _box(ax, 0.32, 0.10, 0.36, 0.17, "Validación y salida", "LOPO: 14 sujetos, 28 ensayos\n1 score OOF por ensayo", "#234f6b")
    for start, end in (((0.64, 0.72), (0.26, 0.57)), ((0.73, 0.72), (0.73, 0.57)), ((0.26, 0.39), (0.42, 0.27)), ((0.73, 0.39), (0.58, 0.27))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=13, color="#6b7f8a", lw=1.3))
    ax.text(0.5, 0.015, "Control temporal separado: duración, número y longitud de ráfagas", ha="center", fontsize=9, color="#9e4141")
    fig.suptitle("Arquitectura entrenada: salida a nivel de ensayo", x=0.03, y=0.98, ha="left", fontsize=14, weight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build() -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "benchmark_results.csv")
    confusions = pd.read_csv(RESULTS / "confusion_matrix.csv")
    trials = pd.read_csv(PROCESSED / "trial_features.csv")
    plot_model_metrics(metrics, FIGURES / "figshare_trial_outcome_metrics.png")
    plot_confusion_matrices(confusions, metrics, FIGURES / "figshare_trial_outcome_confusion.png")
    plot_trial_structure(trials, FIGURES / "figshare_trial_outcome_structure.png")
    plot_architecture(FIGURES / "figshare_trial_outcome_architecture.png")

    selected_models = ["time_only_control", "lr_rms_mdf", "lr_all_features", "svm_rbf", "random_forest", "mlp_compact", "cnn1d_trial_mil", "dummy_prior", "hist_gradient_boosting"]
    model_rows = metrics.set_index("model")
    result_summary = {}
    for model in selected_models:
        row = model_rows.loc[model]
        result_summary[model] = {
            "n_trials": int(row.n_trials),
            "accuracy": float(row.accuracy),
            "macro_f1": float(row.f1_macro),
            "balanced_accuracy": float(row.balanced_accuracy),
            "sensitivity_task_failure": float(row.sensitivity_task_failure),
            "specificity_reference": float(row.specificity_reference),
            "accuracy_ci95": [float(row.accuracy_ci95_low), float(row.accuracy_ci95_high)],
            "macro_f1_ci95": [float(row.f1_macro_ci95_low), float(row.f1_macro_ci95_high)],
        }
    confusion_summary = {}
    for model in ("lr_all_features", "time_only_control", "cnn1d_trial_mil"):
        matrix = confusions[confusions.model == model].sort_values(["true_label", "predicted_label"])
        confusion_summary[model] = [
            [int(matrix[(matrix.true_label == i) & (matrix.predicted_label == j)].iloc[0]["count"]) for j in (0, 1)]
            for i in (0, 1)
        ]
    slide_summary = {
        "title_es": "Ensayo externo por resultado de tarea al fallo",
        "dataset": {
            "doi": "10.6084/m9.figshare.15273906.v1",
            "participants": int(trials.subject_id.nunique()),
            "trials": int(len(trials)),
            "trials_per_class": {str(int(label)): int(count) for label, count in trials.label.value_counts().sort_index().items()},
            "sampling_rate_hz": sorted({int(value) for value in trials.sampling_rate_hz.unique()}),
            "window_s": 0.5,
            "overlap_fraction": 0.5,
            "window_count_median_by_condition": {condition: float(trials.loc[trials.condition == condition, "eligible_window_count"].median()) for condition in ("30", "TLim")},
            "active_bout_count_by_condition": {condition: sorted(set(trials.loc[trials.condition == condition, "active_bout_count"].astype(int))) for condition in ("30", "TLim")},
        },
        "results": result_summary,
        "confusion_matrices": confusion_summary,
        "target_semantics": "one trial-level label: 30 reference contraction versus TLim task carried to task failure; not per-window fatigue state",
        "phase_cascade_status": "not evaluated: this isometric dataset does not supply extension/flexion phase labels",
        "source_controls": "uncalibrated ref_signal only segments active bouts; time-only control is diagnostic, not the EMG model",
        "literature": [
            {"citation": "Rossato et al., Journal of Neurophysiology (2022), DOI 10.1152/jn.00453.2021", "url": "https://journals.physiology.org/doi/10.1152/jn.00453.2021"},
            {"citation": "Zhu et al., Applied Sciences 11(9), 3821 (2021), DOI 10.3390/app11093821", "url": "https://www.mdpi.com/2076-3417/11/9/3821"},
            {"citation": "Hargrove et al., A Strategy for Identifying Locomotion Modes Using Surface EMG", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3025288/"},
        ],
    }
    summary_path = RESULTS / "slide_summary.json"
    summary_path.write_text(json.dumps(slide_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {
        "status": "built",
        "source_hashes": {name: _sha256(RESULTS / name) for name in ("benchmark_results.csv", "confusion_matrix.csv", "run_manifest.json")},
        "output_hashes": {name: _sha256(FIGURES / name) for name in (
            "figshare_trial_outcome_metrics.png",
            "figshare_trial_outcome_confusion.png",
            "figshare_trial_outcome_structure.png",
            "figshare_trial_outcome_architecture.png",
        )},
        "slide_summary_sha256": _sha256(summary_path),
    }
    (RESULTS / "visual_assets_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    _set_style()
    result = build()
    print(json.dumps(result, indent=2))
