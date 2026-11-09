"""Build Spanish and English diagrams separating the executed proxy from the target model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _save_figure(fig, svg_path, png_path):
    """Write compact SVG without trailing path whitespace plus a PNG."""
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_path = Path(svg_path)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=220)


COLORS = {
    "signal": "#DCEAF7",
    "reference": "#E5F2E8",
    "model": "#FFF0D8",
    "proxy": "#F9DEDE",
    "neutral": "#EDF0F4",
    "ink": "#203346",
    "blue": "#356A8A",
    "green": "#467A59",
    "orange": "#A56C17",
    "red": "#9E4141",
    "gray": "#6E7A86",
}


def _box(ax, x, y, w, h, text, color, *, title=None, fontsize=10, edge=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.35,
        edgecolor=edge or COLORS["ink"],
        facecolor=COLORS[color],
    )
    ax.add_patch(patch)
    if title:
        ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center",
                fontsize=fontsize + 0.5, weight="bold", color=COLORS["ink"])
        ax.text(x + w / 2, y + h * 0.34, text, ha="center", va="center",
                fontsize=fontsize - 0.7, color=COLORS["ink"], wrap=True)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=COLORS["ink"], wrap=True)
    return (x, y, w, h)


def _arrow(ax, start, end, *, dashed=False, color=None, bend=0.0, width=1.5):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": width,
            "color": color or COLORS["blue"],
            "linestyle": "--" if dashed else "-",
            "connectionstyle": f"arc3,rad={bend}",
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def _canvas(title: str, status: str):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.035, 0.965, title, fontsize=19, weight="bold", color=COLORS["ink"], va="top", transform=ax.transAxes)
    ax.text(0.965, 0.963, status, fontsize=10, weight="bold", color=COLORS["red"],
            ha="right", va="top", bbox={"boxstyle": "round,pad=0.35", "fc": "#F9DEDE", "ec": "#D7A2A2"}, transform=ax.transAxes)
    return fig, ax


def draw_executed_proxy(output_dir: Path, lang: str = "es") -> list[Path]:
    en = lang == "en"
    title = "Executed experiment: proxy classification" if en else "Experimento ejecutado: clasificación proxy"
    status = "EXECUTED · NOT VALIDATED FATIGUE" if en else "EJECUTADO · NO ES FATIGA VALIDADA"
    fig, ax = _canvas(title, status)
    ax.text(0.035, 0.875, "Common sEMG flow" if en else "Flujo común de sEMG", fontsize=12, weight="bold", color=COLORS["blue"])

    boxes = [
        _box(ax, 0.035, 0.68, 0.17, 0.13, "EMG CSV\ntime + ADC counts" if en else "CSV de EMG\ntiempo + cuentas ADC", "signal"),
        _box(ax, 0.245, 0.68, 0.18, 0.13, "QC + SOS filter\n20–400 Hz" if en else "QC + filtro SOS\n20–400 Hz", "signal"),
        _box(ax, 0.465, 0.68, 0.19, 0.13, "Fixed windows\n250 ms · 50% overlap" if en else "Ventanas fijas\n250 ms · 50% solape", "signal"),
        _box(ax, 0.695, 0.68, 0.235, 0.13, "10 features\nRMS · MAV · WL/s · MDF · MNF · ratio · wavelet" if en else "10 features\nRMS · MAV · WL/s · MDF · MNF · ratio · wavelet", "signal", fontsize=9.4),
    ]
    for left, right in zip(boxes, boxes[1:]):
        _arrow(ax, (left[0] + left[2], left[1] + left[3] / 2), (right[0], right[1] + right[3] / 2))

    ax.text(0.035, 0.58, "Compared procedures for the early/late target" if en else "Procedimientos comparados sobre el objetivo temprano/tardío", fontsize=12,
            weight="bold", color=COLORS["ink"])
    ax.text(0.965, 0.54, "Both paths use the same X matrix of 10 features" if en else "Ambas rutas consumen la misma matriz X de 10 features", fontsize=9.3,
            ha="right", color=COLORS["gray"])
    direct_x = _box(ax, 0.07, 0.39, 0.23, 0.12, "Direct model\nlogistic · SVM · RF" if en else "Modelo directo\nlogística · SVM · RF", "model")
    direct_y = _box(ax, 0.68, 0.39, 0.25, 0.12, "Proxy output\nfirst / last third" if en else "Salida proxy\nprimero / último tercio", "proxy")
    aux1 = _box(ax, 0.07, 0.16, 0.21, 0.13, "Auxiliary RF\nburst half 1 / half 2" if en else "RF auxiliar\nmitad 1 / mitad 2 de ráfaga", "model", fontsize=9.5)
    aux2 = _box(ax, 0.34, 0.16, 0.16, 0.13, "Burst-half score\nOOF" if en else "Score de mitad\nde ráfaga · OOF", "neutral", fontsize=9.5)
    aux3 = _box(ax, 0.55, 0.16, 0.23, 0.13, "Conditioned logistic\nfeatures + score interactions" if en else "Logística condicionada\nfeatures + score e interacciones", "model", fontsize=9.3)
    aux_y = _box(ax, 0.81, 0.16, 0.17, 0.13, "Proxy output\nfirst / last third" if en else "Salida proxy\nprimero / último tercio", "proxy", fontsize=9.1)

    _arrow(ax, (direct_x[0] + direct_x[2], direct_x[1] + direct_x[3] / 2),
           (direct_y[0], direct_y[1] + direct_y[3] / 2))
    _arrow(ax, (aux1[0] + aux1[2], aux1[1] + aux1[3] / 2),
           (aux2[0], aux2[1] + aux2[3] / 2))
    _arrow(ax, (aux2[0] + aux2[2], aux2[1] + aux2[3] / 2),
           (aux3[0], aux3[1] + aux3[3] / 2))
    _arrow(ax, (aux3[0] + aux3[2], aux3[1] + aux3[3] / 2),
           (aux_y[0], aux_y[1] + aux_y[3] / 2))
    ax.text(0.5, 0.055,
            ("The auxiliary stage predicts EMG burst halves; it does not learn mechanical extension/flexion.\n"
             "The final label is temporal; neither path demonstrates physiological fatigue.") if en else
            ("La etapa auxiliar aprende mitades de ráfagas EMG; no aprende extensión/flexión mecánica.\n"
             "La etiqueta final es temporal; ninguna de estas rutas demuestra fatiga fisiológica."),
            ha="center", va="center", fontsize=11, weight="bold", color=COLORS["red"])

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"executed_proxy_architecture_{lang}.svg", output_dir / f"executed_proxy_architecture_{lang}.png"]
    _save_figure(fig, outputs[0], outputs[1])
    plt.close(fig)
    return outputs


def draw_reference_architecture(output_dir: Path, lang: str = "es") -> list[Path]:
    en = lang == "en"
    title = "Target architecture: phase and measured fatigue" if en else "Arquitectura objetivo: fase auxiliar y fatiga medida"
    status = "PROPOSED · REQUIRES REFERENCES" if en else "PROPUESTA · REQUIERE REFERENCIAS"
    fig, ax = _canvas(title, status)
    ax.text(0.035, 0.865, "1 · Preprocess and window sEMG" if en else "1 · Preprocesar y extraer ventanas sEMG", fontsize=12, weight="bold", color=COLORS["blue"])

    emg = _box(ax, 0.035, 0.69, 0.145, 0.12, "sEMG\nchannel + clock" if en else "sEMG\ncanal + reloj", "signal")
    prep = _box(ax, 0.22, 0.69, 0.18, 0.12, "QC + SOS\n20–400 Hz · per segment" if en else "QC + SOS\n20–400 Hz · por tramo", "signal")
    windows = _box(ax, 0.44, 0.69, 0.18, 0.12, "Windows\n500 ms · 50% overlap" if en else "Ventanas\n500 ms · 50% solape", "signal")
    feats = _box(ax, 0.66, 0.69, 0.19, 0.12, "Features\nRMS · MDF · MNF · wavelet", "signal", fontsize=9.3)
    for left, right in zip((emg, prep, windows), (prep, windows, feats)):
        _arrow(ax, (left[0] + left[2], left[1] + left[3] / 2), (right[0], right[1] + right[3] / 2))

    ax.text(0.035, 0.585, "2 · Independent supervision" if en else "2 · Supervisión independiente", fontsize=12, weight="bold", color=COLORS["ink"])
    angle = _box(ax, 0.035, 0.42, 0.20, 0.12, "Synchronized\nangle / video" if en else "Ángulo / video\nsincronizado", "reference")
    phase_labels = _box(ax, 0.285, 0.42, 0.20, 0.12, "Phase labels\nextension / flexion" if en else "Etiquetas fase\nextensión / flexión", "reference", fontsize=9.4)
    phase_model = _box(ax, 0.535, 0.42, 0.18, 0.12, "Phase model\ntrained on X" if en else "Modelo de fase\nentrenado con X", "model", fontsize=9.3)
    phase_oof = _box(ax, 0.765, 0.42, 0.20, 0.12, "OOF phase\nprobabilities for stage 2" if en else "Probabilidades fase\nOOF para etapa 2", "neutral", fontsize=9.2)
    _arrow(ax, (angle[0] + angle[2], angle[1] + angle[3] / 2),
           (phase_labels[0], phase_labels[1] + phase_labels[3] / 2), dashed=True, color=COLORS["green"])
    _arrow(ax, (phase_labels[0] + phase_labels[2], phase_labels[1] + phase_labels[3] / 2),
           (phase_model[0], phase_model[1] + phase_model[3] / 2), dashed=True, color=COLORS["green"])
    _arrow(ax, (phase_model[0] + phase_model[2], phase_model[1] + phase_model[3] / 2),
           (phase_oof[0], phase_oof[1] + phase_oof[3] / 2))

    ax.text(0.035, 0.325, "3 · Mechanical fatigue label and comparison" if en else "3 · Etiqueta mecánica de fatiga y comparación", fontsize=12,
            weight="bold", color=COLORS["ink"])
    torque = _box(ax, 0.035, 0.15, 0.20, 0.12, "Calibrated, synchronized\nforce / torque" if en else "Fuerza / torque\ncalibrado y sincronizado", "reference", fontsize=9.3)
    fatigue_label = _box(ax, 0.285, 0.15, 0.22, 0.12, "Candidate label rule\nTref − MDC95\n3-rep median · 2 checks" if en else "Regla candidata\nTref − MDC95\nmediana 3 rep · 2 eval.", "reference", fontsize=8.8)
    direct = _box(ax, 0.56, 0.22, 0.18, 0.12, "Direct baseline\nX → p(fatigue)" if en else "Baseline directa\nX → p(fatiga)", "model", fontsize=9.4)
    experts = _box(ax, 0.78, 0.15, 0.19, 0.15, "Stage 2\nX + q_phase → experts\n→ p(fatigue)" if en else "Etapa 2\nX + qfase → expertos\n→ p(fatiga)", "model", fontsize=9.0)
    _arrow(ax, (torque[0] + torque[2], torque[1] + torque[3] / 2),
           (fatigue_label[0], fatigue_label[1] + fatigue_label[3] / 2), dashed=True, color=COLORS["green"])
    _arrow(ax, (phase_oof[0] + phase_oof[2] / 2, phase_oof[1]),
           (experts[0] + experts[2] * 0.75, experts[1] + experts[3]), color=COLORS["orange"], bend=0.12)
    ax.text(0.5, 0.055,
            ("Same X matrix for both fatigue models; Y derives from calibrated torque. Nested participant-level folds.\n"
             "Without synchronized torque/SEM and kinematics, this design is untrained and has no metrics.\n"
             "Dashed arrows: reference/supervision; solid arrows: processing/prediction.") if en else
            ("Misma matriz X para ambos modelos de fatiga; Y deriva de torque calibrado. Pliegues anidados por participante.\n"
             "Sin torque/SEM y cinemática sincronizados, esta arquitectura no está entrenada ni tiene métricas.\n"
             "Discontinuas: referencia/supervisión; continuas: procesamiento/predicción."),
            ha="center", va="center", fontsize=9.2, weight="bold", color=COLORS["red"])

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"reference_fatigue_architecture_{lang}.svg", output_dir / f"reference_fatigue_architecture_{lang}.png"]
    _save_figure(fig, outputs[0], outputs[1])
    plt.close(fig)
    return outputs


def draw_windowing_protocol(output_dir: Path, lang: str = "es") -> list[Path]:
    """Show complete sliding windows, their true lengths and their hop sizes."""
    en = lang == "en"
    title = "Windowing protocol" if en else "Protocolo de ventanas"
    status = "HISTORICAL ≠ REFERENCE STUDY" if en else "HISTÓRICO ≠ ESTUDIO CON REFERENCIAS"
    fig, ax = _canvas(title, status)
    ax.set_xlim(-20, 1020)
    ax.set_ylim(0, 1)
    ax.text(0, 0.875, "Same 1-second signal segment (schematic; not an observed label)" if en else
            "Mismo tramo de señal de 1 segundo (esquema; no es una etiqueta observada)",
            fontsize=11, weight="bold", color=COLORS["blue"])

    rows = [
        (0.70, "Executed historical proxy" if en else "Proxy histórico ejecutado",
         250, 125, COLORS["proxy"]),
        (0.49, "Proposed primary reference study" if en else "Estudio con referencias: primario propuesto",
         500, 250, COLORS["reference"]),
        (0.28, "Proposed sensitivity analysis" if en else "Análisis de sensibilidad propuesto",
         250, 125, COLORS["neutral"]),
    ]
    for y, label, length, hop, color in rows:
        starts = list(range(0, 1000 - length + 1, hop))
        summary = (
            f"W={length} ms · H={hop} ms · {len(starts)} complete windows"
            if en else
            f"W={length} ms · H={hop} ms · {len(starts)} ventanas completas"
        )
        ax.text(0, y + 0.075, label, fontsize=9.5, weight="bold", color=COLORS["ink"], va="bottom")
        ax.text(1000, y + 0.075, summary, fontsize=8.2, color=COLORS["gray"], ha="right", va="bottom")
        for index, start in enumerate(starts):
            # Alternating lanes expose each overlapping window boundary instead
            # of making the overlaps look like adjoining time bins.
            lane = (index % 2) * 0.027
            rect_y = y - 0.045 + lane
            patch = FancyBboxPatch(
                (start, rect_y), length, 0.024,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                linewidth=0.9, edgecolor=COLORS["ink"], facecolor=color, alpha=0.78,
            )
            ax.add_patch(patch)
            ax.text(start + length / 2, rect_y + 0.012, f"W{index + 1}",
                    ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
    ax.annotate("", xy=(1000, 0.205), xytext=(0, 0.205),
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": COLORS["gray"]})
    ax.text(0, 0.198, "0", ha="center", va="top", fontsize=7.8, color=COLORS["gray"])
    ax.text(1000, 0.198, "1000", ha="center", va="top", fontsize=7.8, color=COLORS["gray"])
    ax.text(500, 0.173, "Time (ms)" if en else "Tiempo (ms)", ha="center", fontsize=8.2, color=COLORS["gray"])
    ax.text(0, 0.132,
            ("N = round(T × fs); H = round(N × (1 − overlap)); Δf = fs/N. At 1 kHz: 500 ms → N=500, H=250, Δf≈2 Hz; 250 ms → N=250, H=125, Δf≈4 Hz."
             if en else
             "N = round(T × fs); H = round(N × (1 − solape)); Δf = fs/N. A 1 kHz: 500 ms → N=500, H=250, Δf≈2 Hz; 250 ms → N=250, H=125, Δf≈4 Hz."),
            ha="left", va="top", fontsize=8.1, color=COLORS["ink"])
    ax.text(0, 0.075,
            ("Window labels mark complete windows; vertical offset is only for legibility. Longer windows improve nominal frequency resolution; overlapping windows are correlated."
             if en else
             "Las marcas son ventanas completas; el desplazamiento vertical solo ayuda a leer el solape. La ventana larga mejora la resolución nominal; las ventanas solapadas son dependientes."),
            ha="left", va="top", fontsize=7.8, color=COLORS["gray"])
    ax.text(0, 0.035,
            ("Exclude acquisition gaps and reference transitions; compare models on matched support."
             if en else
             "Excluir huecos y transiciones de referencia; comparar modelos sobre el mismo soporte."),
            ha="left", va="top", fontsize=7.8, color=COLORS["gray"])
    ax.axis("off")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"windowing_protocol_{lang}.svg", output_dir / f"windowing_protocol_{lang}.png"]
    _save_figure(fig, outputs[0], outputs[1])
    plt.close(fig)
    return outputs


def draw_reference_models(output_dir: Path, lang: str = "es") -> list[Path]:
    """Compare direct, cascade and phase-expert fatigue architectures."""
    en = lang == "en"
    title = "Reference model architecture" if en else "Arquitectura de modelos objetivo"
    status = "PROPOSED · NOT TRAINED" if en else "PROPUESTA · NO ENTRENADA"
    fig, ax = _canvas(title, status)
    ax.text(0.035, 0.875, "Independent training references" if en else "Referencias independientes de entrenamiento",
            fontsize=11.5, weight="bold", color=COLORS["green"])
    yphase = _box(ax, 0.035, 0.745, 0.43, 0.105,
                  "Y_phase: extension / flexion" if en else "Y_fase: extensión / flexión",
                  "reference", fontsize=9.0, title="Synchronized angle / reviewed video" if en else "Ángulo sincronizado / video revisado")
    yfatigue = _box(ax, 0.535, 0.745, 0.43, 0.105,
                    "Y_fatigue: no fatigue / fatigue" if en else "Y_fatiga: no fatiga / fatiga",
                    "reference", fontsize=9.0, title="Calibrated force / torque" if en else "Fuerza / torque calibrados")

    x = _box(ax, 0.035, 0.515, 0.22, 0.12,
             "X: same processed window features" if en else "X: mismas features de ventana procesada",
             "signal", fontsize=8.6)
    phase = _box(ax, 0.355, 0.515, 0.27, 0.12,
                 ("X → q_ext, q_flex\nTarget: Y_phase" if en else
                  "X → q_ext, q_flex\nObjetivo: Y_fase"),
                 "model", fontsize=9.0, title="Phase classifier" if en else "Clasificador de fase")
    phase_scores = _box(ax, 0.705, 0.515, 0.26, 0.12,
                        "OOF in training; predicted in test" if en else "OOF en entrenamiento; predichas en prueba",
                        "neutral", fontsize=8.2, title="Phase probabilities" if en else "Probabilidades de fase")
    _arrow(ax, (x[0] + x[2], x[1] + x[3] / 2),
           (phase[0], phase[1] + phase[3] / 2))
    _arrow(ax, (yphase[0] + yphase[2] / 2, yphase[1]),
           (phase[0] + phase[2] / 2, phase[1] + phase[3]),
           dashed=True, color=COLORS["green"])
    _arrow(ax, (phase[0] + phase[2], phase[1] + phase[3] / 2),
           (phase_scores[0], phase_scores[1] + phase_scores[3] / 2))

    direct = _box(ax, 0.035, 0.215, 0.29, 0.17,
                  "X → p(fatigue)\nTraining target: Y_fatigue" if en else
                  "X → p(fatiga)\nObjetivo de entrenamiento: Y_fatiga",
                  "model", fontsize=8.9, title="Direct baseline" if en else "Modelo directo")
    cascade = _box(ax, 0.355, 0.215, 0.29, 0.17,
                   "X + q_phase(OOF) → p(fatigue)\nTraining target: Y_fatigue" if en else
                   "X + q_fase(OOF) → p(fatiga)\nObjetivo de entrenamiento: Y_fatiga",
                   "model", fontsize=8.4, title="Phase → fatigue cascade" if en else "Cascada fase → fatiga")
    experts = _box(ax, 0.675, 0.215, 0.29, 0.17,
                   "p = q_ext·f_ext(X) + q_flex·f_flex(X)\nTraining target: Y_fatigue" if en else
                   "p = q_ext·f_ext(X) + q_flex·f_flex(X)\nObjetivo de entrenamiento: Y_fatiga",
                   "model", fontsize=7.8, title="Phase expert heads" if en else "Cabezas expertas por fase")
    _arrow(ax, (phase_scores[0] + phase_scores[2] * 0.38, phase_scores[1]),
           (cascade[0] + cascade[2] / 2, cascade[1] + cascade[3]), color=COLORS["orange"])
    _arrow(ax, (phase_scores[0] + phase_scores[2] * 0.80, phase_scores[1]),
           (experts[0] + experts[2] / 2, experts[1] + experts[3]), color=COLORS["orange"])
    ax.text(0.5, 0.13,
            ("Fatigue models: X, Y_fatigue and held-out groups. Phase classifier target: Y_phase. Fit selection, thresholds and scaling in training folds; report accuracy, macro-F1, balanced accuracy and confusion matrices."
             if en else
             "Modelos de fatiga: X, Y_fatiga y grupos retenidos. Clasificador de fase: Y_fase. Ajuste en pliegues de entrenamiento."),
            ha="center", va="center", fontsize=7.8, color=COLORS["ink"])
    ax.text(0.5, 0.075,
            ("No calibrated force/torque, synchronized kinematics or verified participant IDs in the ZIP: the reference-backed models remain untrained."
             if en else
             "El ZIP no contiene fuerza/torque calibrados, cinemática sincronizada ni IDs verificados: los modelos con referencias siguen sin entrenar."),
            ha="center", va="center", fontsize=8.1, weight="bold", color=COLORS["red"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"reference_models_{lang}.svg", output_dir / f"reference_models_{lang}.png"]
    _save_figure(fig, outputs[0], outputs[1])
    plt.close(fig)
    return outputs


def build(output_dir: Path) -> list[Path]:
    return (draw_executed_proxy(output_dir, "es") + draw_reference_architecture(output_dir, "es") +
            draw_windowing_protocol(output_dir, "es") + draw_reference_models(output_dir, "es") +
            draw_executed_proxy(output_dir, "en") + draw_reference_architecture(output_dir, "en") +
            draw_windowing_protocol(output_dir, "en") + draw_reference_models(output_dir, "en"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    for path in build(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
