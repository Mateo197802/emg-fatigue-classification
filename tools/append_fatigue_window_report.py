"""Append the executed fatigue-window evidence to a private report copy.

The original v6 report is never changed. All tables and figures are generated
from public aggregate CSVs and figures in this repository; person-level rows
and signals are not copied into the revised manuscript.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reference_fatigue"
FIGURES = ROOT / "docs" / "figures"
AUTHORS = "Ethan Guevara; Mateo Gavilanes; Kennedy Rodriguez"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def find(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(f"No row matching {criteria}")


def n(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def ci(row: dict[str, str], metric: str) -> str:
    return f"{n(row[metric + '_ci95_low'])} to {n(row[metric + '_ci95_high'])}"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), fill)
    tc_pr.append(shade)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "5")
        node.set(qn("w:color"), "D9E1E5")


def add_table(doc, headers: list[str], rows: list[list[str]], widths: list[float], font_size=8.2):
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    table.style = "Table Grid"
    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)
    all_rows = [headers, *rows]
    for row_index, values in enumerate(all_rows):
        row = table.rows[0] if row_index == 0 else table.add_row()
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        for i, value in enumerate(values):
            cell = row.cells[i]
            cell.width = Inches(widths[i])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            set_cell_shading(cell, "234F6B" if row_index == 0 else ("F1F5F7" if row_index % 2 == 0 else "FFFFFF"))
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if row_index == 0 or i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(str(value))
            run.font.name = "Arial"
            run.font.size = Pt(font_size)
            run.font.bold = row_index == 0
            run.font.color.rgb = RGBColor(255, 255, 255) if row_index == 0 else RGBColor(24, 42, 54)
    set_borders(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_paragraph(doc, text: str, *, bold_prefix: str | None = None):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.12
    if bold_prefix and text.startswith(bold_prefix):
        first = paragraph.add_run(bold_prefix)
        first.bold = True
        paragraph.add_run(text[len(bold_prefix):])
    else:
        paragraph.add_run(text)
    return paragraph


def add_formula(doc, text: str):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.22)
    paragraph.paragraph_format.right_indent = Inches(0.12)
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.2)
    run.font.color.rgb = RGBColor(35, 79, 107)
    return paragraph


def add_figure(doc, image_path: Path, caption: str, width=6.15):
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(image_path), width=Inches(width))
    cap = doc.add_paragraph(caption, style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)


def configure_styles(doc) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(24, 42, 54)
    for name, size in (("Heading 1", 15), ("Heading 2", 12), ("Heading 3", 10.5)):
        style = doc.styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(35, 79, 107)
        style.paragraph_format.keep_with_next = True
    doc.styles["Caption"].font.name = "Arial"
    doc.styles["Caption"].font.size = Pt(8)
    doc.styles["Caption"].font.italic = True


def append_report(input_docx: Path, output_docx: Path, assets_dir: Path) -> dict:
    metrics = read_csv(RESULTS / "benchmark_metrics.csv")
    specialist = read_csv(RESULTS / "prime_mover_specialist_metrics.csv")
    nested = read_csv(RESULTS / "nested_hgb_metrics.csv")
    confusions = read_csv(RESULTS / "confusion_matrices.csv")
    specialist_conf = read_csv(RESULTS / "prime_mover_specialist_confusions.csv")
    nested_conf = read_csv(RESULTS / "nested_hgb_confusion.csv")
    nested_pairs = read_csv(RESULTS / "nested_hgb_paired_differences.csv")
    specialist_pairs = read_csv(RESULTS / "prime_mover_specialist_paired_differences.csv")
    time_pairs = read_csv(RESULTS / "paired_delta_vs_time_only.csv")
    trial_metrics = read_csv(RESULTS / "figshare_task_level" / "benchmark_results.csv")
    trial_conf = read_csv(RESULTS / "figshare_task_level" / "confusion_matrix.csv")

    doc = Document(input_docx)
    configure_styles(doc)
    doc.add_page_break()
    doc.add_heading("Supplementary Evidence for Fatigue Classification", level=1)
    add_paragraph(doc, "This supplement corrects the target interpretation and records a separate model trained against synchronized, stable self-reports. It distinguishes the original course archive from two external benchmarks. The external window model predicts perceived fatigue, not force loss or a clinical state. The course-archive classifier remains an early-versus-late recording-position model.")
    add_paragraph(doc, "Authors: " + AUTHORS)

    doc.add_heading("H.1  Window target and corpus", level=2)
    add_paragraph(doc, "The external Zenodo v2 dataset contains 13 participants, 156 upper-limb trials, four sEMG channels and self-reported fatigue ratings synchronized to the signal. The source uses three rating states: stable source state 0 is mapped to binary model target y=0 (no perceived fatigue), stable source state 2 is mapped to y=1 (perceived fatigue), and transition state 1 is excluded. A 500 ms EMG window receives a label only when its full duration lies inside one stable reference state. Windows crossing a state change or a reference gap are discarded. The retained support is 125,165 windows: 64,283 with y=0 and 60,882 with y=1.")
    add_paragraph(doc, "This is the experiment in which a model is actually trained to predict fatigue/no-fatigue per window. Its labels are perceived-fatigue self-reports. No synchronized calibrated torque or force-loss measurement is available for the course cohort, so the result must not be called objective mechanical fatigue.")
    add_figure(doc, assets_dir / "perceived_fatigue_window_architecture.png", "Figure H1. Aligned self-report, signal processing, window features, model branches and participant-held-out evaluation.")

    doc.add_heading("H.2  Signal audit and preprocessing", level=2)
    add_paragraph(doc, "The audit covered 624 channel-trial records. It removed 54 terminal rows with a backward clock jump from 42 trials. No remaining gaps or non-monotonic timestamps were found in the processed records. Sensor rail limits were unavailable, so clipping could not be assessed.")
    add_formula(doc, "Δt_med = median(t[n+1] - t[n]);  fs = 1 / Δt_med")
    add_formula(doc, "Butterworth prototype: |Ha(jΩ)|² = 1 / (1 + (Ω/Ωc)^(2n)),  n = 4")
    add_formula(doc, "Digital band-pass: H(z) = product_r [(b0,r + b1,r z^-1 + b2,r z^-2) / (a0,r + a1,r z^-1 + a2,r z^-2)]")
    add_formula(doc, "SOS recurrence: y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) / a0")
    add_paragraph(doc, "The filter uses 20-450 Hz cutoffs, a fourth-order analog prototype, digital order eight and second-order sections. Coefficients are computed from each measured sampling rate. Forward-backward sosfiltfilt gives zero-phase output for offline analysis; segment edges are masked and gaps are not crossed.")
    add_formula(doc, "Line prominence Lf = 10*log10(PSD(f) / median(PSD neighboring bins))")
    add_formula(doc, "Conditional notch: iirnotch(f0, Q=30, fs); apply only when Lf > 10 dB. No 50/60 Hz notch was triggered.")
    add_paragraph(doc, "A 50/60 Hz notch is conditional on the measured line prominence and is not part of the default filter when the threshold is not met. The observed sampling rate was approximately 1,259.26 Hz.")

    doc.add_heading("H.3  Windowing and spectral estimate", level=2)
    add_formula(doc, "T = 0.5 s;  ρ = 0.5;  N = round(T*fs) = 630;  H = round(N*(1-ρ)) = 315")
    add_formula(doc, "Δf = fs/N ≈ 2 Hz;  x_i[n] = x[i*H+n],  n=0,...,N-1")
    add_formula(doc, "Hann: w[n] = 0.5*(1-cos(2πn/(N-1)))")
    add_formula(doc, "Xw[k] = Σ_n x[n]w[n]exp(-j2πkn/N);  Pxx[k] = ck*|Xw[k]|²/(fs*Σ_n w[n]²)")
    add_paragraph(doc, "The one-sided periodogram uses ck=2 for interior positive-frequency bins and ck=1 at DC and Nyquist. The implementation is scipy.signal.periodogram with a Hann window, detrend=False and density scaling; it is not Welch averaging. Overlapping windows are dependent, so uncertainty is bootstrapped across participants rather than across individual windows.")

    doc.add_heading("H.4  Features and formulas", level=2)
    feature_rows = [
        ["RMS", "sqrt(Σx²/N)", "V; effective amplitude"],
        ["MAV", "Σ|x|/N", "V; mean absolute amplitude"],
        ["Waveform length", "Σ|x[n]-x[n-1]|", "V per window"],
        ["Waveform length per second", "Σ|Δx| / ((N-1)/fs)", "V/s"],
        ["Integrated EMG", "Σ|x|/fs", "V·s"],
        ["MDF", "first fm where cumulative Pxx reaches 0.5*ΣPxx", "Hz"],
        ["MNF", "Σ(f*Pxx)/ΣPxx", "Hz"],
        ["Relative band power", "Σ(B Pxx*Δf) / Σ(20-450 Hz Pxx*Δf)", "Bands: 20-60, 60-150, 150-450 Hz"],
        ["Spectral entropy", "-Σ p[k]ln(p[k])/ln(K), p=Pxx/ΣPxx", "0 to 1"],
        ["db4 relative energy", "Ej=Σc_j² / ΣqΣc_q², q in {A3,D3,D2,D1}", "Wavelet level 3; four values"],
    ]
    add_table(doc, ["Feature family", "Definition per channel", "Units / settings"], feature_rows, [1.55, 3.15, 1.55], font_size=7.7)
    add_paragraph(doc, "Fifteen features are computed per channel, producing 60 inputs across four channels. The feature groups capture amplitude, temporal change, spectral distribution, frequency-band power and multiscale wavelet energy. They do not uniquely identify fatigue; task, effort and sensor conditions also affect them.")

    doc.add_heading("H.5  Models, architecture and evaluation", level=2)
    add_paragraph(doc, "The window experiment compares logistic regression, RBF SVM, random forest, HistGradientBoosting, a compact MLP, a majority baseline and a separate time-only diagnostic. The time-only model is not an EMG detector; it quantifies how much the label can be inferred from temporal position in the trial.")
    add_paragraph(doc, "Outer validation is leave-one-participant-out across 13 folds. All trials and windows from a participant stay together. In each training fold, no more than 50 evenly spaced windows per participant × trial × class are sampled; held-out windows remain complete. Time, subject identity and trial identity are not input features. The nested HGB searches 16 fixed settings inside three grouped training-only folds, selecting mean participant macro-F1. Prime-mover experts use the dataset's task metadata for routing and are not extension/flexion experts.")
    add_figure(doc, assets_dir / "perceived_fatigue_window_metrics.png", "Figure H2. Participant-macro Accuracy and macro-F1 with 95% participant-bootstrap intervals.", width=6.1)
    add_figure(doc, assets_dir / "perceived_fatigue_window_confusions.png", "Figure H3. OOF confusion counts and within-reference-class rates.", width=6.1)

    doc.add_heading("H.6  Window-level results", level=2)
    model_rows = []
    names = [
        ("time_only_control", "Time-only diagnostic"),
        ("hist_gradient_boosting", "Global HGB"),
        ("nested_hist_gradient_boosting", "Nested-tuned HGB"),
        ("specialist_by_prime_mover", "HGB experts by prime mover"),
        ("random_forest", "Random forest"),
        ("logistic_regression", "Logistic regression"),
        ("mlp_compact", "Compact MLP"),
        ("svm_rbf", "RBF SVM"),
        ("majority", "Majority baseline"),
    ]
    for model_id, label in names:
        if model_id == "nested_hist_gradient_boosting":
            row = nested[0]
            pooled_acc, pooled_f1 = n(row["accuracy"]), n(row["f1_macro"])
        elif model_id == "specialist_by_prime_mover":
            row = find(specialist, model="specialist_by_prime_mover")
            pooled_acc, pooled_f1 = n(row["accuracy"]), n(row["f1_macro"])
        else:
            row = find(metrics, architecture=model_id)
            pooled_acc, pooled_f1 = n(row["accuracy"]), n(row["f1_macro"])
        model_rows.append([
            label,
            f"{pooled_acc} / {pooled_f1}",
            f"{n(row['accuracy_participant_macro'])} ({ci(row, 'accuracy')})",
            f"{n(row['f1_macro_participant_macro'])} ({n(row['f1_macro_ci95_low'])} to {n(row['f1_macro_ci95_high'])})",
        ])
    add_table(doc, ["Model", "Pooled Accuracy / F1", "Participant Accuracy (95% CI)", "Participant macro-F1 (95% CI)"], model_rows, [1.45, 1.28, 1.92, 1.7], font_size=7.3)

    def matrix_rows(model_id, rows, model_col):
        chosen = [r for r in rows if r[model_col] == model_id]
        cells = {(int(r["actual_label"]), int(r["predicted_label"])): str(int(r["count"])) for r in chosen}
        return [cells[(0, 0)], cells[(0, 1)], cells[(1, 0)], cells[(1, 1)]]

    cm_rows = [
        ["Global HGB", *matrix_rows("hist_gradient_boosting", confusions, "architecture")],
        ["Nested HGB", *matrix_rows("nested_hist_gradient_boosting", nested_conf, "model")],
        ["Prime-mover experts", *matrix_rows("specialist_by_prime_mover", specialist_conf, "model")],
        ["Time-only diagnostic", *matrix_rows("time_only_control", confusions, "architecture")],
    ]
    add_table(doc, ["Model", "TN", "FP", "FN", "TP"], cm_rows, [2.25, 1.03, 1.03, 1.03, 1.01], font_size=8)

    nested_acc = find(nested_pairs, metric="accuracy")
    nested_f1 = find(nested_pairs, metric="f1_macro")
    specialist_acc = find(specialist_pairs, metric="accuracy")
    specialist_f1 = find(specialist_pairs, metric="f1_macro")
    hgb_time_acc = find(time_pairs, emg_model="hist_gradient_boosting", metric="accuracy")
    hgb_time_f1 = find(time_pairs, emg_model="hist_gradient_boosting", metric="f1_macro")
    pair_rows = [
        ["Nested HGB - fixed HGB", f"{n(nested_acc['mean_paired_difference'])} ({n(nested_acc['ci95_low'])} to {n(nested_acc['ci95_high'])})", f"{n(nested_f1['mean_paired_difference'])} ({n(nested_f1['ci95_low'])} to {n(nested_f1['ci95_high'])})"],
        ["Prime-mover experts - global HGB", f"{n(specialist_acc['mean_paired_difference'])} ({n(specialist_acc['ci95_low'])} to {n(specialist_acc['ci95_high'])})", f"{n(specialist_f1['mean_paired_difference'])} ({n(specialist_f1['ci95_low'])} to {n(specialist_f1['ci95_high'])})"],
        ["Global HGB - time-only diagnostic", f"{n(hgb_time_acc['mean_paired_difference'])} ({n(hgb_time_acc['ci95_low'])} to {n(hgb_time_acc['ci95_high'])})", f"{n(hgb_time_f1['mean_paired_difference'])} ({n(hgb_time_f1['ci95_low'])} to {n(hgb_time_f1['ci95_high'])})"],
    ]
    add_table(doc, ["Paired participant comparison", "Δ Accuracy (95% CI)", "Δ macro-F1 (95% CI)"], pair_rows, [3.0, 1.65, 1.7], font_size=7.6)
    add_paragraph(doc, "Nested tuning did not improve on the fixed HGB: paired differences were -0.0019 Accuracy (95% CI -0.0076 to 0.0033) and -0.0029 macro-F1 (-0.0099 to 0.0032). Prime-mover experts also performed worse than global HGB. The time-only model substantially outperformed every EMG model on all 13 participants. This is evidence of temporal confounding, not evidence of a useful deployed fatigue detector.")

    doc.add_heading("H.7  Separate trial-level benchmark", level=2)
    add_paragraph(doc, "The Figshare dataset is a distinct task: 14 participants contribute 14 paired reference-contraction and TLim trials (28 trials total) at 2,048 Hz. It compares a reference contraction at 30% torque with sustained TLim knee extension to task failure. Each complete trial has one label; window labels are not inferred. EMG models use window summaries or multiple-instance pooling to produce one trial prediction.")
    trial_rows = []
    for model_id, label in (("lr_all_features", "Logistic regression, eight EMG features"), ("cnn1d_trial_mil", "CNN-1D with trial-level pooling"), ("time_only_control", "Time and bout-structure diagnostic")):
        row = find(trial_metrics, model=model_id)
        trial_rows.append([label, n(row["accuracy"]), n(row["f1_macro"]), f"{n(row['accuracy_ci95_low'])} to {n(row['accuracy_ci95_high'])}"])
    add_table(doc, ["Trial-level procedure", "Accuracy", "Macro-F1", "Accuracy 95% CI"], trial_rows, [2.8, 1.0, 1.0, 1.55], font_size=7.8)
    trial_cm_rows = []
    for model_id, label in (("lr_all_features", "EMG logistic regression"), ("cnn1d_trial_mil", "CNN-1D MIL"), ("time_only_control", "Time/bout control")):
        chosen = [r for r in trial_conf if r["model"] == model_id]
        cells = {(int(r["true_label"]), int(r["predicted_label"])): str(int(r["count"])) for r in chosen}
        trial_cm_rows.append([label, cells[(0, 0)], cells[(0, 1)], cells[(1, 0)], cells[(1, 1)]])
    add_table(doc, ["Trial model", "Ref→Ref", "Ref→TLim", "TLim→Ref", "TLim→TLim"], trial_cm_rows, [2.1, 1.1, 1.1, 1.1, 0.95], font_size=7.8)
    add_figure(doc, assets_dir / "figshare_trial_outcome_confusion.png", "Figure H4. Trial-level confusion matrices; this target is not instantaneous fatigue.", width=6.1)
    add_figure(doc, assets_dir / "figshare_trial_outcome_structure.png", "Figure H5. The reference and TLim conditions differ in duration and active-bout structure.", width=6.1)
    add_paragraph(doc, "The time/bout control reaches perfect Accuracy and macro-F1 because reference trials contain four active bouts while TLim trials contain one. This exposes a protocol shortcut; the score cannot be interpreted as a physiological detector. This experiment is not merged with the 500 ms perceived-fatigue-window cohort.")

    doc.add_heading("H.8  Interpretation and literature", level=2)
    add_paragraph(doc, "Three targets are kept separate. The original course archive predicts early versus late recording position. The Zenodo experiment predicts self-reported perceived fatigue for reference-aligned windows, but EMG performance is weak against a time-only control. The Figshare benchmark predicts a whole-trial protocol outcome. No dataset in the course ZIP supplies synchronized extension/flexion labels or calibrated force-loss labels. Therefore the proposed extension/flexion-to-fatigue cascade cannot be evaluated from these recordings. Fine-tuning is not transferred across these incompatible channel and label contracts, and reinforcement learning is not suited to a static binary window target without actions or sequential rewards.")
    add_paragraph(doc, "References: Cerqueira S et al. A Dataset of sEMG and Self-Perceived Fatigue Levels for Muscle Fatigue Analysis. Zenodo v2, 10.5281/zenodo.14182446. Rossato J et al. Journal of Neurophysiology 2022;127(2):421-433, 10.1152/jn.00453.2021. Zhu J et al. Applied Sciences 2021;11(9):3821, 10.3390/app11093821.")

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_docx)
    return {"input": str(input_docx), "output": str(output_docx), "tables_added": 5, "figures_added": 5, "supplement_heading": "Supplementary Evidence for Fatigue Classification"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-docx", required=True, type=Path)
    parser.add_argument("--output-docx", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    args = parser.parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "perceived_fatigue_window_architecture.png",
        "perceived_fatigue_window_metrics.png",
        "perceived_fatigue_window_confusions.png",
        "figshare_trial_outcome_confusion.png",
        "figshare_trial_outcome_structure.png",
    ):
        shutil.copy2(FIGURES / name, args.assets_dir / name)
    print(append_report(args.input_docx, args.output_docx, args.assets_dir))


if __name__ == "__main__":
    main()
