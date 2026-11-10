import pandas as pd


def test_public_results_exclude_identity_paths_and_parameters(tmp_path):
    from emg_fatigue.reporting import export_public_results
    frame=pd.DataFrame([{'model':'majority','fold_id':'F01','subject_id':'private-name','source_member':'person/secret.csv','parameters':'private','accuracy':.5,'f1_macro':.4,'n_test':10}])
    p=export_public_results(frame,tmp_path/'public',tmp_path/'private')
    text=p.read_text()
    assert 'private-name' not in text and 'secret.csv' not in text and 'parameters' not in text
    assert 'F01' not in text and 'accuracy' in text and 'f1_macro' in text
    assert p.read_bytes()==export_public_results(frame,tmp_path/'public',tmp_path/'private').read_bytes()


def test_report_equations_cover_the_executed_filter_window_and_features():
    from emg_fatigue.reporting import report_equation_rows

    equations = {name: (formula, implementation) for name, formula, implementation in report_equation_rows()}

    assert {"Butterworth SOS", "Notch sensitivity", "Otsu activity", "Window indices", "Hann PSD",
            "RMS", "MAV", "Waveform length per second", "MDF", "MNF", "Power ratio", "Wavelet energies"} <= set(equations)
    assert "20–400 Hz" in equations["Butterworth SOS"][1]
    assert "sosfiltfilt" in equations["Butterworth SOS"][1]
    assert "250 ms" in equations["Window indices"][1]
    assert "50%" in equations["Window indices"][1]
    assert "√" in equations["MDF"][0] or "min" in equations["MDF"][0]
    assert "sum[n] w[n]^2" in equations["Hann PSD"][0]
    assert "one-sided density periodogram" in equations["Hann PSD"][1]
    assert "db4" in equations["Wavelet energies"][1]


def test_report_includes_reference_window_and_corrected_feature_formulas():
    from emg_fatigue.reporting import report_equation_rows

    equations = {name: (formula, implementation) for name, formula, implementation in report_equation_rows()}
    assert {
        "Butterworth analog prototype", "Reference window indices", "Corrected waveform length",
        "Candidate torque label", "Kinematic phase label", "Direct fatigue model",
        "Phase cascade", "Soft-gated phase experts", "Executed burst-half cascade",
    } <= set(equations)
    assert "|H(jω)|²" in equations["Butterworth analog prototype"][0]
    assert "500 ms" in equations["Reference window indices"][1]
    assert "250 ms" in equations["Reference window indices"][1]
    assert "(N−1)/fs" in equations["Corrected waveform length"][0]
    assert "not used" in equations["Corrected waveform length"][1].lower()
    assert "OOF" in equations["Phase cascade"][1]
    assert "not extension/flexion" in equations["Executed burst-half cascade"][1]


def test_protocol_equation_catalog_imports_without_plotting_or_dataframe_stack():
    import os
    import subprocess
    import sys

    code = r"""
import builtins, sys
sys.path.insert(0, 'src')
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'matplotlib', 'numpy', 'pandas'}:
        raise ModuleNotFoundError('blocked optional analysis dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from emg_fatigue.protocol_equations import report_equation_rows
assert len(report_equation_rows()) >= 20
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
