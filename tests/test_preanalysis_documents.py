from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_report_and_defense_sources_do_not_present_elapsed_time_feature_trajectories():
    paths = [
        ROOT / "tools/build_defense_inputs.py",
        ROOT / "tools/build_defense_v2.mjs",
        ROOT / "tools/build_manuscript.py",
        ROOT / "src/emg_fatigue/reporting.py",
        ROOT / "docs/INFORME_EVIDENCIAS.md",
        ROOT / "docs/PROFESSOR_FEEDBACK_RESPONSE_2026-11-09.md",
        ROOT / "docs/defense-scripts/Guion_defensa_ES.md",
        ROOT / "docs/defense-scripts/Defense_Script_EN.md",
    ]
    banned_claims = (
        "feature_trajectories_by_group.csv",
        "trajectories.png",
        "decil temporal",
        "by recording position and then by group",
        "curvas descriptivas de RMS/MDF/MNF",
    )

    for path in paths:
        source = path.read_text(encoding="utf-8").lower()
        for claim in banned_claims:
            assert claim.lower() not in source, f"{path.relative_to(ROOT)} retains {claim!r}"


def test_preanalysis_narrative_says_verified_repetition_phase_trajectories_are_unavailable():
    report = (ROOT / "docs/INFORME_EVIDENCIAS.md").read_text(encoding="utf-8").lower()
    spanish_script = (ROOT / "docs/defense-scripts/Guion_defensa_ES.md").read_text(encoding="utf-8").lower()
    english_script = (ROOT / "docs/defense-scripts/Defense_Script_EN.md").read_text(encoding="utf-8").lower()

    assert "trayectorias" in report and "no están disponibles" in report
    assert "la ventana es la observación" in report
    assert "no fue entrenado para distinguir fatiga de no fatiga" in report
    assert "repetición" in spanish_script and "no están disponibles" in spanish_script
    assert "trajectories" in english_script and "unavailable" in english_script


def test_manuscript_names_observation_and_class_as_separate_things():
    manuscript = (ROOT / "tools/build_manuscript.py").read_text(encoding="utf-8").lower()

    assert "each feature window is an observation" in manuscript
    assert "the model was not trained to distinguish fatigue from no fatigue" in manuscript
    assert "recording-position proxy" in manuscript
