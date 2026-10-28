def test_variations_are_prespecified_and_leave_primary_unchanged():
    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.sensitivities import sensitivity_variants, ablation_columns
    from emg_fatigue.features import FEATURE_COLUMNS
    c=StudyConfig();v=sensitivity_variants(c)
    assert len(v)==9
    assert c.window_s==.25 and c.boundary_shift==0
    assert v['window_1000ms'].window_s==1
    assert v['no_overlap'].overlap==0
    assert v['boundary_plus_10pct'].boundary_shift==.1
    assert set(ablation_columns()['without_spectral']).isdisjoint({'mdf_hz','mnf_hz','power_ratio_low_high'})
    assert set(ablation_columns()['without_spectral'])<set(FEATURE_COLUMNS)
