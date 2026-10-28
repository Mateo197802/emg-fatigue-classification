from dataclasses import replace
import numpy as np
import pytest


def test_synthetic_workflow_has_real_features_and_private_receipt(tmp_path):
    from emg_fatigue.workflow import synthetic_recordings, process_dataset
    from emg_fatigue.contracts import StudyConfig
    c=StudyConfig(bootstrap_replicates=20)
    r=synthetic_recordings(groups=4,seconds=12)
    bundle=process_dataset(r,c,tmp_path)
    df=bundle['features'];labeled=df[df.label.notna()]
    assert len(labeled)>0 and set(labeled.label)=={0,1}
    assert labeled.subject_id.nunique()==4
    assert (tmp_path/'processing_receipt.json').exists()
    assert set(df.status).issuperset({'labeled','activity_not_detected'})


def test_model_roundtrip_and_reassembled_fragments(tmp_path):
    from emg_fatigue.workflow import synthetic_recordings, process_dataset
    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.inference import fit_final, load_bundle, predict_recording
    from emg_fatigue.io import reassemble_fragments
    import pandas as pd
    c=StudyConfig(inner_folds=2,bootstrap_replicates=20)
    recordings=synthetic_recordings(groups=4,seconds=12)
    data=process_dataset(recordings,c,tmp_path/'dataset')['features'];train=data[data.label.notna()].copy()
    path=tmp_path/'model.joblib'
    fit_final(train,c,path,candidates=[{'family':'logistic_rms_mdf','C':1.}])
    model=load_bundle(path)
    rec=recordings[0];a=predict_recording(rec,model)
    paths=[]
    for i,indices in enumerate(np.array_split(np.arange(len(rec.t)),3)):
        p=tmp_path/f'part{i}.csv';pd.DataFrame({'Relative_Time_s':rec.t[indices],'ADC_Value':rec.x[indices]}).to_csv(p,index=False,float_format='%.17g');paths.append(p)
    merged=reassemble_fragments(paths[::-1]);b=predict_recording(merged,model)
    assert len(a)==len(b)
    np.testing.assert_allclose(a.score.to_numpy(dtype=float),b.score.to_numpy(dtype=float),rtol=1e-8,atol=1e-8,equal_nan=True)
    assert set(a.status).issuperset({'predicted','abstain_activity'})
    p=tmp_path/'model.joblib';p.write_bytes(p.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='checksum'):load_bundle(p)
