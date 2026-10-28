import io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def test_identical_signal_in_distinct_folders_is_rejected(tmp_path):
    from emg_fatigue.io import load_recordings
    data='Relative_Time_s,ADC_Value\n0,1\n0.001,2\n0.002,3\n'
    p=tmp_path/'source.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('A/R1_dr.csv',data);z.writestr('B/R1_dr.csv',data)
    with pytest.raises(ValueError,match='duplicate recording'):load_recordings(p)


@pytest.mark.parametrize('case',['constant','low_rate','short','mixed'])
def test_rejected_recording_intervals_have_stable_schema(case):
    from emg_fatigue.contracts import Recording,StudyConfig
    from emg_fatigue.inference import predict_recording,PREDICTION_COLUMNS
    t=np.arange(2000)/1000;x=np.ones(len(t))
    if case=='low_rate':t=np.arange(1000)/500;x=np.sin(2*np.pi*80*t)
    if case=='short':t=np.arange(100)/1000;x=np.sin(2*np.pi*80*t)
    if case=='mixed':
        t=np.r_[np.arange(1000)/1000,2+np.arange(1000)/1000]
        x=np.r_[np.sin(2*np.pi*80*t[:1000]),np.ones(1000)]
    r=Recording('R','G','test','left','R1',t,x,'hash')
    # Never reaches model prediction: mixed first run is discarded by test mask.
    from emg_fatigue.validation import build_pipeline
    from emg_fatigue.features import FEATURE_COLUMNS
    fit=pd.DataFrame(np.ones((4,len(FEATURE_COLUMNS))),columns=FEATURE_COLUMNS)
    pipe=build_pipeline({'family':'majority'},1);pipe.fit(fit,[0,1,0,1])
    from dataclasses import asdict
    df=predict_recording(r,{'config':asdict(StudyConfig()),'pipeline':pipe,'label_vocabulary':{0:'early_proxy_nonfatigue',1:'late_proxy_fatigue'}})
    assert list(df.columns)==PREDICTION_COLUMNS
    rejected=df[df.row_type=='rejected_run']
    assert len(rejected)>=1
    assert rejected.prediction.isna().all() and rejected.score.isna().all()
    assert rejected.reason.str.len().min()>0


def test_failed_inner_search_persists_audit_and_no_completion(tmp_path,monkeypatch):
    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.workflow import synthetic_recordings,process_dataset,eligible_features
    from emg_fatigue import validation
    from sklearn.exceptions import ConvergenceWarning
    c=StudyConfig(inner_folds=2,bootstrap_replicates=10)
    f=eligible_features(process_dataset(synthetic_recordings(),c,tmp_path)['features'])
    def fail(*args,**kwargs):raise ConvergenceWarning('forced convergence failure')
    monkeypatch.setattr(validation,'fit_pipeline',fail)
    with pytest.raises(RuntimeError):validation.evaluate_nested(f,c,tmp_path/'evaluation',candidates=[{'family':'logistic_rms_mdf','C':1.}])
    assert (tmp_path/'evaluation/failure_receipt.json').exists()
    assert not (tmp_path/'evaluation/evaluation_receipt.json').exists()
    audit=pd.read_csv(tmp_path/'evaluation/search_audit/F01/search_results.csv')
    assert (audit.status=='failed').all() and audit.failure.str.contains('forced convergence').all()


def test_completed_evaluation_rejects_new_features_edited_results_and_failed_rerun(tmp_path,monkeypatch):
    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.workflow import synthetic_recordings,process_dataset,eligible_features
    from emg_fatigue import validation
    from emg_fatigue.reporting import verify_evaluation
    c=StudyConfig(inner_folds=2,bootstrap_replicates=10);candidates=[{'family':'logistic_rms_mdf','C':1.}]
    data=process_dataset(synthetic_recordings(seed=1),c,tmp_path)
    validation.evaluate_nested(eligible_features(data['features']),c,tmp_path/'evaluation',candidates=candidates)
    verify_evaluation(tmp_path,c)
    original=(tmp_path/'evaluation/results.csv').read_bytes()
    (tmp_path/'evaluation/results.csv').write_bytes(original+b'\n')
    with pytest.raises(ValueError,match='checksum'):verify_evaluation(tmp_path,c)
    (tmp_path/'evaluation/results.csv').write_bytes(original)
    process_dataset(synthetic_recordings(seed=2),c,tmp_path)
    with pytest.raises(ValueError,match='feature'):verify_evaluation(tmp_path,c)
    monkeypatch.setattr(validation,'fit_pipeline',lambda *a,**k:(_ for _ in ()).throw(ValueError('forced failure')))
    with pytest.raises(RuntimeError):validation.evaluate_nested(eligible_features(data['features']),c,tmp_path/'evaluation',candidates=candidates)
    assert not (tmp_path/'evaluation/evaluation_receipt.json').exists()


def test_bounded_archive_producer_stops_before_allocating_member(monkeypatch):
    from emg_fatigue.io import run_bounded
    class Producer:
        def __init__(self):self.stdout=io.BytesIO(b'x'*1000);self.stderr=io.BytesIO();self.killed=False
        def kill(self):self.killed=True
        def wait(self,timeout=None):return 0
    producer=Producer()
    import subprocess
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:producer)
    with pytest.raises(ValueError,match='byte limit'):run_bounded(['producer'],16)
    assert producer.killed
    # The stream is deliberately closed by the implementation; no complete read occurs.
