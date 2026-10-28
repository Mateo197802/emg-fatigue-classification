import io
import zipfile
from pathlib import Path
import numpy as np
import pytest


def csv_bytes(times, values):
    return ('Relative_Time_s,ADC_Value,Voltage_mV\n' + '\n'.join(f'{t},{v},0' for t,v in zip(times,values))).encode()


def test_loader_preserves_keys_and_rejects_bad_clock(tmp_path):
    from emg_fatigue.io import read_csv_bytes
    r=read_csv_bytes(csv_bytes([0,.001,.002],[1,-2,3]),'rec','P01','zip','left','R1')
    assert r.subject_id=='P01' and r.x.tolist()==[1,-2,3]
    with pytest.raises(ValueError,match='strictly increasing'):
        read_csv_bytes(csv_bytes([0,.001,.001],[1,2,3]),'r','P','zip','left','R1')


def test_archive_rejects_traversal_and_case_duplicate(tmp_path):
    from emg_fatigue.io import inspect_zip
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('../x.csv','bad')
    with pytest.raises(ValueError,match='unsafe'):inspect_zip(p)
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('p/R1_dr.csv','x');z.writestr('P/r1_DR.CSV','y')
    with pytest.raises(ValueError,match='duplicate'):inspect_zip(p)


def test_gap_and_jitter_are_explicit_and_never_bridged():
    from emg_fatigue.qc import contiguous_runs
    t=np.r_[np.arange(1000)/1000,2+np.arange(1000)/1000]
    runs=contiguous_runs(t)
    assert len(runs)==2 and runs[0]['end']==1000 and runs[1]['start']==1000
    assert all(r['status']=='valid' for r in runs)
    t=np.cumsum(np.tile([.0005,.0015],1000))
    assert contiguous_runs(t)[0]['status']=='jitter_exceeds_limit'


def test_filter_removes_dc_and_low_frequency_without_crossing_gaps():
    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.preprocessing import filter_run
    t=np.arange(4000)/1000;x=100+np.sin(2*np.pi*100*t)+3*np.sin(2*np.pi*2*t)
    y,mask=filter_run(x,1000,StudyConfig())
    assert mask[:100].sum()==0 and mask[-100:].sum()==0
    assert abs(y[mask].mean())<.01
    assert np.sqrt(np.mean(y[mask]**2))==pytest.approx(1/np.sqrt(2),rel=.04)


def test_binary_boundaries_and_quality_are_not_negatives():
    from emg_fatigue.labels import label_interval
    assert label_interval(0,2.9,0,9)=='early_proxy_nonfatigue'
    assert label_interval(3,4,0,9) is None
    assert label_interval(2.9,3.1,0,9) is None
    assert label_interval(6,9,0,9)=='late_proxy_fatigue'
    assert label_interval(0,1,0,9,eligible=False) is None


def test_spectral_features_and_wavelet_energy_are_consistent():
    from emg_fatigue.features import window_features, FEATURE_COLUMNS
    t=np.arange(1000)/1000;x=20+2*np.sin(2*np.pi*100*t)
    f=window_features(x,1000)
    assert f['rms']==pytest.approx(np.sqrt(2),rel=.01)
    assert f['mdf_hz']==pytest.approx(100,abs=1)
    assert f['mnf_hz']==pytest.approx(100,abs=1)
    assert sum(f[k] for k in f if k.startswith('wavelet_'))==pytest.approx(1)
    assert set(f)==set(FEATURE_COLUMNS)
    with pytest.raises(ValueError,match='energy'):window_features(np.zeros(250),1000)
