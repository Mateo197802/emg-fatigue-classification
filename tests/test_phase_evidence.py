from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest


def module(name):
    path=Path(__file__).resolve().parents[1]/'experiments'/'phase_proxy'/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def fixture():
    return pd.DataFrame({'model':['a']*4+['b']*4,'window_id':['w1','w2','w3','w4']*2,
      'label':[0,0,1,1]*2,'prediction':[0,1,0,1,0,0,1,1],'score':[.1,.6,.4,.9,.1,.2,.8,.9]})


def test_confusions_retain_exact_counts_and_target():
    m=module('evidence');r=m.confusion_rows(fixture(),'early_late_proxy')
    a=r[r.model=='a'].iloc[0];b=r[r.model=='b'].iloc[0]
    assert [a.cm_0_0,a.cm_0_1,a.cm_1_0,a.cm_1_1]==[1,1,1,1]
    assert a.norm_0_0==.5 and a.norm_1_1==.5 and a.n==4
    assert b.norm_0_0==1 and b.norm_1_1==1 and b.target=='early_late_proxy'


@pytest.mark.parametrize('failure',['duplicate','missing','label_mismatch','nonbinary'])
def test_oof_validation_rejects_corrupt_support(failure):
    m=module('evidence');f=fixture()
    if failure=='duplicate':f=pd.concat([f,f.iloc[[0]]])
    elif failure=='missing':f=f.drop(f.index[-1])
    elif failure=='label_mismatch':f.loc[4,'label']=1
    elif failure=='nonbinary':f.loc[0,'prediction']=2
    with pytest.raises(ValueError):m.validate_predictions(f,['a','b'],4)


def test_zero_support_normalization_is_explicit():
    m=module('evidence');f=fixture().query('label==0')
    r=m.confusion_rows(f,'burst_half_proxy')
    assert r.norm_1_0.isna().all() and r.norm_1_1.isna().all()


def test_phase_boundaries_and_no_extension_alias():
    m=module('phase_spike');t=np.arange(2000)/1000;active=np.zeros(2000,bool);active[200:1400]=True
    bursts=m.burst_halves(active,t,1000)
    assert m.phase_of_window(250,500,t,bursts)==0
    assert m.phase_of_window(900,1150,t,bursts)==1
    assert m.phase_of_window(700,950,t,bursts) is None
    assert m.phase_of_window(100,350,t,bursts) is None
    assert not m.burst_halves(np.ones(10000,bool),np.arange(10000)/1000,1000)


def test_auxiliary_head_rejects_group_overlap(tmp_path):
    m=module('phase_spike');rng=np.random.default_rng(4)
    frame=pd.DataFrame(rng.normal(size=(8,10)),columns=m.FEATURE_COLUMNS)
    frame['subject_id']=['g1']*4+['g2']*4;frame['phase_proxy']=[0,1]*4;frame['random_proxy']=[1,0]*4
    head=m.PhaseScores(frame,tmp_path)
    with pytest.raises(AssertionError):head.get(['g1'],np.array([0,1]),'phase','overlap')


def test_published_fold_export_has_no_identifiers(tmp_path):
    m=module('evidence');r=pd.DataFrame({'model':['phase_score'],'fold_id':['F01'],'subject_id':['PRIVATE_PERSON'],
       'source_member':['private/record.csv'],'parameters':['private details'],'f1_macro':[.6],'n_test':[4]})
    dest=m.public_fold_export(r,tmp_path/'public',tmp_path/'private')
    out=pd.read_csv(dest)
    assert set(out.columns)=={'model','fold_id','f1_macro','n_test'}
    assert out.fold_id.iloc[0]!='F01'
    assert 'PRIVATE_PERSON' not in Path(dest).read_text() and 'private/record.csv' not in Path(dest).read_text()


def test_export_receipt_does_not_hash_itself(tmp_path):
    m=module('evidence');(tmp_path/'results.csv').write_text('model,n\na,4\n')
    (tmp_path/'evidence_receipt.json').write_text('old receipt')
    assert set(m.aggregate_hashes(tmp_path))=={'results.csv'}
