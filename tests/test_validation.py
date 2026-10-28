import numpy as np
import pandas as pd
import pytest


def fixture():
    from emg_fatigue.features import FEATURE_COLUMNS
    rng=np.random.default_rng(11);n=48
    df=pd.DataFrame(rng.normal(size=(n,len(FEATURE_COLUMNS))),columns=FEATURE_COLUMNS)
    df['label']=np.tile([0,1],24);df['subject_id']=np.repeat(['A','B','C','D'],12)
    df['rms']=df.label+.1*rng.normal(size=n);df['side']=np.tile(['left','right'],24)
    df['window_id']=[f'w{i}' for i in range(n)]
    return df


def test_every_group_held_out_once_with_both_legs_together():
    from emg_fatigue.validation import make_subject_folds
    df=fixture();folds=make_subject_folds(df.subject_id)
    tested=[]
    for fold in folds:
        assert set(df.iloc[fold['train']].subject_id).isdisjoint(df.iloc[fold['test']].subject_id)
        tested.extend(fold['test'])
    assert sorted(tested)==list(range(len(df)))


def test_test_outlier_cannot_change_scaler():
    from emg_fatigue.validation import build_pipeline
    df=fixture();pipe=build_pipeline({'family':'logistic_rms_mdf','C':1.},seed=1)
    train=df.iloc[:36];pipe.fit(train,train.label)
    before=pipe.named_steps['columns'].named_transformers_['features'].mean_.copy()
    test=df.iloc[36:].copy();test['rms']=1e12
    pipe.predict(test)
    np.testing.assert_array_equal(before,pipe.named_steps['columns'].named_transformers_['features'].mean_)
    assert before[0]==pytest.approx(train.rms.mean())


def test_binary_metrics_include_only_two_classes_and_undefined_auc():
    from emg_fatigue.evaluation import binary_metrics
    m=binary_metrics([0,0,1,1],[0,1,1,1],[.1,.6,.8,.9])
    assert m['f1_macro']==pytest.approx((2/3+.8)/2)
    assert m['cm_0_0']==1 and m['cm_0_1']==1 and m['cm_1_1']==2
    z=binary_metrics([0,0],[0,0],[.1,.2])
    assert z['roc_auc'] is None and z['auc_status']=='undefined_single_class'


def test_nested_smoke_has_unique_oof_and_disjoint_inner_groups(tmp_path):
    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.validation import evaluate_nested
    df=fixture()
    candidates=[{'family':'majority'},{'family':'logistic_rms_mdf','C':1.}]
    result=evaluate_nested(df,StudyConfig(inner_folds=2,bootstrap_replicates=20),tmp_path,candidates=candidates)
    assert len(result['results'])==12  # 2 fixed families + nested selector, 4 folds
    oof=result['oof'];assert not oof.duplicated(['model','window_id']).any()
    for _,g in oof.groupby('model'):assert len(g)==len(df)
    for split in result['splits']:
        assert set(split['train_subjects']).isdisjoint(split['test_subjects'])


def test_single_class_training_is_a_recorded_failure():
    from emg_fatigue.validation import require_two_classes
    with pytest.raises(ValueError,match='two classes'):require_two_classes(np.zeros(10))
