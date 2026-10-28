"""Nested group validation. Selection sees only inner validation groups."""
from collections import defaultdict
import json, time, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .contracts import write_json, sha256_file, implementation_provenance
from .features import FEATURE_COLUMNS, AMPLITUDE, SPECTRAL, WAVELET
from .evaluation import binary_metrics, summarize_results, paired_differences

FAMILIES={'amplitude':AMPLITUDE,'spectral':SPECTRAL,'wavelet':WAVELET}


def require_two_classes(y):
    if set(np.unique(y))!={0,1}:raise ValueError('training requires exactly two classes [0,1]')


def make_subject_folds(subject_keys,config=None):
    keys=np.asarray(subject_keys);unique=sorted(set(keys))
    if len(unique)<3:raise ValueError('nested validation requires at least three recording groups')
    return [{'fold_id':f'F{i:02d}','train':np.flatnonzero(keys!=key).tolist(),'test':np.flatnonzero(keys==key).tolist()} for i,key in enumerate(unique,1)]


def candidate_grid():
    grid=[{'family':'majority'}]
    for fam in ['logistic_rms_mdf','logistic_all']:
        grid.extend({'family':fam,'C':c} for c in [.1,1.,10.])
    grid.extend({'family':'svm_rbf','C':c,'gamma':g} for c in [.1,1.,10.] for g in ['scale',.01])
    grid.extend({'family':'random_forest','max_depth':d,'min_samples_leaf':l} for d in [3,6,None] for l in [2,5])
    return grid


def build_pipeline(candidate,seed,columns=None):
    family=candidate['family']
    columns=columns or (['rms','mdf_hz'] if family=='logistic_rms_mdf' else FEATURE_COLUMNS)
    scaler='passthrough' if family in ['random_forest','majority'] else StandardScaler()
    select=ColumnTransformer([('features',scaler,columns)],remainder='drop')
    if family=='majority':est=DummyClassifier(strategy='most_frequent')
    elif family.startswith('logistic'):est=LogisticRegression(C=candidate['C'],class_weight='balanced',max_iter=2000,random_state=seed)
    elif family=='svm_rbf':est=SVC(C=candidate['C'],gamma=candidate['gamma'],class_weight='balanced',random_state=seed)
    elif family=='random_forest':est=RandomForestClassifier(n_estimators=300,max_depth=candidate['max_depth'],min_samples_leaf=candidate['min_samples_leaf'],class_weight='balanced',random_state=seed,n_jobs=2)
    else:raise ValueError('unknown model family')
    return Pipeline([('columns',select),('model',est)])


def fit_pipeline(candidate,train,seed,columns=None):
    require_two_classes(train.label.to_numpy())
    pipe=build_pipeline(candidate,seed,columns)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        with threadpool_limits(limits=1):pipe.fit(train,train.label.astype(int))
    return pipe


def predict_scores(pipe,test):
    est=pipe.named_steps['model'];pred=pipe.predict(test).astype(int)
    if hasattr(est,'predict_proba'):
        score=pipe.predict_proba(test)[:,list(est.classes_).index(1)];kind='native_probability_uncalibrated'
    else:
        raw=pipe.decision_function(test);score=1/(1+np.exp(-np.clip(raw,-700,700)));kind='sigmoid_margin_uncalibrated_score'
    return pred,np.asarray(score),kind


def select_candidates(train,config,candidates,context,columns=None,audit_dir=None):
    n=min(config.inner_folds,train.subject_id.nunique())
    if n<2:raise ValueError('insufficient inner groups')
    folds=list(GroupKFold(n_splits=n).split(train,train.label,groups=train.subject_id))
    rows=[];splits=[];best={}
    for k,(a,b) in enumerate(folds,1):
        splits.append({'context':context,'inner_fold':k,'train_subjects':sorted(set(train.iloc[a].subject_id)),'test_subjects':sorted(set(train.iloc[b].subject_id))})
    for order,c in enumerate(candidates):
        scores=[];failure=None;started=time.perf_counter()
        for a,b in folds:
            tr,va=train.iloc[a],train.iloc[b]
            try:
                pipe=fit_pipeline(c,tr,config.seed,columns);pred,score,_=predict_scores(pipe,va)
                for group in sorted(set(va.subject_id)):
                    mask=va.subject_id.to_numpy()==group
                    scores.append(binary_metrics(va.label.to_numpy()[mask],pred[mask],score[mask])['f1_macro'])
            except (ValueError,RuntimeError,ConvergenceWarning) as e:
                failure=f'{type(e).__name__}: {e}';break
        mean=float(np.mean(scores)) if not failure and scores else None
        row={'context':context,'candidate_order':order,'family':c['family'],'parameters':json.dumps(c,sort_keys=True),'inner_subject_macro_f1':mean,'status':'failed' if failure else 'ok','failure':failure,'seconds':time.perf_counter()-started}
        rows.append(row)
        if audit_dir is not None:
            from pathlib import Path
            audit=Path(audit_dir);audit.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(rows).to_csv(audit/'search_results.csv',index=False)
            write_json(audit/'splits.json',splits)
        if mean is not None and (c['family'] not in best or mean>best[c['family']][0]+1e-12):best[c['family']]=(mean,order,c)
    if not best:raise RuntimeError('all candidate configurations failed; inspect search audit')
    overall=max(best.values(),key=lambda v:(v[0],-v[1]))[2]
    return {fam:v[2] for fam,v in best.items()},overall,rows,splits


def evaluate_nested(features,config,output_dir,candidates=None,columns=None):
    from pathlib import Path
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    completion=out/'evaluation_receipt.json'
    if completion.exists():completion.unlink()
    old_failure=out/'failure_receipt.json'
    if old_failure.exists():old_failure.unlink()
    write_json(out/'RUNNING.json',{'config_sha256':config.digest(),'status':'incomplete'})
    try:
        result=_evaluate_nested(features,config,out,candidates,columns)
        (out/'RUNNING.json').unlink()
        return result
    except BaseException as error:
        if completion.exists():completion.unlink()
        write_json(out/'failure_receipt.json',{'status':'failed_incomplete','error_type':type(error).__name__,'error':str(error),'config_sha256':config.digest()})
        raise


def _evaluate_nested(features,config,output_dir,candidates=None,columns=None):
    from pathlib import Path
    from .workflow import feature_table_digest
    out=Path(output_dir);provenance=implementation_provenance()
    df=features.copy().reset_index(drop=True);require_two_classes(df.label.to_numpy())
    folds=make_subject_folds(df.subject_id,config);results=[];oof=[];search=[];splits=[];selected=[]
    for fold in folds:
        start=time.perf_counter();train=df.iloc[fold['train']];test=df.iloc[fold['test']]
        best,choice,rows,inner=select_candidates(train,config,candidates or candidate_grid(),fold['fold_id'],columns,audit_dir=out/'search_audit'/fold['fold_id'])
        search.extend(rows);splits.extend(inner)
        splits.append({'context':fold['fold_id'],'inner_fold':None,'train_subjects':sorted(set(train.subject_id)),'test_subjects':sorted(set(test.subject_id))})
        selected.append({'fold_id':fold['fold_id'],**choice})
        predictions={}
        for family,c in best.items():
            started=time.perf_counter()
            pipe=fit_pipeline(c,train,config.seed,columns);pred,score,kind=predict_scores(pipe,test)
            predictions[family]=(pred,score,kind)
            m=binary_metrics(test.label.to_numpy(),pred,score)
            results.append({'model':family,'fold_id':fold['fold_id'],'subject_id':str(test.subject_id.iloc[0]),'parameters':json.dumps(c,sort_keys=True),'fit_predict_seconds':time.perf_counter()-started,'score_type':kind,**m})
            for j,(_,r) in enumerate(test.iterrows()):oof.append({'model':family,'fold_id':fold['fold_id'],'window_id':r.window_id,'subject_id':r.subject_id,'side':r.side,'label':int(r.label),'prediction':int(pred[j]),'score':float(score[j])})
        pred,score,kind=predictions[choice['family']]
        results.append({'model':'nested_selector','fold_id':fold['fold_id'],'subject_id':str(test.subject_id.iloc[0]),'parameters':json.dumps(choice,sort_keys=True),'fit_predict_seconds':time.perf_counter()-start,'score_type':kind,**binary_metrics(test.label.to_numpy(),pred,score)})
        for j,(_,r) in enumerate(test.iterrows()):oof.append({'model':'nested_selector','fold_id':fold['fold_id'],'window_id':r.window_id,'subject_id':r.subject_id,'side':r.side,'label':int(r.label),'prediction':int(pred[j]),'score':float(score[j])})
        pd.DataFrame(search).to_csv(out/'search_results.csv',index=False)
        pd.DataFrame(results).to_csv(out/'results.csv',index=False)
        write_json(out/'splits.json',splits)
        print(f'{fold["fold_id"]}/{len(folds)}: {choice["family"]}; macro-F1={results[-1]["f1_macro"]:.4f}; seconds={time.perf_counter()-start:.1f}',flush=True)
    result=pd.DataFrame(results);predictions=pd.DataFrame(oof)
    if predictions.duplicated(['model','window_id']).any():raise RuntimeError('duplicate OOF prediction')
    counts=predictions.groupby('model').size()
    if not (counts==len(df)).all():raise RuntimeError('incomplete OOF coverage')
    predictions.to_parquet(out/'oof_predictions.parquet',index=False)
    summarize_results(result,config.seed,config.bootstrap_replicates).to_csv(out/'metrics_summary.csv',index=False)
    paired_differences(result,config).to_csv(out/'paired_differences.csv',index=False)
    write_json(out/'selected_configurations.json',selected)
    artifacts=['results.csv','search_results.csv','splits.json','oof_predictions.parquet','metrics_summary.csv','paired_differences.csv','selected_configurations.json']
    write_json(out/'evaluation_receipt.json',{'status':'complete','config_sha256':config.digest(),'feature_table_sha256':feature_table_digest(df),'implementation':provenance,'output_hashes':{name:sha256_file(out/name) for name in artifacts},'models':sorted(result.model.unique()),'rows':len(df),'recording_groups':int(df.subject_id.nunique()),'identity_status':config.group_identity_status,'outer_folds':len(folds),'oof_rows':len(predictions),'target':config.label_version,'physiological_reference_available':False,'runtime_scope':'offline','candidate_count':len(candidates or candidate_grid())})
    return {'results':result,'oof':predictions,'search':search,'splits':splits}
