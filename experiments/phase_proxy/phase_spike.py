"""Exploratory grouped cross-fitted activity-burst phase experiment.

The auxiliary reference is EMG burst position, not observed knee extension.
All individual data and predictions remain in this private supplement.
"""
from pathlib import Path
import argparse,hashlib,json,time,traceback
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from emg_fatigue.contracts import read_config,write_json,sha256_file,implementation_provenance
from emg_fatigue.io import load_recordings
from emg_fatigue.preprocessing import preprocess
from emg_fatigue.activity import detect_activity
from emg_fatigue.features import FEATURE_COLUMNS
from emg_fatigue.workflow import verify_feature_cache,eligible_features
from emg_fatigue.reporting import verify_evaluation
from emg_fatigue.validation import make_subject_folds
from emg_fatigue.evaluation import binary_metrics,summarize_results

SEED=20261028
MODELS=['baseline_10','phase_score','phase_conditioned','phase_only','random_conditioned']

def burst_halves(active,t,fs):
    active=np.asarray(active,dtype=bool)
    boundaries=np.diff(np.r_[False,active,False].astype(int))
    runs=[]
    for a,b in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
        duration=float(t[b-1]-t[a]+1/fs)
        if .5<=duration<=8.:
            runs.append((int(a),int(b),(float(t[a])+float(t[b-1]))/2,duration))
    return runs

def phase_of_window(a,b,t,bursts):
    for start,end,mid,duration in bursts:
        if start<=a and b<=end:
            if t[b-1]<mid:return 0
            if t[a]>=mid:return 1
            return None
    return None

def extract_phase(cache,recordings,config,out):
    frame=cache[cache.feature_status=='valid'].copy().reset_index(drop=True)
    frame['phase_proxy']=np.nan;bursts_audit=[]
    for k,rec in enumerate(recordings):
        if rec.series=='R0':continue
        p=detect_activity(preprocess(rec,config),config)
        for r in p['runs']:
            bursts=burst_halves(r['active'],r['t'],r['fs_hz'])
            for a,b,mid,dur in bursts:
                bursts_audit.append({'recording_id':rec.recording_id,'subject_id':rec.subject_id,'run_id':r['run_id'],'start_index':a,'end_index':b,'start_s':float(r['t'][a]),'end_s':float(r['t'][b-1]),'duration_s':dur})
            select=(frame.recording_id==rec.recording_id)&(frame.run_id==r['run_id'])
            for i,row in frame[select].iterrows():
                v=phase_of_window(int(row.start),int(row.end),r['t'],bursts)
                if v is not None:frame.loc[i,'phase_proxy']=v
        if (k+1)%20==0:print(f'phase extraction {k+1}/{len(recordings)}',flush=True)
    frame['random_proxy']=[int(hashlib.sha256((str(SEED)+':'+s).encode()).digest()[0]>=128) for s in frame.window_id]
    pd.DataFrame(bursts_audit).to_csv(out/'candidate_bursts.csv',index=False)
    frame.to_parquet(out/'phase_features.parquet',index=False)
    return frame

def matrix(x,p,model):
    if model=='baseline_10':return x
    if model=='phase_only':return p[:,None]
    if model=='phase_score':return np.column_stack((x,p))
    if model in ['phase_conditioned','random_conditioned']:
        columns=[FEATURE_COLUMNS.index(k) for k in ['rms','mdf_hz','mnf_hz']]
        return np.column_stack((x,p,x[:,columns]*p[:,None]))
    raise ValueError(model)

def fit_second(x,y,c):
    assert set(y)=={0,1}
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=SEED))
    pipe.fit(x,y)
    return pipe

def grouped_mean(y,pred,groups):
    return float(np.mean([binary_metrics(y[groups==g],pred[groups==g],pred[groups==g])['f1_macro'] for g in sorted(set(groups))]))

class PhaseScores:
    def __init__(self,all_features,out):
        self.all=all_features;self.out=out;self.cache={};self.audit=[];self.requests=[]
        self.x=all_features[FEATURE_COLUMNS].to_numpy();self.groups=all_features.subject_id.to_numpy()
        self.known=all_features.phase_proxy.notna().to_numpy()
        self.y={'phase':all_features.phase_proxy.to_numpy(),'random':all_features.random_proxy.to_numpy()}
    def get(self,train_groups,test_indices,kind,context):
        train_groups=tuple(sorted(set(train_groups)))
        test_groups=tuple(sorted(set(self.groups[test_indices])))
        assert not set(train_groups)&set(test_groups),(context,train_groups,test_groups)
        self.requests.append({'context':context,'kind':kind,'train_groups':train_groups,'test_groups':test_groups,'test_rows':len(test_indices)})
        key=(train_groups,kind)
        if key not in self.cache:
            mask=np.isin(self.groups,train_groups)&self.known
            y=self.y[kind][mask].astype(int)
            if set(y)!={0,1}:raise ValueError('auxiliary head missing a class: '+context)
            head=RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_leaf=5,class_weight='balanced',random_state=SEED,n_jobs=1)
            head.fit(self.x[mask],y)
            self.cache[key]=head.predict_proba(self.x)[:,list(head.classes_).index(1)]
            self.audit.append({'kind':kind,'train_groups':train_groups,'train_rows':int(mask.sum()),'class0':int((y==0).sum()),'class1':int((y==1).sum())})
        return self.cache[key][test_indices]
    def crossfit(self,indices,kind,context,n_splits=4):
        groups=self.groups[indices];pred=np.full(len(indices),np.nan);seen=np.zeros(len(indices),int)
        for j,(a,b) in enumerate(GroupKFold(n_splits=min(n_splits,len(set(groups)))).split(indices,groups=groups)):
            pred[b]=self.get(groups[a],indices[b],kind,context+f'/crossfit{j+1}');seen[b]+=1
        assert np.isfinite(pred).all() and (seen==1).all()
        return pred
    def save(self):
        write_json(self.out/'head_training_audit.json',self.audit)
        write_json(self.out/'disjoint_group_requests.json',self.requests)

def evaluate(frame,config,out,primary):
    idx=np.flatnonzero(frame.label.notna().to_numpy())
    data=frame.iloc[idx].copy();data['label']=data.label.astype(int)
    assert len(data)==5509 and data.subject_id.nunique()==15
    reference=eligible_features(pd.read_parquet(primary/'features.parquet'))
    assert list(data.window_id)==list(reference.window_id)
    np.testing.assert_array_equal(data[FEATURE_COLUMNS].to_numpy(),reference[FEATURE_COLUMNS].to_numpy())
    x=data[FEATURE_COLUMNS].to_numpy();y=data.label.to_numpy();groups=data.subject_id.to_numpy()
    heads=PhaseScores(frame,out);results=[];oof=[];search=[];splits=[];aux=[]
    for fold in make_subject_folds(groups):
        started=time.perf_counter();tr=np.array(fold['train']);te=np.array(fold['test'])
        tg=groups[tr];test_groups=sorted(set(groups[te]))
        assert not set(tg)&set(test_groups)
        splits.append({'context':fold['fold_id'],'train_groups':sorted(set(tg)),'test_groups':test_groups})
        inner=list(GroupKFold(n_splits=4).split(tr,groups=tg))
        inner_matrices=[]
        for j,(a,b) in enumerate(inner):
            itr,iva=tr[a],tr[b];ctx=fold['fold_id']+f'/inner{j+1}'
            splits.append({'context':ctx,'train_groups':sorted(set(groups[itr])),'test_groups':sorted(set(groups[iva]))})
            scores={}
            for kind in ['phase','random']:
                pt=heads.crossfit(idx[itr],kind,ctx+'/train',n_splits=3)
                pv=heads.get(groups[itr],idx[iva],kind,ctx+'/validation')
                scores[kind]=(pt,pv)
            inner_matrices.append((itr,iva,scores))
        selected={}
        for model in MODELS:
            kind='random' if model=='random_conditioned' else 'phase';best=(-np.inf,None)
            for c in [.1,1.,10.]:
                values=[]
                for itr,iva,scores in inner_matrices:
                    pt,pv=scores[kind]
                    pipe=fit_second(matrix(x[itr],pt,model),y[itr],c)
                    pred=pipe.predict(matrix(x[iva],pv,model))
                    for g in sorted(set(groups[iva])):
                        mask=groups[iva]==g
                        values.append(binary_metrics(y[iva][mask],pred[mask],pred[mask])['f1_macro'])
                mean=float(np.mean(values));search.append({'fold_id':fold['fold_id'],'model':model,'C':c,'inner_group_macro_f1':mean})
                if mean>best[0]+1e-12:best=(mean,c)
            selected[model]=best[1]
        outer_scores={}
        for kind in ['phase','random']:
            outer_scores[kind]=(heads.crossfit(idx[tr],kind,fold['fold_id']+'/train'),heads.get(groups[tr],idx[te],kind,fold['fold_id']+'/test'))
        for model in MODELS:
            kind='random' if model=='random_conditioned' else 'phase';pt,pv=outer_scores[kind]
            pipe=fit_second(matrix(x[tr],pt,model),y[tr],selected[model])
            pred=pipe.predict(matrix(x[te],pv,model)).astype(int);score=pipe.predict_proba(matrix(x[te],pv,model))[:,1]
            results.append({'model':model,'fold_id':fold['fold_id'],'subject_id':test_groups[0],'C':selected[model],**binary_metrics(y[te],pred,score)})
            for k,i in enumerate(te):oof.append({'model':model,'fold_id':fold['fold_id'],'window_id':data.iloc[i].window_id,'subject_id':groups[i],'label':int(y[i]),'prediction':int(pred[k]),'score':float(score[k])})
        ai=np.flatnonzero((frame.subject_id==test_groups[0]).to_numpy()&frame.phase_proxy.notna().to_numpy())
        if len(ai):
            ap=heads.get(groups[tr],ai,'phase',fold['fold_id']+'/auxiliary_test')
            aux.append({'fold_id':fold['fold_id'],'subject_id':test_groups[0],**binary_metrics(frame.iloc[ai].phase_proxy.to_numpy().astype(int),(ap>=.5).astype(int),ap)})
        pd.DataFrame(results).to_csv(out/'results.csv',index=False)
        pd.DataFrame(search).to_csv(out/'search_results.csv',index=False)
        heads.save();write_json(out/'splits.json',splits)
        latest={r['model']:round(r['f1_macro'],4) for r in results[-len(MODELS):]}
        print(f'{fold["fold_id"]}/15 {latest}; {time.perf_counter()-started:.1f}s',flush=True)
    results=pd.DataFrame(results);oof=pd.DataFrame(oof)
    assert not oof.duplicated(['model','window_id']).any()
    assert (oof.groupby('model').size()==5509).all()
    oof.to_parquet(out/'oof_predictions.parquet',index=False)
    pd.DataFrame(aux).to_csv(out/'auxiliary_phase_metrics.csv',index=False)
    summarize_results(results,SEED,5000).to_csv(out/'metrics_summary.csv',index=False)
    ref_oof=pd.read_parquet(primary/'evaluation/oof_predictions.parquet').query("model=='logistic_all'")
    check=oof.query("model=='baseline_10'").merge(ref_oof,on='window_id',suffixes=('_new','_original'),validate='one_to_one')
    assert len(check)==5509
    assert np.array_equal(check.label_new,check.label_original)
    assert np.array_equal(check.prediction_new,check.prediction_original),'baseline prediction mismatch'
    maxdiff=float(np.max(np.abs(check.score_new-check.score_original)))
    assert maxdiff<1e-9,'baseline score mismatch'
    rng=np.random.default_rng(SEED);paired=[]
    comparisons=[(m,'baseline_10') for m in MODELS if m!='baseline_10']+[('phase_conditioned','random_conditioned')]
    for model,baseline in comparisons:
        a=results.query('model==@model').set_index('fold_id').f1_macro
        b=results.query('model==@baseline').set_index('fold_id').f1_macro
        d=(a-b).to_numpy();boots=rng.choice(d,size=(5000,len(d)),replace=True).mean(axis=1)
        paired.append({'comparison':model+'_minus_'+baseline,'paired_groups':15,'mean_delta':float(d.mean()),'ci95_lower':float(np.quantile(boots,.025)),'ci95_upper':float(np.quantile(boots,.975)),'groups_improved':int((d>1e-12).sum()),'groups_worse':int((d< -1e-12).sum()),'groups_equal':int((abs(d)<=1e-12).sum())})
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    return {'baseline_matches_original':True,'baseline_score_max_abs_difference':maxdiff,'auxiliary_head_fits':len(heads.audit),'disjoint_requests':len(heads.requests),'result_rows':len(results),'oof_rows':len(oof),'window_count':5509,'outer_groups':15,'models':MODELS}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--primary',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'completion_receipt.json').exists():raise ValueError('completed experiment already exists; select a new output directory')
    write_json(out/'RUNNING.json',{'status':'running','started_utc':pd.Timestamp.now(tz='UTC').isoformat()})
    start=time.perf_counter();c=read_config(args.config);primary=Path(args.primary)
    verify_feature_cache(primary,c);verify_evaluation(primary,c)
    frame=extract_phase(pd.read_parquet(primary/'features.parquet'),load_recordings(args.source,c),c,out)
    phase_counts=frame.groupby(['subject_id','phase_proxy']).size().rename('windows').reset_index();phase_counts.to_csv(out/'phase_support_by_group.csv',index=False)
    write_json(out/'phase_extraction_receipt.json',{'reference':'first/second half of EMG activity burst; not knee extension','phase_labeled_windows':int(frame.phase_proxy.notna().sum()),'eligible_bursts':len(pd.read_csv(out/'candidate_bursts.csv')),'all_feature_valid_windows':len(frame),'phase_classes':{str(int(k)):int(v) for k,v in frame.phase_proxy.dropna().value_counts().items()},'source_sha256':sha256_file(args.source),'primary_features_sha256':sha256_file(primary/'features.parquet'),'config_sha256':c.digest()})
    print('phase support',frame.phase_proxy.value_counts().to_dict(),flush=True)
    with threadpool_limits(limits=1):result=evaluate(frame,c,out,primary)
    result.update({'status':'complete','scope':'exploratory EMG-burst phase approximation and weak early/late target; no verified extension or physiological fatigue','seconds':time.perf_counter()-start,'script_sha256':sha256_file(__file__),'protocol_sha256':sha256_file(Path(__file__).with_name('PROTOCOLO_ANTES_DE_RESULTADOS.md')),'implementation':implementation_provenance()})
    result['output_hashes']={f.name:sha256_file(f) for f in out.iterdir() if f.is_file() and f.name not in ['RUNNING.json','failure_receipt.json','completion_receipt.json']}
    write_json(out/'completion_receipt.json',result);(out/'RUNNING.json').unlink();print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
