"""Prepare private presentation inputs from frozen results, without retraining."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from emg_fatigue.contracts import read_config
from emg_fatigue.io import load_recordings
from emg_fatigue.preprocessing import preprocess

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--archive',required=True)
p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.root).resolve();repo=Path(__file__).resolve().parents[1]
out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ev=root/'evaluation';phase=repo/'results/phase_experiment'
for folder,receipt,key in [(ev,'evaluation_receipt.json','output_hashes'),(phase,'evidence_receipt.json','aggregate_hashes')]:
    receipt=json.loads((folder/receipt).read_text(encoding='utf-8'))
    for name,expected in receipt[key].items():assert sha(folder/name)==expected,name
inv=pd.read_csv(root/'inventory.csv');formal=inv[inv.formal.astype(str).str.lower().eq('true')]
coverage=pd.read_csv(root/'coverage.csv');results=pd.read_csv(ev/'results.csv')
trajectory=pd.read_csv(root/'report_assets/feature_trajectories_by_group.csv')
aux=pd.read_csv(phase/'auxiliary_phase_fold_metrics.csv')
assert len(results)==90 and coverage.labeled_windows.sum()==5509 and len(aux)==15
records=[r for r in load_recordings(a.archive) if r.series!='R0'];config=read_config(repo/'configs/study.yaml')
good=next(r for r in records if preprocess(r,config)['runs'])
run=preprocess(good,config)['runs'][0];valid=np.flatnonzero(run['valid'])
indices=np.arange(valid[0],min(valid[0]+1000,valid[-1]),4)
def measures(frame):
    return {model:{metric:{'mean':float(row['mean']),'sd':float(row['sd'])}
        for _,row in group.iterrows() if (metric:=row['metric']) in ['f1_macro','roc_auc','accuracy']}
        for model,group in frame.groupby('model')}
data={'repository':'https://github.com/Mateo197802/emg-fatigue-classification','include_phase':True,
 'corpus':{'recordings':len(inv),'formal_recordings':len(formal),'groups':int(formal.subject_id.nunique()),
  'samples':int(inv.n_samples.sum()),'duration_median_s':float((formal.end_s-formal.start_s).median()),
  'right_recordings':int(formal.side.eq('right').sum()),'left_recordings':int(formal.side.eq('left').sum()),
  'coverage':{k:int(coverage[k].sum()) for k in ['candidate_windows','qc_valid_windows','active_windows','feature_valid_windows','labeled_windows']}},
 'signal':{'time':run['t'][indices].tolist(),'stored':good.x[run['start_index']+indices].tolist(),
  'filtered':run['x'][indices].tolist(),'display_step':4,
  'selection':'first non-R0 record with a QC-valid run; no use of outcomes'},
 'trajectories':{f:[float(trajectory[trajectory.time_bin==i][f].mean()) for i in range(10)] for f in ['rms','mdf_hz','mnf_hz']},
 'primary':measures(pd.read_csv(ev/'metrics_summary.csv')),
 'phase':measures(pd.read_csv(phase/'metrics_summary.csv')),
 'selector_groups':results[results.model.eq('nested_selector')].f1_macro.tolist(),
 'confusions':pd.read_csv(phase/'confusion_matrices.csv').fillna('').to_dict('records'),
 'auxiliary':{m:{'mean':float(aux[m].mean()),'sd':float(aux[m].std(ddof=1))} for m in ['f1_macro','roc_auc']},
 'paired':pd.read_csv(phase/'paired_differences.csv').to_dict('records'),
 'provenance':{'archive_sha256':sha(a.archive),'primary_receipt_sha256':sha(ev/'evaluation_receipt.json'),
  'phase_receipt_sha256':sha(phase/'evidence_receipt.json'),'trajectory_sha256':sha(root/'report_assets/feature_trajectories_by_group.csv')}}
assert data['corpus']['recordings']==140 and data['corpus']['samples']==4256257
out.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':'verified','corpus':data['corpus']},indent=2))
