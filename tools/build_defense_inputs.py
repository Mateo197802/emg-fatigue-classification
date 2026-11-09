"""Prepare private presentation inputs from frozen results, without retraining."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, periodogram, sosfreqz
from emg_fatigue.contracts import read_config
from emg_fatigue.defense_provenance import verify_archive_matches_frozen_sources
from emg_fatigue.defense_figures import write_spanish_defense_figures
from emg_fatigue.evaluation import paired_metric_contrast
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
source_provenance=verify_archive_matches_frozen_sources(Path(a.archive),root)
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
signal_start=int(valid[0]);signal_fs=float(run['fs_hz'])
signal_n=min(int(round(signal_fs)),int(valid[-1])-signal_start+1)
raw_signal=np.asarray(good.x[run['start_index']+signal_start:run['start_index']+signal_start+signal_n],dtype=float)
filtered_signal=np.asarray(run['x'][signal_start:signal_start+signal_n],dtype=float)
if len(raw_signal)!=signal_n or signal_n<8:raise ValueError('QC-selected signal interval is too short for the evidence plot')
raw_signal=raw_signal-raw_signal.mean()
freq,psd_stored=periodogram(raw_signal,fs=signal_fs,window='hann',detrend=False,scaling='density')
_,psd_filtered=periodogram(filtered_signal,fs=signal_fs,window='hann',detrend=False,scaling='density')
sos=butter(config.filter_order,config.band_hz,btype='bandpass',fs=signal_fs,output='sos')
response_hz,response=sosfreqz(sos,worN=2048,fs=signal_fs)
response_db=20*np.log10(np.maximum(np.abs(response),np.finfo(float).tiny))
def measures(frame):
    return {model:{metric:{'mean':float(row['mean']),'sd':float(row['sd'])}
        for _,row in group.iterrows() if (metric:=row['metric']) in ['f1_macro','roc_auc','accuracy']}
        for model,group in frame.groupby('model')}
data={'repository':'https://github.com/Mateo197802/emg-fatigue-classification','include_phase':True,
 'corpus':{'recordings':len(inv),'formal_recordings':len(formal),'groups':int(formal.subject_id.nunique()),
  'samples':int(inv.n_samples.sum()),'duration_median_s':float((formal.end_s-formal.start_s).median()),
  'right_recordings':int(formal.side.eq('right').sum()),'left_recordings':int(formal.side.eq('left').sum()),
  'coverage':{k:int(coverage[k].sum()) for k in ['candidate_windows','qc_valid_windows','active_windows','feature_valid_windows','labeled_windows']}},
 'signal':{'time':run['t'][indices].tolist(),'stored':(good.x[run['start_index']+indices]-np.mean(raw_signal)).tolist(),
  'filtered':run['x'][indices].tolist(),'display_step':4,'sample_rate_hz':signal_fs,
  'psd_frequency_hz':freq.tolist(),'psd_stored':psd_stored.tolist(),'psd_filtered':psd_filtered.tolist(),
  'filter_frequency_hz':response_hz.tolist(),'filter_response_db':response_db.tolist(),
  'selection':'first non-R0 record with a QC-valid run; no use of outcomes'},
 'trajectories':{f:[float(trajectory[trajectory.time_bin==i][f].mean()) for i in range(10)] for f in ['rms','mdf_hz','mnf_hz']},
 'primary':measures(pd.read_csv(ev/'metrics_summary.csv')),
 'phase':measures(pd.read_csv(phase/'metrics_summary.csv')),
 'selector_groups':results[results.model.eq('nested_selector')].f1_macro.tolist(),
 'confusions':pd.read_csv(phase/'confusion_matrices.csv').fillna('').to_dict('records'),
 'auxiliary':{m:{'mean':float(aux[m].mean()),'sd':float(aux[m].std(ddof=1))} for m in ['accuracy','f1_macro','roc_auc']},
 'accuracy_contrasts':{
  'primary':paired_metric_contrast(results,candidate='nested_selector',baseline='logistic_rms_mdf',metric='accuracy',bootstrap_replicates=config.bootstrap_replicates,seed=config.seed),
  'phase':paired_metric_contrast(pd.read_csv(phase/'results.csv'),candidate='phase_conditioned',baseline='baseline_10',metric='accuracy',bootstrap_replicates=config.bootstrap_replicates,seed=config.seed),
 },
 'paired':pd.read_csv(phase/'paired_differences.csv').to_dict('records'),
 'primary_f1_pair':pd.read_csv(ev/'paired_differences.csv').query("comparison=='nested_selector_minus_logistic_rms_mdf'").iloc[0].to_dict(),
 'provenance':{'archive_sha256':source_provenance['archive_sha256'],
  'source_member_manifest_sha256':source_provenance['source_member_manifest_sha256'],
  'source_recordings_verified':source_provenance['recordings_verified'],
  'primary_receipt_sha256':sha(ev/'evaluation_receipt.json'),
  'phase_receipt_sha256':sha(phase/'evidence_receipt.json'),'trajectory_sha256':sha(root/'report_assets/feature_trajectories_by_group.csv')}}
assert data['corpus']['recordings']==140 and data['corpus']['samples']==4256257
out.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
figure_outputs=write_spanish_defense_figures(
 root,data['signal'],
 {'archive_sha256':source_provenance['archive_sha256'],
  'source_member_manifest_sha256':source_provenance['source_member_manifest_sha256'],
  'slides_input_sha256':sha(out)})
print(json.dumps({'status':'verified','corpus':data['corpus'],'spanish_figures':{k:v.name for k,v in figure_outputs.items()}},indent=2))
