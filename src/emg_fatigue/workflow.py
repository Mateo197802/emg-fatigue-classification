from dataclasses import asdict, replace
from pathlib import Path
import hashlib, importlib.metadata, time
import numpy as np
import pandas as pd
from .contracts import Recording, write_json, sha256_file
from .preprocessing import preprocess, line_prominence
from .activity import detect_activity
from .labels import make_windows
from .features import extract_features, FEATURE_COLUMNS


def environment():
    import platform
    return {'python':platform.python_version(),**{m:importlib.metadata.version(m) for m in ['numpy','pandas','scipy','scikit-learn','PyWavelets','matplotlib','joblib','pyarrow']}}


def inventory(recordings,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in recordings:
        dt=np.diff(r.t)
        rows.append({'recording_id':r.recording_id,'subject_id':r.subject_id,'source':r.source,'source_member':r.source_member,'side':r.side,'series':r.series,'n_samples':len(r.t),'start_s':float(r.t[0]),'end_s':float(r.t[-1]),'median_fs_hz':float(1/np.median(dt)),'sha256':r.sha256,'formal':r.series!='R0'})
    df=pd.DataFrame(rows);df.to_csv(out/'inventory.csv',index=False)
    df[['recording_id','subject_id','source','source_member','side','series']].to_csv(out/'source_mapping.csv',index=False)
    write_json(out/'input_hashes.json',[{'recording_id':r.recording_id,'sha256':r.sha256} for r in recordings])
    return df


def process_dataset(recordings,config,output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    inv=inventory(recordings,out);frames=[];qc=[];coverage=[];line=[]
    for rec in recordings:
        if rec.series=='R0':continue
        processed=detect_activity(preprocess(rec,config),config)
        windows=make_windows(processed,config)
        frame=extract_features(processed,windows,config)
        if not frame.empty:frames.append(frame)
        for row in processed['qc']:qc.append({'recording_id':rec.recording_id,'subject_id':rec.subject_id,**row})
        for r in processed['runs']:
            prominence=line_prominence(rec.x[r['start_index']:r['start_index']+len(r['x'])],r['fs_hz'])
            line.append({'recording_id':rec.recording_id,'run_id':r['run_id'],**prominence})
        coverage.append({'recording_id':rec.recording_id,'subject_id':rec.subject_id,'side':rec.side,'series':rec.series,
          'raw_samples':len(rec.t),'qc_valid_samples':int(sum(r['valid'].sum() for r in processed['runs'])),
          'candidate_active_samples':int(sum(r['active'].sum() for r in processed['runs'])),
          'candidate_windows':len(frame),'qc_valid_windows':int(frame.qc_valid.sum()) if len(frame) else 0,
          'active_windows':int(frame.activity_eligible.sum()) if len(frame) else 0,
          'feature_valid_windows':int((frame.feature_status=='valid').sum()) if len(frame) else 0,
          'labeled_windows':int(frame.label.notna().sum()) if len(frame) else 0})
    df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df.to_parquet(out/'features.parquet',index=False)
        label_cols=[x for x in df.columns if x not in FEATURE_COLUMNS]
        df[label_cols].to_csv(out/'labels.csv',index=False)
    pd.DataFrame(qc).to_csv(out/'qc.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    pd.DataFrame(line).to_csv(out/'line_prominence.csv',index=False)
    write_json(out/'feature_schema.json',{'ordered_features':FEATURE_COLUMNS,'amplitude_units':'ADC counts, not calibrated muscle mV','spectral_band_hz':[20,400],'wavelet':'db4 level 3 symmetric; relative coefficient energies','metadata_excluded_from_predictors':['subject_id','side','series','time_fraction','label','recording_id']})
    receipt={'config':asdict(config),'config_sha256':config.digest(),'environment':environment(),
      'source_recordings':len(recordings),'formal_recordings':int(inv.formal.sum()),'source_samples':int(inv.n_samples.sum()),
      'recording_groups_source':int(inv.subject_id.nunique()),'windows_all':len(df),
      'windows_labeled':int(df.label.notna().sum()) if not df.empty else 0,
      'label_counts':{str(int(k)):int(v) for k,v in df.label.dropna().value_counts().items()} if not df.empty else {},
      'source_mode':'synthetic_fixture' if recordings and recordings[0].source=='synthetic' else 'supplied_recordings',
      'physical_channel_mode':'unverified','mvc_available':False,'resting_baseline_available':False,
      'physiological_reference_available':False,'seconds':time.perf_counter()-started}
    receipt['features_sha256']=sha256_file(out/'features.parquet') if not df.empty else None
    receipt['input_hashes_sha256']=sha256_file(out/'input_hashes.json')
    write_json(out/'processing_receipt.json',receipt)
    write_json(out/'label_audit.json',{'version':config.label_version,'evidence':'weak elapsed recording time','original_bounds_before_qc':True,'transition_excluded':True,'qc_and_activity_are_not_negative_class':True,'label_counts':receipt['label_counts'],'status_counts':{str(k):int(v) for k,v in df.status.value_counts().items()} if not df.empty else {}})
    print(f'Processed {receipt["formal_recordings"]} formal recordings; {receipt["windows_labeled"]} labeled windows; {receipt["seconds"]:.1f}s',flush=True)
    return {'features':df,'inventory':inv,'receipt':receipt}


def verify_feature_cache(output_dir,config):
    root=Path(output_dir)
    receipt=__import__('json').loads((root/'processing_receipt.json').read_text(encoding='utf-8'))
    if receipt['config_sha256']!=config.digest():raise ValueError('cached configuration mismatch')
    if sha256_file(root/'features.parquet')!=receipt.get('features_sha256'):
        raise ValueError('feature cache checksum mismatch; regenerate from original recordings')
    return receipt


def feature_table_digest(frame):
    columns=['window_id','subject_id','label']+FEATURE_COLUMNS
    values=pd.util.hash_pandas_object(frame[columns],index=False).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def eligible_features(df):
    valid=df[df.label.notna() & (df.feature_status=='valid')].copy()
    if valid.empty:raise ValueError('no eligible labeled feature windows')
    valid['label']=valid.label.astype(int)
    if not np.isfinite(valid[FEATURE_COLUMNS].to_numpy()).all():raise ValueError('nonfinite eligible features')
    return valid


def synthetic_recordings(groups=4,seconds=12,seed=20261028):
    rng=np.random.default_rng(seed);recordings=[]
    t=np.arange(int(seconds*1000))/1000
    for i in range(groups):
        freq=105-45*t/seconds+2*i
        phase=2*np.pi*np.cumsum(freq)/1000
        amp=np.where((t%2)<1,1.,.06)
        x=amp*(np.sin(phase)+.35*np.sin(phase*1.7))+.02*rng.normal(size=len(t))
        recordings.append(Recording(f'SYN_REC{i+1:02d}',f'SYN_G{i+1:02d}','synthetic','left' if i%2==0 else 'right','R1',t.copy(),x,hashlib.sha256(x.tobytes()).hexdigest(),'software_fixture_only'))
    return recordings
