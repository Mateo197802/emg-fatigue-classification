from dataclasses import asdict
from pathlib import Path
import importlib.metadata, json
import joblib
import numpy as np
import pandas as pd
from .contracts import StudyConfig, write_json, sha256_file
from .features import FEATURE_COLUMNS, extract_features
from .preprocessing import preprocess
from .activity import detect_activity
from .labels import make_windows
from .validation import select_candidates, candidate_grid, fit_pipeline, predict_scores


def fit_final(features,config,model_path,candidates=None):
    from .workflow import feature_table_digest, environment
    path=Path(model_path);path.parent.mkdir(parents=True,exist_ok=True)
    _,choice,search,splits=select_candidates(features,config,candidates or candidate_grid(),'final_full_data_inner_selection')
    pipeline=fit_pipeline(choice,features,config.seed)
    bundle={'format_version':1,'pipeline':pipeline,'config':asdict(config),'feature_columns':FEATURE_COLUMNS,
      'selected_configuration':choice,'label_vocabulary':{0:'early_proxy_nonfatigue',1:'late_proxy_fatigue'},
      'target':config.label_version,'physiological_fatigue_detector_validated':False,
      'sklearn_version':importlib.metadata.version('scikit-learn')}
    before=predict_scores(pipeline,features)[1]
    joblib.dump(bundle,path,compress=3)
    write_json(path.with_suffix('.receipt.json'),{'sha256':sha256_file(path),'config_sha256':config.digest(),
      'feature_table_sha256':feature_table_digest(features),'environment':environment(),
      'selected_configuration':choice,'training_windows':len(features),'recording_groups':int(features.subject_id.nunique()),
      'group_identity_status':config.group_identity_status,'evaluation':'full-data fit is not a new held-out performance estimate',
      'search':search,'splits':splits,'sklearn_version':bundle['sklearn_version']})
    restored=load_bundle(path);after=predict_scores(restored['pipeline'],features)[1]
    if not np.allclose(before,after,rtol=0,atol=1e-12):raise RuntimeError('model roundtrip changed scores')
    return bundle


def load_bundle(path):
    path=Path(path);receipt=json.loads(path.with_suffix('.receipt.json').read_text(encoding='utf-8'))
    if sha256_file(path)!=receipt['sha256']:raise ValueError('model checksum mismatch')
    if receipt['sklearn_version']!=importlib.metadata.version('scikit-learn'):raise ValueError('incompatible scikit-learn version')
    # Load only a trusted, locally generated bundle. A matching receipt is integrity, not authenticity.
    bundle=joblib.load(path)
    if bundle['feature_columns']!=FEATURE_COLUMNS or bundle['format_version']!=1:raise ValueError('incompatible feature schema')
    return bundle


def predict_recording(recording,bundle):
    c=StudyConfig(**bundle['config'])
    processed=detect_activity(preprocess(recording,c),c)
    df=extract_features(processed,make_windows(processed,c),c)
    if df.empty:return pd.DataFrame(columns=['start_s','end_s','status','prediction','score'])
    rows=[]
    for _,r in df.iterrows():
        row={'start_s':r.start_s,'end_s':r.end_s,'qc_valid':bool(r.qc_valid),'activity_occupancy':float(r.activity_occupancy),
          'status':'abstain_qc' if not r.qc_valid else 'abstain_activity' if not r.activity_eligible else 'abstain_features' if r.feature_status!='valid' else 'abstain_transition' if pd.isna(r.label) else 'predicted',
          'prediction':None,'predicted_class':None,'score':None,'score_type':None,'interpretation':'weak temporal proxy; not independently validated fatigue'}
        rows.append(row)
    output=pd.DataFrame(rows);mask=output.status=='predicted'
    if mask.any():
        pred,score,kind=predict_scores(bundle['pipeline'],df.loc[mask])
        output.loc[mask,'prediction']=pred;output.loc[mask,'score']=score;output.loc[mask,'score_type']=kind
        output.loc[mask,'predicted_class']=[bundle['label_vocabulary'][int(x)] for x in pred]
    return output
