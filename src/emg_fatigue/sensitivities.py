from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from .contracts import write_json
from .features import FEATURE_COLUMNS, AMPLITUDE, SPECTRAL, WAVELET
from .workflow import process_dataset, eligible_features
from .validation import evaluate_nested
from .evaluation import binary_metrics


def sensitivity_variants(config):
    return {'no_overlap':replace(config,overlap=0.),
      'window_500ms':replace(config,window_s=.5),'window_1000ms':replace(config,window_s=1.),
      'activity_50pct':replace(config,activity_occupancy=.5),
      'boundary_minus_5pct':replace(config,boundary_shift=-.05),
      'boundary_plus_5pct':replace(config,boundary_shift=.05),
      'boundary_minus_10pct':replace(config,boundary_shift=-.1),
      'boundary_plus_10pct':replace(config,boundary_shift=.1),
      'conditional_notch':replace(config,notch=True)}


def ablation_columns():
    return {'without_amplitude':[x for x in FEATURE_COLUMNS if x not in AMPLITUDE],
      'without_spectral':[x for x in FEATURE_COLUMNS if x not in SPECTRAL],
      'without_wavelet':[x for x in FEATURE_COLUMNS if x not in WAVELET]}


def run_sensitivities(recordings,config,output_root):
    out=Path(output_root);root=out/'sensitivity';root.mkdir(parents=True,exist_ok=True);rows=[]
    # Each variant is fully specified before inspecting its held-out scores.
    for name,c in sensitivity_variants(config).items():
        folder=root/name
        frame=process_dataset(recordings,c,folder)['features']
        try:
            f=eligible_features(frame)
            result=evaluate_nested(f,c,folder/'evaluation',candidates=[{'family':'logistic_rms_mdf','C':1.}])
            metric=result['results'].query("model == 'logistic_rms_mdf'").f1_macro
            rows.append({'analysis':name,'status':'evaluated','n_windows':len(f),'groups':int(f.subject_id.nunique()),'macro_f1_mean':float(metric.mean()),'macro_f1_sd':float(metric.std(ddof=1)),'model':'fixed_logistic_rms_mdf_C1','config_sha256':c.digest()})
        except (ValueError,RuntimeError) as e:
            rows.append({'analysis':name,'status':'not_evaluable','reason':str(e),'config_sha256':c.digest()})
        pd.DataFrame(rows).to_csv(out/'sensitivity_summary.csv',index=False)
    primary=eligible_features(pd.read_parquet(out/'features.parquet'))
    ablation=[]
    for name,columns in ablation_columns().items():
        r=evaluate_nested(primary,config,out/'ablation'/name,
          candidates=[{'family':'logistic_all','C':c} for c in [.1,1.,10.]],columns=columns)
        values=r['results'].query("model == 'logistic_all'").f1_macro
        ablation.append({'ablation':name,'macro_f1_mean':float(values.mean()),'macro_f1_sd':float(values.std(ddof=1)),'columns':json.dumps(columns),'groups':int(primary.subject_id.nunique())})
    pd.DataFrame(ablation).to_csv(out/'ablation_summary.csv',index=False)
    oracle=(primary.time_fraction>=.5).astype(int)
    write_json(out/'time_oracle_diagnostic.json',{'interpretation':'tautological elapsed-time label oracle; uses full-record bounds; not a learned or deployable fatigue model',**binary_metrics(primary.label,oracle,primary.time_fraction)})
    oof=pd.read_parquet(out/'evaluation/oof_predictions.parquet');sides=[]
    for (model,subject,side),g in oof.groupby(['model','subject_id','side']):
        sides.append({'model':model,'subject_id':subject,'side':side,**binary_metrics(g.label,g.prediction,g.score)})
    pd.DataFrame(sides).to_csv(out/'side_metrics.csv',index=False)
    write_json(out/'sensitivity_receipt.json',{'primary_config_sha256':config.digest(),'variations':9,'ablation_subsets':3,'sensitivity_model':'fixed C=1 RMS+MDF logistic; all held-out groups re-evaluated','ablation_model':'logistic with inner grouped C tuning','physiological_validation':False})
