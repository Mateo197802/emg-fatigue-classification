"""Generate numerical tables and figures only from frozen private outputs."""
from pathlib import Path
import hashlib, json, secrets
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .contracts import write_json, sha256_file
from .features import FEATURE_COLUMNS
from .evaluation import binary_metrics

METRICS=['accuracy','precision_macro','recall_macro','f1_macro','balanced_accuracy','roc_auc','average_precision','pr_auc_trapezoidal','cohen_kappa','specificity_class1','precision_class0','recall_class0','f1_class0','support_class0','precision_class1','recall_class1','f1_class1','support_class1','n_test','cm_0_0','cm_0_1','cm_1_0','cm_1_1','auc_status','score_type']


def report_equation_rows():
    """Return equations and status-qualified parameters for the sEMG protocols."""
    return [
        ('Butterworth analog prototype',
         '|H(jω)|² = 1 / (1 + (ω/ωc)^(2n));  s_k = ωc exp(jπ(2k+n−1)/(2n)), k=1..n',
         'Low-pass prototype relation. The executed digital band-pass is obtained by the bilinear-transform design for 20–400 Hz; order n=4 refers to the prototype and yields digital order 8.'),
        ('Butterworth SOS',
         'H(z) = product[r=1..4] (b0,r + b1,r z^-1 + b2,r z^-2) / (1 + a1,r z^-1 + a2,r z^-2); '
         'y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]',
         '20–400 Hz; prototype order 4, digital band-pass order 8; coefficients are designed per run from fs; SOS forward/backward with sosfiltfilt; 100 ms masked at each edge'),
        ('Notch sensitivity',
         'w0 = 2 pi f0/fs; alpha = sin(w0)/(2Q); b = [1,-2cos(w0),1]/(1+alpha); '
         'a = [1,-2cos(w0)/(1+alpha),(1-alpha)/(1+alpha)]',
         'Not applied in primary processing; apply at 50 or 60 Hz only when prominence >10 dB; Q=30'),
        ('Otsu activity',
         't* = argmax[t] w0(t) w1(t) (mu0(t)-mu1(t))^2',
         'RMS envelope frames of 50 ms with 25 ms hop; 128-bin histogram; candidate activity only, not a fatigue label'),
        ('Window indices',
         'N = round(0.25 fs); H = round(N(1-0.50)); window k = x[kH : kH+N]',
         'Current analyzed run uses 250 ms windows, 50% overlap, and at least 75% candidate-active occupancy; at 1 kHz N=250 and H=125 samples'),
        ('Reference window indices',
         'N = round(T fs); H = round(N(1-o)); t_k = t_0 + kH/fs; Δf = fs/N',
         'Proposed reference-backed primary: T=500 ms and o=50%; 250 ms is a sensitivity analysis. At 1 kHz: primary N=500, H=250, Δf≈2 Hz; sensitivity N=250, H=125, Δf≈4 Hz. Not run because Gate 0 fails.'),
        ('Hann PSD',
         'w[n] = 0.5 (1-cos(2 pi n/(N-1))); Pxx[k] = c_k |sum[n] x[n] w[n] exp(-i 2 pi kn/N)|^2 / (fs sum[n] w[n]^2); c_k=1 at DC/Nyquist and 2 at interior positive-frequency bins',
         'SciPy one-sided density periodogram with detrend=False; unrectified centered window; spectral features use 20–400 Hz; nominal bin spacing fs/N is 4 Hz for N=250 at 1 kHz'),
        ('RMS',
         'RMS = sqrt((1/N) sum[n=1..N] x[n]^2)',
         'Amplitude in stored ADC counts; x is centered within each window'),
        ('MAV',
         'MAV = (1/N) sum[n=1..N] abs(x[n])',
         'Amplitude in stored ADC counts; x is centered within each window'),
        ('Waveform length per second',
         'WL/s = (fs/N) sum[n=2..N] abs(x[n]-x[n-1])',
         'Stored counts per second; x is centered within each window'),
        ('Corrected waveform length',
         'WL/s_v2 = sum[n=2..N] abs(x[n]-x[n-1]) / ((N−1)/fs)',
         'Versioned v2 feature using sample-span duration; not used in the historical result and not silently substituted for WL/s'),
        ('MDF',
         'MDF = min f_m such that sum[k:f_k<=f_m] Pxx[k] >= 0.5 sum[k] Pxx[k]',
         'Median frequency of the Hann periodogram restricted to 20–400 Hz; not specific to fatigue'),
        ('MNF',
         'MNF = sum[k] f_k Pxx[k] / sum[k] Pxx[k]',
         'Mean frequency of the Hann periodogram restricted to 20–400 Hz'),
        ('Power ratio',
         'R = sum[k:20<=f_k<60] Pxx[k] / sum[k:60<=f_k<=400] Pxx[k]',
         'Dimensionless ratio from the unrectified periodogram'),
        ('Wavelet energies',
         'E_j = sum[m] c_j[m]^2 / sum[j in {A3,D3,D2,D1}] sum[m] c_j[m]^2',
         'db4 discrete wavelet transform, level 3, symmetric extension; each relative energy is dimensionless'),
        ('Candidate torque label',
         'T_ref = median(T_1,T_2,T_3); MDC95 = 1.96 × sqrt(2) × SEM; fatigue onset candidate if median(T_(r-2:r)) < T_ref − MDC95 for 2 consecutive evaluations',
         'Candidate only; needs approved torque construct, calibration, matched-angle repetitions, repeatability estimate, synchronization and reviewer approval. Not applied to ZIP data.'),
        ('Kinematic phase label',
         'ω(t) = dθ(t)/dt; y_phase = flexion if ω>δ, extension if ω<−δ, transition/abstain if |ω|≤δ',
         'δ must derive from sensor noise and angle sign convention; requires synchronized, calibrated kinematics. Not applied to ZIP data.'),
        ('Direct fatigue model',
         'p_direct = σ(β_0 + βᵀX); σ(z)=1/(1+e^(−z))',
         'Proposed interpretable baseline. X is the prespecified feature vector; target requires independent approved fatigue labels.'),
        ('Phase cascade',
         'p_cascade = σ(a_0 + aᵀX + η_ext q_ext + η_flex q_flex)',
         'Proposed second stage. q_ext and q_flex are cross-fitted out-of-fold (OOF) phase probabilities during training and model predictions at test time; true test labels are never inputs.'),
        ('Soft-gated phase experts',
         'p_expert = q_ext f_ext(X) + q_flex f_flex(X), with q_ext+q_flex=1',
         'Proposed experts, weighted by predicted phase probabilities; compare on the same participant-held-out folds and fatigue-label support as direct/cascade.'),
        ('Executed burst-half cascade',
         'q(X)=B^(−1) Σ_b p_b(burst_half=1|X), B=100; p_proxy=σ(β_0+βᵀX+γq+δ_R q·RMS+δ_M q·MDF+δ_N q·MNF)',
         'Historical auxiliary experiment: predicts EMG burst halves, not extension/flexion; stage 2 predicts early/late recording proxy, not physiological fatigue.'),
    ]


def export_public_results(frame,public_dir,private_dir):
    public_dir=Path(public_dir);private_dir=Path(private_dir)
    public_dir.mkdir(parents=True,exist_ok=True);private_dir.mkdir(parents=True,exist_ok=True)
    salt_path=private_dir/'public_export_salt.txt'
    if not salt_path.exists():salt_path.write_text(secrets.token_hex(32),encoding='utf-8')
    salt=salt_path.read_text(encoding='utf-8')
    columns=['model','fold_id']+[k for k in METRICS if k in frame]
    exported=frame[columns].copy()
    exported['fold_id']=[hashlib.sha256((salt+str(x)).encode()).hexdigest()[:12] for x in exported.fold_id]
    exported=exported.sort_values(['model','fold_id'])
    path=public_dir/'results.csv';exported.to_csv(path,index=False,float_format='%.12g')
    return path


def render_report_assets(output_root,config):
    out=Path(output_root);ev=out/'evaluation';assets=out/'report_assets';assets.mkdir(parents=True,exist_ok=True)
    receipt=verify_evaluation(out,config)
    results=pd.read_csv(ev/'results.csv');summary=pd.read_csv(ev/'metrics_summary.csv')
    oof=pd.read_parquet(ev/'oof_predictions.parquet');features=pd.read_parquet(out/'features.parquet')
    if oof.duplicated(['model','window_id']).any():raise ValueError('OOF duplicates')
    counts=oof.groupby('model').size()
    if not (counts==receipt['rows']).all():raise ValueError('incomplete OOF')
    main=summary.pivot(index='model',columns='metric',values=['mean','sd'])
    main.to_csv(assets/'table_6_1_models.csv')
    selected=oof.query("model=='nested_selector'")
    pooled=binary_metrics(selected.label,selected.prediction,selected.score)
    write_json(assets/'pooled_counts.json',{'scope':'pooled counts for confusion only; primary uncertainty is across recording groups',**pooled})
    fig,ax=plt.subplots(figsize=(7.2,4.3));m=summary.query("metric=='f1_macro'")
    ax.barh(m.model,m['mean'],xerr=m.sd,alpha=.85,color='#285b79');ax.set_xlim(0,1);ax.set_xlabel('Macro F1, mean ± SD across held-out recording groups');ax.set_title('Binary weak temporal labels')
    fig.tight_layout();fig.savefig(assets/'model_comparison.png',dpi=180);plt.close(fig)
    cm=np.array([[pooled['cm_0_0'],pooled['cm_0_1']],[pooled['cm_1_0'],pooled['cm_1_1']]])
    fig,axes=plt.subplots(1,2,figsize=(8,3.6))
    for ax,data,title in [(axes[0],cm,'Absolute out-of-fold counts'),(axes[1],cm/cm.sum(axis=1,keepdims=True),'Row-normalized proportions')]:
        ax.imshow(data,cmap='Blues',vmin=0)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{data[i,j]:.3f}' if title.startswith('Row') else str(int(data[i,j])),ha='center',va='center')
        ax.set_xticks([0,1],['Early proxy','Late proxy']);ax.set_yticks([0,1],['Early proxy','Late proxy']);ax.set_xlabel('Predicted');ax.set_ylabel('Weak reference');ax.set_title(title,fontsize=10)
    fig.tight_layout();fig.savefig(assets/'confusion.png',dpi=180);plt.close(fig)
    r=results.query("model=='nested_selector'")
    fig,ax=plt.subplots(figsize=(7.2,3.5));ax.scatter(np.arange(len(r))+1,r.f1_macro);ax.axhline(r.f1_macro.mean(),color='#777777',linestyle='--');ax.set_ylim(0,1);ax.set_xlabel('Held-out recording group (private order)');ax.set_ylabel('Macro F1');fig.tight_layout();fig.savefig(assets/'held_out_groups.png',dpi=180);plt.close(fig)
    f=features.query("feature_status=='valid'").copy();f['time_bin']=np.minimum((f.time_fraction*10).astype(int),9)
    per_record=f.groupby(['subject_id','recording_id','time_bin'])[['rms','mdf_hz','mnf_hz']].median().reset_index()
    per_group=per_record.groupby(['subject_id','time_bin'])[['rms','mdf_hz','mnf_hz']].median().reset_index()
    fig,axes=plt.subplots(1,3,figsize=(9,3))
    for ax,name in zip(axes,['rms','mdf_hz','mnf_hz']):
        g=per_group.groupby('time_bin')[name].agg(['mean','std']);x=(g.index.to_numpy()+.5)/10
        ax.plot(x,g['mean']);ax.fill_between(x,g['mean']-g['std'],g['mean']+g['std'],alpha=.2);ax.set_title(name);ax.set_xlabel('Elapsed recording fraction')
    fig.suptitle('Candidate-active windows, group summaries ± SD',fontsize=11);fig.tight_layout();fig.savefig(assets/'trajectories.png',dpi=180);plt.close(fig)
    per_group.to_csv(assets/'feature_trajectories_by_group.csv',index=False)
    bilateral=f.groupby(['subject_id','side','series'])[FEATURE_COLUMNS].median().reset_index()
    bilateral.to_csv(assets/'bilateral_feature_medians.csv',index=False)
    pairs=[]
    for feature in ['rms','mdf_hz','mnf_hz']:
        pivot=bilateral.pivot(index=['subject_id','series'],columns='side',values=feature).dropna()
        if {'left','right'}.issubset(pivot.columns):
            d=pivot.left-pivot.right
            pairs.append({'feature':feature,'paired_group_series':len(d),'mean_left_minus_right':float(d.mean()),'median_left_minus_right':float(d.median()),'interpretation':'descriptive electrical difference, no dominance or fatigue-rate claim'})
    pd.DataFrame(pairs).to_csv(assets/'bilateral_summary.csv',index=False)
    if (out/'ablation_summary.csv').exists():
        a=pd.read_csv(out/'ablation_summary.csv');ref=results.query("model=='logistic_all'").f1_macro
        fig,ax=plt.subplots(figsize=(7.2,3.5));names=['all_features']+a.ablation.tolist();means=[ref.mean()]+a.macro_f1_mean.tolist();sds=[ref.std(ddof=1)]+a.macro_f1_sd.tolist();ax.barh(names,means,xerr=sds,color='#4b8067');ax.set_xlim(0,1);ax.set_xlabel('Logistic model macro F1 ± SD');fig.tight_layout();fig.savefig(assets/'ablation.png',dpi=180);plt.close(fig)
    if (out/'sensitivity_summary.csv').exists():
        s=pd.read_csv(out/'sensitivity_summary.csv');fig,ax=plt.subplots(figsize=(7.5,4.5));v=s.query("status=='evaluated'");ax.errorbar(v.macro_f1_mean,v.analysis,xerr=v.macro_f1_sd,fmt='o',color='#285b79');ax.set_xlim(0,1);ax.set_xlabel('Fixed RMS+MDF logistic macro F1 ± SD');fig.tight_layout();fig.savefig(assets/'sensitivity.png',dpi=180);plt.close(fig)
    # Private illustrative traces. Selection is by fixed QC coverage, not by model accuracy.
    write_json(assets/'table_sources.json',{
      'table_6_1_models.csv':['evaluation/results.csv','evaluation/metrics_summary.csv'],
      'confusion.png':['evaluation/oof_predictions.parquet'],
      'held_out_groups.png':['evaluation/results.csv'],
      'trajectories.png':['features.parquet'],
      'bilateral_summary.csv':['features.parquet'],
      'ablation.png':['ablation_summary.csv','evaluation/results.csv'],
      'sensitivity.png':['sensitivity_summary.csv']})
    tracked=[p for p in assets.iterdir() if p.is_file() and p.name!='assets_manifest.json']
    write_json(assets/'assets_manifest.json',{'config_sha256':config.digest(),'files':[{'name':p.name,'sha256':sha256_file(p)} for p in sorted(tracked)]})
    return assets


def verify_evaluation(output_root,config):
    from .workflow import verify_feature_cache,eligible_features,feature_table_digest
    out=Path(output_root);ev=out/'evaluation'
    if (ev/'RUNNING.json').exists() or not (ev/'evaluation_receipt.json').exists():
        raise ValueError('evaluation is incomplete or running')
    receipt=json.loads((ev/'evaluation_receipt.json').read_text(encoding='utf-8'))
    if receipt.get('status')!='complete' or receipt['config_sha256']!=config.digest():raise ValueError('evaluation/config mismatch')
    verify_feature_cache(out,config)
    features=eligible_features(pd.read_parquet(out/'features.parquet'))
    if receipt.get('feature_table_sha256')!=feature_table_digest(features):raise ValueError('evaluation feature identity mismatch')
    for name,expected in receipt['output_hashes'].items():
        if sha256_file(ev/name)!=expected:raise ValueError('evaluation output checksum mismatch: '+name)
    results=pd.read_csv(ev/'results.csv');oof=pd.read_parquet(ev/'oof_predictions.parquet')
    if set(results.model)!=set(receipt['models']) or set(oof.model)!=set(receipt['models']):raise ValueError('evaluation model set mismatch')
    expected_windows=set(features.window_id)
    for model in receipt['models']:
        r=results[results.model==model];o=oof[oof.model==model]
        if len(r)!=receipt['outer_folds'] or r.fold_id.nunique()!=receipt['outer_folds'] or r.subject_id.nunique()!=receipt['outer_folds']:
            raise ValueError('incomplete or duplicate model/fold results')
        if len(o)!=len(features) or o.window_id.duplicated().any() or set(o.window_id)!=expected_windows:raise ValueError('OOF window set mismatch')
        aligned=o.set_index('window_id').loc[features.window_id]
        if not np.array_equal(aligned.label.to_numpy(),features.label.to_numpy()):raise ValueError('OOF label mismatch')
    return receipt
