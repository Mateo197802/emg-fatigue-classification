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
    receipt=json.loads((ev/'evaluation_receipt.json').read_text(encoding='utf-8'))
    if receipt['config_sha256']!=config.digest():raise ValueError('evaluation/config mismatch')
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
