import numpy as np
import pandas as pd
from sklearn import metrics


def binary_metrics(y,pred,score):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int);score=np.asarray(score,dtype=float)
    if not len(y) or y.shape!=pred.shape or y.shape!=score.shape or not np.isfinite(score).all():
        raise ValueError('invalid metric arrays')
    if not set(np.unique(y)).issubset({0,1}) or not set(np.unique(pred)).issubset({0,1}):
        raise ValueError('binary vocabulary is [0,1]')
    cm=metrics.confusion_matrix(y,pred,labels=[0,1]);p,r,f,s=metrics.precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    both=len(np.unique(y))==2
    out={'n_test':int(len(y)),'accuracy':float(metrics.accuracy_score(y,pred)),
      'precision_macro':float(p.mean()),'recall_macro':float(r.mean()),'f1_macro':float(f.mean()),
      'balanced_accuracy':float(r.mean()) if both else None,
      'cohen_kappa':float(metrics.cohen_kappa_score(y,pred)) if both else None,
      'specificity_class1':float(cm[0,0]/cm[0].sum()) if cm[0].sum() else None,
      'roc_auc':float(metrics.roc_auc_score(y,score)) if both else None,
      'average_precision':float(metrics.average_precision_score(y,score)) if both else None,
      'pr_auc_trapezoidal':None,'auc_status':'defined' if both else 'undefined_single_class'}
    if both:
        precision,recall,_=metrics.precision_recall_curve(y,score)
        out['pr_auc_trapezoidal']=float(metrics.auc(recall,precision))
    for i in range(2):
        out.update({f'precision_class{i}':float(p[i]),f'recall_class{i}':float(r[i]),f'f1_class{i}':float(f[i]),f'support_class{i}':int(s[i])})
        for j in range(2):out[f'cm_{i}_{j}']=int(cm[i,j])
    return out


def summarize_results(results,seed=20261028,replicates=5000):
    rows=[];rng=np.random.default_rng(seed)
    metric_names=['accuracy','precision_macro','recall_macro','f1_macro','balanced_accuracy','roc_auc','average_precision','pr_auc_trapezoidal','cohen_kappa','specificity_class1','precision_class0','recall_class0','f1_class0','precision_class1','recall_class1','f1_class1']
    for model,g in results.groupby('model',sort=True):
        for metric in metric_names:
            values=pd.to_numeric(g[metric],errors='coerce').dropna().to_numpy()
            if not len(values):continue
            boots=rng.choice(values,size=(replicates,len(values)),replace=True).mean(axis=1)
            rows.append({'model':model,'metric':metric,'mean':float(values.mean()),'sd':float(values.std(ddof=1)) if len(values)>1 else None,
              'ci95_lower':float(np.quantile(boots,.025)),'ci95_upper':float(np.quantile(boots,.975)),
              'folds_evaluable':len(values),'folds_total':len(g),'uncertainty_unit':'held_out_recording_group'})
    return pd.DataFrame(rows)


def paired_differences(results,config):
    rng=np.random.default_rng(config.seed);rows=[]
    for baseline in ['majority','logistic_rms_mdf']:
        selected=results[results.model=='nested_selector'][['fold_id','f1_macro']]
        base=results[results.model==baseline][['fold_id','f1_macro']]
        pair=selected.merge(base,on='fold_id',suffixes=('_selector','_baseline'))
        if not len(pair):continue
        d=(pair.f1_macro_selector-pair.f1_macro_baseline).to_numpy()
        b=rng.choice(d,size=(config.bootstrap_replicates,len(d)),replace=True).mean(axis=1)
        rows.append({'comparison':f'nested_selector_minus_{baseline}','paired_groups':len(d),'mean_difference':float(d.mean()),'ci95_lower':float(np.quantile(b,.025)),'ci95_upper':float(np.quantile(b,.975))})
    return pd.DataFrame(rows)
