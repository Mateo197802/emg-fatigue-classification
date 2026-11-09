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


def paired_metric_contrast(
    results,
    *,
    candidate,
    baseline,
    metric="accuracy",
    group_column="fold_id",
    bootstrap_replicates=5000,
    seed=20261028,
):
    """Compare one metric with a paired bootstrap over matched held-out groups."""
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    required = {"model", group_column, metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"paired results are missing columns: {sorted(missing)}")
    if candidate == baseline:
        raise ValueError("candidate and baseline must be different models")
    selected = results[results.model.isin([candidate, baseline])][["model", group_column, metric]].copy()
    if selected[["model", group_column]].isna().any().any():
        raise ValueError("model and group IDs must be present")
    if selected.duplicated(["model", group_column]).any():
        raise ValueError("expected one metric row per model and group")
    selected[metric] = pd.to_numeric(selected[metric], errors="raise")
    if not np.isfinite(selected[metric].to_numpy(dtype=float)).all():
        raise ValueError("paired metric values must be finite")
    pivot = selected.pivot(index=group_column, columns="model", values=metric)
    if candidate not in pivot or baseline not in pivot or pivot[[candidate, baseline]].isna().any().any():
        raise ValueError("candidate and baseline must have identical group support")
    differences = (pivot[candidate] - pivot[baseline]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(
        differences,
        size=(bootstrap_replicates, len(differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "mean_difference": float(differences.mean()),
        "ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap, 0.975)),
        "n_paired_groups": int(len(differences)),
        "groups_improved": int((differences > 0).sum()),
        "groups_worse": int((differences < 0).sum()),
        "groups_equal": int((differences == 0).sum()),
        "uncertainty_unit": group_column,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }


def _validate_oof_predictions(predictions):
    required = {"architecture", "group_id", "row_index", "y_true", "y_pred", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("OOF predictions must not be empty")
    if predictions[["architecture", "group_id", "row_index"]].isna().any().any():
        raise ValueError("architecture, group_id, and row_index must be present")
    if predictions.duplicated(["architecture", "row_index"]).any():
        raise ValueError("each architecture must have one OOF row per row_index")
    for column in ("y_true", "y_pred"):
        values = set(pd.to_numeric(predictions[column], errors="raise").unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"{column} must contain only binary labels [0, 1]")
    scores = pd.to_numeric(predictions["score"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("OOF scores must be finite")


def _group_metric_table(predictions):
    _validate_oof_predictions(predictions)
    rows = []
    for (architecture, group_id), group in predictions.groupby(
        ["architecture", "group_id"], sort=False, dropna=False
    ):
        truth = group["y_true"].to_numpy(dtype=int)
        predicted = group["y_pred"].to_numpy(dtype=int)
        score = group["score"].to_numpy(dtype=float)
        rows.append({
            "architecture": architecture,
            "group_id": group_id,
            **binary_metrics(truth, predicted, score),
        })
    return pd.DataFrame(rows)


def summarize_grouped_predictions(
    predictions,
    *,
    metrics_to_summarize=("accuracy", "f1_macro", "balanced_accuracy", "precision_macro", "recall_macro"),
    bootstrap_replicates=5000,
    seed=20261109,
):
    """Summarize OOF metrics with equal group weighting and group bootstrap CIs."""
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    group_metrics = _group_metric_table(predictions)
    rng = np.random.default_rng(seed)
    rows = []
    for architecture, model_groups in group_metrics.groupby("architecture", sort=True):
        for metric in metrics_to_summarize:
            if metric not in model_groups.columns:
                raise ValueError(f"unsupported summary metric: {metric}")
            valid = model_groups[metric].dropna().to_numpy(dtype=float)
            if not len(valid):
                continue
            bootstrap_means = rng.choice(
                valid, size=(bootstrap_replicates, len(valid)), replace=True
            ).mean(axis=1)
            rows.append({
                "architecture": architecture,
                "metric": metric,
                "mean": float(valid.mean()),
                "sd": float(valid.std(ddof=1)) if len(valid) > 1 else None,
                "ci95_lower": float(np.quantile(bootstrap_means, 0.025)),
                "ci95_upper": float(np.quantile(bootstrap_means, 0.975)),
                "n_groups": int(len(valid)),
                "pooled_support": int(model_groups["n_test"].sum()),
                "uncertainty_unit": "group_id",
            })
    return pd.DataFrame(rows)


def paired_group_differences(
    predictions,
    *,
    baseline,
    candidates,
    metrics=("accuracy", "f1_macro"),
    bootstrap_replicates=5000,
    seed=20261109,
):
    """Bootstrap paired candidate-minus-baseline differences across groups."""
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    group_metrics = _group_metric_table(predictions)
    models = set(group_metrics["architecture"])
    if baseline not in models:
        raise ValueError(f"baseline architecture is not present: {baseline}")
    candidate_names = [candidates] if isinstance(candidates, str) else list(candidates)
    if not candidate_names or any(name not in models for name in candidate_names):
        raise ValueError("all candidate architectures must be present")
    baseline_rows = group_metrics[group_metrics["architecture"] == baseline].set_index("group_id")
    rng = np.random.default_rng(seed)
    rows = []
    for candidate in candidate_names:
        candidate_rows = group_metrics[group_metrics["architecture"] == candidate].set_index("group_id")
        if set(baseline_rows.index) != set(candidate_rows.index):
            raise ValueError("paired architectures must have identical group support")
        for metric in metrics:
            if metric not in group_metrics.columns:
                raise ValueError(f"unsupported paired metric: {metric}")
            pair = pd.DataFrame({
                "baseline": baseline_rows[metric],
                "candidate": candidate_rows[metric],
            }).dropna()
            if pair.empty:
                continue
            differences = (pair["candidate"] - pair["baseline"]).to_numpy(dtype=float)
            bootstrap_means = rng.choice(
                differences,
                size=(bootstrap_replicates, len(differences)),
                replace=True,
            ).mean(axis=1)
            rows.append({
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "mean_difference": float(differences.mean()),
                "ci95_lower": float(np.quantile(bootstrap_means, 0.025)),
                "ci95_upper": float(np.quantile(bootstrap_means, 0.975)),
                "n_paired_groups": int(len(differences)),
                "uncertainty_unit": "group_id",
            })
    return pd.DataFrame(rows)


def confusion_matrix_table(predictions):
    """Return pooled OOF confusion counts and row-normalized rates by model."""
    _validate_oof_predictions(predictions)
    rows = []
    for architecture, model_rows in predictions.groupby("architecture", sort=True):
        truth = model_rows["y_true"].to_numpy(dtype=int)
        predicted = model_rows["y_pred"].to_numpy(dtype=int)
        matrix = metrics.confusion_matrix(truth, predicted, labels=[0, 1])
        for reference_class in (0, 1):
            support = int(matrix[reference_class].sum())
            for predicted_class in (0, 1):
                count = int(matrix[reference_class, predicted_class])
                rows.append({
                    "architecture": architecture,
                    "reference_class": reference_class,
                    "predicted_class": predicted_class,
                    "count": count,
                    "reference_support": support,
                    "row_fraction": float(count / support) if support else None,
                    "n_groups": int(model_rows["group_id"].nunique()),
                    "aggregation": "pooled_oof_windows",
                })
    return pd.DataFrame(rows)
