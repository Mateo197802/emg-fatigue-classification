"""Verify completed private runs and export aggregate, target-specific evidence."""
from pathlib import Path
import argparse,json,shutil,importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='emg-phase-evidence-v1'
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix
from emg_fatigue.contracts import read_config,sha256_file,write_json
from emg_fatigue.reporting import verify_evaluation,export_public_results
from emg_fatigue.evaluation import binary_metrics

PRIMARY=['majority','logistic_rms_mdf','logistic_all','svm_rbf','random_forest','nested_selector']
EXPERIMENT=['baseline_10','phase_score','phase_conditioned','phase_only','random_conditioned']
NAMES={'majority':'Training majority','logistic_rms_mdf':'Logistic RMS + MDF','logistic_all':'Logistic 10 features','svm_rbf':'SVM RBF','random_forest':'Random forest','nested_selector':'Nested selector','baseline_10':'Direct logistic 10','phase_score':'10 features + phase score','phase_conditioned':'Phase conditioned','phase_only':'Phase score only','random_conditioned':'Random label control','auxiliary_phase_head':'Auxiliary burst half head'}


def validate_predictions(frame,models,windows):
    if set(frame.model)!=set(models):raise ValueError('OOF model set mismatch')
    if frame.duplicated(['model','window_id']).any():raise ValueError('duplicate OOF window')
    reference=None
    for model in models:
        f=frame[frame.model==model]
        if len(f)!=windows:raise ValueError('OOF support mismatch')
        if not set(f.label).issubset({0,1}) or not set(f.prediction).issubset({0,1}):raise ValueError('nonbinary OOF values')
        labels=f.set_index('window_id').label.sort_index()
        if reference is None:reference=labels
        elif not reference.equals(labels):raise ValueError('OOF window or reference label mismatch')


def confusion_rows(frame,target):
    rows=[]
    for model,g in frame.groupby('model',sort=False):
        cm=confusion_matrix(g.label,g.prediction,labels=[0,1]);support=cm.sum(axis=1)
        norm=np.divide(cm,support[:,None],out=np.full((2,2),np.nan),where=support[:,None]>0)
        r={'target':target,'model':model,'n':int(cm.sum()),'support_0':int(support[0]),'support_1':int(support[1])}
        for i in range(2):
            for j in range(2):r[f'cm_{i}_{j}']=int(cm[i,j]);r[f'norm_{i}_{j}']=float(norm[i,j])
        rows.append(r)
    return pd.DataFrame(rows)


def public_fold_export(frame,public,private):
    return export_public_results(frame,public,private)


def aggregate_hashes(public):
    return {p.name:sha256_file(p) for p in Path(public).iterdir() if p.is_file() and p.name!='evidence_receipt.json'}


def normalize_public_csv(public):
    for p in Path(public).glob('*.csv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))


def resolve_run(experiment):
    root=Path(experiment)
    if (root/'completion_receipt.json').is_file():return root
    if (root/'run/completion_receipt.json').is_file():return root/'run'
    raise ValueError('no completed experiment found at run root or wrapper/run')


def reconstruct_auxiliary(run):
    """Rebuild only stage-1 diagnostics from frozen features; no stage-2 retraining."""
    run=Path(run);script=Path(__file__).with_name('phase_spike.py')
    spec=importlib.util.spec_from_file_location('original_phase_spike',script)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    frame=pd.read_parquet(run/'phase_features.parquet');out=run/'phase_diagnostics';out.mkdir(exist_ok=True)
    heads=original.PhaseScores(frame,out);groups=frame.subject_id.to_numpy();rows=[]
    for group in sorted(set(groups)):
        indices=np.flatnonzero(groups==group)
        score=heads.get(sorted(set(groups)-{group}),indices,'phase','diagnostic/'+str(group))
        part=frame.iloc[indices][['window_id','subject_id','recording_id','phase_proxy','label','time_fraction']].copy()
        part['phase_score']=score;part['phase_prediction']=(score>=.5).astype(int);rows.append(part)
    path=out/'phase_oof_predictions.parquet';pd.concat(rows,ignore_index=True).to_parquet(path,index=False);heads.save()
    write_json(out/'reconstruction_receipt.json',{'phase_features_sha256':sha256_file(run/'phase_features.parquet'),'original_script_sha256':sha256_file(script),'predictions_sha256':sha256_file(path),'source':'group-held-out reconstruction of fixed original stage-1 head; no new fatigue-model fit'})
    return path


def draw_comparison(run,path):
    summary=pd.read_csv(Path(run)/'metrics_summary.csv').query("metric=='f1_macro'").set_index('model')
    results=pd.read_csv(Path(run)/'results.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    axes[0].barh([NAMES[m] for m in EXPERIMENT],[summary.loc[m,'mean'] for m in EXPERIMENT],xerr=[summary.loc[m,'sd'] for m in EXPERIMENT],color='#426b83')
    axes[0].invert_yaxis();axes[0].set_xlim(0,1);axes[0].set_xlabel('Macro-F1 mean ± SD across groups')
    base=results.query("model=='baseline_10'").set_index('fold_id').f1_macro
    for model,color in [('phase_conditioned','#bc704a'),('random_conditioned','#6d8979')]:
        d=results[results.model==model].set_index('fold_id').f1_macro-base
        axes[1].plot(np.arange(1,len(d)+1),d,'o-',label=NAMES[model],color=color)
    axes[1].axhline(0,color='black',ls='--',lw=1);axes[1].set_xlabel('Held-out group (opaque order)');axes[1].set_ylabel('Paired macro-F1 difference');axes[1].legend(fontsize=8)
    fig.suptitle('Exploratory phase experiment — weak early/late target');fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def draw_confusions(table,models,path,title,normalized=False,auxiliary=False):
    ncols=1 if len(models)==1 else 3;nrows=int(np.ceil(len(models)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(12,6.4) if len(models)>1 else (5.5,4.6),squeeze=False)
    for ax,model in zip(axes.ravel(),models):
        r=table[table.model==model].iloc[0];prefix='norm' if normalized else 'cm'
        data=np.array([[r[f'{prefix}_{i}_{j}'] for j in range(2)] for i in range(2)])
        ax.imshow(data,cmap='Blues',vmin=0,vmax=1 if normalized else max(1,r[['cm_0_0','cm_0_1','cm_1_0','cm_1_1']].max()))
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{data[i,j]:.3f}' if normalized else str(int(data[i,j])),ha='center',va='center',fontsize=19,color='white' if data[i,j]>(.55 if normalized else .55*np.nanmax(data)) else '#172c3d')
        ax.set_xticks([0,1],['0','1'],fontsize=12);ax.set_yticks([0,1],['0','1'],fontsize=12)
        ax.set_title(NAMES[model],fontsize=12,pad=5);ax.set_xlabel('Predicted',fontsize=10);ax.set_ylabel('Reference',fontsize=10)
    for ax in axes.ravel()[len(models):]:ax.set_visible(False)
    fig.suptitle(title,fontsize=15)
    legend='0 = first burst half; 1 = second burst half.\nNot verified extension/flexion.' if auxiliary else '0 = early proxy; 1 = late proxy. Not independent physiological fatigue labels.'
    fig.text(.5,.025 if auxiliary else .015,legend,ha='center',fontsize=10 if auxiliary else 11)
    fig.tight_layout(rect=[0,.12 if auxiliary else .055,1,.94]);fig.savefig(path,dpi=180);plt.close(fig)


def architecture(path):
    fig,ax=plt.subplots(figsize=(12,6.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    def box(x,y,w,h,s,color='#eef3f6',size=12):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor=color,edgecolor='#315c78',linewidth=1.2))
        ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=size,color='#17334a')
    def arrow(a,b,label='',dashed=False):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5,'color':'#315c78','linestyle':'--' if dashed else '-'})
        if label:ax.text((a[0]+b[0])/2+.008,(a[1]+b[1])/2+.018,label,fontsize=10,color='#315c78',ha='left')
    box(.02,.55,.16,.15,'Stored EMG\nTime + counts')
    box(.23,.55,.22,.15,'QC / filtering\nActivity / windows')
    box(.51,.55,.19,.15,'X: 10 EMG\nfeatures')
    box(.75,.78,.22,.14,'Stage 1: random forest\n100 trees / depth 3',size=11)
    box(.75,.36,.22,.18,'Stage 2: logistic\nX + p + p×RMS/MDF/MNF\nC chosen within training',size=10)
    box(.75,.08,.22,.15,'Early / late proxy\nUncalibrated score')
    box(.23,.81,.28,.14,'Training reference only\nFirst / second burst half\n0.5–8 s candidate activity',size=10)
    arrow((.18,.625),(.23,.625));arrow((.45,.625),(.51,.625))
    arrow((.70,.66),(.75,.82),'X');arrow((.70,.57),(.75,.50),'X')
    arrow((.86,.78),(.86,.54),'p = second-half score')
    arrow((.86,.36),(.86,.23));arrow((.51,.88),(.75,.88),'Weak phase supervision',True)
    ax.text(.03,.38,'Both stages respect the outer held-out group.\nStage 2 training uses group-out-of-fold stage 1 scores.\nCross-fitting is repeated inside hyperparameter selection.',fontsize=12,va='top',linespacing=1.6,color='#17334a')
    ax.text(.03,.13,'No verified extension/flexion labels.\nNo force / Borg / MVC fatigue reference.\nThis is an offline exploratory experiment.',fontsize=12,va='top',linespacing=1.6,color='#8a4b30')
    fig.suptitle('Two-stage phase experiment: what was actually trained',fontsize=16,y=.99)
    fig.tight_layout();fig.savefig(path,dpi=180);fig.savefig(Path(path).with_suffix('.svg'),metadata={'Date':None});plt.close(fig)


def model_relationships(path):
    fig,ax=plt.subplots(figsize=(12,6.4));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    def block(x,y,w,h,text,color='#eef3f6',size=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#315c78'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.6,color='#17334a')
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5,'color':'#315c78'})
    block(.01,.42,.19,.19,'Same preprocessing\nSame 10 features\nSame held-out groups')
    block(.28,.57,.31,.34,'PRIMARY FAMILIES\nMajority / RMS+MDF logistic\n10-feature logistic / SVM RBF\nRandom forest')
    block(.69,.72,.29,.18,'Nested selector\nChooses in inner folds\nNot a voting ensemble')
    block(.69,.48,.29,.16,'Final private RF artifact\nSeparate full-cohort selection\nNo new OOF estimate')
    block(.28,.05,.31,.35,'SUPPLEMENTAL VARIANTS\nDirect logistic 10 = primary logistic_all\n+ phase score / phase interactions\nPhase score only / random control',size=10)
    block(.69,.12,.29,.24,'AUXILIARY PHASE HEAD\nRF learns EMG burst halves\nScore supplies supplemental variants\nNot extension / flexion ground truth',color='#f8eee7',size=10)
    arrow((.20,.57),(.28,.70));arrow((.20,.46),(.28,.25));arrow((.59,.79),(.69,.81));arrow((.83,.72),(.83,.64));arrow((.69,.24),(.59,.24))
    ax.text(.02,.98,'How the original procedures and the phase experiment relate',fontsize=16,va='top',color='#17334a')
    ax.text(.02,.01,'Fatigue-side evaluation target: weak early/late proxy. Auxiliary target: local burst half.',fontsize=11,color='#8a4b30')
    fig.tight_layout();fig.savefig(path,dpi=180);fig.savefig(Path(path).with_suffix('.svg'),metadata={'Date':None});plt.close(fig)


def build(primary,experiment,out,public,figures,config):
    primary=Path(primary);experiment=Path(experiment);out=Path(out);public=Path(public);figures=Path(figures)
    for p in [out,public,figures]:p.mkdir(parents=True,exist_ok=True)
    verify_evaluation(primary,config)
    run=resolve_run(experiment);receipt=json.loads((run/'completion_receipt.json').read_text())
    if receipt['status']!='complete' or (run/'RUNNING.json').exists():raise ValueError('phase run incomplete')
    for name,expected in receipt['output_hashes'].items():
        if sha256_file(run/name)!=expected:raise ValueError('phase output checksum mismatch: '+name)
    if sha256_file(Path(__file__).with_name('phase_spike.py'))!=receipt['script_sha256']:raise ValueError('phase script identity mismatch')
    if sha256_file(Path(__file__).with_name('PROTOCOLO_ANTES_DE_RESULTADOS.md'))!=receipt['protocol_sha256']:raise ValueError('phase protocol identity mismatch')
    p=pd.read_parquet(primary/'evaluation/oof_predictions.parquet');s=pd.read_parquet(run/'oof_predictions.parquet')
    validate_predictions(p,PRIMARY,5509);validate_predictions(s,EXPERIMENT,5509)
    match=s.query("model=='baseline_10'").merge(p.query("model=='logistic_all'"),on='window_id',suffixes=('_s','_p'),validate='one_to_one')
    if len(match)!=5509 or not np.array_equal(match.label_s,match.label_p) or not np.array_equal(match.prediction_s,match.prediction_p):raise ValueError('baseline mismatch')
    phase=pd.read_parquet(reconstruct_auxiliary(run))
    auxiliary_metrics=pd.read_csv(run/'auxiliary_phase_metrics.csv')
    for group,g in phase[phase.phase_proxy.notna()].groupby('subject_id'):
        measured=binary_metrics(g.phase_proxy.to_numpy().astype(int),g.phase_prediction.to_numpy(),g.phase_score.to_numpy())
        expected=auxiliary_metrics[auxiliary_metrics.subject_id==group]
        if len(expected)!=1:raise ValueError('auxiliary group support mismatch')
        for name in ['n_test','accuracy','f1_macro','roc_auc','support_class0','support_class1']:
            if abs(measured[name]-float(expected.iloc[0][name]))>1e-12:raise ValueError('auxiliary prediction/metric mismatch')
    # The phase diagnostics also carry the fatigue label: drop it before renaming.
    aux=phase[phase.phase_proxy.notna()][['window_id','phase_proxy','phase_prediction','phase_score']].rename(columns={'phase_proxy':'label','phase_prediction':'prediction','phase_score':'score'})
    aux['model']='auxiliary_phase_head';aux['label']=aux.label.astype(int)
    validate_predictions(aux,['auxiliary_phase_head'],3337)
    conf=pd.concat([confusion_rows(p,'early_late_proxy'),confusion_rows(s,'early_late_proxy'),confusion_rows(aux,'burst_half_proxy')],ignore_index=True)
    conf.to_csv(out/'confusion_matrices.csv',index=False);conf.to_csv(public/'confusion_matrices.csv',index=False)
    for name,data,models,target in [('primary',conf,PRIMARY,'Primary procedures'),('phase',conf,EXPERIMENT,'Supplemental procedures')]:
        for normalized in [False,True]:
            file=f'{name}_confusions_'+('normalized' if normalized else 'counts')+'.png'
            draw_confusions(data,models,figures/file,target+' — '+('row proportions' if normalized else 'pooled OOF counts'),normalized)
    for normalized in [False,True]:
        file='auxiliary_phase_confusion_'+('normalized' if normalized else 'counts')+'.png'
        draw_confusions(conf,['auxiliary_phase_head'],figures/file,'Auxiliary burst-half reference — '+('row proportions' if normalized else 'OOF counts'),normalized,True)
    architecture(figures/'phase_architecture.png');model_relationships(figures/'model_relationships.png')
    draw_comparison(run,figures/'phase_comparison.png')
    for name in ['metrics_summary.csv','paired_differences.csv']:shutil.copy2(run/name,public/name)
    private=out/'opaque_mapping';private.mkdir(exist_ok=True)
    shutil.copy2(primary/'public_export/public_export_salt.txt',private/'public_export_salt.txt')
    public_fold_export(pd.read_csv(run/'results.csv'),public,private)
    auxiliary_metrics['model']='auxiliary_phase_head'
    public_fold_export(auxiliary_metrics,out/'auxiliary_export',private)
    shutil.copy2(out/'auxiliary_export/results.csv',public/'auxiliary_phase_fold_metrics.csv')
    registry=[]
    for model in PRIMARY+EXPERIMENT+['auxiliary_phase_head']:
        registry.append({'model':model,'display_name':NAMES[model],'role':'primary_procedure' if model in PRIMARY else 'auxiliary_head' if model=='auxiliary_phase_head' else 'supplemental_variant','target':'burst_half_proxy' if model=='auxiliary_phase_head' else 'early_late_proxy','oof_support':3337 if model=='auxiliary_phase_head' else 5509,'reference_equivalence':'same predictions as primary logistic_all' if model=='baseline_10' else '','deployed_primary_artifact':False})
    registry.append({'model':'final_primary_rf_artifact','display_name':'Serialized final primary RF','role':'full_cohort_fit_not_OOF_estimator','target':'early_late_proxy','oof_support':0,'reference_equivalence':'full-cohort inner selection; no additional OOF estimate','deployed_primary_artifact':True})
    pd.DataFrame(registry).to_csv(public/'model_registry.csv',index=False)
    normalize_public_csv(public)
    # Bundle the original script verbatim separately from this new exporter.
    evidence={'status':'verified','primary_procedures':6,'supplemental_variants':5,'auxiliary_heads':1,'early_late_windows_per_procedure':5509,'auxiliary_reference_windows':3337,'primary_oof_rows':len(p),'supplemental_oof_rows':len(s),'original_phase_script_sha256':receipt['script_sha256'],'protocol_sha256':receipt['protocol_sha256'],'baseline_matches_primary':True,'auxiliary_reconstruction_matches_original_metrics':True,'primary_target':'weak early/late recording position','auxiliary_target':'first/second EMG activity burst half; not verified extension','deployment':'primary model unchanged; exploratory cascade is not promoted','aggregate_hashes':aggregate_hashes(public)}
    write_json(public/'evidence_receipt.json',evidence);write_json(out/'evidence_receipt.json',evidence)
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--primary',required=True);parser.add_argument('--experiment',required=True);parser.add_argument('--private-out',required=True);parser.add_argument('--public-dir',required=True);parser.add_argument('--figures-dir',required=True);parser.add_argument('--config',default='configs/study.yaml');a=parser.parse_args()
    print(json.dumps(build(a.primary,a.experiment,a.private_out,a.public_dir,a.figures_dir,read_config(a.config)),indent=2))
