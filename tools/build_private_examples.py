"""Create QC/split illustrations without selecting on model performance."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emg_fatigue.io import load_recordings
from emg_fatigue.contracts import read_config
from emg_fatigue.preprocessing import preprocess

p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output-root',required=True);a=p.parse_args()
out=Path(a.output_root)/'report_assets';out.mkdir(parents=True,exist_ok=True)
records=[r for r in load_recordings(a.archive) if r.series!='R0'];config=read_config('configs/study.yaml')
good=next(r for r in records if len(preprocess(r,config)['runs']))
processed=preprocess(good,config)['runs'][0];valid=np.flatnonzero(processed['valid']);start=valid[0];end=min(start+1000,valid[-1])
bad=next(r for r in records if np.any(np.diff(r.t)>.001500001));i=int(np.argmax(np.diff(bad.t)));lo=max(0,i-30);hi=min(len(bad.t),i+31)
fig,axes=plt.subplots(2,1,figsize=(7,4.3));axes[0].plot(processed['t'][start:end],processed['x'][start:end],linewidth=.7);axes[0].set_title('Accepted continuous filtered segment',fontsize=11)
axes[1].scatter(bad.t[lo:hi],bad.x[lo:hi],s=5);axes[1].axvspan(bad.t[i],bad.t[i+1],color='#b84d4d',alpha=.2);axes[1].set_title('Timestamp gap: crossing windows excluded',fontsize=11)
for ax in axes:ax.set_xlabel('Original time (s)');ax.set_ylabel('Stored counts')
fig.tight_layout();fig.savefig(out/'qc_examples.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,2.6));ax.axis('off')
items=[(.02,.55,.25,.3,'Outer training groups'),(.38,.55,.25,.3,'Grouped inner tuning'),(.73,.55,.25,.3,'Fit selected pipeline'),(.73,.05,.25,.3,'Held-out group score')]
for x,y,w,h,text in items:
 ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='#edf2f6',edgecolor='#285b79'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9,wrap=True)
for start,end in [((.27,.7),(.38,.7)),((.63,.7),(.73,.7)),((.855,.55),(.855,.35))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->'})
ax.text(.34,.18,'No held-out data fit scaler, model or search',ha='center',fontsize=10)
fig.tight_layout();fig.savefig(out/'split_scheme.png',dpi=180);plt.close(fig)
