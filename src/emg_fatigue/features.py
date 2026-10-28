import numpy as np
import pandas as pd
import pywt
from scipy import signal

AMPLITUDE=['rms','mav','waveform_length_per_s']
SPECTRAL=['mdf_hz','mnf_hz','power_ratio_low_high']
WAVELET=['wavelet_A3','wavelet_D3','wavelet_D2','wavelet_D1']
FEATURE_COLUMNS=AMPLITUDE+SPECTRAL+WAVELET


def window_features(x,fs):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or len(x)<56:raise ValueError('invalid or short feature window')
    x=x-x.mean();energy=float(np.mean(x*x))
    if energy<=1e-20:raise ValueError('zero window energy')
    f,p=signal.periodogram(x,fs,window='hann',detrend=False,scaling='density')
    band=(f>=20)&(f<=400);freq=f[band];power=p[band]
    total=float(power.sum());high=float(p[(f>=60)&(f<=400)].sum())
    if total<=1e-20 or high<=1e-20:raise ValueError('insufficient spectral energy')
    coeffs=pywt.wavedec(x,'db4',mode='symmetric',level=3)
    e=np.array([np.sum(c*c) for c in coeffs]);e=e/e.sum()
    result={'rms':np.sqrt(energy),'mav':float(np.mean(np.abs(x))),
      'waveform_length_per_s':float(np.sum(np.abs(np.diff(x)))/(len(x)/fs)),
      'mdf_hz':float(freq[min(np.searchsorted(np.cumsum(power),.5*total),len(freq)-1)]),
      'mnf_hz':float(np.sum(freq*power)/total),
      'power_ratio_low_high':float(p[(f>=20)&(f<60)].sum()/high)}
    result.update(dict(zip(WAVELET,e.tolist())))
    if not np.isfinite(list(result.values())).all():raise ValueError('nonfinite features')
    return {k:float(v) for k,v in result.items()}


def extract_features(processed,windows,config=None):
    runs={r['run_id']:r for r in processed['runs']};rows=[]
    for w in windows:
        row=dict(w)
        if w['activity_eligible']:
            r=runs[w['run_id']]
            try:
                row.update(window_features(r['x'][w['start']:w['end']],r['fs_hz']));row['feature_status']='valid'
            except ValueError as e:row['feature_status']=str(e);row['label']=None
        else:row['feature_status']='not_eligible'
        rows.append(row)
    return pd.DataFrame(rows)
