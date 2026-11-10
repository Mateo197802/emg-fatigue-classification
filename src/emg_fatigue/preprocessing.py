import numpy as np
import hashlib
import json
from scipy import signal
from .qc import qc_recording


def line_prominence(x,fs):
    n=int(round(2*fs));scores={50.:[],60.:[]}
    for a in range(0,len(x)-n+1,n):
        f,p=signal.periodogram(x[a:a+n],fs,window='hann',detrend='constant')
        for f0 in scores:
            core=(f>=f0-1)&(f<f0+1)
            near=((f>=f0-5)&(f<f0-2))|((f>=f0+2)&(f<f0+5))
            den=float(p[near].mean())
            if den>0 and p[core].mean()>0:scores[f0].append(float(10*np.log10(p[core].mean()/den)))
    return {str(int(k)):float(np.median(v)) if v else None for k,v in scores.items()}


def filter_run(x,fs,config,*,return_design=False):
    try:
        x=np.asarray(x,dtype=float)
        fs=float(fs)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('signal and sampling rate must be numeric') from error
    if x.ndim!=1:raise ValueError('filter input must be one-dimensional')
    if not np.isfinite(x).all():raise ValueError('filter input must contain only finite samples')
    if not np.isfinite(fs) or fs<=0:raise ValueError('sampling rate must be finite and positive')
    if fs<=2*config.band_hz[1]:raise ValueError('sampling rate must exceed the Nyquist limit for the configured band')
    sos=signal.butter(config.filter_order,config.band_hz,btype='bandpass',fs=fs,output='sos')
    pad=3*(2*len(sos)+1)
    edge=max(pad,int(np.ceil(config.edge_s*fs)))
    if len(x)<=2*edge:raise ValueError('insufficient samples after edge mask')
    y=signal.sosfiltfilt(sos,x-x.mean(),padlen=pad)
    prominence_scores=line_prominence(x,fs) if config.notch else {}
    applied_notches=[]
    if config.notch:
        for f0,prom in prominence_scores.items():
            if prom is not None and prom>10:
                center=float(f0);quality_factor=30.
                b,a=signal.iirnotch(center,quality_factor,fs)
                y=signal.filtfilt(b,a,y)
                applied_notches.append({'center_hz':center,'quality_factor':quality_factor,
                    'bandwidth_hz':center/quality_factor,'prominence_db':prom,
                    'b':b.tolist(),'a':a.tolist()})
    mask=np.ones(len(x),dtype=bool);mask[:edge]=False;mask[-edge:]=False
    if not return_design:return y,mask
    design={'schema':'emg_filter_run_design_v1','sampling_rate_hz':fs,
      'bandpass':{'design':'digital Butterworth band-pass in second-order sections',
        'prototype_order':int(config.filter_order),'digital_order':int(2*config.filter_order),
        'cutoff_hz':[float(v) for v in config.band_hz],'nyquist_hz':fs/2,
        'application':'scipy.signal.sosfiltfilt','zero_phase_offline':True,
        'sos_pad_samples':int(pad),'edge_mask_samples_each_end':int(edge),
        'edge_mask_seconds_each_end':float(edge/fs),'sos_coefficients':sos.tolist()},
      'notch':{'enabled':bool(config.notch),'trigger':'measured line prominence > 10 dB',
        'prominence_db_by_candidate_hz':prominence_scores,'applied':applied_notches}}
    design['design_sha256']=hashlib.sha256(json.dumps(design,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return y,mask,design


def preprocess(recording,config):
    processed=[];qc=qc_recording(recording,config)
    for i,row in enumerate(qc):
        a,b=row['start'],row['end']
        if row['status']!='valid':continue
        try:y,mask,design=filter_run(recording.x[a:b],row['fs_hz'],config,return_design=True)
        except ValueError as e:
            row['status']=str(e);continue
        processed.append({'run_id':i,'t':recording.t[a:b],'x':y,'valid':mask,'fs_hz':row['fs_hz'],'start_index':a,'filter_design':design})
    return {'recording':recording,'runs':processed,'qc':qc}
