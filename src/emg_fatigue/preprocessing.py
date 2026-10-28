import numpy as np
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


def filter_run(x,fs,config):
    x=np.asarray(x,dtype=float)
    if fs<=2*config.band_hz[1]:raise ValueError('sampling below band requirement')
    sos=signal.butter(config.filter_order,config.band_hz,btype='bandpass',fs=fs,output='sos')
    pad=3*(2*len(sos)+1)
    edge=max(pad,int(np.ceil(config.edge_s*fs)))
    if len(x)<=2*edge:raise ValueError('insufficient samples after edge mask')
    y=signal.sosfiltfilt(sos,x-x.mean(),padlen=pad)
    if config.notch:
        for f0,prom in line_prominence(x,fs).items():
            if prom is not None and prom>10:
                b,a=signal.iirnotch(float(f0),30,fs)
                y=signal.filtfilt(b,a,y)
    mask=np.ones(len(x),dtype=bool);mask[:edge]=False;mask[-edge:]=False
    return y,mask


def preprocess(recording,config):
    processed=[];qc=qc_recording(recording,config)
    for i,row in enumerate(qc):
        a,b=row['start'],row['end']
        if row['status']!='valid':continue
        try:y,mask=filter_run(recording.x[a:b],row['fs_hz'],config)
        except ValueError as e:
            row['status']=str(e);continue
        processed.append({'run_id':i,'t':recording.t[a:b],'x':y,'valid':mask,'fs_hz':row['fs_hz'],'start_index':a})
    return {'recording':recording,'runs':processed,'qc':qc}
