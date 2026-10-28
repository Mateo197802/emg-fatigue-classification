import numpy as np


def otsu_threshold(values):
    v=np.asarray(values,dtype=float)
    if not len(v) or not np.isfinite(v).all():raise ValueError('finite activity envelope required')
    if np.ptp(v)==0:return float(v[0])
    counts,edges=np.histogram(v,bins=128);centers=(edges[:-1]+edges[1:])/2
    w=np.cumsum(counts);s=np.cumsum(counts*centers);total=w[-1]
    den=w*(total-w);between=np.divide((s[-1]*w-s*total)**2,den,out=np.zeros_like(s),where=den>0)
    return float(centers[int(np.argmax(between))])


def detect_activity(processed,config):
    envelope=[];runs=processed['runs']
    for r in runs:
        n=max(2,int(round(r['fs_hz']*config.activity_frame_s)))
        step=max(1,int(round(r['fs_hz']*config.activity_hop_s)))
        frames=[]
        for a in range(0,len(r['x'])-n+1,step):
            if r['valid'][a:a+n].all():
                rms=float(np.sqrt(np.mean(r['x'][a:a+n]**2)))
                frames.append((a,a+n,rms));envelope.append(rms)
        r['activity_frames']=frames
    threshold=otsu_threshold(envelope) if envelope else None
    for r in runs:
        active=np.zeros(len(r['x']),dtype=bool)
        if threshold is not None:
            for a,b,rms in r['activity_frames']:
                if rms>threshold:active[a:b]=True
        r['active']=active&r['valid'];r['activity_threshold']=threshold
    return processed
