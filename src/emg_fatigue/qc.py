import numpy as np
from .contracts import StudyConfig


def contiguous_runs(times,config=None):
    c=config or StudyConfig();t=np.asarray(times,dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('timestamps must be finite and strictly increasing')
    dt=np.diff(t);median=float(np.median(dt));limit=max(c.gap_min_s,c.gap_multiplier*median)+1e-9
    edges=np.r_[0,np.flatnonzero(dt>limit)+1,len(t)]
    rows=[]
    for a,b in zip(edges[:-1],edges[1:]):
        d=np.diff(t[a:b]);fs=1/float(np.median(d)) if len(d) else None
        jitter=float((np.percentile(d,95)-np.percentile(d,5))/np.median(d)) if len(d) else None
        status='valid'
        if b-a<8:status='insufficient_samples'
        elif jitter>c.jitter_max:status='jitter_exceeds_limit'
        elif fs<=2*c.band_hz[1]:status='sampling_below_band_requirement'
        rows.append({'start':int(a),'end':int(b),'fs_hz':fs,'jitter_relative':jitter,'status':status})
    return rows


def qc_recording(recording,config):
    rows=contiguous_runs(recording.t,config)
    for row in rows:
        x=recording.x[row['start']:row['end']]
        t=recording.t[row['start']:row['end']]
        centered_t=t-t.mean()
        centered_x=x-x.mean()
        time_energy=float(np.dot(centered_t,centered_t))
        trend=float(np.dot(centered_t,centered_x)/time_energy) if time_energy>0 else 0.0
        # These are descriptive diagnostics in stored ADC counts. They do not
        # imply calibrated voltage, resting baseline, or a clipping rail.
        row.update({
            'mean_count':float(np.mean(x)),
            'std_count':float(np.std(x)),
            'rms_count':float(np.sqrt(np.mean(x*x))),
            'peak_to_peak_count':float(np.ptp(x)),
            'p01_count':float(np.quantile(x,.01)),
            'p99_count':float(np.quantile(x,.99)),
            'linear_trend_count_per_s':trend,
        })
        row['constant_signal']=bool(np.ptp(x)==0)
        row['extreme_repeat_fraction']=float(max(np.mean(x==x.min()),np.mean(x==x.max())))
        row['clipping_status']='uncalibrated_extreme_repeat_flag' if row['extreme_repeat_fraction']>.05 else 'no_extreme_repeat_flag'
        if row['constant_signal']:row['status']='constant_signal'
    return rows
