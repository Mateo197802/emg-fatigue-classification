import numpy as np

EARLY_PROXY_LABEL='early_recording_position_proxy'
LATE_PROXY_LABEL='late_recording_position_proxy'
LABELS={EARLY_PROXY_LABEL:0,LATE_PROXY_LABEL:1}
LEGACY_PROXY_LABELS={
    'early_proxy_nonfatigue':EARLY_PROXY_LABEL,
    'late_proxy_fatigue':LATE_PROXY_LABEL,
}


def display_proxy_label(label):
    """Return an honest recording-position name, including for legacy bundles."""
    if label in LABELS:
        return label
    if label in LEGACY_PROXY_LABELS:
        return LEGACY_PROXY_LABELS[label]
    raise ValueError(f'unknown historical proxy label: {label!r}')


def label_interval(start,end,recording_start,recording_end,eligible=True,shift=0.):
    if not eligible:return None
    duration=recording_end-recording_start
    if duration<=0 or start<recording_start or end>recording_end+1e-9 or end<start:
        raise ValueError('invalid window or recording bounds')
    early=recording_start+duration*(1/3+shift)
    late=recording_start+duration*(2/3+shift)
    if end<early:return EARLY_PROXY_LABEL
    if start>=late:return LATE_PROXY_LABEL
    return None


def make_windows(processed,config):
    rec=processed['recording'];rows=[]
    for r in processed['runs']:
        n=int(round(config.window_s*r['fs_hz']));step=max(1,int(round(n*(1-config.overlap))))
        for a in range(0,len(r['x'])-n+1,step):
            b=a+n;valid=bool(r['valid'][a:b].all())
            occupancy=float(np.mean(r['active'][a:b]))
            eligible=valid and occupancy>=config.activity_occupancy
            name=label_interval(float(r['t'][a]),float(r['t'][b-1]),*rec.bounds,eligible=eligible,shift=config.boundary_shift)
            rows.append({'window_id':f'{rec.recording_id}_{r["run_id"]}_{a}_{n}',
                'recording_id':rec.recording_id,'subject_id':rec.subject_id,'side':rec.side,'series':rec.series,
                'run_id':r['run_id'],'start':a,'end':b,'start_s':float(r['t'][a]),'end_s':float(r['t'][b-1]),
                'time_fraction':float((r['t'][a]+r['t'][b-1]-2*rec.t[0])/(2*(rec.t[-1]-rec.t[0]))),
                'qc_valid':valid,'activity_occupancy':occupancy,'activity_eligible':eligible,
                'label':LABELS[name] if name else None,'label_name':name,
                'label_evidence':'weak_elapsed_time','label_version':config.label_version,
                'status':'qc_invalid' if not valid else 'activity_not_detected' if not eligible else 'transition_or_boundary' if name is None else 'labeled'})
    return rows
