"""Read archives without extracting member-controlled paths. Names stay private."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import hashlib, io, re, stat, subprocess, zipfile
import numpy as np
import pandas as pd
from .contracts import Recording

MAX_MEMBER=512*1024**2
MAX_TOTAL=3*1024**3


def read_csv_bytes(data, recording_id, subject_id, source, side, series, source_member=''):
    df=pd.read_csv(io.BytesIO(data),encoding='utf-8-sig')
    if set(df.columns) not in ({'Relative_Time_s','ADC_Value'},{'Relative_Time_s','ADC_Value','Voltage_mV'}):
        raise ValueError('CSV must contain Relative_Time_s,ADC_Value[,Voltage_mV] only')
    a=df[['Relative_Time_s','ADC_Value']].to_numpy(dtype=float)
    if len(a)<2 or not np.isfinite(a).all():raise ValueError('CSV requires at least two finite samples')
    if np.any(np.diff(a[:,0])<=0):raise ValueError('timestamps must be strictly increasing')
    return Recording(recording_id,subject_id,source,side,series,a[:,0],a[:,1],hashlib.sha256(data).hexdigest(),source_member)


def inspect_zip(path):
    with zipfile.ZipFile(path) as z:
        names=[];seen=set();total=0
        for i in z.infolist():
            name=i.filename.replace('\\','/');p=PurePosixPath(name);w=PureWindowsPath(name)
            if p.is_absolute() or w.drive or '..' in p.parts or stat.S_ISLNK(i.external_attr>>16):
                raise ValueError('unsafe ZIP member')
            if name.casefold() in seen:raise ValueError('duplicate ZIP member')
            seen.add(name.casefold());total+=i.file_size
            if i.flag_bits&1 or i.file_size>MAX_MEMBER or total>MAX_TOTAL:
                raise ValueError('encrypted or oversized archive')
            if not i.is_dir() and name.lower().endswith('.csv'):names.append(i.filename)
        if not names:raise ValueError('no CSV recordings')
        return sorted(names)


def _metadata(name,source):
    path=PurePosixPath(name.replace('\\','/'));stem=path.stem
    if source=='rar':
        m=re.fullmatch(r'(P\d+)_set(0[1-5])_(dr|iz)',stem,re.I)
        if not m:raise ValueError('unsupported RAR recording name')
        group=m[1].upper();series=f'R{int(m[2])}';side=m[3].lower()
    else:
        if len(path.parts)!=2:raise ValueError('training ZIP requires one subject folder per recording')
        m=re.search(r'(?:^|_)R([0-5])(?:_1)?(?:_|$)',stem,re.I)
        s=re.search(r'(?:^|_)(dr|iz)(?:_|$)',stem,re.I)
        if not m or not s:raise ValueError('ambiguous series or side in source filename')
        group=path.parts[0];series=f'R{m[1]}';side=s[1].lower()
    return group,series,{'dr':'right','iz':'left'}[side]


def load_recordings(path,config=None):
    path=Path(path).resolve();source=path.suffix.lower().lstrip('.')
    if source=='zip':
        names=inspect_zip(path)
        with zipfile.ZipFile(path) as z:items=[(n,z.read(n)) for n in names]
    elif source=='rar':
        names=subprocess.run(['tar','-tf',str(path)],check=True,capture_output=True,text=True,timeout=60).stdout.splitlines()
        if len(names)!=len(set(n.casefold() for n in names)):raise ValueError('duplicate RAR member')
        for n in names:
            if not re.fullmatch(r'P\d+_set0[1-5]_(dr|iz)\.csv',n,re.I):raise ValueError('unsafe or unsupported RAR member')
        items=[];total=0
        for n in sorted(names):
            b=subprocess.run(['tar','-xOf',str(path),n],check=True,capture_output=True,timeout=60).stdout
            total+=len(b)
            if len(b)>MAX_MEMBER or total>MAX_TOTAL:raise ValueError('oversized archive')
            items.append((n,b))
    else:raise ValueError('training source must be ZIP or RAR')
    metadata=[_metadata(n,source) for n,_ in items]
    groups={g:f'{source.upper()}_G{i:02d}' for i,g in enumerate(sorted({m[0] for m in metadata}),1)}
    return [read_csv_bytes(b,f'{source.upper()}_REC{i:04d}',groups[g],source,side,series,n)
            for i,((n,b),(g,series,side)) in enumerate(zip(items,metadata),1)]


def reassemble_fragments(paths):
    frames=[]
    for p in paths:
        r=read_csv_bytes(Path(p).read_bytes(),'inference','unseen','csv','unknown','unknown')
        frames.append(r)
    frames.sort(key=lambda r:r.t[0])
    for a,b in zip(frames,frames[1:]):
        if b.t[0]<=a.t[-1]:raise ValueError('overlapping fragments or clock reset')
    data=b'\n'.join(Path(p).read_bytes() for p in paths)
    return Recording('inference','unseen','csv','unknown','unknown',np.concatenate([r.t for r in frames]),np.concatenate([r.x for r in frames]),hashlib.sha256(data).hexdigest())
