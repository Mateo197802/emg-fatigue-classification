from dataclasses import dataclass, asdict, fields
from pathlib import Path
import hashlib, json, os
import numpy as np
import yaml


@dataclass(frozen=True)
class StudyConfig:
    label_version: str = 'weak_elapsed_thirds_binary_v1'
    seed: int = 20261028
    window_s: float = .25
    overlap: float = .5
    activity_occupancy: float = .75
    activity_frame_s: float = .05
    activity_hop_s: float = .025
    band_hz: tuple = (20.,400.)
    filter_order: int = 4
    edge_s: float = .1
    gap_min_s: float = .0015
    gap_multiplier: float = 1.5
    jitter_max: float = .05
    inner_folds: int = 4
    bootstrap_replicates: int = 5000
    notch: bool = False
    boundary_shift: float = 0.
    source_kind: str = 'zip'
    group_identity_status: str = 'filename_derived_unverified'
    mode: str = 'offline'

    def __post_init__(self):
        if not (self.window_s>0 and 0<=self.overlap<1 and 0<self.activity_occupancy<=1):
            raise ValueError('invalid window or occupancy configuration')
        if not (0<self.band_hz[0]<self.band_hz[1] and self.inner_folds>=2):
            raise ValueError('invalid frequency band or grouped folds')
        if self.mode!='offline' or self.label_version!='weak_elapsed_thirds_binary_v1':
            raise ValueError('unsupported mode or label version')
        if abs(self.boundary_shift)>.10:raise ValueError('boundary shift outside prespecified range')

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


@dataclass
class Recording:
    recording_id: str
    subject_id: str
    source: str
    side: str
    series: str
    t: np.ndarray
    x: np.ndarray
    sha256: str
    source_member: str = ''

    @property
    def bounds(self):return float(self.t[0]),float(self.t[-1])


def read_config(path):
    data=yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    unknown=set(data)-{f.name for f in fields(StudyConfig)}
    if unknown:raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    if 'band_hz' in data:data['band_hz']=tuple(data['band_hz'])
    return StudyConfig(**data)


def output_root(value=None):
    value=value or os.environ.get('EMG_OUTPUT_ROOT')
    if not value:raise ValueError('set EMG_OUTPUT_ROOT or --output-dir to private storage outside the repository')
    root=Path(value).expanduser().resolve()
    repo=Path(__file__).resolve().parents[2]
    if root==repo or repo in root.parents:raise ValueError('private outputs must remain outside repository')
    root.mkdir(parents=True,exist_ok=True)
    return root


def write_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
