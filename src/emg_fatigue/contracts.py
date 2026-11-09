from dataclasses import dataclass, asdict, fields
from pathlib import Path
import hashlib, json, os
import subprocess
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
        if tuple(self.band_hz)!=(20.,400.) or self.filter_order!=4:
            raise ValueError('this frozen feature contract supports only order-4 prototype band 20-400 Hz')
        for value in [self.activity_frame_s,self.activity_hop_s,self.edge_s,self.gap_min_s,self.gap_multiplier,self.jitter_max,self.bootstrap_replicates]:
            if not np.isfinite(value) or value<=0:raise ValueError('all processing limits must be finite and positive')
        if not np.isfinite(self.boundary_shift):raise ValueError('boundary shift must be finite')
        if self.mode!='offline' or self.label_version!='weak_elapsed_thirds_binary_v1':
            raise ValueError('unsupported mode or label version')
        if abs(self.boundary_shift)>.10:raise ValueError('boundary shift outside prespecified range')

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ReferenceStudyConfig:
    """Separate, gated protocol for labels from independent biomechanical data."""

    study_id: str
    primary_window_s: float
    sensitivity_window_s: float
    overlap: float
    window_type: str
    spectral_taper: str
    psd_method: str
    analysis_band_hz: tuple
    filter_prototype_order: int
    edge_mask_s: float
    data_gate: str
    fatigue_label_definition_id: str
    fatigue_definition_approved: bool
    phase_definition_approved: bool
    participant_ids_verified: bool
    angle_convention: str | None

    def __post_init__(self):
        if self.primary_window_s != 0.5 or self.sensitivity_window_s != 0.25:
            raise ValueError('reference protocol requires 500 ms primary and 250 ms sensitivity windows')
        if self.overlap != 0.5 or self.window_type != 'fixed_length_sliding':
            raise ValueError('unsupported reference window protocol')
        if self.spectral_taper != 'hann' or self.psd_method != 'periodogram':
            raise ValueError('unsupported reference spectral estimator')
        if tuple(self.analysis_band_hz) != (20.0, 400.0) or self.filter_prototype_order != 4:
            raise ValueError('unsupported reference filter specification')
        if not np.isfinite(self.edge_mask_s) or self.edge_mask_s <= 0:
            raise ValueError('edge_mask_s must be finite and positive')
        if self.data_gate not in {'failed', 'passed'}:
            raise ValueError('data_gate must be failed or passed')
        if not self.study_id.strip() or not self.fatigue_label_definition_id.strip():
            raise ValueError('study and label-definition identifiers are required')
        for value in (
            self.fatigue_definition_approved,
            self.phase_definition_approved,
            self.participant_ids_verified,
        ):
            if value is not True and value is not False:
                raise ValueError('approval and identity states must be boolean')

    def require_training_ready(
        self,
        *,
        fatigue_reference_approved=False,
        phase_reference_approved=False,
        participant_ids_verified=False,
    ):
        if fatigue_reference_approved is not True:
            raise ValueError('fatigue reference is missing or unapproved')
        if phase_reference_approved is not True:
            raise ValueError('phase reference is missing or unapproved')
        if participant_ids_verified is not True:
            raise ValueError('participant grouping is not independently verified')
        if not self.fatigue_definition_approved:
            raise ValueError('fatigue label definition approval is absent from this config')
        if not self.phase_definition_approved:
            raise ValueError('phase label definition approval is absent from this config')
        if not self.participant_ids_verified:
            raise ValueError('participant identity approval is absent from this config')
        if self.data_gate != 'passed':
            raise ValueError('G0 data gate has not passed')


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


def read_reference_config(path):
    data=yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    unknown=set(data)-{f.name for f in fields(ReferenceStudyConfig)}
    if unknown:raise ValueError(f'unknown reference configuration keys: {sorted(unknown)}')
    if 'analysis_band_hz' in data:data['analysis_band_hz']=tuple(data['analysis_band_hz'])
    return ReferenceStudyConfig(**data)


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


def implementation_provenance():
    package=Path(__file__).resolve().parent;repo=package.parents[1]
    files={p.name:sha256_file(p) for p in sorted(package.glob('*.py'))}
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True)
    return {'git_revision':revision.stdout.strip() if revision.returncode==0 else 'unavailable',
      'package_source_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'module_hashes':files}
