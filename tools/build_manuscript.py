"""Fill a private copy of the supplied template using frozen numerical outputs.

Run with the bundled document Python runtime (python-docx 1.2.0). This script
reads no raw recordings. The retained template is never modified.
"""
import argparse, csv, hashlib, json, platform, subprocess, sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
import pandas as pd
from docx import Document
from docx.text.paragraph import Paragraph
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

parser=argparse.ArgumentParser()
parser.add_argument('--template',required=True)
parser.add_argument('--output-root',required=True)
parser.add_argument('--previous-report')
parser.add_argument('--include-phase',action='store_true',help='Include verified exploratory phase evidence and appendices')
args=parser.parse_args()
root=Path(args.output_root).resolve();assets=root/'report_assets'
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'src'))
from emg_fatigue.protocol_equations import report_equation_rows
from emg_fatigue.preanalysis import summarize_preanalysis
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
processing=json.loads((root/'processing_receipt.json').read_text())
evaluation=json.loads((root/'evaluation/evaluation_receipt.json').read_text())
summary=rows(root/'evaluation/metrics_summary.csv');results=rows(root/'evaluation/results.csv')
coverage=rows(root/'coverage.csv');sens=rows(root/'sensitivity_summary.csv');abl=rows(root/'ablation_summary.csv')
preanalysis=summarize_preanalysis(pd.DataFrame(rows(root/'qc.csv')),pd.DataFrame(rows(root/'line_prominence.csv')))
refs=json.loads((repo/'references/verified_references.json').read_text(encoding='utf-8'))
phase_public=repo/'results/phase_experiment'
if args.include_phase:
    phase_receipt=json.loads((phase_public/'evidence_receipt.json').read_text())
    for name,digest in phase_receipt['aggregate_hashes'].items():
        if hashlib.sha256((phase_public/name).read_bytes()).hexdigest()!=digest:raise ValueError('phase evidence checksum mismatch: '+name)
    phase_summary=rows(phase_public/'metrics_summary.csv');phase_conf=rows(phase_public/'confusion_matrices.csv');phase_pairs=rows(phase_public/'paired_differences.csv');phase_accuracy_pairs=rows(phase_public/'accuracy_paired_differences.csv')
    phase_aux=rows(phase_public/'auxiliary_phase_fold_metrics.csv')
    refs+=json.loads((repo/'references/phase_references.json').read_text(encoding='utf-8'))
    def phase_ms(model,name):
        r=next(r for r in phase_summary if r['model']==model and r['metric']==name)
        return f"{float(r['mean']):.4f} ± {float(r['sd']):.4f}"
pooled=json.loads((assets/'pooled_counts.json').read_text())
paired=rows(root/'evaluation/paired_differences.csv')
if len(results)!=evaluation['outer_folds']*6:raise RuntimeError('incomplete model/fold table')
def metric(model,name):return next(r for r in summary if r['model']==model and r['metric']==name)
def ms(model,name):
    r=metric(model,name);return f"{float(r['mean']):.3f} ± {float(r['sd']):.3f}"
def ci(model,name):
    r=metric(model,name);return f"{float(r['ci95_lower']):.3f}–{float(r['ci95_upper']):.3f}"

template=Path(args.template).resolve();template_bytes=template.read_bytes();doc=Document(template)
original_headings={p.text.strip():deepcopy(p._p) for p in doc.paragraphs if p.style.name.startswith('Heading')}
table_patterns=[deepcopy(t._tbl) for t in doc.tables]
geometry=[{'width':s.page_width,'height':s.page_height,'top':s.top_margin,'bottom':s.bottom_margin,'left':s.left_margin,'right':s.right_margin} for s in doc.sections]
# All editable body slots are replaced. Section properties, styles, theme,
# header/footer parts, logos and relationship-linked page furniture survive.
body=doc._element.body
for child in list(body):
    if child.tag!=qn('w:sectPr'):body.remove(child)
manuscript=root/'manuscript';manuscript.mkdir(exist_ok=True)
content=[]
def p(text,style=None):
    paragraph=doc.add_paragraph(text,style=style);paragraph.paragraph_format.space_after=Pt(6);paragraph.paragraph_format.keep_together=True
    content.append(text);return paragraph
def h(text,level=1):
    if text in original_headings:
        node=deepcopy(original_headings[text]);body.insert(len(body)-1,node)
        heading=Paragraph(node,body)
    else:heading=doc.add_heading(text,level=level)
    heading.paragraph_format.keep_with_next=True
    content.append('\n'+text+'\n')
def table(headers,data,pattern=1,font_size=9.0):
    node=deepcopy(table_patterns[pattern]);body.insert(len(body)-1,node)
    from docx.table import Table
    t=Table(node,doc._body)
    # Reuse the original table header/body components with the requested rows.
    for r in list(t._tbl.tr_lst)[1:]:t._tbl.remove(r)
    while len(t.columns)<len(headers):t.add_column(Inches(.7))
    if len(t.columns)!=len(headers):
        body.remove(node);t=doc.add_table(rows=1,cols=len(headers));t.style='Table Grid'
    for i,x in enumerate(headers):t.cell(0,i).text=str(x)
    for data_row in data:
        cells=t.add_row().cells
        for i,x in enumerate(data_row):cells[i].text=str(x)
    for j,row in enumerate(t.rows):
        no_split=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(no_split)
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after=Pt(3)
                for run in paragraph.runs:run.font.name='Arial';run.font.size=Pt(font_size)
            tcPr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right','insideH','insideV']:
                tag=OxmlElement('w:'+edge);tag.set(qn('w:val'),'single');tag.set(qn('w:sz'),'4');tag.set(qn('w:color'),'D9D9D9');borders.append(tag)
            tcPr.append(borders)
            if j==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'1F3A5F');tcPr.append(shade)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:run.font.color.rgb=RGBColor(255,255,255)
        if j==0:
            repeat=OxmlElement('w:tblHeader');row._tr.get_or_add_trPr().append(repeat)
    available=doc.sections[0].page_width-doc.sections[0].left_margin-doc.sections[0].right_margin
    n=len(headers)
    weights=([.26,.74] if n==2 else [.33,.22,.45] if n==3 else [.31,.23,.23,.23] if n==4 else [.34,.16,.16,.17,.17] if n==5 else [.20,.16,.16,.16,.16,.16])
    if n==3 and headers[0]=='Stage' and headers[1]=='Equation':weights=[.19,.36,.45]
    if n==3 and headers[0]=='Component' and headers[1]=='Equation':weights=[.22,.38,.40]
    if n==4 and headers[0]=='Feature' and headers[2]=='Equation':weights=[.18,.13,.34,.35]
    if n==6 and headers[0]=='Ref.':weights=[.07,.17,.23,.15,.17,.21]
    t.autofit=False
    for column,weight in zip(t.columns,weights):column.width=int(available*weight)
    for row in t.rows:
        for cell,weight in zip(row.cells,weights):cell.width=int(available*weight)
    content.append('| '+' | '.join(headers)+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in data))
    return t
def figure(name,caption,width=6.0):
    path=assets/name
    if not path.exists():raise FileNotFoundError(path)
    p0=doc.add_paragraph();p0.paragraph_format.keep_with_next=True;p0.add_run().add_picture(str(path),width=Inches(width))
    p(caption);content.append(f'Figure file: {name}')

title='Early-versus-late recording-position classification from sEMG windows'
p0=p(title,'Title')
for run in p0.runs:run.font.color.rgb=RGBColor(0,0,0)
for element in [p0._p.get_or_add_pPr(),doc.styles['Title'].element.get_or_add_pPr()]:
    for border in list(element.findall(qn('w:pBdr'))):element.remove(border)
p('Reanalysis of the supplied recordings using explicit weak labels and held-out recording groups')
identity={}
if args.previous_report:
    previous=Document(args.previous_report)
    for t in previous.tables:
        for row in t.rows:
            if len(row.cells)>=2 and row.cells[0].text.strip() in ['Group number','Members (name - student ID - role)']:
                identity[row.cells[0].text.strip()]=row.cells[1].text.strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
h('Group identification')
table(['Field','Value'],[
 ['Group number',identity.get('Group number','Not attested in the supplied identity table')],
 ['Report title',title],
 ['Members','Ethan Guevara; Mateo Gavilanes; Kennedy Rodriguez'],
 ['Code repository URL','https://github.com/Mateo197802/emg-fatigue-classification'],
 ['Source commit used for this build',commit],['Revision date','9 November 2026']])
h('Abstract')
abstract=(f"Surface electromyography can capture changes associated with muscular fatigue, but a classifier must be evaluated against an explicitly defined reference. "
 f"This study evaluates a binary early-versus-late recording-position proxy from sEMG windows in the supplied knee-extension recordings. The target is not physiological fatigue status. "
 f"The primary archive contains 140 CSV files and 4,256,257 samples grouped into 15 recording-derived subject codes; 138 R1–R5 recordings enter formal processing and two R0 controls remain separate. "
 f"A gap-preserving 20–400 Hz filter, candidate-activity mask, and 250 ms windows with 50% overlap produce {processing['windows_labeled']:,} labeled windows. "
 f"Complete first-third and last-third windows receive the two labels, while transition, invalid-quality and undetected-activity windows are excluded. "
 f"Amplitude, spectral and wavelet features are compared using majority and RMS-plus-MDF logistic baselines, regularized logistic regression, radial-basis SVM and random forest. "
 f"Outer leave-one-recording-group-out validation and grouped inner tuning yield macro-F1 {ms('nested_selector','f1_macro')} for the selection procedure, compared with {ms('logistic_rms_mdf','f1_macro')} for the simple logistic baseline. "
 f"Prespecified window, activity, boundary and notch sensitivities and feature-family ablations quantify dependence on methodological choices. "
 f"Models and preprocessing are serialized and checked after reloading; every table is linked to machine-generated results. "
 f"The central limitation is that no synchronized independent fatigue reference or independently reconciled person identity is available. "
 f"These results therefore demonstrate classification of an operational temporal proxy, rather than confirmed physiological fatigue or generalization to verified new people.")
if args.include_phase:abstract+=' An exploratory cascade learning EMG burst halves before temporal classification showed no clear improvement over direct ten-feature logistic regression.'
if not 200<=len(abstract.split())<=250:raise RuntimeError(f'abstract length {len(abstract.split())}')
p(abstract)
h('Keywords');p('surface electromyography; rectus femoris; recording-position proxy; grouped validation; signal processing; reproducibility')
h('1.  Fatigue: definition and state of the art')
h('1.1  Which fatigue are you detecting?',2)
p('The scientific question concerns acute fatigue during unilateral knee extension, but the model target available in this ZIP is not fatigue status. Each feature window is an observation; the learned class is whether that eligible window falls in the first or final third of its recording. Class 0 is the early recording-position proxy and class 1 is the late recording-position proxy. The model was not trained to distinguish fatigue from no fatigue. No central or peripheral fatigue, force-loss threshold or clinical fatigue diagnosis is independently identified. The prescribed exercise-to-failure context cannot be confirmed per file from the CSV alone. Scores therefore estimate agreement with the temporal proxy, conditional on valid candidate-active windows in the recording extremes.')
h('1.2  Physiological basis',2)
p('Under comparable contraction conditions, changes in motor-unit behavior and membrane propagation can shift the sEMG spectrum. Dynamic knee-extension work with synchronized torque, joint angle and MVC offers a materially stronger basis for interpreting spectral fatigue indices than elapsed file position alone [1]. Time-frequency estimation was developed to address the changing signal properties of cyclic contractions [2]. These observations motivate spectral and wavelet feature families without proving that an observed change in the supplied signal is caused by fatigue.')
p('RMS and MAV describe amplitude, which also depends on effort, recruitment, electrode geometry and movement. A decline in MDF or MNF is not specific to fatigue when contraction phase, force, sensor mode and acquisition transfer functions are unknown. Filtering can remove or attenuate components of biological and contaminating signals together; controlled filter studies [3] support explicit processing choices rather than a claim that the filter which maximizes model accuracy is physiologically best. Frequency-analysis recommendations [10] further motivate stated windows, tapering and bandwidth.')
h('1.3  How fatigue has been detected from EMG - review',2)
p('The comparison below uses ten references, including seven primary research articles. DOI metadata were checked; design statements are limited to the source information actually retrieved. Performance values from different muscles, sensors, labels or splits are not imported into the local results. A paper is not classified as subject-independent merely because it uses cross-validation. In particular, the sample-random 10-fold description in [6] is not equivalent to the present grouped design.')
table(['Ref.','Muscle / task','Reference / label','Features','Model','Performance interpretation'],[
 ['[1]','Rectus femoris; dynamic knee extension','Torque, joint angle and pre/post MVC','Spectral indices','No local classifier imported','Different measured reference'],
 ['[2]','Cyclic dynamic contractions','Time-dependent spectral change','Time-frequency','Spectral analysis','No classifier score imported'],
 ['[3]','Controlled sEMG artifact experiment','Noise/filter conditions','Signal spectra','Filter comparison','Not a fatigue-label classifier'],
 ['[4]','sEMG fatigue task','Detailed label not recovered here','Learned feature fusion','CNN/BLSTM attention','Publisher states cross-subject objective; no number imported'],
 ['[5]','sEMG fatigue task','Detailed label/split not recovered here','Wavelet packet','LSTM','No unsupported score imported'],
 ['[6]','Biceps; sEMG+ECG','Self-report: relaxed / transition / fatigue; every 120 s','Learned sequences','TCN and evidence fusion','Random sample-fold description; not grouped-equivalent'],
 ['[7]','Gastrocnemius; cycling','Anaerobic-threshold/time phases','sEMG+ACC fusion','CNN attention','Different physiology and modalities'],
 ['[8]','ML science methodology','Leakage taxonomy','Not a fatigue feature study','Methodological analysis','Supports clean separation'],
 ['[9]','Small-sample prediction','Simulation/uncertainty','Not a fatigue label study','Cross-validation analysis','Supports reporting variability'],
 ['[10]','sEMG frequency tutorial','Measurement/analysis choices','Spectral estimation','Tutorial','No classifier score imported']],pattern=2)
h('1.4  Gap and contribution',2)
p('The missing local link is an independent fatigue reference. The contribution is an executable binary pipeline with declared weak supervision, physiologically motivated feature families, grouped nested validation, required baselines, full OOF audit, and actual reloadable inference. The historical three-class file-position scores and upper-limb Borg/Cerqueira experiments are retained only in the original archive. They neither train nor validate the new local target.')
h('2.  Dataset and acquisition')
h('2.1  Prescribed protocol and recorded evidence',2)
p('The supplied guide prescribes bilateral rectus femoris MyoWare 2.0 RAW acquisition at at least 1000 Hz per channel; 20 mm electrode separation; a single preferred-leg 1RM; the same resulting 70% reference load for both legs; 2 s concentric and 2 s eccentric movements to technical failure; Borg CR-10 every five repetitions; ten minutes between legs; resting baseline and pre/post MVC. The same absolute load is not necessarily 70% of each leg’s own 1RM. These are intended conditions. The stored files verify only timestamped signed count values and filenames carrying side/series codes. No synchronized force, video, Borg, electrode photo, load, cadence or failure reason is included. The exported Voltage_mV value follows round(ADC_Value × 3300/4095, 2); that arithmetic does not verify physical calibration, so the model input remains ADC_Value.')
h('2.2  Independent reference audit',2)
p('The university instructions describe the planned protocol, including load prescription, cadence, Borg sampling and MVC. The ZIP does not establish that those conditions were executed. The table distinguishes references actually present from intended measurements. The 140 primary CSV files contain time, ADC counts and an exporter conversion; the local source audit found a maximum absolute discrepancy of 0.00 mV from round(ADC_Value × 3300/4095, 2). This verifies the arithmetic conversion only. It does not verify ADC reference voltage, sensor mode, gain, offset or physical muscle voltage.')
table(['Reference','Presence / source path','Units / time resolution','Calibration / synchronization','Label scope and limit'],[
 ['sEMG','Present: primary ZIP → nested EMG archive → 140 CSV','ADC counts; relative timestamp per sample; per-file median ≈1.000 kHz','No physical channel calibration; per-file relative clock only','Sample/file input, no independent fatigue state'],
 ['Force / torque','Absent from archive, CSV and included workbooks','No units or temporal series','No instrument calibration or shared clock','No repetition/window mechanical target'],
 ['MVC pre/post','Absent; mentioned in instructions only','No recorded value or time series','No calibration or synchronization record','Session contrast at most; no window onset'],
 ['Borg CR-10','Absent; instructions state planned query interval','No recorded score or timestamp','No synchronization record','Perceived exertion; not mechanical fatigue ground truth'],
 ['Knee angle / kinematics','Absent from local members','No units or sampled series','No angle calibration, sign convention or synchronized clock','No extension/flexion labels or transition boundaries'],
 ['Video','Absent from local members','No frame sequence or timebase','No alignment event with sEMG','No reviewed movement-phase labels'],
 ['Repetition events','Absent; R1–R5 filenames are series codes, not verified events','No event timestamps','No shared time alignment','No verified repetition, cadence, inversion or failure reason'],
 ['Participant / session IDs','Partial folder and filename groupings only','Not a physical unit; session keys absent','Cross-archive and person identity not independently checked','Recording groups only; unseen-person claim is conditional'],
 ['Acquisition metadata','Partial timestamps, signed ADC counts and file names','Near 1 kHz per-file median; 246 primary gaps >1.5 ms','No verified sensor mode, gain, clock certificate or rail calibration','Instructions are intended conditions, not execution evidence'],
 ['ADC exporter conversion','Present as Voltage_mV in all 140 primary CSVs','mV column computed per sample: round(ADC_Value × 3300/4095, 2)','Arithmetic checked; no physical calibration or known ADC rails','Pipeline keeps ADC_Value; conversion is not biological voltage'],
 ['Fatigue annotations','Absent from all supplied local files','No event time or rating','No independent annotated reference','Early/late labels remain recording-position proxies'],
],pattern=6,font_size=7.3)
h('2.3  Participants',2)
table(['Variable','Value','Evidence'],[['Analyzed recording groups',15,'Folder-derived; person identity unverified'],['Source recordings',140,'138 formal; two controls'],['Sex, age, mass, 1RM','Not available','Not invented or used as predictors'],['Verified repetitions to failure','Not available','No mechanical markers'],['Additional RAR',20,'Two source codes, kept separate']],pattern=3)
h('2.4  Exclusions',2)
p('Rules were frozen before model evaluation: R0 controls excluded from formal learning; nonfinite values or nonmonotonic timestamps rejected; discontinuities split runs without interpolation; excessive relative timestamp jitter and insufficient rate exclude runs; 100 ms filter edges excluded; windows must be complete and satisfy candidate activity occupancy of at least 75%. Transition and boundary-crossing windows are not a third class. Zero-energy or otherwise undefined features are not filled with zero. No recording is removed because of its classification accuracy.')
totals={k:sum(int(r[k]) for r in coverage) for k in ['raw_samples','qc_valid_samples','candidate_active_samples','candidate_windows','qc_valid_windows','active_windows','feature_valid_windows','labeled_windows']}
table(['Denominator','Count'],[[k.replace('_',' '),f'{v:,}'] for k,v in totals.items()])
h('2.5  Signal quality assessment',2)
p('Quality diagnostics preserve the original bytes and flag clock gaps, jitter, constant segments, extreme-value repetition and spectral line prominence. Signed exported counts do not establish calibrated ADC rails. SNR against seated resting noise is not estimable because no identified resting-baseline segment is provided; filter residual is not substituted for true noise. A record can contribute usable segments while windows crossing a timestamp gap remain excluded.')
table(['Diagnostic','Observed result','Interpretation'],[
 ['Contiguous runs',f"{preanalysis['run_count']:,} total; {preanalysis['valid_run_count']:,} pass; {preanalysis['excluded_run_count']:,} excluded",'Run-level time/QC gate'],
 ['Constant runs',f"{preanalysis['constant_run_count']:,}",'Rejected as unusable signal'],
 ['Extreme-repeat flags',f"{preanalysis['uncalibrated_extreme_repeat_flag_count']:,}",'ADC rail calibration unavailable; not verified clipping'],
 ['50 Hz line prominence',f"{preanalysis['line_50hz_above_threshold_count']:,}/{preanalysis['line_50hz_measured_count']:,} above {preanalysis['line_prominence_threshold_db']:.0f} dB",'Notch trigger diagnostic'],
 ['60 Hz line prominence',f"{preanalysis['line_60hz_above_threshold_count']:,}/{preanalysis['line_60hz_measured_count']:,} above {preanalysis['line_prominence_threshold_db']:.0f} dB",'Notch trigger diagnostic'],
 ['Median p99–p01 span',f"{preanalysis['median_p99_minus_p01_count']:.2f} ADC counts",'Across accepted runs; uncalibrated'],
 ['Median RMS',f"{preanalysis['median_rms_count']:.2f} ADC counts",'Across accepted runs; uncalibrated'],
],pattern=7,font_size=8.2)
figure('qc_examples.png','Figure 1. A continuous accepted example and a gap-crossing interval excluded by the timestamp rule. This is a QC illustration selected without model-outcome information.')
h('3.  Labelling strategy')
h('3.1  Chosen strategy and justification',2)
p('For each original file, the elapsed recording fraction is u=(t-t0)/(t1-t0), using original recording bounds before quality exclusions. A complete eligible window receives 0 when its end is before t0+(t1-t0)/3, and receives 1 when its start is at or after t0+2(t1-t0)/3. The central third and boundary-crossing windows are excluded. This implements weak_elapsed_thirds_binary_v1. Repetition-based thirds would be closer to the guide’s example but cannot be reconstructed as verified mechanical events from these files. Borg and MVC approaches were considered and are unavailable locally. The assumption of increasing fatigue with elapsed file time is explicitly weaker than either independent approach.')
h('3.2  Class definition and distribution',2)
counts=processing['label_counts'];n=sum(counts.values())
table(['Class','Exact eligible support','Windows','Percent'],[['0 early recording-position proxy','Complete first elapsed-time third',counts['0'],f"{100*counts['0']/n:.1f}%"],['1 late recording-position proxy','Complete last elapsed-time third',counts['1'],f"{100*counts['1']/n:.1f}%"]],pattern=5)
h('3.3  Circularity check',2)
p('No EMG feature defines label 0 versus 1: labels depend on elapsed recording bounds. RMS enters the candidate-activity eligibility mask, which creates a selection condition, not a fatigue class. This can still bias the analyzed population of windows and is addressed by activity-occupancy sensitivity. Time, file name, recording/group identity, side and series are not model inputs. MDF/MNF threshold-screen outputs from the original project are not labels or predictors in this task. The time oracle diagnostic is tautological by construction and is excluded from deployable models.')
h('3.4  Label reliability',2)
p('No independent local label source exists, so inter-source kappa cannot be estimated. Classifier-versus-proxy kappa is reported separately and is not label reliability. Boundary shifts of ±5% and ±10% recording duration re-evaluate the fixed RMS+MDF baseline. These changes quantify dependence on arbitrary temporal boundaries, not validation against force or subjective fatigue. A one-repetition shift cannot be evaluated without repetition markers.')
h('4.  Signal processing and feature extraction')
h('4.1  Preprocessing chain',2)
equations={name:(formula,implementation) for name,formula,implementation in report_equation_rows()}
processing_names=['Butterworth analog prototype','Butterworth SOS','Notch sensitivity','Otsu activity','Window indices','Reference window indices','Hann PSD']
table(['Stage','Equation','Executed parameters / limitation'],[[name,equations[name][0],equations[name][1]] for name in processing_names],pattern=6,font_size=8.5)
p('For each continuous segment, SOS coefficients use its observed sampling rate. The private filter_designs.json stores the exact SOS rows, sampling rate, edge-mask length and run binding; processing_receipt.json binds the manifest hash. If the optional notch is enabled, its measured trigger, Q, center frequency, bandwidth and actual b/a coefficients are recorded too. Forward-backward filtering masks 100 ms at both edges. Stored values remain ADC counts (not calibrated mV), and the scaler is fit within each training fold. MVC normalization is unavailable. Historical results use 250 ms windows. The separately specified 500 ms primary / 250 ms sensitivity protocol requires independent references and was not evaluated (Gate 0 failed).')
figure('signal_preprocessing_evidence.png','Figure 2. Anonymized QC-selected 1 s sEMG: stored and filtered trace, Hann PSD before/after, and digital band-pass response at fs≈1 kHz. Private evidence; no fatigue or phase labels.',width=5.9)
h('4.2  Features computed',2)
p('Unless stated otherwise, x[n] is the window-centered filtered signal in stored ADC counts. f[k] and Pxx[k] are the frequency bins and Hann-periodogram power restricted to 20–400 Hz.')
feature_specs=[('RMS','Amplitude','RMS','Effective amplitude; effort and electrode placement also affect it'),('MAV','Amplitude','MAV','Mean absolute amplitude'),('Waveform length/s','Amplitude','Waveform length per second','Local absolute change per unit time'),('MDF','Frequency','MDF','Median power frequency; spectral redistribution is not specific to fatigue'),('MNF','Frequency','MNF','Power-weighted mean frequency'),('Low/high ratio','Frequency','Power ratio','Relative 20–60 Hz to 60–400 Hz power'),('A3,D3,D2,D1','Time-frequency','Wavelet energies','db4 level 3 relative energies describe multiscale variation')]
table(['Feature','Family','Equation','Reason / condition'],[[name,family,equations[eq_name][0],reason+'; '+equations[eq_name][1]] for name,family,eq_name,reason in feature_specs],pattern=7,font_size=8.5)
p('Historical WL/s divides by N/fs, while the versioned v2 definition divides by the observed sample span (N−1)/fs. V2 preserves the same adjacent-difference numerator and explicitly changes only the duration convention. It is available for controlled sensitivity work; it was not retroactively inserted into the frozen historical results.')
h('4.2.1  Candidate features reviewed but not added',3)
p('A separate, target-free redundancy audit computed five candidate descriptors from the same QC-valid active windows. They were not used to fit a classifier. For each window, T=N/fs and x[n] is the centered, filtered signal; spectral probabilities use Hann-periodogram bins in the closed 20–400 Hz band.')
table(['Candidate feature','Formula used','Units / decision'],[
 ['Zero crossings per second','C_ZCR/T; C_ZCR=Σ 1[(x[n]≥τx ∧ x[n+1]<−τx) ∨ (x[n]≤−τx ∧ x[n+1]>τx)], τx=0','s⁻¹; candidate only'],
 ['Slope-sign changes per second','d[n]=x[n+1]−x[n]; C_SSC/T; C_SSC=Σ 1[(d[n]>τd ∧ d[n+1]<−τd) ∨ (d[n]<−τd ∧ d[n+1]>τd)], τd=0','s⁻¹; candidate only'],
 ['Integrated EMG','Σ|x[n]|/fs','counts·s; redundant with MAV at fixed duration'],
 ['Population variance','(1/N)Σ(x[n]−x̄)²','counts²; redundant with RMS after centering'],
 ['Normalized spectral entropy','pₖ=Pxx[k]/Σ_{j∈B}Pxx[j]; −Σ_{k∈B}pₖ ln(pₖ)/ln(K), K=|B|','unitless; descriptive candidate only'],
],pattern=7,font_size=7.7)
p('The audit covered 9,289 QC-valid candidate-active windows and used no label or model fit. Spearman correlations were MAV–iEMG 1.000 and RMS–variance 1.000 at both window and group-median levels; adding those pairs would duplicate fixed-duration amplitude information. MNF–entropy was 0.862 across overlapping windows and 0.957 across 15 folder-derived group medians; MDF–zero crossings was 0.740/0.887; wavelet D1–slope changes was 0.840/0.975; and zero crossings–slope changes was 0.736/0.904. These are descriptive correlations, not independent-sample inference or evidence of predictive gain. No candidate was added to the classifier; testing their predictive value requires independent reference labels and nested, training-only feature selection.')
h('4.3  Repetition- and phase-aligned feature trajectories',2)
p(preanalysis['repetition_phase_trajectories_note']+' RMS/MDF/MNF curves by elapsed-time decile were removed: those deciles overlap with the historical proxy definition and cannot serve as fatigue evidence. This analysis is unavailable until synchronized, reviewed repetition and phase markers are supplied.')
h('4.4  Bilateral analysis',2)
p('Left and right recordings remain in one group for every split. Feature summaries pair side within group and series where both are available. There is no evidence that rows from separately exported sides are synchronous, and no paired-sample concatenation is used. Side differences are electrical summaries, not proof of dominant leg or unequal fatigue rates.')
h('4.5  Feature selection or reduction',2)
p('All-feature models use the ten prespecified features; the required simple baseline uses exactly RMS and MDF. No data-driven feature subset is selected on all groups. Family ablations remove amplitude, spectral or wavelet features and repeat grouped inner logistic tuning. Scaling is part of the fitted Pipeline and sees training rows only [8].')
h('5.  Machine learning methodology')
h('5.1  Models evaluated',2)
table(['Model','Rationale','Search','Selection'],[['Majority','Trivial mandatory floor','Training majority','No test-derived prior'],['Logistic RMS+MDF','Mandatory simple reference','C=0.1/1/10','Grouped inner macro-F1'],['Logistic all','Regularized linear reference','C=0.1/1/10','Grouped inner macro-F1'],['SVM RBF','Nonlinear interaction','C=0.1/1/10; gamma=scale/0.01','Grouped inner macro-F1'],['Random forest','Nonlinear feature partition','300 trees; depth=3/6/None; leaf=2/5','Grouped inner macro-F1'],['Nested selector','Evaluate family-selection procedure','Across the same candidate grid','Inner performance only']],pattern=9)
h('5.2  Validation scheme',2)
p('Fifteen outer folds each hold out one recording group completely. Four inner grouped folds select configurations using average per-validation-group macro-F1. Both sides and all series of each group remain together. Test-group observations never fit the scaler, model, hyperparameters, threshold or feature subset. The candidate-activity and deterministic filtering transforms are performed independently per recording under the frozen offline protocol. Folder identity is not independently reconciled, so the unseen-person interpretation remains conditional. OOF predictions cover every eligible window exactly once per model. Small sample uncertainty remains substantial [9].')
figure('split_scheme.png','Figure 4. Outer group holdout and grouped inner selection. Learned preprocessing is fitted only on the training branch.')
h('5.3  Class imbalance',2)
p(f"The analyzed weak-label distribution is {counts['0']} early and {counts['1']} late windows. Logistic, SVM and random-forest candidates use balanced class weights computed within fitting. Majority class is determined from training labels only. No synthetic resampling combines subjects or creates test observations. Group-level evaluation means are not weighted by how many windows a person contributed.")
h('5.4  Reproducibility settings',2)
env=processing['environment'];total_runtime=sum(float(r['fit_predict_seconds']) for r in results if r['model']=='nested_selector')
table(['Item','Value'],[['Seeds','20261028'],['Python and numeric libraries',json.dumps(env,sort_keys=True)],['Hardware',platform.processor()+'; CPU sklearn execution'],['Nested evaluation wall time',f'{total_runtime:.1f} s summed across outer fold timers'],['Regenerate assets','python -m emg_fatigue report --config configs/study.yaml'],['Processing configuration SHA256',processing['config_sha256']]],pattern=11)
timers=[float(r['fit_predict_seconds']) for r in results if r['model']=='nested_selector']
if max(timers)>3*sorted(timers)[len(timers)//2]:p(f'The measured wall timers include an exceptional fold interval of {max(timers):.1f} s. No CPU-only time or steady-state training benchmark is inferred from that interval; it remains included in the audit rather than being removed.')
if args.include_phase:
    h('5.5  Exploratory phase cascade',2)
    p('A separately authorized exploratory experiment tested whether a learned local phase representation improves the same early/late target. Stage 1 is a 100-tree random forest, depth 3, minimum leaf 5, learning first/second halves of candidate EMG activity bursts lasting 0.5–8 s. It is not a verified knee extension/flexion detector. Stage 2 is logistic regression using the ten original features and its uncalibrated phase score; the principal variant adds score interactions with RMS, MDF and MNF. Auxiliary supervision covered 3,337 windows from 996 candidate bursts. All fatigue-side variants retain the identical 5,509 windows and 15 outer groups.')
    p('For feature vector X, the random forest score is q(X)=B⁻¹Σᵦ pᵦ(burst-half=1|X), with B=100 trees. The conditioned logistic probability is p_proxy=σ(β₀+βᵀX+γq+δ_R·q·RMS+δ_M·q·MDF+δ_N·q·MNF), where σ(z)=1/(1+e⁻ᶻ). This formula predicts the early/late temporal proxy. Its auxiliary target is first versus second burst half, not observed knee phase.')
    p('Stage 2 training uses group-out-of-fold stage 1 scores. Cross-fitting repeats inside the grouped internal search over C=0.1/1/10; no held-out group trains either stage or its scaler. The original protocol was frozen before phase extraction and results. A score-only diagnostic and an identically structured random-label control were also specified. This is exploratory analysis after the principal study, not independent confirmation. Architecture, model relationships, counts and validation checks are detailed in Appendices D–G.')
    figure('executed_proxy_architecture_es.png','Figure 8. Executed two-stage proxy experiment. Stage 1 predicts EMG burst half and stage 2 predicts early/late recording position; neither output is verified extension/flexion or physiological fatigue.',width=6.25)
h('5.6  Proposed architecture with independent references',2)
p('This architecture is proposed and has no trained model or performance results: the supplied archive lacks synchronized force or torque, verified knee-angle phase and a participant-session crosswalk. The phase classifier consumes EMG features X and is trained against synchronized video or angle labels; it returns q_ext=P(extension|X) and q_flex=P(flexion|X). The direct fatigue baseline is p_direct=σ(β₀+βᵀX). A cascade can use p_cascade=σ(a₀+aᵀX+η_ext·q_ext+η_flex·q_flex). Phase experts combine as p_expert=q_ext·f_ext(X)+q_flex·f_flex(X). The training inputs to stage 2 must use group-out-of-fold phase probabilities, and test-time predictions must never receive true phase labels.')
table(['Procedure','Probability rule','Supervision'],[
    ['Direct EMG → fatigue','σ(β₀ + βᵀX)','Independent torque-derived label'],
    ['Phase → fatigue cascade','σ(a₀ + aᵀX + η_ext q_ext + η_flex q_flex)','OOF phase scores during training'],
    ['Phase experts','q_ext f_ext(X) + q_flex f_flex(X)','Soft gating from predicted phase']],pattern=9)
doc.add_page_break()
h('5.7  Architecture equations and label rules',2)
architecture_equations=['Kinematic phase label','Candidate torque label','Direct fatigue model','Phase cascade','Soft-gated phase experts','Executed burst-half cascade']
table(['Component','Equation','Status / interpretation'],[[name,equations[name][0],equations[name][1]] for name in architecture_equations],pattern=7,font_size=8.25)
p('The proposed phase reference uses angular velocity omega(t)=dtheta(t)/dt. With the preregistered deadband delta, omega>delta denotes flexion, omega<−delta denotes extension, and |omega|≤delta is a transition excluded from the phase classifier. The angle sign and delta must be supported by sensor convention and noise measurements.')
p('The reference-backed analysis uses 500 ms windows with 50% overlap: N=round(0.5 fs), H=round(0.25 fs). At 1 kHz, N=500 samples, H=250 samples and nominal spectral-bin spacing is 2 Hz. A 250 ms window with 50% overlap is a sensitivity analysis. Windows remain within one verified phase and continuous signal segment; short phases and inversion-crossing windows are not padded or assigned a fabricated class.')
p('A candidate operational fatigue rule for biomechanics review is T_ref = median torque from three valid repetitions in a matched knee-angle range; MDC95 = 1.96 × sqrt(2) × SEM from instrument/test-retest repeatability; and onset at the first of two consecutive evaluations whose three-repetition rolling median is below T_ref − MDC95. This threshold and its exclusions are not frozen or used to label windows until the torque definition, calibration, repeatability source and persistence rule are approved. Borg and pre/post MVC can be secondary checks, but do not locate window-level onset without synchronized mechanical output.')
figure('reference_fatigue_architecture_es.png','Figure 9. Proposed reference-backed architecture. Its dashed references are required to train labels; because those data are absent, this is a design and not a trained result.',width=6.25)
figure('reference_models_es.png','Figure 10. Proposed direct fatigue baseline, phase cascade and soft-gated phase experts, all trained and evaluated against the same independent fatigue reference. No branch is trained on the supplied ZIP.',width=6.25)
figure('windowing_protocol_es.png','Figure 11. The historical 250 ms proxy and the separately defined 500 ms reference-backed primary protocol; rectangles overlap by 50%. This is a schematic window illustration, not a sample label.',width=6.25)
h('6.  Results')
h('6.1  Main comparison table',2)
models=['majority','logistic_rms_mdf','logistic_all','svm_rbf','random_forest','nested_selector']
display_models=['Majority','Logistic RMS + MDF','Logistic all','SVM RBF','Random forest','Nested selector']
table(['Model','Accuracy','Precision macro','Recall macro','F1 macro','ROC AUC'],[[display]+[ms(m,k) for k in ['accuracy','precision_macro','recall_macro','f1_macro','roc_auc']] for m,display in zip(models,display_models)],pattern=12)
p(f"The prespecified nested procedure achieved macro-F1 {ms('nested_selector','f1_macro')} (group-bootstrap 95% CI {ci('nested_selector','f1_macro')}). Average precision was {ms('nested_selector','average_precision')}; trapezoidal PR-AUC was {ms('nested_selector','pr_auc_trapezoidal')}. The RMS+MDF baseline achieved {ms('logistic_rms_mdf','f1_macro')}. Family point estimates do not establish a unique superior model, and choosing a winner from these outer estimates would add post-selection optimism.")
for r in paired:p(f"Paired contrast {r['comparison']}: mean macro-F1 difference {float(r['mean_difference']):+.3f}; 95% group-bootstrap interval {float(r['ci95_lower']):+.3f} to {float(r['ci95_upper']):+.3f}, {r['paired_groups']} matched groups.")
if args.include_phase:
    acc_pair=next(r for r in phase_accuracy_pairs if r['candidate']=='nested_selector')
    p(f"Paired Accuracy difference for nested selector minus RMS+MDF was {float(acc_pair['mean_difference']):+.4f}; 95% group-bootstrap interval [{float(acc_pair['ci95_lower']):+.4f}, {float(acc_pair['ci95_upper']):+.4f}] across {acc_pair['n_paired_groups']} groups. This interval crosses zero. The aggregate calculation and bootstrap settings are in results/phase_experiment/accuracy_paired_differences.csv.")
h('6.2  Per-class performance of the best model',2)
p('Here “best model” means the prespecified nested selection procedure, not the family with the largest retrospectively observed point estimate. Class metrics are mean ± SD across held-out groups; support below is the pooled OOF count.')
table(['Class','Precision','Recall','F1','Support'],[[f'{i} '+('early proxy' if i==0 else 'late proxy'),ms('nested_selector',f'precision_class{i}'),ms('nested_selector',f'recall_class{i}'),ms('nested_selector',f'f1_class{i}'),pooled[f'support_class{i}']] for i in [0,1]],pattern=13)
h('6.3  Confusion matrix',2)
figure('confusion.png','Figure 5. Pooled OOF confusion counts and row-normalized proportions. Window pooling does not replace across-group uncertainty.')
h('6.4  Per-subject performance',2)
figure('held_out_groups.png','Figure 6. One point per held-out recording group; identity follows source folders and is unverified independently. The figure remains in the private report.')
h('6.5  Feature importance or ablation',2)
figure('ablation.png','Figure 7. Logistic feature-family ablations with grouped inner tuning. Lower performance after removal indicates predictive contribution for the proxy, not a causal physiological mechanism.')
table(['Subset','Macro F1 mean','SD'],[[r['ablation'],f"{float(r['macro_f1_mean']):.3f}",f"{float(r['macro_f1_sd']):.3f}"] for r in abl])
h('6.6  Bilateral results',2)
bil=rows(assets/'bilateral_summary.csv')
table(['Feature','Paired group-series','Mean L−R','Median L−R'],[[r['feature'],r['paired_group_series'],f"{float(r['mean_left_minus_right']):.3f}",f"{float(r['median_left_minus_right']):.3f}"] for r in bil])
p('These comparisons summarize candidate-active electrical features in available matched side/series combinations. A group can contribute several series, so the pair count is not an independent-person sample size. Held-out-group side-specific model metrics are provided privately. No dominant-leg, same-relative-intensity or unequal-fatigue-rate conclusion is justified.')
if args.include_phase:
    h('6.7  Supplemental phase experiment',2)
    phase_names=['Direct logistic 10','10 features plus phase score','Phase conditioned','Phase score only','Random label control']
    phase_models=['baseline_10','phase_score','phase_conditioned','phase_only','random_conditioned']
    table(['Procedure','Accuracy mean and SD','Macro F1 mean and SD','ROC AUC mean and SD'],[[name,phase_ms(model,'accuracy'),phase_ms(model,'f1_macro'),phase_ms(model,'roc_auc')] for model,name in zip(phase_models,phase_names)])
    c=next(r for r in phase_pairs if r['comparison']=='phase_conditioned_minus_baseline_10')
    p(f"The principal phase-conditioned contrast against direct ten-feature logistic regression was {float(c['mean_delta']):+.4f} macro-F1, with a paired group-bootstrap 95% interval [{float(c['ci95_lower']):+.4f}, {float(c['ci95_upper']):+.4f}]. Six groups improved, eight worsened and one tied. The interval crosses zero; there is no clear improvement. The random-label control reached {phase_ms('random_conditioned','f1_macro')}. The direct supplemental baseline reproduces primary logistic_all predictions; it is not the nested selection procedure reported in section 6.1.")
    acc_phase_pair=next(r for r in phase_accuracy_pairs if r['candidate']=='phase_conditioned')
    p(f"For paired group Accuracy, conditioned minus direct was {float(acc_phase_pair['mean_difference']):+.4f}, with a 95% group-bootstrap interval [{float(acc_phase_pair['ci95_lower']):+.4f}, {float(acc_phase_pair['ci95_upper']):+.4f}]. This interval also crosses zero; neither Accuracy nor macro-F1 shows a clear gain for the auxiliary phase condition.")
    p(f"The auxiliary phase head achieved Accuracy {mean(float(r['accuracy']) for r in phase_aux):.4f} ± {stdev(float(r['accuracy']) for r in phase_aux):.4f} and macro-F1 {mean(float(r['f1_macro']) for r in phase_aux):.4f} ± {stdev(float(r['f1_macro']) for r in phase_aux):.4f} across held-out groups. Its target is EMG burst halves, not observed knee movement. No extra classifier or split establishes physiological fatigue. All procedure-specific absolute and row-normalized OOF confusion matrices are included in Appendices E–G.")
h('7.  Discussion')
h('7.1  Interpretation',2)
p('The implementation now answers a declared binary operational task with actual inference, but its biological interpretation remains limited by supervision. The exact-time oracle reaches perfect agreement by definition and demonstrates why a high temporal-label score would not itself validate fatigue. The nested model score does not clearly exceed the simple RMS+MDF baseline. The final full-cohort artifact is selected using internal grouped validation, and is not an additional generalization experiment. It should be used as an auditable research prototype for this weak target.')
h('7.2  Comparison with the literature',2)
p('The torque- and MVC-informed knee-extension experiment in [1] addresses a stronger physiological reference than these files. Dynamic spectral estimation [2] motivates attention to nonstationarity and contraction conditions. Fusion methods [4–7] use other architectures, modalities or targets, so their published accuracy cannot be ranked directly against the current grouped macro-F1. The random-sample fold description of [6] further illustrates why a cross-validation label does not guarantee subject separation. We adopt leakage controls [8] and expose small-cohort variability [9] rather than importing a literature score as local evidence.')
h('7.3  Failure analysis',2)
if args.include_phase:p('The phase result does not justify replacing the primary artifact. Its first stage has limited discrimination of an EMG-derived reference, and neither stage observes true movement phase. Cheah et al. [11] used IMU-based concentric/eccentric segmentation and noted that envelope extrema need not coincide with movement phase, particularly under fatigue. Potvin [12] motivates control of kinematics. Guzu et al. [13] combines exercise categories with EMG-derived fatigue states, which differs from our extension-first hypothesis. These references support studying the question; none demonstrates that this implementation must improve. True phase labels, independent fatigue references and new-session/domain tests remain necessary.')
selected=[r for r in results if r['model']=='nested_selector'];worst=min(selected,key=lambda r:float(r['f1_macro']));best=max(selected,key=lambda r:float(r['f1_macro']))
p(f"Held-out-group macro-F1 ranged from {float(worst['f1_macro']):.3f} to {float(best['f1_macro']):.3f}. The least accurate fold contained {worst['n_test']} eligible windows, with confusion counts TN={worst['cm_0_0']}, FP={worst['cm_0_1']}, FN={worst['cm_1_0']}, TP={worst['cm_1_1']}. These are measured error counts, not a diagnosis of its cause. Misclassified windows are preserved in private OOF predictions for review. Without video, force or acquisition metadata, attributing the errors to true fatigue, technique or electrode displacement would be speculation.")
h('7.4  Limitations',2)
p('Labels are approximate and cannot establish an absence of fatigue in the first third. Group identity is inferred from filenames; person-level independence remains conditional. The cohort is small, signals overlap, phase and load are not verified, and candidate activity may include artefact. Transition and low-activity windows are outside the evaluated target, so reported performance is conditional on substantial selection. Physical channel mode and calibration, synchronized force/MVC, Borg and mechanical repetition markers are unavailable. Every transform is offline; zero-phase filtering, per-recording activity thresholds and knowledge of full recording bounds prevent a real-time onset claim. The separate RAR is not independently identified or untouched external validation. Some literature label/split details remain unrecovered and are explicitly marked rather than assumed.')
h('7.5  What you would do differently',2)
p('Before new acquisition, obtain the existing course coordinator’s acquisition metadata and synchronized references if available under the original consent and protocol. Reconcile person/session keys and sensor mode, then define the reference construct before looking at model results. If that information cannot be recovered, retain the present weak-target scope and avoid renaming elapsed-time classification as physiological validation. Any new acquisition would require its own approved course protocol and cannot silently replace the shared cohort.')
h('8.  Conclusions')
p(f"1. An executable binary pipeline produced {processing['windows_labeled']:,} eligible weak-labeled windows from 138 formal recordings, with all quality and selection denominators retained.")
p(f"2. The nested selection procedure achieved macro-F1 {ms('nested_selector','f1_macro')}; its observed mean did not clearly improve on the {ms('logistic_rms_mdf','f1_macro')} RMS+MDF baseline.")
p('3. Signal processing, grouped OOF evaluation, complete model reloading and fragment reassembly are auditable software capabilities; they do not substitute for a fatigue reference.')
p('4. The available evidence supports early/late proxy classification. Independent physiological fatigue detection, calibrated force loss and verified unseen-person generalization remain unestablished.')
if args.include_phase:p('5. Adding an EMG burst-half score did not clearly improve direct ten-feature logistic classification. The first-stage burst-half head, second-stage temporal-proxy classifier and final primary RF artifact are distinct fits and roles; the exploratory cascade is not promoted to the primary model.')
h('9.  Authors')
p('Ethan Guevara; Mateo Gavilanes; Kennedy Rodriguez.')
h('10.  References')
for i,r in enumerate(refs,1):p(f"[{i}] {r['authors']}, “{r['title']},” {r['journal']}, {r['year']}. doi: {r['doi']}.")
h('Appendix A - Additional figures')
figure('sensitivity.png','Prespecified sensitivity analyses, fixed C=1 RMS+MDF logistic reference. Different retained-window populations are stated explicitly.')
table(['Variation','Windows','Groups','Macro F1','SD'],[[r['analysis'],r.get('n_windows','NA'),r.get('groups','NA'),f"{float(r['macro_f1_mean']):.3f}" if r.get('macro_f1_mean') else 'NA',f"{float(r['macro_f1_sd']):.3f}" if r.get('macro_f1_sd') else 'NA'] for r in sens])
h('Appendix B - Full feature list')
p('rms; mav; waveform_length_per_s; mdf_hz; mnf_hz; power_ratio_low_high; wavelet_A3; wavelet_D3; wavelet_D2; wavelet_D1. Metadata are excluded by an explicit column transformer. The all-feature schema is identical in fitting and prediction.')
h('Appendix C - Repository map')
p('io/contracts: source and configuration contracts. qc/preprocessing/activity/labels/features: data transformations. validation/evaluation: nested grouped OOF and metrics. inference: trusted local bundle and prediction. sensitivities: frozen variations and family ablations. reporting: numerical assets and sources. tests: software contracts and regressions. Raw signals, models, OOF predictions and this full report remain private; public code and deidentified fold metrics are separate.')
if args.include_phase:
    h('Appendix D.1 Phase architecture and audit')
    p('The auxiliary target is first/second EMG burst half. The final target is early/late recording position. Neither is a verified extension/flexion or physiological fatigue label. Bursts come from the existing activity mask; complete windows crossing a burst midpoint or edge lack auxiliary labels. Their absence does not exclude them from the fixed fatigue-side comparison.')
    figure('phase/phase_architecture.png','Figure D1. Actual two-stage architecture. Stage 2 preserves the EMG features and receives a cross-fitted auxiliary score. The dotted training reference is used only within training groups.')
    p('The auxiliary head uses 100 RF trees, depth 3 and minimum leaf 5. The primary final artifact uses 300 RF trees, depth 3 and minimum leaf 2. Both use balanced weights but learn different targets. The primary nested selector chooses one family internally; it is not an ensemble vote. The full-cohort final fit supplies a reloadable artifact and no additional OOF performance estimate.')
    doc.add_page_break();h('Appendix D.2 Model relationships',2)
    figure('phase/model_relationships.png','Figure D2. Six principal evaluation procedures, five supplemental configurations, auxiliary phase supervision and the separate full-cohort primary artifact. Supplemental baseline_10 duplicates the primary logistic_all prediction procedure, so it is a matched reference rather than independent corroboration.')
    table(['Variant','Stage 2 input','Purpose'],[['baseline_10','Original 10 features','Matched direct reference'],['phase_score','10 features plus phase score','Cascade without interactions'],['phase_conditioned','10 plus score; interactions with RMS, MDF, MNF','Principal exploratory variant'],['phase_only','Only phase score','Bottleneck diagnostic'],['random_conditioned','Same conditioned input; random auxiliary labels','One capacity control']])
    p('All logit variants select C=0.1/1/10 by four grouped inner folds. Stage 1 uses three-fold cross-fitting within each inner-training set and four-fold cross-fitting in the final outer-training set. Audits preserve 278 distinct auxiliary fits and 645 requests, checking disjoint train/test groups and membership in parent training groups. There are 75 supplemental model×fold rows and 27,545 second-stage OOF rows. Baseline predicted classes match exactly; maximum score difference is 5.10e-13. Reconstructed auxiliary predictions reproduce their original group metrics.')
    doc.add_page_break();h('Appendix E Confusions for primary procedures')
    p('Rows are reference and columns are prediction. Classes 0/1 mean early/late proxy. Each procedure has 2,329 early and 3,180 late windows; repeated procedures do not multiply the biological sample size. Pooled confusion counts describe error totals; principal uncertainty remains between held-out groups.')
    for name,caption in [('primary_confusions_counts.png','Figure E1. Absolute pooled OOF confusion counts for all six principal procedures.'),('primary_confusions_normalized.png','Figure E2. The same principal matrices normalized within each reference row.')]:figure('phase/'+name,caption)
    doc.add_page_break();h('Appendix F Confusions for phase variants')
    p('The five supplemental procedures use the exact same early/late windows and support as Appendix E. The matched direct model is the ten-feature logistic reference, not the nested selector. Predicted stage 1 phase scores are never treated as verified movement labels.')
    for name,caption in [('phase_confusions_counts.png','Figure F1. Absolute pooled OOF confusion counts for all five supplemental variants.'),('phase_confusions_normalized.png','Figure F2. Supplemental confusion matrices normalized within each reference row.')]:figure('phase/'+name,caption)
    doc.add_page_break();h('Appendix G Auxiliary phase reference and integrity')
    p('For this appendix only, 0 means first burst half and 1 means second burst half. Support is 1,663/1,674, total 3,337 windows. These labels come from EMG activity segmentation and do not identify extension or flexion. They must not be confused with the early/late labels used by the second stage.')
    figure('phase/auxiliary_phase_confusion_counts.png','Figure G1. OOF confusion of the first-stage auxiliary head against EMG burst-half supervision.',width=4.8)
    ar=next(r for r in phase_conf if r['model']=='auxiliary_phase_head')
    table(['Reference row','Predicted 0 proportion','Predicted 1 proportion'],[['First burst half',f"{float(ar['norm_0_0']):.4f}",f"{float(ar['norm_0_1']):.4f}"],['Second burst half',f"{float(ar['norm_1_0']):.4f}",f"{float(ar['norm_1_1']):.4f}"]])
    p('The public supplement contains a model registry, per-procedure aggregate matrices, opaque fold metrics, paired contrasts, code and tests. The private delivery retains all granular OOF, candidate bursts, phase support, inner searches, splits, head requests and SHA256 receipts. Evidence tests reject duplicated/missing windows, reference-label disagreement, nonbinary predictions, group overlap, incorrect normalization and private identifier export. The repository and delivery manifests bind the published code and final artifacts. No independent physiological reference is created by these integrity checks.')
for section in doc.sections:
    for paragraph in section.footer.paragraphs:
        for run in paragraph.runs:
            if 'Report Template' in run.text:run.text=run.text.replace('Report Template','Binary sEMG reanalysis')
final=manuscript/'EMG_Binary_Report.docx';doc.save(final)
(manuscript/'EMG_Binary_Report.md').write_text('\n\n'.join(content)+'\n',encoding='utf-8')
if template.read_bytes()!=template_bytes:raise RuntimeError('reference template changed')
write={'template_sha256':hashlib.sha256(template_bytes).hexdigest(),'template_geometry':geometry,'output_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'code_commit':commit,'abstract_word_count':len(abstract.split()),'top_level_sections_preserved':list(range(1,11)),'phase_supplement_included':args.include_phase,'reference_count':len(refs),'body_edit':'all editable template body slots filled; blank placeholder spacing removed; source styles/header/footer/section retained','unresolved':['independent fatigue reference','person identity reconciliation','physical calibration/mode','individual student contribution attestation','selected literature label/split details explicitly marked']}
(manuscript/'manuscript_receipt.json').write_text(json.dumps(write,indent=2)+'\n',encoding='utf-8')
print(final)
