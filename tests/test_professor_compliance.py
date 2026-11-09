import csv
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
ALLOWED={
    'COMPLETED AND VERIFIED',
    'IMPLEMENTED BUT NOT EMPIRICALLY VALIDATED',
    'BLOCKED BY MISSING REFERENCE DATA',
    'FAILED VERIFICATION',
}


def test_professor_requirements_matrix_covers_scientific_acceptance_items():
    path=ROOT/'docs'/'PROFESSOR_REQUIREMENTS_COMPLIANCE_2026-11-09.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)>=12
    assert len({r['requirement_id'] for r in rows})==len(rows)
    assert {r['status'] for r in rows}<=ALLOWED
    assert all(r['evidence'].strip() and r['scope_limit'].strip() for r in rows)
    required={
        'target_fatigue','reference_audit','preprocessing','filter_formulas',
        'features','windowing','phase_labels','fatigue_training','metrics',
        'architectures','report_figures','delivery'
    }
    assert required<={r['requirement_id'] for r in rows}
    status={r['requirement_id']:r['status'] for r in rows}
    assert status['target_fatigue']=='COMPLETED AND VERIFIED'
    assert status['fatigue_training']=='BLOCKED BY MISSING REFERENCE DATA'
    assert status['phase_labels']=='BLOCKED BY MISSING REFERENCE DATA'


def test_reference_audit_documents_presence_units_sync_calibration_and_granularity():
    text=(ROOT/'docs'/'REFERENCE_DATA_REQUIREMENTS.md').read_text(encoding='utf-8').lower()
    for field in ('fuente/ruta','unidad','resolución temporal','calibración',
                  'sincronización','granularidad','fuerza/torque','mvc',
                  'borg cr-10','cinemática','video','marcadores de repetición',
                  'identificadores','metadatos de adquisición','calibración adc',
                  'frecuencia de muestreo','anotaciones de fatiga'):
        assert field in text
    assert 'round(adc_value × 3300 / 4095, 2)' in text
    assert 'no acredita calibración física' in text


def test_report_and_deck_sources_use_the_executed_notch_formula():
    report=(ROOT/'tools'/'build_manuscript.py').read_text(encoding='utf-8')
    deck=(ROOT/'tools'/'build_defense_v2.mjs').read_text(encoding='utf-8')
    assert 'h(\'2.2  Independent reference audit\'' in report
    assert 'round(ADC_Value × 3300/4095, 2)' in report
    assert 'filter_designs.json stores the exact SOS rows' in report
    assert 'β=tan(B/2)' in deck and 'g=1/(1+β)' in deck
    assert 'α=sin(ω₀)/(2Q)' not in deck
    assert 'ADC_Value×3300/4095' in deck
