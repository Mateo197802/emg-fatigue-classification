import json
import pytest


def test_feature_cache_rejects_changed_bytes(tmp_path):
    from emg_fatigue.workflow import verify_feature_cache
    from emg_fatigue.contracts import StudyConfig, sha256_file
    c=StudyConfig();p=tmp_path/'features.parquet';p.write_bytes(b'original')
    (tmp_path/'processing_receipt.json').write_text(json.dumps({'config_sha256':c.digest(),'features_sha256':sha256_file(p)}))
    verify_feature_cache(tmp_path,c)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):verify_feature_cache(tmp_path,c)


def test_unsupported_band_and_invalid_jitter_rejected():
    from emg_fatigue.contracts import StudyConfig
    with pytest.raises(ValueError):StudyConfig(band_hz=(10,300))
    with pytest.raises(ValueError):StudyConfig(jitter_max=-1)
