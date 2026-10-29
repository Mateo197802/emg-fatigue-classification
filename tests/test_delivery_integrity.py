"""Delivery integrity checks use synthetic files and never require course data."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile
import pytest


def verifier():
    path=Path(__file__).resolve().parents[1]/'tools/verify_delivery.py'
    assert path.is_file(), 'Portable delivery verifier has not been implemented'
    spec=importlib.util.spec_from_file_location('delivery_verifier',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def archive(tmp_path, files, entries=None):
    path=tmp_path/'delivery.zip'
    manifest=''.join(hashlib.sha256(data).hexdigest()+'  '+name+'\n' for name,data in files.items())
    with zipfile.ZipFile(path,'w') as z:
        for name,data in (entries if entries is not None else files.items()):z.writestr(name,data)
        z.writestr('SHA256SUMS',manifest)
    return path


def test_complete_archive_is_verified(tmp_path):
    m=verifier();p=archive(tmp_path,{'entrega/report.md':b'verified evidence'})
    result=m.verify_zip(p)
    assert result['status']=='verified' and result['hashed_members']==1


def test_changed_member_is_rejected(tmp_path):
    m=verifier();p=archive(tmp_path,{'report.md':b'original'},[('report.md',b'changed')])
    with pytest.raises(ValueError,match='checksum'):m.verify_zip(p)


def test_unlisted_member_is_rejected(tmp_path):
    m=verifier();p=archive(tmp_path,{'report.md':b'original'},[('report.md',b'original'),('hidden.txt',b'extra')])
    with pytest.raises(ValueError,match='coverage'):m.verify_zip(p)


def test_missing_member_is_rejected(tmp_path):
    m=verifier();p=archive(tmp_path,{'missing.txt':b'absent'},[])
    with pytest.raises(ValueError,match='coverage'):m.verify_zip(p)


def test_duplicate_zip_member_is_rejected(tmp_path):
    m=verifier()
    with pytest.warns(UserWarning,match='Duplicate'):
        p=archive(tmp_path,{'report.md':b'data'},[('report.md',b'data'),('report.md',b'data')])
    with pytest.raises(ValueError,match='duplicate'):m.verify_zip(p)


@pytest.mark.parametrize('name',['../outside.txt','/absolute.txt','C:/absolute.txt','nested\\file.txt'])
def test_nonportable_or_unsafe_member_is_rejected(tmp_path,name):
    m=verifier();p=archive(tmp_path,{name:b'unsafe'})
    with pytest.raises(ValueError,match='unsafe'):m.verify_zip(p)


def test_duplicate_manifest_path_is_rejected(tmp_path):
    m=verifier()
    with pytest.raises(ValueError,match='duplicate'):
        m.parse_manifest(hashlib.sha256(b'x').hexdigest()+'  a.txt\n'+hashlib.sha256(b'x').hexdigest()+'  a.txt\n')


def test_folder_manifest_checks_visible_files(tmp_path):
    m=verifier();(tmp_path/'report.md').write_bytes(b'original')
    (tmp_path/'SHA256SUMS_ENTREGA.txt').write_text(hashlib.sha256(b'original').hexdigest()+'  report.md\n')
    assert m.verify_folder(tmp_path)['hashed_files']==1
    (tmp_path/'report.md').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):m.verify_folder(tmp_path)
