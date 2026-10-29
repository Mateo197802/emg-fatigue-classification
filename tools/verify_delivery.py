"""Verify a private delivery without extracting files or loading model objects.

Python standard library only. Hashes establish consistency, not authenticity
or physiological validation. Run against a trustedly obtained manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import zipfile


def safe_name(name):
    parts=PurePosixPath(name).parts
    if (not name or '\\' in name or name.startswith('/') or
            PureWindowsPath(name).drive or any(p in {'.','..'} for p in name.split('/')) or
            str(PurePosixPath(name))!=name or not parts):
        raise ValueError('unsafe member path: '+name)
    return name


def parse_manifest(text):
    entries={}
    for line in text.splitlines():
        match=re.fullmatch(r'([a-f0-9]{64})  (.+)',line)
        if not match:raise ValueError('invalid manifest line')
        digest,name=match.groups();safe_name(name)
        if name in entries:raise ValueError('duplicate manifest path: '+name)
        entries[name]=digest
    if not entries:raise ValueError('empty manifest')
    return entries


def digest_stream(stream):
    digest=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def verify_zip(path):
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        names=[i.filename for i in archive.infolist() if not i.is_dir()]
        if len(names)!=len(set(names)):raise ValueError('duplicate ZIP member')
        for name in names:safe_name(name)
        if 'SHA256SUMS' not in names:raise ValueError('missing SHA256SUMS')
        entries=parse_manifest(archive.read('SHA256SUMS').decode('utf-8-sig'))
        if set(entries)!=set(names)-{'SHA256SUMS'}:raise ValueError('manifest coverage mismatch')
        for name,expected in entries.items():
            with archive.open(name) as stream:actual=digest_stream(stream)
            if actual!=expected:raise ValueError('checksum mismatch: '+name)
        code_manifest='codigo/manifest/SHA256SUMS'
        if code_manifest in names:
            code=parse_manifest(archive.read(code_manifest).decode('utf-8-sig'))
            actual_code={n[len('codigo/'):] for n in names if n.startswith('codigo/')}-{ 'manifest/SHA256SUMS' }
            if set(code)!=actual_code:raise ValueError('code manifest coverage mismatch')
            for name,expected in code.items():
                if entries['codigo/'+name]!=expected:raise ValueError('code checksum mismatch: '+name)
    with path.open('rb') as stream:digest=digest_stream(stream)
    return {'status':'verified','zip_members':len(names),'hashed_members':len(entries),
            'zip_bytes':path.stat().st_size,'zip_sha256':digest,
            'scope':'All file hashes and ZIP read/CRC checks, plus code manifest when present. No model execution.'}


def verify_folder(folder):
    folder=Path(folder).resolve()
    manifest=folder/'SHA256SUMS_ENTREGA.txt'
    entries=parse_manifest(manifest.read_text(encoding='utf-8-sig'))
    visible={p.name for p in folder.iterdir() if p.is_file()}-{'SHA256SUMS_ENTREGA.txt'}
    if set(entries)!=visible:raise ValueError('visible file manifest coverage mismatch')
    for name,expected in entries.items():
        file=(folder/name).resolve()
        if not file.is_relative_to(folder):raise ValueError('unsafe resolved file: '+name)
        with file.open('rb') as stream:actual=digest_stream(stream)
        if actual!=expected:raise ValueError('checksum mismatch: '+name)
    return {'status':'verified','hashed_files':len(entries)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path)
    parser.add_argument('--folder',type=Path)
    args=parser.parse_args()
    if not(args.zip or args.folder):parser.error('provide --zip and/or --folder')
    try:
        result={'status':'verified'}
        if args.zip:result['archive']=verify_zip(args.zip)
        if args.folder:result['visible_files']=verify_folder(args.folder)
        print(json.dumps(result,indent=2))
    except (ValueError,OSError,zipfile.BadZipFile,KeyError,UnicodeError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)}));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
