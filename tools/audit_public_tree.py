"""Fail closed on restricted file types or exact private archive paths in Git."""
import argparse,re,subprocess,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-archive');a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
forbidden={'.rar','.zip','.joblib','.pkl','.pickle','.parquet','.npz','.npy','.docx','.pdf'}
private_paths=[]
if a.source_archive:
    with zipfile.ZipFile(a.source_archive) as z:private_paths=[n for n in z.namelist() if not n.endswith('/')]
issues=[]
for name in filter(None,tracked):
    f=repo/name
    if f.suffix.lower() in forbidden:issues.append((name,'restricted artifact type'))
    data=f.read_bytes()
    for private in private_paths:
        if private.encode('utf-8') in data:issues.append((name,'private source archive path'))
    if re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}',data):issues.append((name,'possible GitHub credential'))
if issues:
    for name,reason in issues:print(name,reason)
    raise SystemExit(1)
print(f'Public tree audit passed: {len(list(filter(None,tracked)))} tracked paths; no restricted artifacts, exact source paths or GitHub credential pattern.')
