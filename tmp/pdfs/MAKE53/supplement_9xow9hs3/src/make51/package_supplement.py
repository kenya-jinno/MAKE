"""Package a concrete reviewer supplement without publishing or external writes."""
from pathlib import Path
import hashlib, zipfile

ROOT=Path(__file__).resolve().parents[2]
paths=[]
for pattern in [
    'config/experiment_config_MAKE49.json',
    'src/__init__.py','src/make49/*.py','src/metrics/*.py','src/make51/*',
    'results/make49/*.json','results/make49/stage4_cache/*.json',
    'results/make51/*','results/make51/model_validation/*',
    'results/figures/MAKE51/*.pdf','Definitions/*',
    'main_paper_MAKE49.tex','main_paper_MAKE51.tex','main_paper_MAKE51.pdf',
    'MAKE51_revision_notes.md','MAKE51_response_to_reviewers.md',
    'MAKE49_実験設定表.md','MAKE49_段階1_監査結果.md']:
    paths.extend(p for p in ROOT.glob(pattern) if p.is_file())
paths=sorted(set(paths))
assert (ROOT/'results/make51/model_validation/native_hc_s42.json').exists()
assert all((ROOT/p).exists() for p in ['main_paper_MAKE51.pdf','MAKE51_revision_notes.md','MAKE51_response_to_reviewers.md'])
payload={str(p.relative_to(ROOT)):p.read_bytes() for p in paths}
payload['README.md']=(ROOT/'src/make51/SUPPLEMENT_README.md').read_bytes()
manifest=''.join(f'{hashlib.sha256(content).hexdigest()}  {name}\n' for name,content in sorted(payload.items()))
payload['MANIFEST.sha256']=manifest.encode()
out=ROOT/'MAKE51_reproducibility.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,content in sorted(payload.items()):
        entry=zipfile.ZipInfo(name,date_time=(2026,9,19,0,0,0))
        entry.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(entry,content)
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    for line in z.read('MANIFEST.sha256').decode().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
print(f'{out.name}: {len(payload)} files, {out.stat().st_size/1024**2:.1f} MiB; every manifest hash verified.')
