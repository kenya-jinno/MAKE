"""Package the English and Japanese MAKE53 evidence snapshot and verify hashes."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[2]
patterns = [
    "config/experiment_config_MAKE49.json",
    "src/__init__.py", "src/make49/*.py", "src/metrics/*.py",
    "src/make51/*", "src/make52/*.py", "src/make52/*.tex", "src/make52/*.md",
    "src/make53/*", "src/nn51/*.py", "src/nn52/*", "src/nn53/*",
    "results/make49/*.json", "results/make49/stage4_cache/*.json",
    "results/make51/*", "results/make51/model_validation/*",
    "results/make52/*.json", "results/make52/*.tex", "results/make52_*.log",
    "results/make53/*.json", "results/make53/*.tex", "results/nn53/*.json",
    "results/figures/MAKE51/*.pdf", "results/figures/MAKE52/*.pdf",
    "results/figures/NN52/*.pdf", "results/figures/NN53/*.pdf",
    "Definitions/*", "main_paper_MAKE49.tex", "main_paper_MAKE51.tex",
    "main_paper_MAKE52.tex", "main_paper_NN51.tex", "main_paper_NN52.tex",
    "main_paper_MAKE53.tex", "main_paper_MAKE53.pdf",
    "main_paper_NN53.tex", "main_paper_NN53.pdf",
    "MAKE51_revision_notes.md", "MAKE51_response_to_reviewers.md",
    "MAKE52_revision_notes.md", "MAKE52_実験検証結果.md",
    "MAKE52_revision_review.md", "MAKE53_revision_notes.md",
    "MAKE49_実験設定表.md", "MAKE49_段階1_監査結果.md",
]
paths = sorted({p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file()})
payload = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
for required in ["main_paper_MAKE53.pdf", "main_paper_NN53.pdf",
                 "results/make53/native_pruning_summary.json",
                 "results/make52/native_pruned_spec.json"]:
    assert required in payload
payload["README.md"] = (ROOT / "src/make53/SUPPLEMENT_README.md").read_bytes()
payload["MANIFEST.sha256"] = "".join(
    f"{hashlib.sha256(data).hexdigest()}  {name}\n"
    for name, data in sorted(payload.items())).encode()
out = ROOT / "MAKE53_reproducibility.zip"
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for name, data in sorted(payload.items()):
        info = zipfile.ZipInfo(name, date_time=(2026, 9, 19, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, data)
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    for line in z.read("MANIFEST.sha256").decode().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
print(f"{out.name}: {len(payload)} files, {out.stat().st_size / 1024**2:.1f} MiB; all hashes verified.")
