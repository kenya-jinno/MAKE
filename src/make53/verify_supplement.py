"""Extract the delivered archive and execute its no-training rebuild instructions."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
scratch = ROOT / "tmp/pdfs/MAKE53"
scratch.mkdir(parents=True, exist_ok=True)
target = Path(tempfile.mkdtemp(prefix="supplement_", dir=scratch))
archive = ROOT / "MAKE53_reproducibility.zip"
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for line in z.read("MANIFEST.sha256").decode().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(z.read(name)).hexdigest() == digest
    z.extractall(target)
print(f"Extracted {archive.name} to {target}", flush=True)
names = [str(p.relative_to(target)) for version in ["make51", "make52", "make53"]
         for p in (target / "results" / version).glob("*.tex")]
names += ["main_paper_MAKE53.tex", "main_paper_NN53.tex",
          "results/make51/search_replay.csv", "results/make51/search_sensitivity.csv",
          "results/make51/au_predictions.csv",
          "results/make52/manuscript_summary.json",
          "results/make53/native_pruning_summary.json"]
before = {n: (target / n).read_bytes() for n in names}
env = dict(os.environ)
env["MPLCONFIGDIR"] = str(target / ".mplcache")
env["XDG_CACHE_HOME"] = str(target / ".cache")
steps = ["src/make51/rebuild_review.py", "src/make51/replay_search.py",
         "src/make51/summarize_validation.py", "src/make52/summarize_revision.py",
         "src/make53/summarize_native_pruning.py", "src/make53/build_manuscript.py",
         "src/nn53/assemble_blocks.py", "src/nn53/render_figures.py",
         "src/nn53/build_japanese.py"]
for i, name in enumerate(steps):
    with (target / f"rebuild_{i+1}.log").open("w") as log:
        p = subprocess.run([sys.executable, name], cwd=target, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    assert p.returncode == 0, (name, target / f"rebuild_{i+1}.log")
    print(f"Passed: {name}", flush=True)
changed = [n for n, data in before.items() if (target / n).read_bytes() != data]
assert not changed, changed
record = {"archive_integrity_verified": True,
          "extracted_rebuild_commands_passed": steps,
          "byte_identical_rebuilt_tables_manuscripts_and_key_records": sorted(names),
          "model_retraining_performed": False}
(ROOT / "results/make53/supplement_verification.json").write_text(json.dumps(record, indent=2))
print(f"Verified {len(names)} rebuilt tables/manuscripts/key records, all byte-identical.")
