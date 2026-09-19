"""Reuse the verified Japanese figures: MAKE53 changes no figure data."""
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "results/figures/NN53"
out.mkdir(parents=True, exist_ok=True)
files = sorted((ROOT / "results/figures/NN52").glob("*.pdf"))
assert len(files) == 11
for p in files:
    shutil.copyfile(p, out / p.name)
print("Copied 11 unchanged, previously localized figure PDFs.")
