"""Localize MAKE52 figure labels without changing data or plotting operations."""
from pathlib import Path
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures/NN52"
FIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/pdfs/NN52/mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp/pdfs/NN52/cache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = subprocess.check_output(["kpsewhich", "ipaexg.ttf"], text=True).strip()
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=font_path).get_name(),
    "axes.unicode_minus": False,
})
for p in (ROOT / "results/figures/NN51").glob("*.pdf"):
    shutil.copy2(p, FIG / p.name)

arm = json.loads((ROOT / "results/make52/geco_arm_all.json").read_text())
ard = json.loads((ROOT / "results/make52/ard_native_all.json").read_text())
DS = ["MNIST", "FashionMNIST", "dSprites", "CIFAR10"]
NAMES = dict(zip(DS, ["MNIST", "Fashion-MNIST", "dSprites", "CIFAR-10"]))
source = (ROOT / "src/make52/summarize_revision.py").read_text()
code = source[source.index("plt.rcParams.update("):source.index('summary["record_checks"]')]
for en, ja in {
    "Training-pool subset": "学習用プールの部分集合",
    "Validation": "検証",
    "Epoch": "エポック",
    "Training constraint (SSE)": "学習制約（二乗誤差和）",
}.items():
    code = code.replace('"' + en + '"', '"' + ja + '"')
code = code.replace('f"seed ', 'f"シード ')
exec(compile(code, "MAKE52_localized_plot_instructions", "exec"))
assert len(list(FIG.glob("*.pdf"))) == 11
print("Japanese figure set ready: 8 unchanged NN51 figures and 3 localized MAKE52 figures.")
