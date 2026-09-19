"""Reuse unchanged NN52 paragraphs and apply reviewed MAKE53 translations."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
def template(v):
    s = (ROOT / f"src/make{v}/manuscript_body.tex").read_text().strip()
    return re.sub(r"@@section:(\w+)@@",
        lambda m: (ROOT / f"src/make{v}" / (m[1] + ".tex")).read_text().strip(), s).split("\n\n")
def blocks(p):
    a = re.split(r"^@@block:(\d+)@@\n", p.read_text(), flags=re.M)
    return {int(a[i]): a[i+1].strip() for i in range(1, len(a), 2)}
old, new = template(52), template(53)
prior = blocks(ROOT / "src/nn52/blocks_ja.txt")
lookup = {x: prior[i] for i, x in enumerate(old)}
overrides = blocks(ROOT / "src/nn53/new_blocks_ja.txt")
result = {}
for i, s in enumerate(new):
    if i in overrides:
        result[i] = overrides[i]
    elif s in lookup:
        result[i] = lookup[s]
    elif s.replace("MAKE53", "MAKE52") in lookup:
        result[i] = lookup[s.replace("MAKE53", "MAKE52")].replace("MAKE52\\_reproducibility", "MAKE53\\_reproducibility")
    elif re.fullmatch(r"@@table53:\w+@@", s):
        name = s[len("@@table53:"):-2]
        t = (ROOT / "results/make53" / (name + ".tex")).read_text()
        result[i] = "@@table:" + re.search(r"\\label\{tab:([^}]+)\}", t)[1] + "@@"
    else:
        raise ValueError(f"Missing Japanese block {i}: {s[:80]}")
assert len(new) == 166 and len(result) == 166
(ROOT / "src/nn53/blocks_ja.txt").write_text("\n\n".join(
    f"@@block:{i:03d}@@\n{result[i]}" for i in range(len(new))) + "\n")
print(f"Japanese blocks: {len(new)}; explicit revisions: {len(overrides)}")
