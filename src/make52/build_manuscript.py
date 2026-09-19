"""Build MAKE52 from its text template and verified saved-record tables."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]
old = (ROOT / "main_paper_MAKE51.tex").read_text()
pre = old[:old.index(r"\abstract")]
pre = re.sub(r"\\Title\{.*?\}", lambda _: r"""\Title{Intrinsic-Dimension-Guided Bottleneck Selection for
Variational Autoencoders with Active-Unit Diagnostics}""", pre, count=1, flags=re.S)
pre = pre.replace("% MDPI MAKE51: revised against MAKE50_revision_review.md",
                  "% MDPI MAKE52: revision integrating additional experimental records.")
pre = pre.replace("% MAKE51: record-audited revision dated 2026-09-19.",
                  "% MAKE52: title continuity and component-verified revision, 2026-09-19.")
abstract = (ROOT / "src/make52/abstract.tex").read_text()
body = (ROOT / "src/make52/manuscript_body.tex").read_text()
body = re.sub(r"@@section:(\w+)@@",
              lambda m: (ROOT / "src/make52" / (m[1] + ".tex")).read_text(), body)
for prefix, directory in [("table", "make51"), ("table52", "make52")]:
    body = re.sub(r"@@" + prefix + r":(\w+)@@",
                  lambda m: (ROOT / "results" / directory / (m[1] + ".tex")).read_text(), body)
a = json.loads((ROOT / "results/make51/au_paired.json").read_text())
mapping = {
    "AU_C": f"{a['C']:g}", "AU_DIFF": f"{a['paired_difference']:+.3f}",
    "AU_BASE_DIFF": f"{a['mean_delta_au']:+.3f}",
    "AU_BASE_LO": f"{a['delta_au_hierarchical_ci'][0]:+.3f}",
    "AU_BASE_HI": f"{a['delta_au_hierarchical_ci'][1]:+.3f}",
    "PYTHON": a["environment"]["python"], "NUMPY": a["environment"]["numpy"],
    "SKLEARN": a["environment"]["sklearn"],
    "AU_LO": f"{a['paired_hierarchical_ci'][0]:+.3f}",
    "AU_HI": f"{a['paired_hierarchical_ci'][1]:+.3f}",
    "AU100_DIFF": f"{a['noise100_difference']:+.3f}",
    "AU100_LO": f"{a['noise100_hierarchical_ci'][0]:+.3f}",
    "AU100_HI": f"{a['noise100_hierarchical_ci'][1]:+.3f}",
}
mapping.update(json.loads((ROOT / "results/make51/validation_text.json").read_text()))
for k, v in mapping.items():
    body = body.replace("@@" + k + "@@", v)
assert "@@" not in body
refs = old[old.index(r"\begin{thebibliography}"):old.index(r"\end{thebibliography}")]
entries = {m[1]: m[2].strip() for m in re.finditer(
    r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)", refs, re.S)}
order = []
for match in re.finditer(r"\\cite\{([^}]+)\}", body):
    for k in match[1].split(","):
        if k not in order:
            order.append(k)
bib = "\n\\reftitle{References}\n\\begin{thebibliography}{999}\n"
for k in order:
    bib += "\n\\bibitem{" + k + "}\n" + entries[k] + "\n"
bib += "\n\\end{thebibliography}\n\\end{document}\n"
text = pre + abstract + "\n\\begin{document}\n" + body + bib
(ROOT / "main_paper_MAKE52.tex").write_text(text)
print(f"MAKE52: {len(text.splitlines())} lines, {len(order)} references")
