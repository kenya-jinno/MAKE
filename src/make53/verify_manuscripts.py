"""Verify final PDF metadata/text and matching English/Japanese numbering."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
def aux(lang):
    return (ROOT / f"tmp/pdfs/{lang}53/build/main_paper_{lang}53.aux").read_text()
def label_numbers(s):
    return {k: n for k, n in re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", s)
            if not k.endswith("@cref") and k != "LastPage"}
a, b = label_numbers(aux("MAKE")), label_numbers(aux("NN"))
assert a == b and len(a) == 103
def cite_numbers(s):
    return dict(re.findall(r"\\bibcite\{([^}]+)\}\{\{(\d+)\}", s))
assert cite_numbers(aux("MAKE")) == cite_numbers(aux("NN"))
assert len(cite_numbers(aux("MAKE"))) == 30
record = {"matching_label_numbers": len(a), "matching_bibliography_numbers": 30, "pdfs": {}}
for lang, pages in [("MAKE", 45), ("NN", 52)]:
    stem = f"main_paper_{lang}53"
    log = (ROOT / f"tmp/pdfs/{lang}53/build/{stem}.log").read_text()
    assert not re.search(r"Overfull|Missing character|^!|There were undefined|Label\(s\) may have changed", log, re.M)
    assert not re.search(r"(?:Reference|Citation).*undefined", log)
    pdf = ROOT / (stem + ".pdf")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    assert int(re.search(r"Pages:\s+(\d+)", info)[1]) == pages
    text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    assert "??" not in text
    if lang == "NN":
        for name in ["小端", "千佳", "代", "美月", "神野", "健哉"]:
            assert name in text[:1400]
        assert "小幡" not in text and "瑞生" not in text
    record["pdfs"][pdf.name] = {
        "pages": pages, "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "no_overfull_missing_glyphs_or_unresolved_references": True}
ja = ROOT / "results/nn53/translation_verification.json"
translation = json.loads(ja.read_text())
translation.update({"label_numbers_match": len(a), "bibliography_numbers_match": 30,
                    "pdf_pages": 52, "pdf_sha256": record["pdfs"]["main_paper_NN53.pdf"]["sha256"],
                    "author_names_verified": ["小端 千佳", "代 美月", "神野 健哉"]})
ja.write_text(json.dumps(translation, ensure_ascii=False, indent=2))
(ROOT / "results/make53/manuscript_verification.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
