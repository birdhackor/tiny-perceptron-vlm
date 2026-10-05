"""Create concise primary-source snapshots, preserving original acquisition hashes."""
import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
TMP = Path("/tmp/phase4-14_9-primary")
specs = [
    ("roformer", "2104.09864v5", [1, 4, 5], [(1,43),(210,292)],
     "RoFormer: Enhanced Transformer with Rotary Position Embedding", "2023-11-08 arXiv v5; PDF header November 9, 2023",
     "First-page arXiv version stamp; §3.2.1–3.2.2 Eqs.12–16 and p.5 block rotation matrix.") ,
    ("yarn", "2309.00071v3", [1, 3, 4, 5], [(1,25),(113,164),(173,289)],
     "YaRN: Efficient Context Window Extension of Large Language Models", "2026-02-06 arXiv v3",
     "First-page arXiv version stamp; §2.1 Eqs.5–7; §2.2 Eq.8; §3.1–3.2 and Eqs.10–13."),
]
records = []
for name, arxiv, pages, spans, title, version, scope in specs:
    pdf = TMP / f"{name}-{arxiv}.pdf"
    text = TMP / f"{name}.txt"
    raw = pdf.read_bytes()
    pdfsha = hashlib.sha256(raw).hexdigest()
    lines = text.read_bytes().splitlines(keepends=True)
    # pdftotext -layout line numbers are the line-based locators used by sed;
    # split on LF only so form feed remains an actual source byte.
    lines = text.read_bytes().split(b"\n")
    snippets=[]
    for first,last in spans:
        path=OUT/f"{name}-original-lines-{first}-{last}.txt"
        snippet=b"\n".join(lines[first-1:last])+b"\n"
        path.write_bytes(snippet)
        snippets.append({"path":path.name,"original_pdftotext_lf_lines":[first,last],
                         "sha256":hashlib.sha256(snippet).hexdigest()})
    commands=[]
    selected=[]
    for page in pages:
        pattern=TMP/f"{name}-page-%d.pdf"
        command=["pdfseparate","-f",str(page),"-l",str(page),str(pdf),str(pattern)]
        subprocess.run(command,check=True,capture_output=True)
        commands.append(command)
        selected.append(str(TMP/f"{name}-page-{page}.pdf"))
    compact=OUT/f"{name}-primary-selected-pages.pdf"
    command=["pdfunite",*selected,str(compact)]
    subprocess.run(command,check=True,capture_output=True)
    commands.append(command)
    records.append({"title":title,"url":f"https://arxiv.org/pdf/{arxiv}",
                    "version":version,"accessed_on":"2026-10-05",
                    "original_acquired_pdf_sha256":pdfsha,"original_bytes":len(raw),
                    "original_pdf_path_at_acquisition":str(pdf),
                    "authority_reason":"Author-uploaded primary paper at its explicit arXiv version URL, personally checked first-page version and cited sections.",
                    "checked_original_scope":scope,
                    "full_pdf_not_retained_reason":"Permanent proof keeps the exact inspected sections and version page, avoiding unrelated paper content.",
                    "compact_pdf":compact.name,"compact_pdf_sha256":hashlib.sha256(compact.read_bytes()).hexdigest(),
                    "original_page_numbers_in_compact_order":pages,"commands":commands,"text_snapshots":snippets})
result={"records":records,
        "source_limit":"YaRN v3 §2.3 Eq.9 prints g(m)=s·m while defining s=L′/L. This review does not cite Eq.9 for compression; the section's stated /2 and /4 are independently derived from the explicit operation and RoFormer's mθ. YaRN §3.1–3.2 is used only for uniform-scaling tradeoffs and frequency-dependent design.",
        "locator_index_use":"Read keys/types and locator metadata only. Neither required paper was present; originals were acquired directly from explicit version URLs. Locator indices are not evidence."}
(OUT/"primary-source-provenance.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
