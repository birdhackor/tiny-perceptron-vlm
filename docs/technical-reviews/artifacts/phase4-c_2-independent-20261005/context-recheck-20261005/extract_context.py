"""Snapshot raw context bytes and compare this reviewer's own frozen input."""
import difflib
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = HERE.parent / "inputs/0C-frozen.md"
CURRENT = ROOT / "course/chapters/0C.md"
def sha(b): return hashlib.sha256(b).hexdigest()
def section(raw, lesson):
    hs = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    i = next(i for i,h in enumerate(hs) if h[0].startswith(f"## {lesson} ".encode()))
    start,end = hs[i].start(), hs[i+1].start() if i+1<len(hs) else len(raw)
    return raw[start:end], raw[:start].count(b"\n")+1

old,new = OLD.read_bytes(), CURRENT.read_bytes()
(HERE / "0C-current-context-frozen.md").write_bytes(new)
facts = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_c_2",
    "newline_policy": "raw UTF-8 bytes; no stripping or normalization",
    "original_wholefile_frozen_input": {"path": str(OLD.relative_to(ROOT)), "sha256": sha(old)},
    "current_wholefile_context_frozen_input": {"path": str((HERE / "0C-current-context-frozen.md").relative_to(ROOT)), "sha256": sha(new)},
    "selected_sections": {},
    "context_section_byte_identity": {},
}
for lesson in ["C.1", "C.2", "C.3", "C.4", "C.5", "C.6", "C.7"]:
    ob,_=section(old,lesson); nb,line=section(new,lesson)
    facts["context_section_byte_identity"][lesson] = {"identical":ob==nb,"old_sha256":sha(ob),"current_sha256":sha(nb)}
    if lesson in ["C.2", "C.4"]:
        target = HERE / f"{lesson}-current.md"; target.write_bytes(nb)
        old_target=HERE / f"{lesson}-original-context.md";old_target.write_bytes(ob)
        facts["selected_sections"][lesson] = {"current_path":str(target.relative_to(ROOT)),"current_sha256":sha(nb),"original_path":str(old_target.relative_to(ROOT)),"original_sha256":sha(ob),"current_first_line":line}
        print(f"CURRENT {lesson} first line {line}")
        print(nb.decode("utf-8"))
    if lesson == "C.4":
        d="".join(difflib.unified_diff(ob.decode("utf-8").splitlines(keepends=True),nb.decode("utf-8").splitlines(keepends=True),fromfile="own original frozen C.4",tofile="current C.4"))
        (HERE / "C.4-own-frozen-vs-current.diff").write_text(d,encoding="utf-8")
        print(d)
assert facts["selected_sections"]["C.2"]["current_sha256"] == "959cfac0b9ca1a19aeede2abb6c9186de3df41a3ec2cb11249273ceb35676c2b"
assert facts["original_wholefile_frozen_input"]["sha256"] == "d9959533b8acd624fcc8d20120bb7830905f60737f92f78f200d37d582d9f577"
(HERE / "context-input-facts.json").write_text(json.dumps(facts,ensure_ascii=False,indent=2)+"\n")
