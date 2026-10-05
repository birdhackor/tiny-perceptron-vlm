"""Independent bounded R.1 probe; never builds or installs the course."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
import time

import nbformat
from nbclient import NotebookClient
import torch

from scripts.build_course import notebook
from scripts.export_course import checked_notebook

ROOT = Path.cwd()
ART = ROOT / "docs/technical-reviews/artifacts/natural-v4-factual/R.1"
RAW = ROOT / "outputs/natural-v4/factual-research/R.1"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def section(path, lesson):
    text = path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## .+$", text, re.M))
    for i, h in enumerate(headings):
        if h[0].startswith("## " + lesson + " "):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            return text[h.start():end]
    raise ValueError(lesson)

body = section(ROOT / "course/README.md", "R.1")
intro = (ROOT / "course/README.md").read_text().split("## ", 1)[0]
assert hashlib.sha256(body.encode()).hexdigest() == "4a0e0a4bb3d5ce5c88db5dffaf511955a07bdb260b234ba4a9bdf967d2179c54"
assert hashlib.sha256(intro.encode()).hexdigest() == "66df1da0da24fa57bb5664b96b12f96c22dad6953789d64e05ac843e29474b2d"
(ART / "fresh-assigned-body.md").write_text(body)
(ART / "fresh-assigned-introduction.md").write_text(intro)
lesson = section(ROOT / "course/chapters/01.md", "1.1")
original = json.loads((ROOT / "notebooks/01/1.1.ipynb").read_text())
generated = notebook("1.1", "文字怎麼變成數字？", lesson.split("\n", 1)[1], ROOT / "course/chapters/01.md")
assert original == generated
record = checked_notebook(original, ROOT / "outputs/notebooks/01/1.1.ipynb")
main = re.search(r"```python\n(.*?)```", lesson, re.S)[1]
assert main.rstrip() == "".join(original["cells"][3]["source"])
assert not any(w in main for w in ["Transformer", "cuda", "backward", "optimizer"])
result = {
    "environment": {"python": platform.python_version(), "executable": sys.executable,
                    "torch": torch.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available(),
                    "nbformat": nbformat.__version__, "platform": platform.platform()},
    "sources": {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "notebooks/01/1.1.ipynb", ROOT / "outputs/notebooks/01/1.1.ipynb", ROOT / "scripts/build_course.py", ROOT / "scripts/export_course.py", ROOT / "course/figures/character_ids.svg"]},
    "generator_matches_current_notebook": True,
    "original_record_inspection": {"own_execution": False, "source_cells_match_current": True,
        "code_execution_counts": [c["execution_count"] for c in record["cells"] if c["cell_type"] == "code"],
        "main_stdout": "".join(o["text"] for o in record["cells"][3]["outputs"] if o["output_type"] == "stream")},
    "toy_runs": [],
}
for name, code in [
    ("main", main),
    ("reverse_exercise", main.replace("chars = sorted(set(text))", "chars = sorted(set(text), reverse=True)")),
    ("W.1_changed_input", main.replace("貓看狗，狗看貓。", "鳥看狗，狗看鳥。")),
]:
    ns = {}; out = io.StringIO()
    with contextlib.redirect_stdout(out): exec(compile(code, "<"+name+">", "exec"), ns)
    assert ns["restored"] == ns["text"]
    if name == "main":
        assert ns["chars"] == ["。", "狗", "看", "貓", "，"]
        assert ns["ids"] == [3, 2, 1, 4, 1, 2, 3, 0]
        assert ns["ids"][0] == ns["ids"][6] == 3
        try: [ns["to_id"][c] for c in "貓看鳥"]
        except KeyError as e: result["unknown_character"] = {"exception": type(e).__name__, "key": e.args[0]}
        else: raise AssertionError("Missing expected KeyError")
    if name == "reverse_exercise": assert ns["ids"] == [1, 2, 3, 0, 3, 2, 1, 4]
    result["toy_runs"].append({"name":name,"text":ns["text"],"characters":len(ns["text"]),"unique_characters":len(ns["chars"]),
                              "chars":ns["chars"],"ids":ns["ids"],"restored":ns["restored"],"stdout":out.getvalue(),"assertion_passed":True})
warm = section(ROOT / "course/first-steps.md", "W.2")
warm_code = re.search(r"```python\n(.*?)```", warm, re.S)[1]
out=io.StringIO()
with contextlib.redirect_stdout(out): exec(compile(warm_code,"<W.2>","exec"),{})
assert out.getvalue() == "貓 2\n貓在睡覺\n看到 貓\n看到 狗\n"
result["W.2_sample_stdout"]=out.getvalue()
nb=nbformat.read(ROOT / "notebooks/01/1.1.ipynb",as_version=4)
start=time.perf_counter()
NotebookClient(nb,timeout=60,kernel_name="tiny-perceptron",resources={"metadata":{"path":str(ROOT)}},allow_errors=False).execute()
nbformat.write(nb,RAW / "own-executed-1.1.ipynb")
own_main="".join(o["text"] for o in nb.cells[3].outputs if o.output_type == "stream")
assert own_main == result["toy_runs"][0]["stdout"]
result["own_notebook_execution"]={"kernel":"tiny-perceptron", "kernel_executable":"/workspace/tiny-perceptron-vlm/.venv/bin/python", "elapsed_seconds":time.perf_counter()-start,
    "cells_total":len(nb.cells),"code_cells":sum(c.cell_type == "code" for c in nb.cells),
    "execution_counts":[c.execution_count for c in nb.cells if c.cell_type == "code"], "main_stdout":own_main,
    "errors":[],"figure_output_mimetypes":list(nb.cells[5].outputs[0].data)}
print(json.dumps(result,ensure_ascii=False,indent=2))
(ART / "fresh-cpu-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
