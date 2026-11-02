"""本人 source15 technical callback：精確比較與收據，不重跑已驗的未改程式。"""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
REVIEWS = ROOT / "docs/course-revision-20261029-omni/reviews"
CACHE = Path("/workspace/work/reference/omni-technical-technical_5_7-v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section(raw, lesson):
    return re.search(rb"^## " + re.escape(lesson.encode()) + rb"\s.*?(?=^## |\Z)", raw, re.M | re.S).group()


original_path = REVIEWS / "technical/technical_5_7/technical-original-v1.json"
assert sha(original_path) == "a52bb4ddb0668b2a07bc485f75fdcf471b9c9c145f044f5b9b0c4241d23cc743"
original = json.loads(original_path.read_bytes())
old_freeze = json.loads((REVIEWS / "source-freeze-technical-fix14.json").read_bytes())
new_freeze_path = REVIEWS / "source-freeze-technical-fix15.json"
assert sha(new_freeze_path) == "1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"
new_freeze = json.loads(new_freeze_path.read_bytes())
old_page = next(p for p in old_freeze["pages"] if p["page_id"] == "5.7")
new_page = next(p for p in new_freeze["pages"] if p["page_id"] == "5.7")
old_source = (ROOT / old_page["snapshot"]).read_bytes()
new_source = (ROOT / new_page["snapshot"]).read_bytes()
assert hashlib.sha256(new_source).hexdigest() == new_page["source_sha256"]
assert section((ROOT / new_page["source"]).read_bytes(), "5.7") == new_source
old_quote = "只接受 tensor 與一般資料，不執行檔案裡夾帶的程式"
new_quote = "限制可還原的型別與可呼叫的函式，降低載入時執行任意程式的風險"
assert old_source.replace(old_quote.encode(), new_quote.encode(), 1) == new_source
assert "讀進來的檔案仍應來自可信的來源。".encode() in new_source
old_code = re.search(rb"```python\n(.*?)```", old_source, re.S).group(1)
new_code = re.search(rb"```python\n(.*?)```", new_source, re.S).group(1)
assert old_code == new_code
prior_execution = json.loads((ART / "execution.json").read_bytes())
assert hashlib.sha256(new_code).hexdigest() == prior_execution["embedded_sha256"]

receipt = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": original["reviewer_task"],
    "callback_kind": "initial technical source15 callback by original reviewer",
    "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": sha(original_path)},
    "freeze": {"path": str(new_freeze_path.relative_to(ROOT)), "sha256": sha(new_freeze_path)},
    "source": {"snapshot": new_page["snapshot"], "sha256": new_page["source_sha256"], "current_section_matches_snapshot": True},
    "only_change": {"before": old_quote, "after": new_quote, "exact_single_replacement": True, "trusted_source_condition_preserved": True},
    "code": {"unchanged": True, "sha256": hashlib.sha256(new_code).hexdigest(), "reuse_execution": {"path": str((ART / "execution.json").relative_to(ROOT)), "sha256": sha(ART / "execution.json")}, "original_both_assertions_passed": prior_execution["embedded"]["both_assertions_passed"], "original_exercise": prior_execution["exercise"], "new_training_or_gpu_run": False},
    "repository_sources": [],
    "prerequisites": [],
    "retained_artifacts": [],
    "notebooks": [],
}
for item in original["sources"]:
    if item["kind"] == "repository_code":
        actual = sha(ROOT / item["path"])
        assert actual == item["sha256"]
        receipt["repository_sources"].append({"path": item["path"], "sha256": actual, "unchanged": True})
for item in original["artifacts"]:
    assert sha(ROOT / item["path"]) == item["sha256"]
    receipt["retained_artifacts"].append({"path": item["path"], "sha256": item["sha256"]})
prior_inspection = json.loads((ART / "inspection.json").read_bytes())
for item in prior_inspection["prerequisites"]:
    path, lesson = item["source"].split("#")
    actual = hashlib.sha256(section((ROOT / path).read_bytes(), lesson)).hexdigest()
    assert actual == item["sha256"]
    receipt["prerequisites"].append({"source": item["source"], "sha256": actual, "unchanged": True})

official_doc = CACHE / "pytorch-docs_source_notes_serialization.md"
official_api = CACHE / "pytorch-torch_serialization.py"
assert sha(official_doc) == "d7bfc78aefe95ac8edd2972d0749b338e824736f67f8d55aa4fbd7e72fe14b92"
assert sha(official_api) == "4a25e7d5a44d55ce2578164451b3cfbb65b4bf1be8bdba2fd7b9e008d2cf9021"
receipt["primary_reread"] = {
    "pin": "5c4886908584029761b579af026dcfb627c84070",
    "version": "torch 2.14.1+cpu",
    "documentation": {"url": "https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/serialization.md", "cache": str(official_doc), "sha256": sha(official_doc), "locator": "255-310, especially weights_only security 264-278", "reread": True},
    "api": {"url": "https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/serialization.py", "cache": str(official_api), "sha256": sha(official_api), "locator": "torch.load 1315-1395; warning and weights_only argument", "reread": True},
    "finding": "新句的限制型別/函式與降低任意程式執行風險，符合官方 restricted unpickler 與 narrows the surface 的範圍；保留可信來源條件，沒有絕對保證。",
}

nb_path = ROOT / new_page["notebook"]
assert sha(nb_path) == new_page["notebook_sha256"]
notebook = json.loads(nb_path.read_bytes())
sources = [c["source"] for c in notebook["cells"]]
notebook_code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
assert new_code.decode().rstrip("\n") in [code.rstrip("\n") for code in notebook_code]
markdown = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
assert new_quote in markdown and old_quote not in markdown and "讀進來的檔案仍應來自可信的來源。" in markdown
receipt["notebooks"].append({"path": str(nb_path.relative_to(ROOT)), "sha256": sha(nb_path), "freeze_matches": True, "code_matches_source": True, "comparison_note": "Notebook cell omits only the fenced code's final newline; every code line is identical.", "corrected_prose_present": True, "old_absolute_claim_absent": True, "trusted_source_condition_present": True})
for path in ["outputs/omni-notebooks-technical-fix15-run/05/5.7.ipynb", "outputs/omni-notebooks-technical-fix15-current/05/5.7.ipynb"]:
    executed_path = ROOT / path
    executed = json.loads(executed_path.read_bytes())
    assert [c["source"] for c in executed["cells"]] == sources
    outputs = [o for c in executed["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])]
    assert all(o["output_type"] != "error" for o in outputs)
    stdout = "".join("".join(o["text"]) for o in outputs if o.get("name") == "stdout")
    assert stdout == prior_execution["embedded"]["stdout"]
    receipt["notebooks"].append({"path": path, "sha256": sha(executed_path), "provenance": "author-supplied kernel result independently inspected by reviewer; not reviewer execution", "all_sources_match_current_notebook": True, "execution_counts": [c.get("execution_count") for c in executed["cells"] if c["cell_type"] == "code"], "error_outputs": 0, "stdout": stdout, "stdout_equals_reviewer_original_execution": True})
kernel_path = ROOT / "outputs/omni-notebooks-technical-fix15-run/validation-kernel.json"
kernel = json.loads(kernel_path.read_bytes())
assert kernel["total"] == kernel["passed"] == 1 and kernel["failed"] == 0 and kernel["failures"] == []
receipt["author_kernel_manifest"] = {"path": str(kernel_path.relative_to(ROOT)), "sha256": sha(kernel_path), "raw_values": kernel, "provenance": "author result; parsed and checked, not reviewer execution"}
receipt["figures"] = {"source15": new_page["figures_sha256"], "source14": old_page["figures_sha256"], "reason": "both empty; no diagram to render/view"}
assert new_page["figures_sha256"] == old_page["figures_sha256"] == {}
receipt["stage_scope"] = "本件僅為原 reviewer 的 initial technical source15 callback；checker 排除原 packet 真 reader /root/rv1019_reader_ch05。proper reader02 → technical02 → coherence02 尚未到件，不聲稱新 reader 或 stage seal 完成。"
print(json.dumps(receipt, ensure_ascii=False, indent=2))
