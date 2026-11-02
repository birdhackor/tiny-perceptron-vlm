"""Same reviewer's immutable fix11 callback procedure for complete lesson 9.18."""

import contextlib
import difflib
import hashlib
import importlib.util
import io
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
REPORT = "docs/course-revision-20261029-omni/reviews/technical/technical_9_18/technical-original-v1.json"
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
ORIGINAL_SHA = "f47129150de2b57d0628abcb3d895ff55aebb4f9298228b37ea154d4f8db9a0f"
KERNEL = "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix11-cumulative.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(code):
    output, namespace = io.StringIO(), {}
    with contextlib.redirect_stdout(output):
        exec(compile(code, "current-9.18-inline.py", "exec"), namespace)
    return namespace, output.getvalue()


assert sha(ROOT / REPORT) == ORIGINAL_SHA
original = json.loads((ROOT / REPORT).read_text())
prior_run = json.loads((A / "execution.json").read_text())
freeze = json.loads((ROOT / FREEZE).read_text())
pages = {p["page_id"]: p for p in freeze["pages"]}
page = pages["9.18"]
snapshot = ROOT / page["snapshot"]
assert sha(snapshot) == page["source_sha256"]
source = snapshot.read_text()
old_source = (ROOT / prior_run["source_binding"]["snapshot"]).read_text()
assert page["figures_sha256"] == {}
assert re.search(r"^## 9\.18 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S)[0] == source
code = re.findall(r"```python\n(.*?)\n```", source, re.S)[0]
old_code = re.findall(r"```python\n(.*?)\n```", old_source, re.S)[0]
expected = re.findall(r"```text\n(.*?)\n```", source, re.S)[0]
assert code == old_code
assert expected == re.findall(r"```text\n(.*?)\n```", old_source, re.S)[0]
assert hashlib.sha256(code.encode()).hexdigest() == prior_run["source_binding"]["python_fence_sha256"]
new_clause = "同生成器只換seed，仍是同來源的重新抽樣；只有另按模板或組合分組，並排除各份資料之間的重疊，才可稱相應的模板或組合留出。"
assert new_clause in source
missing_quotes = [c["id"] for c in original["claims"] if c["source_quote"] not in source]
assert missing_quotes == ["c22"]
assert old_source.replace("要聲稱適用新來源，須真的留出不同來源；同生成器換seed只算模板或組合留出。", "要聲稱適用新來源，須真的留出不同來源。" + new_clause) == source
prior_artifacts = []
for artifact in original["artifacts"]:
    actual = sha(ROOT / artifact["path"])
    assert actual == artifact["sha256"]
    prior_artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})
prerequisites = []
for key, prior in prior_run["prerequisite_bindings"].items():
    p = pages[key]
    assert p["source_sha256"] == prior["sha256"] == sha(ROOT / p["snapshot"])
    for figure, digest in p["figures_sha256"].items():
        assert sha(ROOT / figure) == digest
    prerequisites.append({"lesson": key, "snapshot": p["snapshot"], "sha256": p["source_sha256"], "unchanged_since_complete_original_read": True, "figures_sha256": p["figures_sha256"]})
original_receipts = json.loads((A / "source-receipts.json").read_text())
source_checks = []
for item in original_receipts["originals"]:
    p = Path(item["cache_path_git_external"])
    assert sha(p) == item["sha256"]
    if item.get("text_file"):
        assert sha(p.parent / item["text_file"]) == item["text_sha256"]
    source_checks.append({"id": item["id"], "url": item["url"], "version": item["version"], "sha256": item["sha256"], "locator": item["locator"], "unchanged": True})
render = json.loads((A / "render-receipt.json").read_text())
for image in render["rendered"]:
    assert sha(Path(image["png_git_external"])) == image["png_sha256"]
namespace, stdout = run(code)
assert stdout.strip() == expected and stdout == prior_run["lesson_stdout"]
assert namespace["T"] == 1.0 and namespace["tau"] == 0.9
results = {"policy": namespace["evaluate"](namespace["policy"], 0.9), **{name: namespace["evaluate"](rows, 0.9) for name, rows in namespace["final"].items()}, "全部final": namespace["evaluate"](sum(namespace["final"].values(), []), 0.9), "零接受": namespace["evaluate"](namespace["policy"], 1.0)}
assert {k: list(v) for k, v in results.items()} == prior_run["original_results"]
exercise_from = "row([4., 0., -2.], 0, info_pred=True, info_gold=False)"
assert code.count(exercise_from) == 1
exercise, exercise_stdout = run(code.replace(exercise_from, "row([4., 0., -2.], 0, info_pred=False, info_gold=False)"))
exercise_results = {"新措辭": exercise["evaluate"](exercise["final"]["新措辭"], exercise["tau"]), "全部final": exercise["evaluate"](sum(exercise["final"].values(), []), exercise["tau"])}
assert {k: list(v) for k, v in exercise_results.items()} == prior_run["exercise_results"]
assert exercise["T"] == 1.0 and exercise["tau"] == 0.9
official = next(x for x in original_receipts["originals"] if x["id"] == "s-kev")
spec = importlib.util.spec_from_file_location("callback_pinned_metrics", official["cache_path_git_external"])
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
import numpy as np

restored = metrics.raw_row({"p": namespace["softmax"]([4, 0, -2], 2), "logits": [2, 0, -1], "inference_temperature": 2})
assert restored["logits"] == [4.0, 0.0, -2.0]
assert np.allclose(restored["p"], namespace["softmax"]([4, 0, -2]), atol=1e-14, rtol=0)
assert metrics.evaluate_threshold([0.99, 0.9], [True, False], None)["risk"] is None
notebook = ROOT / page["notebook"]
assert sha(notebook) == page["notebook_sha256"]
nb = json.loads(notebook.read_text())
assert sum("".join(c["source"]).strip() == code.strip() for c in nb["cells"] if c["cell_type"] == "code") == 1
kernel = json.loads((ROOT / KERNEL).read_text())
assert kernel["source_freeze"] == {"path": FREEZE, "sha256": sha(ROOT / FREEZE)}
record = next(r for r in kernel["records"] if r["lesson"] == "9.18")
assert record["notebook_sha256"] == sha(notebook)
assert sha(Path(record["source"])) == sha(Path(record["bundle_path"])) == record["executed_sha256"]
executed = json.loads(Path(record["bundle_path"]).read_text())
assert [c["source"] for c in executed["cells"]] == [c["source"] for c in nb["cells"]]
matching_cells = [c for c in executed["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip() == code.strip()]
assert len(matching_cells) == 1
cell = matching_cells[0]
assert cell["execution_count"] is not None
assert not any(o.get("output_type") == "error" for c in executed["cells"] for o in c.get("outputs", []))
kernel_stdout = "".join("".join(o["text"]) for o in cell["outputs"] if o["output_type"] == "stream" and o["name"] == "stdout")
assert kernel_stdout == stdout
cv = Path(next(x for x in original_receipts["originals"] if x["id"] == "s-cv")["cache_path_git_external"]).read_text().splitlines()
decision_path = ROOT / "docs/course-revision-20261029-omni/reviews/technical-author-decisions-fix09.json"
receipt = {
    "created_at": datetime.now(UTC).isoformat(), "reviewer_task": original["reviewer_task"],
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-18-v1/callback-verification-v1.py",
    "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "device": "CPU", "platform": platform.platform(), "torch": "lesson and reviewer rerun do not use torch; saved author kernel setup imported torch"},
    "source_binding": {"freeze": FREEZE, "freeze_sha256": sha(ROOT / FREEZE), "snapshot": page["snapshot"], "source_sha256": sha(snapshot), "notebook": page["notebook"], "notebook_sha256": sha(notebook), "python_fence_sha256": hashlib.sha256(code.encode()).hexdigest(), "figure_sha256": {}},
    "original_report": {"path": REPORT, "sha256": ORIGINAL_SHA, "unchanged": True},
    "full_source_diff": "".join(difflib.unified_diff(old_source.splitlines(True), source.splitlines(True), fromfile="reviewed-fix08", tofile="current-fix11")),
    "changed_claim_ids": missing_quotes, "current_corrected_quote": new_clause,
    "original_artifacts_rehashed": prior_artifacts, "prerequisites_rehashed": prerequisites,
    "original_source_fingerprints_rechecked": source_checks,
    "original_evidence_reinspection": {"s-cv": "Read again lines 60–81, 310–330, 626–658 and 747–763: final-test leakage, same generative process, dependent groups and explicit group disjointness.", "s-guo": "Read again §4 opening and §4.2 Eq.9/positive T: held-out same-distribution fitting and within-query argmax only.", "s-kev": "Read again risk/coverage 103–164, raw_row 232–243, fit-temperature/weights 269–294 and groups 297–330; the source-stratified fold still includes each source.", "s-selective": "Read again §2 and §3 empirical/population risk, i.i.d. finite-sample bound assumptions and §5 selection/test separation.", "s-kev-readme": "Read again lines48,374–375 to confirm actual unseen-source provenance; line450 blanket cross-confidence-order claim remains rejected."},
    "small_original_excerpt": {"source_id": "s-cv", "locator": "1.7.2 cross_validation.rst lines 639–642", "quote": "\n".join(cv[638:642])},
    "lesson_stdout": stdout, "results": results, "exercise_results": exercise_results,
    "exercise_changed_output_lines": [line for line in exercise_stdout.splitlines() if line.startswith(("新措辭", "全部final"))],
    "official_software_rerun": {"restored_logits": restored["logits"], "zero_accepted_risk": None},
    "saved_kernel_inspection": {"manifest": KERNEL, "manifest_sha256": sha(ROOT / KERNEL), "record": record, "all_current_cell_sources_match": True, "main_cell_execution_count": cell["execution_count"], "main_stdout_matches_current_rerun": True, "no_saved_cell_error": True, "scope": "Only lesson9.18 inspected; not an independent audit of all331 entries."},
    "author_decision_seen_after_original": {"path": decision_path.relative_to(ROOT).as_posix(), "sha256": sha(decision_path), "json_pointer": "/items/0", "used_as_authority_for_verdict": False},
    "result": "All callback assertions passed: corrected scope, exact unchanged code/outputs, fresh full inline/exercise rerun, original-evidence fingerprints, current notebook cell match and saved lesson9.18 kernel output.",
    "scope": "Full current9.18 read by the same reviewer. This local CPU arithmetic and saved-kernel inspection do not establish real model quality, OOD transfer or confidence-interval coverage. No other reviewer or reader report used. Prior prerequisite render/view evidence is reused only after SVG/PNG and artifact fingerprints were rechecked."
}
out = A / "callback-execution-v1.json"
with out.open("x") as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"artifact": out.relative_to(ROOT).as_posix(), "sha256": sha(out), "result": receipt["result"]}, ensure_ascii=False))
