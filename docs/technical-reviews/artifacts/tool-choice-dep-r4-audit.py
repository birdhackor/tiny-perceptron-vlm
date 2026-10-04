"""Fresh R.4 dependency audit: no training and no notebook re-execution."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from scripts.export_course import checked_notebook, joined

torch.set_num_threads(1)
assert torch.version.cuda is None
ART = ROOT / "docs/technical-reviews/artifacts"


def load(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class Reading(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "h2":
            self.active = a.get("id") == "R.4"
        if self.active and tag == "a" and a.get("href"):
            self.links.append(a["href"])

    def handle_data(self, text):
        if self.active:
            self.text.append(text)

    def handle_endtag(self, tag):
        if tag == "article":
            self.active = False


result = {
    "reviewer_task": "/root/technical_dep_r4",
    "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
    "scope": "Source inspections, all current executed-copy/source comparisons, one selected public download and CPU inference. No 226-kernel rerun, no training, no GPU, no exhaustive quality reassessment.",
}
original = ROOT / "outputs/reviewer-tools/tool-choice-dep-r4-original"
result["original_section"] = {
    "extraction": json.loads((original / "extraction.json").read_text()),
    "execution": json.loads((original / "execution.json").read_text()),
}
assert result["original_section"]["execution"]["executed"] is False
for name in ("section.md", "extraction.json", "execution.json"):
    (ART / ("tool-choice-dep-r4-original-" + name)).write_bytes((original / name).read_bytes())

index = load("course/lesson-index.json")
assert len(index) == 226
copies = []
for item in index:
    source = ROOT / item["notebook"]
    executed = ROOT / "outputs/tool-choice-site-kernels" / source.relative_to(ROOT / "notebooks")
    notebook = checked_notebook(json.loads(source.read_text()), executed)
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    rendered = (ROOT / "outputs/zensical/docs" / (item["id"] + ".md")).read_text()
    labelled = [c for c in code if not c.get("metadata", {}).get("course_setup") and not c.get("metadata", {}).get("course_figure") and c.get("outputs")]
    assert rendered.count("實際執行結果 · CPU") == len(labelled)
    copies.append({"lesson": item["id"], "source_sha256": sha(source), "executed_sha256": sha(executed), "code_cells": len(code), "cpu_labelled_output_cells": len(labelled)})
result["notebook_copy_audit"] = {"input": "outputs/tool-choice-site-kernels", "count": len(copies), "fresh_kernel_executions_in_this_audit": 0, "entries": copies}
build = load("outputs/site/build-info.json")
assert build["lessons"] == len(index) and build["executed_cpu_outputs"] is True
home = (ROOT / "outputs/zensical/docs/index.md").read_text()
assert f"全部 {len(index)} 個小節" in home
page = Reading()
page.feed((ROOT / "outputs/site/course.html").read_text())
text = "".join(page.text)
for phrase in ("CPU 上驗證這段例子", "沒有接續原訓練所需的完整狀態", "不能推成模型像人一樣理解", "固定訓練資料包", "還得比較正常圖片和錯配圖片的回答"):
    assert phrase in text, phrase
for target in ("training.html", "training.html#T.3", "3.1.html", "15.4.html", "2.1.html", "training-assets.html"):
    assert target in page.links, target
assert any("docs/course-experiments/README.md" in target for target in page.links)
result["site"] = {"build_info": build, "course_page_sha256": sha("outputs/site/course.html"), "R.4_links": page.links, "R.4_visible_text": text.strip(), "home_count": len(index)}

plan = load("docs/course-experiments/plan.json")
reports = []
for item in plan["sequence"]:
    r = load(f"docs/course-experiments/results/{item['id']}.json")
    assert r["evidence_status"] == "complete_run" and r["step_scale"] == 1
    assert not r.get("unfinished_schedules")
    reports.append({"id": item["id"], "report_sha256": sha(f"docs/course-experiments/results/{item['id']}.json"), "device": r["device"], "seed": r["seed"], "code_fingerprints": len(r["code_sha256"]), "result_fields": list(r["results"]), "evidence_status": r["evidence_status"]})
assert len(reports) == 30
result["formal_report_inventory"] = reports
simple = load("docs/course-experiments/results/simple_models.json")["results"]
assert [simple["data"][s]["records"] for s in ("train", "validation", "test")] == [9, 1, 2]
assert all(run["steps"] == 200 for run in simple["runs"].values())
projector = load("docs/course-experiments/results/projector.json")["results"]
assert projector["training"]["steps"] == 300
assert projector["test"]["correct"] == 0 and projector["test"]["examples"] == 6
assert len(projector["test"]["samples"]) == 6
tool = load("docs/course-experiments/results/tool_choice.json")["results"]
assert tool["training"]["steps"] == 900
assert tool["test"]["metrics"]["accuracy"]["numerator"] == tool["test"]["metrics"]["accuracy"]["denominator"] == 96
assert tool["paraphrase_diagnostic"]["metrics"]["needed_tool_selected"]["numerator"] == 0
result["manually_inspected_report_examples"] = {
    "simple_models": {"split_records": [9, 1, 2], "runs": {k: {"steps": v["steps"], "before_nll": v["before_nll"], "after_nll": v["after_nll_same_post_update_time"]} for k,v in simple["runs"].items()}, "scope": simple["scope"]},
    "projector": {"updates": 300, "test_correct": 0, "test_examples": 6, "all_six_failed_answers_saved": True, "first_sample": projector["test"]["samples"][0], "scope": projector["scope"]},
    "tool_choice": {"updates": 900, "effective_targets": tool["training"]["effective_tokens"], "test_accuracy": tool["test"]["metrics"]["accuracy"], "new_wording_needed_tool_selected": tool["paraphrase_diagnostic"]["metrics"]["needed_tool_selected"], "scope": tool["limitations"]},
}
result["kernel_probe_exception"] = {"id": "flash_probe", "experiment_kind": load("docs/course-experiments/results/flash_probe.json")["experiment_kind"], "scope": "Separately identified numerical-kernel supplement; fixture is not trained weights. Not counted as an independent training-quality experiment."}

assets = load("assets/training/manifest.json")["assets"]
result["training_assets"] = []
for a in assets:
    upstream = load(a["source_metadata"])
    declared = upstream["license"]
    identifier = declared["identifier"] if isinstance(declared, dict) else declared
    assert identifier == a["license"]
    result["training_assets"].append({"id": a["id"], "training_records": a["training_records"], "declared_license": a["license"], "source_metadata_sha256": sha(a["source_metadata"]), "archive_sha256": a["archive_sha256"]})
assert len(assets) == 8
assert len({a["license"] for a in assets}) > 1
assert "Permission is hereby granted" in (ROOT / "LICENSE").read_text()
assert "不是已訓練權重" in (ROOT / "course/README.md").read_text()

commands = [
    [sys.executable, "scripts/fetch_course_models.py", "--model", "text_foundation", "--output", "outputs/reviewer-tools/tool-choice-dep-r4-download"],
    [sys.executable, "scripts/infer.py", "outputs/reviewer-tools/tool-choice-dep-r4-download/text_foundation/model.pt", "--prompt", "color=blue;shape=circle;", "--tokens", "32", "--device", "cpu", "--json"],
]
result["selected_model_commands"] = []
for command in commands:
    p = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=45)
    result["selected_model_commands"].append({"command": command, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
    assert p.returncode == 0, p.stderr
model_spec = next(m for m in load("docs/course-experiments/public-models.json")["models"] if m["id"] == "text_foundation")
model_root = ROOT / "outputs/reviewer-tools/tool-choice-dep-r4-download/text_foundation"
for f in model_spec["files"]:
    p = model_root / f["output"]
    assert sha(p) == f["sha256"] and p.stat().st_size == f["bytes"]
payload = torch.load(model_root / "model.pt", map_location="cpu", weights_only=True)
for forbidden in ("optimizer", "torch_rng", "python_rng", "cuda_rng", "mps_rng", "training_state", "step"):
    assert forbidden not in payload
generated = json.loads(result["selected_model_commands"][-1]["stdout"])
assert generated["answer"] == "side=right." and generated["eos"] and not generated["invalid_special_tokens"]
result["selected_student_model"] = {"id": "text_foundation", "public_revision": model_spec["revision"], "files_verified": len(model_spec["files"]), "payload_keys": sorted(payload), "missing_resume_state": True, "actual_generation": generated, "quality_scope": "Prompt supplies color and shape but no side target; this is load/generate/stop evidence, not correctness evidence."}

v1, v2 = torch.tensor([1.,2.,3.]), torch.tensor([3.,2.,1.])
image_ignoring_readout = torch.zeros((1,3))
assert v1.shape == v2.shape == (3,)
assert torch.equal(image_ignoring_readout @ v1, image_ignoring_readout @ v2)
result["dimension_counterexample"] = {"vectors": [v1.tolist(),v2.tolist()], "size": 3, "readout": image_ignoring_readout.tolist(), "outputs": [(image_ignoring_readout @ v).tolist() for v in (v1,v2)], "inference": "Matching dimensions permit multiplication; zero coefficients can ignore images entirely. Controlled image changes test dependence, not human-like understanding."}

fingerprints = ["scripts/export_course.py", "scripts/build_course.py", "scripts/check_notebooks.py", "scripts/fetch_course_models.py", "scripts/course_release.py", "scripts/infer.py", "tiny_perceptron/model.py", "tiny_perceptron/training.py", "tiny_perceptron/modal_data.py", "course/training.md", "course/chapters/03.md", "course/chapters/15.md", "course/chapters/02.md", "assets/training/README.md", "assets/training/manifest.json", "docs/course-experiments/README.md", "docs/course-experiments/plan.json", "docs/course-experiments/public-models.json", "LICENSE", "pyproject.toml"]
result["inspected_dependency_fingerprints"] = {p: sha(p) for p in fingerprints}
(ART / "tool-choice-dep-r4-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status":"passed", "notebook_copy_comparisons":len(copies), "formal_report_inventory":len(reports), "downloaded_model_files_verified":len(model_spec["files"]), "cpu_answer":generated["answer"], "kernel_reruns":0, "training_updates":0},ensure_ascii=False))
