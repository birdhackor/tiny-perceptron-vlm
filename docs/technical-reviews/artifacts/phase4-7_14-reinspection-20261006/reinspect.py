"""Original technical owner: bounded verification of changed baseline claim.

Reuse previous real evidence only after exact hash checks. Do not train/load weights,
rerun unchanged fences, or read author review/correction notes from result JSON.
"""
import ast
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-7_14-independent"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


prior_path = BASE / "history/prior-report.json"
assert sha(prior_path) == "485867d9efbb911d2df70fb06530a156668bc76625311f1c91d1e9fa8858dd9f"
prior = json.loads(prior_path.read_bytes())  # Own prior report only, after opaque backup.
integrity = []
for item in prior["artifacts"]:
    digest = sha(ROOT / item["path"])
    assert digest == item["sha256"]
    integrity.append({"id": item["id"], "path": item["path"], "sha256": digest, "unchanged": True})
facts_path = ROOT / "docs/review-tools/section_facts.py"
ns = {}
tree = ast.parse(facts_path.read_bytes())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "original_section"]
import re
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(facts_path), "exec"), {"re": re}, ns)
section, chapter, line = ns["original_section"](ROOT / "course/chapters/07.md", "7.14")
context, _, ctx_line = ns["original_section"](ROOT / "course/chapters/07.md", "7.13")
assert hashlib.sha256(section).hexdigest() == "4869f3338eb523ab38d70d12589e2931553d0f4e8a8769704e9e72c1ce251a59"
inputs = BASE / "inputs"
inputs.mkdir(exist_ok=True)
(inputs / "current-section.md").write_bytes(section)
(inputs / "current-context-7.13.md").write_bytes(context)
(inputs / "current-frozen-chapter-07.md").write_bytes(chapter)
shutil.copyfile(facts_path, inputs / "current-section-facts.py")
shutil.copyfile(ROOT / "docs/review-tools/factual-reviewer-instructions.md", inputs / "current-factual-reviewer-instructions.md")
shutil.copyfile(ROOT / "scripts/check_technical_reviews.py", inputs / "current-checker.py")
extract_root = Path("/tmp/phase4-7_14-reinspection-20261006")
for name in ["extraction.json", "fence-1.py", "bootstrap.py"]:
    shutil.copyfile(extract_root / name, inputs / name)
extraction = json.loads((inputs / "extraction.json").read_bytes())
assert extraction["figure_sha256"] == {} and not extraction["svg_references"]
assert (inputs / "fence-1.py").read_bytes() == (OLD / "inputs/current/fence-1.py").read_bytes()
prior_section = (OLD / "inputs/current/section.md").read_bytes()
diff = "".join(difflib.unified_diff(prior_section.decode().splitlines(keepends=True), section.decode().splitlines(keepends=True), fromfile="own-frozen-prior-7.14", tofile="current-7.14"))
(BASE / "section.diff").write_text(diff, encoding="utf-8")

# Explicit raw-measurement pointers; no author correction, note, or review values.
sft_path = ROOT / "docs/course-experiments/results/sft.json"
ablation_path = ROOT / "docs/course-experiments/results/sft_ablation.json"
assert sft_path.read_bytes() == (OLD / "inputs/current/docs/course-experiments/results/sft.json").read_bytes()
assert ablation_path.read_bytes() == (OLD / "inputs/current/docs/course-experiments/results/sft_ablation.json").read_bytes()
sft = json.loads(sft_path.read_bytes())
ablation = json.loads(ablation_path.read_bytes())
assert sft["results"]["checkpoint"] == "model.pt"
assert sft["results"]["training"]["steps"] == 900
assert sft["step_scale"] == 1.0
assert sft["results"]["training"]["records"] == 45
assert ablation["results"]["before"]["A_attributes"] == sft["results"]["after"]
for split in ["validation", "test"]:
    assert ablation["results"]["data"]["attributes"][split]["sha256"] == sft["results"]["data"][split]["sha256"]

revision = sft["revision"]
source_integrity = []
for path in ["scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/model.py"]:
    frozen = inputs / "direct-sft-original" / path
    assert frozen.read_bytes() == subprocess.check_output(["git", "show", revision + ":" + path], cwd=ROOT)
    assert sha(frozen) == sft["code_sha256"][path]
    source_integrity.append({"original_path": path, "revision": revision, "sha256": sha(frozen), "recorded_code_hash_match": True})
text_path = inputs / "direct-sft-original/scripts/course_experiments/text.py"
tree = ast.parse(text_path.read_bytes())
run_sft = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_sft")
steps_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_steps")
assert any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "seed" for n in run_sft.body)
model_assignment = next(n for n in run_sft.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model" for t in n.targets))
assert model_assignment.value.func.id == "new_lm"
training_assignment = next(n for n in run_sft.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "training" for t in n.targets))
assert training_assignment.value.func.id == "fit_lm"
kwargs = {k.arg: k.value for k in training_assignment.value.keywords}
assert kwargs["name"].value == "model" and kwargs["mode"].value == "sft"
assert kwargs["steps"].func.id == "_steps" and kwargs["steps"].args[1].value == 900
step_ns = {}
exec(compile(ast.Module(body=[steps_node], type_ignores=[]), str(text_path), "exec"), step_ns)
assert step_ns["_steps"](SimpleNamespace(step_scale=sft["step_scale"]), 900) == 900
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ["load_lm", "load_checkpoint"] for n in ast.walk(model_assignment))

# Original PyTorch sources retained; installed documented CE bytes remain identical.
import torch.nn.modules.loss
assert Path(torch.nn.modules.loss.__file__).read_bytes() == (OLD / "sources/pytorch-loss.py").read_bytes()
assert torch.version.git_version == "5c4886908584029761b579af026dcfb627c84070"
assert torch.version.cuda is None and not torch.cuda.is_available()
render_info = json.loads((BASE / "render/render-info.json").read_bytes())
assert render_info["status"] == 200 and render_info["article_image_count"] == 0 and render_info["article_svg_count"] == 0
assert render_info["baseline_link"].endswith("/7.13.html")
shutil.copyfile("/tmp/phase4-7_14-current.html", BASE / "render/served-current.html")
for name in ["stdout", "stderr"]:
    shutil.copyfile("/tmp/phase4-7_14-render." + name + ".txt", BASE / ("render/failed-cli." + name + ".txt"))

receipt = {
    "kind": "actual_original_owner_reinspection",
    "reviewer_task": "/root/phase4_factual_coordinator/factual_7_14",
    "reviewed_at": datetime.now(UTC).isoformat(),
    "prior_report": {"path": prior_path.relative_to(ROOT).as_posix(), "sha256": sha(prior_path), "preserved_before_reading_current": True},
    "current_section": {"source": "course/chapters/07.md#7.14", "sha256": hashlib.sha256(section).hexdigest(), "snapshot": (inputs / "current-section.md").relative_to(ROOT).as_posix(), "first_line": line, "entire_section_personally_read": True},
    "current_context": {"source": "course/chapters/07.md#7.13", "sha256": hashlib.sha256(context).hexdigest(), "snapshot": (inputs / "current-context-7.13.md").relative_to(ROOT).as_posix(), "first_line": ctx_line, "entire_needed_section_personally_read": True},
    "frozen_complete_chapter": {"meaning": "2026-10-06 reinspection input snapshot, not an evergreen full-chapter assertion", "path": (inputs / "current-frozen-chapter-07.md").relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(chapter).hexdigest()},
    "changes_inspected": [
        {"change": "Traditional character corrections 我們/代價/乾淨/形狀", "affected_claims": ["C1", "C5", "C6"], "finding": "Meaning, arithmetic, objective, code, and measured values unchanged; previous real evidence reused after hash checks"},
        {"change": "Baseline reference changed from 7.12 to 7.13; direct SFT attribute model, explicitly from scratch for 900 updates", "affected_claims": ["C5"], "finding": "Direct original sft revision a253d... run_sft builds new_lm, saves start step0, fits mode=sft/name=model at _steps(900); original result checkpoint=model.pt, training.steps=900 and step_scale=1.0. Current7.13 states same baseline; ablation before.A_attributes equals direct sft after. No pretrain-sft checkpoint is loaded by ablation."},
    ],
    "original_measurement_pointers_actually_checked": ["sft:/revision", "sft:/code_sha256/scripts~1course_experiments~1text.py", "sft:/code_sha256/scripts~1course_experiments~1common.py", "sft:/code_sha256/tiny_perceptron~1model.py", "sft:/step_scale", "sft:/results/checkpoint", "sft:/results/training/steps", "sft:/results/training/records", "sft:/results/training/effective_tokens", "sft:/results/after", "sft:/results/data/validation/sha256", "sft:/results/data/test/sha256", "ablation:/results/before/A_attributes", "ablation:/results/data/attributes/validation/sha256", "ablation:/results/data/attributes/test/sha256"],
    "original_result_full_hashes_unchanged": {"sft.json": sha(sft_path), "sft_ablation.json": sha(ablation_path)},
    "direct_sft_original_sources": source_integrity,
    "actual_source_reading": ["Direct-SFT original text.py AST run_sft556-566 and _steps34-36, common.py seed41-42/new_lm45-47, model.py TinyLM.__init__53-67. Read only needed construction/training branches, not returned author explanations.", "Personally reapplied the unchanged own official CE/source, numeric/gradient/boundary, split/sampler/ID/NLL evidence to every current claim after verifying all43 prior artifact hashes. No fresh official-source fetch claimed."],
    "own_prior_evidence_integrity": integrity,
    "code_changed": False,
    "code_sha256": sha(inputs / "fence-1.py"),
    "original_fence_rerun_this_time": False,
    "reason_no_reexecution": "Code bytes and all relevant numeric/empirical inputs unchanged; current change adds baseline provenance, checked through original AST and original recorded900-step measurement, not through retraining.",
    "visual_inspection": {"url": render_info["url"], "source_figures": {}, "successful_render_command": "timeout 35 .venv/bin/python docs/technical-reviews/artifacts/phase4-7_14-reinspection-20261006/render/render.py", "render_exit_code": 0, "render_environment": {"playwright": render_info["playwright"], "chromium": render_info["chromium"], "viewport": "1280x800"}, "screenshot": (BASE / "render/desktop-expanded.png").relative_to(ROOT).as_posix(), "screenshot_sha256": sha(BASE / "render/desktop-expanded.png"), "personally_viewed_with_view_image": True, "finding": "Current expanded supplemental paragraph shows link to7.13 and900-update baseline; table remains2/5 vs1/5,7/10 vs6/10,.37101 vs.34698. Article contains no image/SVG figures. Screenshot's sticky header partially overlays title; this review verifies changed paragraph and evidence, not a reader/usability pass.", "attempt_history": "Initial chromium CLI render timed out20s(exit124), produced no image; first view_image attempt therefore failed. Switched to installed Playwright with DOM-contentloaded navigation; actual screenshot succeeded and was personally viewed. No unavailable visual check was recorded as completed."},
    "environment": {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "CPU metadata and bounded AST/JSON checks; no model instantiated"},
    "verdict": "pass",
    "unresolved_substantive_questions": [],
    "actual_scope": "Current7.14 whole section and necessary current7.13 context personally read. Changed900-step direct-SFT lineage verified from original-version construction/training branches and original raw measurement pointers; every unchanged own artifact/code/primary measurement rehashed and supporting scope reapplied. Own previous true fence/numeric/gradient/data/ID/NLL checks reused openly. No new training, GPU, weights, model/data download, fresh paper search, or full pipeline; stored CUDA NLL sums remain original measurements.",
    "verification_attempts": [{"exit_code": 1, "finding": "First own AST assertion assumed seed was statement5, but original source has seed at statement4. Original source was read; verifier changed to test actual seed expression rather than fixed ordinal. Failed script/stdout/stderr retained in attempts/first-ast-check. This was a verifier assertion error, not an unresolved教材claim."}, {"exit_code": 0, "finding": "Corrected bounded AST/hash/raw-pointer verification passed; no original implementation or teaching text modified."}],
}
dump(BASE / "reinspection-receipt.json", receipt)
print(json.dumps({"verdict": "pass", "current_source_sha256": receipt["current_section"]["sha256"], "prior_report_sha256": sha(prior_path), "unchanged_own_artifact_count": len(integrity), "direct_sft_original_revision": revision, "checkpoint": "model.pt", "original_training_steps": sft["results"]["training"]["steps"], "direct_sft_step_scale": sft["step_scale"], "reinspection_receipt_sha256": sha(BASE / "reinspection-receipt.json"), "new_model_training": False}, ensure_ascii=False, indent=2))
