import ast
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-20261005-T_2-independent"
REVIEWER = "/root/phase4_factual_coordinator/factual_t_2"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
prior_raw = (OUT / "prior-T.2.opaque.json").read_bytes()
assert sha(prior_raw) == "4c6b191fc90479a0eb255f7300b669698462bd98e6f1892b8e80132e3cda6927"
report = json.loads(prior_raw)
assert report["reviewer_task"] == REVIEWER and report["lesson_id"] == "T.2"
print("OWN PRIOR REPORT TOP-LEVEL KEYS/TYPES", json.dumps({k: type(v).__name__ for k, v in report.items()}))
primary, _, first_line = facts.original_section(ROOT / "course/training.md", "T.2")
assert sha(primary) == report["source_sha256"]
assert primary == (OLD / "section.md").read_bytes()
(OUT / "current-T.2.md").write_bytes(primary)
print("CURRENT T.2 RAW SECTION", primary.decode("utf-8"))
current_context, _, context_start = facts.original_section(ROOT / "course/chapters/05.md", "5.1")
context_needed = current_context.split(b"<details>")[0]
prior_context_needed = (OLD / "5.1.md").read_bytes().split(b"<details>")[0]
(OUT / "current-5.1.md").write_bytes(current_context)
(OUT / "current-5.1-necessary-slice.md").write_bytes(context_needed)
(OUT / "prior-5.1-necessary-slice.md").write_bytes(prior_context_needed)
print("CURRENT NECESSARY 5.1 SLICE", context_needed.decode("utf-8"))
slice_diff = "".join(difflib.unified_diff(prior_context_needed.decode().splitlines(True), context_needed.decode().splitlines(True), fromfile="prior-5.1-necessary-slice", tofile="current-5.1-necessary-slice"))
(OUT / "necessary-5.1.diff.txt").write_text(slice_diff, encoding="utf-8")
print("NECESSARY CONTEXT DIFF", slice_diff)
current_fences = facts.fences(context_needed, context_start)
prior_fences = facts.fences(prior_context_needed, context_start)
assert len(current_fences) == len(prior_fences) == 1
assert current_fences[0]["raw"] == prior_fences[0]["raw"]
fence = current_fences[0]["raw"]
tree = ast.parse(fence.decode())
methods = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("backward", "norm", "step"):
        methods.append({"method": node.func.attr, "fence_line": node.lineno, "expression": ast.unparse(node)})
print("CURRENT CONTEXT NECESSARY AST CALLS", json.dumps(methods, ensure_ascii=False))
assert any(m["expression"] == "model.embedding.weight.grad.norm()" for m in methods)
assert any(m["expression"] == "optimizer.step()" for m in methods)

artifact_checks = []
for artifact in report["artifacts"]:
    current_sha = sha((ROOT / artifact["path"]).read_bytes())
    assert current_sha == artifact["sha256"], artifact["path"]
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"], "sha256": current_sha, "unchanged": True})
code_checks = []
for source in report["sources"]:
    if source["kind"] == "repository_code":
        assert sha((ROOT / source["path"]).read_bytes()) == source["sha256"]
        code_checks.append({"id": source["id"], "path": source["path"], "sha256": source["sha256"], "unchanged": True})

# Read this reviewer's original measurements only at named, necessary pointers.
measurement = json.loads((OLD / "execution/verification.json").read_text())
print("OWN MEASUREMENT TOP-LEVEL KEYS/TYPES", json.dumps({k: type(v).__name__ for k, v in measurement.items()}))
selected = {"/environment": measurement["environment"], "/original_train_sha256": measurement["original_train_sha256"], "/bounded_variation": {k: measurement["bounded_variation"][k] for k in ("loss", "parameter_before", "parameter_after_backward_and_clip", "gradient_before_clipping", "returned_total_norm", "unused_parameter_gradient")}}
for i, record in enumerate(measurement["records"]):
    for field in ("task", "command_argv", "exit_code", "entry"):
        selected[f"/records/{i}/{field}"] = record[field]
    for field in ("optimizer_step_calls", "save_checkpoint_calls", "torch_save_calls", "changed_parameter_tensors", "gradient_norm", "scored_positions_in_batch"):
        selected[f"/records/{i}/observations/{field}"] = record["observations"][field]
print("OWN PRIOR RAW MEASUREMENT POINTERS", json.dumps(selected, ensure_ascii=False, indent=2))
for record in measurement["records"]:
    assert record["exit_code"] == 0
    obs = record["observations"]
    assert all(obs[f] == 0 for f in ("optimizer_step_calls", "save_checkpoint_calls", "torch_save_calls", "changed_parameter_tensors"))

# Local package metadata and generated version constants, without importing torch,
# constructing a model, executing training, fetching sources or accessing caches.
dist = importlib.metadata.distribution("torch")
version_file = Path(dist.locate_file("torch/version.py"))
version_ast = ast.parse(version_file.read_text())
version_constants = {}
for node in version_ast.body:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    for target in targets:
        if isinstance(target, ast.Name) and target.id in ("__version__", "git_version", "cuda"):
            version_constants[target.id] = ast.literal_eval(node.value)
environment = {"python": platform.python_version(), "python_executable": sys.executable, "torch_distribution": dist.version, "torch_git_version": str(version_constants["git_version"]), "cuda_build": str(version_constants["cuda"]), "inspection_kind": "CPU metadata, bytes, AST and JSON only; original model/CLI runs deliberately not replayed"}
assert environment["torch_distribution"] == measurement["environment"]["torch"]
assert environment["torch_git_version"] == measurement["environment"]["torch_git_version"]
assert environment["cuda_build"] == measurement["environment"]["cuda_build"]
print("CURRENT INSPECTION ENVIRONMENT", json.dumps(environment))

# Re-read only the relevant preserved original authority functions to check the
# magnitude interpretation. Byte hashes already established exact prior versions.
authority_paths = ["official-sources/torch__nn__utils__clip_grad.py--_get_total_norm.txt", "official-sources/torch__nn__utils__clip_grad.py--clip_grad_norm_.txt"]
for name in authority_paths:
    print("PRESERVED ORIGINAL AUTHORITY", name, (OLD / name).read_text())
other_context, _, _ = facts.original_section(ROOT / "course/chapters/10.md", "10.7")
prior_other_context = (OLD / "10.7.md").read_bytes()
assert other_context == prior_other_context

inspection = {
    "reviewer_task": REVIEWER,
    "callback_kind": "same original independent factual reviewer; narrow context callback",
    "primary": {"source": "course/training.md#T.2", "sha256": sha(primary), "first_line": first_line, "snapshot": (OUT / "current-T.2.md").relative_to(ROOT).as_posix(), "personally_read_complete": True, "unchanged": True, "intro": None, "figures": {}},
    "necessary_context": {"source": "course/chapters/05.md#5.1", "first_line": context_start, "whole_section_sha256": sha(current_context), "whole_section_hash_meaning": "Byte version of preserved section; only the pre-details necessary slice was semantically inspected. Appendix measurements were not re-reviewed.", "necessary_slice_sha256": sha(context_needed), "necessary_slice_snapshot": (OUT / "current-5.1-necessary-slice.md").relative_to(ROOT).as_posix(), "prior_necessary_slice_sha256": sha(prior_context_needed), "python_fence_sha256": sha(fence), "python_fence_unchanged": True, "methods": methods, "support_scope": "T.2 only points to 5.1 as a parameter-specific embedding-gradient check. Its unchanged code and current magnitude wording still provide that cross-reference. This callback does not endorse all 5.1 claims, its 40-step outcome or its appendix training measurements."},
    "other_cross_reference": {"source": "course/chapters/10.md#10.7", "sha256": sha(other_context), "unchanged": True, "action": "Raw bytes/version equality checked only; no new semantic reread or figure validation."},
    "prior_canonical": {"path": (OUT / "prior-T.2.opaque.json").relative_to(ROOT).as_posix(), "sha256": sha(prior_raw)},
    "reused_artifacts": artifact_checks,
    "unchanged_runtime_contracts": code_checks,
    "measurement_file_sha256": sha((OLD / "execution/verification.json").read_bytes()),
    "measurement_pointers_personally_checked": list(selected),
    "environment": environment,
    "judgment": "The wording 'gradient magnitude greater than 0' is consistent with the unchanged norm() inspection and the original authoritative L2/magnitude semantics. The current necessary cross-reference continues to support T.2; no substantive uncertainty is introduced.",
    "reuse_scope": "All original T.2 concept/CLI/CPU/source evidence remains at exact preserved hashes and its original support scope. No original diagnostic was relabeled as a new execution or current full-chapter acceptance.",
    "events_and_limits": "No source edits, new reviewer/child, full reader session, model execution, training, GPU, weights/data download, external-source fetch or new paper search. No old other-review/author-correction summary was read. Prior report/proofs remain untouched; own original report metadata and named own raw measurement leaves were consulted.",
}
(OUT / "current-inspection.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("INSPECTION FINAL", json.dumps({"primary_sha256": sha(primary), "necessary_context_section_sha256": sha(current_context), "necessary_slice_sha256": sha(context_needed), "prior_artifacts_verified": len(artifact_checks), "runtime_sources_verified": len(code_checks), "inspection_sha256": sha((OUT / "current-inspection.json").read_bytes())}, ensure_ascii=False))
