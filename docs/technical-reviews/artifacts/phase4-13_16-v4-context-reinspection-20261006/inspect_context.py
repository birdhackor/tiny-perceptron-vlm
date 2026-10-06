"""Original owner's V4 source/context/figure fingerprint inspection; never runs a model."""
from pathlib import Path
import ast
import difflib
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, UTC

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-13_16-independent"
PREVIOUS = ROOT / "docs/technical-reviews/artifacts/phase4-13_16-context-reinspection-20261005"
OWNER = "/root/phase4_factual_coordinator/factual_13_16"
def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
prior_path = A / "own-prior-canonical-a3eb7278.json"
prior_raw = prior_path.read_bytes()
assert sha(prior_raw) == "a3eb727878ab8c6dd9d707cf488f14fd7e9b08322c87e467a141ab652463b44e"
assert (ROOT / "docs/technical-reviews/13.16.json").read_bytes() == prior_raw
prior = json.loads(prior_raw)
assert prior["reviewer_task"] == OWNER
spec = importlib.util.spec_from_file_location("sf", ROOT / "docs/review-tools/section_facts.py")
sf = importlib.util.module_from_spec(spec); spec.loader.exec_module(sf)
expected = {"13.16": "a8da849365b9da1151dd269f6cb0b31043439d286885049953d7ce0ebfa1ce70", "13.12": "4e52b0f5e6afa038e0202979e85de86831e39746eca746a3fc8deb8b78fc4ffc", "13.14": "ba2361526193f3321b9d8974a6889117fe4602a5e2d6d132413d0d22ab9d6ab4"}
sections = {}
diffs = []
for lesson in ["13.16", "13.12", "13.14"]:
    current, whole, first = sf.original_section(ROOT / "course/chapters/13.md", lesson)
    assert sha(current) == expected[lesson]
    snapshot = A / (lesson + "-current.md")
    assert snapshot.read_bytes() == current
    previous_path = PREVIOUS / (lesson + "-current.md")
    previous = previous_path.read_bytes()
    old_fences = sf.fences(previous, first)
    new_fences = sf.fences(current, first)
    assert len(old_fences) == len(new_fences)
    fence_facts = []
    for index, (before, after) in enumerate(zip(old_fences, new_fences, strict=True), 1):
        assert before["raw"] == after["raw"]
        assert before["language"] == after["language"] == "python"
        assert ast.dump(ast.parse(before["raw"]), include_attributes=False) == ast.dump(ast.parse(after["raw"]), include_attributes=False)
        fence_facts.append({"index": index, "language": "python", "raw_sha256": sha(after["raw"]), "raw_bytes_identical": True, "python_ast_identical": True, "executed_this_round": False})
    difference = "".join(difflib.unified_diff(previous.decode().splitlines(keepends=True), current.decode().splitlines(keepends=True), fromfile=previous_path.relative_to(ROOT).as_posix(), tofile=snapshot.relative_to(ROOT).as_posix()))
    if difference: diffs.append(difference)
    sections[lesson] = {"source": "course/chapters/13.md#" + lesson, "sha256": sha(current), "current_locator": f"course/chapters/13.md:{first}; complete section including details", "snapshot": snapshot.relative_to(ROOT).as_posix(), "prior_snapshot": previous_path.relative_to(ROOT).as_posix(), "prior_sha256": sha(previous), "raw_bytes_identical_to_prior": current == previous, "personally_read_this_round": "complete section including details", "fence_facts": fence_facts}
    if lesson == "13.16": assert current == previous
(A / "current-chapter13-frozen-input.md").write_bytes(whole)
(A / "context-diff-from-own-previous.patch").write_text("\n".join(diffs))
method = ROOT / "docs/review-tools/factual-reviewer-instructions.md"
(A / "reviewer-method-current.md").write_bytes(method.read_bytes())

unchanged = []
for artifact in prior["artifacts"]:
    actual = sha((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"], artifact["path"]
    unchanged.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})
method_sources = []
for source in prior["sources"]:
    if source["kind"] != "repository_code": continue
    actual = sha((ROOT / source["path"]).read_bytes())
    assert actual == source["sha256"]
    method_sources.append({"id": source["id"], "path": source["path"], "sha256": actual, "unchanged": True})

figure = ROOT / "course/figures/rewrite-13-model-roles.svg"
figure_sha = sha(figure.read_bytes())
assert figure_sha == "d7589ab9e44118540111e8f331b555d4d6c4db3f548e7d373bf7d40750eba406"
assert (A / "model-roles-current.svg").read_bytes() == figure.read_bytes()
assert (A / "model-roles-current.png").is_file()
environment = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "device": "CPU for byte/hash/AST inspection and SVG rendering only; no torch/model executed", "inkscape": subprocess.run(["inkscape", "--version"], capture_output=True, text=True, check=True).stdout.strip()}
dump(A / "inspection-environment.json", environment)
facts = {"schema_version": 1, "kind": "same_owner_v4_context_inspection_facts", "reviewer_task": OWNER, "recorded_on": datetime.now(UTC).isoformat(), "own_prior_opaque": {"path": prior_path.relative_to(ROOT).as_posix(), "sha256": sha(prior_raw), "bytes_equal_to_canonical_before_new_write": True}, "sections": sections, "frozen_whole_input": {"snapshot": (A / "current-chapter13-frozen-input.md").relative_to(ROOT).as_posix(), "sha256": sha(whole), "meaning": "This inspection's frozen full input bytes, not a claim of full chapter reading or a future current chapter hash; prior frozen snapshots retain their original meanings/hashes."}, "method_instructions_sha256": sha(method.read_bytes()), "unchanged_registered_prior_artifacts": unchanged, "unchanged_current_method_sources": method_sources, "necessary_context_figure": {"source": figure.relative_to(ROOT).as_posix(), "sha256": figure_sha, "snapshot": (A / "model-roles-current.svg").relative_to(ROOT).as_posix(), "render": (A / "model-roles-current.png").relative_to(ROOT).as_posix(), "render_sha256": sha((A / "model-roles-current.png").read_bytes()), "render_exit_code": 0, "personally_viewed": True, "actual_observation": "Five vertical labeled role boxes readable. Reference is labeled PPO starting point, emits starting-point card probabilities, and does not update during PPO. Old is collected log-probability snapshot fixed within rollout; reward fixed and policy/critic updated. SVG desc states reference is after demonstration training. Labels and timing match personally read current13.14 and original implementation; no clipped label or ambiguous numeric axis."}, "narrow_original_rereads": [{"path": (OLD / "posttraining-original.py").relative_to(ROOT).as_posix(), "locator": "lines278–299", "sha256": sha((OLD / "posttraining-original.py").read_bytes()), "support": "Reference copied/frozen at289 after SFT loop278–287, before reward/PPO branch training."}, {"path": (OLD / "instructgpt-v1-original.txt").relative_to(ROOT).as_posix(), "locator": "lines302–319 (sec3.1 three-step process);483–509 (sec3.5 RL and eq2)", "sha256": sha((OLD / "instructgpt-v1-original.txt").read_bytes()), "support": "PPO fine-tunes supervised policy; KL uses SFT reference. This is the already personally verified arXiv2203.02155v1 original; no source retrieval or paper-wide reread this round."}], "environment": environment, "scope": "Personally read complete current13.16/13.12/13.14; read source SVG; actually rendered/viewed current required context figure; compared own prior snapshots and every registered own evidence/current method fingerprint. Narrow original implementation/paper excerpt reread, no other owner report/repair expectation read, no model CPU/recipe/GPU/train.", "source_bytes_policy": "Original UTF-8 section bytes without whitespace or newline normalization", "inspection_code_sha256": sha(Path(__file__).read_bytes())}
dump(A / "inspection-facts.json", facts)
print(json.dumps({"source_sha256": sections["13.16"]["sha256"], "contexts": {k: sections[k]["sha256"] for k in ["13.12", "13.14"]}, "unchanged_prior_artifacts": len(unchanged), "method_fingerprints_unchanged": True, "figure_sha256": figure_sha, "figure_rendered_and_personally_viewed": True, "model_executions": 0, "scope": facts["scope"]}, ensure_ascii=False, indent=2))
