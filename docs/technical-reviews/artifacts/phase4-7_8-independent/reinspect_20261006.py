"""Same-owner real reinspection of the current 7.8; preserve all old proof."""
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[3]
OUT = BASE / "reinspection-20261006"
OUT.mkdir(exist_ok=False)
EXPECTED_PRIOR = "23cac4ecdec456327f9aa40a92300ccfdbaac7969754deac75bc8238bcf0bf6a"
EXPECTED_CURRENT = "453b1a38c02163c8613cefc6122b903b984e03a4921fcd69168ae15ab7046618"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return path


report_path = REPO / "docs/technical-reviews/7.8.json"
prior_bytes = report_path.read_bytes()
assert hashlib.sha256(prior_bytes).hexdigest() == EXPECTED_PRIOR
history = REPO / "docs/technical-reviews/history" / f"phase4-7_8-own-before-reinspection-20261006-{EXPECTED_PRIOR}.json"
history.parent.mkdir(exist_ok=True)
if history.exists():
    assert history.read_bytes() == prior_bytes
else:
    history.write_bytes(prior_bytes)
report = json.loads(prior_bytes)  # Only this same technical owner's exact frozen report.
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_7_8"

spec = importlib.util.spec_from_file_location("facts", REPO / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
current, whole, first_line = facts.original_section(REPO / "course/chapters/07.md", "7.8")
assert hashlib.sha256(current).hexdigest() == EXPECTED_CURRENT
old = (BASE / "inputs/section.md").read_bytes()
assert current == old.replace("正式入口须先拒絕空題".encode(), "正式入口須先拒絕空題".encode())
(OUT / "current-7.8.md").write_bytes(current)
diff = "".join(difflib.unified_diff(old.decode().splitlines(True), current.decode().splitlines(True), fromfile="own-frozen-7.8", tofile="current-7.8"))
(OUT / "current-7.8.diff").write_text(diff)
old_fence = (BASE / "inputs/fence-1.py").read_bytes()
current_fences = facts.fences(current, first_line)
assert len(current_fences) == 1 and current_fences[0]["closed"] and current_fences[0]["raw"] == old_fence
(OUT / "current-7.8-fence-1.py").write_bytes(current_fences[0]["raw"])

reused_artifacts = []
for artifact in report["artifacts"]:
    actual = digest(REPO / artifact["path"])
    assert actual == artifact["sha256"], artifact["id"]
    reused_artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})
implementations = []
for name in ["model.py", "attention.py", "data.py", "modern.py"]:
    current_path = REPO / "tiny_perceptron" / name
    assert current_path.read_bytes() == (BASE / "inputs" / name).read_bytes()
    implementations.append({"path": str(current_path.relative_to(REPO)), "sha256": digest(current_path), "unchanged": True})
figures = []
for path, expected in report["context_figure_sha256"].items():
    assert digest(REPO / path) == expected
    figures.append({"path": path, "sha256": expected, "unchanged": True,
                    "render_policy": "Reused original actual Inkscape raster after SVG and raster hash verification; PNG personally reopened with view_image during this reinspection."})

contexts = []
for lesson in ["7.4", "7.7"]:
    body, _, line = facts.original_section(REPO / "course/chapters/07.md", lesson)
    prior = (BASE / "inputs" / f"prerequisite-{lesson}.md").read_bytes()
    (OUT / f"current-{lesson}.md").write_bytes(body)
    (OUT / f"current-{lesson}.diff").write_text("".join(difflib.unified_diff(prior.decode().splitlines(True), body.decode().splitlines(True), fromfile=f"own-frozen-{lesson}", tofile=f"current-{lesson}")))
    if lesson == "7.4":
        assert body == prior.replace("接着".encode(), "接著".encode())
        change = "接着→接著 glyph edit only; unchanged input/target alignment claims, code and SVG."
    else:
        assert body == prior.replace(b"assert torch.allclose(base, actual, atol=1e-6)", b"assert torch.allclose(base, actual, atol=1e-6, rtol=0.0)")
        change = "Only assertion strengthens comparison by explicit rtol=0.0; model/forward/valid/positions unchanged. One bounded current context fence executed below."
        fence = facts.fences(body, line)[0]
        assert fence["closed"]
        (OUT / "current-7.7-fence-1.py").write_bytes(fence["raw"])
    contexts.append({"lesson": lesson, "sha256": hashlib.sha256(body).hexdigest(), "change": change,
                     "read_scope": "Current prerequisite section reread in full as necessary context; not a separate technical-owner review of that section."})

env_overrides = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                 "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(REPO)}
argv = [str(REPO / ".venv/bin/python"), str(OUT / "current-7.7-fence-1.py")]
executed = subprocess.run(argv, cwd=REPO, env={**os.environ, **env_overrides}, capture_output=True, timeout=30, check=False)
(OUT / "context-7.7.stdout.txt").write_bytes(executed.stdout)
(OUT / "context-7.7.stderr.txt").write_bytes(executed.stderr)
assert executed.returncode == 0, executed.stderr.decode()
observed_text = executed.stdout.decode().strip()
assert observed_text.startswith("有效位置最大差 ")
observed_difference = float(observed_text.rsplit(" ", 1)[1])
assert observed_difference <= 1e-6
context_run = {"command_argv": argv, "command": shlex.join(argv), "cwd": str(REPO), "timeout_seconds": 30,
               "environment_overrides": env_overrides, "exit_code": executed.returncode,
               "script_sha256": digest(OUT / "current-7.7-fence-1.py"), "stdout_sha256": digest(OUT / "context-7.7.stdout.txt"),
               "stderr_sha256": digest(OUT / "context-7.7.stderr.txt"), "observed_max_abs_difference": observed_difference,
               "assertion": "torch.allclose(base,actual,atol=1e-6,rtol=0.0)", "scope": "Only changed prerequisite tolerance assertion; freshly initialized random width8 TinyLM inference, no training/evaluation of existing weights."}
save("context-run.json", context_run)

environment_argv = [str(REPO / ".venv/bin/python"), "-c", "import json,sys,torch;print(json.dumps({'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torch_git_version':torch.version.git_version,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device':'cpu'}))"]
environment = json.loads(subprocess.run(environment_argv, cwd=REPO, env={**os.environ, **env_overrides}, capture_output=True, text=True, timeout=15, check=True).stdout)
assert environment["cuda_build"] is None and environment["cuda_available"] is False
save("environment.json", environment)
(OUT / "reviewer-method.md").write_bytes((REPO / "docs/review-tools/factual-reviewer-instructions.md").read_bytes())
claim_coverage = [{"claim_id": claim["id"], "technical_statement_changed": False, "status": "verified",
                   "actual_reinspection": "Compared the whole current section with own frozen input, reread this claim and its original evidence/support scope; code, primary sources and actual CPU proof hashes verified unchanged.",
                   "reused_original_support": [{"source_id": e["source_id"], "locator": e["locator"], "supports": e["supports"]} for e in claim["evidence"]]}
                  for claim in report["claims"]]
receipt = {
    "kind": "same_owner_real_technical_reinspection", "schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": report["reviewer_task"], "source": report["source"], "source_sha256": EXPECTED_CURRENT,
    "prior_source_sha256": report["source_sha256"], "prior_report_history_path": str(history.relative_to(REPO)), "prior_report_sha256": digest(history),
    "current_section_read": "Full original UTF-8 7.8 read directly; own original raw bytes diffed; no author repair summary or other review read.",
    "actual_change": {"text": "正式入口须先拒絕空題→正式入口須先拒絕空題", "nature": "Traditional character glyph correction only", "code_changed": False, "figures_changed": False, "technical_claims_changed": False},
    "source_file_frozen_input": {"description": "Complete current chapter hash observed during this exact reinspection, not asserted as later chapter state; whole chapter was not read", "observed_sha256": hashlib.sha256(whole).hexdigest()},
    "unchanged_artifacts_verified": reused_artifacts, "implementations_verified": implementations, "context_figures_verified": figures,
    "necessary_contexts": contexts, "new_cpu_execution": context_run, "environment": environment, "environment_command_argv": environment_argv,
    "official_source_reuse": "All six own original official snapshots checked against their saved hashes; existing URL/version/authority/inspection/access dates retained. Personally reread causal logits definition629-641, generation2881-2923, NumPy negative/paired indexing304-340, saved TinyLM.forward68-86 and attention_mask10-18. No new papers or external source fetch needed for a glyph change.",
    "claim_coverage": claim_coverage,
    "evidence_reuse_scope": "Original 7.8 fence and exercise/exhaustive15-mask/gather/empty-row and TinyLM mechanics executions are reused only after unmodified code and exact artifacts verified. No claim that those checks were rerun today. Current 7.7 fence run is separately recorded.",
    "visual_scope": "7.8 still has no figures. Original two prerequisite Inkscape PNG renders were personally reopened via view_image after source/raster hash verification; no new render or browser desktop/mobile page check claimed.",
    "verdict": "pass", "unresolved_questions": [], "excluded_work": "No GPU, training, paid compute, dataset/model download, existing weight evaluation, or full pipeline. No body/SVG edits by reviewer."
}
receipt_path = save("reinspection-receipt.json", receipt)


def add_artifact(identifier, path, kind, description, command=None, result=None):
    artifact = {"id": identifier, "path": str(path.relative_to(REPO)), "sha256": digest(path), "kind": kind, "description": description}
    if kind == "execution":
        artifact.update(command=command, result=result, environment={key: str(value) for key, value in environment.items()})
    report["artifacts"].append(artifact)


receipt_id = "same-owner-reinspection-20261006"
add_artifact(receipt_id, receipt_path, "execution", "原technical owner本人讀current全節/必要ctx、核全部舊證據指紋與支持、glyph變更與ctx短CPU執行之真回查收據。",
             f"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python {Path(__file__).relative_to(REPO)}", "pass; glyph-only7.8 change; 30 proof artifacts unchanged; same8 technical claims supported; new7.7 strict-tolerance CPU fence exits0")
add_artifact("current-section-20261006", OUT / "current-7.8.md", "source_snapshot", "本次親讀的current7.8原始UTF-8 bytes，source SHA對應此稿。")
add_artifact("section-diff-20261006", OUT / "current-7.8.diff", "source_snapshot", "相對本人original frozen7.8的實際diff；仅须→須，不替代整節真閱讀。")
add_artifact("reinspection-code-20261006", Path(__file__), "code", "本次本人指紋、diff、ctx fence執行與canonical更新實際腳本。")
for lesson in ["7.4", "7.7"]:
    add_artifact(f"current-context-{lesson}-20261006", OUT / f"current-{lesson}.md", "source_snapshot", "本人重讀的必要current前置完整bytes，diff與範圍記於收據。")
add_artifact("strict-context-code-20261006", OUT / "current-7.7-fence-1.py", "code", "current前置7.7實際原fence，包括新rtol0.0嚴格assert。")
add_artifact("strict-context-execution-20261006", OUT / "context-7.7.stdout.txt", "execution", "真正執行current前置7.7原fence的新CPU stdout；不冒稱重新跑7.8舊checks。", context_run["command"], f"exit0; observed max abs difference {observed_difference}; atol1e-6,rtol0.0 assertion passes")
report["source_sha256"] = EXPECTED_CURRENT
report["verdict"] = "pass"
report["reinspection"] = {"artifact_id": receipt_id, "receipt_path": str(receipt_path.relative_to(REPO)), "receipt_sha256": digest(receipt_path),
                          "prior_report_history_path": str(history.relative_to(REPO)), "prior_report_sha256": digest(history),
                          "scope": "Full same-owner reinspection; only glyph change in7.8; changed prerequisite7.7 tolerance run separately; all original8 claims/evidence genuinely verified applicable."}
report["review_history"].append({"stage": "same_owner_current_reinspection", "source_sha256": EXPECTED_CURRENT, "verdict": "pass",
                                 "artifact_id": receipt_id, "details": "Full current7.8 and needed7.4/7.7 read, own frozen diff verified; glyph change does not alter substantive claims. All original proof and implementation/SVG hashes checked; current7.7 stricter assertion executed successfully."})
report["checks"]["factual_accuracy"]["details"] += " 2026-10-06 same-owner真回查：须→須，不改8項技術主張；新canonical綁定本次receipt。"
report["checks"]["figure_consistency"]["details"] += " 本次再次核SVG/render指紋並view原實際PNG，兩前置圖仍支持本節語義；无新render或網頁檢查聲稱。"
report["checks"]["source_verification"]["details"] += " 本次沿用自己已親讀之不變官方原始快照，核指紋/支持並重讀真正相关原碼段落，不重抓來源。"
report["claims"][6]["artifact_ids"] += [receipt_id, "strict-context-execution-20261006"]
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
summary = {"verdict": "pass", "report_path": str(report_path.relative_to(REPO)), "report_sha256": digest(report_path),
           "source_sha256": EXPECTED_CURRENT, "prior_history_path": str(history.relative_to(REPO)), "prior_history_sha256": digest(history),
           "canonical_artifact_id": receipt_id, "receipt_path": str(receipt_path.relative_to(REPO)), "receipt_sha256": digest(receipt_path),
           "actual_scope": "Current7.8 full text, own-frozen diff, current7.4/7.7 full context, primary support excerpts,30 old artifact hashes, unchanged implementation/SVG hashes, two personally viewed original PNGs, one current7.7 strict-tolerance CPU fence", "unresolved_questions": []}
save("reinspection-summary.json", summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
