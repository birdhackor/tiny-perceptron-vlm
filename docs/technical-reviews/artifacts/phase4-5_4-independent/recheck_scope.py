"""Record the original reviewer's actual complete-section scope recheck, preserving initial history."""
import copy
import difflib
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RECHECK = OUT / "recheck"
RECHECK.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "docs/review-tools"))
from section_facts import fences, original_section

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def rawsha(b):
    return hashlib.sha256(b).hexdigest()
def save(name, data):
    (RECHECK / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

initial_path = OUT / "report-revise-initial.json"
initial_report = json.loads(initial_path.read_text())
initial_sha = sha(initial_path)
assert initial_sha == "6b6fa749b0dded74e1b85fc1d26d7a02672f6356858dc8f2dc479768b9306764"
opaque_history = ROOT / "docs/technical-reviews/history/phase4-5_4-own-initial-revise-6b6fa749b0dded74e1b85fc1d26d7a02672f6356858dc8f2dc479768b9306764.json"
assert initial_path.read_bytes() == opaque_history.read_bytes()
assert initial_report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_5_4"
assert initial_report["verdict"] == "revise"

raw, whole, first_line = original_section(ROOT / "course/chapters/05.md", "5.4")
old_raw = (OUT / "original/section.md").read_bytes()
body = raw.decode("utf-8")
assert body.startswith("## 5.4 ")
assert "在這個從零歷史開始的第一步，應減去的更新量與當前梯度同號" in body
assert "後續更新量的正負由歷史平均 `m_hat` 決定，可能與當前梯度不同號" in body
assert "Adam調步幅沒有顛倒下降的基本方向" not in body
new_fences = fences(raw, first_line)
assert len(new_fences) == 1
assert new_fences[0]["raw"] == (OUT / "original/fence-1.py").read_bytes()
assert rawsha(new_fences[0]["raw"]) == "90bd4020dc21e94d3b09c458d787a8f7d1814fcd670089c118e5e1e39830f7e3"
old_lines = old_raw.splitlines(keepends=True)
new_lines = raw.splitlines(keepends=True)
assert len(old_lines) == len(new_lines)
changed_lines = [(i + first_line, a.decode("utf-8"), b.decode("utf-8"))
                 for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
assert len(changed_lines) == 1 and changed_lines[0][0] == 141
(RECHECK / "section.md").write_bytes(raw)
diff = "".join(difflib.unified_diff(old_raw.decode().splitlines(keepends=True), body.splitlines(keepends=True), fromfile="initial-review-5.4", tofile="current-5.4"))
(RECHECK / "section.diff").write_text(diff)

reused = []
for art in initial_report["artifacts"]:
    p = ROOT / art["path"]
    observed = sha(p)
    assert observed == art["sha256"], (art["id"], observed, art["sha256"])
    reused.append({"id": art["id"], "path": art["path"], "sha256": observed, "unchanged": True})

results = json.loads((OUT / "bounded-results.json").read_text())
example = results["variants"]["negative_second_gradient"]["values"]
history = results["history_and_resume"]
assert example["g"]["values"][1] < 0 and example["m_hat"]["values"][1] < 0 and example["update"]["values"][1] < 0
assert history["gradient_sequence"][1][1] < 0 and history["m_hat2"][1] > 0 and history["update2"][1] > 0

command = ".venv/bin/python docs/technical-reviews/artifacts/phase4-5_4-independent/recheck_scope.py > docs/technical-reviews/artifacts/phase4-5_4-independent/recheck/recheck.stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_4-independent/recheck/recheck.stderr.txt"
record = {
    "reviewer_task": initial_report["reviewer_task"], "rechecked_utc": datetime.now(UTC).isoformat(),
    "review_mode": "Same independent technical reviewer personally reread the complete revised 5.4, after initial revise; no reader answers used.",
    "original_source_sha256": initial_report["source_sha256"], "new_source_sha256": rawsha(raw),
    "source_file_sha256": rawsha(whole), "section_first_line": first_line,
    "read_scope": "Complete current course/chapters/05.md#5.4, including opening, entire original fence, bias correction, epsilon, continuation and complete revised exercise/details; prior linked 1.12 sign semantics retained.",
    "actual_changed_lines": changed_lines, "current_fence_sha256": rawsha(new_fences[0]["raw"]),
    "fence_byte_identity": True, "figure_references": [],
    "primary_source_reinspection": [
        {"source": "sources/adam-1412.6980v9.txt and previously personally viewed original PDF page2", "locator": "paper p2 Algorithm1 and §2; p3 §2.1 update and direction-of-m_hat paragraph", "finding": "At zero-history t=1, m_hat=g. General update uses m_hat/(sqrt(v_hat)+epsilon), with positive denominator and learning rate, so sign follows m_hat rather than necessarily current g. The revised wording states precisely this scope."},
        {"source": "sources/torch-adam-installed-commit.py", "locator": "L300–311 formula; L456–475 first/second moment updates; L528–546 bias corrections and negative addcdiv", "finding": "The recurrence, positive denominator and subtraction support the new explicit distinction between zero-history first-step sign and general moment-history sign."},
    ],
    "retained_first_step_evidence": {"gradient": example["g"]["values"], "m_hat": example["m_hat"]["values"], "update": example["update"]["values"], "meaning": "For the revised first-step exercise, second g/m_hat/update are all negative, and subtracting the update raises that parameter."},
    "retained_history_counterexample": {"gradient_sequence": history["gradient_sequence"], "m_hat2": history["m_hat2"], "update2": history["update2"], "meaning": "The earlier counterexample now illustrates the newly stated caveat: second current g is negative but historical m_hat and amount to subtract are positive."},
    "prior_execution_reuse": {"rerun": False, "reason": "Only source line141 changed; original fence bytes, primary snapshots and all existing evidence hashes are unchanged. No changed arithmetic or API needs rerunning. Prior CPU checks remain dated initial-review executions, not new measurements.", "artifacts_hash_checked": reused},
    "initial_report": {"path": str(initial_path.relative_to(ROOT)), "sha256": initial_sha, "opaque_coordinator_history_path": str(opaque_history.relative_to(ROOT)), "byte_identical": True},
    "issue": "I1-direction-scope", "issue_resolution": "Revised exercise explicitly limits current-gradient agreement to zero-history first step, retains subtraction-of-negative meaning, and states later update sign is from m_hat and may differ from current g. This removes the initial scope ambiguity without implying monotonic loss decrease.",
    "verdict": "pass", "command": command,
    "environment": {"python": sys.version, "executable": sys.executable, "device": "cpu evidence reused", "cwd": str(ROOT), "shell": "bash", "login": "false"},
    "restrictions_observed": "Only own report and artifacts changed. No body/figures changed, child agents, training, GPU, data/model downloads, uploads, paid compute, reader answers, or commit.",
}
save("recheck-record.json", record)

report = copy.deepcopy(initial_report)
report["source_sha256"] = rawsha(raw)
report["verdict"] = "pass"
report["summary"] = "C1–C8原始公式、数值、API与程式证据保持有效；完整新版复读后，C9明确零历史首步与后续m_hat方向范围，I1已解决。"
report["review_scope"] += " 同一独立reviewer親读完整修訂版5.4，重新对照原方法/官方更新式與保留反例；只改末段，原CPU证据透明沿用，没有冒充重新运行。"
report["reviewed_on"] = "2026-10-05"
report["author_tasks"] = ["/root/phase4_factual_coordinator"]
report["author_task_scope"] = "此task已明確以協調者身分作本次line141局部修订；未猜測更早作者身分。"
report["history"] = [{"verdict": "revise", "source_sha256": initial_report["source_sha256"], "report_path": str(initial_path.relative_to(ROOT)), "report_sha256": initial_sha, "claim_id": "C9", "issue_id": "I1-direction-scope", "original_claim": copy.deepcopy(next(c for c in initial_report["claims"] if c["id"] == "C9")), "original_issue": copy.deepcopy(initial_report["issues"][0]), "preserved": True}]
report["rechecks"] = [{"record_path": str((RECHECK / "recheck-record.json").relative_to(ROOT)), "record_sha256": sha(RECHECK / "recheck-record.json"), "reviewer_task": initial_report["reviewer_task"], "read_full_section": True, "old_source_sha256": initial_report["source_sha256"], "new_source_sha256": rawsha(raw), "issue_id": "I1-direction-scope", "result": "resolved", "prior_cpu_checks_reused": True, "cpu_rerun": False}]
for source in report["sources"]:
    if source["id"] == "original_code":
        source["version"] = f"原初审节SHA-256 {initial_report['source_sha256']}；原fence L121–132；现修订节SHA-256 {rawsha(raw)}的fence逐byte相同。"
        source["inspection_note"] += " 本次完整复读新版并实际核对原fence bytes/hash未变，透明沿用原CPU运行事实。"
for art in report["artifacts"]:
    if art["id"] == "inspection":
        art["description"] = "保留初审亲读／推导与当时revise的scope疑问历史；本次解决记录另见recheck_scope，未覆盖或伪造旧判断。"
def artifact(identifier, path, kind, desc):
    return {"id": identifier, "path": str(path.relative_to(ROOT)), "sha256": sha(path), "kind": kind, "description": desc}
report["artifacts"].extend([
    artifact("initial_review_history", initial_path, "source_snapshot", "本人初始revise报告原bytes，C9/I1原问题与完整证据保留；与协调者opaque history SHA相同。"),
    artifact("rechecked_section", RECHECK / "section.md", "source_snapshot", "本人完整重读的目前5.4原始UTF-8 bytes，未正规化换行。"),
    artifact("recheck_diff", RECHECK / "section.diff", "source_snapshot", "初审版到完整新版实际diff；仅line141作用域与m_hat符号说明改变。"),
    artifact("recheck_code", Path(__file__), "code", "本次实际执行的原bytes/hash比对、证据沿用核验与本人报告更新代码。"),
    {**artifact("recheck_scope", RECHECK / "recheck-record.json", "execution", "本人真实完整复读、原来源再读定位、反例重新核对、原CPU证据hash实际验证与I1解决记录。"), "command": command, "result": "Only line141 changed; original fence byte-identical; all 24 prior artifact hashes unchanged; current wording agrees with original paper/official source and retained first-step/history examples. No CPU arithmetic rerun.", "environment": {"python": "3.13.5", "torch_for_retained_cpu_evidence": "2.14.1+cpu", "device": "cpu", "cwd": str(ROOT), "shell": "bash login:false"}},
])
c9 = next(c for c in report["claims"] if c["id"] == "C9")
c9["original_statement"] = c9["statement"]
c9["original_status"] = c9["status"]
c9["statement"] = "零历史第一步，应减去的更新量与当前梯度同号；后续更新量正负由历史平均m_hat决定，可能与当前梯度不同号。"
c9["scope"] = "本節α=.001、ε=1e−8，标准Adam零初始化；第一步m_hat=g，因此正分母保留g符号。后续正分母保留m_hat符号，m_hat与当前g不必同号。没有每步代价下降或每步沿当前负梯度方向保证。"
c9["status"] = "verified"
c9["evidence"].append({"source_id": "bounded_execution", "locator": "retained bounded-results.json variants.negative_second_gradient and history_and_resume; actual new wording inspected in recheck/recheck-record.json", "supports": "同一原首步和两步反例与新版两个分别限定的方向陈述一致。原程序/证据hash未变，未把沿用说成复跑。"})
c9["artifact_ids"].extend(["recheck_scope", "rechecked_section", "initial_review_history"])
issue = report["issues"][0]
issue["initial_resolution"] = issue["resolution"]
issue["status"] = "resolved"
issue["resolution"] = record["issue_resolution"]
issue["resolved_source_sha256"] = rawsha(raw)
issue["resolved_by"] = initial_report["reviewer_task"]
issue["recheck_artifact_id"] = "recheck_scope"
report["checks"]["factual_accuracy"]["status"] = "pass"
report["checks"]["factual_accuracy"]["details"] = "C1–C8已有primary-source与实际CPU证据，代码/hash未变；本人完整读新版后C9作用域与原更新式、保留两步反例相符，I1解决。"
report["checks"]["limitations"]["status"] = "pass"
report["checks"]["limitations"]["details"] = "新版明示首步与后续m_hat方向区别，没有一般每步当前梯度下降方向保证。仍仅手填梯度/计算应减量，非求导/实际训练/评测；原CPU证据透明沿用。"
report["verification_boundary"] += " 本次不重跑未变CPU示例；实际读取完整新节与精确原来源片段、检查所有旧artifact hash并核对原反例，新增recheck证据明确标示scope。"
(ROOT / "docs/technical-reviews/5.4.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"verdict": "pass", "source_sha256": rawsha(raw), "changed_lines": [r[0] for r in changed_lines], "fence_unchanged": True, "reused_artifact_hashes": len(reused), "cpu_rerun": False, "issue": "I1-direction-scope resolved by same reviewer"}, ensure_ascii=False))
