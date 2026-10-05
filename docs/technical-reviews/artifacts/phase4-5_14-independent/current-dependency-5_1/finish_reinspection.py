"""This original owner's narrow semantic 5.1 dependency reinspection receipt/report."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from html.parser import HTMLParser

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[3]
REPORT = ROOT / "docs/technical-reviews/5.14.json"
OWNER = "/root/phase4_factual_coordinator/factual_5_14"

def h(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rel(path):
    return path.relative_to(ROOT).as_posix()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

prior_path = next(HERE.glob("prior-report-opaque-*.json"))
prior = json.loads(prior_path.read_text())
assert REPORT.read_bytes() == prior_path.read_bytes(), "preserve initial own report before update"
assert prior["reviewer_task"] == OWNER
own = HERE / "current-5.14.md"
current_ctx = HERE / "current-5.1.md"
frozen_ctx = BASE / "inputs/context-5.1.md"
assert own.read_bytes() == (BASE / "inputs/section.md").read_bytes()
assert h(own) == prior["source_sha256"]
frozen_sentence = "梯度為正只說這張表收到訊號，代價下降再說更新改善了固定題。"
current_sentence = "梯度大小大於0，只說這張表收到訊號；代價下降才說明更新改善了固定題。"
assert frozen_ctx.read_text().replace(frozen_sentence, current_sentence) == current_ctx.read_text()
integrity = json.loads((HERE / "dependency-integrity.json").read_text())
assert all(x.get("unchanged", x.get("byte_identical", False)) for x in integrity)

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = 0
        self.href = None
        self.label = []
        self.items = []
    def handle_starttag(self, tag, attrs):
        if tag == "article": self.inside += 1
        if tag == "a" and self.inside:
            self.href = dict(attrs).get("href")
            self.label = []
    def handle_endtag(self, tag):
        if tag == "article": self.inside -= 1
        if tag == "a" and self.href is not None:
            self.items.append({"href": self.href, "label": "".join(self.label)})
            self.href = None
    def handle_data(self, data):
        if self.href is not None: self.label.append(data)

parser = Links()
parser.feed((HERE / "ownpage-5.14.html").read_text())
links = [x for x in parser.items if x["label"].strip() in {"5.1", "5.2", "5.15"}]
assert {x["label"].strip() for x in links} == {"5.1", "5.2", "5.15"}
page_text = (HERE / "ownpage-article-text.txt").read_text()
assert "optimizer.step()" in page_text and "代碼每輪只計算new" in page_text
assert (BASE / "inputs/stdout.txt").read_text().strip() in page_text

receipt = {
    "schema_version": 1,
    "kind": "original_owner_current_dependency_reinspection",
    "reviewer_task": OWNER,
    "performed_at": datetime.now(UTC).isoformat(),
    "scope": "Only 5.14 necessary 5.1 dependency after a frozen current5.1 wording revision; not whole-chapter/new-identity review.",
    "prior_opaque": {"path": rel(prior_path), "sha256": h(prior_path), "bytes": prior_path.stat().st_size, "preservation": "original report bytes copied before reading/updating; historical original claims/verdict/evidence retained"},
    "current_own": {"source": "course/chapters/05.md#5.14", "path": rel(own), "source_sha256": h(own), "read_slice": "entire section original UTF-8; current lines520–551", "byte_identical_to_original_saved_own_source": True, "figure_sha256": {}},
    "current_context": {"source": "course/chapters/05.md#5.1", "path": rel(current_ctx), "section_sha256": h(current_ctx), "actual_read": "entire saved current5.1 from original MD; essential support slice is lines5–40, before details", "required_slice": {"path": rel(HERE / "current-5.1-required-slice.md"), "sha256": h(HERE / "current-5.1-required-slice.md"), "first_line": 5, "last_line": 40}},
    "frozen_original_context": {"source": "course/chapters/05.md#5.1", "path": rel(frozen_ctx), "sha256": h(frozen_ctx), "meaning": "actual originally saved context bytes; historical input, not current5.1"},
    "context_change_personally_read": {"old": frozen_sentence, "current": current_sentence, "other_bytes_in_5_1_identical": True, "meaning": "Correction refers to .grad.norm() magnitude rather than sign of each gradient coordinate. Current5.14 scalar gradient=-4 already shows negative gradient can carry valid signal; no scalar/function/update method changed."},
    "necessary_support_mapping": [
        {"claim_id": "update-diagnostic", "requires_from_5_1": "zero_grad -> forward/loss -> backward -> step sequence, correct optimizer parameters, gradient cleared only before calculation; do not infer improvement from signal alone", "current_contract": "same code and diagnostic sequence; corrected norm-magnitude sentence makes its signal/improvement distinction precise", "impact": "none; same claim remains verified by official step contract and own already-executed scalar backward-vs-step/precision probes"},
        {"claim_id": "answer-diagnostic", "requires_from_5_1": "all-ignored target is rejected, not success; actual averaging/count authority comes from repo loss_sum/masked_loss and necessary5.2", "current_contract": "same all-ignored exercise; model.py92–105 unchanged and personally reread named AST slices", "impact": "none; original count1/log3/ignored-grad0/all-ignored-error evidence remains valid"},
        {"claim_ids": ["local-direction", "start1-numbers", "fence-contract", "exercise-ratio", "pilot-search", "comparison-controls", "nonfinite-diagnostic"], "requires_from_5_1": "no dependence on norm-sign wording or5.1 model scores", "current_contract": "own5.14 original bytes, scalar fence, helper code and cited primary sources unchanged", "impact": "none; original numeric/CPU/primary-source evidence reused within its original scope"},
    ],
    "other_dependencies": json.loads((HERE / "other-required-context-integrity.json").read_text()),
    "dependency_integrity": {"path": rel(HERE / "dependency-integrity.json"), "sha256": h(HERE / "dependency-integrity.json"), "result": "all cited primary snapshots and current model/train/training dependencies match owner's original saved versions"},
    "new_source_contract_rereads": [
        {"path": rel(BASE / "sources/pytorch-v2.8.0-sgd.py"), "sha256": h(BASE / "sources/pytorch-v2.8.0-sgd.py"), "read_locator": "step105–150", "support": "optimizer.step performs parameter update; does not guarantee every step decreases objective"},
        {"path": rel(BASE / "sources/installed-sgd.py"), "sha256": h(BASE / "sources/installed-sgd.py"), "read_locator": "347–379", "support": "installed momentum/negative-LR update semantics remain same relevant contract"},
        {"path": rel(BASE / "sources/pytorch-v2.8.0-loss.py"), "sha256": h(BASE / "sources/pytorch-v2.8.0-loss.py"), "read_locator": "CrossEntropyLoss1158–1197,1236–1239", "support": "ignored answers do not contribute input gradient or mean denominator"},
        {"path": "tiny_perceptron/model.py", "full_file_sha256": h(ROOT / "tiny_perceptron/model.py"), "read_locator": "AST-named loss_sum92–100 and masked_loss103–105", "support": "same count-zero rejection,sum/count; saved exact reread slices current-loss_sum.py/current-masked_loss.py"},
    ],
    "unchanged_primary_paper_reuse": "Bengio v2/Smith v6 original owner inspected source snapshots unchanged; no new download/search/re-reading claim; retain original algorithm/scope limitations",
    "reused_cpu": {"path": rel(BASE / "bounded-results.json"), "sha256": h(BASE / "bounded-results.json"), "actually_rechecked_pointers": ["/backward_vs_step", "/display_precision", "/target_denominator", "/assertions"], "pointer_snapshot": rel(HERE / "reused-cpu-pointers.json"), "pointer_snapshot_sha256": h(HERE / "reused-cpu-pointers.json"), "new_cpu_execution": False, "reason_reuse_valid": "unchanged source fence, actual helper and optimizer update semantics; only context prose norm-sign clarification; no uncertainty requires rerun"},
    "exact_ownpage_read": {"url": "http://127.0.0.1:8765/5.14.html", "observed_status": 200, "actual_read": "actual HTML response and its5.14 article text/code, displayed CPU result and dependency links, not root existence or global parity", "html_path": rel(HERE / "ownpage-5.14.html"), "html_sha256": h(HERE / "ownpage-5.14.html"), "article_text_path": rel(HERE / "ownpage-article-text.txt"), "article_text_sha256": h(HERE / "ownpage-article-text.txt"), "relevant_links": links, "visual_layout_test": "not performed/no browser tool; this technical reinspection makes no browser layout claim; own5.14 has no figures"},
    "method": {"path": rel(HERE / "current-factual-reviewer-instructions.md"), "sha256": h(HERE / "current-factual-reviewer-instructions.md"), "actual_read": "complete latest methods before semantic reinspection"},
    "original_full_chapter_hash_semantics": "Original extraction/provenance file whole-chapter hash remains an at-that-time observation, not current chapter version. No full old chapter was saved then; actual preserved inputs are original5.14 bytes and named context slices. This receipt only asserts saved own-section and required-context versions, not fictional whole-chapter parity.",
    "excluded_support": "5.1 details'600-step GPU model measurements were incidentally read with the complete current context but are not evidence for5.14, whose scalar outputs and diagnostic claims depend on neither those numbers nor a model-training result. No original result JSON inspection or new model score is claimed.",
    "verdict": "pass",
    "unresolved_issues": [],
    "training_gpu_downloads": "none",
    "report_update_code": {"path": rel(Path(__file__)), "sha256": h(Path(__file__))},
}
receipt_path = HERE / "reinspection-receipt.json"
write(receipt_path, receipt)
receipt_id = "reinspection-current-5_1-20261005"

report = prior
original_artifact_ids = {x["id"] for x in report["artifacts"]}
new_artifacts = []
for n, p in enumerate(sorted(HERE.iterdir()), 1):
    if not p.is_file() or p.name.startswith("checker-"):
        continue
    identifier = receipt_id if p == receipt_path else f"dependency5_1-{n}"
    assert identifier not in original_artifact_ids
    new_artifacts.append({"id": identifier, "path": rel(p), "sha256": h(p), "kind": "code" if p.suffix == ".py" else "source_snapshot", "description": "原owner窄current5.1依賴複查：" + p.name})
report["artifacts"].extend(new_artifacts)
report["actual_read_scope"] += "；原owner2026-10-05窄複查：最新完整方法、current5.14原MD全文與精確ownpage5.14.html article、current5.1原MD全文（必要支持slice5–40）；named loss_sum/masked_loss與官方/installed SGD、CrossEntropyLoss必要行；5.2/5.15相同原bytes只核完整指紋後復用原讀取，不宣稱重讀/重跑。"
report["dependency_reinspection"] = {"artifact_id": receipt_id, "path": rel(receipt_path), "sha256": h(receipt_path), "reviewer_task": OWNER, "verdict": "pass", "unresolved_issues": []}
report["current_context_versions"] = [receipt["current_context"], *receipt["other_dependencies"]]
report["frozen_context_versions"] = [receipt["frozen_original_context"]]
report["prior_report_history"] = receipt["prior_opaque"]
for c in report["claims"]:
    if c["id"] in {"update-diagnostic", "answer-diagnostic"}:
        c["artifact_ids"].append(receipt_id)
        c["scope"] += "；current5.1僅把訊號說法由梯度正號改為梯度範數大於0，更新/空答案契約未變；原owner具體依賴複查見reinspection receipt。"
for name in ["factual_accuracy", "source_verification", "limitations"]:
    report["checks"][name]["details"] += " 原ownercurrent5.1窄依賴複查完成：norm大小用詞修正，step/空答案方法和原主源/CPU證據支持不變；未重新訓練。"
report["summary"] += " 原owner完成current5.1必要依賴窄複查，維持pass，舊報告opaque及舊證據保留。"
write(REPORT, report)
assert json.loads(REPORT.read_text())["dependency_reinspection"]["sha256"] == h(receipt_path)
print(json.dumps({"new_report_path": rel(REPORT), "new_report_sha256": h(REPORT), "own_source_sha256": h(own), "current_context5_1_sha256": h(current_ctx), "frozen_context5_1_sha256": h(frozen_ctx), "figure_sha256": {}, "prior_opaque_path": rel(prior_path), "prior_opaque_sha256": h(prior_path), "own_receipt_id": receipt_id, "own_receipt_path": rel(receipt_path), "own_receipt_sha256": h(receipt_path), "verdict": "pass", "unresolved_issues": []}, ensure_ascii=False, indent=2))
