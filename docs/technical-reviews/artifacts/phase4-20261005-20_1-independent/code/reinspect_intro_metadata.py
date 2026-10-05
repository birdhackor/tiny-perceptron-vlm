"""Record a real narrow metadata reinspection, preserving the initial PASS bytes."""
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
TASK = "/root/phase4_factual_coordinator/factual_20_1"
INITIAL_SHA = "043f6af13a946b9127a01e28e4673012256da271966201968a2a122475cc1b08"
REPORT = ROOT / "docs/technical-reviews/20.1.json"
HISTORY = BASE / "history/20.1.initial-pass.opaque.json"
RECEIPT = BASE / "execution/intro-metadata-reinspection.json"
RECEIPT_ID = "intro-metadata-reinspection"

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def path(p):
    return p.relative_to(ROOT).as_posix()

initial_bytes = REPORT.read_bytes()
assert digest(initial_bytes) == INITIAL_SHA
assert digest(HISTORY.read_bytes()) == INITIAL_SHA
assert HISTORY.read_bytes() == initial_bytes
report = json.loads(initial_bytes)
assert report["reviewer_task"] == TASK
assert TASK == "/root/phase4_factual_coordinator/factual_20_1"

chapter = (ROOT / "course/chapters/20.md").read_bytes()
chapter.decode("utf-8")
headings = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter))
section_index = next(i for i, heading in enumerate(headings) if heading[0].startswith(b"## 20.1 "))
intro = chapter[:headings[0].start()]
section = chapter[headings[section_index].start():headings[section_index + 1].start() if section_index + 1 < len(headings) else len(chapter)]
figure_path = ROOT / "course/figures/rewrite-20-input-routes.svg"
figure = figure_path.read_bytes()
assert intro == (BASE / "inputs/intro.md").read_bytes()
assert section == (BASE / "original-fence/section.md").read_bytes()
assert figure == (BASE / "figures/rewrite-20-input-routes.svg").read_bytes()
assert digest(section) == report["source_sha256"]
assert digest(figure) == report["figure_sha256"][path(figure_path)]
assert digest(intro) == report["introduction"]["sha256"]

summary = "本章延伸沿用已訓練的 Qwen 圖文核心，語音由 Whisper 先轉成文字再交給同一核心；其起點有別於第 19 章的隨機初始化主線。導言說明 LoRA 候選未被驗證支持採用，因此保留底座；短 CPU 範例只隔離機制，不能代表完整成熟模型的能力。"
receipt = {
    "reviewer_task": TASK,
    "recorded_at": datetime.now(UTC).isoformat(),
    "kind": "narrow_metadata_reinspection",
    "reason": "Coordinator identified missing current-round top-level intro_sha256 and intro_summary; the initial report already contained a genuine introduction read/hash/summary. This correction adds required metadata without changing the original first-read record.",
    "initial_report_history": {"path": path(HISTORY), "sha256": INITIAL_SHA, "preservation": "Exact original report bytes were preserved before any report change; initial verdict was pass, and it was not counted by the coordinator's current-round introduction gate."},
    "actual_read_scope": ["Personally reread all current raw UTF-8 chapter introduction bytes before the first ## heading", "Personally reread all current raw UTF-8 20.1 section bytes, including the fence and figure caption", "Compared current SVG bytes/hash to the exact SVG already rendered and personally viewed; no repeated visual inspection or browser run"],
    "version_checks": {
        "introduction": {"source": "course/chapters/20.md: raw bytes before first ##", "current_sha256": digest(intro), "saved_path": path(BASE / "inputs/intro.md"), "saved_sha256": digest((BASE / "inputs/intro.md").read_bytes()), "equal_bytes": True},
        "section": {"source": "course/chapters/20.md#20.1", "current_sha256": digest(section), "saved_path": path(BASE / "original-fence/section.md"), "saved_sha256": digest((BASE / "original-fence/section.md").read_bytes()), "equal_bytes": True},
        "figure": {"source": path(figure_path), "current_sha256": digest(figure), "saved_path": path(BASE / "figures/rewrite-20-input-routes.svg"), "saved_sha256": digest((BASE / "figures/rewrite-20-input-routes.svg").read_bytes()), "equal_bytes": True}
    },
    "top_level_fields_added": {"intro_sha256": digest(intro), "intro_summary": summary},
    "unchanged_evidence": "Original source/paper inspections, original fence and bounded CPU results, existing-data selection recomputation, and personally viewed frozen-SVG renders remain the earlier real evidence because the relevant introduction/section/figure bytes are identical. They were not rerun or backdated.",
    "canonical_task_assertion": "passed",
    "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-20261005-20_1-independent/code/reinspect_intro_metadata.py",
    "environment": {"python": sys.version, "device": "CPU metadata/byte verification only"},
    "unresolved": []
}
RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
receipt_sha = digest(RECEIPT.read_bytes())
report["intro_sha256"] = digest(intro)
report["intro_summary"] = summary
report["metadata_reinspection"] = {"artifact_id": RECEIPT_ID, "path": path(RECEIPT), "sha256": receipt_sha, "initial_report_history": receipt["initial_report_history"], "scope": "Actual narrow introduction/section reread and unchanged SVG-byte verification; metadata-only report correction."}
report["artifacts"].extend([
    {"id": "initial-pass-history", "path": path(HISTORY), "sha256": INITIAL_SHA, "kind": "source_snapshot", "description": "Opaque exact initial PASS report preserved before correction of missing top-level introduction metadata; first-read and prior checker history remain unchanged."},
    {"id": RECEIPT_ID, "path": path(RECEIPT), "sha256": receipt_sha, "kind": "source_snapshot", "description": "Actual fresh narrow metadata reinspection receipt: current/saved raw introduction, 20.1 and SVG identity, own top-level introduction summary, original report history path/hash and true non-rerun scope."},
    {"id": "intro-metadata-code", "path": path(Path(__file__).resolve()), "sha256": digest(Path(__file__).read_bytes()), "kind": "code", "description": "Exact source for this reviewer's narrow metadata correction and receipt generation."}
])
assert report["reviewer_task"] == TASK
assert report["intro_sha256"] == report["introduction"]["sha256"]
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reviewer_task": TASK, "canonical_task_assertion": "passed", "verdict": report["verdict"], "source_sha256": report["source_sha256"], "intro_sha256": report["intro_sha256"], "report_sha256": digest(REPORT.read_bytes()), "history_path": path(HISTORY), "history_sha256": INITIAL_SHA, "receipt_artifact_id": RECEIPT_ID, "receipt_path": path(RECEIPT), "receipt_sha256": receipt_sha, "actual_scope": receipt["actual_read_scope"], "unresolved": []}, ensure_ascii=False))
