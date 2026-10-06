"""Inspect declared byte versions; do not infer scientific correctness or scope."""

from pathlib import Path
import datetime
import hashlib
import json
import re
import sys


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
digest = lambda raw: hashlib.sha256(raw).hexdigest()
progress_path = ROOT / "docs/course-revision-20261005/factual-review-progress.json"
ledger_path = ROOT / "docs/course-revision-20261005/continuity/original-review-callbacks.json"
progress_raw, ledger_raw = progress_path.read_bytes(), ledger_path.read_bytes()
progress, ledger = json.loads(progress_raw), json.loads(ledger_raw)
report_inputs, checked, candidates = [], [], []


def section_hash(source):
    path, sid = source.rsplit("#", 1)
    raw = (ROOT / path).read_bytes()
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    selected = [i for i, h in enumerate(headings) if h[0].startswith(("## " + sid + " ").encode())]
    if len(selected) != 1:
        raise ValueError(source)
    index = selected[0]
    end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
    return digest(raw[headings[index].start():end])


def compare(sid, pointer, source, expected, kind):
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        return
    actual = section_hash(source) if "#" in source else digest((ROOT / source).read_bytes())
    row = {"lesson_id": sid, "pointer": pointer, "source": source,
           "declared_sha256": expected, "actual_sha256": actual, "kind": kind,
           "matches": expected == actual}
    checked.append(row)
    if not row["matches"]:
        row["same_original_callback_status"] = ledger["records"].get(sid, {}).get("technical", {}).get("status")
        candidates.append(row)


def walk_current_context(sid, value, pointer, inherited_source=None):
    """Only explicit current context fields; historical/frozen inputs stay historical."""
    if isinstance(value, list):
        for i, item in enumerate(value):
            walk_current_context(sid, item, pointer + "/" + str(i), inherited_source)
    elif isinstance(value, dict):
        source = value.get("source", inherited_source)
        path = value.get("path")
        for key, item in value.items():
            child_pointer = pointer + "/" + key
            if any(word in key.lower() for word in ["prior", "previous", "initial", "old", "frozen", "historical"]):
                continue
            if key in ["source_sha256", "section_sha256", "current_sha256", "current_section_sha256", "current_context_sha256"] and isinstance(source, str):
                compare(sid, child_pointer, source, item, "explicit_current_context_section")
            elif key == "sha256" and isinstance(path, str):
                compare(sid, child_pointer, path, item, "explicit_current_context_file_or_snapshot")
            elif isinstance(item, (list, dict)):
                next_source = source
                if key in progress["records"]:
                    next_source = progress["records"][key]["source"]
                walk_current_context(sid, item, child_pointer, next_source)


for sid in progress["section_order"]:
    path = ROOT / "docs/technical-reviews" / (sid + ".json")
    raw = path.read_bytes()
    report = json.loads(raw)
    report_inputs.append({"lesson_id": sid, "path": str(path.relative_to(ROOT)), "sha256": digest(raw)})
    for i, source in enumerate(report["sources"]):
        if source.get("kind") == "repository_code":
            compare(sid, "/sources/" + str(i), source["path"], source.get("sha256"), "formal_repository_source")
    for path_text, expected in report.get("figure_sha256", {}).items():
        compare(sid, "/figure_sha256/" + path_text, path_text, expected, "declared_own_or_context_figure")
    for key, value in report.items():
        if key == "current_context_versions":
            walk_current_context(sid, value, "/" + key)
        elif key in ["context_reinspection", "context_reinspections"]:
            # Reinspection lists may hold genuine historical sessions. The latest
            # explicit current context map is inspected without rewriting history.
            latest = value[-1] if isinstance(value, list) and value else value
            if isinstance(latest, dict):
                for context_key, context_value in latest.items():
                    if context_key.startswith("current_context_"):
                        walk_current_context(sid, context_value, "/" + key + "/" + context_key)

assert progress_path.read_bytes() == progress_raw and ledger_path.read_bytes() == ledger_raw, "Progress changed during metadata scan."
assert all((ROOT / item["path"]).read_bytes() and digest((ROOT / item["path"]).read_bytes()) == item["sha256"] for item in report_inputs), "Report changed during metadata scan."
receipt = {
    "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "scope": "Formal repository-source hashes, declared figure hashes and explicitly structured current context versions. Historical/frozen whole-file inputs are not relabeled current. Unstructured context necessity is resolved by the same original owners and their callback receipts; this program does not review claims or infer that a dependency is scientifically sufficient.",
    "program_sha256": digest(Path(__file__).read_bytes()),
    "inputs": {"factual_progress_sha256": digest(progress_raw), "callback_ledger_sha256": digest(ledger_raw), "reports": report_inputs},
    "checked_entries": checked,
    "mismatch_candidates": candidates,
    "status": "no_byte_mismatch_candidates" if not candidates else "requires_original_owner_scope_resolution",
}
(OUT / "current-dependency-scan.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reports": len(report_inputs), "checked_entries": len(checked), "mismatch_candidates": candidates, "status": receipt["status"]}, ensure_ascii=False))
raise SystemExit(bool(candidates))
