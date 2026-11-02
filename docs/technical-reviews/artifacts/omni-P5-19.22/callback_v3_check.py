"""Local wording callback; --acquire inspects primary metadata, default is portable.

No neural model, training, timing benchmark, or final question is executed.
"""

import argparse
import difflib
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REVIEW = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-19.22"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


parser = argparse.ArgumentParser()
parser.add_argument("--acquire", action="store_true")
args = parser.parse_args()
freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"
assert sha(freeze_path) == "8df31b044f819c10ba0db5db9ae65f36bd39134c017a646e902b7376e78cbcaa"
page = next(p for p in read(freeze_path)["pages"] if p["page_id"] == "19.22")
body = (ROOT / page["snapshot"]).read_text()
assert sha(ROOT / page["snapshot"]) == page["source_sha256"] == "6a8bf3cc3a61adc7610c5ac79c3a278ba378154e0ffb5f2ce981ed8f9c323cfb"
previous_path = REVIEW / "technical-current-v2.json"
assert sha(previous_path) == "1fe54354ce162634a3b0eaa07b7e9939857dc597de0f274e78bb61395e933fed"
previous = read(previous_path)
previous_freeze = read(ROOT / previous["source_freeze"]["path"])
old_page = next(p for p in previous_freeze["pages"] if p["page_id"] == "19.22")
old = (ROOT / old_page["snapshot"]).read_text()
old_phrase, new_phrase = "都是尚未實測的預定門檻", "都是在實測前固定的驗收門檻"
assert old.count(old_phrase) == body.count(new_phrase) == 1
assert old.replace(old_phrase, new_phrase, 1) == body
chapter_slice = re.search(r"(?ms)^## 19\.22 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text()).group()
assert chapter_slice == body
codes = re.findall(r"```python\n(.*?)\n```", body, re.S)
assert len(codes) == 2 and codes == re.findall(r"```python\n(.*?)\n```", old, re.S)
sources = {s["path"]: sha(ROOT / s["path"]) == s["sha256"] for s in previous["sources"] if s["kind"] == "repository_code"}
artifacts = {a["path"]: sha(ROOT / a["path"]) == a["sha256"] for a in previous["artifacts"]}
assert all(sources.values()) and all(artifacts.values())
assert page["figures_sha256"] == previous["figure_sha256"]
for path, digest in page["figures_sha256"].items():
    assert sha(ROOT / path) == digest
assert sha(ROOT / page["notebook"]) == page["notebook_sha256"] == "09837ac26f7736375e48e2a4300fa2d32544d0ed57589bb20f8040ecd6cfb671"
notebook = read(ROOT / page["notebook"])
cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
assert all(any(code.strip() == cell.strip() for cell in cells) for code in codes)
metadata_path = HERE / "callback-v3-provenance.json"
if args.acquire:
    raw_root = Path("/workspace/work/omni-final-completed-41004229262/verified")
    names = {"metrics": "frozen/metrics.json", "opening": "frozen/final-opening.json", "origin": "measurements/sound/part-origin.json", "completion": "measurements/sound/part-completion.json"}
    full = {key: read(raw_root / name) for key, name in names.items()}
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "kind": "reviewer primary metadata projection only; not collection/seal", "original_archive": read(HERE / "primary-projection.json")["primary_inputs"]["final_archive"], "inputs": {key: {"archive_member": name, **ref(raw_root / name), "created_at": full[key]["created_at"], **{k: full[key][k] for k in ["final_opened", "metrics_sha256", "measurement_completed", "results_sha256"] if k in full[key]}} for key, name in names.items()}, "sound_thresholds": full["metrics"]["sound"], "sound_results": ref(raw_root / "measurements/sound/results.json")}
    bundle_path = ROOT / "outputs/omni-P5-current-kernel-bundle-v8-receipt.json"
    bundle = read(bundle_path)
    row = next(row for row in bundle["rows"] if row["id"] == "19.22")
    metadata["author_bundle_provenance_only"] = {"path": bundle_path.relative_to(ROOT).as_posix(), **ref(bundle_path), "relevant_row": row, "scope": "Receipt metadata inspected only; author execution is not reviewer execution and is not a future-clone dependency."}
    assert (ROOT / "docs/omni-products/v1/metrics.json").read_bytes() == (raw_root / names["metrics"]).read_bytes()
    assert not metadata_path.exists()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
metadata = read(metadata_path)
metrics_ref = metadata["inputs"]["metrics"]
assert sha(ROOT / "docs/omni-products/v1/metrics.json") == metrics_ref["sha256"] == "cb291a08265a96526477bf8e0350243c8ddf9564600ba691d7a7f6aee3f6b336"
assert not metrics_ref["final_opened"]
assert metadata["inputs"]["opening"]["metrics_sha256"] == metadata["inputs"]["origin"]["metrics_sha256"] == metrics_ref["sha256"]
times = [metadata["inputs"][key]["created_at"] for key in ["metrics", "opening", "origin", "completion"]]
assert all(datetime.fromisoformat(a) < datetime.fromisoformat(b) for a, b in zip(times, times[1:]))
assert metadata["inputs"]["completion"]["measurement_completed"]
assert metadata["sound_results"]["sha256"] == metadata["inputs"]["completion"]["results_sha256"] == read(HERE / "primary-projection.json")["primary_inputs"]["sound_rows"]["sha256"]
t = metadata["sound_thresholds"]
assert (t["active_reference_snr_minimum_db"], t["gold_program_minimum_accuracy"], t["actual_AV_chain_minimum_accuracy"], t["legal_output_minimum_accuracy"], t["sample_rate"], t["maximum_seconds"], t["maximum_events"]) == (10, 0.85, 0.8, 1.0, 8000, 2, 4)
assert metadata["author_bundle_provenance_only"]["relevant_row"]["source"]["sha256"] == page["notebook_sha256"]
lines = body.splitlines()
result = {"created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": "/root/omni_p5_technical_19_22", "environment": {"python": platform.python_version(), "device": "CPU; stdlib file/JSON/hash checks only"}, "mode": "primary metadata acquisition and local recheck" if args.acquire else "portable saved metadata and source binding replay", "source_freeze": {"path": freeze_path.relative_to(ROOT).as_posix(), "sha256": sha(freeze_path)}, "page": page, "previous_full_review": {"path": previous_path.relative_to(ROOT).as_posix(), "sha256": sha(previous_path)}, "reading_scope": "Local same-reviewer callback: personally read L192–226 including complete threshold paragraph, previous/next transitions and formal-model lead-in, plus L450–476 covering prior issue closure, training completion, measured table and final timing limitations. Not a new blind or whole-page reread.", "exact_change": {"old": old_phrase, "new": new_phrase, "only_replacement_plus_no_other_bytes": True}, "diff": "".join(difflib.unified_diff(old.splitlines(True), lines_with_newlines := body.splitlines(True), fromfile="v6/current-v2", tofile="v8")), "read_excerpts": {"threshold_and_transitions_L205_215": "\n".join(lines[204:215]), "completed_measurement_status_L456_475": "\n".join(lines[455:475])}, "both_python_blocks_unchanged": True, "code_block_sha256": [hashlib.sha256(code.encode()).hexdigest() for code in codes], "current_notebook_code_matches": True, "current_notebook_sha256": page["notebook_sha256"], "figure_bytes_unchanged": page["figures_sha256"], "registered_implementation_sources_unchanged": sources, "previous_registered_artifacts_unchanged": artifacts, "metadata_projection": {"path": metadata_path.relative_to(ROOT).as_posix(), **ref(metadata_path)}, "threshold_timeline": {"order": ["metrics fixed", "final opened", "sound part started", "sound part completed"], "created_at_UTC": times, "metrics_hash_bound_by_opening_and_measurement": True, "sound_measurement_completed": True}, "passed": True, "no_formal_execution": True, "limits": "Uses original personally verified raw-measurement provenance and prior full-v6 review for unchanged material. No new NN, training, formal-final question, latency benchmark, waveform analysis, kernel, rendering, original-paper reread, or new provider-clock attestation performed. Default replay reads only tracked repository files and this small metadata projection; optional original-package and author-bundle inputs serve provenance only."}
print(json.dumps(result, ensure_ascii=False, indent=2))
