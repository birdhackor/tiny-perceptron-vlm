"""本人 source15 依賴 callback；保留 source14 實查原件。"""
import datetime
import difflib
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REVIEWS = ROOT / "docs/course-revision-20261029-omni/reviews"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


if __name__ == "__main__":
    original_path = REVIEWS / "technical/technical_validation/technical-original-v1.json"
    original = read(original_path)
    freeze_paths = {str(n): REVIEWS / f"source-freeze-technical-fix{n}.json" for n in (13, 14, 15)}
    freezes = {n: read(p) for n, p in freeze_paths.items()}
    pages = {n: {x["page_id"]: x for x in f["pages"]} for n, f in freezes.items()}
    changed = [pid for pid in pages["14"] if pages["14"][pid] != pages["15"].get(pid)]
    assessments = {
        "5.7": "Read complete old/current section and exact delta. The change narrows weights_only=True from an absolute no-program-execution guarantee to restrictions reducing risk. validation makes no pickle/security guarantee. Its checkpoint/resume statements use unchanged actual trainer/smoke receipts and code, not this old safety sentence; no necessary premise changes.",
        "chapter-10": "Read both complete introductions. The change names10.9–10.11 and adds the10.12 timestamp destination. validation's distinction between runnable image inputs and learned ability already rests on unchanged implementation and heldout results, not this introduction's last-three wording. No claim of new A training is introduced.",
        "chapter-14": "Read both complete introductions. The change corrects the one-layer model link4.5->4.6 and adds14.19 as the final coordinate-rotation section. validation's B/C parameter, update and performance claims are grounded in exact historical raw receipts/journals, not the introduction's route numbering. No measured capability or necessary mechanism changes.",
    }
    deltas = []
    for pid in changed:
        old, new = pages["14"][pid], pages["15"][pid]
        a, b = ROOT / old["snapshot"], ROOT / new["snapshot"]
        assert sha(a) == old["source_sha256"] and sha(b) == new["source_sha256"]
        deltas.append({"page_id": pid, "before": old, "after": new, "diff": list(difflib.unified_diff(a.read_text().splitlines(), b.read_text().splitlines(), lineterm="")), "necessary_dependency_assessment": assessments[pid]})
    sources = []
    for source in original["sources"]:
        if source["kind"] == "repository_code":
            p = ROOT / source["path"]
            sources.append({"id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": sha(p), "unchanged": sha(p) == source["sha256"]})
    artifacts = [{"id": a["id"], **ref(ROOT / a["path"]), "unchanged": sha(ROOT / a["path"]) == a["sha256"]} for a in original["artifacts"]]
    probes = read(HERE / "bounded-probes-v2.json")
    raw = []
    for label, row in [(k, v) for k, v in probes.items() if isinstance(v, dict) and "log_path" in v] + [("global_gate_" + str(i), v) for i, v in enumerate(probes["current_global_gates"])]:
        p = Path(row["log_path"])
        raw.append({"label": label, "path": str(p), "sha256": sha(p), "bytes": p.stat().st_size, "matches_saved_receipt": sha(p) == row["log_sha256"]})
    target = {n: {"freeze": ref(freeze_paths[n]), "page": p["validation"]} for n, p in pages.items()}
    assert set(changed) == set(assessments)
    assert all(p["validation"]["source_sha256"] == original["source_sha256"] and p["validation"]["figures_sha256"] == {} for p in pages.values())
    assert [s["id"] for s in sources if not s["unchanged"]] == ["s_policy"]
    assert all(a["unchanged"] for a in artifacts) and all(x["matches_saved_receipt"] for x in raw)
    assert sha(ROOT / "docs/course-revision-20261029-omni/REVIEW-PROTOCOL.md") == "bcfa804b49c029e875f83174ff75990ce14fe4d5fad06f382c1c70aa289ea10e"
    result = {"created_at": datetime.datetime.now(datetime.UTC).isoformat(), "command": "uv run --extra cpu python " + str(Path(__file__).relative_to(ROOT)), "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU metadata only"}, "original": ref(original_path), "target_freezes": target, "source14_to15_complete_canonical_delta": deltas, "registered_repository_sources": sources, "original_artifacts": artifacts, "callback_raw_log_integrity": raw, "claims_covered": [c["id"] for c in original["claims"]], "claim_dependency_judgment": "All99 original full-page claims remain within the same necessary premises: target and necessary W.1/20.8/20.12/20.13 material and registered code/primary inputs unchanged. s_policy alone changed since original; its genuine semantic review is preserved in callback-input-audit-v2.json. Current protocol stage order/full affected-unit recheck rules remain unchanged; the new final figure/publication requirements are not asserted as completed by this page. Source15's three exact deltas above add no required verification for the validation claims.", "limits": ["Post-original informed callback, not a second fresh original.", "Source14 probe executions retain their actual times and source15 confirms relevant source bytes unchanged; no rerun, full331 notebook, full pytest, final seal or global publication pass is inferred.", "The five original pytest failures and isolated baseline comparison remain immutable; the133-case later subset passed and does not replace the old full-suite record.", "No training, model forward, GPU, paid action, source/checker edit or commit."]}
    out = HERE / "source15-dependency-audit-v1.json"
    with out.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"output": ref(out), "changed_pages": changed, "claims": len(result["claims_covered"]), "raw_logs_verified": len(raw)}))
