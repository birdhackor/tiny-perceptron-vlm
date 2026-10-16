"""REVIEW-LATE-01 窄範圍獨立fixture；不將虛構收據冒稱真實派工。"""

import copy
import hashlib
import importlib.util
import json
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = datetime.now(UTC).isoformat()
    spec = importlib.util.spec_from_file_location("independent_late_review", ROOT / "docs/review-tools/incremental_revision.py")
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    rootwork = ROOT / "work"
    rootwork.mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(dir=rootwork, prefix="late-collection-") as folder:
        folder = Path(folder)
        counter = 0

        def save(value):
            nonlocal counter
            counter += 1
            path = folder / f"receipt-{counter}.json"
            path.write_text(json.dumps(value) + "\n")
            return {"path": path.relative_to(folder).as_posix(), "sha256": digest(path)}

        def fixture():
            refs = [save({"synthetic_report": n}) for n in (1, 2)]
            seals = {}
            for i, role in enumerate(review.ROLES, 1):
                seals[role] = save({"role": role, "status": "completed", "started_at": f"2026-10-16T0{i}:00:00+00:00",
                                    "completed_at": f"2026-10-16T0{i}:20:00+00:00",
                                    "report_sha256s": [r["sha256"] for r in refs] if role == "continuity" else []})
            return [{"id": "fixture", "seals": seals}], refs

        def load(ref):
            return json.loads((folder / ref["path"]).read_bytes())

        def run(name, mutation, *, selected=("reader", "technical"), expect="reject", shape="section"):
            waves, refs = fixture()
            proof = {"wave": "fixture", "collection_role": "continuity"}
            if shape == "section":
                proof.update(mode="section-composite", section_reports=refs)
                required = refs
            else:
                proof["report"] = refs[0]
                required = refs[:1]
            # 每個反例都先以同一批正常收據通過，避免壞基底冒稱阻擋。
            sealed = review.review_seals(folder, waves, selected)
            review.require_collected(proof, "technical", sealed, required)
            mutation(waves, proof, refs)
            try:
                sealed = review.review_seals(folder, waves, selected)
                review.require_collected(proof, "technical", sealed, required)
            except (ValueError, KeyError) as error:
                actual, failure = "reject", str(error)
            else:
                actual, failure = "pass", None
            results.append({"id": name, "shape": shape, "selected": selected, "base_passed": True,
                            "expected": expect, "actual": actual, "failure": failure, "passed": actual == expect})

        run("late_section_reports_accepted", lambda *_: None, expect="pass")
        run("late_tool_audit_accepted", lambda *_: None, expect="pass", shape="tool")

        def remove_member(waves, proof, refs):
            seal = load(waves[0]["seals"]["continuity"])
            seal["report_sha256s"].remove(refs[-1]["sha256"])
            waves[0]["seals"]["continuity"] = save(seal)

        run("late_section_all_members_required", remove_member)
        run("late_tool_member_required", lambda w, p, r: remove_member(w, p, r[:1]), shape="tool")
        run("late_section_cannot_claim_early_technical", lambda w, p, r: p.pop("collection_role"))
        run("late_tool_cannot_claim_early_technical", lambda w, p, r: p.pop("collection_role"), shape="tool")
        run("technical_cannot_collect_as_reader", lambda w, p, r: p.update(collection_role="reader"))
        run("unknown_collection_role_rejected", lambda w, p, r: p.update(collection_role="invented"))
        run("missing_later_collection_stage_rejected", lambda w, p, r: w[0]["seals"].pop("continuity"))

        def change_seal(waves, field, value):
            seal = load(waves[0]["seals"]["continuity"])
            seal[field] = value
            waves[0]["seals"]["continuity"] = save(seal)

        run("technical_subgate_checks_later_overlap", lambda w, p, r: change_seal(w, "started_at", "2026-10-16T02:10:00+00:00"))
        run("reader_subgate_checks_later_overlap", lambda w, p, r: change_seal(w, "started_at", "2026-10-16T02:10:00+00:00"), selected=("reader",))
        run("reader_subgate_checks_later_wrong_role", lambda w, p, r: change_seal(w, "role", "technical"), selected=("reader",))
        run("technical_subgate_checks_later_reversed_duration", lambda w, p, r: change_seal(w, "completed_at", "2026-10-16T02:59:00+00:00"))

        def tamper_seal(waves, proof, refs):
            (folder / waves[0]["seals"]["continuity"]["path"]).write_text("{}")

        run("reader_subgate_checks_later_seal_sha", tamper_seal, selected=("reader",))

        def overlap_wave(waves, proof, refs):
            waves.append({"id": "later-fixture", "seals": {"reader": waves[0]["seals"]["reader"]}})

        run("provided_later_seal_constrains_next_wave", overlap_wave, selected=("reader",))
        waves, refs = fixture()
        waves[0]["seals"] = {"reader": waves[0]["seals"]["reader"]}
        sealed = review.review_seals(folder, waves, ("reader",))
        assert set(sealed["fixture"]) == {"reader"}
        results.append({"id": "reader_only_unprovided_future_not_invented", "expected": "pass", "actual": "pass", "passed": True})

    output = {"reviewer_task": "/root/tech_20261016_review_tools", "initial_context": "fresh", "current_context": "informed",
              "started_at": started, "completed_at": datetime.now(UTC).isoformat(),
              "environment": {"executable": sys.executable, "python": sys.version, "platform": platform.platform()},
              "checked_implementation": {"path": "docs/review-tools/incremental_revision.py", "sha256": digest(ROOT / "docs/review-tools/incremental_revision.py")},
              "results": results, "all_passed": all(r["passed"] for r in results),
              "limit": "全部收據為隔離synthetic fixture；不是當前真實all manifest或歷史收件的獨立證明。"}
    path = ART / "tool-audit-callback-late-probes-results-04.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"probes": len(results), "all_passed": output["all_passed"], "failures": [r for r in results if not r["passed"]],
                      "result_sha256": digest(path)}, ensure_ascii=False))
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
