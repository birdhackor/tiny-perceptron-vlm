"""同 owner 收件後的定點 retention：只核身分與既有原始觀察。"""

import hashlib
import json
import math
import platform
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("retention-audit.json")
PREFIX = "tiny-perceptron-c-v3/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw(path):
    return (ROOT / path).read_bytes()


def load(path):
    return json.loads(raw(path))


def main():
    started = datetime.now(UTC).isoformat()
    original_path = "docs/course-revision-20261101-c-v3/technical-v21/canonical_19_13/original-v1.json"
    intake_path = "docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/canonical-19-13-original-v1.json"
    assert sha(raw(original_path)) == "c87b75357359ee693b143190f52d3d7c1f65e0da8fa143c0a0fca5713163f96f"
    assert sha(raw(intake_path)) == "860d18e1918c065d0863e89d52c3ed8ca1bb08c0a2d003bb4e9a3a7aaf6b0250"
    intake = load(intake_path)
    assert intake["report"]["sha256"] == sha(raw(original_path))
    prior_path = "docs/course-revision-20261029-omni/reviews/technical/technical_19_13/technical-original-v1.json"
    prior = load(prior_path)
    assert sha(raw(prior_path)) == "472c94afcee6a58937b8d575fb01d4b8d23126c984d9fb2c235f4cb0d208c962"
    code_identity = []
    for source in prior["sources"]:
        if source["kind"] == "repository_code":
            actual = sha(raw(source["path"]))
            assert actual == source["sha256"]
            code_identity.append({"source_id": source["id"], "path": source["path"], "sha256": actual})
    retained_artifacts = []
    for item in prior["artifacts"]:
        actual = sha(raw(item["path"]))
        assert actual == item["sha256"]
        retained_artifacts.append({"id": item["id"], "path": item["path"], "sha256": actual})
    old = load("docs/technical-reviews/artifacts/omni-technical-19-13-v1/execution-v1.json")
    original_sources = load("docs/technical-reviews/artifacts/omni-technical-19-13-v1/source-receipts-v1.json")
    histories = {}
    for name in ("b", "c"):
        execution = load(f"docs/lesson-products/results/initial-{name}-execution.json")
        inspection = load(f"docs/lesson-products/results/initial-{name}-inspection.json")
        canonical = json.dumps(execution, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert sha(canonical) == inspection["receipt_canonical_sha256"]
        training = execution["training"]
        for key in ("seed", "steps_this_attempt", "initial_weights_sha256", "selected_weights_sha256"):
            assert training[key] == old["history"][name][key]
        assert training["initial_weights_sha256"] != training["selected_weights_sha256"]
        histories[name] = {key: training[key] for key in ("run_identity", "seed", "steps_this_attempt", "batch_size", "initial_weights_sha256", "selected_weights_sha256", "environment")}
    moe = load("docs/selftrained/results/public-raw/moe/test/metrics.json")
    tool = moe["per_task_final_reply"]["tool_call"]["tool_roundtrip"]
    assert tool["numerator"] == 0 and tool["denominator"] == 276
    assert tool["rate"] < moe["thresholds"]["tool_roundtrip"]
    bc = load("docs/lesson-products/results/final-report-v1.json")
    assert bc["settings_freeze_sha256"] == sha(raw("docs/lesson-products/results/settings-freeze-v1.json"))
    assert bc["b"]["5"]["boxes"]["correct"] == 2928
    assert bc["b"]["5"]["sum"]["correct"] == 2
    freeze = load("docs/course-revision-20261101-c-v3/source-freeze-v21.json")
    dependency_boundaries = []
    for lesson in ("7.23", "9.11", "19.15", "19.19", "19.20", "19.21", "19.22", "19.23", "19.24"):
        page = next(p for p in freeze["pages"] if p["page_id"] == lesson)
        data = raw(page["snapshot"])
        assert sha(data) == page["source_sha256"]
        paragraphs = re.split(r"\n\s*\n", data.decode())
        selected = [p for p in paragraphs if not p.startswith(("    ", "|")) and any(k in p for k in ("這個CPU", "本節展示", "正式模型", "本節C-v2", "CPU計算", "兩段程式", "思考投入", "Jev", "上述固定HF", "來源："))]
        dependency_boundaries.append({"lesson_id": lesson, "snapshot": page["snapshot"], "sha256": sha(data), "same_as_prior_dependency": old["page_bindings"][lesson]["source_sha256"] == sha(data), "boundary_excerpts": [p[:2000] for p in selected[:6]]})
    descriptor = load("docs/decision-product/v3/public-release.json")
    archive = Path("/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip")
    assert sha(archive.read_bytes()) == descriptor["sha256"]
    with zipfile.ZipFile(archive) as saved:
        seen_members = {}

        def member(name):
            data = saved.read(PREFIX + name)
            seen_members[name] = {"bytes": len(data), "sha256": sha(data)}
            return data

        def obj(name):
            return json.loads(member(name))

        qualified = obj("bundle/bundle.json")
        reviewed = obj("evidence/reviewed-candidate-bundle.json")
        index = obj("evidence/reviews/index.json")
        assert seen_members["evidence/reviews/index.json"]["sha256"] == qualified["review_index_sha256"]
        assert seen_members["evidence/reviewed-candidate-bundle.json"]["sha256"] == qualified["source_candidate_bundle_sha256"] == index["candidate_bundle_sha256"]
        assert qualified["config"] == reviewed["config"]
        assert qualified["files"] == reviewed["files"]
        assert qualified["final_opening_sha256"] == reviewed["final_opening_sha256"]
        expected_implementation = {Path(p).name: digest for p, digest in descriptor["implementation"].items()}
        assert index["implementation"] == expected_implementation
        assert sha(member("README.txt")) == index["readme_sha256"]
        reports = []
        actors = []
        for reference in index["reviews"]:
            name = "evidence/reviews/" + reference["path"]
            report = obj(name)
            assert seen_members[name]["sha256"] == reference["sha256"]
            assert report["reviewer_task"] == reference["actor"]
            assert report["implementation"] == expected_implementation
            assert report["readme_sha256"] == index["readme_sha256"]
            assert report["candidate_bundle_sha256"] == index["candidate_bundle_sha256"]
            assert datetime.fromisoformat(report["created_at"]) <= datetime.fromisoformat(index["created_at"])
            actors.append(reference["actor"])
            closures = report.get("original_issue_resolutions", report.get("actual_current_rechecks", report.get("original_necessary_rechecks", [])))
            reports.append({"kind": reference["kind"], "actor": reference["actor"], "member": name, "sha256": reference["sha256"], "created_at": report["created_at"], "all17_implementation_equal": True, "candidate_and_readme_equal": True, "original_closure_count": len(closures), "closure_ids": [row.get("issue_id", row.get("id", row.get("original_id"))) for row in closures]})
        assert len(set(actors)) == 3 and set(x["kind"] for x in reports) == {"readability", "correctness", "coherence"}
        attachments = obj("evidence/review-attachments-index.json")
        for item in attachments["files"]:
            data = member(item["delivered_path"])
            assert len(data) == item["bytes"] and sha(data) == item["sha256"]
        opening = obj("evidence/final-opening.json")
        final = obj("evidence/final-results.json")
        observations = obj("evidence/final-observations.json")
        families = obj("evidence/final-families.json")
        manual = obj("evidence/product-checks.json")
        legacy = obj("evidence/legacy-regression.json")
        reviewed_summary = obj("evidence/reviewed-qualification-summary.json")
        policy = obj("bundle/policy.json")
        assert reviewed_summary["final"] == final
        assert reviewed_summary["manual_product_metrics"] == manual["metrics"]
        assert reviewed_summary["training"]["weights_sha256"] == qualified["files"]["weights.safetensors"] == opening["weights_sha256"]
        assert opening["implementation"] == expected_implementation
        assert final["observations_sha256"] == seen_members["evidence/final-observations.json"]["sha256"]
        assert final["families_sha256"] == seen_members["evidence/final-families.json"]["sha256"]
        assert opening["product_checks_sha256"] == seen_members["evidence/product-checks.json"]["sha256"]
        assert opening["legacy_regression_sha256"] == seen_members["evidence/legacy-regression.json"]["sha256"]
        # 僅重算已保存logits的計數；不載模型，不生成新題，不重跑final。
        counts, correct, family_counts = Counter(), Counter(), Counter()
        accepted = errors = 0
        for row in observations:
            kind = row["kind"]
            counts[kind] += 1
            family_counts[row["family"]] += 1
            logits = [v if v is not None else -math.inf for v in row["logits"]]
            chosen = max(range(len(logits)), key=logits.__getitem__)
            correct[kind] += chosen == row["label"]
            if kind == "choice":
                scaled = [v / policy["temperatures"][kind] for v in logits]
                largest = max(scaled)
                confidence = 1 / sum(math.exp(v - largest) for v in scaled)
                adopt = row["ids"][chosen] != "none" and row["eligible"][chosen] and confidence >= policy["threshold"]
                accepted += adopt
                errors += adopt and chosen != row["label"]
        assert len(families) == len(family_counts) == 1500
        assert set(family_counts.values()) == {3}
        assert counts == correct == {"choice": 1500, "info": 1500, "rubric": 1500}
        assert accepted == 1025 and errors == 0
        for kind in counts:
            assert final["metrics"][kind]["correct"] == correct[kind]
            assert final["metrics"][kind]["families"] == counts[kind]
        c3 = {"archive_sha256": descriptor["sha256"], "index_created_at": index["created_at"], "collection_scope": "Original three reports are collected by path/full SHA in this original index; no invented per-report received_at.", "reports": reports, "attachment_members_checked": len(attachments["files"]), "reviewed_and_qualified_config_files_equal": True, "raw_observation_arithmetic": {"observations": len(observations), "families": len(family_counts), "observations_per_family": 3, "per_kind": dict(counts), "correct_per_kind": dict(correct), "choice_accepted": accepted, "choice_accepted_errors": errors, "choice_coverage_all": accepted / counts["choice"]}, "manual_retained": {key: manual[key] for key in ("metrics", "accepted_errors", "known_action_accepted", "known_action_observations", "known_action_coverage", "invariance_failures", "statistical_unit")}, "legacy_retained": {key: legacy[key] for key in ("metrics", "accepted", "accepted_errors", "scope")}, "members": seen_members}
    result = {"schema": "canonical-19.13-retention-audit-v1", "reviewer_task": "/root/c_v3_technical_coord/canonical_19_13", "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-19-13/retention-audit.py", "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "CPU JSON/hash arithmetic only; no model inference"}, "original": {"path": original_path, "sha256": sha(raw(original_path)), "intake_path": intake_path, "intake_sha256": sha(raw(intake_path)), "received_at": intake["received_at"]}, "prior": {"path": prior_path, "sha256": sha(raw(prior_path)), "reviewer_task": prior["reviewer_task"], "created_at": prior["created_at"], "source_sha256": prior["source_sha256"], "source_identity": code_identity, "artifacts": retained_artifacts, "verdict_flag_used_as_evidence": False}, "legacy_execution": {"original_owner": prior["reviewer_task"], "recorded_at": old["created_at"], "scope": old["scope"], "histories": histories, "historical_moe": old["historical_moe_test"], "historical_bc": old["historical_bc_test"], "new_model_execution_by_this_owner": False}, "retained_original_source_receipts": [{k: row[k] for k in ("url", "sha256", "accessed_at", "locator", "short_excerpt", "inspection_scope")} for row in original_sources["original_sources"]], "dependency_boundaries": dependency_boundaries, "c3_retention": c3, "limitations": ["Current owner checked identity and narrow semantic applicability after original actual intake; prior pass flags are not technical proof.", "Old NN17 engineering reviews and old model executions retain their original owners, dates and limits; no new acceptance run is claimed.", "Only saved-logit arithmetic and opaque member/hash comparisons ran. No model, training, final opening, network, publication or commit.", "Bounded typed3–6device service only; primary fresh evidence4–6, manual/old60regression are exposed/correlated; no arbitrary natural-language or fresh3device reliability claim."]}
    with OUT.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"path": OUT.relative_to(ROOT).as_posix(), "sha256": sha(OUT.read_bytes()), "old_repository_sources_equal": len(code_identity), "retained_artifacts_equal": len(retained_artifacts), "c3_review_actors": actors, "c3_attachments_checked": len(attachments["files"]), "raw_saved_logit_counts": dict(correct)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
