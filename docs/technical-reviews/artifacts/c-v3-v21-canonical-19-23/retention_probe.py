"""Read immutable 2026-10-30 receipts; recalculate saved numbers, never run a model."""

import ast
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "docs/technical-reviews/artifacts/omni-P5-19.23"
ORIGINAL = "docs/course-revision-20261101-c-v3/technical-v21/canonical_19_23/original-v1.json"
INTAKE = "docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/canonical-19-23-original-v1.json"
PRIOR = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-19.23/technical-current-v2.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer(row):
    return row["ids"][min(range(len(row["ids"])), key=lambda i: (-row["logits"][i], row["ids"][i]))]


def nll(row, temperature):
    scores = [x / temperature for x in row["logits"]]
    peak = max(scores)
    return peak + math.log(sum(math.exp(x - peak) for x in scores)) - scores[row["ids"].index(row["gold"])]


def pmax(row, temperature):
    scores = [x / temperature for x in row["logits"]]
    peak = max(scores)
    return 1 / sum(math.exp(x - peak) for x in scores)


def policy(rows, temperature, threshold):
    accepted = errors = 0
    for row in rows:
        selected = answer(row["tasks"]["choice"])
        take = pmax(row["tasks"]["choice"], temperature) >= threshold and selected != "none" and answer(row["choice_info"]) == "yes"
        accepted += take
        errors += take and (selected != row["tasks"]["choice"]["gold"] or not row["world_supported"] or not row["choice_information_sufficient"])
    return {"families": len(rows), "accepted": accepted, "errors": errors, "coverage": accepted / len(rows), "risk": errors / accepted if accepted else None}


def main():
    started = datetime.now(UTC).isoformat()
    assert sha(ROOT / ORIGINAL) == "8dc913a7ccc2f7584ee89da6ea82346d846f0ed2949eeee5c242c730d1362112"
    assert sha(ROOT / INTAKE) == "13436546874af90a7db2a977a9dcc6e82a8695319b590b1c0c6866c88789c2c8"
    assert sha(ROOT / PRIOR) == "1a557f400690277fd63985b44da9facb42543b893cea5f748f09b817ae3f9932"
    intake = json.loads((ROOT / INTAKE).read_text())
    assert intake["report"]["sha256"] == sha(ROOT / ORIGINAL)
    assert started > intake["received_at"]
    expected = {"fixed-inputs.json": "2e5bc645940e424c1910e71cdb3734f222859f5916c008e83e45fd5fedf63be5", "primary-provenance.json": "f00e7a1a44d4323df850a54993ba2141aa7d226e2b926a850d1846a1b5888aba", "recompute-output.json": "2d01d5a4e27ef2a5b13cf1629891ba0d77073b01d0867edca872631cb18e03d0"}
    for name, digest in expected.items():
        assert sha(OLD / name) == digest
    raw = json.loads((OLD / "fixed-inputs.json").read_text())
    provenance = json.loads((OLD / "primary-provenance.json").read_text())
    old_recompute = json.loads((OLD / "recompute-output-current-v1.json").read_text())
    prior = json.loads((ROOT / PRIOR).read_text())
    identities, drifts = [], []
    for group in ("sources", "artifacts"):
        for item in prior[group]:
            if "path" not in item:
                continue
            actual = sha(ROOT / item["path"])
            rec = {"group": group, "id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"], "actual_sha256": actual}
            (identities if actual == item["sha256"] else drifts).append(rec)
    assert len(drifts) == 1 and drifts[0]["path"] == "docs/omni-products/README.md"
    guide = (ROOT / drifts[0]["path"]).read_text()
    assert "C-v2的最終正確率與接受後風險未達標" in guide and "合格C-v3操作" in guide and "raw Question介面不互換" in guide
    original_sources = []
    for filename, ranges in (("model.py", ((205, 224), (364, 369), (404, 426), (447, 461))), ("metrics.py", ((233, 243), (269, 295)))):
        receipt = next(x for x in provenance["third_party_originals_actually_read"] if x["url"].endswith("/kev/" + filename))
        saved = Path(receipt["full_text_outside_git"])
        assert sha(saved) == receipt["sha256"]
        lines = saved.read_text().splitlines()
        excerpts = []
        for left, right in ranges:
            text = "\n".join(lines[left - 1:right])
            excerpts.append({"lines": f"{left}-{right}", "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
        original_sources.append({"url": receipt["url"], "version": "2074cc44256200bb37540103fa4572901faa10f3", "original_acquired_on": receipt["accessed_on"], "current_inspected_on": datetime.now(UTC).date().isoformat(), "full_original_sha256": receipt["sha256"], "full_original_bytes": receipt["bytes"], "small_supporting_excerpts": excerpts, "new_network_retrieval": False})

    stats = {}
    for group in ("primary", "compact"):
        rows = raw[group]
        assert len(rows) == len({r["family"] for r in rows}) == 60
        stats[group] = {kind: {"correct": sum(answer(r["tasks"][kind]) == r["tasks"][kind]["gold"] for r in rows), "families": len(rows)} for kind in ("choice", "info", "rubric")}
        assert all(answer(record) == record["answer"] for r in rows for record in r["tasks"].values())
    assert [v["correct"] for v in stats["primary"].values()] == [39, 55, 35]
    assert [v["correct"] for v in stats["compact"].values()] == [28, 42, 21]
    assert {r["family"] for r in raw["primary"]} == {r["family"] for r in raw["compact"]}
    labels = {r["family"]: r for r in raw["original_final_labels"]}
    assert len(labels) == 60
    assert all(r["tasks"][k]["gold"] == labels[r["family"]]["golds"][k] for r in raw["primary"] for k in ("choice", "info", "rubric"))
    stats["panels"] = {name: {"correct": sum(answer(r["tasks"]["choice"]) == r["tasks"]["choice"]["gold"] for r in raw["primary"] if r["panel"] == name), "families": sum(r["panel"] == name for r in raw["primary"])} for name in ("complete", "removed", "missing")}
    stats["nll"] = {"raw": sum(nll(r["tasks"]["choice"], 1) for r in raw["primary"]) / 60, "fixed_temperature": sum(nll(r["tasks"]["choice"], raw["temperature"]) for r in raw["primary"]) / 60, "families": 60}
    assert abs(stats["nll"]["raw"] - 2.690098901766063) < 1e-12 and abs(stats["nll"]["fixed_temperature"] - 3.962385753167719) < 1e-12
    stats["final_acceptance"] = policy(raw["primary"], raw["temperature"], raw["threshold"])
    assert stats["final_acceptance"]["accepted"] == 53 and stats["final_acceptance"]["errors"] == 19
    stats["candidate_diagnostics"] = {}
    for count in range(2, 9):
        rows = [r for r in raw["diagnostic"] if r["total_choices_including_none"] == count]
        assert len(rows) == len({r["family"] for r in rows}) == 60
        assert all(len(r["ids"]) == count and answer(r) == r["answer"] for r in rows)
        stats["candidate_diagnostics"][str(count)] = {"correct": sum(answer(r) == r["gold"] for r in rows), "observations": len(rows)}
    assert [v["correct"] for v in stats["candidate_diagnostics"].values()] == [37, 22, 29, 37, 37, 32, 26]
    calibration = raw["calibration"]
    assert len(calibration) == len({r["family"] for r in calibration}) == 80
    assert len(raw["policy"]) == len({r["family"] for r in raw["policy"]}) == 60
    assert not ({r["family"] for r in calibration} & {r["family"] for r in raw["policy"]})
    grid = [{"temperature": t, "nll": sum(sum(nll(r["tasks"][k], t) for k in ("choice", "info", "rubric")) / 3 for r in calibration) / 80} for t in raw["original_C_metrics"]["temperature_grid"]]
    selected = min(grid, key=lambda r: (r["nll"], abs(r["temperature"] - 1), r["temperature"]))
    assert selected["temperature"] == raw["temperature"]
    policy_grid = [{"threshold": t, **policy(raw["policy"], raw["temperature"], t)} for t in raw["original_C_metrics"]["threshold_grid"]]
    feasible = [r for r in policy_grid if r["accepted"] and r["risk"] <= 0.05]
    best = max(feasible, key=lambda r: (r["coverage"], r["threshold"]))
    assert best["threshold"] == raw["threshold"] == 0.6
    stats["retained_selection_recalculation"] = {"temperature_grid": grid, "policy_grid": policy_grid, "selected_temperature": selected, "selected_policy": best, "no_final_parameter_selection": True}
    times = provenance["pre_final_git_evidence"]
    commit = times["last_selection_commit_hash_author_time_committer_time"].splitlines()
    assert times["model_selection_at_revision_bytes_equal_original_archive"]
    assert datetime.fromisoformat(raw["recorded_selection"]["created_at"]) < datetime.fromisoformat(commit[2]) < datetime.fromisoformat(raw["recorded_opening_created_at"]) < datetime.fromisoformat(raw["choice_part_origin"]["created_at"]) < datetime.fromisoformat(raw["choice_part_completion"]["created_at"])
    stats["recorded_pre_final_commitment"] = times
    training = {}
    for mode, bundle in raw["training"].items():
        result, log = bundle["result"], bundle["log"]
        best_log = max(log, key=lambda r: (r["dev_score"], -r["update"]))
        assert result["complete"] and result["updates"] == log[-1]["update"] == 6000 and len(log) == 12
        assert best_log["update"] == result["best"]["update"] == 5000
        for name, digest in result["implementation_sha256"].items():
            assert sha(ROOT / "tiny_perceptron/omni_products" / name) == digest
        training[mode] = {"updates": 6000, "dev_checkpoints": 12, "selected_update": 5000, "parameters": result["trainable_parameters"], "initial_sha256": result["initial_parameter_sha256"], "seed": result["config"]["seed"], "core_plus_route_questions": result["config"]["train_questions"], "source_identity_matches": True}
    assert training["augmented"]["initial_sha256"] == training["compact"]["initial_sha256"]
    stats["retained_training_log_inspection"] = training
    data = old_recompute["statistics"]["data_inspection"]
    assert data["family_counts"] == raw["original_frozen_C_family_counts"] == {"train": 903, "dev": 64, "calibration": 80, "policy": 60, "final": 60, "auto": 30}
    assert data["core_train_questions_augmented"] == 903 * 3 * 4 == 10836 and data["compact_variant0"] == 903 * 4 == 3612
    assert not any(data["cross_split_visible_fact_intersections"].values())
    stats["retained_original_data_generation_inspection"] = {**data, "execution_owner": provenance["reviewer"], "execution_date": "2026-10-30", "new_data_or_final_generated": False}
    invariant_rows = raw["development_invariants"]["rows"]
    maximum = max(v for r in invariant_rows for k, v in r.items() if k.endswith("_max_abs"))
    assert len(invariant_rows) == 8 and maximum <= 1e-5 and all(r["permutation_answer_same"] for r in invariant_rows)
    stats["retained_invariant_rows"] = {"development_families": 8, "largest_saved_max_abs": maximum, "atol": 1e-5, "trained_model_reexecuted": False}

    # Execute only the original batch-selection closure, with string markers and identity collate.
    tree = ast.parse((ROOT / "tiny_perceptron/omni_products/training_runtime.py").read_text())
    train_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_choice")
    batch_node = next(n for n in train_node.body if isinstance(n, ast.FunctionDef) and n.name == "batch")
    ns = {"torch": torch, "config": {"batch_size": 32}, "core": ["core" + str(i) for i in range(12)], "route_groups": [[name] for name in ("A", "B", "C", "none")], "device": "cpu", "collate_questions": lambda selected, device: selected}
    exec(compile(ast.Module(body=[batch_node], type_ignores=[]), "original-batch-closure", "exec"), ns)
    batch = ns["batch"](torch.Generator().manual_seed(29))
    assert len(batch) == 32 and all(x.startswith("core") for x in batch[:16]) and all(x in {"A", "B", "C", "none"} for x in batch[16:])
    assert "torch.randint(4" in ast.unparse(batch_node)
    from tiny_perceptron.omni_products.routing import Request
    router = Request("request-only", True, False, True).router_state()
    assert router == "request: request-only\nimage=1 audio=0 video=1"
    old_entry = (ROOT / "tiny_perceptron/lesson_products/__main__.py").read_text()
    assert 'default=Path("outputs/lesson-products/c")' in old_entry
    stats["batch_and_routing_source_probe"] = {"batch_size": len(batch), "core_count": 16, "route_count": 16, "sampled_route_counts": dict(Counter(batch[16:])), "equal_weight_mechanism": "actual closure calls torch.randint(4) for each route; not equal realized counts in every batch", "router_state": router, "legacy_C_default": "outputs/lesson-products/c", "training_or_NN_forward": False}
    original_runs = []
    for artifact in prior["artifacts"]:
        if artifact["kind"] == "execution":
            original_runs.append({k: artifact[k] for k in ("path", "sha256", "command", "result", "environment")})
    out = {"schema": "c-v3-v21-canonical-19-23-retention-audit-v1", "reviewer_task": "/root/c_v3_technical_coord/canonical_19_23", "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-19-23/retention_probe.py", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "CPU", "mode": "identity, saved-row arithmetic, no neural forward"}, "original_intake_sha256": sha(ROOT / INTAKE), "original_sha256_unchanged": sha(ROOT / ORIGINAL), "immutable_primary_receipts": expected, "unchanged_prior_references": identities, "changed_prior_references": drifts, "current_guide_scope_inspected": True, "Kev_original_source_inspection": original_sources, "original_2026_10_30_execution_owner": provenance["reviewer"], "original_2026_10_30_execution_receipts": original_runs, "statistics": stats, "limits": ["All model training, final inference and original archive acquisition are retained 2026-10-30 evidence, not this actor's new runs.", "This actor only checked exact identities, necessary local authoritative source slices, saved logits/logs and the isolated sampler/router code.", "No new final opening/generation, trained model forward, optimizer update, network retrieval, package installation, publication or product action.", "Prior report is a source index and retention lineage only; its verdict was not used as evidence."]}
    target = HERE / "retention-audit.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": sha(target), "unchanged_references": len(identities), "drift": drifts[0]["path"], "primary": stats["primary"], "acceptance": stats["final_acceptance"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
