"""摘錄授權的原始執行收據並核對教材複查所需的指紋與先後。"""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ORIGINAL = "docs/course-review-20261019/reviews/rv1019_technical_validation/technical-validation-original-01.json"
READS = {}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(path):
    path = Path(path)
    content = path.read_bytes()
    READS[str(path.relative_to(ROOT) if path.is_absolute() else path)] = {"sha256": digest(content), "bytes": len(content)}
    return content


def read(path):
    return json.loads(raw(ROOT / path))


def selected(value, names):
    return {k: value[k] for k in names if k in value}


def command(argv):
    process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    return {"argv": argv, "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr}


def main():
    original = read(ORIGINAL)
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
              "environment": {"python": platform.python_version(), "device": "cpu", "platform": platform.platform()},
              "revision": command(["git", "rev-parse", "HEAD"])["stdout"].strip()}
    result["source_inventory"] = []
    for source in original["sources"]:
        if source["kind"] == "repository_code":
            current = digest(raw(ROOT / source["path"]))
            result["source_inventory"].append({"id": source["id"], "path": source["path"],
                                              "previous_sha256": source["sha256"], "current_sha256": current,
                                              "unchanged": current == source["sha256"]})
    result["previous_artifacts"] = []
    for artifact in original["artifacts"]:
        current = digest(raw(ROOT / artifact["path"]))
        result["previous_artifacts"].append({"path": artifact["path"], "sha256": current,
                                            "unchanged": current == artifact["sha256"]})
    result["page_diff"] = command(["git", "diff", "--no-index",
        "docs/course-review-20261019/freeze-04/sources/validation.md",
        "docs/course-review-20261019/freeze-05/sources/validation.md"])
    result["chapter_diff"] = command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/chapters/20.md"])
    result["guides_diff"] = command(["git", "diff", "b74f590c4dfa5e70dd2d60cb572746dc5a57503e", "HEAD", "--",
                                   "docs/editorial-guide.md", "docs/technical-review-guide.md"])
    for path in ["docs/validation.md", "docs/course-review-20261019/freeze-05/sources/validation.md"]:
        raw(ROOT / path)
    freeze = read("docs/course-review-20261019/freeze-06.json")
    result["frozen_validation_page"] = next(p for p in freeze["pages"] if p["page_id"] == "validation")
    suite_path = "outputs/course-review-20261019/executed-05/validation-kernel.json"
    result["authorized_full_book_summary"] = {"path": suite_path, "result": read(suite_path)}
    result["kernel_probe"] = read("docs/technical-reviews/artifacts/review-20261019-validation/kernel-results-current-01.json")["checks"]
    manifest = read("docs/selftrained/v2-manifest.json")
    expected_data = {v["path"]: v["sha256"] for v in manifest["records"]}
    stages = {}
    for directory in sorted((ROOT / "docs/selftrained/results/training-raw").iterdir()):
        if not directory.is_dir():
            continue
        base = str(directory.relative_to(ROOT))
        receipt = read(base + "/receipt.json")
        training = read(base + "/raw/train-receipt.json")
        execution = read(base + "/raw/execution.json")
        files = {item["path"]: item for item in receipt["files"]}
        raw_matches = {
            name: READS[base + "/raw/" + name]["sha256"] == files[name]["sha256"]
                  and READS[base + "/raw/" + name]["bytes"] == files[name]["bytes"]
            for name in ["train-receipt.json", "execution.json"]}
        job = execution["job"]
        origin = training["origin"]
        history = [{k: row[k] for k in ["stage", "step", "checkpoint_sha256", "selection"] if k in row}
                   for row in training.get("stage_history", [])]
        fields = selected(training, ["schema", "stage", "architecture", "steps", "tokens", "target_tokens",
            "train_records", "validation_records", "test_used_for_selection", "seed", "selection",
            "best_validation_loss", "total_parameters", "trainable_parameters", "interrupted",
            "completed_requested_steps", "selected_checkpoint_available", "inference_exported"])
        stages[directory.name] = {
            "receipt": selected(receipt, ["run_id", "stage", "revision", "manifest_sha256", "started_at", "finished_at",
                                          "status", "resource_spec", "returncode"]),
            "data_readiness": selected(receipt["data_readiness"], ["status", "manifest_sha256", "archive_sha256",
                                        "archive_bytes", "record_files_verified", "asset_files_verified"]),
            "training": fields, "origin": selected(origin, ["kind", "seed"]),
            "stage_history": history,
            "execution": selected(execution, ["run_id", "status", "returncode", "started_at", "finished_at", "command"]),
            "job": selected(job, ["architecture", "stage", "steps", "init_checkpoint", "resume", "seed"]),
            "selected_files": {name: files[name] for name in ["best.pt", "model.safetensors", "execution.json", "train-receipt.json"]},
            "checks": {"raw_hashes_and_sizes_match_receipt": all(raw_matches.values()),
                       "run_identity_matches": receipt["run_id"] == execution["run_id"],
                       "recorded_complete": receipt["status"] == execution["status"] == "completed"
                                            and execution["returncode"] == 0
                                            and training["completed_requested_steps"] is True
                                            and training["interrupted"] is False,
                       "actual_step_count_matches_job": training["steps"] == job["steps"],
                       "random_origin": origin["kind"] == "all-neural-weights-random",
                       "data_hashes_match_manifest": training["data_sha256"] == expected_data,
                       "test_not_used_for_selection": training["test_used_for_selection"] is False},
        }
    by_run = {record["receipt"]["run_id"]: record for record in stages.values()}
    for record in stages.values():
        parent = record["job"].get("init_checkpoint")
        if parent:
            previous = by_run[parent["run_id"]]
            record["parent"] = {"run_id": parent["run_id"], "sha256": parent["sha256"],
                                "parent_best_sha256": previous["selected_files"]["best.pt"]["sha256"],
                                "matches": parent["sha256"] == previous["selected_files"]["best.pt"]["sha256"]}
        else:
            record["parent"] = {"none": True, "pretrain_root": record["training"]["stage"] == "pretrain"}
    result["training_stages"] = stages
    selection = read("docs/selftrained/results/v2-final-training-source-selection.json")
    result["source_selection"] = {
        "header": selected(selection, ["schema", "decided_at", "decision_scope", "test_used_for_selection"]),
        "moe_joint": selected(selection["moe_joint"], ["source_training_run", "source_training_revision",
            "selected_step", "completed_requested_steps", "sha256", "safe_weights_sha256"]),
        "dense_joint": selection["dense_joint"],
    }
    final_public = read("docs/selftrained/results/v2-final-public-results.json")
    result["final_tests"] = {}
    for arch, stage in [("moe", "moe-native"), ("dense", "dense-weighted")]:
        path = f"docs/selftrained/results/v2-{arch}-public-final-test-stage-index.json"
        index = read(path)
        cp = index["checkpoint"]
        record = stages[stage]
        public = final_public["architectures"][arch]
        result["final_tests"][arch] = {
            "index_path": path,
            "run": index["run"], "split": index["split"], "expected_count": index["expected_count"],
            "completed_count": index["completed_count"],
            "checkpoint": selected(cp, ["architecture", "source_training_run", "source_training_revision", "selected_step",
                                        "completed_requested_steps", "sha256", "safe_weights_sha256"]),
            "per_task_counts": {name: value["count"] for name, value in index["per_task_final_reply"].items()},
            "generation_audit": selected(index["generation_audit"], ["main_call_count", "control_call_count",
               "total_logged_calls", "neural_calls", "status_counts", "stop_reason_counts", "all_call_eos",
               "main_reply_failure_rows", "perception_failure_rows", "exhausted_attempts", "capped_neural_generations"]),
            "lineage": selected(index["lineage"], ["manifest_sha256", "job_sha256", "conditions_sha256",
                                                 "outputs_sha256", "completed_ids_sha256"]),
            "checks": {
                "training_run_identity": cp["source_training_run"] == record["receipt"]["run_id"],
                "selected_best_identity": cp["sha256"] == record["selected_files"]["best.pt"]["sha256"] == public["selected_source"]["sha256"],
                "safe_weights_identity": cp["safe_weights_sha256"] == record["selected_files"]["model.safetensors"]["sha256"] == public["safe_weights"]["sha256"],
                "completed_steps_match_receipt": cp["completed_requested_steps"] == record["training"]["steps"],
                "counts_match": index["completed_count"] == index["expected_count"] == 3734
                                == sum(v["count"] for v in index["per_task_final_reply"].values()),
                "public_test_run": index["run"]["attempt"] == public["final_test"]["run_id"],
            },
        }
    evaluation_path = ROOT / "scripts/selftrained/evaluate.py"
    evaluation = raw(evaluation_path).decode()
    marker_start = evaluation.index("        test_marker = protocol_path.with_suffix")
    marker_end = evaluation.index("    elif args.split", marker_start)
    result["single_protocol_guard_excerpt"] = evaluation[marker_start:marker_end]
    joint_counts = {name: selected(stages[name]["training"], ["train_records", "validation_records"])
                    for name in ["moe-joint", "moe-native", "dense-joint", "dense-weighted"]}
    result["dataset_counts"] = {"joint_actual_trainer_counts": joint_counts,
        "final_count_each": {arch: record["completed_count"] for arch, record in result["final_tests"].items()},
        "all_joint_counts_match": all(v == {"train_records": 28876, "validation_records": 2435} for v in joint_counts.values()),
        "scope": "核對實際trainer讀取後記錄的列數及其manifest-bound資料SHA；本次未重新下載LFS資料或解包重數。"}
    final = read("docs/lesson-products/results/final-report-v1.json")
    settings = read("docs/lesson-products/results/settings-freeze-v1.json")
    package = read("docs/lesson-products/results/publication-package-v1.json")
    settings_hash = READS["docs/lesson-products/results/settings-freeze-v1.json"]["sha256"]
    identities = [{"product": product, "file": name, "freeze_sha256": expected,
        "package_sha256": package["files"][product + "/" + name]["sha256"],
        "matches": expected == package["files"][product + "/" + name]["sha256"]}
        for product, files in settings["products"].items() for name, expected in files.items()]
    result["scoring_publication_identity"] = {
        "settings_sha256": settings_hash, "final_recorded_settings_sha256": final["settings_freeze_sha256"],
        "package_settings_sha256": package["files"]["settings-freeze.json"]["sha256"],
        "package_report_sha256": package["files"]["report.json"]["sha256"],
        "final_report_sha256": READS["docs/lesson-products/results/final-report-v1.json"]["sha256"],
        "product_files": identities,
        "checks": {"same_scoring_freeze": final["settings_freeze_sha256"] == settings_hash,
                   "same_package_freeze": package["files"]["settings-freeze.json"]["sha256"] == settings_hash,
                   "same_package_report": package["files"]["report.json"]["sha256"] == READS["docs/lesson-products/results/final-report-v1.json"]["sha256"],
                   "all_product_files_same": all(v["matches"] for v in identities)}}
    audits = {}
    for label in ["b-mechanisms", "c"]:
        prefix = "docs/lesson-products/reviews/engineering-final-" + label
        original_review = read(prefix + "-original.json")
        execution = read(prefix + "-execution.json")
        if label == "b-mechanisms":
            review_excerpt = selected(original_review, ["reviewer_task", "reviewer_context", "verdict_scope", "verification",
                                                       "scope", "boundary", "auto", "limits"])
            execution_excerpt = selected(execution, ["reviewer_task", "reviewer_context", "command", "script_sha256",
                                                      "environment", "checked_documents", "checks", "summary", "boundary", "scope"])
        else:
            review_excerpt = selected(original_review, ["reviewer_task", "reviewer_context", "scope", "raw_reference",
                                                        "settings_freeze_sha256", "final_report_sha256", "limits", "execution_artifacts"])
            execution_excerpt = selected(execution, ["reviewer_task", "reviewer_context", "command", "probe_script_sha256",
                                                      "environment", "checks", "raw_reference", "scope"])
        audits[label] = {"original_path": prefix + "-original.json", "execution_path": prefix + "-execution.json",
                         "original_excerpt": review_excerpt, "execution_excerpt": execution_excerpt}
    result["two_saved_recalculation_records"] = audits
    result["recalculation_identity_checks"] = {
        "different_reviewers": audits["b-mechanisms"]["original_excerpt"]["reviewer_task"]
                              != audits["c"]["original_excerpt"]["reviewer_task"],
        "b_original_execution_identity": audits["b-mechanisms"]["original_excerpt"]["reviewer_task"]
                                       == audits["b-mechanisms"]["execution_excerpt"]["reviewer_task"],
        "c_original_execution_identity": audits["c"]["original_excerpt"]["reviewer_task"]
                                       == audits["c"]["execution_excerpt"]["reviewer_task"],
        "b_same_raw": audits["b-mechanisms"]["original_excerpt"]["verification"]["raw_sha256"] == final["predictions"]["sha256"],
        "c_same_raw": audits["c"]["original_excerpt"]["raw_reference"]["sha256"] == final["predictions"]["sha256"],
    }
    result["git_timeline"] = command(["git", "log", "--reverse", "--format=%H %cI %s", "--",
        "docs/lesson-products/results/settings-freeze-v1.json", "docs/lesson-products/results/final-report-v1.json",
        "docs/lesson-products/results/publication-package-v1.json",
        "docs/lesson-products/reviews/engineering-final-b-mechanisms-original.json",
        "docs/lesson-products/reviews/engineering-final-c-original.json"])
    result["runtime_observations"] = {
        "ipython_warning_excerpt": "IPython dir '/Users/yclin/.ipython' is not a writable location, using a temp directory.",
        "ipython_cause": "父程序在設定artifact內IPYTHONDIR之前匯入ipykernel，IPython以可寫性檢查改用系統暫存目錄；這造成非預期的寫入範圍偏離。沒有開啟該暫存目錄，也沒有修改既有檔。",
        "shutdown_error_excerpt": "PermissionError: [Errno 1] Operation not permitted (originated from sysctl() malloc 1/3)",
        "shutdown_scope": "psutil讀取程序清單時被拒，屬讀取權限錯誤，並非寫檔被拒；課程Notebook exit0、隔離與故障回報探針均通過。"}
    result["authorized_evidence_boundary"] = {
        "extra_outputs_read": [suite_path],
        "engineering_reviews_read": [p for p in READS if p.startswith("docs/lesson-products/reviews/")],
        "other_reviews_read": False,
        "scope": "這四份工程紀錄只用來核對頁面所稱的兩次保存分數重算；未沿用其pass裁決作本頁其他主張的依據。"}
    result["input_files"] = READS
    result["checks"] = {
        "old_artifacts_unchanged": all(v["unchanged"] for v in result["previous_artifacts"]),
        "all_training_stage_checks": all(all(v["checks"].values()) for v in stages.values()),
        "all_parent_links": all(v["parent"].get("matches", v["parent"].get("pretrain_root", False)) for v in stages.values()),
        "all_final_test_links": all(all(v["checks"].values()) for v in result["final_tests"].values()),
        "all_dataset_counts": result["dataset_counts"]["all_joint_counts_match"],
        "publication_identity": all(result["scoring_publication_identity"]["checks"].values()),
        "two_audit_record_identities": all(result["recalculation_identity_checks"].values()),
        "kernel_probes": all(result["kernel_probe"].values()),
    }
    output = HERE / "evidence-results-current-01.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"checks": result["checks"], "stages": len(stages), "source_drift":
        [v["path"] for v in result["source_inventory"] if not v["unchanged"]]}, ensure_ascii=False))
    if not all(result["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
