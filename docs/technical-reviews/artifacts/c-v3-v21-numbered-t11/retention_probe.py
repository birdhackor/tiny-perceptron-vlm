"""Read retained raw artifacts and recompute arithmetic; never invoke final evaluation."""

import copy
import hashlib
import io
import json
import math
import pathlib
import platform
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime

import torch

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def local(name):
    return json.loads((ROOT / name).read_bytes())


def identity(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}


def parse_facts(example):
    target = re.findall(r"(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])", example["question"])
    assert len(target) == 1
    result = {}
    for row in re.split(r"[。.]", example["state"]):
        if not row.strip():
            continue
        names = re.findall(r"(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])", row)
        task = re.findall(r"record|play|save", row)
        assert len(names) == len(task) == 1
        if example["source"] == "chinese":
            p = re.search(r"許可(是|否|未知)", row)[1]
            r = re.search(r"就緒(是|否|未知)", row)[1]
            mapping = {"是": 1, "否": 0, "未知": -1}
        else:
            p = re.search(r"permission(?:=| is | )(yes|no|unknown)", row)[1]
            r = re.search(r"read(?:y=|iness is |iness )(yes|no|unknown)", row)[1]
            mapping = {"yes": 1, "no": 0, "unknown": -1}
        result[names[0]] = (task[0], mapping[p], mapping[r])
    return target[0], result


def declared_gold(example, target, facts):
    task, p, r = facts[target]
    if example["kind"] == "choice":
        action = "none" if -1 in (p, r) else "stop" if p == 0 else "wait" if r == 0 else task
        correct = [key for key, desc in example["candidates"] if re.search(r"(?:Device |裝置)" + target + r"[:：]\s*" + action + r"$", desc)]
        return correct[0] if correct else "none"
    if example["kind"] == "info":
        required = (p, r) if task == "record" else (p,) if task == "play" else (r,)
        return "yes" if -1 not in required else "no"
    return "none" if -1 in (p, r) else str(p + r if task == "record" else (2 if p else r) if task == "play" else (2 if r else p))


def c_v3():
    path = pathlib.Path("/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip")
    descriptor = local("docs/decision-product/v3/public-release.json")
    acquisition = local("docs/decision-product/v3/metadata/acquisition.json")
    result = {"archive": identity(path)}
    assert result["archive"]["sha256"] == descriptor["sha256"]
    assert result["archive"]["bytes"] == descriptor["bytes"]
    prefix = "tiny-perceptron-c-v3/"
    with zipfile.ZipFile(path) as archive:
        # Deliberately never read evidence/reviews or evidence/review-attachments.
        result["metadata_acquisition_matches"] = all(sha(archive.read(row["archive_member"])) == row["sha256"] == sha((ROOT / row["path"]).read_bytes()) for row in acquisition["files"])
        result["implementation_matches"] = {name: sha((ROOT / name).read_bytes()) == digest == sha(archive.read(prefix + name)) for name, digest in descriptor["implementation"].items()}
        result["bundle"] = {name: {"bytes": len(archive.read(prefix + "bundle/" + name)), "sha256": sha(archive.read(prefix + "bundle/" + name)), "matches_repo_descriptor": sha(archive.read(prefix + "bundle/" + name)) == digest, "matches_installed": sha((ROOT / "checkpoints/decision-product-v3" / name).read_bytes()) == digest} for name, digest in descriptor["bundle"].items()}
        result["continuation_state_members"] = [name for name in archive.namelist() if name.endswith(".pt")]
        evidence = lambda name: json.loads(archive.read(prefix + "evidence/" + name + ".json"))
        final, observations, examples, families = [evidence(name) for name in ("final-results", "final-observations", "final-examples", "final-families")]
        result["raw_final_bindings"] = {name: {"sha256": sha(archive.read(prefix + "evidence/" + name + ".json")), "matches_result_receipt": sha(archive.read(prefix + "evidence/" + name + ".json")) == final[key]} for name, key in (("final-observations", "observations_sha256"), ("final-examples", "examples_sha256"), ("final-families", "families_sha256"), ("final-opening", "final_opening_sha256"))}
        selection, opening = evidence("selection"), evidence("final-opening")
        policy = json.loads(archive.read(prefix + "bundle/policy.json"))
        result["ordering"] = {"completion": evidence("completion")["created_at"], "selection": selection["created_at"], "opening": opening["created_at"], "result": final["created_at"], "weights_match_at_selection_opening": selection["weights_sha256"] == opening["weights_sha256"] == descriptor["bundle"]["weights.safetensors"], "selection_binding": opening["selection_sha256"] == sha(archive.read(prefix + "evidence/selection.json")), "policy": policy}
        seen = set()
        metrics = defaultdict(lambda: {"n": 0, "correct": 0})
        selective = {"families": 0, "accepted": 0, "errors": 0}
        family_map = {row["key"]: row for row in families}
        assert len(family_map) == len(families) == 1500
        conflicts = Counter()
        for observation, example in zip(observations, examples, strict=True):
            key = observation["family"], observation["kind"]
            assert key not in seen
            seen.add(key)
            assert key == (example["family"], example["kind"])
            target, facts = parse_facts(example)
            expected = declared_gold(example, target, facts)
            ids = [pair[0] for pair in example["candidates"]] + ["none"]
            assert ids == observation["ids"]
            assert expected == example["gold"] == observation["gold_id"] == ids[observation["label"]]
            fact_rows = [(('record', 'play', 'save').index(task), p, r) for task, p, r in facts.values()]
            family_key = sha(json.dumps(sorted(fact_rows), separators=(",", ":")).encode())
            assert family_key == example["family"]
            assert sorted(fact_rows) == sorted(tuple(row) for row in family_map[family_key]["facts"])
            # The retained JSON encodes the info head's masked none logit as null.
            logits = [float("-inf") if value is None else value for value in observation["logits"]]
            selected = max(range(len(logits)), key=lambda i: logits[i])
            correct = selected == observation["label"]
            for group in (example["kind"], example["kind"] + "/source/" + example["source"], example["kind"] + "/panel/" + example["panel"]):
                metrics[group]["n"] += 1
                metrics[group]["correct"] += correct
            if example["kind"] == "choice":
                expected_eligible = [re.findall(r"(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])", desc) == [target] for _, desc in example["candidates"]] + [False]
                assert expected_eligible == observation["eligible"]
                values = [math.exp((x - max(logits)) / policy["temperatures"]["choice"]) for x in logits]
                confidence = values[selected] / sum(values)
                accepted = ids[selected] != "none" and expected_eligible[selected] and confidence >= policy["threshold"]
                selective["families"] += 1
                selective["accepted"] += accepted
                selective["errors"] += accepted and not correct
        result["recomputed_metrics"] = dict(metrics)
        result["recomputed_selective"] = {**selective, "coverage": selective["accepted"] / selective["families"], "risk": selective["errors"] / selective["accepted"]}
        result["observations"] = len(observations)
        result["independent_final_families"] = len(families)
        result["device_counts_in_final"] = dict(Counter(len(row["facts"]) for row in families))
        result["statistical_unit"] = "1500 independent visible-fact families per kind; three correlated questions per family, never4500 independent families"
        result["gold_eligibility_and_visible_fact_hash_checks"] = "all4500 saved examples agree with independently parsed visible facts and declared fixed rules; all1500 family hashes agree"
        final_keys = set(family_map)
        excluded = {key for entry in evidence("registry-before-final")["entries"] for key in entry["family_keys"]}
        result["previous_exposed_worlds"] = len(excluded)
        result["final_previous_exposure_overlap"] = len(final_keys & excluded)
        result["calibration_policy_counts"] = {name: len(keys) for name, keys in selection["families"].items()}
        result["final_calibration_policy_overlap"] = {name: len(final_keys & set(keys)) for name, keys in selection["families"].items()}
        for name, expected_field in (("product-checks", "expected"), ("legacy-regression", "gold")):
            original = evidence(name)
            counts = defaultdict(lambda: {"n": 0, "correct": 0, "accepted_errors": 0})
            for row in original["observations"]:
                prediction, expected = row["prediction"], row[expected_field]
                correct = prediction["model_choice"] == expected
                counts[row["kind"]]["n"] += 1
                counts[row["kind"]]["correct"] += correct
                counts[row["kind"]]["accepted_errors"] += prediction["status"] == "accepted" and not correct
            result[name] = {"member_sha256": sha(archive.read(prefix + "evidence/" + name + ".json")), "recomputed": dict(counts), "device_counts": sorted({row["devices"] for row in original["observations"]}) if name == "product-checks" else "old60families;180 correlated questions; engineering regression, neverfreshfinal"}
    return result


def omni_retention():
    selection = local("docs/omni-products/v1/model-selection.json")
    recipe = local("docs/omni-products/v1/formal-recipes.json")
    path = pathlib.Path("/workspace/work/omni-core-completed-40988311100/core-checkpoints.zip")
    result = {"archive": identity(path), "jobs": {}}
    with zipfile.ZipFile(path) as archive:
        for job, binding in selection["all_formal_jobs"].items():
            prefix = f"core/{job}/"
            training_raw = archive.read(prefix + "training/training-result.json")
            training = json.loads(training_raw)
            job_raw = archive.read(prefix + "job-result.json")
            run = json.loads(job_raw)
            assert sha(job_raw) == binding["job_result_sha256"]
            assert sha(training_raw) == run["training_result_sha256"]
            assert run["success"] and training["complete"] and training["optimizer_updates_observed"]
            assert training["updates"] == recipe["jobs"][job]["training"]["updates"]
            assert training["initial_parameter_sha256"] != training["latest_parameter_sha256"]
            state = {}
            for name, file_binding in binding["files"].items():
                raw = archive.read(prefix + "training/" + name)
                assert len(raw) == file_binding["bytes"] and sha(raw) == file_binding["sha256"]
                if name.endswith(".pt"):
                    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                    state[name] = {"step": value["step"], "keys": list(value), "optimizer_state_entries": len(value["optimizer"]["state"]), "rng_types": {name: type(value[name]).__name__ for name in ("python_rng", "torch_rng", "sampler_rng")}}
            result["jobs"][job] = {"kind": run["kind"], "started_at": run["started_at"], "finished_at": run["finished_at"], "runtime": run["runtime"], "execution_host": run["execution_host"], "updates": training["updates"], "best_update": training["best"]["update"], "config": training["config"], "source_binding_matches": all(sha((ROOT / "tiny_perceptron/omni_products" / name).read_bytes()) == digest for name, digest in training["implementation_sha256"].items()), "continuation": state, "identity_checks": "job-result SHA matches selection; training-result SHA matches job; all4 training files match selection bytes/SHA"}
    result["selected_four_weights"] = {name: selection["model_files"][name]["weights"] for name in ("av", "choice", "codec", "talker")}
    result["selected_four_bytes"] = sum(row["bytes"] for row in result["selected_four_weights"].values())
    result["selected_four_mib"] = result["selected_four_bytes"] / 2**20
    result["selection_time"] = selection["created_at"]
    result["selected_temperature"] = selection["temperature"]
    result["selected_threshold"] = selection["threshold"]
    path = ROOT / "outputs/omni-P5-public-raw-results-v1/final-results.zip"
    result["fixed_final_archive"] = identity(path)
    assert result["fixed_final_archive"]["sha256"] == "50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3"
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("measurements/choice/results.json")
        assert sha(raw) == "f0c3ec5fe8a504331c768df43c064d6f59a82a0469b4f2ea4d3a5ac9ab4cfdfd"
        original = json.loads(raw)["primary"]
        rows = original["policy"]["rows"]
        accepted, errors = sum(row["accepted"] for row in rows), sum(row["accepted"] and row["error"] for row in rows)
        result["c_v2_fixed_final"] = {"types": {kind: row["primary"] for kind, row in original["types"].items()}, "policy_recomputed": {"families": len(rows), "accepted": accepted, "errors": errors, "risk": errors / accepted}, "predeclared_thresholds": recipe["metrics_sha256"], "required": local("docs/omni-products/v1/metrics.json")["C"], "opening_member_sha256": sha(archive.read("frozen/final-opening.json")), "frozen_selection_matches": sha(archive.read("frozen/model-selection.json")) == sha((ROOT / "docs/omni-products/v1/model-selection.json").read_bytes())}
    return result


def history():
    result = {}
    for experiment in ("rag", "tools", "reasoning"):
        name = f"docs/course-experiments/results/{experiment}.json"
        value = local(name)
        result[experiment] = {"identity": identity(ROOT / name), "revision": value["revision"], "seed": value["seed"], "device": value["device"], "results_keys": list(value["results"]), "hf_revision": value["hf"]["revision"], "hf_result_revision": value["hf"].get("result_revision"), "preserved_original": "Read original saved input/output/denominator records; no model run, no re-training, no historical hash updated."}
    data = local("docs/course-experiments/results/tools.json")["results"]
    steps, mismatch, id_count = 0, 0, 0
    for key in ("samples", "one_step_cap_samples"):
        for episode in data[key]:
            first = next(event["generation"] for event in episode["trace"] if "generation" in event)
            current = copy.deepcopy(first["messages"][:2])
            for event in episode["trace"]:
                if "generation" not in event:
                    continue
                generation = event["generation"]
                ids = [1]
                for message in current:
                    ids += [{"system": 7, "user": 3, "assistant": 4}[message["role"]], *[byte + 8 for byte in message["content"].encode()], 2]
                ids.append(4)
                assert ids == generation["input_ids"] and len(ids) == generation["input_tokens"]
                steps += 1
                mismatch += current != generation["messages"]
                id_count += len(ids)
                if event.get("executed") and event.get("finite_result"):
                    current += [{"role": "assistant", "content": generation["samples"][0]["generated"]}, {"role": "user", "content": f"TOOL_RESULT:{event['tool_result']}"}]
    result["tools_input_snapshot_reconstruction"] = {"episodes": {"normal": len(data["samples"]), "one_step_cap": len(data["one_step_cap_samples"])}, "generations": steps, "messages_mismatched": mismatch, "input_ids_verified": id_count, "scope": "Independent reconstruction of historical byte-token inputs; original data unchanged; no model invocation."}
    manifest = local("docs/selftrained/v2-manifest.json")
    result["selftrained_v2_identity"] = {"manifest": identity(ROOT / "docs/selftrained/v2-manifest.json"), "initialization": manifest["initialization"], "config": manifest["model_config"], "package": manifest["package"], "record_files": len(manifest["records"])}
    result["lesson_product_retention"] = {name: {"identity": identity(ROOT / name), "product": local(name)["product"], "files": local(name)["files"]} for name in ("docs/lesson-products/results/b-artifact-manifest.json", "docs/lesson-products/results/c-artifact-manifest.json")}
    name = "docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json"
    original = local(name)
    result["selftrained_original_training_receipt"] = {"identity": identity(ROOT / name), **{key: original[key] for key in ("schema", "stage", "architecture", "steps", "tokens", "target_tokens", "train_records", "validation_records", "test_used_for_selection", "seed", "selection", "origin", "completed_requested_steps", "interrupted", "inference_exported")}}
    assert original["steps"] == 4000 and original["completed_requested_steps"] and not original["interrupted"]
    result["selftrained_original_final"] = {}
    for basename in ("receipt.json", "evaluation-receipt.json", "metrics.json"):
        name = "docs/selftrained/results/public-raw/moe/test/" + basename
        value = local(name)
        result["selftrained_original_final"][basename] = {"identity": identity(ROOT / name), "top_level_keys": list(value)}
    exported = local("docs/selftrained/v2-jobs/moe-public-final-test.json")["public_export"]
    result["selftrained_pinned_public_export"] = exported
    result["old_b_c_training"] = {}
    for product in ("b", "c"):
        name = f"docs/lesson-products/results/initial-{product}-execution.json"
        value = local(name)
        result["old_b_c_training"][product] = {"identity": identity(ROOT / name), "run_id": value["run_id"], "code_revision": value["code_revision"], "training": value["training"], "job": value["job"], "source_identity_matches": {name: sha((ROOT / name).read_bytes()) == digest for name, digest in value["source_hashes"].items() if name.startswith("tiny_perceptron/lesson_products/")}}
    value = local("docs/lesson-products/results/final-report-v1.json")
    result["old_b_c_fixed_final"] = {"identity": identity(ROOT / "docs/lesson-products/results/final-report-v1.json"), "recorded_at": value["recorded_at"], "data_manifest_sha256": value["data_manifest_sha256"], "settings_freeze_sha256": value["settings_freeze_sha256"], "result_keys": list(value), "prediction_record_type": type(value["predictions"]).__name__, "scope": value["scope"]}
    natural = (ROOT / "tiny_perceptron/natural_assistant.py").read_text()
    result["mature_model_retained_source"] = {"identity": identity(ROOT / "tiny_perceptron/natural_assistant.py"), "pinned_constants": [line for line in natural.splitlines() if line.startswith(("MODEL_ID =", "MODEL_REVISION =", "ASR_ID =", "ASR_REVISION ="))], "pretrained_loader_lines": [number for number, line in enumerate(natural.splitlines(), 1) if ".from_pretrained(" in line]}
    return result


def main():
    report = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/c_v3_technical_coord/numbered_t11", "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"}, "scope": "Fresh execution of a read-only identity and arithmetic audit over retained original files. No old review reports or attachments read; no original data modified; no final evaluation or training invoked.", "c_v3": c_v3(), "omni": omni_retention(), "history": history()}
    (OUT / "retention-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"c_v3_metrics": report["c_v3"]["recomputed_selective"], "c_v3_previous_overlap": report["c_v3"]["final_previous_exposure_overlap"], "omni_jobs": len(report["omni"]["jobs"]), "omni_four_weight_bytes": report["omni"]["selected_four_bytes"], "c_v2_policy": report["omni"]["c_v2_fixed_final"]["policy_recomputed"], "tool_trace": report["history"]["tools_input_snapshot_reconstruction"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
