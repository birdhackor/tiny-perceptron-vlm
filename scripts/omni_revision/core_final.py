"""固定候選／T／tau之後才可單次開final；能力不足不重挑權重。"""

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path

import torch

from scripts.omni_revision import core_evaluation as measured
from scripts.omni_revision.core_selection import formal_result, immutable_json, predict_families
from scripts.omni_revision.formal_local import FROZEN, ROOT, utc
from tiny_perceptron.omni_products.data import Event
from tiny_perceptron.omni_products.inference import CJudge, OmniAssistant, load_trained
from tiny_perceptron.omni_products.media_bindings import OLD_A, OLD_B
from tiny_perceptron.omni_products.reliability import TEMPERATURES, THRESHOLDS
from tiny_perceptron.omni_products.training_runtime import file_sha, json_file, source_binding

AUDIT = ROOT / "docs/course-revision-20261029-omni/infrastructure/final-evaluation-audit-v1.json"


def evaluation_sources():
    files = [
        path
        for directory in (
            "tiny_perceptron",
            "tiny_perceptron/omni_products",
            "tiny_perceptron/selftrained",
            "tiny_perceptron/lesson_products",
        )
        for path in (ROOT / directory).glob("*.py")
    ]
    files.extend(
        ROOT / name
        for name in (
            "scripts/omni_revision/core_final.py",
            "scripts/omni_revision/core_evaluation.py",
            "scripts/omni_revision/core_selection.py",
            "scripts/omni_revision/core_report.py",
            "scripts/omni_revision/formal_local.py",
            "scripts/omni_revision/__init__.py",
        )
    )
    return {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(files)}


def legacy_preflight(data_root, a_root, b_root):
    from tiny_perceptron.lesson_products.inference import Product, rule_depth
    from tiny_perceptron.selftrained.inference import InferenceAssistant

    data = json.loads((FROZEN / "data-definition.json").read_bytes())
    legacy_path = ROOT / "docs/selftrained/v2-manifest.json"
    if file_sha(legacy_path) != data["files_sha256"]["docs/selftrained/v2-manifest.json"]:
        raise ValueError("舊資料manifest不同")
    legacy = json.loads(legacy_path.read_bytes())
    total = 0
    for row in [*legacy["records"], *legacy["assets"]]:
        path = data_root / row["path"]
        if path.is_symlink() or path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            raise ValueError("保留能力的實際資料與固定manifest不同")
        total += path.stat().st_size
    if file_sha(b_root / "artifact-manifest.json") != OLD_B["manifest_sha256"]:
        raise ValueError("舊B manifest不同")
    old_a = InferenceAssistant(a_root, data_root, device="cpu", manifest_sha256=OLD_A["manifest_sha256"])
    old_b = Product(b_root, "b", device="cpu", verify=True)
    if rule_depth(old_b.auto_policy, "sum") != 1:
        raise ValueError("不可改原B深度policy")
    return (
        old_a,
        old_b,
        {
            "created_at": utc(),
            "stage": "legacy-assets-and-loader-preflight",
            "no_model_predictions": True,
            "old_A": old_a.receipt,
            "old_B_manifest_sha256": file_sha(b_root / "artifact-manifest.json"),
            "data_files": len(legacy["records"]) + len(legacy["assets"]),
            "data_bytes": total,
            "old_B_depth": 1,
        },
    )


def checked_selection(core, expected):
    path = FROZEN / "model-selection.json"
    if file_sha(path) != expected:
        raise ValueError("需明示固定model selection的真SHA")
    selected = json.loads(path.read_bytes())
    recipes = json.loads((FROZEN / "formal-recipes.json").read_bytes())
    required = {
        "schema",
        "final_opened",
        "implementation_sha256",
        "selected",
        "all_formal_jobs",
        "model_files",
        "temperature",
        "policy_feasible",
        "threshold",
        "sound_map",
        "selection_evidence",
        "formal_recipe_sha256",
        "data_definition_sha256",
        "metrics_sha256",
        "codec_selection_sha256",
    }
    if not required <= selected.keys() or selected["schema"] != "omni-core-model-selection-v1":
        raise ValueError("候選選擇缺少完整schema／權重／T／tau／map／evidence，不能開封")
    kinds = {"av", "choice", "choice_control", "codec", "talker"}
    if (
        set(selected["selected"]) != kinds
        or set(selected["model_files"]) != kinds
        or set(selected["all_formal_jobs"]) != set(recipes["jobs"])
    ):
        raise ValueError("十個完整正式工作與五個選定模組不可省略")
    a_names = [name for name in recipes["job_order"] if recipes["jobs"][name]["kind"] == "av"]
    names = selected["selected"]
    if (
        names["av"] not in a_names
        or names["choice"] != "c-augmented"
        or names["choice_control"] != "c-compact"
        or names["codec"] != "codec"
        or names["talker"] != "talker"
    ):
        raise ValueError("候選種類與事先規定的對照不同")
    if (
        type(selected["temperature"]) not in (int, float)
        or not math.isfinite(selected["temperature"])
        or not any(math.isclose(selected["temperature"], value, rel_tol=1e-12) for value in TEMPERATURES)
        or type(selected["policy_feasible"]) is not bool
        or (selected["policy_feasible"] and selected["threshold"] not in THRESHOLDS)
        or (not selected["policy_feasible"] and selected["threshold"] is not None)
    ):
        raise ValueError("T／tau須先由固定網格選定；無可行策略必須全轉交")
    from tiny_perceptron.omni_products.inference import SOUND_MAP

    if selected["sound_map"] != {key: [event.__dict__ for event in program] for key, program in SOUND_MAP.items()}:
        raise ValueError("semantic音型字典與選定凍結不同")
    if selected["final_opened"] is not False or selected["implementation_sha256"] != source_binding():
        raise ValueError("候選必須先選定，來源不可漂移")
    training = {}
    for name, value in selected["all_formal_jobs"].items():
        _, _, _, actual = formal_result(core, FROZEN, name)
        training[name] = actual
        if (
            file_sha(core / name / "job-result.json") != value["job_result_sha256"]
            or value["best"] != actual["best"]
            or value["files"] != actual["files"]
        ):
            raise ValueError("正式job收據已變")
        if (
            not {"weights.safetensors", "best-continuation.pt", "latest-continuation.pt", "training-log.json"}
            <= actual["files"].keys()
        ):
            raise ValueError("正式weights與完整continuation不可省略")
        for filename, binding in actual["files"].items():
            artifact = core / name / "training" / filename
            if (
                Path(filename).name != filename
                or artifact.is_symlink()
                or artifact.stat().st_size != binding["bytes"]
                or file_sha(artifact) != binding["sha256"]
            ):
                raise ValueError("正式成品收到的bytes／SHA不同")
        load_trained(core / name / "training", recipes["jobs"][name]["kind"])
    expected_a = max(
        a_names,
        key=lambda name: (training[name]["best"]["dev_score"], -training[name]["best"]["update"], -a_names.index(name)),
    )
    if names["av"] != expected_a:
        raise ValueError("AV不是開final前依固定dev規則選出的候選")
    if selected["formal_recipe_sha256"] != file_sha(FROZEN / "formal-recipes.json"):
        raise ValueError("正式recipe已變")
    for key, name in (
        ("data_definition_sha256", "data-definition.json"),
        ("metrics_sha256", "metrics.json"),
        ("codec_selection_sha256", "codec-selection.json"),
    ):
        if selected[key] != file_sha(FROZEN / name):
            raise ValueError("final前固定資料／度量／codec已變")
    for kind, binding in selected["model_files"].items():
        if binding["job"] != names[kind] or binding["weights"] != training[names[kind]]["files"]["weights.safetensors"]:
            raise ValueError("選定模組binding與正式job不相符")
        path = core / binding["job"] / "training"
        if (
            file_sha(path / "training-result.json") != binding["training_result_sha256"]
            or file_sha(path / "weights.safetensors") != binding["weights"]["sha256"]
        ):
            raise ValueError("選定weights與實際不同")
    evidence = ROOT / "outputs/omni-core-selection-v1"
    if set(selected["selection_evidence"]) != {
        "calibration.json",
        "temperature.json",
        "policy-predictions.json",
        "policy.json",
    }:
        raise ValueError("calibration80／policy60的選擇證據不可省略")
    for filename, binding in selected["selection_evidence"].items():
        artifact = evidence / filename
        if (
            artifact.is_symlink()
            or artifact.stat().st_size != binding["bytes"]
            or file_sha(artifact) != binding["sha256"]
        ):
            raise ValueError("calibration／policy證據已改")
    temperature = json.loads((evidence / "temperature.json").read_bytes())
    policy = json.loads((evidence / "policy.json").read_bytes())
    data = json.loads((FROZEN / "data-definition.json").read_bytes())
    for filename, split in (("calibration.json", "calibration"), ("policy-predictions.json", "policy")):
        rows = json.loads((evidence / filename).read_bytes())
        expected_families = {row["family"] for row in data["C"]["panels"][split]}
        if len(rows) != len(expected_families) or {row["family"] for row in rows} != expected_families:
            raise ValueError("calibration80／policy60的固定家族不可省略或重複")
        if any(set(row["tasks"]) != {"choice", "info", "rubric"} for row in rows):
            raise ValueError("calibration／policy的三種題型不可省略")
    if (
        temperature["temperature"] != selected["temperature"]
        or policy["feasible"] != selected["policy_feasible"]
        or policy["threshold"] != selected["threshold"]
    ):
        raise ValueError("T／tau與當時選擇收據不同；不在這裡重新調參")
    return selected


def checked_audit():
    sources = evaluation_sources()
    audit = json.loads(AUDIT.read_bytes())
    if (
        audit.get("status") != "pass_for_reviewed_scope"
        or not audit.get("reviewer_task")
        or audit["reviewer_task"] == "/root"
    ):
        raise ValueError("final量測入口需要獨立技術原判／回核")
    for name, digest in sources.items():
        if audit.get("reviewed_files_sha256", {}).get(name) != digest:
            raise ValueError("final執行來源偏離稽核")
        if subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError("final來源須先提交")
    for path in (
        AUDIT,
        *(
            FROZEN / name
            for name in (
                "data-definition.json",
                "metrics.json",
                "formal-recipes.json",
                "codec-selection.json",
                "model-selection.json",
            )
        ),
    ):
        name = str(path.relative_to(ROOT))
        if subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT) != path.read_bytes():
            raise ValueError("final前稽核與選定manifest須先提交")
    return sources


def opening(core, expected, create=False):
    selected = checked_selection(core, expected)
    sources = checked_audit()
    value = {
        "schema": "omni-core-final-opening-v1",
        "model_selection_sha256": expected,
        "data_definition_sha256": selected["data_definition_sha256"],
        "metrics_sha256": selected["metrics_sha256"],
        "evaluation_sources_sha256": sources,
        "independent_audit_sha256": file_sha(AUDIT),
        "runtime": {"device": "cpu", "torch": str(torch.__version__), "python": platform.python_version()},
    }
    path = FROZEN / "final-opening.json"
    if create:
        immutable_json(path, {**value, "created_at": utc(), "final_opened": True})
    else:
        actual = json.loads(path.read_bytes())
        if actual.get("final_opened") is not True or any(actual.get(key) != item for key, item in value.items()):
            raise ValueError("final開封後來源／權重／策略／backend不可變更")
    return selected


def assistant(core, selected, old_a=None, old_b=None):
    names = selected["selected"]
    return OmniAssistant(
        load_trained(core / names["av"] / "training", "av"),
        load_trained(core / names["choice"] / "training", "choice"),
        load_trained(core / "codec/training", "codec"),
        load_trained(core / "talker/training", "talker"),
        old_a=old_a,
        old_b=old_b,
        temperature=selected["temperature"],
        threshold=selected["threshold"],
    )


def evaluate(part, core, expected, output, legacy):
    selected = opening(core, expected)
    frozen_opening = json.loads((FROZEN / "final-opening.json").read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    json_file(
        output / "part-origin.json",
        {
            "part": part,
            "created_at": utc(),
            "final_opened": True,
            "model_selection_sha256": expected,
            "final_opening_sha256": file_sha(FROZEN / "final-opening.json"),
            "data_definition_sha256": frozen_opening["data_definition_sha256"],
            "metrics_sha256": frozen_opening["metrics_sha256"],
            "evaluation_sources_sha256": frozen_opening["evaluation_sources_sha256"],
        },
    )
    data = json.loads((FROZEN / "data-definition.json").read_bytes())
    if part == "av":
        models = {
            name: measured.evaluate_av(load_trained(core / name / "training", "av"), data["AV"]["final"])
            for name in selected["all_formal_jobs"]
            if name.startswith("av-")
        }
        value = {"models": models, "paired_family_comparison": measured.paired_AV_comparison(models)}
    elif part == "sound":
        product = assistant(core, selected)
        value = {
            "codec_and_gold_program": measured.evaluate_sound(
                product.codec, product.talker, data["sound"]["final"], output / "sound"
            ),
            "actual_A_chain": measured.evaluate_chain(product, data["AV"]["final"], output / "chain"),
        }
        times = []
        for i, program in enumerate(data["sound"]["final"][:10]):
            try:
                times.append(
                    {
                        "family_index": i,
                        **measured.measure_playable_chunks(product, tuple(Event(**item) for item in program)),
                    }
                )
            except Exception as error:
                times.append({"family_index": i, "error": measured.error_record(error)})
        value["actual_playable_timing"] = times
    elif part == "choice":
        product = assistant(core, selected)
        primary = predict_families(product.c, data["C"]["panels"]["final"])
        control = CJudge(load_trained(core / "c-compact/training", "choice"))
        ablation = predict_families(control, data["C"]["panels"]["final"])
        value = {
            "primary": measured.evaluate_c(primary, selected["temperature"], selected["threshold"]),
            "data_control_raw_only": measured.evaluate_c(ablation, 1.0, None),
            "data_control_scope": "raw accuracy ablation; no separately fitted control T/tau or zero-acceptance policy comparison",
            "development_invariants": measured.choice_invariants(product.c.model, data["C"]["panels"]["dev"][:8]),
            "candidate_count_secondary": measured.choice_candidate_diagnostics(
                product.c, data["C"]["panels"]["final"], data["C"]["diagnostic_candidate_pool"]
            ),
        }
    elif part == "integration":
        old_a, old_b, receipt = legacy_preflight(*legacy)
        product = assistant(core, selected, old_a, old_b)
        value = {
            "legacy_binding": receipt,
            "auto": measured.evaluate_auto(product, data["Auto"]["families"]),
            "old_A_retention": measured.evaluate_old_a(
                product, data["old_A_retention"], data["counterfactual_tool_diagnostics"]
            ),
            "B_retention": measured.evaluate_b(product, data["B_retention"]["values"]),
        }
    else:
        raise ValueError("未知final部分")
    json_file(output / "results.json", value)
    json_file(
        output / "part-completion.json",
        {
            "part": part,
            "created_at": utc(),
            "measurement_completed": True,
            "capability_pass_not_implied": True,
            "results_sha256": file_sha(output / "results.json"),
            "part_origin_sha256": file_sha(output / "part-origin.json"),
            "final_opening_sha256": file_sha(FROZEN / "final-opening.json"),
            "model_selection_sha256": expected,
        },
    )


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("operation", choices=("legacy-preflight", "open", "av", "sound", "choice", "integration"))
    parser.add_argument("--core", type=Path, default=ROOT / "outputs/omni-formal-core-v1")
    parser.add_argument("--selection-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--old-a-root", type=Path)
    parser.add_argument("--old-b-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.operation != "open" and args.output is None:
        raise ValueError("量測部分／legacy preflight需要獨立輸出路徑")
    legacy = (args.data_root, args.old_a_root, args.old_b_root)
    if args.operation == "legacy-preflight":
        if any(path is None for path in legacy):
            raise ValueError("舊權重與資料路徑須明示")
        _, _, receipt = legacy_preflight(*legacy)
        immutable_json(args.output, receipt)
    elif args.operation == "open":
        opening(args.core, args.selection_sha256, create=True)
    else:
        evaluate(args.operation, args.core, args.selection_sha256, args.output, legacy)


if __name__ == "__main__":
    main()
