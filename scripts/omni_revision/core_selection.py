"""只用正式訓練紀錄／calibration／policy選候選；沒有final執行入口。"""

import argparse
import json
from pathlib import Path

from scripts.omni_revision.formal_local import FROZEN, ROOT, utc, verified_formal_training
from tiny_perceptron.omni_products.inference import SOUND_MAP, CJudge, load_trained
from tiny_perceptron.omni_products.materialize import public_question
from tiny_perceptron.omni_products.reliability import select_policy, select_temperature
from tiny_perceptron.omni_products.training_runtime import file_sha, json_file, source_binding


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def formal_result(output, frozen, name):
    return verified_formal_training(output, frozen, name)


def freeze_codec(output, frozen):
    _, _, result, training = formal_result(output, frozen, "codec")
    load_trained(output / "codec/training", "codec")
    value = {
        "schema": "omni-formal-codec-selection-v1",
        "created_at": utc(),
        "final_opened": False,
        "formal_recipe_sha256": result["formal_recipe_sha256"],
        "training_result_sha256": result["training_result_sha256"],
        "weights_sha256": training["files"]["weights.safetensors"]["sha256"],
        "model": training["config"]["model"],
        "dev_best": training["best"],
        "selection_rule": "fixed profile-selected architecture; checkpoint chosen only by development SNR",
    }
    immutable_json(frozen / "codec-selection.json", value)
    return value


def prediction(judge, question):
    result = judge.predict(public_question(question))
    # candidate_hidden是研究tensor，不把它誤序列化為公開JSON。
    return {key: result[key] for key in ("answer", "ids", "logits", "probabilities", "p_max")}


def predict_families(judge, rows):
    result = []
    for row in rows:
        tasks = {
            kind: {**prediction(judge, row["questions"][kind]), "gold": row["questions"][kind]["gold"]}
            for kind in ("choice", "info", "rubric")
        }
        result.append(
            {
                "family": row["family"],
                "source": row["source"],
                "panel": row["panel"],
                "tasks": tasks,
                "choice_info": prediction(judge, row["questions"]["choice_info"]),
                "world_supported": row["world_supported"],
                "choice_information_sufficient": row["choice_information_sufficient"],
            }
        )
    return result


def select_models(output, frozen, evidence):
    document = json.loads((frozen / "formal-recipes.json").read_bytes())
    all_results = {name: formal_result(output, frozen, name)[3] for name in document["job_order"]}
    a_names = [name for name in document["job_order"] if document["jobs"][name]["kind"] == "av"]
    if len(a_names) != 6:
        raise ValueError("六個正式A工作不可省略")
    for seed in (11, 29, 47):
        a, b = (all_results[f"av-{kind}-s{seed}"] for kind in ("additive", "time-rope"))
        if (
            a["initial_parameter_sha256"] != b["initial_parameter_sha256"]
            or a["trainable_parameters"] != b["trainable_parameters"]
        ):
            raise ValueError("同seed的AV表示對照初值／可學參數不同")
    left, right = (all_results[f"c-{mode}"] for mode in ("augmented", "compact"))
    if left["initial_parameter_sha256"] != right["initial_parameter_sha256"]:
        raise ValueError("C資料對照不可換初值")
    selected_a = max(
        a_names,
        key=lambda name: (
            all_results[name]["best"]["dev_score"],
            -all_results[name]["best"]["update"],
            -a_names.index(name),
        ),
    )
    judge = CJudge(load_trained(output / "c-augmented/training", "choice"))
    data = json.loads((frozen / "data-definition.json").read_bytes())
    calibration = predict_families(judge, data["C"]["panels"]["calibration"])
    temperature = select_temperature(calibration)
    policy_predictions = predict_families(judge, data["C"]["panels"]["policy"])
    policy = select_policy(policy_predictions, temperature["temperature"])
    evidence.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("calibration", calibration),
        ("temperature", temperature),
        ("policy-predictions", policy_predictions),
        ("policy", policy),
    ):
        json_file(evidence / f"{name}.json", value)
    selected = {
        "av": selected_a,
        "choice": "c-augmented",
        "choice_control": "c-compact",
        "codec": "codec",
        "talker": "talker",
    }
    value = {
        "schema": "omni-core-model-selection-v1",
        "created_at": utc(),
        "final_opened": False,
        "formal_recipe_sha256": file_sha(frozen / "formal-recipes.json"),
        "data_definition_sha256": document["data_definition_sha256"],
        "metrics_sha256": document["metrics_sha256"],
        "implementation_sha256": source_binding(),
        "selected": selected,
        "model_files": {
            name: {
                "job": job,
                "weights": all_results[job]["files"]["weights.safetensors"],
                "training_result_sha256": file_sha(output / job / "training/training-result.json"),
            }
            for name, job in selected.items()
        },
        "all_formal_jobs": {
            name: {
                "job_result_sha256": file_sha(output / name / "job-result.json"),
                "best": result["best"],
                "files": result["files"],
            }
            for name, result in all_results.items()
        },
        "temperature": temperature["temperature"],
        "policy_feasible": policy["feasible"],
        "threshold": policy["threshold"],
        "selection_evidence": {
            path.name: {"sha256": file_sha(path), "bytes": path.stat().st_size} for path in sorted(evidence.iterdir())
        },
        "sound_map": {label: [event.__dict__ for event in program] for label, program in SOUND_MAP.items()},
        "codec_selection_sha256": file_sha(frozen / "codec-selection.json"),
        "selection_sources_sha256": {str(Path(__file__).relative_to(ROOT)): file_sha(__file__)},
        "scope": "model/codec/T/tau/map selection only; no final predictions, capability verdict or publication",
    }
    immutable_json(frozen / "model-selection.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("operation", choices=("freeze-codec", "select-models"))
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/omni-formal-core-v1")
    parser.add_argument("--evidence", type=Path, default=ROOT / "outputs/omni-core-selection-v1")
    args = parser.parse_args()
    value = (
        freeze_codec(args.output_root, args.frozen)
        if args.operation == "freeze-codec"
        else select_models(args.output_root, args.frozen, args.evidence)
    )
    print(json.dumps({"schema": value["schema"], "created_at": value["created_at"], "final_opened": False}))


if __name__ == "__main__":
    main()
