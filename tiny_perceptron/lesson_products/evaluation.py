"""calibration／policy選設定；固定bytes後，才由另一入口評最後test。"""

import hashlib
import itertools
import json
import platform
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from .data import CANDIDATES, read_dataset
from .inference import Product, auto_solve, parse_state, rule_depth
from .model import Config, LoopedSolver
from .training import encoded_data, metrics, predict, save_json, state_digest

TEMPERATURES = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0, 8.0)
SOURCE_FILES = tuple(
    Path(__file__).with_name(name) for name in ("data.py", "model.py", "training.py", "inference.py", "evaluation.py")
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def c_logits(product, rows):
    data = encoded_data(rows, "c", product.tokenizer, product.config, product.device)
    return predict(product.model, data)


def b_logits(product, rows, depths=(1, 3, 5)):
    data = encoded_data(rows, "b", product.tokenizer, product.config, product.device)
    return {depth: predict(product.model, data, loops=depth) for depth in depths}


def chosen_meanings(logits, rows):
    # 與公開judge一致：實際相等分數時用語義ID，不能用候選順序。
    return [
        min(zip(row["candidates"], scores.tolist(), strict=False), key=lambda item: (-item[1], item[0]))[0]
        for scores, row in zip(logits, rows, strict=True)
    ]


def c_report(logits, rows, temperatures):
    report = {}
    selected = chosen_meanings(logits, rows)
    for question in CANDIDATES:
        indices = [i for i, row in enumerate(rows) if row["question"] == question]
        if not indices:
            continue
        labels = [rows[i]["candidates"].index(rows[i]["gold"]) for i in indices]
        result = {
            "before": metrics(logits[indices], labels),
            "after": metrics(logits[indices], labels, temperatures[question]),
        }
        correct = sum(selected[i] == rows[i]["gold"] for i in indices)
        for value in result.values():
            value.update(correct=correct, accuracy=correct / len(indices))
        report[question] = result
    return report


def auto_inputs(b_rows):
    return [
        {
            "state": row["state"],
            "question": question,
            "candidates": list(CANDIDATES[question]),
            "gold": row["task"] if question == "choice" else "yes",
        }
        for row in b_rows
        for question in ("choice", "info")
    ]


def auto_quality(b_rows, predictions, c_predictions, policy, fixed=None):
    success, covered, conditional, total_loops = 0, 0, 0, 0
    for i, row in enumerate(b_rows):
        task, info = c_predictions[2 * i : 2 * i + 2]
        accepted = task in ("boxes", "sum") and info == "yes"
        if not accepted:
            continue
        covered += 1
        depth = fixed or rule_depth(policy, task, parse_state(row["state"]).get("event_count", 0))
        total_loops += depth
        correct = int(predictions[depth][i]) == row["gold"]
        success += correct
        conditional += correct
    count = len(b_rows)
    return {
        "count_complete_eligible": count,
        "correct_end_to_end": success,
        "accuracy_end_to_end": success / count,
        "covered": covered,
        "coverage": covered / count,
        "conditional_accuracy": conditional / covered if covered else None,
        "mean_actual_b_loops_all_complete": total_loops / count,
        "mean_b_loops_accepted": total_loops / covered if covered else None,
        "c_calls_per_input": 2,
        "denominator_scope": "Every predeclared truly supported complete B input, including C rejections as failures. C cost is additional; loop counts are not end-to-end time.",
    }


def make_artifact_manifest(directory, product, data_sha, freeze_sha):
    directory = Path(directory)
    names = [
        "model.pt",
        "config.json",
        "tokenizer.json",
        "training-receipt.json",
        "calibration.json" if product == "c" else "auto-policy.json",
    ]
    value = {
        "version": 1,
        "product": product,
        "data_manifest_sha256": data_sha,
        "settings_freeze_sha256": freeze_sha,
        "files": {
            name: {"sha256": sha(directory / name), "bytes": (directory / name).stat().st_size} for name in names
        },
    }
    save_json(directory / "artifact-manifest.json", value)
    return value


def select_settings(data_directory, b_directory, c_directory, output, device="cpu", mechanism_directories=None):
    """不執行最後test；所有搜尋都是預定grid、只看開發切分。"""
    torch.set_num_threads(2)
    output = Path(output)
    if output.exists():
        raise ValueError("每次凍結另立新檔，不覆寫舊設定或紀錄")
    b, c = Product(b_directory, "b", device, verify=False), Product(c_directory, "c", device, verify=False)
    receipts = {
        p: json.loads((Path(d) / "training-receipt.json").read_bytes())
        for p, d in (("b", b_directory), ("c", c_directory))
    }
    data_sha = sha(Path(data_directory) / "manifest.json")
    for product, receipt in receipts.items():
        if (
            not receipt["weights_changed"]
            or not receipt["resume_verification"]["exact_next_step"]
            or "L4" not in (receipt["environment"]["gpu"] or "")
        ):
            raise ValueError("需真實L4更新與精確續訓驗證，不接受僅有檔案的成品：" + product)
        if receipt["data_manifest_sha256"] != data_sha:
            raise ValueError("成品訓練資料不是目前這一份manifest")
        instance = b if product == "b" else c
        selected = torch.load(instance.directory / "best.pt", weights_only=True, map_location="cpu")
        meta = selected["metadata"]
        actual = state_digest(instance.model.state_dict())
        if (
            receipt["selected_weights_sha256"] != actual
            or state_digest(selected["model"]) != actual
            or selected["config"] != asdict(instance.config)
            or meta["tokenizer_sha256"] != sha(instance.directory / "tokenizer.json")
            or meta["product"] != product
            or meta["mode"] != ("multi" if product == "b" else "candidate")
            or meta["data_manifest_sha256"] != data_sha
            or meta["step"] != receipt["selected_step"]
            or meta["run_identity"] != receipt["run_identity"]
        ):
            raise ValueError("實際成品權重／config／tokenizer與原selected best checkpoint及訓練收據不符")
    if not mechanism_directories or set(mechanism_directories) != {"fixed3", "odd"}:
        raise ValueError("必須先完成兩個配對機制訓練，再凍結與查看最後測試")
    mechanism = {}
    for mode, directory in mechanism_directories.items():
        value = torch.load(Path(directory) / "last.pt", weights_only=True, map_location="cpu")
        metadata = value["metadata"]
        if metadata["mode"] != mode or metadata["data_manifest_sha256"] != data_sha:
            raise ValueError("機制權重配方或資料不符")
        mechanism[mode] = {
            "last.pt": sha(Path(directory) / "last.pt"),
            "training-receipt.json": sha(Path(directory) / "training-receipt.json"),
            "initial_weights_sha256": metadata["initial_weights_sha256"],
            "last_step": metadata["step"],
            "seed": metadata["seed"],
            "config": value["config"],
            "training_recipe": metadata["training_recipe"],
            "depth_weights": metadata["depth_weights"],
            "tokenizer_sha256": sha(Path(directory) / "tokenizer.json"),
        }
        if metadata["tokenizer_sha256"] != mechanism[mode]["tokenizer_sha256"] or mechanism[mode][
            "tokenizer_sha256"
        ] != sha(b.directory / "tokenizer.json"):
            raise ValueError("機制tokenizer不同，不能用同一編碼冒稱配對")
    if any(
        mechanism["fixed3"][key] != mechanism["odd"][key]
        for key in (
            "initial_weights_sha256",
            "last_step",
            "seed",
            "config",
            "training_recipe",
            "depth_weights",
            "tokenizer_sha256",
        )
    ):
        raise ValueError("機制比較需相同初始化、seed及實際更新次數，不能拿不同policy best稱配對")
    cal_rows = read_dataset(data_directory, "c", "calibration")
    cal_logits = c_logits(c, cal_rows)
    temperatures = {}
    for question in CANDIDATES:
        indices = [i for i, row in enumerate(cal_rows) if row["question"] == question]
        labels = [cal_rows[i]["candidates"].index(cal_rows[i]["gold"]) for i in indices]
        temperatures[question] = min(
            TEMPERATURES, key=lambda t: (metrics(cal_logits[indices], labels, t)["cross_entropy"], t)
        )
    calibration = {
        "version": 1,
        "temperatures": temperatures,
        "grid": list(TEMPERATURES),
        "selected_on": "calibration only; minimize per-question mean CE",
        "measurements": c_report(cal_logits, cal_rows, temperatures),
    }
    save_json(Path(c_directory) / "calibration.json", calibration)
    policy_rows = read_dataset(data_directory, "b", "policy")
    solver_logits = b_logits(b, policy_rows)
    predictions = {depth: values.argmax(-1) for depth, values in solver_logits.items()}
    routing = auto_inputs(policy_rows)
    c_predictions = chosen_meanings(c_logits(c, routing), routing)
    grid = []
    for low, medium, arithmetic in itertools.product(range(5), range(5), (1, 3, 5)):
        if low > medium:
            continue
        rule = {"version": 1, "low_max_events": low, "medium_max_events": medium, "sum_depth": arithmetic}
        quality = auto_quality(policy_rows, predictions, c_predictions, rule)
        grid.append({"rule": rule, "quality": quality})
    feasible = [row for row in grid if row["quality"]["mean_actual_b_loops_all_complete"] <= 3]
    selected = min(
        feasible,
        key=lambda row: (
            -row["quality"]["accuracy_end_to_end"],
            row["quality"]["mean_actual_b_loops_all_complete"],
            json.dumps(row["rule"], sort_keys=True),
        ),
    )
    rule = {
        **selected["rule"],
        "selected_on": "policy only; maximum end-to-end accuracy subject mean actual B loops over all complete eligible<=3; ties lower mean loops then lexical rule",
        "policy_quality": selected["quality"],
        "grid_count": len(grid),
    }
    save_json(Path(b_directory) / "auto-policy.json", rule)
    model_files = {}
    for product, directory in (("b", b_directory), ("c", c_directory)):
        names = (
            "model.pt",
            "config.json",
            "tokenizer.json",
            "training-receipt.json",
            "auto-policy.json" if product == "b" else "calibration.json",
        )
        model_files[product] = {name: sha(Path(directory) / name) for name in names}
    freeze = {
        "version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "data_manifest_sha256": sha(Path(data_directory) / "manifest.json"),
        "products": model_files,
        "source_sha256s": {p.name: sha(p) for p in SOURCE_FILES},
        "protocol": {
            "b_primary": "boxes, high=5 loops, accuracy>=0.90",
            "b_secondary": "sum accuracy target0.85; any failure is reported",
            "c_primary": "Choice accuracy>=0.90 and semantic reverse-reorder consistency>=0.99",
            "auto": "all supported complete heldout examples; low1/medium3/high5/auto; C cost included in integrated timings",
            "mechanism": "separate matched actual last-step fixed3 vs sampled1/3/5 checkpoints, evaluation1..6; no monotonic guarantee",
        },
        "calibration": calibration,
        "auto_search": grid,
        "test_outcomes_observed": False,
        "mechanisms": mechanism,
    }
    save_json(output, freeze)
    freeze_sha = sha(output)
    for product, directory in (("b", b_directory), ("c", c_directory)):
        make_artifact_manifest(directory, product, freeze["data_manifest_sha256"], freeze_sha)
    return {"settings_freeze_sha256": freeze_sha, "temperatures": temperatures, "auto_rule": rule}


def b_report(logits, rows):
    groups = {task: [i for i, row in enumerate(rows) if row["task"] == task] for task in ("boxes", "sum")}
    for length in range(1, 7):
        groups[f"boxes_length_{length}"] = [
            i for i, row in enumerate(rows) if row["task"] == "boxes" and len(row["world"]["events"]) == length
        ]
    for name in ("answer_from_initial", "last_write_to_other_box", "write_then_swap", "repeated_swap"):
        groups[name] = [i for i, row in enumerate(rows) if row["task"] == "boxes" and row["strata"][name]]
    labels = [row["gold"] for row in rows]
    return {
        name: {
            **metrics(logits[indices], [labels[i] for i in indices]),
            "families": len({rows[i]["family_id"] for i in indices}),
        }
        for name, indices in groups.items()
        if indices
    }


def frozen_guard(freeze_path, data_directory, b_directory, c_directory, mechanism_directories):
    freeze = json.loads(Path(freeze_path).read_bytes())
    if freeze["test_outcomes_observed"] is not False or freeze["data_manifest_sha256"] != sha(
        Path(data_directory) / "manifest.json"
    ):
        raise ValueError("不是未看最後測試時固定的相同資料")
    if freeze["source_sha256s"] != {p.name: sha(p) for p in SOURCE_FILES}:
        raise ValueError("評估程式在凍結後改動；不可默默重選或重跑")
    for product, directory in (("b", b_directory), ("c", c_directory)):
        for name, expected in freeze["products"][product].items():
            if sha(Path(directory) / name) != expected:
                raise ValueError("凍結後成品或推論設定改動：" + product + "/" + name)
        manifest = json.loads((Path(directory) / "artifact-manifest.json").read_bytes())
        if manifest["data_manifest_sha256"] != freeze["data_manifest_sha256"] or manifest[
            "settings_freeze_sha256"
        ] != sha(freeze_path):
            raise ValueError("成品manifest的資料／設定凍結身分不符")
    for mode, directory in mechanism_directories.items():
        for name in ("last.pt", "training-receipt.json"):
            if sha(Path(directory) / name) != freeze["mechanisms"][mode][name]:
                raise ValueError("凍結後機制權重改動")
        if sha(Path(directory) / "tokenizer.json") != freeze["mechanisms"][mode]["tokenizer_sha256"]:
            raise ValueError("凍結後機制tokenizer改動")
    if set(mechanism_directories) != set(freeze["mechanisms"]):
        raise ValueError("機制配對不完整")
    return freeze


def latency_report(b, c, rows):
    # 穩定ID抽樣，不依題目答對與否選快／簡單案例。
    sample = sorted(rows, key=lambda row: hashlib.sha256(("latency-v1:" + row["id"]).encode()).hexdigest())[:24]
    result = {}
    for effort in ("low", "medium", "high", "auto"):
        for row in sample[:3]:
            auto_solve(b, c, row["state"], effort)
        measurements, c_times, direct = [], [], []
        for _ in range(3):
            for row in sample:
                answer = auto_solve(b, c, row["state"], effort)
                measurements.append(answer["seconds"])
                c_times.append(answer["c_seconds"])
                if effort != "auto":
                    depth = {"low": 1, "medium": 3, "high": 5}[effort]
                    direct.append(b.solve(row["state"], depth)["seconds"])
        result[effort] = {
            "requests": len(measurements),
            "integrated_mean_seconds": statistics.mean(measurements),
            "integrated_median_seconds": statistics.median(measurements),
            "integrated_p95_seconds": sorted(measurements)[int(0.95 * (len(measurements) - 1))],
            "c_mean_seconds": statistics.mean(c_times),
            "direct_b_mean_seconds": statistics.mean(direct) if direct else None,
        }
    return {
        "sample_ids": [row["id"] for row in sample],
        "warmup_per_effort": 3,
        "repetitions": 3,
        "model_reload_excluded": True,
        "batch_size": 1,
        "device": str(b.device),
        "threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "measurements": result,
    }


def evaluate_frozen(freeze_path, data_directory, b_directory, c_directory, output, mechanism_directories, device="cpu"):
    torch.set_num_threads(2)
    output = Path(output)
    freeze_marker = Path(freeze_path).with_suffix(".test-started.json")
    if freeze_marker.exists() or output.exists() or output.with_suffix(".started.json").exists():
        raise ValueError("最後測試只能由此凍結首次啟動；不覆寫成績或啟動紀錄")
    freeze = frozen_guard(freeze_path, data_directory, b_directory, c_directory, mechanism_directories)
    started = time.perf_counter()
    # marker綁在固定freeze路徑，換output也不能再啟動；exclusive create避免競態。
    marker_value = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "settings_freeze_sha256": sha(freeze_path),
        "output": str(output),
        "scope": "One start for this fixed freeze file, across output names. Not a global lock across copied files/machines; deliberate protocol bypass is outside the tool guarantee.",
    }
    with freeze_marker.open("x") as stream:
        json.dump(marker_value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    save_json(
        output.with_suffix(".started.json"),
        {
            "recorded_at": datetime.now(UTC).isoformat(),
            "settings_freeze_sha256": sha(freeze_path),
            "scope": "Written before computing any final heldout model predictions; frozen weights/settings/data/code checked.",
        },
    )
    b, c = Product(b_directory, "b", device), Product(c_directory, "c", device)
    extra_raw = []
    rows = read_dataset(data_directory, "b", "test")
    solver_logits = b_logits(b, rows, range(1, 7))
    solver_report = {str(depth): b_report(logits, rows) for depth, logits in solver_logits.items()}
    predictions = {depth: logits.argmax(-1) for depth, logits in solver_logits.items()}
    routing = auto_inputs(rows)
    c_predictions = chosen_meanings(c_logits(c, routing), routing)
    integrated = {
        effort: auto_quality(rows, predictions, c_predictions, b.auto_policy, fixed=depth)
        for effort, depth in (("low", 1), ("medium", 3), ("high", 5), ("auto", None))
    }
    c_rows = read_dataset(data_directory, "c", "test")
    scores = c_logits(c, c_rows)
    semantic = chosen_meanings(scores, c_rows)
    temperatures = c.calibration["temperatures"]
    judgments = c_report(scores, c_rows, temperatures)
    choice_indices = [i for i, row in enumerate(c_rows) if row["question"] == "choice"]
    reordered = [{**c_rows[i], "candidates": list(reversed(c_rows[i]["candidates"]))} for i in choice_indices]
    reverse_choices = chosen_meanings(c_logits(c, reordered), reordered)
    reversed_by_index = dict(zip(choice_indices, reverse_choices, strict=True))
    consistent = sum(semantic[i] == value for i, value in zip(choice_indices, reverse_choices, strict=True))
    missing = [i for i in choice_indices if c_rows[i].get("missing_best")]
    errors = []
    for i, row in enumerate(c_rows):
        if semantic[i] == row["gold"]:
            continue
        probabilities = (scores[i, : len(row["candidates"])] / temperatures[row["question"]]).softmax(-1)
        n = len(row["candidates"])
        errors.append(
            {
                "id": row["id"],
                "state": row["state"],
                "question": row["question"],
                "gold": row["gold"],
                "predicted": semantic[i],
                "concentration": (n * float(probabilities.max()) - 1) / (n - 1),
            }
        )
    errors.sort(key=lambda row: (-row["concentration"], row["id"]))
    shifts = {}
    for split in ("wording_shift", "entity_shift", "long_shift"):
        shifted = read_dataset(data_directory, "b", split)
        shifted_scores = b_logits(b, shifted, (5,))[5]
        shifts[split] = b_report(shifted_scores, shifted)
        extra_raw.extend(
            {
                "product": "b",
                "split": split,
                "id": row["id"],
                "family_id": row["family_id"],
                "task": row["task"],
                "gold": row["gold"],
                "depth": 5,
                "raw_scores": score.tolist(),
            }
            for row, score in zip(shifted, shifted_scores, strict=True)
        )
    c_shift = read_dataset(data_directory, "c", "wording_shift")
    shifted_c_scores = c_logits(c, c_shift)
    shifts["c_wording"] = c_report(shifted_c_scores, c_shift, temperatures)
    extra_raw.extend(
        {
            "product": "c",
            "split": "wording_shift",
            "id": row["id"],
            "family_id": row["family_id"],
            "question": row["question"],
            "candidates": row["candidates"],
            "gold": row["gold"],
            "raw_scores": score[: len(row["candidates"])].tolist(),
        }
        for row, score in zip(c_shift, shifted_c_scores, strict=True)
    )
    mechanisms = {}
    for mode, directory in mechanism_directories.items():
        value = torch.load(Path(directory) / "last.pt", weights_only=True, map_location="cpu")
        model = LoopedSolver(Config(**value["config"])).to(device).eval()
        model.load_state_dict(value["model"])
        data = encoded_data(rows, "b", b.tokenizer, model.config, device)
        mechanism_scores = {depth: predict(model, data, depth) for depth in range(1, 7)}
        mechanisms[mode] = {
            "actual_last_step": value["metadata"]["step"],
            "same_update_count_last_weights": True,
            "terminal_supervision": [3] if mode == "fixed3" else [1, 3, 5],
            "evaluation": {str(depth): b_report(score, rows) for depth, score in mechanism_scores.items()},
        }
        extra_raw.extend(
            {
                "product": "mechanism",
                "mode": mode,
                "split": "test",
                "id": row["id"],
                "family_id": row["family_id"],
                "task": row["task"],
                "gold": row["gold"],
                "scores_by_depth": {str(depth): score[i].tolist() for depth, score in mechanism_scores.items()},
            }
            for i, row in enumerate(rows)
        )
    raw = output.with_suffix(".predictions.jsonl")
    with raw.open("x") as stream:
        for i, row in enumerate(rows):
            stream.write(
                json.dumps(
                    {
                        "product": "b",
                        "id": row["id"],
                        "family_id": row["family_id"],
                        "task": row["task"],
                        "gold": row["gold"],
                        "answers_by_depth": {str(d): int(p[i]) for d, p in predictions.items()},
                        "scores_by_depth": {str(d): score[i].tolist() for d, score in solver_logits.items()},
                        "split": "test",
                        "c_choice": c_predictions[2 * i],
                        "c_info": c_predictions[2 * i + 1],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for i, row in enumerate(c_rows):
            stream.write(
                json.dumps(
                    {
                        "product": "c",
                        "split": "test",
                        "id": row["id"],
                        "family_id": row["family_id"],
                        "question": row["question"],
                        "candidates": row["candidates"],
                        "gold": row["gold"],
                        "selected": semantic[i],
                        "reverse_selected": reversed_by_index.get(i),
                        "raw_scores": scores[i, : len(row["candidates"])].tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for record in extra_raw:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    labels = [r["gold"] for r in rows if r["task"] == "boxes"]
    import re

    last_number = [int(re.findall(r"[0-9]+", r["state"])[-1]) for r in rows if r["task"] == "boxes"]
    report = {
        "version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "settings_freeze_sha256": sha(freeze_path),
        "data_manifest_sha256": freeze["data_manifest_sha256"],
        "b": solver_report,
        "c": judgments,
        "c_reorder": {
            "count": len(choice_indices),
            "consistent": consistent,
            "consistency": consistent / len(choice_indices),
        },
        "c_missing_best": {"count": len(missing), "correct": sum(semantic[i] == c_rows[i]["gold"] for i in missing)},
        "c_high_concentration_errors": {
            "threshold": 0.9,
            "count": sum(row["concentration"] >= 0.9 for row in errors),
            "total_errors": len(errors),
            "highest_concentration_examples": errors[:10],
        },
        "auto": integrated,
        "shifts": shifts,
        "mechanisms": mechanisms,
        "last_number_baseline_boxes_accuracy": sum(a == b for a, b in zip(labels, last_number, strict=True))
        / len(labels),
        "latency": latency_report(b, c, rows),
        "predictions": {"file": raw.name, "sha256": sha(raw), "bytes": raw.stat().st_size},
        "gates": {
            "b_primary": solver_report["5"]["boxes"]["accuracy"] >= 0.90,
            "b_secondary_target": solver_report["5"]["sum"]["accuracy"] >= 0.85,
            "c_choice": judgments["choice"]["after"]["accuracy"] >= 0.90,
            "c_reorder": consistent / len(choice_indices) >= 0.99,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Finite fixed grammar; C includes disclosed identical generic observations across family splits. Heldout-family scores are not open-domain natural-language generalization. More loops may worsen accuracy/CE. Latency is CPU/device-specific; concentration is not probability of correctness.",
    }
    save_json(output, report)
    return report
