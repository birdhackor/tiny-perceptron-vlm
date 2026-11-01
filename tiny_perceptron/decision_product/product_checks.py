"""手寫真值表與實際服務驗收；不把這些相關壓力測試當作獨立 final 樣本。"""

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from .data import family_key
from .experiment import write_json
from .qualification import inference_sources, load_experiment, register_exposure, sha256
from .service import Action, DecisionEngine, Device

# 三項期望皆由人工明列，驗收時不呼叫資料生成器的標籤函式。
# task, permission, ready, action, information, grade
TRUTH = (
    ("record", False, False, "stop", "yes", "0"),
    ("record", False, True, "stop", "yes", "1"),
    ("record", True, False, "wait", "yes", "1"),
    ("record", True, True, "record", "yes", "2"),
    ("record", None, False, "none", "no", "none"),
    ("record", None, True, "none", "no", "none"),
    ("record", False, None, "none", "no", "none"),
    ("record", True, None, "none", "no", "none"),
    ("record", None, None, "none", "no", "none"),
    ("play", False, False, "stop", "yes", "0"),
    ("play", False, True, "stop", "yes", "1"),
    ("play", True, False, "wait", "yes", "2"),
    ("play", True, True, "play", "yes", "2"),
    ("play", None, False, "none", "no", "none"),
    ("play", None, True, "none", "no", "none"),
    ("play", False, None, "none", "yes", "none"),
    ("play", True, None, "none", "yes", "none"),
    ("play", None, None, "none", "no", "none"),
    ("save", False, False, "stop", "yes", "0"),
    ("save", False, True, "stop", "yes", "2"),
    ("save", True, False, "wait", "yes", "1"),
    ("save", True, True, "save", "yes", "2"),
    ("save", None, False, "none", "yes", "none"),
    ("save", None, True, "none", "yes", "none"),
    ("save", False, None, "none", "no", "none"),
    ("save", True, None, "none", "no", "none"),
    ("save", None, None, "none", "no", "none"),
)
IDS = ("前台錄音室", "備援播放台", "archive / NAS", "stage-monitor", "unit-05", "封存備援六")


def fixture_devices(task, permission, ready, count, target_position):
    rows = [Device(IDS[i], ("record", "play", "save")[i % 3], bool(i % 2), bool((i + 1) % 2)) for i in range(count)]
    rows[target_position] = Device(IDS[target_position], task, permission, ready)
    return rows


def fixture_family_keys():
    keys = set()
    for task, permission, ready, *_ in TRUTH:
        for count in (3, 4, 5, 6):
            for position in range(count):
                devices = fixture_devices(task, permission, ready, count, position)
                facts = [
                    (
                        ("record", "play", "save").index(item.task),
                        -1 if item.permission is None else int(item.permission),
                        -1 if item.ready is None else int(item.ready),
                    )
                    for item in devices
                ]
                keys.add(family_key(facts))
    return sorted(keys)


def run_product_checks(engine):
    rng = random.Random(826039)
    metrics = defaultdict(lambda: [0, 0])
    failures, observations, latencies = [], [], []
    invariance_failures, accepted_errors, known_accepted, known = 0, 0, 0, 0
    for task, permission, ready, action, information, grade in TRUTH:
        for language in ("zh", "en"):
            for count in (3, 4, 5, 6):
                for position in range(count):
                    devices = fixture_devices(task, permission, ready, count, position)
                    target = devices[position].id
                    session = engine.session(devices, language)
                    expected = {"choice": action, "info": information, "rubric": grade}
                    for kind, invoke in (
                        ("choice", session.choose),
                        ("info", session.information),
                        ("rubric", session.grade),
                    ):
                        started = time.perf_counter()
                        prediction = invoke(target)
                        latencies.append(1000 * (time.perf_counter() - started))
                        correct = prediction.model_choice == expected[kind]
                        for key in (
                            kind,
                            f"{kind}/language/{language}",
                            f"{kind}/devices/{count}",
                            f"{kind}/target-slot/{position}",
                        ):
                            metrics[key][0] += correct
                            metrics[key][1] += 1
                        row = {
                            "task": task,
                            "permission": permission,
                            "ready": ready,
                            "language": language,
                            "devices": count,
                            "target_position": position,
                            "kind": kind,
                            "expected": expected[kind],
                            "prediction": prediction.to_dict(),
                        }
                        observations.append(row)
                        if not correct:
                            failures.append(row)
                        if kind == "choice":
                            accepted_errors += prediction.status == "accepted" and not correct
                            known += action != "none"
                            known_accepted += action != "none" and prediction.status == "accepted"
                            options = [
                                Action(value, target, value) for value in ("stop", "wait", "record", "play", "save")
                            ]
                            rng.shuffle(options)
                            shuffled = session.choose(target, options)
                            uncached = session.choose(target, options, use_cache=False)
                            consistent = (
                                shuffled.model_choice == prediction.model_choice
                                and shuffled.status == prediction.status
                                and abs(shuffled.confidence - prediction.confidence) < 1e-5
                                and uncached.model_choice == shuffled.model_choice
                                and abs(uncached.confidence - shuffled.confidence) < 1e-5
                            )
                            invariance_failures += not consistent
                    if action != "none":
                        # 七項中有他人的正確動作，卻沒有目標的正確動作；必須拒答。
                        others = [
                            Action(f"{i}-{value}", row.id, value)
                            for i, row in enumerate(devices)
                            for value in ("stop", "wait", "record", "play", "save")
                            if row.id != target or value != action
                        ]
                        options = rng.sample(others, 7)
                        prediction = session.choose(target, options)
                        correct = prediction.model_choice == "none"
                        metrics["removed"][0] += correct
                        metrics["removed"][1] += 1
                        accepted_errors += prediction.status == "accepted" and not correct
                        row = {
                            "task": task,
                            "permission": permission,
                            "ready": ready,
                            "language": language,
                            "devices": count,
                            "target_position": position,
                            "kind": "removed",
                            "expected": "none",
                            "prediction": prediction.to_dict(),
                        }
                        observations.append(row)
                        if not correct:
                            failures.append(row)
    result = {key: {"correct": c, "observations": n, "accuracy": c / n} for key, (c, n) in sorted(metrics.items())}
    gates = {
        "manual_kind_accuracy": all(
            result[kind]["accuracy"] >= 0.98 for kind in ("choice", "info", "rubric", "removed")
        ),
        "manual_subgroups": all(group["accuracy"] >= 0.95 for key, group in result.items() if "/" in key),
        "no_wrong_accepted": accepted_errors == 0,
        "known_action_coverage": known_accepted / known >= 0.90,
        "branch_permutation_and_cache": invariance_failures == 0,
    }
    return {
        "schema": "c-product-manual-service-checks-v1",
        "pass": all(gates.values()),
        "gates": gates,
        "metrics": result,
        "accepted_errors": accepted_errors,
        "known_action_accepted": known_accepted,
        "known_action_observations": known,
        "known_action_coverage": known_accepted / known,
        "invariance_failures": invariance_failures,
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
            "scope": "single sequential CPU request, includes schema-generated text, SHA cache check and inference; mixed cold/warm states",
        },
        "statistical_unit": "Deterministic engineering stress cases. Repeated local facts, renderings, slots and counterfactuals are correlated; not extra independent final families.",
        "failures": failures,
        "observations": observations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root, output = Path(args.experiment), Path(args.output)
    if output.exists():
        raise ValueError("不可覆寫既有驗收結果")
    model, tokenizer, best = load_experiment(root)
    # 手寫題若參與開發或驗收，同一可見世界也不能重回 fresh final。
    register_exposure(root.parent, fixture_family_keys(), "manual-engineering-fixture", sha256(Path(__file__)))
    selection = (
        json.loads((root / "qualification/selection.json").read_text())
        if (root / "qualification/selection.json").exists()
        else None
    )
    temperatures = selection["temperatures"] if selection else {kind: 1.0 for kind in ("choice", "info", "rubric")}
    threshold = selection["selected"]["threshold"] if selection and selection["selected"] else 0.9
    engine = DecisionEngine._for_development(model, tokenizer, temperatures, threshold)
    result = run_product_checks(engine)
    result["provenance"] = {
        "weights_sha256": best["weights_sha256"],
        "tokenizer_sha256": sha256(root / "tokenizer.json"),
        "checker_sha256": sha256(Path(__file__)),
        "service_sha256": sha256(Path(__file__).with_name("service.py")),
        "selection_sha256": sha256(root / "qualification/selection.json") if selection else None,
        "threshold": threshold,
        "temperatures": temperatures,
        "inference_sources": inference_sources(),
    }
    write_json(output, result)
    print(
        json.dumps({key: value for key, value in result.items() if key not in {"observations", "failures", "metrics"}}),
        flush=True,
    )


if __name__ == "__main__":
    main()
