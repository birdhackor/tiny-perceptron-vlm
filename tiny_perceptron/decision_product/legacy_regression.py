"""在完整 repo 核對舊60情境的工作語意；已暴露舊題只作回歸，不當新 final。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .data import family_key
from .experiment import write_json
from .qualification import inference_sources, load_experiment, register_exposure, sha256
from .service import Action, DecisionEngine, Device


def check(root):
    # 舊教材的模組只在這個 repo 內回歸工具使用，成品推論完全不依賴它們。
    from tiny_perceptron.omni_products import data as old_data
    from tiny_perceptron.omni_products import questions as old_questions

    panels = old_questions.panel_c("final")
    visible = [
        old_data.visible_c_facts(tuple(panel["parent"]), index, panel["panel"]) for index, panel in enumerate(panels)
    ]
    keys = [
        family_key(
            [
                (
                    item["task"],
                    -1 if item["permission"] is None else int(item["permission"]),
                    -1 if item["ready"] is None else int(item["ready"]),
                )
                for item in facts
            ]
        )
        for facts in visible
    ]
    register_exposure(root.parent, keys, "old-exposed-final-semantic-regression", sha256(Path(__file__)))
    model, tokenizer, best = load_experiment(root)
    selection = json.loads((root / "qualification/selection.json").read_text())
    engine = DecisionEngine._for_development(
        model, tokenizer, selection["temperatures"], selection["selected"]["threshold"]
    )
    groups, rows = defaultdict(lambda: [0, 0]), []
    accepted, errors = 0, 0
    for panel, facts in zip(panels, visible):
        devices = [
            Device(name, ("record", "play", "save")[fact["task"]], fact["permission"], fact["ready"])
            for name, fact in zip(("A", "B", "C"), facts)
        ]
        session = engine.session(devices, "zh" if panel["source"] == "chinese" else "en")
        predictions = {
            "choice": session.choose(
                "A", [Action(option.id, "A", option.id) for option in panel["questions"]["choice"].candidates]
            ),
            "info": session.information("B"),
            "rubric": session.grade("C"),
        }
        for kind, prediction in predictions.items():
            gold = panel["questions"][kind].gold
            correct = prediction.model_choice == gold
            groups[kind][0] += correct
            groups[kind][1] += 1
            if kind == "choice":
                accepted += prediction.status == "accepted"
                errors += prediction.status == "accepted" and not correct
            rows.append(
                {
                    "original_family": panel["family"],
                    "visible_family": family_key(
                        [
                            (
                                item["task"],
                                -1 if item["permission"] is None else int(item["permission"]),
                                -1 if item["ready"] is None else int(item["ready"]),
                            )
                            for item in facts
                        ]
                    ),
                    "original_source": panel["source"],
                    "kind": kind,
                    "gold": gold,
                    "prediction": prediction.to_dict(),
                }
            )
    metrics = {kind: {"correct": c, "families": n, "accuracy": c / n} for kind, (c, n) in groups.items()}
    gates = {
        "choice": metrics["choice"]["accuracy"] >= 0.95,
        "info": metrics["info"]["accuracy"] >= 0.97,
        "rubric": metrics["rubric"]["accuracy"] >= 0.95,
        "no_accepted_error": errors == 0,
        "accepted_coverage": accepted / 60 >= 0.60,
    }
    return {
        "schema": "c-product-legacy-semantic-regression-v1",
        "pass": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "accepted": accepted,
        "accepted_errors": errors,
        "observations": rows,
        "scope": "Original60 exposed final worlds and original primary gold labels; new typed API renders these facts into its own supported text. This is semantic regression, not old raw Question interface compatibility or a fresh held-out test. No matched comparison against the old byte model on raw phrasing is claimed.",
        "provenance": {
            "weights_sha256": best["weights_sha256"],
            "selection_sha256": sha256(root / "qualification/selection.json"),
            "checker_sha256": sha256(Path(__file__)),
            "service_sha256": sha256(Path(__file__).with_name("service.py")),
            "inference_sources": inference_sources(),
            "old_input_sources": {
                "data.py": sha256(Path(old_data.__file__)),
                "questions.py": sha256(Path(old_questions.__file__)),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    path = Path(args.output)
    if path.exists():
        raise ValueError("不覆寫既有回歸結果")
    result = check(Path(args.experiment))
    write_json(path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "observations"}), flush=True)


if __name__ == "__main__":
    main()
