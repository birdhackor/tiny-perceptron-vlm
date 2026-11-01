"""只用 calibration/policy 選參數；新 final 必須先建立不可改寫的開封收據。"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .data import KINDS, evaluation_examples, make_families
from .experiment import write_json
from .model import Tokenizer
from .runtime import candidate_eligibility, collate_for_model, create_model, encode_for_model

TEMPERATURES = tuple(math.exp(math.log(0.25) + i * math.log(16) / 32) for i in range(33))
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inference_sources():
    return {
        name: sha256(Path(__file__).with_name(name))
        for name in ("data.py", "model.py", "record_model.py", "indexed_model.py", "runtime.py", "service.py", "cli.py")
    }


def upper95(errors, n):
    if n == 0:
        return 1.0
    z, p = 1.6448536269514722, errors / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def load_experiment(root):
    verify_training_receipts(root)
    best = json.loads((root / "best.json").read_text())
    if sha256(root / "best.safetensors") != best["weights_sha256"]:
        raise ValueError("權重收據不符")
    tokenizer = Tokenizer(json.loads((root / "tokenizer.json").read_text()))
    model = create_model(best["config"])
    model.load_state_dict(load_file(root / "best.safetensors"))
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("權重含非有限值")
    model.eval().requires_grad_(False)
    return model, tokenizer, best


def verify_training_receipts(root, require_complete=False):
    freeze = json.loads((root / "source-freeze.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("training_driver") == "hardening":
        names = ("data.py", "model.py", "record_model.py", "indexed_model.py", "hardening_experiment.py")
    elif plan.get("architecture") == "indexed-record-state-memory-v1":
        names = ("data.py", "model.py", "record_model.py", "indexed_model.py", "indexed_experiment.py")
    elif plan.get("architecture") == "record-local-state-set-v1":
        names = ("data.py", "model.py", "record_model.py", "record_experiment.py")
    else:
        names = ("data.py", "model.py", "experiment.py")
    current = {name: sha256(Path(__file__).with_name(name)) for name in names}
    if freeze["sources"] != current or plan["source_receipts"] != freeze["sources"]:
        raise ValueError("目前實作與實際訓練來源不符")
    for name, digest in freeze["sources"].items():
        if sha256(root / "source-snapshot" / name) != digest:
            raise ValueError("訓練來源快照不符")
    for name, digest in freeze["artifacts"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"訓練輸入收據不符：{name}")
    if require_complete:
        completion = json.loads((root / "completion.json").read_text())
        if (
            completion["completed_updates"] != plan["steps"]
            or completion["process_exit_code"] != 0
            or completion["source_freeze_sha256"] != sha256(root / "source-freeze.json")
        ):
            raise ValueError("訓練尚未完整完成或完成收據不符")
        for name, digest in completion["artifacts"].items():
            if sha256(root / name) != digest:
                raise ValueError("完成後訓練輸出發生變更")
    return current


def register_exposure(round_root, family_keys, kind, source_digest):
    registry_path = round_root.parent / "decision-product-exposed-families.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock = registry_path.with_suffix(".lock")
    with lock.open("x") as handle:
        handle.write(kind + "\n")
    try:
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"entries": []}
        identity = (kind, source_digest, sorted(set(family_keys)))
        if any(
            (entry["kind"], entry.get("source_sha256"), sorted(entry["family_keys"])) == identity
            for entry in registry["entries"]
        ):
            return
        registry["entries"].append(
            {
                "kind": kind,
                "created_at": datetime.now(UTC).isoformat(),
                "source_sha256": source_digest,
                "family_keys": identity[2],
            }
        )
        pending = registry_path.with_suffix(".pending.json")
        write_json(pending, registry)
        pending.replace(registry_path)
    finally:
        lock.unlink()


def active_contract(round_root):
    pointer = json.loads((round_root / "round-contract-active.json").read_text())
    if Path(pointer["file"]).name != pointer["file"]:
        raise ValueError("contract 路徑須為同目錄檔案")
    path = round_root / pointer["file"]
    if sha256(path) != pointer["sha256"]:
        raise ValueError("contract 指紋不符")
    return json.loads(path.read_text()), path


def quality_gates(metrics, contract):
    limits = contract["fresh_qualification"]
    gates = {f"{kind}_accuracy": metrics[kind]["accuracy"] >= limits[f"{kind}_accuracy_min"] for kind in KINDS}
    gates["source_kind_accuracy"] = all(
        value["accuracy"] >= limits["source_kind_accuracy_min"] for key, value in metrics.items() if "/source/" in key
    )
    gates["class_accuracy"] = all(
        value["accuracy"] >= limits["class_accuracy_min"]
        for key, value in metrics.items()
        if "/gold/" in key or "/action/" in key
    )
    gates["choice_panel_accuracy"] = all(
        value["accuracy"] >= limits["choice_panel_accuracy_min"]
        for key, value in metrics.items()
        if key.startswith("choice/panel/")
    )
    if "selective_choice" in metrics:
        selective = metrics["selective_choice"]
        gates.update(
            {
                "accepted_risk": selective["risk"] <= limits["choice_accepted_risk_max"],
                "risk_upper95": selective["risk_one_sided_wilson95_upper"]
                <= limits["choice_accepted_risk_one_sided_wilson95_upper_max"],
                "coverage": selective["coverage"] >= limits["choice_overall_accepted_coverage_min"],
            }
        )
    return gates


def reserve_final_families(root, opening):
    """同 round 只選一個模型；跨 round 排除所有已暴露及結構測試 family。"""
    round_root = root.parent
    registry_path = round_root.parent / "decision-product-exposed-families.json"
    lock = registry_path.with_suffix(".lock")
    with lock.open("x") as handle:
        handle.write(str(root.resolve()) + "\n")
    try:
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"entries": []}
        excluded = {key for entry in registry["entries"] for key in entry["family_keys"]}
        opening["previous_registry_sha256"] = sha256(registry_path) if registry_path.exists() else None
        opening["selected_experiment"] = root.name
        opening["previous_exposed_family_count"] = len(excluded)
        round_opening = round_root / "round-final-opening.json"
        # 先寫 round 層級收據，任何子實驗都不能重新開同一 round。
        with round_opening.open("x") as handle:
            handle.write(json.dumps(opening, ensure_ascii=False, indent=2) + "\n")
        previous = root / "qualification/registry-before-final.json"
        with previous.open("x") as handle:
            handle.write(registry_path.read_text() if registry_path.exists() else json.dumps(registry, indent=2) + "\n")
        local = root / "qualification/final-opening.json"
        with local.open("x") as handle:
            handle.write(round_opening.read_text())
        protocol = opening["protocol"]
        generated = make_families("final", protocol["families"] + len(excluded), protocol["family_seed"])
        families = [family for family in generated if family["key"] not in excluded][: protocol["families"]]
        if len(families) != protocol["families"]:
            raise ValueError("排除歷次暴露後 final 數量不足")
        registry["entries"].append(
            {
                "kind": "formal_final_opened",
                "created_at": datetime.now(UTC).isoformat(),
                "round": round_root.name,
                "experiment": root.name,
                "opening_sha256": sha256(round_opening),
                "family_keys": [family["key"] for family in families],
            }
        )
        pending = registry_path.with_suffix(".pending.json")
        write_json(pending, registry)
        pending.replace(registry_path)
        return families
    finally:
        lock.unlink()


@torch.inference_mode()
def observations(model, tokenizer, examples, batch_size=32):
    rows = [encode_for_model(example, tokenizer, model) for example in examples]
    result = []
    for start in range(0, len(rows), batch_size):
        logits = model(collate_for_model(rows[start : start + batch_size], model))
        for example, row, scores in zip(examples[start : start + batch_size], rows[start : start + batch_size], logits):
            count = len(example.candidates)
            valid = scores[:count] if example.kind == "info" else scores[: count + 1]
            if not bool(torch.isfinite(valid).all()) or (
                example.kind == "info" and not bool(torch.isneginf(scores[count]))
            ):
                raise RuntimeError("驗收神經輸出含非有限值，不得宣告品質通過")
            action = (
                "none"
                if example.gold == "none"
                else example.candidates[row["label"]][1].rsplit(":", 1)[-1].rsplit("：", 1)[-1].strip()
            )
            result.append(
                {
                    "family": example.family,
                    "kind": example.kind,
                    "source": example.source,
                    "panel": example.panel,
                    "ids": [key for key, _ in example.candidates] + ["none"],
                    "label": row["label"],
                    "logits": [float(value) if math.isfinite(value) else None for value in scores[: count + 1]],
                    "answerable": example.gold != "none",
                    "gold_id": example.gold,
                    "gold_action": action if example.kind == "choice" else None,
                    "eligible": candidate_eligibility(example),
                }
            )
    return result


def scores_of(row):
    result = torch.tensor([value if value is not None else -math.inf for value in row["logits"]])
    if bool(torch.isnan(result).any()) or bool(torch.isposinf(result).any()) or not bool(torch.isfinite(result).any()):
        raise ValueError("觀測資料的 logits 不合法")
    return result


def nll(rows, temperature):
    return sum(
        float(F.cross_entropy(scores_of(row)[None] / temperature, torch.tensor([row["label"]]))) for row in rows
    ) / len(rows)


def decision(row, temperature, threshold):
    probability = (scores_of(row) / temperature).softmax(-1)
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("觀測資料的機率非有限值")
    index = int(probability.argmax())
    confidence = float(probability[index])
    accepted = row["ids"][index] != "none" and confidence >= threshold and row["eligible"][index]
    return index, confidence, accepted


def selective_metrics(rows, temperature, threshold):
    accepted, errors = 0, 0
    for row in rows:
        index, _, accept = decision(row, temperature, threshold)
        accepted += accept
        errors += accept and index != row["label"]
    return {
        "families": len(rows),
        "accepted": accepted,
        "errors": errors,
        "coverage": accepted / len(rows),
        "risk": errors / accepted if accepted else 1.0,
        "risk_one_sided_wilson95_upper": upper95(errors, accepted),
    }


def summary(rows, temperatures, threshold):
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        correct = int(scores_of(row).argmax()) == row["label"]
        keys = [
            row["kind"],
            f"{row['kind']}/source/{row['source']}",
            f"{row['kind']}/panel/{row['panel']}",
            f"{row['kind']}/gold/{'none' if row['gold_id'] == 'none' else 'answer' if row['kind'] == 'choice' else row['gold_id']}",
        ]
        if row["kind"] == "choice":
            keys.append(f"choice/action/{row['gold_action']}")
        for key in keys:
            groups[key][0] += correct
            groups[key][1] += 1
    metrics = {key: {"correct": c, "families": n, "accuracy": c / n} for key, (c, n) in sorted(groups.items())}
    choice = [row for row in rows if row["kind"] == "choice"]
    metrics["selective_choice"] = selective_metrics(choice, temperatures["choice"], threshold)
    return metrics


def calibrate(root):
    verify_training_receipts(root, require_complete=True)
    contract, _ = active_contract(root.parent)
    best_metrics = json.loads((root / "best.json").read_text())["metrics"]
    if not all(quality_gates(best_metrics, contract).values()):
        raise ValueError("開發門檻未達，不使用 calibration/policy 選參數")
    destination = root / "qualification"
    destination.mkdir(exist_ok=True)
    if (destination / "selection.json").exists():
        raise ValueError("本次參數已選定；不能覆寫選擇收據")
    model, tokenizer, best = load_experiment(root)
    temperatures, metrics = {}, {}
    calibration = make_families("calibration", 600, 826027)
    policy = make_families("policy", 600, 826029)
    rows = observations(model, tokenizer, evaluation_examples(calibration, 826031))
    write_json(destination / "calibration-observations.json", rows)
    for kind in KINDS:
        group = [row for row in rows if row["kind"] == kind]
        search = [{"temperature": temperature, "nll": nll(group, temperature)} for temperature in TEMPERATURES]
        chosen = min(search, key=lambda entry: (entry["nll"], abs(math.log(entry["temperature"]))))
        temperatures[kind] = chosen["temperature"]
        metrics[kind] = {"raw_nll": nll(group, 1.0), "calibrated_nll": chosen["nll"], "search": search}
    rows = observations(model, tokenizer, evaluation_examples(policy, 826033))
    write_json(destination / "policy-observations.json", rows)
    choice = [row for row in rows if row["kind"] == "choice"]
    search = [
        {"threshold": threshold, **selective_metrics(choice, temperatures["choice"], threshold)}
        for threshold in THRESHOLDS
    ]
    limits = contract["fresh_qualification"]
    feasible = [
        entry
        for entry in search
        if entry["coverage"] >= limits["choice_overall_accepted_coverage_min"]
        and entry["risk"] <= limits["choice_accepted_risk_max"]
        and entry["risk_one_sided_wilson95_upper"] <= limits["choice_accepted_risk_one_sided_wilson95_upper_max"]
    ]
    selected = (
        max(feasible, key=lambda entry: (entry["accepted"], -entry["errors"], entry["threshold"])) if feasible else None
    )
    write_json(
        destination / "selection.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "kind": "calibration-policy-selection-only-no-final",
            "weights_sha256": best["weights_sha256"],
            "tokenizer_sha256": sha256(root / "tokenizer.json"),
            "sources": verify_training_receipts(root),
            "inference_sources": inference_sources(),
            "qualification_sha256": sha256(Path(__file__)),
            "source_freeze_sha256": sha256(root / "source-freeze.json"),
            "completion_sha256": sha256(root / "completion.json"),
            "temperatures": temperatures,
            "calibration": metrics,
            "policy_search": search,
            "selected": selected,
            "policy_metrics": summary(rows, temperatures, selected["threshold"] if selected else 1.0),
            "families": {
                "calibration": [family["key"] for family in calibration],
                "policy": [family["key"] for family in policy],
            },
            "statistical_unit": "600 families per kind; three related observations are never pooled as 1800 independent trials",
        },
    )
    print(json.dumps({"event": "policy_selected", "selected": selected, "temperatures": temperatures}), flush=True)


def final(root):
    verify_training_receipts(root, require_complete=True)
    destination = root / "qualification"
    selection_path = destination / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["selected"] is None:
        raise ValueError("policy 未達門檻，不開封 final")
    model, tokenizer, best = load_experiment(root)
    if (
        selection["weights_sha256"] != best["weights_sha256"]
        or selection["tokenizer_sha256"] != sha256(root / "tokenizer.json")
        or selection["sources"] != verify_training_receipts(root)
        or selection["qualification_sha256"] != sha256(Path(__file__))
        or selection["source_freeze_sha256"] != sha256(root / "source-freeze.json")
        or selection["completion_sha256"] != sha256(root / "completion.json")
        or selection["inference_sources"] != inference_sources()
    ):
        raise ValueError("選參數後來源／權重／tokenizer 發生變更")
    contract, contract_path = active_contract(root.parent)
    if not all(quality_gates(best["metrics"], contract).values()) or not all(
        quality_gates(selection["policy_metrics"], contract).values()
    ):
        raise ValueError("dev 或 policy 未達完整品質門檻，不開封 final")
    checks_path = destination / "product-checks.json"
    checks = json.loads(checks_path.read_text())
    provenance = checks["provenance"]
    expected = {
        "weights_sha256": best["weights_sha256"],
        "tokenizer_sha256": sha256(root / "tokenizer.json"),
        "checker_sha256": sha256(Path(__file__).with_name("product_checks.py")),
        "service_sha256": sha256(Path(__file__).with_name("service.py")),
        "selection_sha256": sha256(selection_path),
        "threshold": selection["selected"]["threshold"],
        "temperatures": selection["temperatures"],
        "inference_sources": inference_sources(),
    }
    if not checks["pass"] or not all(checks["gates"].values()) or provenance != expected:
        raise ValueError("實際服務驗收未通過或來源／採用政策不符")
    legacy_path = destination / "legacy-regression.json"
    legacy = json.loads(legacy_path.read_text())
    legacy_expected = {
        "weights_sha256": best["weights_sha256"],
        "selection_sha256": sha256(selection_path),
        "checker_sha256": sha256(Path(__file__).with_name("legacy_regression.py")),
        "service_sha256": sha256(Path(__file__).with_name("service.py")),
        "inference_sources": inference_sources(),
        "old_input_sources": {
            name: sha256(Path(__file__).parents[1] / "omni_products" / name) for name in ("data.py", "questions.py")
        },
    }
    if not legacy["pass"] or not all(legacy["gates"].values()) or legacy["provenance"] != legacy_expected:
        raise ValueError("舊三裝置情境的語意回歸未通過或版本不符")
    opening = {
        "created_at": datetime.now(UTC).isoformat(),
        "weights_sha256": best["weights_sha256"],
        "selection_sha256": sha256(selection_path),
        "contract_sha256": sha256(contract_path),
        "active_contract_sha256": sha256(root.parent / "round-contract-active.json"),
        "product_checks_sha256": sha256(checks_path),
        "legacy_regression_sha256": sha256(legacy_path),
        "implementation": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "protocol": {"families": 1500, "family_seed": 826035, "rendering_seed": 826037},
        "thresholds": contract["fresh_qualification"],
        "statistical_unit": "one independent visible-fact family per kind; no pooling kinds",
    }
    families = reserve_final_families(root, opening)
    write_json(destination / "final-families.json", families)
    known = {family["key"] for family in json.loads((root / "train-families.json").read_text())}
    known.update(family["key"] for family in make_families("dev", 600, 826021))
    known.update(family["key"] for family in json.loads((root / "curriculum-families.json").read_text()))
    known.update(selection["families"]["calibration"])
    known.update(selection["families"]["policy"])
    if known.intersection(family["key"] for family in families):
        raise ValueError("跨 split family 洩漏")
    examples = evaluation_examples(families, 826037)
    rows = observations(model, tokenizer, examples)
    write_json(destination / "final-observations.json", rows)
    write_json(destination / "final-examples.json", [example.to_dict() for example in examples])
    metrics = summary(rows, selection["temperatures"], selection["selected"]["threshold"])
    gates = quality_gates(metrics, contract)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "schema": "c-product-v3-final-v1",
        "measurement_completed": True,
        "primary_gates_pass": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "final_opening_sha256": sha256(destination / "final-opening.json"),
        "observations_sha256": sha256(destination / "final-observations.json"),
        "examples_sha256": sha256(destination / "final-examples.json"),
        "families_sha256": sha256(destination / "final-families.json"),
        "scope": contract["scope"],
        "further_qualification_required": "Input/schema/cache/branch-permutation/counterfactual/manual product checks. This report alone does not declare release-ready.",
    }
    write_json(destination / "final-results.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "final"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    (calibrate if args.phase == "calibrate" else final)(Path(args.experiment))


if __name__ == "__main__":
    main()
