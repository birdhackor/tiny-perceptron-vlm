"""9.18 reviewer: exact inline CPU run and arithmetic over saved logits only."""

import contextlib
import copy
import hashlib
import io
import json
import math
import platform
import re
import runpy
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(logits, temperature):
    scaled = [z / temperature for z in logits]
    maximum = max(scaled)
    exps = [math.exp(z - maximum) for z in scaled]
    denominator = sum(exps)
    return [p / denominator for p in exps]


def loss(record, temperature):
    z = [v / temperature for v in record["logits"]]
    maximum = max(z)
    gold = record["ids"].index(record["gold"])
    return math.log(sum(math.exp(v - maximum) for v in z)) + maximum - z[gold]


def winner(record):
    index = min(range(len(record["ids"])), key=lambda i: (-record["logits"][i], record["ids"][i]))
    return record["ids"][index]


def fixed_rule(rows, temperature, threshold):
    decisions = []
    for row in rows:
        choice = row["tasks"]["choice"]
        answer = winner(choice)
        info_answer = winner(row["choice_info"])
        maximum = max(probabilities(choice["logits"], temperature))
        accepted = maximum >= threshold and answer != "none" and info_answer == "yes"
        reasons = {
            "choice_wrong": answer != choice["gold"],
            "unsupported": not row["world_supported"],
            "information_insufficient": not row["choice_information_sufficient"],
        }
        error = accepted and any(reasons.values())
        decisions.append({"family": row["family"], "accepted": accepted, "error": error,
                          "answer": answer, "p_max": maximum, "info_answer": info_answer,
                          "error_reasons": reasons})
    n = sum(r["accepted"] for r in decisions)
    e = sum(r["error"] for r in decisions)
    return {"families": len(rows), "accepted": n, "errors": e, "coverage": n / len(rows),
            "empirical_risk": e / n if n else None, "rows": decisions}


def main():
    import torch
    from tiny_perceptron.omni_products.reliability import accepted_rows

    data = json.loads((HERE / "preserved-inputs-v1.json").read_bytes())
    receipts = json.loads((HERE / "inspection-receipts-v1.json").read_bytes())
    page = receipts["snapshot"]
    frozen = (ROOT / page["snapshot"]).read_text()
    assert hashlib.sha256(frozen.encode()).hexdigest() == page["source_sha256"]
    expected = re.search(r"```text\n(.*?)\n```", frozen, re.S).group(1).strip()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        namespace = runpy.run_path(str(HERE / "inline-v1.py"))
    assert output.getvalue().strip() == expected
    inline = {"stdout": output.getvalue(), "exact_expected_output_match": True,
              "T": namespace["T"], "tau": namespace["tau"],
              "nll_unrounded": namespace["nll"](namespace["T"]),
              "policy_grid": [{"tau": t, "values": namespace["evaluate"](namespace["policy"], t)}
                              for t in namespace["taus"]]}
    changed = copy.deepcopy(namespace["final"])
    changed["新措辭"][-1]["info_pred"] = False
    inline["exercise_changed_new"] = namespace["evaluate"](changed["新措辭"], namespace["tau"])
    inline["exercise_changed_all"] = namespace["evaluate"](sum(changed.values(), []), namespace["tau"])
    assert inline["exercise_changed_new"] == (1, 1, 1 / 3, 1.0)
    assert inline["exercise_changed_all"] == (3, 1, 0.5, 1 / 3)
    artificial = {
        "unsupported": namespace["evaluate"]([namespace["row"]([4., 0., -2.], 0, supported_gold=False)], .9),
        "predicted_no": namespace["evaluate"]([namespace["row"]([4., 0., -2.], 0, info_pred=False)], .9),
        "none_winner": namespace["evaluate"]([namespace["row"]([0., 0., 4.], 2)], .9),
        "stable_id_tie": namespace["evaluate"]([namespace["row"]([4., 4., -2.], 1)], .4),
    }
    assert artificial["unsupported"] == (1, 1, 1., 1.)
    assert artificial["predicted_no"] == artificial["none_winner"] == (0, 0, 0., None)
    assert artificial["stable_id_tie"] == (1, 0, 1., 0.)

    # Run the generated local setup and unchanged main cell, without entering Colab.
    notebook = json.loads((ROOT / page["notebook"]).read_bytes())
    assert digest(ROOT / page["notebook"]) == page["notebook_sha256"]
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert cells[1].strip() == (HERE / "inline-v1.py").read_text().strip()
    notebook_output = io.StringIO()
    notebook_namespace = {"__name__": "__review_notebook__"}
    with contextlib.redirect_stdout(notebook_output):
        for cell in cells:
            exec(compile(cell, page["notebook"], "exec"), notebook_namespace)
    assert notebook_namespace["in_colab"] is False
    assert notebook_output.getvalue() == output.getvalue()

    temperature = data["fixed_temperature"]
    threshold = data["fixed_threshold"]
    # Audit the recorded pre-final grids. This does not select a new policy or change a file.
    calibration_audit = []
    for recorded in data["recorded_temperature_grid"]:
        t = recorded["temperature"]
        mean = sum(sum(loss(r["tasks"][k], t) for k in ("choice", "info", "rubric")) / 3
                   for r in data["calibration"]) / len(data["calibration"])
        assert abs(mean - recorded["family_equal_type_equal_nll"]) < 1e-12
        calibration_audit.append({"temperature": t, "mean_nll": mean})
    recorded_minimum = min(calibration_audit, key=lambda r: (r["mean_nll"], abs(r["temperature"] - 1), r["temperature"]))
    assert recorded_minimum["temperature"] == temperature
    policy_audit = []
    for recorded in data["recorded_policy_grid"]:
        scored = fixed_rule(data["policy"], temperature, recorded["threshold"])
        for key in ("families", "accepted", "errors", "coverage", "empirical_risk"):
            assert scored[key] == recorded[key]
        policy_audit.append({k: v for k, v in recorded.items()})
    feasible = [r for r in policy_audit if r["accepted"] and r["empirical_risk"] <= .05]
    assert max(feasible, key=lambda r: (r["coverage"], r["threshold"]))["threshold"] == threshold

    splits = {}
    for split in ("calibration", "policy", "final"):
        rows = data[split]
        ids = {r["family"] for r in rows}
        assert len(ids) == len(rows)
        assert ids == set(data["expected_family_ids"][split])
        splits[split] = ids
    assert not (splits["calibration"] & splits["policy"] or splits["calibration"] & splits["final"]
                or splits["policy"] & splits["final"])
    final = data["final"]
    raw_nll = sum(loss(r["tasks"]["choice"], 1.) for r in final) / len(final)
    calibrated_nll = sum(loss(r["tasks"]["choice"], temperature) for r in final) / len(final)
    correct = sum(winner(r["tasks"]["choice"]) == r["tasks"]["choice"]["gold"] for r in final)
    for row in final:
        choice = row["tasks"]["choice"]
        p = probabilities(choice["logits"], temperature)
        index = min(range(len(p)), key=lambda i: (-p[i], choice["ids"][i]))
        assert choice["ids"][index] == winner(choice) == choice["answer"]
    scored = fixed_rule(final, temperature, threshold)
    official_local = accepted_rows(final, temperature, threshold)
    for key in ("families", "accepted", "errors", "coverage", "empirical_risk"):
        assert scored[key] == official_local[key] == data["recorded_final_metrics"]["policy"][key]
    for independently, project in zip(scored["rows"], official_local["rows"]):
        for key in ("family", "accepted", "error", "answer"):
            assert independently[key] == project[key]
        assert abs(independently["p_max"] - project["p_max"]) < 1e-14
    for label, value in (("raw", raw_nll), ("calibrated", calibrated_nll)):
        assert abs(value - data["recorded_final_metrics"]["choice"][label]["NLL"]) < 1e-12
    assert correct == data["recorded_final_metrics"]["choice"]["primary"]["correct"] == 39
    assert (scored["accepted"], scored["errors"]) == (53, 19)
    assert round(raw_nll, 4) == 2.6901 and round(calibrated_nll, 4) == 3.9624
    assert round(scored["coverage"] * 100, 1) == 88.3
    assert round(scored["empirical_risk"] * 100, 1) == 35.8
    # These two failures already prevent C promotion; no other module or model is rerun.
    rules = data["fixed_metric_rules"]
    assert correct < rules["choice_minimum_correct"]
    assert scored["empirical_risk"] > rules["maximum_empirical_accepted_risk"]
    assert data["promotion_projection"]["promote_C_service"] is False
    by_source = {}
    for source in sorted({r["source"] for r in final}):
        panel = [r for r in final if r["source"] == source]
        by_source[source] = {k: v for k, v in fixed_rule(panel, temperature, threshold).items() if k != "rows"}
    causes = {name: sum(r["accepted"] and r["error_reasons"][name] for r in scored["rows"])
              for name in ("choice_wrong", "unsupported", "information_insufficient")}

    z, t0 = [1., 0., -1.], .5
    served_z = [v / t0 for v in z]
    restored = [v * t0 for v in served_z]
    assert restored == z
    result = {
        "reviewer_task": "/root/omni_p5_technical_9_18",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU"},
        "scope": "Exact synthetic inline/Notebook local execution plus saved logits arithmetic. No model loaded, trained or inferred; no new T/tau chosen or written.",
        "portable_input_sha256": digest(HERE / "preserved-inputs-v1.json"),
        "inline": inline,
        "artificial_rule_checks": artificial,
        "notebook_local_setup_and_main": {"cells": len(cells), "in_colab": False, "stdout_equal_to_inline": True},
        "pre_final_audit": {"T": temperature, "tau": threshold, "calibration_families": 80,
                            "calibration_grid_all33_agree_atol1e12": True, "recorded_minimum": recorded_minimum,
                            "policy_families": 60, "policy_grid": policy_audit,
                            "split_ids_match_frozen_definition_and_are_disjoint": True},
        "final": {"families": len(final), "correct_before_and_after": correct,
                  "raw_mean_NLL": raw_nll, "calibrated_mean_NLL": calibrated_nll,
                  "fixed_policy": {k: v for k, v in scored.items() if k != "rows"},
                  "accepted_error_causes_may_overlap": causes,
                  "by_authored_surface_template": by_source,
                  "candidate_count_histogram": dict(Counter(len(r["tasks"]["choice"]["ids"]) for r in final)),
                  "project_accepted_rows_agrees": True,
                  "non_promotion_supported_by_choice_and_policy_failures": True},
        "temperature_restore_artificial_check": {"raw": z, "T0": t0, "served": served_z, "restored": restored},
        "all_assertions_passed": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
