"""9.18 reviewer-owned CPU verification of the exact frozen inline program.

Third-party originals stay in /workspace/work/reference; this file stores only
the review procedure. It does not train or run a language model.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix08.json"
ORIGINALS = Path("/workspace/work/reference/omni-technical-9-18-v1")
METRICS_SHA = "167a586334cfcced4adcc197758535c3bcda832b855c151095ef2d1968cf8983"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_inline(code):
    out = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(out):
        exec(compile(code, "frozen-9.18-inline.py", "exec"), namespace)
    return namespace, out.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    freeze = json.loads((ROOT / FREEZE).read_bytes())
    pages = {page["page_id"]: page for page in freeze["pages"]}
    page = pages["9.18"]
    snapshot = ROOT / page["snapshot"]
    assert sha(snapshot) == page["source_sha256"]
    source = snapshot.read_text()
    assert page["figures_sha256"] == {}
    code_blocks = re.findall(r"```python\n(.*?)\n```", source, re.S)
    assert len(code_blocks) == 1
    code = code_blocks[0]
    expected = re.findall(r"```text\n(.*?)\n```", source, re.S)[0]
    current = (ROOT / page["source"]).read_text()
    section = re.search(r"^## 9\.18 .*?(?=^## |\Z)", current, re.M | re.S)[0]
    assert section == source
    notebook = ROOT / page["notebook"]
    assert sha(notebook) == page["notebook_sha256"]
    cells = json.loads(notebook.read_text())["cells"]
    assert sum("".join(c["source"]).strip() == code.strip() for c in cells if c["cell_type"] == "code") == 1
    prerequisites = {}
    for key in ("9.9", "9.10", "9.13", "9.14", "9.15", "9.16"):
        entry = pages[key]
        assert sha(ROOT / entry["snapshot"]) == entry["source_sha256"]
        prerequisites[key] = {"snapshot": entry["snapshot"], "sha256": entry["source_sha256"]}

    ns, stdout = run_inline(code)
    assert stdout.strip() == expected
    assert ns["T"] == 1.0 and ns["tau"] == 0.9
    assert len(ns["calibration"]) == 5 and len(ns["temperatures"]) == 33
    assert ns["temperatures"][0] == 0.25 and ns["temperatures"][-1] == 4.0
    # Closed forms are independent of the inline softmax implementation.
    closed_forms = {
        "T1_X": math.e / (math.e + 2),
        "T1_Y": math.exp(0.5) / (math.exp(0.5) + 1 + math.exp(-10)),
        "T0.1_X": math.exp(10) / (math.exp(10) + 2),
        "T0.1_Y": math.exp(5) / (math.exp(5) + 1 + math.exp(-100)),
    }
    for t, name in ((1.0, "T1"), (0.1, "T0.1")):
        assert math.isclose(max(ns["softmax"]([1.0, 0.0, 0.0], t)), closed_forms[name + "_X"], abs_tol=1e-14)
        assert math.isclose(max(ns["softmax"]([0.5, 0.0, -10.0], t)), closed_forms[name + "_Y"], abs_tol=1e-14)
    expected_nll = -0.6 * math.log(0.6) - 0.4 * math.log(0.2)
    assert math.isclose(ns["nll"](1), expected_nll, abs_tol=1e-14)
    evaluate = ns["evaluate"]
    original_results = {
        "policy": evaluate(ns["policy"], ns["tau"]),
        **{name: evaluate(rows, ns["tau"]) for name, rows in ns["final"].items()},
        "全部final": evaluate(sum(ns["final"].values(), []), ns["tau"]),
        "零接受": evaluate(ns["policy"], 1.0),
    }
    assert original_results == {
        "policy": (3, 0, 0.5, 0.0), "原措辭": (2, 0, 2 / 3, 0.0),
        "新措辭": (2, 2, 2 / 3, 1.0), "全部final": (4, 2, 2 / 3, 0.5),
        "零接受": (0, 0, 0.0, None),
    }
    threshold_audit = [{"tau": tau, "result": evaluate(ns["policy"], tau)} for tau in ns["taus"]]
    old = "row([4., 0., -2.], 0, info_pred=True, info_gold=False)"
    assert code.count(old) == 1
    exercise_code = code.replace(old, "row([4., 0., -2.], 0, info_pred=False, info_gold=False)")
    exercise, exercise_stdout = run_inline(exercise_code)
    exercise_results = {
        "新措辭": exercise["evaluate"](exercise["final"]["新措辭"], exercise["tau"]),
        "全部final": exercise["evaluate"](sum(exercise["final"].values(), []), exercise["tau"]),
    }
    assert exercise["T"] == 1.0 and exercise["tau"] == 0.9
    assert exercise_results == {"新措辭": (1, 1, 1 / 3, 1.0), "全部final": (3, 1, 0.5, 1 / 3)}
    row = ns["row"]
    branches = {
        "supported_gold_false": evaluate([row([4, 0, -2], 0, supported_gold=False)], 0.9),
        "info_gold_false_prediction_yes": evaluate([row([4, 0, -2], 0, info_gold=False)], 0.9),
        "info_prediction_no": evaluate([row([4, 0, -2], 0, info_pred=False)], 0.9),
        "none_prediction": evaluate([row([0, 0, 4], 2)], 0.9),
    }
    assert branches["supported_gold_false"] == branches["info_gold_false_prediction_yes"] == (1, 1, 1.0, 1.0)
    assert branches["info_prediction_no"] == branches["none_prediction"] == (0, 0, 0.0, None)
    policy_code = code[code.index("taus ="):code.index('print("固定tau"')]
    rejected = dict(ns, policy=[row([0, 0, 4], 2)])
    try:
        exec(compile(policy_code, "frozen-9.18-threshold-code.py", "exec"), rejected)
    except RuntimeError as error:
        no_feasible = str(error)
    else:
        raise AssertionError("no-feasible threshold branch failed to stop")

    metrics_path = ORIGINALS / "kev-metrics-2074cc4.py"
    assert sha(metrics_path) == METRICS_SHA
    spec = importlib.util.spec_from_file_location("pinned_kev_metrics", metrics_path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    import numpy as np

    p_scaled = ns["softmax"]([4, 0, -2], 2.0)
    served = {"p": p_scaled, "logits": [2.0, 0.0, -1.0], "inference_temperature": 2.0}
    restored = metrics.raw_row(served)
    assert restored["logits"] == [4.0, 0.0, -2.0]
    assert np.allclose(restored["p"], ns["softmax"]([4, 0, -2], 1.0), atol=1e-14, rtol=0)
    try:
        metrics.raw_row({"p": p_scaled, "inference_temperature": 2.0})
    except ValueError as error:
        missing_logits = str(error)
    else:
        raise AssertionError("raw_row accepted absent logits")
    calibration_rows = [
        {"p": ns["softmax"](z), "logits": z, "label": label, "type": "choice", "task": "choice",
         "variant": "clean", "source": "reviewer-toy", "inference_temperature": 1.0}
        for z, label in ns["calibration"]
    ]
    official_fit = metrics.fit_temperature(calibration_rows, aggregation="micro", points=33)
    assert math.isclose(official_fit, 1.0, abs_tol=1e-14)
    try:
        metrics.fit_temperature([dict(calibration_rows[0], inference_temperature=2.0)])
    except ValueError as error:
        repeated_scaling = str(error)
    else:
        raise AssertionError("fit_temperature accepted already-scaled rows")
    confidence, correct = [0.99, 0.9, 0.8], [True, False, True]
    official_curve = metrics.risk_coverage_curve(confidence, correct)
    official_tau = metrics.select_threshold(confidence, correct, 0.05)
    assert official_tau == 0.99
    official_zero = metrics.evaluate_threshold(confidence, correct, None)
    assert official_zero["risk"] is None and official_zero["coverage"] == 0
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": "/root/omni_technical_coordinator/technical_9_18",
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-18-v1/verification.py --output docs/technical-reviews/artifacts/omni-technical-9-18-v1/execution.json",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "device": "CPU", "platform": platform.platform(), "torch": "not used by the lesson or this probe"},
        "source_binding": {"freeze": FREEZE, "freeze_sha256": sha(ROOT / FREEZE), "snapshot": page["snapshot"], "source_sha256": sha(snapshot), "python_fence_sha256": hashlib.sha256(code.encode()).hexdigest(), "notebook": page["notebook"], "notebook_sha256": sha(notebook), "notebook_main_cell_matches": True, "live_section_matches": True, "figure_sha256": {}},
        "prerequisite_bindings": prerequisites,
        "lesson_stdout": stdout, "lesson_stdout_equals_frozen_expected": True,
        "closed_form_values": closed_forms, "closed_form_nll_at_T1": expected_nll,
        "threshold_audit": threshold_audit, "original_results": original_results,
        "exercise_change": old + " -> info_pred=False only", "exercise_results": exercise_results,
        "exercise_changed_output_lines": [line for line in exercise_stdout.splitlines() if line.startswith(("新措辭", "全部final"))],
        "branch_probes": branches, "no_feasible_threshold_exception": no_feasible,
        "official_metrics": {"pin": "2074cc44256200bb37540103fa4572901faa10f3", "sha256": METRICS_SHA, "restored_logits": restored["logits"], "missing_logits_exception": missing_logits, "fit_33_point_micro": official_fit, "repeated_scaling_exception": repeated_scaling, "diagnostic_curve": official_curve, "selected_policy_threshold": official_tau, "zero_acceptance": official_zero},
        "result": "All assertions passed; exact inline output, exercise, split bindings and pinned-source software probes verified.",
        "scope": "Deterministic artificial arithmetic on CPU only. No language-model inference, training, real quality, interval coverage or distribution-shift experiment. Notebook main cell was compared, not remote Colab setup executed.",
    }
    output = ROOT / args.output
    with output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"result": result["result"], "output": args.output, "sha256": sha(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
