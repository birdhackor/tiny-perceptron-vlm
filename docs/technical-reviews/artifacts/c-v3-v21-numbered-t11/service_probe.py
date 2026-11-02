"""Small fresh typed-API probes and isolated adoption-guard software probes."""

import copy
import json
import pathlib
import platform
import sys
from datetime import UTC, datetime

import torch

from scripts.course_product import dispatch_request
from tiny_perceptron.decision_product.cli import predict_request
from tiny_perceptron.decision_product.data import Example
from tiny_perceptron.decision_product.runtime import candidate_eligibility, encode_for_model
from tiny_perceptron.decision_product.service import Action, DecisionEngine, Device

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    engine = DecisionEngine(ROOT / "checkpoints/decision-product-v3")
    original = json.loads((ROOT / "examples/decision_product/record-ready.json").read_text())
    records, forwards = [], []
    hook = engine._model.register_forward_hook(lambda module, args, output: forwards.append(list(output.shape)))
    for name, mutate, expected in [
        ("record-ready", lambda value: None, ("record", "yes", "2")),
        ("play-ready", lambda value: value["devices"][0].update(task="play"), ("play", "yes", "2")),
        ("permission-false", lambda value: value["devices"][0].update(permission=False), ("stop", "yes", "1")),
        ("readiness-unknown", lambda value: value["devices"][0].update(ready=None), ("none", "no", "none")),
        ("answer-removed", lambda value: value.update(candidates=[{"id": "only-play", "device": "主錄音室", "action": "play"}]), ("none", "yes", "2")),
    ]:
        request = copy.deepcopy(original)
        mutate(request)
        result = predict_request(engine, request)
        observed = tuple(result["results"][kind]["model_choice"] for kind in ("choice", "info", "rubric"))
        assert observed == expected
        records.append({"name": name, "request": request, "expected": list(expected), "result": result})
    device_count_probes = []
    for count in (2, 3, 4, 5, 6, 7):
        devices = [Device(str(index), "record", True, True) for index in range(count)]
        try:
            session = engine.session(devices)
            result = session.choose("0").to_dict()
            assert 3 <= count <= 6
            assert result["model_choice"] == "record"
            device_count_probes.append({"devices": count, "accepted_schema": True, "result": result})
        except ValueError as error:
            assert count in (2, 7)
            device_count_probes.append({"devices": count, "accepted_schema": False, "error": str(error)})
    hook.remove()
    # Controlled logits below test service branch behavior, never model quality.
    original_forward = engine._model.forward
    guard_records = []
    session = engine.session([Device("target", "record", True, True), Device("other", "play", True, True), Device("third", "save", False, False)])
    try:
        engine._model.forward = lambda batch, memory=None: torch.tensor([[20.0, 0.0]])
        result = session.choose("target", [Action("wrong-device", "other", "record")]).to_dict()
        assert result["status"] == "abstained" and result["model_choice"] == "wrong-device"
        guard_records.append({"case": "high confidence wrong device ID", "forced_logits": [20, 0], "result": result})
        result = session.choose("target", [Action("target-match", "target", "record")]).to_dict()
        assert result["status"] == "accepted"
        guard_records.append({"case": "high confidence real same-target candidate", "forced_logits": [20, 0], "result": result})
        engine._model.forward = lambda batch, memory=None: torch.tensor([[0.0, 0.0]])
        result = session.choose("target", [Action("target-match", "target", "record")]).to_dict()
        assert result["status"] == "abstained"
        guard_records.append({"case": "confidence below frozen threshold", "forced_logits": [0, 0], "result": result})
        engine._model.forward = lambda batch, memory=None: torch.tensor([[0.0, 20.0]])
        result = session.choose("target", [Action("target-match", "target", "record")]).to_dict()
        assert result["status"] == "abstained" and result["model_choice"] == "none"
        guard_records.append({"case": "none argmax", "forced_logits": [0, 20], "result": result})
        def sentinel(*args, **kwargs):
            raise RuntimeError("fresh-review-neural-forward-required")
        engine._model.forward = sentinel
        try:
            session.choose("target")
            raise AssertionError("neural forward was bypassed")
        except RuntimeError as error:
            assert str(error) == "fresh-review-neural-forward-required"
            guard_records.append({"case": "neural forward cannot be bypassed", "observed_error": str(error)})
    finally:
        engine._model.forward = original_forward
    eligibility = []
    for state in ("A: task=record; permission=yes; ready=yes. B: task=play; permission=no; ready=no. C: task=save; permission=yes; ready=no.", "A: task=save; permission=unknown; ready=no. B: task=record; permission=yes; ready=yes. C: task=play; permission=no; ready=unknown."):
        example = Example(state, "Choose A's action.", (("arbitrary-a", "Device A: stop"), ("arbitrary-b", "Device B: record")), "", "choice", "", "", "")
        row = encode_for_model(example, engine._tokenizer, engine._model)
        mask = candidate_eligibility(example)
        assert row["target_row"] == 0 and mask == [True, False, False]
        eligibility.append({"state": state, "target_row": row["target_row"], "mask": mask})
    try:
        dispatch_request(engine, {"state": "三數加法;1;2;3"})
        raise AssertionError("typed entry accepted B state")
    except ValueError as error:
        b_error = str(error)
    report = {"created_at": datetime.now(UTC).isoformat(), "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()}, "scope": "Five fresh handwritten example variants and six device-count/schema probes, plus controlled-logit software branches. Not a new held-out final, not independent population-quality evidence.", "model_type": type(engine._model).__name__, "policy": engine._policy, "real_model_forwards": forwards, "real_model_requests": records, "device_count_probes": device_count_probes, "controlled_logit_guard_probes": guard_records, "id_only_invariance": eligibility, "B_state_rejection": b_error}
    (OUT / "service-probes.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"real_requests": len(records), "real_model_forward_calls": len(forwards), "device_count_probe_cases": len(device_count_probes), "adoption_guard_cases": len(guard_records), "typed_B_rejected": bool(b_error)}))


if __name__ == "__main__":
    main()
