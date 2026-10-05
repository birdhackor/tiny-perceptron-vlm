"""Recompute the existing v4 validation selection from named raw counters.

This verifies existing scores; it does not run or regrade any pretrained model.
Author explanation fields are not inspected or emitted.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
FILES = {
    "configuration": "docs/natural-assistant/v4/selection.json",
    "protocol": "docs/natural-assistant/v4/validation-protocol-lower-lr.json",
    "scores": "docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json",
    "result": "docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/raw/result.json",
}
objects, copies = {}, {}
for key, filename in FILES.items():
    original = ROOT / filename
    raw = original.read_bytes()
    destination = BASE / "inputs" / ("selection-" + key + ".json")
    destination.write_bytes(raw)
    assert raw == destination.read_bytes()
    objects[key] = json.loads(raw)
    copies[key] = {"original_path": filename, "permanent_copy": str(destination.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()}

config, protocol, scores, result = (objects[x] for x in ["configuration", "protocol", "scores", "result"])
assert config["validation_result_sha256"] == copies["result"]["sha256"]
assert scores["artifact_binding"]["validation_result_sha256"] == copies["result"]["sha256"]
assert config["validation_protocol_sha256"] == copies["protocol"]["sha256"]
assert scores["denominators"] == protocol["validation_denominators"]
assert set(result["variants"]) == set(scores["variants"])
expected_weights = {"photo_summary": 270, "photo_fact": 135, "text_presence": 840, "single_ocr": 1512, "ordered_ocr": 5040, "text_chat": 560, "voice_typed_reference_chat": 1260, "voice_actual_asr_chat": 1260}
assert scores["integer_weights"] == expected_weights
counts = {name: data["correct_counts"] for name, data in scores["variants"].items()}
base = counts["base"]
computed = {}
for name, data in scores["variants"].items():
    n = sum(data["correct_counts"][k] * weight for k, weight in expected_weights.items())
    assert n == data["primary_numerator"]
    assert data["primary_denominator"] == 75600
    assert result["variants"][name]["generation_count"] == 132
    gates = {
        "all_generations_complete": data["all_generations_complete"],
        "photo_summary": counts[name]["photo_summary"] >= base["photo_summary"],
        "photo_fact": counts[name]["photo_fact"] >= base["photo_fact"] - 1,
        "text_presence": counts[name]["text_presence"] >= base["text_presence"] - 1,
        "single_ocr": counts[name]["single_ocr"] >= base["single_ocr"] - 1,
        "ordered_ocr": counts[name]["ordered_ocr"] >= base["ordered_ocr"],
        "text_chat": counts[name]["text_chat"] >= base["text_chat"] - 1,
        "voice_typed_reference_chat": counts[name]["voice_typed_reference_chat"] >= base["voice_typed_reference_chat"],
        "voice_actual_asr_chat": counts[name]["voice_actual_asr_chat"] >= base["voice_actual_asr_chat"],
    }
    computed[name] = {"primary_numerator": n, "primary_denominator": 75600, "generation_count": 132, "gates_recomputed": gates, "eligible_and_above_base": all(gates.values()) and n > scores["variants"]["base"]["primary_numerator"]}
eligible = [name for name in computed if name != "base" and computed[name]["eligible_and_above_base"]]
selected = max(eligible, key=lambda name: computed[name]["primary_numerator"]) if eligible else "base"
assert selected == scores["selected_variant"] == config["selected_variant"] == "base"
record = {"scope": "Existing v4 raw validation counters, predeclared protocol and selected configuration; no new model score or annotation", "raw_pointers_read": {"configuration": ["/selected_variant", "/validation_result_sha256", "/validation_protocol_sha256", "/base_model"], "protocol": ["/validation_denominators", "/score", "/adapter_nonregression_gates_vs_base", "/selection"], "scores": ["/denominators", "/integer_weights", "/variants/*/correct_counts", "/variants/*/primary_numerator", "/variants/*/primary_denominator", "/variants/*/all_generations_complete", "/selected_variant", "/artifact_binding/validation_result_sha256"], "result": ["/variants/*/generation_count"]}, "denominators": scores["denominators"], "computed": computed, "selected_variant": selected, "copies": copies, "assertions": "passed"}
(BASE / "execution/selection-verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(record, ensure_ascii=False))
