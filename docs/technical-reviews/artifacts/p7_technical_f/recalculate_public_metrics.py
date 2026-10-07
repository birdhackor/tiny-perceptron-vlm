import json
from pathlib import Path

for arch in ("moe", "dense"):
    path = Path(f"docs/selftrained/results/public-raw/{arch}/test/metrics.json")
    raw = json.loads(path.read_text())
    tasks = raw["per_task_final_reply"]
    measured = {}
    for name, task in tasks.items():
        field = "tool_roundtrip" if name == "tool_call" else "semantic"
        rate = task[field]
        assert rate["denominator"] == task["count"]
        assert abs(rate["rate"] - rate["numerator"] / rate["denominator"]) < 1e-12
        measured[name] = {"numerator": rate["numerator"], "denominator": rate["denominator"], "recalculated_rate": rate["numerator"] / rate["denominator"], "source_groups": task["source_groups"], "recording_assets": task["recording_assets"]}
    assert sum(task["count"] for task in tasks.values()) == raw["count"] == 3734
    print(json.dumps({"architecture": arch, "count": raw["count"], "evaluation_complete": raw["evaluation_complete"], "limited_smoke": raw["limited_smoke"], "weight_source": raw["weight_source"], "safe_weights_sha256": raw["safe_weights_sha256"], "thresholds": raw["thresholds"], "tasks": measured, "ocr_perception": raw["perception"]["ocr"], "voice_topic_continuation": raw["voice_topic_continuation"], "voice_scope": raw["voice_scope"], "toy_micro": 12/18, "toy_macro": (1+0)/2}, ensure_ascii=False))
