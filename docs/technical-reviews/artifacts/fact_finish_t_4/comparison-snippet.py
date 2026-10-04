import json
import math
from pathlib import Path

kind = "attributes"
before = json.loads(Path(f"outputs/{kind}-before.json").read_text())
after = json.loads(Path(f"outputs/{kind}-after.json").read_text())
assert before["data"] == after["data"]
assert before["declared_split"] == after["declared_split"] == "validation"
assert before["effective_tokens"] == after["effective_tokens"] > 0
assert before["skipped"] == after["skipped"] == []
assert [s["row"] for s in before["samples"]] == [s["row"] for s in after["samples"]]
comparison = {}
for stage, report in (("before", before), ("after", after)):
    assert math.isfinite(report["mean_token_nll"])
    rate = report["exact_match"]
    assert rate is None or 0 <= rate <= 1
    comparison[stage] = {
        key: report[key] for key in ("mean_token_nll", "effective_tokens", "exact_match", "samples", "skipped")
    }
text = json.dumps(comparison, ensure_ascii=False, indent=2)
Path(f"outputs/{kind}-comparison.json").write_text(text + "\n")
print(text)
print("固定提示開始：", Path(f"outputs/{kind}-prompt-before.txt").read_text())
print("固定提示結束：", Path(f"outputs/{kind}-prompt-after.txt").read_text())
