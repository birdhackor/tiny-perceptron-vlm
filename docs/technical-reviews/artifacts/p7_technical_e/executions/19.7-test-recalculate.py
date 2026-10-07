import json
from pathlib import Path
v=json.loads(Path("docs/selftrained/results/public-raw/moe/test/metrics.json").read_text())
r=v["per_task_final_reply"]["tool_call"]["tool_roundtrip"];threshold=v["thresholds"]["tool_roundtrip"]
assert r["numerator"]==0 and r["denominator"]==276 and r["rate"]==r["numerator"]/r["denominator"]<threshold==.95
print(json.dumps({"original_path":"docs/selftrained/results/public-raw/moe/test/metrics.json","roundtrip":r,"threshold":threshold,"recalculation":r["numerator"]/r["denominator"],"scope":"original stored measurement only, not new full model evaluation"},ensure_ascii=False))
