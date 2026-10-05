import hashlib
import json
import platform
from pathlib import Path

here = Path(__file__).resolve().parent
source = (here / "original/section.md").read_bytes()
target = (here / "original/linked-7.12.md").read_bytes()
text = target.decode("utf-8")
result = {
    "source_sha256": hashlib.sha256(source).hexdigest(),
    "target_sha256": hashlib.sha256(target).hexdigest(),
    "source_anchor_label_present": "[7.12實測](#7.12)" in source.decode("utf-8"),
    "target_heading": text.splitlines()[0],
    "target_no_training_statement": "沒有訓練模型或產生答題成績" in text,
    "target_original_json_link": "docs/course-experiments/results/sft.json" in text,
    "target_contains_claimed_score_phrases": {s: s in text for s in ["1題答對", "7題答對", "5／10", "250"]},
    "target_scope_statement": "本節只示範材料與切分，沒有新訓練結果" in text,
    "environment": {"python": platform.python_version(), "device": "cpu; text inspection only"},
    "scope": "Link target and phrase facts only. Reviewer separately reads both sections to judge attribution, including the valid indirect JSON link.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
