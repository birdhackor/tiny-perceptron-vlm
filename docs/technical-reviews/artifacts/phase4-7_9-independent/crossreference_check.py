"""Save and compare raw section bytes for the claimed 7.12 score location."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/review-tools"))
from section_facts import original_section

base = Path(__file__).parent
body, whole, line = original_section(ROOT / "course/chapters/07.md", "7.12")
(base / "section-7.12.md").write_bytes(body)
text = body.decode("utf-8")
source_text = (base / "section.md").read_text()
claim = "[7.12](#7.12)會列出成績"
assert claim in source_text
counterstatement = "沒有訓練模型或產生答題成績"
assert counterstatement in text
print(json.dumps({"python": sys.version, "source_claim": claim,
    "7.12_source_sha256": hashlib.sha256(body).hexdigest(),
    "7.12_first_line": line, "read_scope": "complete original 7.12 raw subsection",
    "7.12_explicit_counterstatement": counterstatement,
    "7.12_sft_result_link_present": "docs/course-experiments/results/sft.json" in text,
    "judgment": "7.12 links the complete independent report but does not list the scores; current forward-reference wording is inaccurate.",
    "verification_scope": "Document-location check, not re-review of 7.12 or old reader/technical reports"}, ensure_ascii=False, indent=2))
