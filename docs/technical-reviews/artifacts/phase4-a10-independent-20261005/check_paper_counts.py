"""Check the cited total in three original-paper locations, not dataset recounting."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
text = (HERE / "longmemeval-2410.10813v2.txt").read_text()
lines = text.split("\n")
assert "arXiv:2410.10813v2 [cs.CL] 4 Mar 2025" in lines[16]
assert "With 500 meticulously" in lines[27]
assert "consists of 500 manually created questions" in lines[84]
assert "Personal  50k 500" in lines[152]
print(json.dumps({"version": "arXiv:2410.10813v2, 4 Mar 2025", "reported_question_count": 500, "cross_checked_text_lines_one_based": [28, 85, 153], "scope": "500 is the original paper's reported full benchmark total, not a dataset recount, number of sessions, local sample denominator, or reproduced model score."}, indent=2))
