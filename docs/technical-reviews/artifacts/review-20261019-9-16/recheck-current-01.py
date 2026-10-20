"""核對凍結差異、沿用證據與成本估計的成立條件，不更新原證據。"""

import hashlib
import json
import os
import platform
import re
import subprocess
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-9-16"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_9-16/technical-9-16-original-01.json"
    assert sha(original_path) == "c73cdd773cdf970122b0227998b641c81f1123847c2939ea76f5f35217378f20"
    original = json.loads(original_path.read_text())
    old_path = ROOT / "docs/course-review-20261019/freeze-01/sources/9.16.md"
    current_path = ROOT / "docs/course-review-20261019/freeze-05/sources/9.16.md"
    assert sha(old_path) == "fa94160307cf56025270d43bb170081204943b41485f381fab5489787a9852b3"
    assert sha(current_path) == "6f2a91a3d5ac8d803f810a1090c3860d33b6b487811cfdb5eebd04692c36d139"
    old, current = old_path.read_bytes(), current_path.read_bytes()
    before = "期望成本都會算錯".encode()
    after = "就不能保證期望成本算得正確".encode()
    assert old.count(before) == 1 and old.replace(before, after) == current
    chapter = (ROOT / "course/chapters/09.md").read_bytes()
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter))
    index = next(i for i, match in enumerate(headings) if match[0].startswith(b"## 9.16 "))
    extracted = chapter[headings[index].start():headings[index + 1].start()]
    assert extracted == current
    fence = rb"```python\n(.*?)```"
    assert re.findall(fence, old, re.S) == re.findall(fence, current, re.S)
    audit = []
    for artifact in original["artifacts"]:
        actual = sha(ROOT / artifact["path"])
        assert actual == artifact["sha256"]
        audit.append({"path": artifact["path"], "recorded_sha256": artifact["sha256"], "actual_sha256": actual})
    for source in original["sources"]:
        if source["kind"] == "repository_code":
            actual = sha(ROOT / source["path"])
            assert actual == source["sha256"]
            audit.append({"path": source["path"], "recorded_sha256": source["sha256"], "actual_sha256": actual})
    command = [".venv/bin/python", str((BASE / "verify.py").relative_to(ROOT))]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, check=True)
    assert result.stderr == ""
    repeated = json.loads(result.stdout)
    previous = json.loads((BASE / "verification.json").read_text())
    assert repeated == previous
    diff_command = ["git", "diff", "--no-index", str(old_path.relative_to(ROOT)), str(current_path.relative_to(ROOT))]
    diff = subprocess.run(diff_command, cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 1 and not diff.stderr
    p, truth = Fraction(3, 5), Fraction(2, 5)
    estimate, actual = 5 * (1 - p), 5 * (1 - truth)
    assert estimate == 2 and actual == 3 and estimate - actual == 5 * (truth - p)
    concentration = (3 * Fraction(3, 5) - 1) / 2
    concentration_estimate = 5 * (1 - concentration)
    concentration_actual = 5 * (1 - Fraction(3, 5))
    assert concentration == Fraction(2, 5) and concentration_estimate == 3 and concentration_actual == 2
    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-16/recheck-current-01.py",
        "environment": repeated["environment"],
        "result": "全部斷言通過；新版僅改缺乏保證的敘述，所有沿用證據與 repo 來源雜湊相符，補充驗算重跑結果相同。",
        "original_report_sha256": sha(original_path), "source_sha256": sha(current_path),
        "current_chapter_section_matches_freeze": True, "python_fences_identical": True,
        "changed_wording": {"before": before.decode(), "after": after.decode()},
        "diff": {"command": diff_command, "exit_code": diff.returncode, "stdout": diff.stdout},
        "preserved_evidence": audit,
        "repeated_verification": {"command": command, "exit_code": result.returncode,
                                  "result_equals_saved_output": True, "stdout": result.stdout, "stderr": result.stderr},
        "cost_estimation": {"derivation": "設真實答對機率為 q，代入 p 的估計誤差是 5*(1-p)-5*(1-q)=5*(q-p)。缺少 p=q 的依據，無法保證誤差為零；這不表示每次必有非零誤差。",
                            "uncalibrated_example": {"p": str(p), "true_success_probability": str(truth),
                                                     "estimated_cost": str(estimate), "actual_cost": str(actual)},
                            "concentration_example": {"candidate_probabilities": ["3/5", "1/5", "1/5"],
                                                      "concentration": str(concentration), "true_success_probability": "3/5",
                                                      "estimated_cost_using_concentration": str(concentration_estimate),
                                                      "actual_cost": str(concentration_actual)},
                            "scope": "這些是指定母體與成本的精確推導，沒有實測部署品質，也不把校準當成單題成功保證。"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
