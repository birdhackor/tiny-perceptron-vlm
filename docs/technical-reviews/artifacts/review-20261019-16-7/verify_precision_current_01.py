"""核對凍結原文差異、沿用證據、當次執行與 Python 取位規約。"""

import difflib
import hashlib
import html
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-16-7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_16-7/technical-16-7-original-01.json"
assert digest(original_path) == "66a066161634c288c71b63f1d5277e4b7d6c6c3b1921d2715281b0e7d062a597"
original = json.loads(original_path.read_bytes())
old_source = ROOT / "docs/course-review-20261019/freeze-01/sources/16.7.md"
new_source = ROOT / "docs/course-review-20261019/freeze-05/sources/16.7.md"
old_text, new_text = old_source.read_text(), new_source.read_text()
assert digest(old_source) == original["source_sha256"]
assert digest(new_source) == "60abfb5779f7341d6bb95574bc5ae26f672f58a6d8d5f2c45897c4e0061818db"
old_phrase, new_phrase = "最後四捨五入到四位小數。", "最後取到四位小數。"
assert new_text.count(new_phrase) == 1
assert new_text.replace(new_phrase, old_phrase, 1) == old_text
assert original["issues"] == []
audit = []
for group in ("sources", "artifacts"):
    for item in original[group]:
        if "path" not in item:
            continue
        path = ROOT / item["path"]
        observed = digest(path)
        assert observed == item["sha256"]
        audit.append({"group": group, "id": item["id"], "path": item["path"],
                      "expected_sha256": item["sha256"], "observed_sha256": observed})
old_check = BASE / "self-check.txt"
assert digest(old_check) == "59a89b2d4c404045dd7267ee58a53ffa57b282089309fca04b07b6ab4e873020"

run = ROOT / "outputs/reviewer-tools/runs/20261020T084432469680Z"
execution = json.loads((run / "execution.json").read_bytes())
environment = json.loads((run / "environment.json").read_bytes())
extraction = json.loads((run / "extraction.json").read_bytes())
assert execution["exit_code"] == 0
assert extraction["source_sha256"] == digest(new_source)
assert extraction["figure_sha256"] == {}
old_execution = json.loads((BASE / "section-execution.json").read_bytes())
assert (run / "fence-1.py").read_text() == old_execution["executed_code"]
assert (run / "stdout.txt").read_text() == (BASE / "section-stdout.txt").read_text()
section_copy = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/16.md#16.7" --execute',
                "execution": execution, "environment": environment, "extraction": extraction,
                "stdout": (run / "stdout.txt").read_text(), "stderr": (run / "stderr.txt").read_text(),
                "executed_code": (run / "fence-1.py").read_text(),
                "copied_original_sha256": {name: digest(run / name) for name in
                                           ("execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt", "fence-1.py")}}

command = [str(ROOT / ".venv/bin/python"), str(BASE / "verify_precision.py")]
replay = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
replayed = json.loads(replay.stdout)
assert replayed["result"] == "所有斷言通過"
prior_verification = json.loads((BASE / "precision-verification.json").read_bytes())
assert replayed == prior_verification

cases = [(1.03125, 4, 1.0312), (1.09375, 4, 1.0938),
         (2.5, 0, 2.0), (3.5, 0, 4.0), (0.075, 4, 0.075)]
rounding = []
for value, digits, expected in cases:
    observed = round(value, digits)
    assert observed == expected
    rounding.append({"value": value, "ndigits": digits, "expected": expected,
                     "observed": observed, "display": str(observed)})
raw_error = replayed["matrix_cases"]["original"]["max_error"]
assert round(raw_error, 4) == 0.0075

url = "https://docs.python.org/3.13/library/functions.html#round"
curl = ["curl", "-L", "--fail", "--silent", "--show-error", url]
fetched = subprocess.run(curl, capture_output=True, check=True)
raw = fetched.stdout
text = raw.decode()
start = text.index('id="round"')
end = text.find('<dl class="py function">', start + 20)
section = text[start:end] if end > start else text[start:start + 13000]
plain = re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", section)))
quote = "if two multiples are equally close, rounding is done toward the even choice"
assert quote in plain
assert "closest multiple" in plain and "ndigits" in plain
source_receipt = {"title": "Built-in Functions — round", "url": url,
                  "version": "Python 3.13.16 documentation / round(number, ndigits=None)",
                  "accessed_on": datetime.now(UTC).date().isoformat(),
                  "command_argv": curl, "exit_code": fetched.returncode,
                  "original_bytes": len(raw), "original_sha256": hashlib.sha256(raw).hexdigest(),
                  "checked_original": True, "locator": "round(number, ndigits=None)",
                  "short_excerpt": quote,
                  "inspection_note": "原文以最接近的10**(-ndigits)倍數取值；等距取偶，float可能有二進位表示差異。新句取到四位小數沒有指定五一律進位，也沒有承諾補零顯示。"}

print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-7/verify_precision_current_01.py",
                  "environment": {"python": platform.python_version(), "torch": replayed["environment"]["torch"], "device": "cpu", "platform": platform.platform()},
                  "original_report": {"path": relative(original_path), "sha256": digest(original_path), "issues": original["issues"]},
                  "frozen_sources": {relative(old_source): digest(old_source), relative(new_source): digest(new_source)},
                  "diff": "".join(difflib.unified_diff(old_text.splitlines(keepends=True), new_text.splitlines(keepends=True), fromfile=relative(old_source), tofile=relative(new_source))),
                  "changed_text": {"before": old_phrase, "after": new_phrase, "only_change": True},
                  "unchanged_evidence_audit": audit,
                  "prior_self_check_sha256": digest(old_check),
                  "current_section_execution": section_copy,
                  "unchanged_fence_and_stdout": True,
                  "supplementary_replay": {"command_argv": command, "exit_code": replay.returncode,
                                           "stderr": replay.stderr, "matches_prior_output_exactly": True, "result": replayed},
                  "rounding_cases": rounding, "actual_max_error": raw_error, "actual_rounded_max_error": round(raw_error, 4),
                  "official_source": source_receipt,
                  "result": "全部核對通過；新句成立；沿用證據未變。"}, ensure_ascii=False, indent=2, allow_nan=False))
