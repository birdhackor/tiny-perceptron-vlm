"""核對整數算例、候選紀錄、實際驗證器與原始來源的小份收據。"""

import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def example(steps):
    used = [steps[0][0]] + [b for a, b, claimed in steps]
    return {
        "steps": steps,
        "valid": [a + b == claimed for a, b, claimed in steps],
        "linked": [steps[i][0] == steps[i - 1][2] for i in range(1, len(steps))],
        "used": used,
        "sorted_used": sorted(used),
        "sorted_question": sorted([2, 2]),
        "task_operands_valid": sorted(used) == sorted([2, 2]),
        "final_correct": steps[-1][2] == 2 + 2,
    }


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def original_source(url, version, locator, excerpts, markers):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "25", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    parser = VisibleText()
    parser.feed(completed.stdout.decode())
    text = " ".join(" ".join(parser.parts).split())
    for excerpt in excerpts:
        assert excerpt in text, "來源摘錄須存在於當次讀取的原文"
    for marker in markers:
        assert marker in text, "來源定位須存在於當次讀取的原文"
    return {
        "url": url,
        "version": version,
        "accessed_at": datetime.now(UTC).isoformat(),
        "command_argv": command,
        "exit_code": completed.returncode,
        "response_bytes": len(completed.stdout),
        "response_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "locator": locator,
        "excerpts": excerpts,
        "locator_markers_checked": markers,
        "full_text_retained": False,
    }


frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/C.2.md"
freeze = ROOT / "docs/course-review-20261019/freeze-04.json"
current = (ROOT / "course/chapters/0C.md").read_bytes()
headings = list(re.finditer(rb"(?m)^## [^\r\n]+", current))
position = next(index for index, heading in enumerate(headings) if heading[0].startswith(b"## C.2 "))
end = headings[position + 1].start() if position + 1 < len(headings) else len(current)
section = current[headings[position].start():end]
assert section == frozen.read_bytes(), "當前小節與凍結原文須相同"

original = example([(2, 2, 5), (5, -1, 4)])
modified = example([(2, 2, 4), (5, -1, 4)])
detour = example([(2, 2, 4), (4, 3, 7), (7, -3, 4)])
assert original["valid"] == [False, True]
assert original["linked"] == [True]
assert original["used"] == [2, 2, -1]
assert original["final_correct"] and not original["task_operands_valid"]
assert modified["valid"] == [True, True] and modified["linked"] == [False]
assert detour["valid"] == [True, True, True] and detour["linked"] == [True, True]
assert detour["final_correct"] and not detour["task_operands_valid"]
assert ((2 + 2) + 3) - 3 == 2 + 2 == 4

implementation = ROOT / "scripts/course_experiments/applications.py"
tree = ast.parse(implementation.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_verify_reasoning")
namespace = {"re": re}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(implementation), "exec"), namespace)
verify = namespace["_verify_reasoning"]
record_path = ROOT / "docs/course-experiments/results/reasoning.json"
record = json.loads(record_path.read_bytes())
recorded_implementation_sha = record["code_sha256"][implementation.relative_to(ROOT).as_posix()]
budgets = record["results"]["comparison"]["steps"]["budgets"]
assert [budget["candidate_count"] for budget in budgets] == [1, 2, 4, 8]
budget_summary = []
matches = []
all_candidates = []
for budget_index, budget in enumerate(budgets):
    assert len(budget["samples"]) == 24
    candidate_total = 0
    for sample_index, sample in enumerate(budget["samples"]):
        assert sample["truth"] == sample["a"] + sample["b"] + sample["c"]
        assert sample["question"] == f"({sample['a']}+{sample['b']})+{sample['c']}=?"
        assert len(sample["candidates"]) == budget["candidate_count"]
        for candidate_index, candidate in enumerate(sample["candidates"]):
            recomputed = verify(candidate, sample, "steps")
            for key, value in recomputed.items():
                assert candidate[key] == value, "保存的判定須符合當前驗證器重算結果"
            assert candidate["fully_verified"] == all(
                candidate[key] for key in ("equations_valid", "linked", "task_operands_valid", "final_correct")
            )
            candidate_total += 1
            all_candidates.append(candidate)
            if candidate["generated"] == "5+0=5;5+3=10;answer=10":
                matches.append({
                    "budget_index": budget_index,
                    "candidate_count": budget["candidate_count"],
                    "sample_index": sample_index,
                    "candidate_index": candidate_index,
                    "question": sample["question"],
                    "truth": sample["truth"],
                    "candidate": candidate,
                    "recomputed": recomputed,
                })
    budget_summary.append({
        "candidate_count": budget["candidate_count"],
        "samples": len(budget["samples"]),
        "candidates": candidate_total,
    })
assert matches, "正文引用的生成須存在於紀錄"

row = {"a": 5, "b": 0, "c": 3, "truth": 8}
cases = [
    ("正確回答", "5+0=5;5+3=8;answer=8", []),
    ("等式錯誤", "5+0=5;5+3=10;answer=10", []),
    ("接續錯誤", "5+0=5;6+3=9;answer=9", []),
    ("換掉題目運算元", "4+1=5;5+3=8;answer=8", []),
    ("最後答案與最後等式不一致", "5+0=5;5+3=8;answer=9", []),
    ("格式錯誤但答案正確", "5 + 0=5;5+3=8;answer=8", []),
    ("含非法特殊token", "5+0=5;5+3=8;answer=8", [0]),
]
injected = []
for name, generated, invalid in cases:
    result = verify({"generated": generated, "invalid_special_tokens": invalid}, row, "steps")
    assert result["fully_verified"] == (name == "正確回答")
    injected.append({"case": name, "generated": generated, "invalid_special_tokens": invalid, "result": result})

sources = [
    original_source(
        "https://arxiv.org/html/2305.20050v1",
        "arXiv:2305.20050v1, 2023-05-31",
        "Outcome-supervised Reward Models (ORMs), HTML paragraph S2.SS5.p1.1；Methods 與 Scope",
        ["false positives solutions that reach the correct answer with incorrect reasoning will be misgraded."],
        ["Outcome-supervised Reward Models (ORMs)", "there is no simple way to automate process supervision"],
    ),
    original_source(
        "https://arxiv.org/html/2305.04388v2",
        "arXiv:2305.04388v2, 2023-12-09",
        "Introduction 中討論正確 reasoning 與 faithfulness 的差別；Evaluating Systematic Unfaithfulness",
        [
            "even if CoT explanations seem plausible and have correct reasoning",
            "We should not expect CoT explanations to be faithful by default",
        ],
        ["Counterfactual Simulatability", "Evaluating Systematic Unfaithfulness"],
    ),
    original_source(
        "https://docs.python.org/3.13/reference/expressions.html#binary-arithmetic-operations",
        "Python 3.13 Language Reference, Expressions，當次 HTTPS 文件",
        "Binary arithmetic operations，HTML anchors index-70 與 index-71",
        [
            "The + (addition) operator yields the sum of its arguments.",
            "The - (subtraction) operator yields the difference of its arguments.",
        ],
        ["Binary arithmetic operations"],
    ),
]

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-C-2/verify_facts.py",
    "environment": {"python": sys.version, "device": "cpu", "platform": platform.platform()},
    "source_bindings": {
        "frozen_source_sha256": sha(frozen),
        "freeze_sha256": sha(freeze),
        "current_section_matches_freeze": True,
        "implementation_path": implementation.relative_to(ROOT).as_posix(),
        "implementation_sha256": sha(implementation),
        "recorded_implementation_sha256": recorded_implementation_sha,
        "implementation_matches_recorded_code_sha256": sha(implementation) == recorded_implementation_sha,
        "record_path": record_path.relative_to(ROOT).as_posix(),
        "record_sha256": sha(record_path),
    },
    "examples": {"original": original, "changed_first_result": modified, "detour": detour},
    "derivation": {
        "original_problem_value": 2 + 2,
        "expanded_expression": "((2 + 2) + 3) - 3",
        "expanded_expression_value": ((2 + 2) + 3) - 3,
        "arithmetic_equality": True,
        "note": "引入相抵的常數不改變算術值；是否准許額外數字屬題目另訂的解答限制。",
    },
    "record_metadata": {key: record[key] for key in ["revision", "seed", "device", "gpu", "python_version", "torch_version", "evidence_status", "status"]},
    "budget_summary": budget_summary,
    "candidate_recomputation": {
        "total": len(all_candidates),
        "mismatches": 0,
        "parsed": sum(candidate["parsed"] for candidate in all_candidates),
        "parsed_but_final_incorrect": sum(candidate["parsed"] and not candidate["final_correct"] for candidate in all_candidates),
        "fully_verified": sum(candidate["fully_verified"] for candidate in all_candidates),
        "scope": "重算保存的候選及評分，沒有重新訓練或重新抽樣。候選不是獨立重複訓練的樣本。",
    },
    "quoted_generation_matches": matches,
    "injected_verifier_cases": injected,
    "original_sources": sources,
    "result": "所有算例斷言、候選欄位重算與來源摘錄定位均通過。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
