"""複查凍結原文、既有證據與加法驗證規則；輸出另存新檔。"""

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def example(steps):
    used = [steps[0][0]] + [b for a, b, claimed in steps]
    return {
        "steps": steps,
        "valid": [a + b == claimed for a, b, claimed in steps],
        "linked": [steps[i][0] == steps[i - 1][2] for i in range(1, len(steps))],
        "used": used,
        "task_operands_valid": sorted(used) == sorted([2, 2]),
        "final_correct": steps[-1][2] == 2 + 2,
    }


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_C-2/technical-C-2-original-01.json"
assert digest(original_path) == "f3618d6162952533f47b317851458b78ada378317e1f87b1c4d1a6c113788521"
original = json.loads(original_path.read_bytes())
old_source = ROOT / "docs/course-review-20261019/freeze-01/sources/C.2.md"
new_source = ROOT / "docs/course-review-20261019/freeze-05/sources/C.2.md"
freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
assert digest(old_source) == original["source_sha256"]
assert digest(new_source) == "df11c67ed4dc656f40d362858af2e4486d6cd979c293bea9d1b7d387f3dd8bac"
assert digest(freeze_path) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
freeze = json.loads(freeze_path.read_bytes())
page = next(page for page in freeze["pages"] if page["page_id"] == "C.2")
assert page["snapshot"] == new_source.relative_to(ROOT).as_posix()
assert page["source_sha256"] == digest(new_source)
assert page["figures_sha256"] == {}

old_artifacts = []
for item in original["artifacts"]:
    actual = digest(ROOT / item["path"])
    assert actual == item["sha256"], "既有證據不可改寫"
    old_artifacts.append({"path": item["path"], "sha256": actual, "unchanged": True})
repo_sources = []
for item in original["sources"]:
    if item["kind"] == "repository_code":
        actual = digest(ROOT / item["path"])
        assert actual == item["sha256"], "沿用的程式與紀錄版本必須一致"
        repo_sources.append({"path": item["path"], "sha256": actual, "unchanged": True})

command = ["git", "diff", "--no-index", str(old_source.relative_to(ROOT)), str(new_source.relative_to(ROOT))]
diff = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
assert diff.returncode == 1 and not diff.stderr
changed = [line[1:] for line in diff.stdout.splitlines() if line.startswith("+") and not line.startswith("+++")]
assert len(changed) == 2
assert "這是本實驗另訂的規則：題目的數字各用一次，不引入別的常數" in changed[0]
assert "這份回答的算術與最後答案都對" in changed[1]
assert "不符合本例「只用題目數字」的規則" in changed[1]
assert "不是在解原題" not in new_source.read_text()

section_command = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/0C.md#C.2", "--execute"]
section = subprocess.run(section_command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, check=False)
assert section.returncode == 0, "小節原始程式執行必須成功"
summary = json.loads(section.stdout)
assert summary["source_sha256"] == digest(new_source)
assert summary["python_fences"] == 1 and summary["svg_references"] == 0
assert summary["execution_exit_code"] == 0
run = Path(summary["artifact_directory"])
assert (run / "section.md").read_bytes() == new_source.read_bytes()
copied_files = {}
for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py"):
    raw = (run / name).read_bytes()
    copied_files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "content_utf8": raw.decode()}
expected_stdout = "等式算對 [False, True]\n接上前步 [True]\n只用題目的數字 False\n最後答案對 True\n"
assert copied_files["stdout.txt"]["content_utf8"] == expected_stdout
assert copied_files["stderr.txt"]["content_utf8"] == ""
old_execution = json.loads((ARTIFACTS / "section-execution.json").read_bytes())
assert copied_files["fence-1.py"]["sha256"] == old_execution["copied_files"]["fence-1.py"]["sha256"]

implementation = ROOT / "scripts/course_experiments/applications.py"
tree = ast.parse(implementation.read_bytes())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_verify_reasoning")
namespace = {"re": re}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(implementation), "exec"), namespace)
verify = namespace["_verify_reasoning"]
record = json.loads((ROOT / "docs/course-experiments/results/reasoning.json").read_bytes())
budgets = record["results"]["comparison"]["steps"]["budgets"]
assert [budget["candidate_count"] for budget in budgets] == [1, 2, 4, 8]
count = 0
matches = []
for budget_index, budget in enumerate(budgets):
    assert len(budget["samples"]) == 24
    for sample_index, sample in enumerate(budget["samples"]):
        assert len(sample["candidates"]) == budget["candidate_count"]
        assert sample["truth"] == sample["a"] + sample["b"] + sample["c"]
        for candidate_index, candidate in enumerate(sample["candidates"]):
            observed = verify(candidate, sample, "steps")
            assert all(candidate[key] == value for key, value in observed.items())
            count += 1
            if candidate["generated"] == "5+0=5;5+3=10;answer=10":
                matches.append({"budget_index": budget_index, "sample_index": sample_index, "candidate_index": candidate_index, "result": observed})
assert count == 360 and len(matches) == 5
prior_verification = json.loads((ARTIFACTS / "verification-output.json").read_bytes())
repeated_injected = []
row = {"a": 5, "b": 0, "c": 3, "truth": 8}
for case in prior_verification["injected_verifier_cases"]:
    observed = verify(case, row, "steps")
    assert observed == case["result"]
    repeated_injected.append({"case": case["case"], "result": observed})

examples = {
    "original": example([(2, 2, 5), (5, -1, 4)]),
    "changed_first_result": example([(2, 2, 4), (5, -1, 4)]),
    "detour": example([(2, 2, 4), (4, 3, 7), (7, -3, 4)]),
}
for name, values in examples.items():
    for key, value in values.items():
        assert prior_verification["examples"][name][key] == json.loads(json.dumps(value))
assert examples["detour"]["valid"] == [True, True, True]
assert examples["detour"]["linked"] == [True, True]
assert examples["detour"]["final_correct"] is True
assert examples["detour"]["task_operands_valid"] is False
assert ((2 + 2) + 3) - 3 == 2 + 2 == 4

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-C-2/recheck-facts-current-01.py",
    "environment": {"python": sys.version, "device": "cpu", "platform": platform.platform()},
    "original_report": {"path": original_path.relative_to(ROOT).as_posix(), "sha256": digest(original_path)},
    "source_freeze": {"path": freeze_path.relative_to(ROOT).as_posix(), "sha256": digest(freeze_path)},
    "source_sha256": digest(new_source),
    "prior_artifacts": old_artifacts,
    "repository_sources": repo_sources,
    "diff": {"command_argv": command, "exit_code": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr},
    "changed_sentences": [
        {"sentence": changed[0], "verified": True, "basis": "_verify_reasoning 的 task 依有序(a,b,c)檢查題目運算元；這是特定任務規則。全部保存候選與注入例重算相符。"},
        {"sentence": changed[1], "verified": True, "basis": "逐步等式、接續與最後答案都成立，僅運算元多重集合不同；((2+2)+3)-3=4。"},
    ],
    "section_execution": {"command_argv": section_command, "exit_code": section.returncode, "stdout": section.stdout, "stderr": section.stderr, "copied_from": str(run), "copied_files": copied_files},
    "examples": examples,
    "candidate_recomputation": {"candidates": count, "mismatches": 0, "budget_sample_counts": [len(b["samples"]) for b in budgets], "quoted_generation_matches": matches},
    "repeated_injected_cases": repeated_injected,
    "limits": ["未重新訓練或重新生成。", "現行實作與舊紀錄的全檔hash差異仍存在；沿用先前已保存的追溯限制。", "文字驗算不揭露內部因果計算。"],
    "result": "原文與凍結版本相同、舊證據未改、算例與全部保存候選重算一致；兩處新敘述均有直接依據。",
}
output = ARTIFACTS / "recheck-evidence-current-01.json"
with output.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"path": output.relative_to(ROOT).as_posix(), "sha256": digest(output), "result": result["result"]}, ensure_ascii=False))
