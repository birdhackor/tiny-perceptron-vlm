"""核對固定候選、選法、分母與算術驗證規則；不重新生成或訓練模型。"""

import ast
import copy
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def independent_check(candidate, row, mode):
    text = candidate["generated"].strip()
    if candidate["invalid_special_tokens"]:
        return None, False, False
    if mode == "direct":
        final = int(text) if re.fullmatch(r"-?[0-9]+", text) else None
        return final, final == row["truth"], final == row["truth"]
    suffix = re.search(r";answer=(-?[0-9]+)$", text)
    final = int(suffix[1]) if suffix else None
    parts = text.split(";")
    equations = [re.fullmatch(r"(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+)", p) for p in parts[:2]]
    valid = False
    if len(parts) == 3 and all(equations) and re.fullmatch(r"answer=-?[0-9]+", parts[2]):
        left, right = [tuple(map(int, match.groups())) for match in equations]
        valid = (
            left[0] + left[1] == left[2]
            and right[0] + right[1] == right[2]
            and left[2] == right[0]
            and right[2] == final
            and (left[0], left[1], right[1]) == (row["a"], row["b"], row["c"])
            and final == row["truth"]
        )
    return final, final == row["truth"], valid


code_path = "scripts/course_experiments/applications.py"
tree = ast.parse((ROOT / code_path).read_text())
names = {"_rate", "_verify_reasoning", "_reasoning_samples"}
definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(definitions) == len(names)
namespace = {"Counter": Counter, "re": re}
exec(compile(ast.Module(body=definitions, type_ignores=[]), code_path, "exec"), namespace)

data_path = "docs/course-experiments/results/reasoning.json"
record = json.loads((ROOT / data_path).read_text())
summary = []
examples = {}
checked_candidates = 0
for mode, model in record["results"]["comparison"].items():
    budgets = model["budgets"]
    assert [b["candidate_count"] for b in budgets] == [1, 2, 4, 8]
    lookup = {b["candidate_count"]: {s["question"]: s for s in b["samples"]} for b in budgets}
    rows = [{key: s[key] for key in ("family", "a", "b", "c", "question", "truth")} for s in budgets[0]["samples"]]
    assert len(rows) == len({s["question"] for s in rows}) == 24

    def stored_sample(model, messages, ctx, count, tokens, temperature):
        sample = lookup[count][messages[0]["content"]]
        assert sample["generation"]["temperature"] == temperature == 0.7
        assert sample["generation"]["max_new_tokens"] == tokens == (8 if mode == "direct" else 48)
        fields = ("generated", "generated_ids", "eos", "invalid_special_tokens", "generated_tokens")
        return {
            **copy.deepcopy(sample["generation"]),
            "samples": [{key: copy.deepcopy(c[key]) for key in fields} for c in sample["candidates"]],
        }

    namespace["_sample"] = stored_sample
    replay = namespace["_reasoning_samples"](None, rows, None, mode)
    assert replay == budgets
    for budget in budgets:
        coverage = majority = verifier = correct_invalid = conditional = 0
        for sample in budget["samples"]:
            assert sample["truth"] == sample["a"] + sample["b"] + sample["c"]
            assert len(sample["candidates"]) == budget["candidate_count"]
            evaluated = [independent_check(c, sample, mode) for c in sample["candidates"]]
            for candidate, values in zip(sample["candidates"], evaluated, strict=True):
                assert values == (candidate["final_answer"], candidate["final_correct"], candidate["fully_verified"])
                checked_candidates += 1
            answers = [x[0] for x in evaluated if x[0] is not None]
            counts = {answer: answers.count(answer) for answer in answers}
            selected = max(counts, key=counts.get) if counts else None
            accepted = next((x[0] for x in evaluated if x[2]), None)
            covered = any(x[1] for x in evaluated)
            assert sample["majority_answer"] == selected
            assert sample["verified_answer"] == accepted
            assert sample["oracle_coverage"] == covered
            coverage += covered
            majority += selected == sample["truth"]
            verifier += accepted == sample["truth"]
            conditional += covered and selected == sample["truth"]
            correct_invalid += sum(final_correct and not fully_verified for _, final_correct, fully_verified in evaluated)
            if (mode == "direct" and budget["candidate_count"] == 4 and sample["question"] == "(3+3)+0=?") or (
                mode == "steps" and budget["candidate_count"] == 2 and sample["question"] == "(5+0)+3=?"
            ):
                examples[mode] = {
                    "question": sample["question"],
                    "truth": sample["truth"],
                    "generated": [c["generated"] for c in sample["candidates"]],
                    "final_answers": answers,
                    "majority_answer": selected,
                    "verified_answer": accepted,
                }
        assert budget["oracle_coverage"]["numerator"] == coverage
        assert budget["majority_accuracy"]["numerator"] == majority
        assert budget["verifier_accuracy"]["numerator"] == verifier
        assert budget["majority_accuracy_given_coverage"]["numerator"] == conditional
        assert budget["majority_accuracy_given_coverage"]["denominator"] == coverage
        assert budget["final_correct_but_steps_invalid"]["numerator"] == correct_invalid == 0
        assert coverage == verifier
        summary.append({
            "mode": mode, "k": budget["candidate_count"], "questions": len(budget["samples"]),
            "candidates": sum(len(s["candidates"]) for s in budget["samples"]),
            "coverage": coverage, "majority_correct": majority, "verifier_correct": verifier,
            "majority_accuracy_given_coverage": f"{conditional}/{coverage}",
            "final_correct_but_steps_invalid": correct_invalid,
        })

source = (ROOT / "docs/course-review-20261019/freeze-01/sources/C.4.md").read_text()
fence = re.search(r"```python\n(.*?)\n```", source, re.S)[1]
toy = []
for candidates in ([3, 3, 4], [3, 3, 5], []):
    adapted = fence.replace("candidates = [3, 3, 4]", f"candidates = {candidates!r}")
    if not candidates:
        toy.append({"candidates": candidates, "verifier": next((x for x in candidates if x == 2 + 2), None)})
        continue
    stdout = io.StringIO()
    local = {}
    with redirect_stdout(stdout):
        exec(compile(adapted, "candidate-selection-example", "exec"), local)
    assert local["majority"] == 3
    assert local["verified"] == (4 if 4 in candidates else None)
    toy.append({"candidates": candidates, "most_common": Counter(candidates).most_common(1), "stdout": stdout.getvalue()})

visited = []


def observe(x):
    visited.append(x)
    return x == 4


lazy = (x for x in [3, 4, 4] if observe(x))
assert visited == []
assert next(lazy, None) == 4
assert visited == [3, 4]
assert Counter([10, 8]).most_common(1) == [(10, 1)]
assert Counter([8, 10]).most_common(1) == [(8, 1)]

row = {"a": 5, "b": 0, "c": 3, "truth": 8}
probes = []
for text, special in [
    ("5+0=5;5+3=8;answer=8", []),
    ("5+0=5;5+3=9;answer=8", []),
    ("5+0=5;6+3=9;answer=8", []),
    ("4+1=5;5+3=8;answer=8", []),
    ("5+0=5;5+3=8;answer=7", []),
    ("junk;answer=8", []),
    ("5+0=5;5+3=8;answer=8", [1]),
]:
    result = namespace["_verify_reasoning"]({"generated": text, "invalid_special_tokens": special}, row, "steps")
    assert result["fully_verified"] == (text == "5+0=5;5+3=8;answer=8" and not special)
    probes.append({"generated": text, "invalid_special_tokens": special, "result": result})

freeze_path = "docs/course-review-20261019/freeze-04.json"
freeze = json.loads((ROOT / freeze_path).read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "C.4")
figure_path = "course/figures/rewrite-C-candidate-choice.svg"
receipt_path = "docs/course-review-20261019/figure-renders-04.json"
receipt = next(r for r in json.loads((ROOT / receipt_path).read_text())["records"] if r["page_id"] == "C.4")
assert digest(page["snapshot"]) == page["source_sha256"] == "43c1977aa09ea08c588563f6b11c6c71a59ff5c2591204abf5d9090ecff507b2"
assert digest(freeze_path) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
assert digest(figure_path) == page["figures_sha256"][figure_path] == receipt["frozen_svg"]["sha256"]
assert digest(receipt["png"]["path"]) == receipt["png"]["sha256"]
links = {
    "Counter": "from collections import Counter" in (ROOT / "course/chapters/01.md").read_text(),
    "most_common": "pairs.most_common(1)[0][0]" in (ROOT / "course/chapters/06.md").read_text(),
}
assert all(links.values())
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-C-4/verify.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "torch": importlib.metadata.version("torch"), "device": "cpu"},
    "inputs": {p: digest(p) for p in (data_path, code_path, page["snapshot"], freeze_path, figure_path, receipt_path, "course/chapters/01.md", "course/chapters/06.md")},
    "original_run": {k: record[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "evidence_status")},
    "method": "從現行原始碼擷取評分與選法函式，餵入已保存的候選重放；另外獨立解析每個候選並重算等式、選法與分母。沒有重新訓練或抽樣。",
    "checked_candidates": checked_candidates, "all_replayed_budgets_equal_record": True,
    "budgets": summary, "examples": examples, "toy": toy,
    "tie_rule": {"forward": Counter([10, 8]).most_common(1), "reversed": Counter([8, 10]).most_common(1)},
    "generator": {"initially_unevaluated": True, "observations_until_first_match": visited, "stops_after_first_match": True},
    "verifier_probes": probes, "linked_example_checks": links,
    "figure_receipt": receipt,
    "result": "所有核對成立。固定紀錄的候選與彙總一致，練習輸出符合預測。",
}, ensure_ascii=False, indent=2))
