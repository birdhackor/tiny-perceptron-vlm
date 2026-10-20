"""獨立重算評分範例、有限動作與已保存樣本；不執行訓練。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path.cwd()
DESTINATION = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def selected_nodes(text, names):
    return [
        node
        for node in ast.parse(text).body
        if getattr(node, "name", None) in names
        or (isinstance(node, ast.Assign) and any(getattr(target, "id", None) in names for target in node.targets))
    ]


def extracted_section(path, heading):
    text = path.read_text()
    start = text.index("## " + heading + " ")
    end = text.find("\n## ", start + 1)
    return text[start : end + 1 if end >= 0 else len(text)]


source = ROOT / "docs/course-review-20261019/freeze-01/sources/C.6.md"
frozen = source.read_bytes()
assert digest(frozen) == "6e21aa76d91ac562728363562863a15a391056c2d92d32d1d69c076efd524c3e"
assert extracted_section(ROOT / "course/chapters/0C.md", "C.6").encode() == frozen
code = re.findall(r"```python\n(.*?)```", frozen.decode(), flags=re.S)[0]
exercise = code.replace('"答案是4"]', '"答案是4", "0 1 2 3 4 5"]')
weak = exercise.replace("float(parsed == truth)", "float(str(truth) in text.split())")
example_outputs = {}
for name, snippet in (("original", code), ("exercise_strict", exercise), ("exercise_weak", weak)):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(snippet, name, "exec"), {})
    example_outputs[name] = stdout.getvalue()
assert example_outputs["original"] == "'3' 解析值 3 獎勵 0.0\n'4' 解析值 4 獎勵 1.0\n'5' 解析值 5 獎勵 0.0\n'答案是4' 解析值 None 獎勵 0.0\n"
assert example_outputs["exercise_strict"].endswith("'0 1 2 3 4 5' 解析值 None 獎勵 0.0\n")
assert example_outputs["exercise_weak"].endswith("'0 1 2 3 4 5' 解析值 None 獎勵 1.0\n")
assert "'答案是4' 解析值 None 獎勵 0.0" in example_outputs["exercise_weak"]

parser_cases = []
for text, truth, expected in [
    (" 4\n", 4, 1.0), ("-4", -4, 1.0), ("+4", 4, 0.0),
    ("4 5", 4, 0.0), ("答案是4", 4, 0.0), ("", 4, 0.0),
    ("4.0", 4, 0.0), ("4\n5", 4, 0.0), ("04", 4, 1.0), ("４", 4, 0.0),
]:
    clean = text.strip()
    parsed = int(clean) if re.fullmatch(r"-?[0-9]+", clean) else None
    reward = float(parsed == truth)
    assert reward == expected
    parser_cases.append({"text": text, "truth": truth, "clean": clean, "parsed": parsed, "reward": reward})
api_checks = {
    "match_object_truthy": bool(re.fullmatch(r"-?[0-9]+", "4")),
    "nonmatch_is_none": re.fullmatch(r"-?[0-9]+", "答案是4") is None,
    "raw_backslash_characters": list(r"\n"),
    "raw_pattern_equals_plain_without_backslash": r"-?[0-9]+" == "-?[0-9]+",
    "float_booleans": [float(True), float(False)],
    "repr_preserves_visible_quotes_and_whitespace": repr(" 4\n"),
    "split_with_whitespace": " 0\t1\n4 ".split(),
    "string_truth": str(4),
    "conditional_chosen_branch_only": (7 if True else 1 / 0),
}

result_path = ROOT / "docs/course-experiments/results/reasoning.json"
result = json.loads(result_path.read_bytes())
application_path = ROOT / "scripts/course_experiments/applications.py"
common_path = ROOT / "scripts/course_experiments/common.py"
application = application_path.read_text()
common = common_path.read_text()
policy_names = {"_FinitePolicy", "_policy_features", "_POLICY_ACTIONS", "_policy_reward", "_evaluate_policy", "_reasoning_records", "_rate", "_split_summary"}
training_names = {"_reinforce"}
common_names = {"split_records", "records_sha256"}
namespace = {"torch": torch, "nn": nn, "re": re, "random": random, "json": json, "hashlib": hashlib}
for text, names in ((application, policy_names), (common, common_names)):
    exec(compile(ast.Module(body=selected_nodes(text, names), type_ignores=[]), "repository_excerpt", "exec"), namespace)
actions = namespace["_POLICY_ACTIONS"]
assert actions == [str(number) for number in range(16)] + [" ".join(map(str, range(16)))]
all_rows = namespace["_reasoning_records"]()
splits = namespace["split_records"](all_rows, result["seed"])
assert namespace["_split_summary"](splits) == result["results"]["split"]
reward_table = []
for truth in range(16):
    strict = [namespace["_policy_reward"](action, truth) for action in range(17)]
    proxy = [namespace["_policy_reward"](action, truth, proxy=True) for action in range(17)]
    assert sum(strict) == 1 and strict[truth] == 1 and strict[16] == 0
    assert sum(proxy) == 2 and proxy[truth] == 1 and proxy[16] == 1
    reward_table.append({"truth": truth, "strict": strict, "proxy": proxy})

record = result["results"]["reinforce"]["weak_proxy"]
after = record["after"]
row_summaries = []
totals = {"sample_accuracy": 0, "sample_proxy_reward": 0, "enumeration_action_rate": 0}
assert len(after["samples"]) == 24
for row, expected in zip(after["samples"], splits["test"], strict=True):
    assert all(row[key] == value for key, value in expected.items())
    assert row["truth"] == row["a"] + row["b"] + row["c"]
    assert len(row["samples"]) == 16
    assert len(row["probabilities"]) == 17
    assert abs(sum(row["probabilities"]) - 1) < 1e-6
    for sample in row["samples"]:
        assert sample["generated"] == actions[sample["action_id"]]
        clean = sample["generated"].strip()
        strict = float(bool(re.fullmatch(r"-?[0-9]+", clean)) and int(clean) == row["truth"])
        proxy = float(str(row["truth"]) in sample["generated"].split())
        assert strict == sample["strict_reward"] and proxy == sample["proxy_reward"]
        totals["sample_accuracy"] += int(strict)
        totals["sample_proxy_reward"] += int(proxy)
        totals["enumeration_action_rate"] += int(sample["action_id"] == 16)
    row_summaries.append({
        "question": row["question"], "truth": row["truth"],
        "action_ids": [sample["action_id"] for sample in row["samples"]],
        "strict_sum": sum(sample["strict_reward"] for sample in row["samples"]),
        "proxy_sum": sum(sample["proxy_reward"] for sample in row["samples"]),
        "enumeration_probability": row["probabilities"][16],
    })
for key, count in totals.items():
    assert after[key] == {"numerator": count, "denominator": 384, "rate": count / 384}
assert totals == {"sample_accuracy": 0, "sample_proxy_reward": 384, "enumeration_action_rate": 384}
assert row_summaries[0]["question"] == "(0+3)+3=?" and row_summaries[0]["truth"] == 6

torch.manual_seed(42)
model = namespace["_FinitePolicy"]()
before_parameters = {name: value.clone() for name, value in model.state_dict().items()}
evaluation = namespace["_evaluate_policy"](model, splits["test"], SimpleNamespace(device="cpu"))
assert all(torch.equal(value, model.state_dict()[name]) for name, value in before_parameters.items())
assert len(evaluation["samples"]) == 24 and all(len(row["samples"]) == 16 for row in evaluation["samples"])
assert all(sample["generated"] == actions[sample["action_id"]] for row in evaluation["samples"] for sample in row["samples"])

historical_command = ["git", "show", result["revision"] + ":scripts/course_experiments/applications.py"]
local_lookup = subprocess.run(historical_command, capture_output=True)
historical_url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/" + result["revision"] + "/scripts/course_experiments/applications.py"
retrieval_command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "35", historical_url]
retrieved = subprocess.run(retrieval_command, capture_output=True, check=True)
historical = retrieved.stdout
assert digest(historical) == result["code_sha256"]["scripts/course_experiments/applications.py"]
compared = policy_names | training_names
current_ast = {getattr(node, "name", "_POLICY_ACTIONS"): ast.dump(node, include_attributes=False) for node in selected_nodes(application, compared)}
historical_ast = {getattr(node, "name", "_POLICY_ACTIONS"): ast.dump(node, include_attributes=False) for node in selected_nodes(historical.decode(), compared)}
assert current_ast == historical_ast

related_sections = {
    "reward_model": extracted_section(ROOT / "course/chapters/13.md", "13.10"),
    "conditional": extracted_section(ROOT / "course/chapters/07.md", "7.1"),
    "reward_hacking": extracted_section(ROOT / "course/chapters/13.md", "13.16"),
}
assert "訓練時，把同一題較佳卡與較差卡" in related_sections["reward_model"]
assert "A if 條件 else B" in related_sections["conditional"]
assert "策略模型利用評分弱點拿分" in related_sections["reward_hacking"]

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu"},
    "source_sha256": digest(frozen), "current_section_matches_freeze": True,
    "example_outputs": example_outputs, "parser_cases": parser_cases, "api_checks": api_checks,
    "actions": actions, "reward_table": reward_table,
    "repository_files": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in (application_path, common_path, result_path, ROOT / "course/chapters/13.md", ROOT / "course/chapters/07.md")},
    "code_excerpts": {getattr(node, "name", "_POLICY_ACTIONS"): ast.get_source_segment(application, node) for node in selected_nodes(application, compared)},
    "historical_code": {
        "local_lookup_command_argv": historical_command,
        "local_lookup_exit_code": local_lookup.returncode,
        "local_lookup_stderr": local_lookup.stderr.decode(),
        "url": historical_url, "retrieval_command_argv": retrieval_command,
        "retrieval_exit_code": retrieved.returncode, "sha256": digest(historical),
        "relevant_ast_matches_current": True,
    },
    "recorded_run": {key: result[key] for key in ("revision", "seed", "python_version", "torch_version", "device", "gpu", "evidence_status")},
    "data_split": namespace["_split_summary"](splits),
    "training": record["training"],
    "sample_count": 24 * 16, "recomputed_totals": totals,
    "reported_metrics": {key: after[key] for key in totals},
    "test_rows": row_summaries,
    "current_cpu_evaluation": {"parameter_values_unchanged": True, "sample_count": 384, "generated_strings_match_sampled_action_ids": True},
    "scope": "重算已保存逐筆樣本，執行現行評分器與抽樣器；未重新訓練，也未宣稱重現原 GPU RNG 序列。列舉必得寬鬆分數僅對正解 0 到 15 的任務成立。",
    "checks_passed": True,
}
with (DESTINATION / "verification-output.json").open("x", encoding="utf-8") as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"checks_passed": True, "samples": 384, "totals": totals}, ensure_ascii=False))
