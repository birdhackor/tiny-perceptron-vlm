"""複查現行依據、重算範例與保存樣本，並保存指定工具的執行收據。"""

import ast
import contextlib
import hashlib
import html
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path.cwd()
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.posttraining import ppo_clipped_objective


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    return digest((ROOT / path).read_bytes())


def command(argv):
    result = subprocess.run(argv, capture_output=True, check=True)
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def section(text, identifier):
    match = re.search(r"^## " + re.escape(identifier) + r" .+\n", text, re.M)
    next_heading = re.search(r"^## ", text[match.end():], re.M)
    return text[match.start(): match.end() + next_heading.start() if next_heading else len(text)]


original_path = "docs/course-review-20261019/reviews/rv1019_technical_C-6/technical-C-6-original-01.json"
original_raw = (ROOT / original_path).read_bytes()
assert digest(original_raw) == "4ce1d9938016c55cbfdb16966e798a4caab69a15b15f8c95fc30469550c37025"
original = json.loads(original_raw)
assert original["issues"] == []
artifact_checks = []
for artifact in original["artifacts"]:
    actual = file_digest(artifact["path"])
    assert actual == artifact["sha256"]
    artifact_checks.append({"path": artifact["path"], "expected_sha256": artifact["sha256"], "current_sha256": actual, "unchanged": True})
prior_verification = json.loads((BASE / "verification-output.json").read_bytes())
repo_checks = []
for path, prior in prior_verification["repository_files"].items():
    actual = file_digest(path)
    same = actual == prior
    if not same:
        assert path in {"course/chapters/13.md", "course/chapters/07.md"}
    repo_checks.append({"path": path, "prior_sha256": prior, "current_sha256": actual, "unchanged": same})

frozen_path = "docs/course-review-20261019/freeze-01/sources/C.6.md"
frozen = (ROOT / frozen_path).read_bytes()
assert digest(frozen) == original["source_sha256"]
assert section((ROOT / "course/chapters/0C.md").read_text(), "C.6").encode() == frozen
assert file_digest("docs/course-review-20261019/freeze-05.json") == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"

file_diffs = []
section_checks = []
for path, identifiers in (("course/chapters/13.md", ("13.10", "13.16")), ("course/chapters/07.md", ("7.1",))):
    old = command(["git", "show", "5ef56bc756f3:" + path])
    target = command(["git", "show", "3af7f13ef959:" + path])
    current = (ROOT / path).read_text()
    assert current == target["stdout"]
    assert digest(old["stdout"].encode()) == next(s["sha256"] for s in original["sources"] if s.get("path") == path)
    diff = command(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", path])
    file_diffs.append(diff)
    for identifier in identifiers:
        before = section(old["stdout"], identifier)
        after = section(current, identifier)
        assert before == after
        section_checks.append({"path": path, "heading": after.splitlines()[0], "prior_sha256": digest(before.encode()), "current_sha256": digest(after.encode()), "unchanged": True})
assert "A if 條件 else B" in section((ROOT / "course/chapters/07.md").read_text(), "7.1")
assert "訓練時，把同一題較佳卡與較差卡" in section((ROOT / "course/chapters/13.md").read_text(), "13.10")
assert "策略模型利用評分弱點拿分" in section((ROOT / "course/chapters/13.md").read_text(), "13.16")

snippet = re.findall(r"```python\n(.*?)```", frozen.decode(), re.S)[0]
exercise = snippet.replace('"答案是4"]', '"答案是4", "0 1 2 3 4 5"]')
weak = exercise.replace("float(parsed == truth)", "float(str(truth) in text.split())")
example_outputs = {}
for name, code in (("original", snippet), ("exercise_strict", exercise), ("exercise_weak", weak)):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, name, "exec"), {})
    example_outputs[name] = stream.getvalue()
assert example_outputs == prior_verification["example_outputs"]

result_path = "docs/course-experiments/results/reasoning.json"
result = json.loads((ROOT / result_path).read_bytes())
traces = result["results"]["reinforce"]["weak_proxy"]["after"]["samples"]
actions = [str(value) for value in range(16)] + [" ".join(map(str, range(16)))]
assert len(traces) == 24
counts = {"sample_accuracy": 0, "sample_proxy_reward": 0, "enumeration_action_rate": 0}
for row in traces:
    assert row["truth"] == row["a"] + row["b"] + row["c"]
    assert len(row["samples"]) == 16
    for sample in row["samples"]:
        text = sample["generated"]
        assert text == actions[sample["action_id"]]
        strict = float(bool(re.fullmatch(r"-?[0-9]+", text)) and int(text) == row["truth"])
        proxy = float(str(row["truth"]) in text.split())
        assert strict == sample["strict_reward"] and proxy == sample["proxy_reward"]
        counts["sample_accuracy"] += int(strict)
        counts["sample_proxy_reward"] += int(proxy)
        counts["enumeration_action_rate"] += int(sample["action_id"] == 16)
assert counts == prior_verification["recomputed_totals"]

round_result = {"rounded": [round(x, 4) for x in (-2.2395448, -0.2395448)], "tie_to_even": [round(1.23445, 4), round(1.23455, 4)]}
old_log = torch.tensor([0.2, 0.2], dtype=torch.float64).log()
new_log = torch.tensor([0.4, 0.1], dtype=torch.float64).log()
advantage = torch.tensor([0.6, -0.6], dtype=torch.float64)
objective = ppo_clipped_objective(new_log, old_log, advantage)
assert torch.allclose(objective["unclipped"], torch.tensor([1.2, -0.3], dtype=torch.float64), atol=1e-12, rtol=0)
assert torch.allclose(objective["clipped"], torch.tensor([0.72, -0.48], dtype=torch.float64), atol=1e-12, rtol=0)
assert torch.allclose(objective["surrogate"], torch.tensor([0.72, -0.48], dtype=torch.float64), atol=1e-12, rtol=0)
ppo_result = {key: value.tolist() for key, value in objective.items()}

torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=10, width=8))
ids = torch.tensor([[0, 0, 1, 2, 3]])
valid = ids != 0
padding_results = []
with torch.no_grad():
    baseline = model(torch.tensor([[1, 2, 3]]))["logits"]
    for left, right in ((0, 0), (1, 2), (127, 127), (127, 0)):
        positions = torch.tensor([[left, right, 0, 1, 2]])
        actual = model(ids, valid=valid, positions=positions)["logits"]
        difference = (baseline - actual[:, 2:]).abs().max().item()
        assert tuple(actual.shape) == (1, 5, 10)
        assert torch.allclose(baseline, actual[:, 2:], atol=1e-6, rtol=0)
        padding_results.append({"pad_positions": [left, right], "real_logits_max_abs_difference": difference})
    try:
        model(ids, valid=valid, positions=torch.tensor([[-1, 0, 0, 1, 2]]))
    except IndexError as error:
        negative_position = {"exception_type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("表外位置未被拒絕")
logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 4.0, 1.0], [1.0, 2.0, 5.0]]], dtype=torch.float64)
labels = torch.tensor([[0, 1, 2]])
expected_loss = -logits.log_softmax(-1)[0, torch.arange(3), labels[0]].mean()
actual_loss = masked_loss(logits, labels)
assert torch.allclose(expected_loss, actual_loss, atol=1e-12, rtol=0)

external = []
for identifier, url, queries in (
    ("round_current", "https://docs.python.org/3.13/library/functions.html", (("round", "Return number rounded", 510),)),
    ("ppo_current", "https://raw.githubusercontent.com/openai/spinningup/master/docs/algorithms/ppo.rst", (("Key Equations", "L(s,a,\\theta_k,\\theta) =", 405), ("You Should Know", "While this kind of clipping", 250))),
):
    argv = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "35", url]
    completed = subprocess.run(argv, capture_output=True, check=True)
    raw = completed.stdout
    text = raw.decode()
    if identifier == "round_current":
        text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))
    excerpts = []
    for locator, phrase, length in queries:
        index = text.index(phrase)
        excerpts.append({"locator": locator, "excerpt": text[index:index + length]})
    external.append({"id": identifier, "url": url, "command_argv": argv, "exit_code": completed.returncode, "accessed_at_utc": datetime.now(UTC).isoformat(), "retrieved_sha256": digest(raw), "excerpts": excerpts})

origin = ROOT / "outputs/reviewer-tools/runs/20261020T095630581022Z"
section_execution = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/0C.md#C.6" --execute', "origin": str(origin.relative_to(ROOT)), "copied_files": {}}
for name in ("execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (origin / name).read_bytes()
    section_execution["copied_files"][name] = {"sha256": digest(raw), "utf8_content": raw.decode()}
assert json.loads(section_execution["copied_files"]["execution.json"]["utf8_content"])["exit_code"] == 0
assert section_execution["copied_files"]["stdout.txt"]["utf8_content"] == example_outputs["original"]
for item in artifact_checks:
    assert file_digest(item["path"]) == item["expected_sha256"]

source_paths = ("tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/posttraining.py")
source_records = {path: {"sha256": file_digest(path)} for path in source_paths}
for path, names in ((source_paths[0], {"TinyLM", "loss_sum", "masked_loss"}), (source_paths[1], {"attention_mask"}), (source_paths[2], {"ppo_clipped_objective"})):
    text = (ROOT / path).read_text()
    source_records[path]["excerpts"] = {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body if getattr(node, "name", None) in names}

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "original_report": {"path": original_path, "sha256": digest(original_raw), "issues": original["issues"]},
    "source_sha256": digest(frozen), "page_unchanged": True, "artifact_checks": artifact_checks,
    "repository_file_checks": repo_checks, "dependency_diffs": file_diffs, "cited_section_checks": section_checks,
    "example_outputs": example_outputs, "sample_count": 384, "recomputed_totals": counts,
    "supplementary_changed_sentence_verification": {
        "scope": "前置檔異動均在本頁實際引用段落之外；此處補核對其新增措辭，不視為本頁新增主張或整節重審。",
        "round": round_result, "ppo_clipping": ppo_result,
        "same_position_loss": {"expected": expected_loss.item(), "observed": actual_loss.item(), "tolerance": "atol=1e-12, rtol=0"},
        "padding": padding_results, "negative_pad_position": negative_position,
        "clipping_limit": "目標函式移除不當增大的誘因，不保證每次更新的機率變化被硬性限制。",
    },
    "external_sources": external, "supplementary_repository_code": source_records,
    "section_execution": section_execution, "checks_passed": True,
    "limits": "未重跑 GPU 訓練或載入 checkpoint；未重審前置整節；未測易讀性。"
}
with (BASE / "recheck-current-01.json").open("x", encoding="utf-8") as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"checks_passed": True, "claims_retained": len(original["claims"]), "cited_sections_unchanged": len(section_checks), "samples": 384, "totals": counts}, ensure_ascii=False))
