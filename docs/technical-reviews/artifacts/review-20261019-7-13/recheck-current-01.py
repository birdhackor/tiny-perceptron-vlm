"""複查教材依賴、保全原始證據，並重跑本節示例。"""
import ast
import contextlib
import hashlib
import html
import io
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.training import choose_device

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def run(argv):
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120)
    return {"command_argv": argv, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def code_run(code):
    namespace = {}
    output = io.StringIO()
    error = None
    with contextlib.redirect_stdout(output):
        try:
            exec(compile(code, "section-example", "exec"), namespace)
        except Exception as e:
            error = type(e).__name__
    return {"stdout": output.getvalue(), "exception": error, "scores": namespace.get("scores")}

original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_7-13/technical-7-13-original-01.json"
original_raw = original_path.read_bytes()
assert sha(original_raw) == "4a542edb1ed9a81125bd1057e3c08a51f9ef390fe49e84f8ba00c3f971dcc62d"
original = json.loads(original_raw)
assert original["issues"] == []
old_artifacts = []
for artifact in original["artifacts"]:
    raw = (ROOT / artifact["path"]).read_bytes()
    actual = sha(raw)
    assert actual == artifact["sha256"]
    old_artifacts.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
old_check = A / "self-check.txt"
old_artifacts.append({"path": str(old_check.relative_to(ROOT)), "sha256": sha(old_check.read_bytes()), "unchanged": True})
source_checks = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = sha((ROOT / source["path"]).read_bytes())
    source_checks.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"],
                          "current_sha256": actual, "unchanged": actual == source["sha256"]})
assert [s["path"] for s in source_checks if not s["unchanged"]] == ["course/training.md"]
training = ROOT / "course/training.md"
assert sha(training.read_bytes()) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
diff = run(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
old_training = run(["git", "show", "5ef56bc756f3:course/training.md"])
new_training = run(["git", "show", "6a7aa1de5227:course/training.md"])
assert all(c["exit_code"] == 0 for c in [diff, old_training, new_training])
assert sha(old_training["stdout"].encode()) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
assert new_training["stdout"].encode() == training.read_bytes()
recipe_pattern = r'<details id="fixed-sft-recipe" markdown="1">.*?</details>'
old_recipe = re.search(recipe_pattern, old_training["stdout"], re.S)[0]
current_recipe = re.search(recipe_pattern, training.read_text(), re.S)[0]
assert old_recipe == current_recipe
snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/7.13.md"
chapter = (ROOT / "course/chapters/07.md").read_bytes()
start = re.search(rb"(?m)^## 7\.13 ", chapter).start()
end_match = re.search(rb"(?m)^## ", chapter[start + 1:])
end = start + 1 + end_match.start() if end_match else len(chapter)
section = chapter[start:end]
assert section == snapshot.read_bytes()
assert sha(section) == original["source_sha256"]
freeze = ROOT / "docs/course-review-20261019/freeze-06.json"
assert sha(freeze.read_bytes()) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
section_command = run([str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/07.md#7.13", "--execute"])
assert section_command["exit_code"] == 0
run_summary = json.loads(section_command["stdout"])
assert run_summary["source_sha256"] == original["source_sha256"]
assert run_summary["svg_references"] == 0
run_dir = Path(run_summary["artifact_directory"])
copied_files = []
for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"]:
    raw = (run_dir / name).read_bytes()
    copied_files.append({"name": name, "sha256": sha(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()})
code = re.search(r"```python\n(.*?)```", section.decode(), re.S)[1]
correct = code.replace("3}", "2}")
fixed = correct.replace('and not scores["1+1內容正確"]', 'and scores["1+1內容正確"]')
renamed = correct.replace('reply = \'{"answer": 2}\'', 'reply = \'{"result": 2}\'')
variants = {"original": code_run(code), "answer_two_original_assertion": code_run(correct),
            "answer_two_both_true_assertion": code_run(fixed), "renamed_result": code_run(renamed)}
previous_verification = json.loads((A / "verification.json").read_text())
assert variants == previous_verification["example_variants"]
record = json.loads((ROOT / "docs/course-experiments/results/sft.json").read_bytes())
tok = ByteTokenizer()
test = record["results"]["after"]["test"]
recomputed = []
for sample in test["samples"]:
    ids = sample["generated_ids"]
    raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
    exact = raw == tok.encode(sample["expected"])
    eos = tok.eos_id in ids
    assert exact is sample["exact"] and eos is sample["eos"]
    recomputed.append({"expected": sample["expected"], "exact": exact, "eos": eos})
assert sum(s["exact"] for s in recomputed) == test["matches"] == 5
assert len(recomputed) == test["records"] == 10
assert round(record["elapsed_seconds"], 1) == 31.7
assert record["results"]["training"]["steps"] == 900
assert record["results"]["pretrain_then_sft"]["after_sft"]["test"]["matches"] == 7

files = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/quantize.py", "scripts/prepare_data.py",
         "scripts/fetch_training_assets.py", "scripts/course_experiments/run.py", "tiny_perceptron/training.py"]
cli_commands = []
source_inspection = []
for name in files:
    body = (ROOT / name).read_text()
    tree = ast.parse(body)
    options = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            if "--device" in flags:
                default = next((ast.literal_eval(k.value) for k in node.keywords if k.arg == "default"), None)
                options.append({"flags": flags, "default": default, "source": ast.get_source_segment(body, node)})
    source_inspection.append({"path": name, "sha256": sha((ROOT / name).read_bytes()), "device_options": options,
                              "device_calls": [line for line in body.splitlines() if any(x in line for x in ("choose_device(", "load_checkpoint(", 'map_location="cpu"', 'return torch.device(', 'torch.cuda.is_available()', 'torch.backends.mps.is_available()'))]})
    if name.startswith("scripts/"):
        result = run([str(ROOT / ".venv/bin/python"), name, "--help"])
        assert result["exit_code"] == 0
        cli_commands.append(result)
lookup = {s["path"]: s for s in source_inspection}
for name in ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py"]:
    assert lookup[name]["device_options"][0]["default"] == "auto"
assert lookup["scripts/course_experiments/run.py"]["device_options"][0]["default"] == "cpu"
for name in ["scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py"]:
    assert lookup[name]["device_options"] == []
    help_result = next(c for c in cli_commands if c["command_argv"][1] == name)
    assert "--device" not in help_result["stdout"]
selection = []
for cuda, mps, requested, expected in [(True, True, "auto", "cuda"), (False, True, "auto", "mps"),
                                        (False, False, "auto", "cpu"), (True, True, "cpu", "cpu")]:
    with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
        observed = str(choose_device(requested))
    assert observed == expected
    selection.append({"mock_cuda_available": cuda, "mock_mps_available": mps, "requested": requested, "expected": expected, "observed": observed})

url = "https://docs.pytorch.org/docs/2.14/notes/randomness.html"
curl = run(["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url])
assert curl["exit_code"] == 0
page = curl["stdout"]
plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
excerpt = "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
assert excerpt in plain
pytorch = {"url": url, "version": "PyTorch 2.14 documentation", "title": html.unescape(re.search(r"<title>(.*?)</title>", page, re.S)[1]),
           "accessed_at": datetime.now(UTC).isoformat(), "original_sha256": sha(page.encode()), "original_bytes": len(page.encode()),
           "command_argv": curl["command_argv"], "exit_code": curl["exit_code"], "stderr": curl["stderr"],
           "checked_original": True, "excerpt": excerpt,
           "inspection_note": "實讀Reproducibility開頭、CPU/GPU同seed仍不保證重現、亂數控制及非確定運算說明。只支持跨環境結果可能不同，不證明任何特定學生執行已發生差異。全文不存檔。"}
for old in old_artifacts:
    assert sha((ROOT / old["path"]).read_bytes()) == old["sha256"]
assert sha(original_path.read_bytes()) == sha(original_raw)
evidence = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-13/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": sha(original_raw), "issues": original["issues"]},
    "source_sha256": sha(section), "source_freeze_sha256": sha(freeze.read_bytes()),
    "previous_source_hash_checks": source_checks, "previous_artifact_hash_checks": old_artifacts,
    "training_diff": diff, "training_versions": {"old_revision": "5ef56bc756f3", "new_revision": "6a7aa1de5227", "old_sha256": sha(old_training["stdout"].encode()), "current_sha256": sha(training.read_bytes())},
    "fixed_sft_recipe": {"unchanged": old_recipe == current_recipe, "sha256": sha(current_recipe.encode()), "current_text": current_recipe},
    "section_execution": {"command_receipt": section_command, "copied_from": str(run_dir.relative_to(ROOT)), "copied_files": copied_files},
    "example_variants": variants,
    "sample_recomputation": {"samples": recomputed, "matches": 5, "records": 10, "eos_count": sum(s["eos"] for s in recomputed), "pretrain_sft_matches": 7, "steps": 900, "elapsed_rounded": 31.7},
    "cli_commands": cli_commands, "source_inspection": source_inspection, "mocked_device_selection": selection,
    "pytorch_original": pytorch,
    "scope": "不重跑訓練、量化匯出或實際下載；裝置選擇以mock可用性測試，不代表在實體CUDA/MPS裝置執行。舊證據中training.md的hash是歷史收據，此次以current_sha256及未變的fixed-sft-recipe作現行依據。",
    "result": "原報告及舊artifact保持原樣；所有頁面主張仍有支持，固定SFT配方未改動；新裝置說明與CLI／程式一致。"
}
with (A / "recheck-evidence-current-01.json").open("x", encoding="utf-8") as target:
    json.dump(evidence, target, ensure_ascii=False, indent=2)
    target.write("\n")
print(json.dumps({"result": evidence["result"], "section_exit_code": section_command["exit_code"], "changed_repo_sources": [s["path"] for s in source_checks if not s["unchanged"]], "test": evidence["sample_recomputation"]}, ensure_ascii=False))
