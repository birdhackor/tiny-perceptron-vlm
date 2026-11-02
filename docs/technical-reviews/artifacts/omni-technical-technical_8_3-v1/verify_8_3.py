"""8.3 獨立技術審閱：原封程式、資料／計分重建與既有小模型 CPU 推論。"""

import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.course_experiments.behavior import _style_metrics, _style_record
from scripts.course_experiments.common import evaluate_lm, records_sha256, split_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import masked_loss
from tiny_perceptron.training import load_checkpoint

OUT = ROOT / "docs/technical-reviews/artifacts/omni-technical-technical_8_3-v1/verification-output.json"
CACHE = Path("/workspace/work/reference/omni-technical-technical_8_3-v1")
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
started = time.perf_counter()
torch.set_num_threads(2)
torch.manual_seed(42)
tok = ByteTokenizer()
checks = []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(name, observed, expected):
    checks.append({"name": name, "observed": observed, "expected": expected, "pass": observed == expected})


def execute(code):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "frozen-8.3-code-block", "exec"), {})
    return stream.getvalue()


freeze = json.loads((ROOT / FREEZE).read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "8.3")
raw = (ROOT / page["snapshot"]).read_bytes()
body = raw.decode()
check("frozen source digest", sha(raw), page["source_sha256"])
current_body = re.search(r"^## 8\.3 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S)[0]
check("current source equals packet snapshot", current_body, body)
check("no figure references", re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body), [])
code_blocks = re.findall(r"```python\n(.*?)```", body, re.S)
check("number of textbook code blocks", len(code_blocks), 1)
stdout = execute(code_blocks[0])
check("verbatim textbook stdout", stdout, "短答版 總位置 10 有效目標數 2\n積木比喻版 總位置 52 有效目標數 44\n")
exercise_code = code_blocks[0].replace('"積木比喻版": "4，像把兩組積木合在一起再數。"', '"積木比喻版": "4，共有四個"')
exercise_stdout = execute(exercise_code)
check("exercise stdout", exercise_stdout, "短答版 總位置 10 有效目標數 2\n積木比喻版 總位置 25 有效目標數 17\n")

counts = []
for answer, size, valid in [("4", 10, 2), ("4，像把兩組積木合在一起再數。", 52, 44), ("4，共有四個", 25, 17)]:
    x, y = render_chat([{"role": "user", "content": "2+2=?"}, {"role": "assistant", "content": answer}], tok)
    target = y[y != -100]
    check("answer length " + answer, len(x), size)
    check("valid targets " + answer, int((y != -100).sum()), valid)
    check("target byte IDs plus EOS " + answer, target.tolist(), tok.encode(answer) + [tok.eos_id])
    check("exactly eight ignored positions " + answer, int((y == -100).sum()), 8)
    check("label and input dtype " + answer, [str(x.dtype), str(y.dtype)], ["torch.int64", "torch.int64"])
    logits = torch.zeros(1, len(x), 264, dtype=torch.float64, requires_grad=True)
    loss = masked_loss(logits, y.unsqueeze(0))
    loss.backward()
    check("ignored positions have zero direct logit gradient " + answer, bool((logits.grad[0][y == -100] == 0).all()), True)
    check("uniform logits cross entropy equals ln(264) " + answer, abs(loss.item() - math.log(264)) < 1e-12, True)
    counts.append({"answer": answer, "answer_utf8_bytes": len(answer.encode()), "per_character_utf8_bytes": [len(c.encode()) for c in answer], "input_positions": len(x), "effective_targets": len(target), "ignored_positions": 8, "loss_at_uniform_logits": loss.item()})

saved_path = ROOT / "docs/course-experiments/results/style.json"
saved = json.loads(saved_path.read_text())
results = saved["results"]
parts = split_records(arithmetic_records(), seed=saved["seed"])
check("train/validation/test record counts", {k: len(v) for k, v in parts.items()}, {"train": 49, "validation": 8, "test": 7})
check("2+2 absent from training", any(r["a"] == 2 and r["b"] == 2 for r in parts["train"]), False)
check("test family absent from training", bool({r["family"] for r in parts["train"]} & {r["family"] for r in parts["test"]}), False)
data_checks = {}
sample_checks = {}
for style in ("concise", "vivid"):
    style_parts = {split: [_style_record(row, style, False) for row in rows] for split, rows in parts.items()}
    run = results["default_style_runs"][style]
    data_checks[style] = {}
    for split, rows in style_parts.items():
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        digest = sha(encoded)
        check(style + " " + split + " JSONL SHA", digest, run["data"][split]["sha256"])
        check(style + " " + split + " questions have no style prefix", any("style=" in row["messages"][0]["content"] for row in rows), False)
        data_checks[style][split] = {"records": len(rows), "jsonl_sha256": digest}
    check(style + " records SHA", records_sha256(style_parts["train"]), run["training"]["records_sha256"])
    lengths = [len(row["messages"][-1]["content"].encode()) + 1 for row in style_parts["train"]]
    sampler = random.Random(saved["seed"])
    draws = [length for _ in range(450) for length in sampler.choices(lengths, k=16)]
    effective = sum(draws)
    check(style + " reconstructed sampled effective targets", effective, run["training"]["effective_tokens"])
    data_checks[style]["training"] = {"steps": 450, "batch_size": 16, "sampled_record_instances": len(draws), "effective_targets": effective, "mean_effective_targets_per_record_instance": effective / len(draws), "one_digit_record_instances": sum(n == min(lengths) for n in draws), "two_digit_record_instances": sum(n == max(lengths) for n in draws), "sampler": "random.Random(42).choices(examples, k=16), independently reset for each branch"}
    for when in ("before", "after"):
        test = copy.deepcopy(run[when]["test"])
        original_rubric = copy.deepcopy(test["rubric"])
        _style_metrics(test, style_parts["test"])
        check(style + " " + when + " recomputed rubric", test["rubric"], original_rubric)
        records = []
        for row, sample in zip(style_parts["test"], test["samples"], strict=True):
            raw_ids = sample["generated_ids"]
            check(style + " " + when + " raw byte identity " + row["family"], raw_ids, tok.encode(sample["generated"]) + [tok.eos_id])
            true_answer = str(row["a"] + row["b"])
            content_correct = sample["generated"].split("，", 1)[0].strip() == true_answer
            check(style + " " + when + " content rubric " + row["family"], content_correct, sample["content_correct"])
            records.append({"question": row["messages"][0]["content"], "true_value": true_answer, "generated": sample["generated"], "content_correct": sample["content_correct"], "style_correct": sample["style_correct"], "generated_ids_sha256": sha(json.dumps(raw_ids).encode()), "valid_raw_bytes_and_eos": True})
        sample_checks[style + "_" + when] = {"rubric": test["rubric"], "samples": records}

release_path = ROOT / "docs/course-experiments/public-releases/style.json"
release = json.loads(release_path.read_text())["release"]
weight_runs = {}
base_state = None
for name, style in (("content", "concise"), ("default-concise", "concise"), ("default-vivid", "vivid")):
    manifest = next(e for e in release["public_manifest"]["files"] if e["output"] == name + ".pt")
    path = CACHE / "weights" / manifest["output"]
    weight_sha = sha(path.read_bytes())
    check(name + " checkpoint SHA", weight_sha, manifest["sha256"])
    model, payload = load_checkpoint(path, device="cpu")
    model.eval()
    check(name + " parameter count", sum(p.numel() for p in model.parameters()), 141568)
    state = model.state_dict()
    if name == "content":
        base_state = copy.deepcopy(state)
    changed_tensors = sum(not torch.equal(state[k], base_state[k]) for k in state)
    if name != "content":
        check(name + " parameters differ from base", changed_tensors > 0, True)
    style_rows = [_style_record(row, style, False) for row in parts["test"]]
    evaluated = _style_metrics(evaluate_lm(model, style_rows, mode="sft", tokens=96), style_rows)
    old_evaluation = results["content_evaluation"]["test"] if name == "content" else results["default_style_runs"][style]["after"]["test"]
    check(name + " all seven raw generations equal saved GPU run", [s["generated_ids"] for s in evaluated["samples"]], [s["generated_ids"] for s in old_evaluation["samples"]])
    weight_runs[name] = {"source_url": f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/{manifest['path']}", "local_path": str(path), "sha256": weight_sha, "bytes": path.stat().st_size, "parameters": sum(p.numel() for p in model.parameters()), "changed_state_tensors_from_base": changed_tensors, "rubric": evaluated["rubric"], "samples": [{"question": s["messages"][0]["content"], "generated": s["generated"], "content_correct": s["content_correct"], "style_correct": s["style_correct"]} for s in evaluated["samples"]]}

code_paths = ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/training.py", "scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"]
code_receipts = {}
for path in code_paths:
    current_sha = sha((ROOT / path).read_bytes())
    old_bytes = subprocess.check_output(["git", "show", saved["revision"] + ":" + path])
    check(path + " historical code hash equals experiment manifest", sha(old_bytes), saved["code_sha256"][path])
    code_receipts[path] = {"current_sha256": current_sha, "historical_revision": saved["revision"], "historical_sha256": sha(old_bytes), "exact_current_match": current_sha == sha(old_bytes)}

official_source = Path(inspect.getsourcefile(F.cross_entropy)).read_bytes()
check("installed PyTorch functional equals pinned original", sha(official_source), sha((CACHE / "pytorch-functional.py").read_bytes()))
failed = [entry["name"] for entry in checks if not entry["pass"]]
small_checks = [{k: v for k, v in entry.items() if k not in ("observed", "expected")} for entry in checks]
output = {"reviewer_task": "/root/omni_technical_coordinator/technical_8_3", "created_at": datetime.now(UTC).isoformat(), "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_8_3-v1/verify_8_3.py", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_revision": torch.version.git_version, "device": "cpu", "dtype": "int64 input/labels, float64 diagnostic logits, float32 checkpoint weights", "threads": "2", "training_updates": "0"}, "source_freeze": FREEZE, "source_sha256": sha(raw), "source_code_block_sha256": sha(code_blocks[0].encode()), "textbook_stdout": stdout, "exercise_stdout": exercise_stdout, "counts": counts, "style_result_sha256": sha(saved_path.read_bytes()), "style_result_original_environment": {key: saved[key] for key in ["revision", "seed", "device", "gpu", "torch_version", "python_version", "step_scale"]}, "release_manifest_sha256": sha(release_path.read_bytes()), "split_counts": {key: len(rows) for key, rows in parts.items()}, "data_and_denominator_reconstruction": data_checks, "saved_samples_recomputed": sample_checks, "public_checkpoint_cpu_inference": weight_runs, "primary_code_fingerprints": code_receipts, "pytorch_official_source": {"url": f"https://github.com/pytorch/pytorch/blob/{torch.version.git_version}/torch/nn/functional.py#L3478", "sha256": sha(official_source), "cross_entropy_start_line": inspect.getsourcelines(F.cross_entropy)[1]}, "check_count": len(checks), "checks": small_checks, "failed_checks": failed, "duration_seconds": time.perf_counter() - started, "scope": "Only exact textbook execution, deterministic data/count/rubric reconstruction, and inference of existing 141568-parameter checkpoints. No new training, GPU, or mature model evaluation."}
with OUT.open("x", encoding="utf-8") as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"output": str(OUT.relative_to(ROOT)), "sha256": sha(OUT.read_bytes()), "checks": len(checks), "failed_checks": failed, "textbook_stdout": stdout, "exercise_stdout": exercise_stdout, "denominators": data_checks, "cpu_rubrics": {k: v["rubric"] for k, v in weight_runs.items()}, "duration_seconds": output["duration_seconds"]}, ensure_ascii=False))
raise SystemExit(bool(failed))
