import codecs
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch
from torch import nn

from scripts import fetch_training_assets
from scripts.course_experiments import run
from scripts.course_experiments.common import evaluate_lm, records_sha256, split_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, generate, masked_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "argv": argv,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def original(url, version, selections, start_at=None):
    argv = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    raw = subprocess.check_output(argv)
    lines = raw.decode("utf-8").splitlines()
    if start_at is not None:
        lines = lines[next(i for i, line in enumerate(lines) if start_at in line):]
    excerpts = []
    for needle, count in selections:
        index = next(i for i, line in enumerate(lines) if needle in line)
        excerpts.append({
            "locator": needle,
            "text": "\n".join(lines[index:index + count]),
        })
    return {
        "url": url,
        "version": version,
        "accessed_on": "2026-10-20",
        "command_argv": argv,
        "original_sha256": sha(raw),
        "original_bytes": len(raw),
        "checked_original": True,
        "excerpts": excerpts,
    }


class ScriptedLM(nn.Module):
    def __init__(self, tokens, prompt_length):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_length=128)
        self.tokens = tokens
        self.prompt_length = prompt_length

    def forward(self, ids, **kwargs):
        index = max(0, min(ids.shape[1] - self.prompt_length, len(self.tokens) - 1))
        logits = torch.full((*ids.shape, 264), -20.0)
        logits[..., self.tokens[index]] = 20.0
        return {"logits": logits, "cache": None}


tok = ByteTokenizer()
torch.set_num_threads(1)
messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
x, y = render_chat(messages)
logits = torch.zeros((1, len(y), tok.vocab_size), requires_grad=True)
loss = masked_loss(logits, y.unsqueeze(0))
loss.backward()
assert y[-1].item() == tok.eos_id == 2
assert y[y != IGNORE].tolist() == [73, 2]
assert logits.grad[0, -1, tok.eos_id].item() < 0
assert torch.count_nonzero(logits.grad[0, y == IGNORE]).item() == 0


def visible(tokens, budget):
    result = []
    for token in tokens[:budget]:
        if token == tok.eos_id:
            break
        result.append(token)
    return tok.decode(result)


original_tokens = [tok.encode("A")[0], tok.eos_id, tok.encode("B")[0]]
assert original_tokens == [73, 2, 74]
assert visible(original_tokens, 2) == "A"
assert visible(tok.encode("ABCD"), 2) == "AB"
assert tok.decode(list(range(8)) + tok.encode("A") + [tok.eos_id]) == "A"

prompt = [tok.bos_id, tok.user_id] + tok.encode("Q") + [tok.eos_id, tok.assistant_id]
prompt_tensor = torch.tensor([prompt])
model = ScriptedLM(original_tokens, len(prompt))
with_eos = generate(model, prompt_tensor, max_new_tokens=5)[0].tolist()
assert with_eos == prompt + [73, 2]
assert model.training is True
with_budget = generate(ScriptedLM(tok.encode("ABCD"), len(prompt)), prompt_tensor, 2)[0].tolist()
assert with_budget == prompt + [73, 74]
actual_shape = list(TinyLM(ModelConfig())(prompt_tensor)["logits"].shape)
assert actual_shape == [1, len(prompt), tok.vocab_size]

raw = "貓".encode("utf-8")
decoder = codecs.getincrementaldecoder("utf-8")()
partial = decoder.decode(raw[:2])
pending = list(decoder.getstate()[0])
complete = decoder.decode(raw[2:])
assert partial == "" and pending == [232, 178] and complete == "貓"

period_with_eos = generate(ScriptedLM(tok.encode("A.") + [2], len(prompt)), prompt_tensor, 8)[0].tolist()
period_at_budget = generate(ScriptedLM(tok.encode("A.B"), len(prompt)), prompt_tensor, 2)[0].tolist()
assert tok.decode(period_with_eos[len(prompt):]) == tok.decode(period_at_budget[len(prompt):]) == "A."
assert period_with_eos[-1] == 2 and period_at_budget[-1] != 2

wrong_record = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
wrong = evaluate_lm(ScriptedLM([74, 2], len(prompt)), [wrong_record], mode="sft", tokens=8)
no_eos = evaluate_lm(ScriptedLM([73, 74], len(prompt)), [wrong_record], mode="sft", tokens=1)
assert wrong["samples"][0]["generated_ids"] == [74, 2]
assert wrong["samples"][0]["exact"] is False and wrong["samples"][0]["eos"] is True
assert no_eos["samples"][0]["exact"] is True and no_eos["samples"][0]["eos"] is False

record_path = "docs/course-experiments/results/sft.json"
record = json.loads((ROOT / record_path).read_bytes())
parts = split_records(generate_records("attributes-sft"), seed=record["seed"])
assert {k: len(v) for k, v in parts.items()} == {"train": 45, "validation": 5, "test": 10}
assert records_sha256(parts["train"]) == record["results"]["training"]["records_sha256"]
summary = {}
for split in ("validation", "test"):
    report = record["results"]["after"][split]
    rows = report["samples"]
    checked = []
    for sample, row in zip(rows, parts[split], strict=True):
        ids = sample["generated_ids"]
        assert sample["messages"] == row["messages"][:-1]
        assert sample["expected"] == row["messages"][-1]["content"]
        before_eos = ids[:ids.index(2)] if 2 in ids else ids
        assert sample["generated"] == tok.decode(before_eos)
        assert sample["eos"] == (2 in ids)
        assert sample["exact"] == (before_eos == tok.encode(sample["expected"]))
        checked.append({k: sample[k] for k in ("expected", "generated", "generated_ids", "exact", "eos")})
    exact = sum(row["exact"] for row in rows)
    ended = sum(row["eos"] for row in rows)
    assert report["records"] == len(rows) == len(parts[split])
    assert report["matches"] == exact
    assert report["eos_rate"] == ended / len(rows)
    assert report["exact_match"] == exact / len(rows)
    summary[split] = {"records": len(rows), "exact": exact, "eos": ended, "errors": len(rows) - exact, "samples": checked}
first = summary["validation"]["samples"][0]
assert first == {"expected": "circle", "generated": "cire", "generated_ids": [107, 113, 122, 109, 2], "exact": False, "eos": True}
assert record["results"]["training"]["steps"] == 900
assert record["gpu"] == "NVIDIA L4"
assert f'{record["elapsed_seconds"]:.1f}' == "31.7"

cli = [command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--list-assets", "sft"]),
       command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"]),
       command([str(ROOT / ".venv/bin/python"), "scripts/fetch_training_assets.py", "--help"])]
assert all(item["exit_code"] == 0 for item in cli)
assert cli[0]["stdout"].strip() == "assets/training/ultrachat-sft-v1.tar.gz"
captured_run = {}


def intercepted_execute(*args, **kwargs):
    captured_run.update(args=[str(arg) for arg in args], kwargs=kwargs)
    return {"experiment_id": "sft", "elapsed_seconds": 0, "evidence_status": "intercepted", "artifacts": []}


with patch.object(sys, "argv", ["run", "--experiment", "sft", "--device", "cuda"]), patch.object(run, "execute", intercepted_execute), contextlib.redirect_stdout(io.StringIO()):
    run.main()
assert captured_run["args"][0:2] == ["sft", "cuda"]
assert captured_run["args"][2] == str(ROOT / "outputs/course-experiments/course-v1/sft")
assert captured_run["kwargs"] == {"step_scale": 1.0}
captured_fetch = {}


def intercepted_unpack(archive, asset, output):
    captured_fetch.update(archive=str(archive), asset=asset["id"], output=str(output))
    return {"asset": asset["id"], "intercepted": True}


with patch.object(sys, "argv", ["fetch_training_assets", "--asset", "ultrachat-sft"]), patch.object(fetch_training_assets, "unpack_asset", intercepted_unpack), patch.object(fetch_training_assets.subprocess, "run") as lfs, contextlib.redirect_stdout(io.StringIO()):
    fetch_training_assets.main()
    captured_fetch["lfs_calls"] = [list(call.args[0]) for call in lfs.call_args_list]
assert captured_fetch["asset"] == "ultrachat-sft"
assert captured_fetch["archive"] == str(ROOT / "assets/training/ultrachat-sft-v1.tar.gz")
assert captured_fetch["output"] == str(ROOT / "data/training")

originals = {
    "pytorch_seq2seq": original(
        "https://raw.githubusercontent.com/pytorch/tutorials/c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6/intermediate_source/seq2seq_translation_tutorial.py",
        "pytorch/tutorials commit c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6",
        [("for idx, (inp, tgt) in enumerate(pairs):", 7), ("loss = criterion(", 5), ("# predicts the EOS token we stop there.", 2)],
    ),
    "hf_stopping": original(
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/stopping_criteria.py",
        "transformers v4.57.1",
        [("class EosTokenCriteria(", 22)],
    ),
    "hf_config": original(
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/configuration_utils.py",
        "transformers v4.57.1",
        [("        max_new_tokens (`int`", 2)],
    ),
    "hf_generation": original(
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py",
        "transformers v4.57.1",
        [("            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)", 1),
         ("            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)", 2),
         ("            return input_ids", 1)],
        start_at="    def _sample(",
    ),
    "python_codecs": original(
        "https://raw.githubusercontent.com/python/cpython/v3.13.9/Doc/library/codecs.rst",
        "CPython v3.13.9",
        [("   .. method:: decode(object, final=False)", 16)],
    ),
}

paths = [record_path, "tiny_perceptron/data.py", "tiny_perceptron/model.py",
         "scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
         "scripts/course_experiments/run.py", "scripts/fetch_training_assets.py",
         "scripts/prepare_data.py", "assets/training/manifest.json",
         "docs/course-review-20261019/freeze-04.json",
         "docs/course-review-20261019/freeze-01/sources/7.9.md"]
hashes = {name: sha((ROOT / name).read_bytes()) for name in paths}
code_match = {name: hashes[name] == expected for name, expected in record["code_sha256"].items() if name in hashes}
assert all(code_match[name] for name in ("tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py"))
runner_path = "scripts/course_experiments/run.py"
history_argv = ["git", "rev-list", "HEAD", "--", runner_path]
commits = subprocess.check_output(history_argv, cwd=ROOT, text=True).splitlines()
runner = None
for commit in commits:
    argv = ["git", "show", commit + ":" + runner_path]
    old = subprocess.check_output(argv, cwd=ROOT)
    if sha(old) == record["code_sha256"][runner_path]:
        lines = old.decode().splitlines()
        excerpts = []
        for needle, count in (("    started = time.perf_counter()", 8),
                              ("    elapsed = time.perf_counter() - started", 1),
                              ('        "timing_scope":', 1)):
            index = next(i for i, line in enumerate(lines) if needle in line)
            excerpts.append({"locator": needle, "text": "\n".join(lines[index:index + count])})
        runner = {"path": runner_path, "commit": commit, "sha256": sha(old),
                  "history_argv": history_argv, "candidate_count": len(commits),
                  "command_argv": argv, "excerpts": excerpts}
        break
assert runner is not None

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-9/verify_facts.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "input_sha256": hashes,
    "render_chat": {"x": x.tolist(), "y": y.tolist(), "effective_targets": y[y != IGNORE].tolist(), "eos_gradient": logits.grad[0, -1, 2].item()},
    "stopping": {"scripted_tokens": original_tokens, "original_visible": visible(original_tokens, 2), "exercise_visible": visible(tok.encode("ABCD"), 2), "prompt_with_user_eos": prompt, "with_eos": with_eos, "with_budget": with_budget, "logits_shape": actual_shape, "decode_all_specials_and_A": tok.decode(list(range(8)) + [73]), "period_eos_raw": period_with_eos[len(prompt):], "period_budget_raw": period_at_budget[len(prompt):], "same_visible_period": "A."},
    "utf8": {"raw_bytes": list(raw), "partial_visible": partial, "pending_bytes": pending, "completed_visible": complete, "direct_partial_decode": tok.decode(tok.encode("貓")[:2])},
    "evaluation_probe": {"wrong_but_eos": wrong["samples"], "correct_content_without_eos": no_eos["samples"], "scope": "預先指定 logits 的機制測試，不是模型訓練成果。"},
    "record_check": {"path": record_path, "sha256": hashes[record_path], "training_steps": 900, "training_records": 45, "seed": record["seed"], "generation_budget": 32, "training_records_sha256": records_sha256(parts["train"]), "summary": summary, "elapsed_seconds": record["elapsed_seconds"], "rounded_elapsed_seconds": f'{record["elapsed_seconds"]:.1f}', "timing_scope": record["timing_scope"], "gpu": record["gpu"], "python": record["python_version"], "torch": record["torch_version"], "recorded_revision": record["revision"], "code_hash_match": code_match, "scope": "重新計算已保存紀錄與目前資料生成器，未重跑 GPU 訓練或驗證硬體計時。"},
    "cli": {"commands": cli, "parsed_run_with_execute_intercepted": captured_run, "parsed_fetch_with_side_effects_intercepted": captured_fetch, "scope": "實跑 help、資產列舉與參數路由；下載、解包及訓練副作用由替身攔截，未執行。"},
    "original_sources": originals,
    "recorded_runner": runner,
    "historical_runner_lookup": command(["git", "show", record["revision"] + ":scripts/course_experiments/run.py"]),
    "result": "所有確定性斷言與既有實驗紀錄核對通過。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
