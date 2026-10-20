"""核對配方算例、保存的逐題結果及確定性資料預算；不訓練模型。"""

import contextlib
import hashlib
import inspect
import io
import json
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import fit_lm, records_sha256, split_records, text_examples
from scripts.course_experiments.run import experiment_spec
from scripts.course_experiments.text import arithmetic_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    (Path(__file__).parent / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


command = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-16/audit.py"
environment = {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}
run_dir = ROOT / "outputs/reviewer-tools/runs/20261020T052654812907Z"
copies = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run_dir / name).read_bytes()
    copies[name] = {"sha256": sha(raw), "utf8_content": raw.decode("utf-8")}
assert json.loads(copies["execution.json"]["utf8_content"])["exit_code"] == 0
save("section-execution.json", {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/07.md#7.16" --execute',
    "launcher_stdout": '{"artifact_directory": "/Users/yclin/project/tiny-perceptron-vlm/outputs/reviewer-tools/runs/20261020T052654812907Z", "source_sha256": "d9c74585a360e3a4f41e67902b8c033ac128d568f9713a72e7f01cc9bc47878e", "python_fences": 1, "svg_references": 0, "execution_exit_code": 0}\n',
    "environment": environment,
    "copied_files": copies,
})

ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(copies["fence-1.py"]["utf8_content"], ns)
a, b = ns["a"], ns["b"]


def recipe_stats(rows):
    return {
        "sources": [r["source"] for r in rows],
        "records": len(rows),
        "effective_targets": sum(r["answer_tokens"] for r in rows),
        "targets_by_source": {
            source: sum(r["answer_tokens"] for r in rows if r["source"] == source)
            for source in ("A", "B")
        },
        "distinct_texts": len({r["text"] for r in rows}),
    }


recipes = {"original": recipe_stats(a + b), "b_only_four": recipe_stats(b * 2),
           "b_only_two": recipe_stats(b), "exercise": recipe_stats([a[0], b[0], b[1], b[0]])}
for row in b:
    row["answer_tokens"] = 4
recipes["longer_b_exercise"] = recipe_stats([a[0], b[0], b[1], b[0]])
assert recipes["original"]["effective_targets"] == recipes["exercise"]["effective_targets"] == 8
assert recipes["longer_b_exercise"]["targets_by_source"] == {"A": 2, "B": 12}

result_path = ROOT / "docs/course-experiments/results/sft_ablation.json"
recorded = json.loads(result_path.read_bytes())
results = recorded["results"]
parts = {
    "A_attributes": split_records(generate_records("attributes-sft"), seed=recorded["seed"]),
    "B_arithmetic": split_records(arithmetic_records(), seed=recorded["seed"]),
}
splits = {}
for task, dataset in parts.items():
    saved_task = "attributes" if task == "A_attributes" else "arithmetic"
    splits[task] = {}
    for split, rows in dataset.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        got = {"records": len(rows), "families": len({r["family"] for r in rows}), "sha256": sha(raw)}
        expected = results["data"][saved_task][split]
        assert all(got[k] == expected[k] for k in got), "重建資料與保存的資料指紋不同"
        splits[task][split] = got
    family_sets = [{r["family"] for r in dataset[s]} for s in ("train", "validation", "test")]
    assert not any(family_sets[i] & family_sets[j] for i in range(3) for j in range(i + 1, 3))

training = {}
for branch in ("b-only", "replay"):
    rows = parts["B_arithmetic"]["train"]
    if branch == "replay":
        rows = parts["A_attributes"]["train"] + rows
    expected = results["runs"][branch]["training"]
    assert records_sha256(rows) == expected["records_sha256"]
    examples = text_examples(rows, "sft", 128)
    sampler = random.Random(recorded["seed"])
    targets, input_tokens, padded_slots, source_a, empty_a_batches = 0, 0, 0, 0, 0
    for _ in range(expected["steps"]):
        indices = sampler.choices(range(len(examples)), k=16)
        batch = [examples[i] for i in indices]
        targets += sum(int((y != IGNORE).sum()) for x, y in batch)
        input_tokens += sum(len(x) for x, y in batch)
        padded_slots += 16 * max(len(x) for x, y in batch)
        batch_a = sum(i < len(parts["A_attributes"]["train"]) for i in indices) if branch == "replay" else 0
        source_a += batch_a
        empty_a_batches += int(batch_a == 0)
    assert targets == expected["effective_tokens"], "重建抽樣的有效目標總量不符"
    training[branch] = {
        "steps": expected["steps"], "batch_size": 16, "effective_tokens": targets,
        "records": len(rows), "records_sha256": records_sha256(rows),
        "input_tokens": input_tokens, "padded_input_slots": padded_slots,
        "sampled_A_records": source_a, "sampled_B_records": 500 * 16 - source_a,
        "batches_without_A": empty_a_batches,
    }

tok = ByteTokenizer()
evaluations = {}
for state, state_result in (("before", results["before"]),
                            ("b-only", results["runs"]["b-only"]),
                            ("replay", results["runs"]["replay"])):
    evaluations[state] = {}
    for task in parts:
        evaluations[state][task] = {}
        for split in ("validation", "test"):
            ev = state_result[task][split]
            rows = parts[task][split]
            assert len(ev["samples"]) == ev["records"] == len(rows)
            exacts, eos, checks = 0, 0, []
            for sample, row in zip(ev["samples"], rows, strict=True):
                assert sample["messages"] == row["messages"][:-1]
                assert sample["expected"] == row["messages"][-1]["content"]
                ids = sample["generated_ids"]
                raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
                exact = raw == tok.encode(sample["expected"])
                ended = tok.eos_id in ids
                assert exact == sample["exact"] and ended == sample["eos"]
                assert tok.decode(raw) == sample["generated"]
                exacts += exact
                eos += ended
                checks.append({"expected": sample["expected"], "generated_ids": ids,
                               "exact_recomputed": exact, "eos_recomputed": ended})
            assert exacts == ev["matches"] and exacts / len(rows) == ev["exact_match"]
            assert eos / len(rows) == ev["eos_rate"] == 1.0
            evaluations[state][task][split] = {
                "matches": exacts, "records": len(rows), "eos_count": eos, "samples": checks,
            }

source_paths = [
    "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py", "scripts/prepare_data.py", "tiny_perceptron/data.py",
    "tiny_perceptron/model.py", "course/training.md", "docs/course-experiments/plan.json",
]
code_checks = {}
for path in source_paths:
    current = (ROOT / path).read_bytes()
    lookup = subprocess.run(["git", "show", f"{recorded['revision']}:{path}"], cwd=ROOT,
                            capture_output=True)
    prior = lookup.stdout if lookup.returncode == 0 else None
    record_hash = recorded["code_sha256"].get(path)
    if record_hash and path != "scripts/course_experiments/run.py":
        assert sha(current) == record_hash, "目前程式與實驗紀錄的程式指紋不同"
    code_checks[path] = {"current_sha256": sha(current), "recorded_revision_sha256": sha(prior) if prior else None,
                         "recorded_code_sha256": record_hash,
                         "current_equals_recorded_code_hash": sha(current) == record_hash if record_hash else None,
                         "historical_lookup_command": lookup.args, "historical_lookup_exit_code": lookup.returncode,
                         "historical_lookup_stderr": lookup.stderr.decode()}

runner_path = "scripts/course_experiments/run.py"
runner_version = "918652bb9dcdd97b0397d797403a26de2f996c99"
runner_command = ["git", "show", f"{runner_version}:{runner_path}"]
runner_raw = subprocess.run(runner_command, cwd=ROOT, check=True, capture_output=True).stdout
assert sha(runner_raw) == recorded["code_sha256"][runner_path]
runner_text = runner_raw.decode()
code_checks[runner_path]["matching_historical_source"] = {
    "command_argv": runner_command, "sha256": sha(runner_raw), "exit_code": 0,
    "timing_source_excerpt": runner_text[runner_text.index("    started = time.perf_counter()"):
                                       runner_text.index("    artifacts = [")],
}

cli = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"],
                     check=True, capture_output=True, text=True, cwd=ROOT)
save("verification.json", {
    "command": command, "environment": environment, "result": "全部斷言通過；未進行模型訓練或 GPU 計時。",
    "recipes": recipes, "splits": splits, "training_reconstruction": training,
    "effective_target_ratio": training["replay"]["effective_tokens"] / training["b-only"]["effective_tokens"],
    "evaluations_recomputed_from_saved_ids": evaluations,
    "recorded_environment": {k: recorded[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status")},
    "elapsed_seconds_rounded": round(recorded["elapsed_seconds"], 1),
    "result_path": result_path.relative_to(ROOT).as_posix(), "result_sha256": sha(result_path.read_bytes()),
    "code_checks": code_checks, "fit_lm_signature": str(inspect.signature(fit_lm)),
    "experiment_spec": experiment_spec("sft_ablation"), "cli_help_stdout": cli.stdout,
    "limitations": "逐題成績依保存的原始生成 ID 重算，沒有重新產生模型回答。L4 裝置與歷史耗時依原始結果及計時實作核對，沒有量測本機 CUDA。",
})

abstract_url = "https://arxiv.org/abs/1902.10486v4"
pdf_url = "https://arxiv.org/pdf/1902.10486v4"
retrievals = {}
for url in (abstract_url, pdf_url):
    proc = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", url],
                          check=True, capture_output=True)
    retrievals[url] = {"command_argv": proc.args, "exit_code": proc.returncode,
                       "bytes": len(proc.stdout), "sha256": sha(proc.stdout)}
    if url == abstract_url:
        html = proc.stdout.decode()
        assert "4 Jun 2019" in html and "jointly trains" in html
    else:
        assert proc.stdout.startswith(b"%PDF"), "原始論文下載內容不是 PDF"
save("original-source.json", {
    "accessed_on": datetime.now(timezone.utc).date().isoformat(), "retrievals": retrievals,
    "version": "arXiv:1902.10486v4，2019-06-04",
    "title": "On Tiny Episodic Memories in Continual Learning",
    "read_locator": "原始摘要；PDF 的 Experience Replay 與 Algorithm 1（紙本頁 3–4）；PDF 文字另以瀏覽工具檢視。",
    "minimal_excerpt": "θ ← SGD(Bn ∪ BM, θ, lr)",
    "inspection_note": "Algorithm 1 抽取當前任務及記憶的例子，以當前參數 θ 對合併 batch 做一次 SGD；Experience Replay 說明以兩份梯度的平均更新。不是儲存歷史梯度再套用。論文另指出只對新任務微調會沿用前一任務的參數。",
    "scope": "只支持回放的定義與當前參數重算機制；不把論文的基準成績或成本結論移用到專案的小模型。第三方全文僅在程序記憶體與瀏覽工具中讀取，未保存到 repo。",
})
print(json.dumps({"result": "全部斷言通過", "effective_tokens": {k: v["effective_tokens"] for k, v in training.items()},
                  "A_test_matches": {k: v["A_attributes"]["test"]["matches"] for k, v in evaluations.items()},
                  "B_test_matches": {k: v["B_arithmetic"]["test"]["matches"] for k, v in evaluations.items()},
                  "ratio": training["replay"]["effective_tokens"] / training["b-only"]["effective_tokens"]}, ensure_ascii=False))
