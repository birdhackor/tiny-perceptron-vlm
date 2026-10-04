"""Fresh C.1 audit: rebuild hashed data and replay CPU loss, never retrain schedules."""

# Imports below the repository path bootstrap permit replay as a direct script.
# ruff: noqa: E402

import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.applications import (
    _conversation,
    _nll_examples,
    _reasoning_records,
    _reasoning_sft,
    _sft_nll,
    _split_summary,
    _verify_reasoning,
)
from scripts.course_experiments.common import Context, new_lm, split_records, text_examples, write_json
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import loss_sum
from tiny_perceptron.training import load_checkpoint

ARTIFACTS = Path(__file__).parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    formal = json.loads((ROOT / "docs/course-experiments/results/reasoning.json").read_text())
    result = formal["results"]
    manifest = {item["path"]: item for item in formal["artifacts"]}
    receipts = {"command": " .venv/bin/python " + str(Path(__file__).relative_to(ROOT))}
    receipts["environment"] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "threads": "2",
        "formal_environment": {key: str(formal[key]) for key in ("device", "gpu", "torch_version", "python_version")},
    }
    receipts["source_code_versions"] = {}
    for path in (
        "scripts/course_experiments/applications.py",
        "scripts/course_experiments/common.py",
        "scripts/course_experiments/run.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/model.py",
        "tiny_perceptron/training.py",
    ):
        actual = digest(ROOT / path)
        assert actual == formal["code_sha256"][path]
        receipts["source_code_versions"][path] = actual

    tok = ByteTokenizer()
    short, long = "5", "從2開始數三次：3、4、5，所以是5。"
    modified = long[:-2] + "6。"
    lengths = [len(tok.encode(text)) for text in (short, long, modified)]
    assert lengths == [1, 47, 47]
    receipts["byte_example"] = {"texts": [short, long, modified], "lengths_without_eos": lengths}

    def save_rebuilt(name, value):
        path = ARTIFACTS / ("rebuilt-" + name)
        write_json(path, value)
        actual = digest(path)
        assert actual == manifest[name]["sha256"], (name, actual)
        return {"sha256": actual, "bytes": path.stat().st_size, "matches_formal_artifact": True}

    splits = split_records(_reasoning_records(), 42)
    assert _split_summary(splits) == result["split"]
    receipts["rebuilt_data"] = {"dataset.json": save_rebuilt("dataset.json", splits)}
    for left in splits:
        for right in splits:
            if left != right:
                assert not {row["family"] for row in splits[left]} & {row["family"] for row in splits[right]}
    mode_splits = {}
    receipts["target_budgets"] = {}
    for mode in ("direct", "steps"):
        mode_splits[mode] = {name: _reasoning_sft(rows, mode) for name, rows in splits.items()}
        receipts["rebuilt_data"][mode + "-dataset.json"] = save_rebuilt(mode + "-dataset.json", mode_splits[mode])
        examples = text_examples(mode_splits[mode]["train"], "sft")
        sampler = random.Random(42)
        tokens = sum(int((y != IGNORE).sum()) for _ in range(900) for _, y in sampler.choices(examples, k=24))
        assert tokens == result["comparison"][mode]["training"]["effective_tokens"]
        totals = {
            name: sum(len(tok.encode(row["messages"][1]["content"])) + 1 for row in rows)
            for name, rows in mode_splits[mode].items()
        }
        assert totals["test"] == result["comparison"][mode]["test"]["effective_tokens"]
        receipts["target_budgets"][mode] = {
            "updates": 900,
            "sampled_examples": 21600,
            "effective_training_targets": tokens,
            "dataset_targets": totals,
        }
        index_sampler = random.Random(42)
        indices = [index for _ in range(900) for index in index_sampler.choices(range(len(examples)), k=24)]
        receipts["target_budgets"][mode]["sampled_indices_sha256"] = hashlib.sha256(
            json.dumps(indices).encode()
        ).hexdigest()
    assert (
        receipts["target_budgets"]["direct"]["sampled_indices_sha256"]
        == receipts["target_budgets"]["steps"]["sampled_indices_sha256"]
    )
    receipts["target_budget_ratio"] = 466322 / 48803
    temperature_logits = torch.tensor([3.0, 1.0, 0.0])
    receipts["temperature_check"] = {
        "logits": temperature_logits.tolist(),
        "temperature_1": temperature_logits.softmax(-1).tolist(),
        "temperature_0.7": (temperature_logits / 0.7).softmax(-1).tolist(),
        "operation": "softmax(logits/T); only output scores are transformed, no optimizer update",
    }
    assert receipts["temperature_check"]["temperature_0.7"][0] > receipts["temperature_check"]["temperature_1"][0]

    asset = formal["assets"][0]
    archive = ROOT / asset["archive"]
    assert digest(archive) == asset["archive_sha256"]
    with tarfile.open(archive) as package:
        for row in asset["files"]:
            data = package.extractfile(row["path"]).read()
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
        raw = package.extractfile("reasoning-initial/gsm8k-train-first200.jsonl").read()
    upstream_url = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"
    with urllib.request.urlopen(upstream_url, timeout=40) as response:
        upstream = response.read()
    upstream_receipt = next(
        row
        for row in json.loads((ARTIFACTS / "external-fetch.json").read_text())
        if row["name"] == "upstream-gsm8k-train.jsonl"
    )
    assert hashlib.sha256(upstream).hexdigest() == upstream_receipt["sha256"]
    assert len(upstream) == upstream_receipt["bytes"]
    assert b"".join(upstream.splitlines(keepends=True)[:200]) == raw
    (ARTIFACTS / "upstream-gsm8k-first200.jsonl").write_bytes(raw)
    original = [json.loads(line) for line in raw.splitlines()]
    assert len(original) == 200 and all(set(row) == {"question", "answer"} for row in original)
    records = [
        {
            "family": hashlib.sha256(row["question"].encode()).hexdigest(),
            "source_index": i,
            "provenance": "GSM8K original human-authored training answer; not teacher generation",
            "messages": _conversation(row["question"], row["answer"]),
        }
        for i, row in enumerate(original)
    ]
    gsm_splits = split_records(records, 42)
    assert _split_summary(gsm_splits) == result["gsm8k_training_pilot"]["split"]
    receipts["rebuilt_data"]["gsm8k-dataset.json"] = save_rebuilt("gsm8k-dataset.json", gsm_splits)
    gsm_examples, packing = {}, {}
    target_conservation = []
    for name, rows in gsm_splits.items():
        gsm_examples[name], metadata, skipped = [], [], 0
        for row in rows:
            x, labels = render_chat(row["messages"])
            all_targets = []
            for start in range(0, len(x), 128):
                segment = (x[start : start + 128], labels[start : start + 128])
                count = int((segment[1] != IGNORE).sum())
                if not count:
                    skipped += 1
                    continue
                all_targets += segment[1][segment[1] != IGNORE].tolist()
                gsm_examples[name].append(segment)
                metadata.append(
                    {
                        "family": row["family"],
                        "source_index": row["source_index"],
                        "start": start,
                        "length": len(segment[0]),
                        "effective_tokens": count,
                    }
                )
            expected = tok.encode(row["messages"][1]["content"]) + [tok.eos_id]
            assert all_targets == expected
            target_conservation.append(
                {
                    "split": name,
                    "source_index": row["source_index"],
                    "answer_bytes_plus_eos": len(expected),
                    "ordered_target_sha256": hashlib.sha256(bytes(all_targets)).hexdigest()
                    if all(t < 256 for t in all_targets)
                    else hashlib.sha256(json.dumps(all_targets).encode()).hexdigest(),
                    "exactly_once_and_in_order": True,
                }
            )
        packing[name] = {
            "segments": metadata,
            "question_only_segments_skipped": skipped,
            "effective_tokens": sum(row["effective_tokens"] for row in metadata),
        }
    receipts["rebuilt_data"]["gsm8k-packing.json"] = save_rebuilt("gsm8k-packing.json", packing)
    sampler = random.Random(42)
    used = sum(int((y != IGNORE).sum()) for _ in range(150) for _, y in sampler.choices(gsm_examples["train"], k=16))
    assert used == 216776
    receipts["gsm8k"] = {
        "original_train_rows": len(upstream.splitlines()),
        "selected_complete_rows": 200,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "updates": 150,
        "batch_segments": 16,
        "sampled_segments": 2400,
        "effective_training_targets": used,
        "packing": {
            name: {
                "segments": len(value["segments"]),
                "question_only_segments_skipped": value["question_only_segments_skipped"],
                "effective_tokens": value["effective_tokens"],
            }
            for name, value in packing.items()
        },
        "target_conservation": target_conservation,
    }

    candidates = []
    for mode, comparison in result["comparison"].items():
        for budget in comparison["budgets"]:
            assert len(budget["samples"]) == 24
            correct = 0
            for row in budget["samples"]:
                assert row["truth"] == row["a"] + row["b"] + row["c"]
                assert len(row["candidates"]) == budget["candidate_count"]
                assert row["generation"]["temperature"] == 0.7
                assert row["generation"]["max_new_tokens"] == (8 if mode == "direct" else 48)
                for sample in row["candidates"]:
                    ids = sample["generated_ids"]
                    raw_ids = ids[:-1] if ids and ids[-1] == tok.eos_id else ids
                    assert sample["generated"] == tok.decode(raw_ids)
                    assert sample["generated_tokens"] == len(ids)
                    assert sample["invalid_special_tokens"] == [t for t in raw_ids if t < 8]
                    assert sample["eos"] == bool(ids and ids[-1] == tok.eos_id)
                    scored = _verify_reasoning(sample, row, mode)
                    assert all(sample[key] == value for key, value in scored.items())
                    candidates.append(
                        {
                            "mode": mode,
                            "k": budget["candidate_count"],
                            "family": row["family"],
                            "question": row["question"],
                            "truth": row["truth"],
                            "generated": sample["generated"],
                            "ids": ids,
                            **scored,
                        }
                    )
                correct += any(sample["final_correct"] for sample in row["candidates"])
            assert budget["oracle_coverage"]["numerator"] == correct
    write_json(ARTIFACTS / "all-candidates-rechecked.json", candidates)
    receipts["candidate_audit"] = {
        "total": len(candidates),
        "all_raw_ids_decode_eos_and_rescores_match": True,
        "single_candidate": {
            mode: sum(c["final_correct"] for c in candidates if c["mode"] == mode and c["k"] == 1)
            for mode in ("direct", "steps")
        },
        "denominator_single_candidate": 24,
    }
    gsm = result["gsm8k_training_pilot"]
    lookup = {row["family"]: row for row in gsm_splits["test"]}
    for item in gsm["continuations"]:
        prefix = lookup[item["family"]]["messages"][1]["content"].encode()[:40].decode("utf-8", errors="ignore")
        assert item["generation"]["messages"][0]["content"] == "Continue human answer: " + prefix
        assert item["prompt_mode"] == "human-answer prefix continuation, not problem solving"
    receipts["continuations"] = {
        "count": len(gsm["continuations"]),
        "all_20_human_answer_prefixes_match": True,
        "problem_solving_accuracy_computed": False,
    }

    with tempfile.TemporaryDirectory(prefix="fact_finish_c_1_dependency-") as temporary:
        ctx = Context("cpu", Path(temporary), Path(temporary), ROOT / "assets/training")
        base = new_lm(ctx, width=64, layers=2, max_length=128, heads=2, backend="sdpa")
        base_hash = hashlib.sha256(
            b"".join(value.detach().cpu().numpy().tobytes() for value in base.state_dict().values())
        ).hexdigest()
        assert (
            base_hash
            == result["comparison"]["direct"]["base_state_sha256"]
            == result["comparison"]["steps"]["base_state_sha256"]
        )
        receipts["base_initialization"] = {
            "regenerated_random_parameter_bytes_sha256": base_hash,
            "both_formal_hashes_equal": True,
            "loaded_pretrain_or_sft_base": False,
        }
        initial_gsm = new_lm(ctx, width=32, layers=1, max_length=128, heads=1, backend="sdpa")
        initial_nll = _nll_examples(initial_gsm, gsm_examples["train"], ctx)
        assert abs(initial_nll["nll"] - gsm["training"]["initial_loss"]) < 1e-5
        receipts["gsm_initial_cpu_nll"] = initial_nll
        entry = next(
            row
            for row in json.loads((ROOT / "docs/course-experiments/public-models.json").read_text())["models"]
            if row["id"] == "reasoning"
        )
        weight_receipts = []
        receipts["cpu_nll"] = {}
        for name in ("direct.pt", "steps.pt", "gsm8k-pilot.pt", "policy-strict.pt"):
            spec = next(row for row in entry["files"] if row["output"] == name)
            url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{spec['path']}"
            path = Path(temporary) / name
            with urllib.request.urlopen(url, timeout=40) as response:
                path.write_bytes(response.read())
            assert digest(path) == spec["sha256"] and path.stat().st_size == spec["bytes"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            tensors = [
                {
                    "name": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "finite": bool(torch.isfinite(value).all()),
                    "sha256": hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
                }
                for key, value in payload["model"].items()
            ]
            assert all(row["finite"] for row in tensors)
            weight_receipts.append(
                {
                    "name": name,
                    "url": url,
                    "revision": entry["revision"],
                    "bytes": path.stat().st_size,
                    "file_sha256": digest(path),
                    "keys": list(payload),
                    "config": payload.get("config"),
                    "tensor_checks": tensors,
                }
            )
            if name == "policy-strict.pt":
                try:
                    load_checkpoint(path)
                except ValueError as error:
                    assert str(error) == "不支援的 checkpoint 格式"
                    receipts["policy_language_loader_rejection"] = str(error)
                else:
                    raise AssertionError("finite policy unexpectedly accepted by LM loader")
                continue
            model, _ = load_checkpoint(path)
            mode = name.removesuffix(".pt")
            receipts["cpu_nll"][mode] = {}
            for split in ("train", "validation", "test"):
                measured = (
                    _nll_examples(model, gsm_examples[split], ctx)
                    if mode == "gsm8k-pilot"
                    else _sft_nll(model, mode_splits[mode][split], ctx)
                )
                expected = (
                    gsm["training"]["final_loss"]
                    if mode == "gsm8k-pilot" and split == "train"
                    else gsm[split]["nll"]
                    if mode == "gsm8k-pilot"
                    else result["comparison"][mode]["training"]["final_loss"]
                    if split == "train"
                    else result["comparison"][mode][split]["nll"]
                )
                assert abs(measured["nll"] - expected) < 1e-5, (mode, split, measured, expected)
                receipts["cpu_nll"][mode][split] = {
                    **measured,
                    "formal_nll": expected,
                    "absolute_difference": abs(measured["nll"] - expected),
                    "tolerance": 1e-5,
                }
            if mode == "steps":
                command = [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/infer.py",
                    str(path),
                    "--prompt",
                    "(1+2)+3=?",
                    "--chat",
                    "--tokens",
                    "48",
                    "--temperature",
                    "0",
                    "--device",
                    "cpu",
                    "--json",
                ]
                infer = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
                receipts["inference_cli"] = {
                    "command": command,
                    "exit_code": infer.returncode,
                    "stdout": infer.stdout,
                    "stderr": infer.stderr,
                }
        write_json(ARTIFACTS / "weights-inspected.json", weight_receipts)

    logits = torch.tensor([[[0.0, math.log(3)], [1.0, 0.0], [0.0, 0.0]]])
    labels = torch.tensor([[1, IGNORE, 0]])
    summed, count = loss_sum(logits, labels)
    expected = -math.log(0.75) - math.log(0.5)
    assert int(count) == 2 and abs(float(summed) - expected) < 1e-6
    receipts["manual_nll_check"] = {
        "probabilities": [0.75, 0.5],
        "ignored_positions": 1,
        "formula": "[-ln(0.75)-ln(0.5)]/2",
        "sum_expected": expected,
        "sum_observed": float(summed),
        "effective_count": int(count),
        "average": float(summed / count),
        "tolerance": 1e-6,
    }
    receipts["scopes"] = [
        "Private fixed raw artifacts returned 401 without existing HF credential; deterministic raw JSON rebuilt byte-for-byte to formal hashes.",
        "No long training rerun, no paid computation, CPU verifies loss and interfaces only, no CUDA time extrapolation.",
        "Release draft_role appeared incidentally while locating fixed raw files; no review verdict or author narrative was used as evidence.",
        "C.1 itself has no figure; prerequisite figures are not claims in this section.",
    ]
    write_json(ARTIFACTS / "replay-results.json", receipts)
    print(
        json.dumps(
            {
                "passed": True,
                "data_artifacts_matching_formal_sha": list(receipts["rebuilt_data"]),
                "candidate_count": len(candidates),
                "single_candidate_correct": receipts["candidate_audit"]["single_candidate"],
                "byte_lengths": lengths,
                "target_budget_ratio": receipts["target_budget_ratio"],
                "base_sha256": base_hash,
                "cpu_nll": receipts["cpu_nll"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
