"""Independently recompute lesson 9.1 evidence; never publish private text."""

import argparse
import ast
import copy
import hashlib
import io
import json
import math
import platform
import random
import re
import tarfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments.behavior import _conversation
from scripts.course_experiments.common import Context, _nll, new_lm, split_records, text_examples
from scripts.course_experiments.text import _digest, _utf8_prefix
from scripts.course_release import validate_approval
from scripts.ingest_course_result import PRIVATE_TEXT_KEYS, redact
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.model import generate, loss_sum

ROOT = Path(__file__).resolve().parents[3]
PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_v2_09_01_"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_source(path, name):
    module = ast.parse(path.read_text())
    return ast.dump(next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name))


class FixedNextToken(torch.nn.Module):
    """Exercise actual stopping rules with a controlled next-token logit."""

    def __init__(self, next_id, max_length):
        super().__init__()
        self.next_id = next_id
        self.config = SimpleNamespace(max_length=max_length)

    def forward(self, ids, cache=None):
        scores = torch.zeros((*ids.shape, 264))
        scores[:, :, self.next_id] = 1
        return {"logits": scores, "cache": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-result", type=Path)
    parser.add_argument("--remote-receipt", type=Path)
    args = parser.parse_args()
    if args.private_result:
        args.private_result = args.private_result.resolve()
    torch.set_num_threads(2)
    formal_path = ROOT / "docs/course-experiments/results/safety.json"
    formal = json.loads(formal_path.read_text())
    pilot = formal["results"]["pku_pilot"]
    archive = ROOT / "assets/training/pku-safe-rlhf-v1.tar.gz"
    asset = next(item for item in formal["assets"] if item["id"] == "pku-safe-rlhf")
    assert digest(archive) == asset["archive_sha256"]
    with tarfile.open(archive) as bundle:
        member = next(item for item in bundle.getmembers() if item.name.endswith("train-first-100.jsonl"))
        raw_bytes = bundle.extractfile(member).read()
        viewer_file = next(item for item in bundle.getmembers() if item.name.endswith("source-viewer-response.json"))
        viewer = json.load(bundle.extractfile(viewer_file))
        api_file = next(item for item in bundle.getmembers() if item.name.endswith("source-api.json"))
        source_api = json.load(bundle.extractfile(api_file))
    assert hashlib.sha256(raw_bytes).hexdigest() == "c4a88d08ef7456669766f1a3b908f82d918be5625a24794d9843d7f6df1d552b"
    raw = [json.loads(line) for line in raw_bytes.splitlines()]
    assert len(raw) == 100
    assert source_api["sha"] == "9421ffafec3fa40a1f1a7d567b4d525079477ecb"
    assert [item["row_idx"] for item in viewer["rows"]] == list(range(100))
    assert [item["row"] for item in viewer["rows"]] == raw
    assert all(not item["truncated_cells"] for item in viewer["rows"])
    records, selection = [], []
    for index, row in enumerate(raw):
        safer = row["safer_response_id"]
        selected = row[f"is_response_{safer}_safe"]
        selection.append({"source_row": index, "safer_response_id": safer, "selected_is_safe": selected})
        if selected:
            records.append(
                _conversation(
                    _utf8_prefix(row["prompt"], 120),
                    _utf8_prefix(row[f"response_{safer}"], 120),
                    _digest(row["prompt"]),
                    source_row=index,
                    source_record_sha256=_digest(row),
                    safer_response_id=safer,
                    better_response_id=row["better_response_id"],
                )
            )
    assert len(records) == 60
    parts = split_records(records, seed=42)
    split_audit = {}
    families = set()
    for split, rows in parts.items():
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        assert hashlib.sha256(encoded).hexdigest() == pilot["data"][split]["sha256"]
        local_families = {row["family"] for row in rows}
        assert not families & local_families
        families.update(local_families)
        examples = text_examples(rows, mode="sft", max_length=256)
        items = []
        for row, (x, y) in zip(rows, examples, strict=True):
            question, answer = (message["content"] for message in row["messages"])
            answer_tokens = len(answer.encode())
            assert x.dtype == y.dtype == torch.int64
            assert int((y != IGNORE).sum()) == answer_tokens + 1
            assert y[-1].item() == ByteTokenizer.eos_id
            assert len(x) == len(question.encode()) + answer_tokens + 4 <= 244
            assert len(question.encode()) <= 120 and answer_tokens <= 120
            items.append(
                {
                    "source_row": row["source_row"],
                    "family": row["family"],
                    "prompt_bytes": len(question.encode()),
                    "answer_bytes": answer_tokens,
                    "sequence_length": len(x),
                    "effective_targets_including_eos": answer_tokens + 1,
                    "record_sha256": _digest(row),
                }
            )
        split_audit[split] = {
            "records": len(rows),
            "families": len(local_families),
            "jsonl_sha256": hashlib.sha256(encoded).hexdigest(),
            "rows": items,
            "effective_targets": sum(item["effective_targets_including_eos"] for item in items),
        }
    sampler = random.Random(42)
    effective = sum(
        int((y != IGNORE).sum())
        for _ in range(80)
        for _, y in sampler.choices(text_examples(parts["train"], "sft", 256), k=4)
    )
    assert effective == pilot["training"]["effective_tokens"] == 37709
    ctx = Context("cpu", ROOT / "unused-audit-output", ROOT, ROOT, 42)
    model = new_lm(ctx, width=32, layers=1, max_length=256)
    initial = _nll(model, text_examples(parts["train"], "sft", 256))
    assert abs(initial["nll"] - pilot["training"]["initial_loss"]) < 2e-6
    assert model.description()["parameters"] == 37728
    lesson = dict(sections(ROOT / "course/chapters/09.md"))["9.1"]
    snippet = re.search(r"```python\n(.*?)```", lesson, re.S)[1]
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(compile(snippet, "course/chapters/09.md#9.1", "exec"), {})
    original_output = stdout.getvalue()
    assert original_output == "True True True\n較安全回答可直接當安全正例 False\n"
    variant = snippet.replace('"回答0安全": False', '"回答0安全": True')
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(compile(variant, "course/chapters/09.md#9.1 exercise", "exec"), {})
    assert stdout.getvalue() == "True True True\n較安全回答可直接當安全正例 True\n"
    crop_tests = {
        text: {
            "codepoints": len(text),
            "bytes": len(text.encode()),
            "prefix_bytes": len(_utf8_prefix(text, 120).encode()),
        }
        for text in ["貓" * 41, "🙂" * 31, "a" * 121]
    }
    assert crop_tests["貓" * 41]["prefix_bytes"] == 120
    assert crop_tests["🙂" * 31]["prefix_bytes"] == 120
    logits = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    labels = torch.tensor([[1, IGNORE]])
    summed, count = loss_sum(logits, labels)
    manual = math.log(math.exp(2) + 2) - 2
    assert count.item() == 1 and abs(summed.item() - manual) < 1e-14
    prompt = torch.tensor([[1, 3, 81, 2, 4]])
    eos_output = generate(FixedNextToken(2, 256), prompt, 128)
    context_output = generate(FixedNextToken(81, 8), prompt, 128)
    token_output = generate(FixedNextToken(81, 256), prompt, 128)
    assert eos_output.shape[1] == 6 and eos_output[0, -1].item() == 2
    assert context_output.shape[1] == 8 and token_output.shape[1] == 133
    approval = json.loads((ROOT / "docs/course-experiments/releases/safety.json").read_text())
    public_manifest_path = Path(str(PREFIX) + "public-export-manifest.json")
    public_manifest = json.loads(public_manifest_path.read_text())
    public_release = json.loads((ROOT / "docs/course-experiments/public-releases/safety.json").read_text())
    expected_manifest = next(
        item
        for item in public_release["release"]["public_manifest"]["files"]
        if item["output"] == "export-manifest.json"
    )
    assert digest(public_manifest_path) == expected_manifest["sha256"]
    assert [item["output"] for item in public_manifest["files"]] == [item["path"] for item in approval["files"]]
    assert [item["source_sha256"] for item in public_manifest["files"]] == [
        item["sha256"] for item in approval["files"]
    ]
    validate_approval(approval, "safety", "course-v1", approval["private_source"]["repo"])
    rejected = []
    for field, value in [("pku_derived", True), ("visibility", "private")]:
        test = copy.deepcopy(approval)
        test[field] = value
        try:
            validate_approval(test, "safety", "course-v1", test["private_source"]["repo"])
        except ValueError as error:
            rejected.append(str(error))
    assert len(rejected) == 2
    cleaned = redact({"private_only": True, "samples": ["DO NOT PUBLISH"], "records": 6})
    assert not PRIVATE_TEXT_KEYS & cleaned.keys() and cleaned["records"] == 6
    frozen = Path(str(PREFIX) + "formal-behavior.txt")
    current = ROOT / "scripts/course_experiments/behavior.py"
    same_functions = {
        name: function_source(current, name) == function_source(frozen, name)
        for name in ["_pku_pilot", "_conversation"]
    }
    assert all(same_functions.values())
    sample_audit = {"status": "unavailable", "reason": "public formal report redacts private-branch samples"}
    if args.private_result:
        full = json.loads(args.private_result.read_text())
        full_pilot = full["results"]["pku_pilot"]
        assert full_pilot["training"] == pilot["training"]
        assert redact(full_pilot) == pilot
        assert full["preflight"]["checkpoint_private"] is True
        assert "pku-pilot.pt" in full["hf"]["verified_checkpoints"]
        all_rows = {}
        for split in ["validation", "test"]:
            samples = full_pilot["evaluation"][split]["samples"]
            assert len(samples) == 6
            checked = []
            for row, sample in zip(parts[split], samples, strict=True):
                ids = sample["generated_ids"]
                raw_generated = ids[: ids.index(2)] if 2 in ids else ids
                expected_ids = ByteTokenizer().encode(row["messages"][-1]["content"])
                assert sample["messages"] == row["messages"][:-1]
                assert sample["expected"] == row["messages"][-1]["content"]
                assert sample["exact"] == (raw_generated == expected_ids)
                assert sample["eos"] == (2 in ids)
                prompt_length = len(row["messages"][0]["content"].encode()) + 4
                assert len(ids) == min(128, 256 - prompt_length)
                checked.append(
                    {
                        "source_row": row["source_row"],
                        "exact": sample["exact"],
                        "eos": sample["eos"],
                        "generated_tokens": len(ids),
                        "generated_ids_sha256": _digest(ids),
                        "prompt_length": prompt_length,
                    }
                )
            assert sum(item["exact"] for item in checked) == sum(item["eos"] for item in checked) == 0
            all_rows[split] = checked
        sample_audit = {
            "status": "all_12_original_samples_checked",
            "full_result_sha256": digest(args.private_result),
            "rows": all_rows,
            "private_result_path": str(args.private_result.relative_to(ROOT)),
            "preflight_checkpoint_private": full["preflight"]["checkpoint_private"],
            "verified_private_backup": full["hf"],
        }
    remote_audit = None
    if args.remote_receipt:
        receipt = json.loads(args.remote_receipt.read_text())
        proof = receipt["proof"]
        assert proof["formal_result_sha256"] == digest(formal_path)
        assert receipt["probe_sha256"] == proof["probe_sha256"] == digest(Path(str(PREFIX) + "checkpoint_probe.py"))
        for verified in receipt["verified_files"]:
            original = next(item for item in formal["artifacts"] if item["path"] == verified["path"])
            assert verified == original
        for path, source_hash in proof["code_sha256"].items():
            assert source_hash == formal["code_sha256"][path] == digest(ROOT / path)
        assert proof["step"] == 80 and proof["optimizer_step_values"] == [80.0]
        assert proof["effective_update_targets"] == 37709
        assert proof["final_train_nll"]["effective_tokens"] == 5599
        assert abs(proof["final_train_nll"]["nll"] - pilot["training"]["final_loss"]) < 2e-6
        for split in ["validation", "test"]:
            observed = proof["complete_evaluations"][split]
            assert observed["effective_tokens"] == 726
            assert abs(observed["nll"] - pilot["evaluation"][split]["nll"]) < 2e-6
            assert len(observed["all_samples"]) == 6
            for checked, original in zip(observed["all_samples"], sample_audit["rows"][split], strict=True):
                for key in ["source_row", "exact", "eos", "generated_tokens", "generated_ids_sha256"]:
                    assert checked[key] == original[key]
                assert checked["matches_original_gpu_ids"] is True
        remote_audit = {
            "receipt_sha256": digest(args.remote_receipt),
            "run_url": "https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37173099328",
            "checkpoint_final_cpu_nll": proof["final_train_nll"],
            "all_12_cpu_ids_equal_original_gpu": True,
            "environment": proof["environment"],
        }
    output = {
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_09_01_audit.py"
        + (f" --private-result {args.private_result.relative_to(ROOT)}" if args.private_result else "")
        + (f" --remote-receipt {args.remote_receipt}" if args.remote_receipt else ""),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "result": "all implemented assertions passed",
        "lesson_source_sha256": hashlib.sha256(lesson.encode()).hexdigest(),
        "formal_result_sha256": digest(formal_path),
        "selection": selection,
        "splits": split_audit,
        "initial_cpu_nll": initial,
        "effective_update_targets": effective,
        "formal_training": pilot["training"],
        "formal_environment": {
            key: formal[key]
            for key in ["device", "seed", "torch_version", "python_version", "gpu", "timing_scope", "step_scale"]
        },
        "original_snippet_output": original_output,
        "exercise_output": stdout.getvalue(),
        "crop_tests": crop_tests,
        "ce_example": {"manual": manual, "executed": summed.item(), "denominator": count.item()},
        "publication_rejections": rejected,
        "public_weights": [item["path"] for item in approval["files"]],
        "anonymous_public_export_manifest_sha256": digest(public_manifest_path),
        "frozen_functions_match": same_functions,
        "original_samples": sample_audit,
        "generation_limit_source": "tiny_perceptron/model.py:109-132; EOS or min(new-token cap, context cap)",
        "generation_stopping_execution": {
            "prompt_tokens": 5,
            "eos_new_tokens": 1,
            "context_cap_8_new_tokens": 3,
            "max_new_tokens_128": 128,
        },
        "remote_receipt_checked": remote_audit,
    }
    destination = Path(str(PREFIX) + "audit.json")
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": output["result"],
                "output": str(destination.relative_to(ROOT)),
                "splits": {
                    name: {key: value for key, value in item.items() if key != "rows"}
                    for name, item in split_audit.items()
                },
                "initial_cpu_nll": initial,
                "effective_update_targets": effective,
                "original_samples": sample_audit["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
