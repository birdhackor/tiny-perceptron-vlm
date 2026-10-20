"""核對候選重排、有效目標及教師生成回答的實際資料路徑。"""

import ast
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import compression
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import IGNORE, SPECIALS, ByteTokenizer, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def function_text(raw, names):
    text = raw.decode()
    tree = ast.parse(text)
    return {
        node.name: ast.get_source_segment(text, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    }


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-6/alignment-probe.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    }
    p = torch.tensor([0.8, 0.2])
    q = torch.tensor([0.2, 0.8])
    order = [["狗", "貓"].index(token) for token in ["貓", "狗"]]
    exercise_q = torch.tensor([0.8, 0.2])
    output["permutation"] = {
        "order": order,
        "dtype": str(p.dtype),
        "reordered": q[order].tolist(),
        "mismatched_mae": (p - q).abs().mean().item(),
        "aligned_mae": (p - q[order]).abs().mean().item(),
        "exercise_order": [["貓", "狗"].index(token) for token in ["貓", "狗"]],
        "exercise_mae": (p - exercise_q).abs().mean().item(),
        "vocabulary_only_changed_mae": (p - q[[0, 1]]).abs().mean().item(),
        "item_python_type": type(p.mean().item()).__name__,
    }
    assert order == [1, 0] and torch.equal(p, q[order])
    assert abs(output["permutation"]["mismatched_mae"] - 0.6) < 1e-6
    assert output["permutation"]["exercise_mae"] == 0

    tok = ByteTokenizer()
    records = [
        {"question": "color=green;shape=square;pitch=low;color?", "answer": "green", "family": "color"},
        {"question": "shape=square;shape?", "answer": "square", "family": "shape"},
    ]
    examples = compression._examples(records, 128)
    x, y, valid = pad_batch(examples)
    teacher = TinyLM(ModelConfig(width=64, layers=2, max_length=128)).eval()
    cache, _ = compression._cache_text(teacher, records, "cpu")
    with torch.no_grad():
        logits = teacher(x, valid=valid)["logits"]
    assert all(torch.equal(cached, logits[i, y[i] != IGNORE]) for i, cached in enumerate(cache))
    output["tokenizer_and_cache"] = {
        "vocab_size": tok.vocab_size,
        "special_ids": dict(zip(SPECIALS, range(len(SPECIALS)))),
        "unicode_ids": tok.encode("小貓追"),
        "byte_offset": len(SPECIALS),
        "ascii_byte_values": len(set(tok.encode(bytes(range(128)).decode("ascii")))),
        "input_shape": list(x.shape),
        "cache_shapes": [list(value.shape) for value in cache],
        "valid_targets_per_record": [int((labels != IGNORE).sum()) for _, labels in examples],
        "prompt_targets_ignored": [bool(torch.all(labels[:len(compression._prompt(record)) - 1] == IGNORE)) for record, (_, labels) in zip(records, examples)],
        "pad_targets_ignored": bool(torch.all(y[~valid] == IGNORE)),
        "cache_matches_selected_teacher_rows": True,
    }
    assert all(output["tokenizer_and_cache"]["prompt_targets_ignored"])
    assert output["tokenizer_and_cache"]["pad_targets_ignored"]

    class StopBeforeUpdate(Exception):
        pass

    alignment = []
    for width in (16, 32):
        student = TinyLM(ModelConfig(width=width, layers=1, heads=2, max_length=128))
        ctx = SimpleNamespace(seed=42, device="cpu", output=Path(__file__).parent)
        plan = [random.Random(ctx.seed)]
        indices = [plan[0].randrange(len(examples)) for _ in range(16)]
        expected_x, expected_y, expected_valid = pad_batch([examples[i] for i in indices])
        expected_targets = torch.cat([cache[i] for i in indices])[None]
        captured = {}

        def forward_hook(module, args, kwargs, result):
            captured["x"] = args[0].detach().clone()
            captured["valid"] = kwargs["valid"].detach().clone()
            captured["logits"] = result["logits"].detach().clone()

        def capture_kl(selected, targets, labels, temperature):
            assert torch.equal(captured["x"], expected_x)
            assert torch.equal(captured["valid"], expected_valid)
            assert torch.equal(selected.detach(), captured["logits"][expected_y != IGNORE][None])
            assert torch.equal(targets, expected_targets)
            assert torch.equal(labels, expected_y[expected_y != IGNORE][None])
            value = distillation_kl(selected, targets, labels, temperature)
            try:
                distillation_kl(selected, targets[:, :-1], labels, temperature)
            except ValueError as error:
                mismatch_error = str(error)
            else:
                raise AssertionError("目標數不同時應拒絕比較")
            alignment.append({
                "student_width": width,
                "vocab_size": student.config.vocab_size,
                "sampled_indices": indices,
                "selected_shape": list(selected.shape),
                "labels_shape": list(labels.shape),
                "same_gold_inputs_and_cache_order": True,
                "kl": value.item(),
                "count_mismatch_error": mismatch_error,
                "optimizer_updates": 0,
            })
            raise StopBeforeUpdate

        handle = student.register_forward_hook(forward_hook, with_kwargs=True)
        before = compression._parameter_hash(student)
        try:
            with patch.object(compression, "distillation_kl", capture_kl):
                compression._fit_text(ctx, student, records, "alignment", steps=1, teacher_cache=cache)
        except StopBeforeUpdate:
            pass
        finally:
            handle.remove()
        assert before == compression._parameter_hash(student)
    output["student_selection"] = alignment

    student_logits = torch.randn(*logits.shape, requires_grad=True)
    loss = distillation_kl(student_logits, logits.detach(), y, temperature=2.0)
    loss.backward()
    excluded = y == IGNORE
    assert torch.count_nonzero(student_logits.grad[excluded]).item() == 0
    assert student_logits.grad[~excluded].abs().sum().item() > 0
    output["kl_mask"] = {
        "excluded_positions": int(excluded.sum()),
        "pad_positions": int((~valid).sum()),
        "included_positions": int((~excluded).sum()),
        "excluded_gradient_nonzero_elements": int(torch.count_nonzero(student_logits.grad[excluded])),
        "included_gradient_absolute_sum": student_logits.grad[~excluded].abs().sum().item(),
    }

    ctx = SimpleNamespace(output=Path(__file__).parent, device="cpu")
    hard, audit = compression._hard_targets(ctx, teacher, records, "probe", tokens=3)
    hard_checks = []
    for record, row in zip(records, hard):
        hx, hy = compression._example(row, 128)
        prefix = compression._prompt(record)
        assert hy[hy != IGNORE].tolist() == row["_hard_ids"]
        assert torch.equal(hx[:len(prefix)], prefix)
        hard_checks.append({"question": record["question"], "teacher_ids": row["_hard_ids"], "effective_labels": hy[hy != IGNORE].tolist(), "same_prompt": True, "gold_ids": tok.encode(record["answer"]) + [tok.eos_id]})
    output["hard_targets"] = {"checks": hard_checks, "file": str(Path(audit["file"]).relative_to(ROOT)), "sha256": audit["sha256"], "records": audit["records"]}

    result_path = ROOT / "docs/course-experiments/results/distillation.json"
    result_raw = result_path.read_bytes()
    result = json.loads(result_raw)
    attributes = result["results"]["tasks"]["attributes"]
    historical = {}
    for relative, names in {
        "scripts/course_experiments/compression.py": {"_chat", "_example", "_examples", "_prompt", "_cache_text", "_fit_text", "_hard_targets", "_distill_case", "run_distillation"},
        "tiny_perceptron/data.py": {"ByteTokenizer", "render_chat", "pad_batch"},
        "tiny_perceptron/alignment.py": {"distillation_kl"},
        "tiny_perceptron/model.py": {"ModelConfig"},
    }.items():
        command = ["git", "show", f"{result['revision']}:{relative}"]
        retrieval = subprocess.run(command, cwd=ROOT, capture_output=True)
        current = (ROOT / relative).read_bytes()
        expected = result["code_sha256"][relative]
        comparison = {"command": command, "exit_code": retrieval.returncode, "stderr": retrieval.stderr.decode(), "recorded_sha256": expected, "current_sha256": digest(current)}
        raw = retrieval.stdout if retrieval.returncode == 0 else None
        if raw is None and digest(current) != expected:
            log_command = ["git", "log", "--all", "--format=%H", "--", relative]
            commits = subprocess.check_output(log_command, cwd=ROOT).decode().splitlines()
            comparison["fallback_log_command"] = log_command
            comparison["searched_file_revisions"] = len(commits)
            for commit in commits:
                candidate = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT, capture_output=True)
                if candidate.returncode == 0 and digest(candidate.stdout) == expected:
                    raw = candidate.stdout
                    comparison["matching_source_command"] = ["git", "show", f"{commit}:{relative}"]
                    break
        if raw is None and digest(current) == expected:
            raw = current
            comparison["matching_source"] = "current working tree"
        if raw is not None:
            old_functions = function_text(raw, names)
            new_functions = function_text(current, names)
            comparison.update(retrieved_sha256=digest(raw), recorded_hash_matches=digest(raw) == expected, functions_equal={name: old_functions.get(name) == new_functions.get(name) for name in sorted(names)})
        historical[relative] = comparison
    output["experiment_record"] = {
        "path": str(result_path.relative_to(ROOT)),
        "sha256": digest(result_raw),
        "revision": result["revision"],
        "teacher_config": attributes["teacher_provenance"]["config"],
        "student_runs": {name: {"parameters": run["storage"]["parameter_count"], "objective": run["training"]["objective"]} for name, run in attributes["runs"].items() if name in {"w16_ce", "w16_teacher_hard", "w16_ce_kl", "w32_ce", "w32_teacher_hard", "w32_ce_kl"}},
        "historical_source_comparison": historical,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
