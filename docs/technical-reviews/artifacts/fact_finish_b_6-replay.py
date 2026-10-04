"""Independent CPU verification for B.6; never reruns the 900-step schedule."""

import argparse
import contextlib
import hashlib
import io
import json
import platform
import random
import re
from collections import Counter
from pathlib import Path

import torch

from scripts.course_experiments.applications import _prompt_ids, _sample
from scripts.course_experiments.common import Context, _nll, fit_lm, new_lm, records_sha256, text_examples
from scripts.course_experiments.tool_choice import build_records, parse_action, split_records, summarize
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_finish_b_6"


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "outputs/tool-choice-development/fixed-900/tool-choice.pt"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = {
        "reviewer_task": "/root/fact_finish_b_6",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": str(torch.get_num_threads()),
            "platform": platform.platform(),
        },
    }
    body = PREFIX.with_name(PREFIX.name + "-section-B.6.md").read_text()
    snippet = re.search(r"```python\n(.*?)```", body, re.S)[1]
    scope = {}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(snippet, "B.6-read-snapshot", "exec"), scope)
    x, labels, tok = scope["x"], scope["labels"], scope["tokenizer"]
    answer = labels[labels != -100].tolist()
    assert answer == [92, 87, 87, 84, 2]
    assert tok.decode(answer) == "TOOL" and len(answer) == 5
    messages = [dict(row) for row in scope["messages"]]
    messages[-1]["content"] = "ASK"
    ask_x, ask_labels = render_chat(messages, tok)
    ask_ids = ask_labels[ask_labels != -100].tolist()
    assert ask_ids == [73, 91, 83, 2]
    result["short_example"] = {
        "actual_stdout": printed.getvalue(),
        "x_dtype": str(x.dtype),
        "labels_dtype": str(labels.dtype),
        "x_shape": list(x.shape),
        "labels_shape": list(labels.shape),
        "input_min": int(x.min()),
        "input_max": int(x.max()),
        "answer_ids": answer,
        "valid_positions": (labels != -100).nonzero().flatten().tolist(),
        "ignored_positions": int((labels == -100).sum()),
        "ask_ids": ask_ids,
        "ask_valid_positions": int((ask_labels != -100).sum()),
        "ask_input_length": len(ask_x),
        "ask_labels_are_not_validated_against_policy": True,
    }

    # Check direct logit gradients separately from gradients through prefix input.
    torch.manual_seed(17)
    tiny = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=192))
    logits = tiny(x.unsqueeze(0))["logits"]
    logits.retain_grad()
    loss = masked_loss(logits, labels.unsqueeze(0))
    loss.backward()
    ignored_gradient = float(logits.grad[0, labels == -100].abs().max())
    supervised_gradient = float(logits.grad[0, labels != -100].abs().sum())
    prefix_id = tok.encode("1")[0]
    prefix_gradient = float(tiny.embedding.weight.grad[prefix_id].norm())
    assert ignored_gradient == 0 and supervised_gradient > 0 and prefix_gradient > 0
    result["gradient_check"] = {
        "loss": float(loss.detach()),
        "ignored_logits_gradient_max": ignored_gradient,
        "supervised_logits_gradient_sum": supervised_gradient,
        "user_only_digit_1_embedding_id": prefix_id,
        "user_only_digit_1_embedding_gradient_norm": prefix_gradient,
        "scope": "A single CPU random model proves the two gradient routes can differ; it is not a training-quality benchmark.",
    }

    dataset_path = PREFIX.with_name(PREFIX.name + "-dataset.json")
    formal_path = PREFIX.with_name(PREFIX.name + "-formal-tool-choice.json")
    dataset = json.loads(dataset_path.read_text())
    formal = json.loads(formal_path.read_text())
    saved = formal["results"]
    regenerated = split_records(build_records(), seed_value=42)
    families = {row["family"] for row in regenerated["test"]}
    regenerated["paraphrase_diagnostic"] = [row for row in build_records(paraphrase=True) if row["family"] in families]
    assert regenerated == dataset
    assert not ({row["family"] for row in dataset["train"]} & families)
    lengths = {}
    for split, rows in dataset.items():
        examples = text_examples(rows, mode="sft", max_length=160)
        lengths[split] = {
            "records": len(rows),
            "families": sorted({row["family"] for row in rows}),
            "actions": dict(Counter(row["expected_action"] for row in rows)),
            "max_x_length": max(len(inputs) for inputs, _ in examples),
            "supervised_tokens": sum(int((targets != -100).sum()) for _, targets in examples),
            "records_sha256": records_sha256(rows),
            "all_records_exactly_equal_to_rebuilt_source": True,
        }
        assert lengths[split]["records_sha256"] == saved["data"][split]["sha256"]
    train_examples = text_examples(dataset["train"], "sft", 160)
    sampler = random.Random(42)
    sampled_counts = [
        sum(int((targets != -100).sum()) for _, targets in sampler.choices(train_examples, k=24)) for _ in range(900)
    ]
    assert sum(sampled_counts) == saved["training"]["effective_tokens"] == 121297
    result["dataset_checks"] = {
        "dataset_bytes_sha256": file_sha(dataset_path),
        "formal_bytes_sha256": file_sha(formal_path),
        "splits": lengths,
        "no_CALC_or_COPY_user_prefixes": all(
            not row["messages"][1]["content"].startswith(("CALC", "COPY")) for rows in dataset.values() for row in rows
        ),
        "sampling_effective_tokens": sum(sampled_counts),
        "each_step_effective_tokens": sampled_counts,
    }

    model, payload = load_checkpoint(args.checkpoint, device="cpu")
    tensors = []
    for name, tensor in sorted(payload["model"].items()):
        assert torch.isfinite(tensor).all()
        tensors.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "elements": tensor.numel(),
                "all_finite": True,
                "sha256": hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest(),
                "strict_loaded_equal": torch.equal(tensor, model.state_dict()[name]),
            }
        )
    assert file_sha(args.checkpoint) == saved["frozen_evaluation"]["checkpoint_sha256"]
    assert state_sha(payload["model"]) == saved["frozen_evaluation"]["weights_sha256"]
    assert payload["step"] == 900
    assert payload["config"] == saved["reproducibility"]["config"]
    assert model.description()["parameters"] == 143616
    result["checkpoint_checks"] = {
        "checkpoint_sha256": file_sha(args.checkpoint),
        "state_sha256": state_sha(payload["model"]),
        "format_version": payload["format_version"],
        "config": payload["config"],
        "step": payload["step"],
        "metadata": payload["metadata"],
        "tokenizer": payload["tokenizer"],
        "training_state": {k: v for k, v in payload["training_state"].items() if k != "sampler_rng"},
        "optimizer_parameter_steps": sorted({int(v["step"]) for v in payload["optimizer"]["state"].values()}),
        "tensors": tensors,
        "parameters": model.description()["parameters"],
        "binary_not_copied": True,
        "replay_requires_matching_checkpoint_or_regeneration": True,
    }

    ctx = Context("cpu", ROOT / "outputs/fact-finish-b6-short-training", ROOT / "outputs", ROOT / "assets", 42)
    random_model = new_lm(ctx, width=64, layers=2, heads=2, max_length=160)
    initial = _nll(random_model, train_examples)
    final = _nll(model, train_examples)
    assert abs(initial["nll"] - saved["training"]["initial_loss"]) < 1e-7
    # FP32 loss aggregation is approximate; label/ID comparisons below stay exact.
    assert abs(final["nll"] - saved["training"]["final_loss"]) < 1e-7
    before = state_sha(random_model.state_dict())
    ctx.output.mkdir(parents=True, exist_ok=True)
    one_step = fit_lm(
        random_model, dataset["train"], ctx, mode="sft", steps=1, batch_size=24, lr=0.003, name="review-smoke"
    )
    after = state_sha(random_model.state_dict())
    assert before != after and one_step["steps"] == 1
    result["initial_final_loss_and_single_update"] = {
        "recorded_initial_loss": saved["training"]["initial_loss"],
        "recreated_random_initial": initial,
        "recorded_final_loss": saved["training"]["final_loss"],
        "recomputed_final": final,
        "final_nll_absolute_tolerance": 1e-7,
        "final_nll_absolute_difference": abs(final["nll"] - saved["training"]["final_loss"]),
        "initial_state_sha256": before,
        "after_one_step_sha256": after,
        "one_step": one_step,
        "scope": "One new optimizer update only. Original 900-step schedule and original timing were not rerun.",
    }

    replayed = {}
    compact_rows = []
    for split in ("validation", "test", "paraphrase_diagnostic"):
        samples = saved[split]["samples"]
        assert len(samples) == len(dataset[split])
        predictions = []
        for index, (row, sample) in enumerate(zip(dataset[split], samples, strict=True)):
            assert sample["record"] == row
            generation = sample["generation"]
            assert generation["messages"] == row["messages"][:-1]
            assert generation["input_ids"] == _prompt_ids(row["messages"][:-1], ByteTokenizer())
            original = generation["samples"][0]
            assert original["generated"] == ByteTokenizer().decode(original["generated_ids"])
            actual = _sample(model, row["messages"][:-1], ctx, tokens=8)["samples"][0]
            assert actual == original
            action = parse_action(actual)
            assert action == sample["chosen_action"]
            predictions.append(action)
            compact_rows.append(
                {
                    "split": split,
                    "index": index,
                    "family": row["family"],
                    "intent": row["intent"],
                    "system": row["messages"][0]["content"],
                    "question": row["messages"][1]["content"],
                    "expected": row["expected_action"],
                    "generated": actual,
                    "identical_to_formal_raw_record": True,
                }
            )
        recomputed = summarize(dataset[split], predictions)
        assert recomputed == saved[split]["metrics"]
        independent_correct = sum(
            a == row["expected_action"] for a, row in zip(predictions, dataset[split], strict=True)
        )
        assert independent_correct == recomputed["accuracy"]["numerator"]
        replayed[split] = {
            "generation_records": len(samples),
            "all_prompt_ids_checked": True,
            "all_generated_ids_and_EOS_reproduced_exactly": True,
            "recomputed_metrics": recomputed,
        }
    result["replayed_generations"] = replayed
    result["all_assertions_passed"] = True
    PREFIX.with_name(PREFIX.name + "-execution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    PREFIX.with_name(PREFIX.name + "-generation-replay.json").write_text(
        json.dumps(compact_rows, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "short_example": result["short_example"],
                "gradient_check": result["gradient_check"],
                "records": {k: v["records"] for k, v in lengths.items()},
                "effective_tokens": sum(sampled_counts),
                "checkpoint_step": payload["step"],
                "checkpoint_tensors": len(tensors),
                "recreated_initial_nll": initial["nll"],
                "recomputed_final_nll": final["nll"],
                "replayed_records": sum(v["generation_records"] for v in replayed.values()),
                "all_assertions_passed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
