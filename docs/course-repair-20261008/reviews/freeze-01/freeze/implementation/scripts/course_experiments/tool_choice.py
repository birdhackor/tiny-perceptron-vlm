"""Train a tiny next-token model to choose an action from natural-language requests.

This is a routing-component experiment: the model emits DIRECT, TOOL, or ASK.
It does not generate calculator arguments, execute a tool, or produce a final answer.
The specified policy prefers a calculator for every numerical addition when one is
available, including 1 + 2. This is learned task policy, not measured introspection.
"""

import argparse
import hashlib
import platform
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_perceptron.data import ByteTokenizer

from .applications import _sample
from .common import Context, fit_lm, new_lm, records_sha256, seed, write_json

ACTIONS = ("DIRECT", "TOOL", "ASK")
PLANNED_STEPS = 900
RESERVED_TEST_FAMILY = "pair:1:2"
POLICY = {
    "numerical_addition": "TOOL when the calculator is available; ASK otherwise",
    "explanation": "DIRECT",
    "copy": "DIRECT",
    "missing_quantity": "ASK",
    "scope": "A stipulated task policy, not a confidence-conditioned or cost-optimal policy.",
}


def _templates(a, b, paraphrase):
    if paraphrase:
        return [
            ("numerical_addition", f"幫我把 {a} 與 {b} 相加，給我答案。"),
            ("explanation", f"小朋友怎麼理解 {a} + {b}？"),
            ("copy", f"這裡寫著 {a}/{b}，請原封不動重寫。"),
            ("missing_quantity", f"商品 {a} 元，加收 {b} 元，數量未知，結帳要付多少？"),
        ]
    return [
        ("numerical_addition", f"{a} 加 {b} 等於多少？"),
        ("numerical_addition", f"請算出 {b} + {a} 的結果。"),
        ("explanation", f"請解釋 {a} 加 {b} 的意思。"),
        ("explanation", f"用積木說明 {b} 加 {a}。"),
        ("copy", f"請照抄這段：{a},{b}。"),
        ("copy", f"把 {b},{a} 原樣寫出來。"),
        ("missing_quantity", f"單價 {a} 元，運費 {b} 元，總價多少？"),
        ("missing_quantity", f"每個 {b} 元，運費 {a} 元，但數量沒說，求總價。"),
    ]


def _expected_action(intent, calculator_available):
    if intent == "numerical_addition":
        return "TOOL" if calculator_available else "ASK"
    return "ASK" if intent == "missing_quantity" else "DIRECT"


def build_records(*, paraphrase=False):
    """Keep operand order, all intents, and both availability conditions together."""
    rows = []
    for a in range(10):
        for b in range(a, 10):
            family = f"pair:{a}:{b}"
            for template, (intent, question) in enumerate(_templates(a, b, paraphrase)):
                for available in (True, False):
                    action = _expected_action(intent, available)
                    rows.append(
                        {
                            "family": family,
                            "operands": [a, b],
                            "template": template,
                            "paraphrase": paraphrase,
                            "intent": intent,
                            "calculator_available": available,
                            "expected_action": action,
                            "ask_reason": (
                                "missing_quantity"
                                if intent == "missing_quantity"
                                else "calculator_unavailable"
                                if intent == "numerical_addition" and not available
                                else None
                            ),
                            "expected_human_step": (
                                "請補充購買數量。"
                                if intent == "missing_quantity"
                                else "請啟用計算器，或提供可核對的計算結果。"
                                if intent == "numerical_addition" and not available
                                else None
                            ),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "計算器可用。" if available else "計算器停用。",
                                },
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": action},
                            ],
                        }
                    )
    return rows


def split_records(records, seed_value=42):
    """Predeclare 1 + 2 as test; split other unordered operand families 44/5/5."""
    families = sorted({row["family"] for row in records})
    if len(families) != 55 or RESERVED_TEST_FAMILY not in families:
        raise ValueError("Expected the complete 55-family dataset")
    families.remove(RESERVED_TEST_FAMILY)
    random.Random(seed_value).shuffle(families)
    partition = {
        "train": set(families[:44]),
        "validation": set(families[44:49]),
        "test": set(families[49:]) | {RESERVED_TEST_FAMILY},
    }
    return {name: [row for row in records if row["family"] in keys] for name, keys in partition.items()}


def parse_action(sample):
    """Strict protocol: a complete label and EOS; reject hidden control tokens."""
    if not sample["eos"] or sample["invalid_special_tokens"]:
        return None
    return sample["generated"] if sample["generated"] in ACTIONS else None


def _rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def summarize(rows, predictions):
    """Score chosen actions only; no tool or final-answer correctness is inferred."""
    if len(rows) != len(predictions):
        raise ValueError("Every evaluated request needs exactly one prediction")
    pairs = list(zip(rows, predictions, strict=True))
    needed = [(row, action) for row, action in pairs if row["expected_action"] == "TOOL"]
    not_needed = [(row, action) for row, action in pairs if row["expected_action"] != "TOOL"]
    unavailable = [(row, action) for row, action in pairs if not row["calculator_available"]]
    confusion = {
        expected: dict(Counter(action or "INVALID" for row, action in pairs if row["expected_action"] == expected))
        for expected in ACTIONS
    }
    return {
        "accuracy": _rate(sum(action == row["expected_action"] for row, action in pairs), len(rows)),
        "valid_action": _rate(sum(action in ACTIONS for action in predictions), len(rows)),
        "needed_tool_selected": _rate(sum(action == "TOOL" for _, action in needed), len(needed)),
        "needed_tool_missed": _rate(sum(action != "TOOL" for _, action in needed), len(needed)),
        "unnecessary_tool_selected": _rate(sum(action == "TOOL" for _, action in not_needed), len(not_needed)),
        "unavailable_tool_selected": _rate(sum(action == "TOOL" for _, action in unavailable), len(unavailable)),
        "confusion": confusion,
        "by_intent": {
            intent: _rate(
                sum(action == row["expected_action"] for row, action in pairs if row["intent"] == intent),
                sum(row["intent"] == intent for row in rows),
            )
            for intent in sorted({row["intent"] for row in rows})
        },
        "by_availability": {
            str(available).lower(): _rate(
                sum(
                    action == row["expected_action"]
                    for row, action in pairs
                    if row["calculator_available"] == available
                ),
                sum(row["calculator_available"] == available for row in rows),
            )
            for available in (True, False)
        },
    }


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _split_summary(rows):
    return {
        "records": len(rows),
        "families": sorted({row["family"] for row in rows}),
        "action_counts": dict(Counter(row["expected_action"] for row in rows)),
        "sha256": records_sha256(rows),
    }


def evaluate(model, records, ctx):
    started = time.perf_counter()
    samples, predictions = [], []
    for row in records:
        generation = _sample(model, row["messages"][:-1], ctx, tokens=8)
        action = parse_action(generation["samples"][0])
        predictions.append(action)
        samples.append({"record": row, "generation": generation, "chosen_action": action})
    return {
        "metrics": summarize(records, predictions),
        "baselines": {
            "always_direct": summarize(records, ["DIRECT"] * len(records)),
            "always_tool": summarize(records, ["TOOL"] * len(records)),
        },
        "samples": samples,
        "seconds": time.perf_counter() - started,
        "records_sha256": records_sha256(records),
    }


def run(ctx):
    """Run the single fixed schedule; never select a checkpoint using held-out scores."""
    wall_started = time.perf_counter()
    ctx.output.mkdir(parents=True, exist_ok=True)
    seed(ctx.seed)
    splits = split_records(build_records(), ctx.seed)
    test_families = {row["family"] for row in splits["test"]}
    paraphrases = [row for row in build_records(paraphrase=True) if row["family"] in test_families]
    write_json(ctx.output / "dataset.json", {**splits, "paraphrase_diagnostic": paraphrases})
    model = new_lm(ctx, width=64, layers=2, heads=2, max_length=160)
    training = fit_lm(
        model,
        splits["train"],
        ctx,
        mode="sft",
        steps=PLANNED_STEPS,
        batch_size=24,
        lr=0.003,
        name="tool-choice",
    )
    frozen_sha = _state_sha256(model)
    validation = evaluate(model, splits["validation"], ctx)
    test = evaluate(model, splits["test"], ctx)
    diagnostic = evaluate(model, paraphrases, ctx)
    unchanged = _state_sha256(model) == frozen_sha
    if not unchanged:
        raise RuntimeError("Evaluation changed model weights")
    checkpoint = ctx.output / "tool-choice.pt"
    result = {
        "schema_version": 1,
        "experiment": "tool_choice",
        "policy": POLICY,
        "model": model.description(),
        "training": training,
        "data": {
            **{name: _split_summary(rows) for name, rows in splits.items()},
            "paraphrase_diagnostic": _split_summary(paraphrases),
        },
        "validation": validation,
        "test": test,
        "paraphrase_diagnostic": diagnostic,
        "frozen_evaluation": {
            "weights_sha256": frozen_sha,
            "unchanged": unchanged,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "selection": "Final predeclared 900-step checkpoint; no tuning or checkpoint selection on validation/test/diagnostic.",
        },
        "reproducibility": {
            "seed": ctx.seed,
            "config": asdict(model.config),
            "tokenizer": ByteTokenizer().state(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": ctx.device,
            "cpu_threads": torch.get_num_threads(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "Synthetic short Traditional Chinese templates and numbers 0..9 only; no general-language benchmark.",
            "Template-based numeric-family test and unseen-paraphrase diagnostic answer different generalization questions.",
            "Only action selection is trained and evaluated; labels are not tool arguments or final answers.",
            "ASK covers missing quantity and unavailable calculator; explanation of why to ask is not trained.",
            "The calculator preference is stipulated in training labels, not learned from accuracy/cost rewards or self-awareness.",
            "No confidence calibration, dynamic cost optimization, tool execution, or real safety guarantee is established.",
        ],
        "total_seconds": time.perf_counter() - wall_started,
    }
    write_json(ctx.output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/tool-choice-development/fixed-900"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    ctx = Context(args.device, args.output, args.output, Path("assets"), args.seed)
    result = run(ctx)
    print({"parameters": result["model"]["parameters"], "training_seconds": result["training"]["seconds"]})
    for name in ("validation", "test", "paraphrase_diagnostic"):
        print(name, result[name]["metrics"])


if __name__ == "__main__":
    main()
