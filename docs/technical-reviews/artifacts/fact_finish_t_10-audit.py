"""Fresh T.10 record, denominator, objective and pinned CPU inference audit."""

import hashlib
import json
import math
import platform
import random
import sys
import tarfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.behavior import (  # noqa: E402
    _conversation,
    _style_record,
)
from scripts.course_experiments.common import Context, split_records  # noqa: E402
from scripts.course_experiments.compression import (  # noqa: E402
    _evaluate,
    _examples,
    _parameter_hash,
    _prompt,
    _style_scores,
)
from scripts.course_experiments.modalities import (  # noqa: E402
    evaluate_modal,
    modal_inputs,
)
from scripts.course_experiments.text import arithmetic_records  # noqa: E402
from scripts.prepare_data import generate_records  # noqa: E402
from tiny_perceptron.alignment import distillation_kl  # noqa: E402
from tiny_perceptron.data import IGNORE, ByteTokenizer  # noqa: E402
from tiny_perceptron.training import load_checkpoint  # noqa: E402

OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_finish_t_10"
torch.set_num_threads(2)
tok = ByteTokenizer()
receipt = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_t_10-audit.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "scope": "Read every retained raw row and recompute denominators; pinned public CPU inference, no training rerun or GPU timing claim.",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_report(name):
    return json.loads((OUT / f"{PREFIX}-snapshot-docs_course-experiments_results_{name}.json").read_text())


def persist(suffix, value):
    path = OUT / f"{PREFIX}-{suffix}.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return path


text_report = read_report("distillation")
modal_report = read_report("multimodal_distillation")
tasks = text_report["results"]["tasks"]
mtasks = modal_report["results"]["tasks"]
datasets = {"attributes": split_records(generate_records("attributes-sft"), seed=42)}
arithmetic = split_records(arithmetic_records(), seed=42)
dates = []
for day in range(1, 25):
    date = f"2026-10-{day:02d}"
    dates.extend(
        [
            _conversation(
                f"task=date;date={date};confirm", "已確認" + date + "。", "date-" + date, style="clarification"
            ),
            _conversation(f"task=date;id={day};date=?;confirm", "請提供日期。", "date-" + date, style="clarification"),
        ]
    )
date_parts = split_records(dates, seed=42)
datasets["style_transfer"] = {
    split: [_style_record(row, style, True) for row in rows for style in ("concise", "vivid", "json")]
    + date_parts[split]
    for split, rows in arithmetic.items()
}
assets = json.loads((ROOT / "assets/training/manifest.json").read_text())["assets"]


def archive_rows(asset_id, basename):
    item = next(item for item in assets if item["id"] == asset_id)
    path = ROOT / item["archive"]
    assert digest(path.read_bytes()) == item["archive_sha256"]
    with tarfile.open(path) as archive:
        matches = [member for member in archive.getmembers() if Path(member.name).name == basename]
        assert len(matches) == 1
        data = archive.extractfile(matches[0]).read()
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()], digest(data)


stories, _ = archive_rows("tinystories", "tinystories-train-512.jsonl")
for row in stories:
    fallback = digest(json.dumps([{"text": row["text"]}], sort_keys=True, ensure_ascii=False).encode())
    row["family"] = row.get("text_sha256", fallback)
datasets["moe_to_dense"] = split_records(stories, seed=42)
for name in ["vqa", "joint"]:
    source = json.loads((ROOT / f"docs/course-experiments/results/{name}.json").read_text())
    datasets[name] = {split: value["records"] for split, value in source["results"]["data"]["splits"].items()}

data_receipts = {}
for name, parts in datasets.items():
    reference = tasks[name] if name in tasks else mtasks[name]
    snapshot = persist(f"dataset-{name}", parts)
    assert digest(snapshot.read_bytes()) == reference["data"]["sha256"], name
    identities = {split: {str(row["family"]) for row in rows} for split, rows in parts.items()}
    assert all(
        not identities[a] & identities[b]
        for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]
    )
    data_receipts[name] = {
        "path": str(snapshot.relative_to(ROOT)),
        "sha256": digest(snapshot.read_bytes()),
        "counts": {split: len(rows) for split, rows in parts.items()},
    }
receipt["datasets"] = data_receipts

full_rows = []
evaluation_receipts = []


def inspect_evaluations(value, locator):
    if isinstance(value, dict):
        key = "generated_samples" if "generated_samples" in value else "samples" if "samples" in value else None
        if key and value[key] and "examples" in value and "correct" in value:
            rows = value[key]
            if "generated_ids" in rows[0]:
                assert len(rows) == value["examples"]
                hits, eos, invalid = 0, 0, 0
                for row in rows:
                    expected = row.get("expected", row.get("target"))
                    ids = row["generated_ids"]
                    raw = ids[: ids.index(tok.eos_id)] if tok.eos_id in ids else ids
                    exact = raw == tok.encode(expected)
                    assert exact == row.get("exact", row.get("exact_match"))
                    ended = tok.eos_id in ids
                    assert ended == row.get("ended_with_eos", row.get("eos"))
                    assert tok.decode(raw) == row["generated"]
                    hits += exact
                    eos += ended
                    invalid += sum(i < 8 for i in raw)
                    full_rows.append({"locator": locator, **row})
                assert hits == value["correct"]
                assert hits / len(rows) == value["exact_match"]
                assert eos / len(rows) == value["eos_rate"]
                if "nll_sum" in value:
                    assert math.isclose(
                        value["nll_sum"] / value["supervised_tokens"], value["answer_nll"], abs_tol=1e-12
                    )
                evaluation_receipts.append(
                    {
                        "locator": locator,
                        "rows": len(rows),
                        "correct_recounted": hits,
                        "eos_recounted": eos,
                        "invalid_control_ids_recounted": invalid,
                    }
                )
        for key, child in value.items():
            if key not in ["samples", "generated_samples"]:
                inspect_evaluations(child, f"{locator}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            inspect_evaluations(child, f"{locator}[{index}]")


inspect_evaluations(text_report["results"], "distillation.results")
inspect_evaluations(modal_report["results"], "multimodal_distillation.results")
persist("all-generation-rows", full_rows)
receipt["all_retained_evaluations"] = evaluation_receipts

plan_receipts = []
training_steps = 0
for name, task in tasks.items():
    original = datasets[name]["train"][:32] if name == "moe_to_dense" else datasets[name]["train"]
    hard = task["hard_target_generation"]
    if hard:
        assert len(hard["audit"]) == len(original) == hard["records"]
        correct = 0
        for audit, row in zip(hard["audit"], original, strict=True):
            assert audit["family"] == row["family"]
            assert audit["question"] == row["messages"][0]["content"]
            assert audit["gold_answer"] == row["messages"][-1]["content"]
            ids = audit["teacher_ids"]
            raw = ids[: ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            correct += raw == tok.encode(audit["gold_answer"])
            assert audit["valid_target_tokens"] == len(ids)
            assert audit["teacher_correct"] == (raw == tok.encode(audit["gold_answer"]))
        assert correct == hard["correct"]
    init_by_width, plan_by_width = {}, {}
    for name_run, run in task["runs"].items():
        if "training" not in run:
            continue
        train = run["training"]
        examples = _examples(original, 128)
        counts = [int((labels != IGNORE).sum()) for _, labels in examples]
        if "teacher_hard" in name_run:
            counts = [len(row["teacher_ids"]) for row in hard["audit"]]
        rng = random.Random(42)
        plan = [[rng.randrange(len(examples)) for _ in range(16)] for _ in range(train["steps"])]
        plan_sha = digest(json.dumps(plan).encode())
        effective = sum(counts[index] for indices in plan for index in indices)
        assert plan_sha == train["batch_plan_sha256"]
        assert effective == train["effective_supervised_tokens"]
        width = name_run.split("_")[0]
        assert init_by_width.setdefault(width, train["initialization_sha256"]) == train["initialization_sha256"]
        assert plan_by_width.setdefault(width, plan_sha) == plan_sha
        for entry in train["loss_trace"]:
            expected_loss = (1 - train["alpha"]) * entry["ce"] + train["alpha"] * entry["kl"]
            assert math.isclose(entry["loss"], expected_loss, abs_tol=2e-6)
        training_steps += train["steps"]
        plan_receipts.append(
            {
                "task": name,
                "run": name_run,
                "steps": train["steps"],
                "effective_targets_reconstructed": effective,
                "plan_sha256": plan_sha,
                "initialization_sha256": train["initialization_sha256"],
            }
        )
    for split in ["validation", "test"]:
        examples = _examples(datasets[name][split], 128)
        count = sum(int((y != IGNORE).sum()) for _, y in examples)
        assert count == task[f"teacher_{split}"]["supervised_tokens"]
        assert len(examples) == task[f"teacher_{split}"]["nll_sequence_chunks"]
assert training_steps == 3900
receipt["text_student_updates_recounted"] = training_steps
for name, task in mtasks.items():
    for run_name, run in task["runs"].items():
        train = run["training"]
        rows = datasets[name]["train"]
        rng = random.Random(42)
        plan = [[rng.randrange(len(rows)) for _ in range(4)] for _ in range(train["steps"])]
        count = sum(len(tok.encode(rows[index]["answer"])) + 1 for indices in plan for index in indices)
        assert count == train["effective_answer_tokens"]
        assert digest(json.dumps(plan).encode()) == train["batch_plan_sha256"]
        assert train["steps"] == train["optimizer_updates"] == 350
        plan_receipts.append(
            {
                "task": name,
                "run": run_name,
                "steps": 350,
                "effective_targets_reconstructed": count,
                "plan_sha256": train["batch_plan_sha256"],
                "initialization_sha256": train["initialization_sha256"],
            }
        )
receipt["plans"] = plan_receipts

gsm, gsm_sha = archive_rows("gsm8k", "gsm8k-train-first200.jsonl")
selection = tasks["attributes"]["out_of_domain_gsm8k"]["selection"]
assert selection["source_file_sha256"] == gsm_sha
eligible, skipped = [], []
for index, row in enumerate(gsm):
    final = row["answer"].split("####")[-1].strip().replace(",", "")
    prefix = len(_prompt({"question": row["question"], "answer": final}))
    answer_count = len(tok.encode(final)) + 1
    if prefix + 24 <= 128 and answer_count <= 24:
        eligible.append(
            {
                "source_row": index,
                "question": row["question"],
                "answer": final,
                "prefix_tokens": prefix,
                "answer_tokens_with_eos": answer_count,
            }
        )
    else:
        skipped.append({"source_row": index, "prefix_tokens": prefix, "answer_tokens_with_eos": answer_count})
assert [row["source_row"] for row in eligible] == selection["selected_source_rows"] == [14, 94]
assert len(skipped) == selection["skipped_rows"] == 198
assert [row["answer"] for row in eligible] == ["5", "60"]
receipt["gsm8k"] = {
    "source_sha256": gsm_sha,
    "source_rows": len(gsm),
    "eligible": eligible,
    "skipped_rows": len(skipped),
}

# Independent scalar formula, mask and detached teacher-gradient check.
student_logits = torch.tensor([[[0.1, -0.5, 1.1], [0.2, 0.3, -0.4], [2.0, -1.0, 0.0]]], requires_grad=True)
teacher_logits = torch.tensor([[[0.4, 1.3, -0.1], [-0.2, 0.7, 0.9], [0.2, -0.1, 0.8]]], requires_grad=True)
labels = torch.tensor([[1, IGNORE, 2]])
temperature = 2.0
terms = []
for row_index in [0, 2]:
    t = teacher_logits.detach()[0, row_index].tolist()
    s = student_logits.detach()[0, row_index].tolist()
    pt = [math.exp(v / temperature) for v in t]
    qs = [math.exp(v / temperature) for v in s]
    pt = [v / sum(pt) for v in pt]
    qs = [v / sum(qs) for v in qs]
    terms.append(sum(p * math.log(p / q) for p, q in zip(pt, qs, strict=True)))
manual = sum(terms) / 2 * temperature**2
actual = distillation_kl(student_logits, teacher_logits, labels, temperature)
assert math.isclose(actual.item(), manual, abs_tol=3e-7)
actual.backward()
assert teacher_logits.grad is None
assert student_logits.grad[0, 1].abs().sum() == 0
receipt["objective"] = {
    "hand_formula": "T^2 / n_valid * sum_valid sum_vocab p_T(log(p_T)-log(q_T)); p_T=softmax(t/T), q_T=softmax(s/T)",
    "manual_scalar": manual,
    "executed": actual.item(),
    "teacher_grad": None,
    "masked_student_gradient": student_logits.grad[0, 1].tolist(),
    "tolerance": "3e-7, FP32 arithmetic",
}

manifest = json.loads((ROOT / "docs/course-experiments/public-models.json").read_text())
weight_receipts, replays = [], {}
ctx = Context("cpu", Path("/tmp/fact_finish_t_10"), Path("/tmp/fact_finish_t_10"), ROOT / "assets/training", seed=42)
for model_name in ["distillation", "multimodal_distillation"]:
    public = next(item for item in manifest["models"] if item["id"] == model_name)
    for file in public["files"]:
        if not file["output"].endswith(".pt"):
            continue
        path = ROOT / "checkpoints/course" / model_name / file["output"]
        assert path.stat().st_size == file["bytes"]
        assert digest(path.read_bytes()) == file["sha256"]
        model, payload = load_checkpoint(path, "cpu")
        state_hash = _parameter_hash(model)
        tensors = []
        for name, value in payload["model"].items():
            assert torch.isfinite(value).all()
            assert torch.equal(value, model.state_dict()[name].cpu())
            tensors.append(
                {
                    "name": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "elements": value.numel(),
                    "bytes": value.numel() * value.element_size(),
                    "sha256": digest(value.contiguous().numpy().tobytes()),
                }
            )
        weight_receipts.append(
            {
                "public_id": model_name,
                "file": file["output"],
                "url": f"https://huggingface.co/{public['repo']}/resolve/{public['revision']}/{file['path']}",
                "revision": public["revision"],
                "sha256": file["sha256"],
                "file_bytes": file["bytes"],
                "format": str(payload["format_version"]),
                "config": payload["config"],
                "modal_config": payload.get("modal_config"),
                "model_state_sha256": state_hash,
                "tensor_bytes": sum(row["bytes"] for row in tensors),
                "tensors": tensors,
            }
        )
        if model_name == "distillation":
            task_name = (
                "attributes"
                if file["output"].startswith("sft")
                else "style_transfer"
                if file["output"].startswith("style")
                else "moe_to_dense"
            )
            task = tasks[task_name]
            teacher = "-teacher.pt" in file["output"]
            stem = file["output"].split("-", 1)[1].removesuffix(".pt").replace("-", "_")
            expected = task["teacher_test"] if teacher else task["runs"][stem]["test"]
            if not teacher and "training" in task["runs"][stem]:
                assert state_hash == task["runs"][stem]["training"]["final_sha256"]
            actual_eval = _evaluate(
                model, datasets[task_name]["test"], "cpu", 96 if task_name == "style_transfer" else 24
            )
            assert [s["generated_ids"] for s in actual_eval["generated_samples"]] == [
                s["generated_ids"] for s in expected["generated_samples"]
            ]
            assert actual_eval["supervised_tokens"] == expected["supervised_tokens"]
            assert actual_eval["correct"] == expected["correct"]
            assert math.isclose(actual_eval["answer_nll"], expected["answer_nll"], abs_tol=3e-6)
            if task_name == "style_transfer":
                actual_eval["style"] = _style_scores(actual_eval, datasets[task_name]["test"])
                reference_style = task["teacher_style"] if teacher else task["runs"][stem]["style"]
                assert actual_eval["style"]["rubric"] == reference_style["rubric"]
            replays[file["output"]] = actual_eval
        else:
            name = "joint" if file["output"].startswith("joint") else "vqa"
            task = mtasks[name]
            teacher = "-teacher.pt" in file["output"]
            method = file["output"].split("-", 1)[1].removesuffix(".pt")
            expected = task["teacher_test"] if teacher else task["runs"][method]["test"]
            if not teacher:
                assert state_hash == task["runs"][method]["training"]["final_sha256"]
            actual_eval = evaluate_modal(model, datasets[name]["test"], ctx)
            assert [s["generated_ids"] for s in actual_eval["samples"]] == [
                s["generated_ids"] for s in expected["samples"]
            ]
            assert actual_eval["effective_tokens"] == expected["effective_tokens"]
            assert math.isclose(actual_eval["mean_token_nll"], expected["mean_token_nll"], abs_tol=3e-6)
            replays[file["output"]] = {"test": actual_eval}
            if not teacher:
                for ablation in ["blank_image", "blank_audio"] if name == "joint" else ["blank_image"]:
                    observed = evaluate_modal(model, datasets[name]["test"], ctx, ablation)
                    expected_ablation = task["runs"][method][ablation + "_test"]
                    assert [s["generated_ids"] for s in observed["samples"]] == [
                        s["generated_ids"] for s in expected_ablation["samples"]
                    ]
                    replays[file["output"]][ablation] = observed
            else:
                validation = evaluate_modal(model, datasets[name]["validation"], ctx)
                assert validation["correct"] == task["teacher_validation"]["correct"]
                replays[file["output"]]["validation"] = validation
            if name == "joint" and method == "ce_kl":
                original_ids = [s["generated_ids"] for s in actual_eval["samples"]]
                blank_ids = [s["generated_ids"] for s in replays[file["output"]]["blank_image"]["samples"]]
                assert original_ids == blank_ids
            row = datasets[name]["train"][random.Random(42).randrange(len(datasets[name]["train"]))]
            ids, labels, image, waveform, _ = modal_inputs(row, ctx)
            with torch.no_grad():
                output = model(ids, labels, image=image, waveform=waveform)
            prediction_rows = (output["labels"][0] != IGNORE).nonzero().flatten().tolist()
            replays[file["output"]]["alignment_probe"] = {
                "family": row["family"],
                "prediction_rows": prediction_rows,
                "answer_ids": output["labels"][0, output["labels"][0] != IGNORE].tolist(),
            }
        print("CPU replay verified", file["output"], flush=True)
persist("weights", weight_receipts)
persist("cpu-replays", replays)
receipt["weights_verified"] = len(weight_receipts)
receipt["replays_verified"] = len(replays)
ce = tasks["attributes"]["runs"]["w32_ce"]["test"]["generated_samples"]
kl = tasks["attributes"]["runs"]["w32_ce_kl"]["test"]["generated_samples"]
receipt["ce_to_kl"] = {
    "new_correct": sum(not a["exact"] and b["exact"] for a, b in zip(ce, kl, strict=True)),
    "new_wrong": sum(a["exact"] and not b["exact"] for a, b in zip(ce, kl, strict=True)),
    "teacher_agreement": [tasks["attributes"]["runs"][k]["teacher_agreement"] for k in ["w32_ce", "w32_ce_kl"]],
}
receipt["result"] = (
    "All assertions passed. Every raw evaluation row, reconstructed plan, selected GSM8K input and 24 pinned public checkpoints verified; CPU inference ID replay matches L4 records."
)
persist("audit-result", receipt)
print(
    json.dumps(
        {"result": receipt["result"], "objective": receipt["objective"], "ce_to_kl": receipt["ce_to_kl"]},
        ensure_ascii=False,
        indent=2,
    )
)
