"""Independent bounded CPU audit of lesson 7.14; no training or environment changes."""

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
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.check_technical_reviews import sections
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum, masked_loss

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_07_14_"
torch.set_num_threads(1)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


body = dict(sections(ROOT / "course/chapters/07.md"))["7.14"]
(OUT / f"{PREFIX}section.md").write_text(body, encoding="utf-8")
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(blocks[0], "course/chapters/07.md#7.14", "exec"), {})
(OUT / f"{PREFIX}original_stdout.txt").write_text(stdout.getvalue())

numeric = []
for score in (1.0, 5.0, 8.0):
    values = [0.0, 0.0, 3.0, score]
    z = torch.tensor([values], dtype=torch.float64, requires_grad=True)
    norm = sum(math.exp(x) for x in values)
    manual_probs = [math.exp(x) / norm for x in values]
    for target in (2, 3):
        target_tensor = torch.tensor([target], dtype=torch.long)
        loss = F.cross_entropy(z, target_tensor)
        manual = math.log(norm) - values[target]
        assert abs(float(loss.detach()) - manual) < 1e-12
        gradient = torch.autograd.grad(loss, z)[0][0]
        expected_gradient = torch.tensor(manual_probs, dtype=torch.float64)
        expected_gradient[target] -= 1
        assert torch.allclose(gradient, expected_gradient, atol=1e-12, rtol=0)
        numeric.append(
            {
                "logits": values,
                "dtype": str(z.dtype),
                "target": target,
                "probabilities": manual_probs,
                "manual_loss": manual,
                "torch_loss": float(loss.detach()),
                "gradient": gradient.tolist(),
                "tolerance": "1e-12 absolute for float64",
            }
        )

# A finite small update on an established correct preference need not reverse argmax.
prior = torch.tensor([[0.0, 0.0, 8.0, 0.0]], dtype=torch.float64, requires_grad=True)
wrong_loss = F.cross_entropy(prior, torch.tensor([3]))
grad = torch.autograd.grad(wrong_loss, prior)[0]
after = (prior - 0.01 * grad).detach()
assert int(prior.argmax()) == int(after.argmax()) == 2
prior_demo = {
    "before_logits": prior.detach().tolist(),
    "after_one_wrong_target_logit_gradient_step_lr_0.01": after.tolist(),
    "correct_argmax_before_and_after": 2,
    "scope": "Possibility example in logit space, not evidence of LLM robustness or quality",
}

tok = ByteTokenizer()
mapping = {str(i): tok.encode(str(i)) for i in range(4)}
assert mapping == {"0": [56], "1": [57], "2": [58], "3": [59]}

report_path = ROOT / "docs/course-experiments/results/sft_ablation.json"
report = json.loads(report_path.read_text())
formal = report["results"]
sft_path = ROOT / "docs/course-experiments/results/sft.json"
sft = json.loads(sft_path.read_text())
assert formal["before"]["A_attributes"] == sft["results"]["after"]
frozen_generator = OUT / f"{PREFIX}frozen_prepare_data.py.txt"
original_namespace = {"__name__": "independent_review_original_generator"}
exec(compile(frozen_generator.read_text(), str(frozen_generator), "exec"), original_namespace)
original_records = original_namespace["generate_records"]("attributes-sft")
assert original_records == generate_records("attributes-sft")
parts = split_records(original_records, seed=42)
split_records_output = {}
for name, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    data_path = OUT / f"{PREFIX}attributes_{name}.jsonl"
    data_path.write_bytes(raw)
    manifest = formal["data"]["attributes"][name]
    assert len(rows) == manifest["records"]
    assert digest(raw) == manifest["sha256"]
    families = {row["family"] for row in rows}
    split_records_output[name] = {"records": len(rows), "families": sorted(families), "sha256": digest(raw)}
    for row in rows:
        user, assistant = row["messages"]
        match = re.fullmatch(r"color=(.*);shape=(.*);pitch=(.*);(.*)", user["content"])
        assert match
        color, shape, pitch, question = match.groups()
        independently_correct = {
            "describe": shape,
            "shape?": shape,
            "color?": color,
            "pitch?": pitch,
            "joint?": shape + "," + pitch,
        }[question]
        assert assistant["content"] == independently_correct
for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
    assert not (set(split_records_output[left]["families"]) & set(split_records_output[right]["families"]))

clean = parts["train"]
noisy = copy.deepcopy(clean)
corruptions = []
for i, row in enumerate(noisy):
    answer = row["messages"][-1]["content"]
    if answer in ("circle", "square") and len(corruptions) < max(1, len(noisy) // 10):
        row["messages"][-1]["content"] = "square" if answer == "circle" else "circle"
        corruptions.append({"row": i, "family": row["family"], "correct": answer, "wrong": row["messages"][-1]["content"]})
assert corruptions == formal["corruptions"]
assert len(corruptions) == 4 and len({r["family"] for r in corruptions}) == 2
assert all(a["messages"][0] == b["messages"][0] for a, b in zip(clean, noisy, strict=True))
examples = {"clean": text_examples(clean, "sft", 128), "noisy": text_examples(noisy, "sft", 128)}
assert all(
    torch.equal(a[1] != IGNORE, b[1] != IGNORE)
    for a, b in zip(examples["clean"], examples["noisy"], strict=True)
)
schedules = {}
for name in ("clean", "noisy"):
    sampler = random.Random(42)
    schedule, total, changed_records_drawn, changed_targets = [], 0, 0, 0
    for _ in range(300):
        indices = sampler.choices(range(45), k=16)
        schedule.extend(indices)
        _, y, valid = pad_batch([examples[name][i] for i in indices])
        total += int((y != IGNORE).sum())
        assert bool(((y != IGNORE) <= valid).all())
        changed_records_drawn += sum(i in {0, 1, 5, 6} for i in indices)
        for i in indices:
            changed_targets += int((examples["clean"][i][1] != examples["noisy"][i][1]).sum())
    training = formal["runs"][name]["training"]
    assert total == training["effective_tokens"] == 33733
    assert training["steps"] == 300 and training["records"] == 45
    assert training["records_sha256"] == records_sha256(clean if name == "clean" else noisy)
    schedules[name] = {
        "drawn_records": len(schedule),
        "effective_targets": total,
        "schedule_sha256": digest(json.dumps(schedule).encode()),
        "corrupted_records_drawn": changed_records_drawn,
        "changed_target_positions": changed_targets,
        "wrong_answer_target_positions_including_EOS": changed_records_drawn * 7,
        "unchanged_EOS_in_corrupt_answers": changed_records_drawn,
    }
assert schedules["clean"] == schedules["noisy"]

full_samples, table = {}, {}
for name, expected_matches in (("clean", (2, 7)), ("noisy", (1, 6))):
    table[name] = {}
    full_samples[name] = {}
    for split, expected_count in zip(("validation", "test"), expected_matches, strict=True):
        evaluation = formal["runs"][name]["attributes"][split]
        samples = evaluation["samples"]
        assert len(samples) == len(parts[split]) == evaluation["records"]
        full_samples[name][split] = []
        for sample, row in zip(samples, parts[split], strict=True):
            assert sample["messages"] == row["messages"][:-1]
            assert sample["expected"] == row["messages"][-1]["content"]
            ids = sample["generated_ids"]
            ended = tok.eos_id in ids
            assert ended
            answer_ids = ids[: ids.index(tok.eos_id)]
            exact = answer_ids == tok.encode(row["messages"][-1]["content"])
            assert exact == sample["exact"] and ended == sample["eos"]
            assert tok.decode(answer_ids) == sample["generated"]
            full_samples[name][split].append({**sample, "independently_recomputed_exact": exact})
        targets = sum(int((y != IGNORE).sum()) for _, y in text_examples(parts[split], "sft", 128))
        expected_targets = 36 if split == "validation" else 69
        assert targets == evaluation["effective_tokens"] == expected_targets
        matches = sum(s["independently_recomputed_exact"] for s in full_samples[name][split])
        assert matches == evaluation["matches"] == expected_count
        nll = evaluation["nll_sum"] / targets
        assert abs(nll - evaluation["nll"]) < 1e-15
        table[name][split] = {
            "matches": matches,
            "records": len(samples),
            "effective_targets": targets,
            "nll_sum": evaluation["nll_sum"],
            "nll": nll,
            "nll_rounded_5dp": f"{nll:.5f}",
            "all_EOS": True,
        }
assert table["clean"]["test"]["nll_rounded_5dp"] == "0.37101"
assert table["noisy"]["test"]["nll_rounded_5dp"] == "0.34698"

# Meaningful software check: same effective-target reduction on a fresh small CPU model.
torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8))
x, y, valid = pad_batch(examples["clean"][:2])
logits = model(x, valid=valid)["logits"]
summed, count = loss_sum(logits, y)
loss = masked_loss(logits, y)
selected_logits, selected_labels = logits[y != IGNORE], y[y != IGNORE]
direct = F.cross_entropy(selected_logits, selected_labels)
assert int(count) == 14
assert torch.allclose(loss, direct)
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

official = OUT / f"{PREFIX}functional.py.txt"
installed = Path(inspect.getsourcefile(F.cross_entropy))
assert digest(installed.read_bytes()) == digest(official.read_bytes())
identity = {}
for filename in (
    "scripts/course_experiments/text.py",
    "scripts/course_experiments/common.py",
    "tiny_perceptron/model.py",
    "tiny_perceptron/data.py",
    "tiny_perceptron/training.py",
):
    actual = digest((ROOT / filename).read_bytes())
    assert actual == report["code_sha256"][filename]
    identity[filename] = actual
frozen_runner = OUT / f"{PREFIX}frozen_run.py.txt"
assert digest(frozen_runner.read_bytes()) == report["code_sha256"]["scripts/course_experiments/run.py"]

output = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "threads": "1",
    },
    "source_sha256": digest(body.encode()),
    "original_code_stdout": stdout.getvalue(),
    "numeric_and_gradients": numeric,
    "finite_update_possibility_demo": prior_demo,
    "byte_digit_mapping": mapping,
    "formal_report_sha256": digest(report_path.read_bytes()),
    "base_sft_report_sha256": digest(sft_path.read_bytes()),
    "base_heldout_results_exactly_equal_sft_after": True,
    "frozen_generator_records_exactly_equal_current_generator": True,
    "formal_environment": {k: report[k] for k in ("revision", "seed", "device", "torch_version", "python_version", "gpu", "step_scale", "status", "timing_scope", "elapsed_seconds")},
    "formal_generation": {"max_new_tokens": 32, "temperature": 0.0, "cache": False, "vocab": 264, "EOS": 2},
    "training_configuration": {"steps": 300, "batch_size": 16, "lr": 0.003, "optimizer": "fresh AdamW per branch", "seed": 42, "clip_norm": 1.0},
    "split_records": split_records_output,
    "corruptions": corruptions,
    "corruption_fraction": 4 / 45,
    "schedules": schedules,
    "table_recomputed_from_all_samples": table,
    "full_heldout_samples": full_samples,
    "software_forward_backward": {"shape": list(logits.shape), "dtype": str(logits.dtype), "effective_targets": int(count), "loss_sum": float(summed.detach()), "masked_loss": float(loss.detach()), "direct_selected_target_mean": float(direct.detach()), "all_parameter_gradients_finite": True},
    "official_installed_functional_sha256": digest(installed.read_bytes()),
    "current_code_matches_formal_hashes": identity,
    "frozen_runner_sha256_matches_formal_report": True,
    "scope": "No paid run, no retraining. CPU execution verifies arithmetic, implementation, exact data/sampler reconstruction and reported complete GPU observations; it does not reproduce GPU training or audit unavailable per-token logits.",
}
(OUT / f"{PREFIX}audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": "all audit assertions passed", "source_sha256": output["source_sha256"], "numeric": numeric, "schedules": schedules, "table": table}, ensure_ascii=False, indent=2))
