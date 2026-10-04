"""Fresh, CPU-only evidence for section 13.8; no training or environment changes."""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
import platform
import random
import re
import shutil
import subprocess
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments import behavior
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.data import ByteTokenizer, render_chat

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_08"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, content):
    path = OUT / f"{PREFIX}_{name}"
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def execute(code):
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        exec(compile(code, "13.8-original", "exec"), namespace)
    return output.getvalue(), namespace


chapter = ROOT / "course/chapters/13.md"
body = dict(sections(chapter))["13.8"]
write("section.txt", body)
prerequisites = []
for name, ids in [("13.md", {"13.1", "13.4"}), ("09.md", {"9.3"})]:
    for section_id, prerequisite in sections(ROOT / "course/chapters" / name):
        if section_id in ids:
            prerequisites.append({"source": f"course/chapters/{name}#{section_id}",
                                  "section_sha256": hashlib.sha256(prerequisite.encode()).hexdigest(),
                                  "text": prerequisite})
write("prerequisites.json", prerequisites)
code = re.findall(r"```python\n(.*?)\n```", body, re.S)
assert len(code) == 1
code = code[0] + "\n"
write("snippet.py", code)
stdout, namespace = execute(code)
pair = namespace["pair"]
expected = ("真值 4\n較佳 2+2等於4；把兩組各2個物件合起來，就是4個。\n"
            "較差 完全正確，你說得很好！\n偏好理由 溫和修正錯誤，並提供可核對解釋\n")
assert stdout == expected
write("snippet_stdout.txt", stdout)
tree = ast.parse(code)
assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
assert [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)] == ["print"] * 4
exercise = code.replace(pair["chosen"], "笨蛋，當然是4").replace(
    pair["reason"], "算術正確，但侮辱使用者；語氣差於原先溫和解釋，不應視為同等優質正例"
)
exercise_stdout, exercise_namespace = execute(exercise)
assert exercise_namespace["pair"]["chosen"] == "笨蛋，當然是4"
assert exercise_stdout.startswith("真值 4\n較佳 笨蛋，當然是4\n")
write("exercise.py", exercise)
write("exercise_stdout.txt", exercise_stdout)

# A label swap changes the sign of the DPO score-space gradient, not arithmetic truth.
score_directions = []
for flipped in (False, True):
    factual = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    agreeing = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    ref_factual, ref_agreeing = torch.tensor([-4.0]), torch.tensor([-3.0])
    loss = (dpo_loss(agreeing, factual, ref_agreeing, ref_factual, beta=0.1) if flipped else
            dpo_loss(factual, agreeing, ref_factual, ref_agreeing, beta=0.1))
    loss.backward()
    sign = 1 if flipped else -1
    assert math.isclose(loss.item(), math.log(2), abs_tol=1e-12)
    assert math.isclose(factual.grad.item(), sign * 0.05, abs_tol=1e-12)
    assert math.isclose(agreeing.grad.item(), -sign * 0.05, abs_tol=1e-12)
    score_directions.append({"agreeing_response_marked_chosen": flipped, "loss": loss.item(),
                             "factual_logp_gradient": factual.grad.item(),
                             "agreeing_logp_gradient": agreeing.grad.item()})

original_pair = behavior._pair_examples([pair], 256)[0]
changed_reason_pair = behavior._pair_examples([{**pair, "reason": "completely different metadata"}], 256)[0]
assert all(torch.equal(a, b) for side_a, side_b in zip(original_pair, changed_reason_pair, strict=True)
           for a, b in zip(side_a, side_b, strict=True))
token_details = [{"side": side, "input_shape": list(x.shape), "dtype": str(x.dtype),
                  "effective_answer_targets": int((y != -100).sum()), "final_target": int(y[-1])}
                 for side, (x, y) in zip(("chosen", "rejected"), original_pair, strict=True)]

# Inspect the entire existing experiment result and every saved preference/generation row.
# This is an audit of recorded results, not a rerun of its NVIDIA L4 training.
formal_path = ROOT / "docs/course-experiments/results/dpo.json"
formal = json.loads(formal_path.read_text())
formal_result = formal["results"]
data_root = ROOT / "outputs/text-behavior-interface-check/dpo"
raw_sets = {}
split_audits = []
for group, manifest in [("data", formal_result["data"]),
                        ("format-pairs", formal_result["format_only"]["data"]),
                        ("ultrafeedback-excerpts", formal_result["ultrafeedback_pilot"]["data"])]:
    raw_sets[group] = {}
    family_sets = []
    for split in ("train", "validation", "test"):
        source_path = data_root / group / f"{split}.jsonl"
        assert digest(source_path) == manifest[split]["sha256"]
        rows = [json.loads(line) for line in source_path.read_text().splitlines()]
        assert len(rows) == manifest[split]["records"]
        assert len({row["family"] for row in rows}) == manifest[split]["families"]
        raw_sets[group][split] = rows
        family_sets.append({row["family"] for row in rows})
        raw_copy = write(f"raw_{group}_{split}.jsonl", source_path.read_text())
        split_audits.append({"group": group, "split": split, "records": len(rows),
                             "families": len(family_sets[-1]), "sha256": digest(source_path),
                             "preserved_path": str(raw_copy.relative_to(ROOT))})
    assert not family_sets[0] & family_sets[1]
    assert not family_sets[0] & family_sets[2]
    assert not family_sets[1] & family_sets[2]

token_audits = []
branches = [("model", "data", formal_result["runs"]["model"]["training"]),
            ("beta1", "data", formal_result["runs"]["beta1"]["training"]),
            ("format", "format-pairs", formal_result["format_only"]["training"]),
            ("ultrafeedback", "ultrafeedback-excerpts", formal_result["ultrafeedback_pilot"]["training"])]
for name, group, training in branches:
    sampler = random.Random(formal["seed"])
    lengths = []
    for row in raw_sets[group]["train"]:
        effective = 0
        for side in ("chosen", "rejected"):
            _, labels = render_chat([{"role": "user", "content": row["prompt"]},
                                     {"role": "assistant", "content": row[side]}])
            assert labels[-1].item() == ByteTokenizer().eos_id
            effective += int((labels != -100).sum())
        lengths.append(effective)
    observed = sum(sum(sampler.choices(lengths, k=8)) for _ in range(training["steps"]))
    assert observed == training["effective_answer_tokens_both_sides"]
    token_audits.append({"branch": name, "steps": training["steps"], "batch_pairs": 8,
                         "seed": formal["seed"], "effective_answer_tokens_both_sides": observed,
                         "definition": "sum of non--100 targets on both sampled answers, including EOS"})

preference_audits = []
generation_audits = []


def audit_tables(value, path):
    if isinstance(value, dict):
        if "samples" in value and value["samples"]:
            samples = value["samples"]
            if "policy_margin" in samples[0]:
                assert len(samples) == value["records"]
                for row in samples:
                    assert math.isclose(row["policy_margin"],
                                        row["policy_chosen_logp"] - row["policy_rejected_logp"], abs_tol=1e-9)
                    assert math.isclose(row["relative_margin"],
                                        row["policy_margin"] - row["reference_margin"], abs_tol=1e-9)
                    for side in ("chosen", "rejected"):
                        _, labels = render_chat([{"role": "user", "content": row["prompt"]},
                                                 {"role": "assistant", "content": row[side]}])
                        assert int((labels != -100).sum()) == row[f"{side}_answer_tokens"]
                assert sum(row["policy_margin"] > 0 for row in samples) == value["chosen_higher_absolute_probability"]
                assert sum(row["relative_margin"] > 0 for row in samples) == value["relative_preference_improved"]
                preference_audits.append({"path": path, "records": len(samples), "all_samples": samples})
            elif "generated" in samples[0]:
                assert all(row["exact"] == (row["generated"] == row["expected"]) for row in samples)
                generation_audits.append({"path": path, "all_samples": samples})
        for key, child in value.items():
            if key != "samples":
                audit_tables(child, f"{path}.{key}")


audit_tables(formal_result, "results")
assert "13.8" not in formal_result["sections"]
write("formal_dpo.json", formal)
write("formal_all_rows.json", {"preference": preference_audits, "generation": generation_audits})
write("formal_code_excerpts.txt", "\n\n".join(inspect.getsource(fn) for fn in (
    behavior._pair_examples, behavior._dpo_train, behavior._preference_evaluate,
    behavior._natural_dpo_pilot, behavior.run_dpo)))

paper_snapshots = []
for title, original in [
    ("sycophancy_v4", "outputs/technical-sources/text-behavior/sycophancy2023.pdf"),
    ("dpo_v3", "outputs/posttrain-design/ppo-reference/dpo-original.pdf"),
    ("instructgpt_v1", "outputs/posttrain-design/ppo-reference/instructgpt-original.pdf"),
]:
    pdf = OUT / f"{PREFIX}_{title}.pdf"
    text = OUT / f"{PREFIX}_{title}.txt"
    shutil.copyfile(ROOT / original, pdf)
    conversion = subprocess.run(["pdftotext", "-layout", str(pdf), str(text)], capture_output=True, text=True)
    assert conversion.returncode == 0, conversion.stderr
    paper_snapshots.append({"original": original, "pdf": str(pdf.relative_to(ROOT)),
                            "pdf_sha256": digest(pdf), "text": str(text.relative_to(ROOT)),
                            "text_sha256": digest(text), "conversion_command": f"pdftotext -layout {pdf} {text}"})

audit = {
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                    "platform": platform.platform(), "cuda_available": str(torch.cuda.is_available())},
    "section_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "snippet": {"stdout": stdout, "exact_output": True, "call_names": ["print"] * 4,
                "imports": [], "model_updates": 0, "automated_preference_scoring": False},
    "arithmetic_derivation": "Two disjoint groups of cardinality 2 give 1+1+1+1=4; 2+2=4, and 4!=5.",
    "exercise_stdout": exercise_stdout,
    "dpo_label_swap_score_space_only": score_directions,
    "pair_rendering": token_details,
    "reason_metadata_is_ignored_by_formal_pair_renderer": True,
    "formal_experiment": {
        "source": str(formal_path.relative_to(ROOT)), "sha256": digest(formal_path),
        "metadata": {k: formal[k] for k in ("revision", "device", "seed", "torch_version", "python_version",
                                             "gpu", "elapsed_seconds", "timing_scope", "step_scale")},
        "sections": formal_result["sections"], "split_audits": split_audits,
        "token_audits": token_audits, "preference_tables": len(preference_audits),
        "preference_rows_checked": sum(x["records"] for x in preference_audits),
        "generated_rows_checked": sum(len(x["all_samples"]) for x in generation_audits),
        "scope": formal_result["scope"],
        "no_sycophancy_measurement": "13.8 is not a declared experiment section; arithmetic, format and cropped "
                                     "UltraFeedback comparisons do not measure agreement with false user premises.",
        "retraining_performed": False,
    },
    "paper_snapshots": paper_snapshots,
    "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
        ROOT / "tiny_perceptron/alignment.py", ROOT / "tiny_perceptron/data.py",
        ROOT / "scripts/course_experiments/behavior.py")},
    "result": "All assertions passed; code prints the declared arithmetic and labels. Label direction is not a truth checker. "
              "Existing DPO records provide no measured anti-sycophancy result for section 13.8.",
}
write("audit.json", audit)
print(json.dumps(audit, ensure_ascii=False, indent=2))
