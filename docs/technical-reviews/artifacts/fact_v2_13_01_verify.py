"""Independent CPU audit of lesson 13.1, without training or modifying course files."""

import contextlib
import hashlib
import io
import json
import platform
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from playwright.sync_api import sync_playwright

from scripts.check_technical_reviews import sections
from scripts.course_experiments.behavior import _pair_examples, _preference_parts
from scripts.course_experiments.common import split_records
from scripts.course_experiments.posttraining import CONFIG, build_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
from tiny_perceptron.capstone import build_dataset, preference_pairs
from tiny_perceptron.data import IGNORE, pad_batch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
PREFIX = "fact_v2_13_01_"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def answer_counts(rows):
    return [[int((y != IGNORE).sum()) for _, y in pair] for pair in _pair_examples(rows, 128)]


def sampled_targets(rows, steps):
    counts = answer_counts(rows)
    sampler = random.Random(42)
    return sum(sum(lengths) for _ in range(steps) for lengths in sampler.choices(counts, k=8))


def main():
    report = {"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_01_verify.py"}
    report["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "platform": platform.platform(),
        "chromium": "/usr/bin/chromium",
        "dtype": "torch.float64 for the analytic loss check; torch.int64 for token labels",
    }
    chapter = ROOT / "course/chapters/13.md"
    raw = chapter.read_bytes()
    intro = raw[:re.search(rb"^## ", raw, re.M).start()]
    body = dict(sections(chapter))["13.1"]
    assert body.encode() == raw[raw.index(body.encode()):raw.index(body.encode()) + len(body.encode())]
    (OUT / (PREFIX + "intro.txt")).write_bytes(intro)
    (OUT / (PREFIX + "section.md")).write_bytes(body.encode())
    report["source_sha256"] = digest(body.encode())
    report["intro_sha256"] = digest(intro)
    report["prerequisites"] = {
        f"course/chapters/{name}#{lesson}": digest(dict(sections(ROOT / "course/chapters" / name))[lesson].encode())
        for name, ids in [("08.md", ["8.5", "8.1", "8.3"]), ("09.md", ["9.1"]), ("13.md", ["13.7"])]
        for lesson in ids
    }
    code = re.search(r"```python\n(.*?)```", body, re.S)[1]
    buffer = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "13.1 exact fenced snippet", "exec"), namespace)
    expected = "條件 只回數字：2+2=?\n比較 4 ／ 4，就像兩雙筷子共有四根。\n理由 先正確，再遵循只回數字 選擇 A\n"
    assert buffer.getvalue() == expected
    namespace["pair"]["prompt"] = namespace["pair"]["prompt"].replace("只回數字：", "")
    assert namespace["chosen"] == "A"
    report["snippet"] = {"output": buffer.getvalue(), "chosen_after_prompt_only_change": namespace["chosen"]}
    original_path = ROOT / "docs/course-experiments/results/dpo.json"
    original = json.loads(original_path.read_bytes())
    style = json.loads((ROOT / "docs/course-experiments/results/style.json").read_bytes())
    (OUT / (PREFIX + "dpo_original_report.json")).write_bytes(original_path.read_bytes())
    arithmetic = split_records(arithmetic_records(), seed=42)
    pairs = _preference_parts(arithmetic)
    assert sum(map(len, pairs.values())) == 8 * 8 == 64
    assert {name: len(rows) for name, rows in pairs.items()} == {"train": 49, "validation": 8, "test": 7}
    families = {name: {row["family"] for row in rows} for name, rows in pairs.items()}
    assert not any(families[a] & families[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
    assert len(set.union(*families.values())) == 36
    audit = {}
    for name, rows in pairs.items():
        assert digest(jsonl_bytes(rows)) == original["results"]["data"][name]["sha256"]
        assert digest(jsonl_bytes(arithmetic[name])) == style["results"]["arithmetic_data"][name]["sha256"]
        for row in rows:
            a, b = map(int, row["prompt"].removesuffix("=?").split("+"))
            assert row["chosen"] == str(a + b)
            assert row["rejected"] == str(a + b + 1)
            assert row["family"] == f"{min(a, b)}+{max(a, b)}"
        counts = answer_counts(rows)
        examples = _pair_examples(rows, 128)
        x, y, valid = pad_batch([example[0] for example in examples])
        assert x.dtype == y.dtype == torch.int64 and valid.dtype == torch.bool
        assert int((y != IGNORE).sum()) == sum(lengths[0] for lengths in counts)
        audit[name] = {
            "records": len(rows), "families": sorted(families[name]), "pair_jsonl_sha256": digest(jsonl_bytes(rows)),
            "style_jsonl_sha256": digest(jsonl_bytes(arithmetic[name])), "rows": rows,
            "answer_targets_chosen_rejected_including_eos": counts,
            "targets_chosen_rejected": [sum(lengths[side] for lengths in counts) for side in [0, 1]],
            "padded_tensor_shape": list(x.shape), "mask": "assistant UTF-8 bytes plus EOS; prompt and padding = -100",
        }
    report["dataset_audit"] = audit
    report["original_run"] = {
        key: original[key] for key in ["revision", "seed", "device", "gpu", "python_version", "torch_version", "step_scale", "evidence_status", "elapsed_seconds", "timing_scope"]
    }
    report["original_run"]["audit_scope"] = "Original CUDA result inspected; only deterministic data and target counts reproduced on this CPU. No GPU timing or training reproduction."
    format_pairs = {name: [{**row, "rejected": row["chosen"] + "; answer complete"} for row in rows] for name, rows in pairs.items()}
    effective = {"model": sampled_targets(pairs["train"], 250), "beta1": sampled_targets(pairs["train"], 250), "format-model": sampled_targets(format_pairs["train"], 200)}
    assert effective == {"model": 9448, "beta1": 9448, "format-model": 34554}
    for name in ["model", "beta1"]:
        assert original["results"]["runs"][name]["training"]["effective_answer_tokens_both_sides"] == effective[name]
    assert original["results"]["format_only"]["training"]["effective_answer_tokens_both_sides"] == effective["format-model"]
    report["training_configuration_audit"] = {
        "seed": 42, "arithmetic_steps": 250, "format_steps": 200, "pair_batch_size": 8,
        "optimizer": "AdamW, lr=0.001, constant schedule, clip_grad_norm_=1.0; default optimizer settings",
        "betas": [0.1, 1.0], "model": "style/content.pt, width=64, layers=2, max_length=128; same initial state per branch",
        "answer_targets_both_sides_including_eos": effective,
        "count_scope": "policy answer labels in both sides, not extra reference-forward work or FLOPs",
    }
    sample_audit = {}
    for stage, run in [("before", original["results"]["before"]), *[(key, value["preference"]) for key, value in original["results"]["runs"].items()], ("format", original["results"]["format_only"]["preference"])]:
        stage_rows = format_pairs if stage == "format" else pairs
        sample_audit[stage] = {}
        for split in ["validation", "test"]:
            samples = run[split]["samples"]
            assert len(samples) == run[split]["records"] == len(stage_rows[split])
            assert [r["prompt"] for r in samples] == [r["prompt"] for r in stage_rows[split]]
            for observed, expected_row, counts in zip(samples, stage_rows[split], answer_counts(stage_rows[split]), strict=True):
                assert all(observed[key] == expected_row[key] for key in ["family", "prompt", "chosen", "rejected"])
                assert [observed["chosen_answer_tokens"], observed["rejected_answer_tokens"]] == counts
                assert abs(observed["policy_margin"] - (observed["policy_chosen_logp"] - observed["policy_rejected_logp"])) < 1e-9
                assert abs(observed["relative_margin"] - (observed["policy_margin"] - observed["reference_margin"])) < 1e-9
            assert run[split]["chosen_higher_absolute_probability"] == sum(r["policy_margin"] > 0 for r in samples)
            assert run[split]["relative_preference_improved"] == sum(r["relative_margin"] > 0 for r in samples)
            sample_audit[stage][split] = samples
    report["per_question_audit"] = sample_audit
    p = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)
    r = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    baseline = torch.tensor([-2.0], dtype=torch.float64)
    loss = dpo_loss(p, r, baseline, torch.tensor([-3.0], dtype=torch.float64))
    loss.backward()
    assert abs(loss.item() - 0.6931471805599453) < 1e-14
    logits = torch.zeros((1, 3, 4), dtype=torch.float64)
    labels = torch.tensor([[IGNORE, 1, 2]])
    summed = sequence_log_probability(logits, labels).item()
    assert abs(summed - (-2 * 1.3862943611198906)) < 1e-14
    report["loss_cpu_check"] = {"equal_policy_reference_loss": loss.item(), "chosen_gradient": p.grad.item(), "rejected_gradient": r.grad.item(), "masked_sum_two_targets_vocab4": summed}
    finite = build_records()
    assert len(finite) == 165
    for row in finite:
        assert row["candidate_meets_full_request"][row["expected_action"]]
    report["finite_policy_intro_audit"] = {"records": len(finite), "config": CONFIG, "candidate_source": finite[0]["candidate_source"], "label_source": finite[0]["label_source"], "examples": [finite[0], finite[1], finite[2]]}
    capstone_splits, _ = build_dataset(seed=42)
    report["multimodal_preference_scope"] = {}
    for name, rows in capstone_splits.items():
        demo_pairs = preference_pairs(rows)
        assert all(pair["chosen"] == pair["row"]["answer"] and pair["rejected"] == pair["chosen"] + "，祝你愉快！" for pair in demo_pairs)
        report["multimodal_preference_scope"][name] = {"pairs": len(demo_pairs), "tasks": sorted({pair["row"]["task"] for pair in demo_pairs}), "fact_preserved": True}
    report["source_hashes"] = {}
    for name in ["scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "scripts/course_experiments/behavior.py", "tiny_perceptron/data.py", "tiny_perceptron/alignment.py", "tiny_perceptron/capstone.py", "scripts/course_experiments/posttraining.py"]:
        h = digest((ROOT / name).read_bytes())
        report["source_hashes"][name] = {"current_sha256": h, "original_run_sha256": original["code_sha256"].get(name), "same_as_original_run": h == original["code_sha256"].get(name)}
    figure = ROOT / "course/figures/multimodal_preference_pair.svg"
    xml = ET.fromstring(figure.read_bytes())
    assert xml.attrib["viewBox"] == "0 0 900 360"
    report["figure"] = {"sha256": digest(figure.read_bytes()), "labels": [item.text for item in xml.findall("{http://www.w3.org/2000/svg}text")], "paths": [item.attrib["d"] for item in xml.findall("{http://www.w3.org/2000/svg}path")]}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 900, "height": 360}, device_scale_factor=1)
        page.set_content('<style>html,body{margin:0;padding:0;width:900px;height:360px}</style>' + figure.read_text())
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(OUT / (PREFIX + "figure.png")))
        report["environment"]["chromium_version"] = browser.version
        report["figure"]["screenshot"] = PREFIX + "figure.png"
        report["figure"]["render_method"] = "Chromium native SVG rendering from exact XML via page.set_content; local file navigation is blocked by browser policy"
        browser.close()
    report["test_usage_scope"] = "run_dpo evaluates before.validation and before.test before branch training; fixed beta/steps do not use these scores programmatically for selection. Test exclusivity until final evaluation is not established."
    report["result"] = "All deterministic arithmetic, split identity, per-question labels/targets and CPU checks passed; no training performed."
    (OUT / (PREFIX + "execution.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "counts": {k: len(v) for k, v in pairs.items()}, "effective_targets": effective, "output": str(OUT / (PREFIX + "execution.json"))}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
