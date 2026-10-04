"""Independent bounded CPU audit of 7.17; no formal training is repeated."""

import contextlib
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from playwright.sync_api import sync_playwright

from scripts.check_technical_reviews import sections
from scripts.course_experiments.common import split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.training import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_07_17"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def weights(model):
    return digest(b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values()))


def main():
    torch.set_num_threads(1)
    section = dict(sections(ROOT / "course/chapters/07.md"))["7.17"]
    (DEST / f"{PREFIX}_section.txt").write_text(section, encoding="utf-8")
    prerequisites = {}
    for chapter, lesson in [("07", "7.11"), ("07", "7.1"), ("07", "7.12"), ("05", "5.12"), ("19", "19.4")]:
        raw = dict(sections(ROOT / f"course/chapters/{chapter}.md"))[lesson]
        prerequisites[lesson] = {"source": f"course/chapters/{chapter}.md#{lesson}", "sha256": digest(raw.encode())}
    block = re.findall(r"```python\n(.*?)```", section, re.S)[0]
    numeric = []
    for cost in [20, 5]:
        output = io.StringIO()
        namespace = {}
        with contextlib.redirect_stdout(output):
            exec(block.replace("article_cost, checked_chat_cost = 1, 20", f"article_cost, checked_chat_cost = 1, {cost}"), namespace)
        observed = [int(line.split()[-1]) for line in output.getvalue().splitlines()]
        expected = [1000, 50 * cost, 1000 * cost]
        assert observed == expected
        numeric.append({"checked_chat_cost": cost, "expected": expected, "observed": observed, "stdout": output.getvalue()})

    formal_path = ROOT / "docs/course-experiments/results/sft.json"
    formal = json.loads(formal_path.read_text())
    ablation = json.loads((ROOT / "docs/course-experiments/results/sft_ablation.json").read_text())
    results = formal["results"]
    parts = split_records(generate_records("attributes-sft"), seed=formal["seed"])
    dataset = {}
    for name, rows in parts.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        assert digest(raw) == results["data"][name]["sha256"]
        target_count = sum(int((y != IGNORE).sum()) for _, y in text_examples(rows, mode="sft"))
        dataset[name] = {"records": len(rows), "families": sorted({row["family"] for row in rows}), "sha256": digest(raw), "effective_targets": target_count}
        (DEST / f"{PREFIX}_{name}.jsonl").write_bytes(raw)
    families = {name: set(item["families"]) for name, item in dataset.items()}
    assert not families["train"] & (families["validation"] | families["test"])
    assert not families["validation"] & families["test"]

    tokenizer = ByteTokenizer()
    stages = {"scratch_sft": results["after"], "text_before_sft": results["pretrain_then_sft"]["before_sft"], "text_after_sft": results["pretrain_then_sft"]["after_sft"]}
    audits = {}
    for stage, evaluations in stages.items():
        audits[stage] = {}
        for split, evaluation in evaluations.items():
            samples = evaluation["samples"]
            assert len(samples) == len(parts[split])
            matches = 0
            for sample, row in zip(samples, parts[split], strict=True):
                assert sample["messages"] == row["messages"][:-1]
                assert sample["expected"] == row["messages"][-1]["content"]
                generated = sample["generated_ids"]
                eos = tokenizer.eos_id in generated
                raw = generated[:generated.index(tokenizer.eos_id)] if eos else generated
                exact = raw == tokenizer.encode(sample["expected"])
                assert exact == sample["exact"] and eos == sample["eos"]
                assert tokenizer.decode(raw) == sample["generated"]
                matches += int(exact)
            assert matches == evaluation["matches"]
            assert evaluation["effective_tokens"] == dataset[split]["effective_targets"]
            assert abs(evaluation["nll_sum"] / dataset[split]["effective_targets"] - evaluation["nll"]) < 1e-12
            audits[stage][split] = evaluation
    assert [audits[name]["test"]["matches"] for name in stages] == [5, 1, 7]

    text_records = [{"text": row["messages"][0]["content"] + row["messages"][1]["content"], "family": row["family"]} for row in parts["train"]]
    training = results["training"]
    pretraining = results["pretrain_then_sft"]["pretraining"]
    continuation = results["pretrain_then_sft"]["sft"]
    expected_text_digest = digest(json.dumps(text_records, sort_keys=True, ensure_ascii=False).encode())
    assert expected_text_digest == pretraining["records_sha256"]
    assert training["records_sha256"] == continuation["records_sha256"]
    assert [training["steps"], pretraining["steps"], continuation["steps"]] == [900, 250, 900]
    sampled_target_totals = {}
    for stage, rows, mode, report in [("scratch_sft", parts["train"], "sft", training), ("text", text_records, "text", pretraining), ("continuation_sft", parts["train"], "sft", continuation)]:
        examples = text_examples(rows, mode=mode)
        sampler = random.Random(42)
        count = sum(int((y != IGNORE).sum()) for _ in range(report["steps"]) for _, y in sampler.choices(examples, k=16))
        assert count == report["effective_tokens"]
        sampled_target_totals[stage] = count

    frozen = {}
    for source in ["scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "tiny_perceptron/training.py", "scripts/prepare_data.py"]:
        raw = subprocess.check_output(["git", "show", f"{formal['revision']}:{source}"], cwd=ROOT)
        if source in formal["code_sha256"]:
            assert digest(raw) == formal["code_sha256"][source]
        name = source.replace("/", "_").replace(".py", ".txt")
        path = DEST / f"{PREFIX}_formal_{name}"
        path.write_bytes(raw)
        frozen[source] = {"revision": formal["revision"], "sha256": digest(raw), "snapshot": str(path.relative_to(ROOT))}

    torch.manual_seed(42)
    base = TinyLM(ModelConfig(width=8))
    base_hash = weights(base)
    x, y, valid = pad_batch(text_examples(parts["train"][:2], mode="sft"))
    with tempfile.TemporaryDirectory(prefix=PREFIX) as temporary:
        path = Path(temporary) / "base.pt"
        save_checkpoint(path, base, step=0, metadata={"task": "independent bounded audit"})
        checkpoint_hash = digest(path.read_bytes())
        branches = []
        for index in range(2):
            branch, payload = load_checkpoint(path)
            before_hash = weights(branch)
            assert before_hash == base_hash
            optimizer = torch.optim.AdamW(branch.parameters(), lr=0.003)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(branch(x[index:index + 1], valid=valid[index:index + 1])["logits"], y[index:index + 1])
            loss.backward()
            optimizer.step()
            after_hash = weights(branch)
            assert after_hash != before_hash
            branches.append({"before_hash": before_hash, "after_hash": after_hash, "loss_before_single_update": float(loss.detach()), "saved_step": payload["step"]})
        assert weights(base) == base_hash
        assert branches[0]["after_hash"] != branches[1]["after_hash"]

    svg_path = ROOT / "course/figures/posttrain_stages.svg"
    tree = ET.fromstring(svg_path.read_text())
    ns = {"svg": "http://www.w3.org/2000/svg"}
    labels = [element.text for element in tree.findall(".//svg:text", ns)]
    svg_paths = [element.attrib for element in tree.findall(".//svg:path", ns)]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 820, "height": 460}, device_scale_factor=1)
        page.set_content('<html><body style="margin:0">' + svg_path.read_text() + "</body></html>")
        page.evaluate("document.fonts.ready")
        screenshot = DEST / f"{PREFIX}_posttrain_stages.png"
        page.screenshot(path=str(screenshot), full_page=True)
        browser_version = browser.version
        browser.close()

    output = {
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform(), "chromium": browser_version, "threads": "1"},
        "source_sha256": digest(section.encode()), "prerequisites": prerequisites,
        "numeric_original_and_exercise": numeric,
        "formal_run": {"revision": formal["revision"], "seed": formal["seed"], "device": formal["device"], "gpu": formal["gpu"], "torch": formal["torch_version"], "step_scale": formal["step_scale"], "status": formal["status"], "file_sha256": digest(formal_path.read_bytes())},
        "dataset": dataset, "all_45_held_out_generation_records": audits,
        "training_configuration": {"width": 64, "layers": 2, "batch_size": 16, "lr": 0.003, "max_length": 128, "greedy_temperature": 0.0, "max_new_tokens": 32, "scratch_updates": 900, "text_updates": 250, "continuation_updates": 900, "text_train_records": 45, "sampled_target_totals": sampled_target_totals},
        "frozen_formal_code": frozen,
        "ablation_scope": {"sections": ablation["results"]["sections"], "matched_budget": ablation["results"]["matched_budget"], "scope": ablation["results"]["scope"], "supports_7_17_before_after_counts": False},
        "checkpoint_bounded_probe": {"base_weights_sha256": base_hash, "checkpoint_sha256": checkpoint_hash, "reload_and_update_two_branches": branches, "base_unchanged": True, "updates_per_branch": 1},
        "figure": {"source_sha256": digest(svg_path.read_bytes()), "viewBox": tree.attrib["viewBox"], "labels": labels, "paths": svg_paths, "render": str(screenshot.relative_to(ROOT)), "render_sha256": digest(screenshot.read_bytes())},
        "scope": "Formal CUDA outcomes are audited from the complete recorded results, dataset hashes and run-revision source. No formal training or checkpoint inference was rerun. CPU execution only demonstrates arithmetic, target denominators and save/reload/one-update mechanics."
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
