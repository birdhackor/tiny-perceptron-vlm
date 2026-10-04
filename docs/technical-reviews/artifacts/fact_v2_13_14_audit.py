"""Fresh independent audit of lesson 13.14; persist actual CPU and render evidence."""

import hashlib
import inspect
import json
import math
import platform
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from playwright.sync_api import sync_playwright
from torch.nn import functional as F

from scripts.check_technical_reviews import sections
from scripts.course_experiments import posttraining as experiment
from tiny_perceptron.posttraining import (
    FiniteResponsePolicy,
    FiniteRewardModel,
    FiniteValueModel,
    exact_kl,
)

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_14_"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    path = ART / (PREFIX + name)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run(command, output_name):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (ART / (PREFIX + output_name + "_stdout.txt")).write_text(result.stdout, encoding="utf-8")
    (ART / (PREFIX + output_name + "_stderr.txt")).write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    environment = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "torch.float32 for lesson; torch.float64 cross-check",
        "cpu_threads_at_start": str(torch.get_num_threads()),
        "grad_enabled": str(torch.is_grad_enabled()),
    }
    bodies = dict(sections(ROOT / "course/chapters/13.md"))
    body = bodies["13.14"]
    (ART / (PREFIX + "section.md")).write_text(body, encoding="utf-8")
    (ART / (PREFIX + "prerequisites.md")).write_text(
        "\n".join(bodies[key] for key in ("13.10", "13.11", "13.12")), encoding="utf-8"
    )
    raw_snippet = re.search(r"```python\n(.*?)```", body, re.S)[1]
    # Preserve the original lesson in section.md; only separate import groups for Ruff.
    snippet = raw_snippet.replace("import torch\nfrom ", "import torch\n\nfrom ")
    snippet_path = ART / (PREFIX + "snippet.py")
    snippet_path.write_text(snippet, encoding="utf-8")
    exercise_path = ART / (PREFIX + "exercise.py")
    exercise_path.write_text(
        snippet.replace("[0.4]", "[1.4]").replace("[[0.8, 0.2]]", "[[0.5, 0.5]]"), encoding="utf-8"
    )
    executions = {
        "lesson": run([str(ROOT / ".venv/bin/python"), str(snippet_path)], "snippet"),
        "exercise": run([str(ROOT / ".venv/bin/python"), str(exercise_path)], "exercise"),
    }
    assert "0.36 -1.2" in executions["lesson"]["stdout"]
    assert "0.1927" in executions["lesson"]["stdout"]
    assert "0.16 0.8" in executions["exercise"]["stdout"]
    assert "KL 0.0" in executions["exercise"]["stdout"]
    numeric = []
    for dtype in (torch.float32, torch.float64):
        for value_number in (0.4, 1.4):
            value = torch.tensor([value_number], requires_grad=True, dtype=dtype)
            target = torch.tensor([1.0], dtype=dtype)
            loss = (value - target).square().mean()
            loss.backward()
            expected_loss = (value_number - 1) ** 2
            expected_grad = 2 * (value_number - 1)
            assert abs(loss.item() - expected_loss) < 1e-6
            assert abs(value.grad.item() - expected_grad) < 1e-6
            numeric.append({"dtype": str(dtype), "input_shape": [1], "mean_denominator": 1,
                            "value": value_number, "target": 1, "loss": loss.item(),
                            "gradient": value.grad.item(), "expected_loss": expected_loss,
                            "expected_gradient": expected_grad})
        p = torch.tensor([[0.8, 0.2]], dtype=dtype)
        q = torch.tensor([[0.5, 0.5]], dtype=dtype)
        policy = p.log().requires_grad_()
        reference = q.log().requires_grad_()
        kl = exact_kl(policy, reference)
        kl.sum().backward()
        expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
        assert abs(kl.item() - expected) < 1e-6
        assert reference.grad is None
        assert policy.grad is not None
        assert torch.allclose(policy.softmax(-1), p, atol=1e-7, rtol=1e-7)
        assert exact_kl(q.log(), q.log()).item() == 0
        numeric.append({"dtype": str(dtype), "shape": [1, 2], "candidate_sum_denominator": 2,
                        "p": p.tolist(), "q": q.tolist(), "kl": kl.tolist(), "expected": expected,
                        "identity_kl": 0, "reference_gradient": None,
                        "policy_gradient": policy.grad.tolist(), "mask": "none; all two actions enumerated"})

    official = {}
    urls = {
        "openai_train_policy.txt": "https://raw.githubusercontent.com/openai/lm-human-preferences/"
        "cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py",
        "torch_functional_remote.txt": "https://raw.githubusercontent.com/pytorch/pytorch/"
        + torch.version.git_version + "/torch/nn/functional.py",
    }
    for name, url in urls.items():
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
                status = response.status
            path = ART / (PREFIX + name)
            path.write_bytes(data)
            official[name] = {"url": url, "status": status, "path": str(path.relative_to(ROOT)),
                              "sha256": digest(path), "bytes": len(data)}
        except Exception as error:
            official[name] = {"url": url, "error": str(error)}
    installed_path = Path(inspect.getfile(F))
    installed_lines = installed_path.read_text().splitlines()
    source_ranges = ((2176, 2220), (2293, 2325), (4271, 4375))
    snapshot = "\n".join(
        f"Original installed source {installed_path}; SHA256 {digest(installed_path)}; torch {torch.__version__}\n"
        + "\n".join(f"{i + 1}: {installed_lines[i]}" for i in range(start - 1, end))
        for start, end in source_ranges
    )
    (ART / (PREFIX + "torch_installed_source.txt")).write_text(snapshot + "\n", encoding="utf-8")
    official["installed_torch"] = {"path": str(installed_path), "sha256": digest(installed_path),
                                  "version": str(torch.__version__), "git_version": torch.version.git_version,
                                  "ranges": source_ranges}
    save("source_fetch.json", official)

    svg = ROOT / "course/figures/ppo_roles.svg"
    xml = ET.parse(svg).getroot()
    labels = [{"text": "".join(node.itertext()), "x": node.get("x"), "y": node.get("y")}
              for node in xml.findall(".//{http://www.w3.org/2000/svg}text")]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path="/usr/bin/chromium")
        page = browser.new_page(viewport={"width": 820, "height": 520}, device_scale_factor=1)
        page.set_content("<style>html,body{margin:0}</style>" + svg.read_text())
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(ART / (PREFIX + "ppo_roles.png")))
        render = {"browser": browser.version, "viewport": [820, 520], "device_scale_factor": 1,
                  "font_check": page.evaluate('document.fonts.check("18px \\"Noto Sans CJK TC\\"")'),
                  "svg_sha256": digest(svg), "xml_viewBox": xml.get("viewBox"), "labels": labels}
        browser.close()
    save("figure_render.json", render)

    run_path = ART / (PREFIX + "cpu_run")
    executions["formal_cpu"] = run(
        [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.posttraining",
         "--output", str(run_path)], "formal_cpu"
    )
    formal_path = ROOT / "docs/course-experiments/results/posttraining.json"
    formal = json.loads(formal_path.read_text())
    (ART / (PREFIX + "published_posttraining.json")).write_bytes(formal_path.read_bytes())
    fresh = json.loads((run_path / "result.json").read_text())
    original_records_path = ROOT / "outputs/posttraining-development/fixed/records.json"
    original_records = json.loads(original_records_path.read_text())
    (ART / (PREFIX + "original_records.json")).write_bytes(original_records_path.read_bytes())
    expected_records = experiment.split_records(experiment.build_records())
    assert original_records == expected_records
    assert formal["results"]["config"] == fresh["results"]["config"] == experiment.CONFIG
    for path, expected_hash in formal["code_sha256"].items():
        assert digest(ROOT / path) == expected_hash
    row_check = {}
    for split, records in expected_records.items():
        published_eval = formal["results"]["evaluations"][split]
        fresh_eval = fresh["results"]["evaluations"][split]
        assert published_eval == fresh_eval
        assert len(published_eval["rows"]) == len(records)
        for raw, saved in zip(records, published_eval["rows"], strict=True):
            assert all(saved[key] == value for key, value in raw.items())
            for policy in saved["policies"].values():
                probabilities = policy["probabilities"]
                assert len(probabilities) == 4 and abs(sum(probabilities) - 1) < 1e-6
                choice = max(range(4), key=probabilities.__getitem__)
                assert choice == policy["chosen_action"]
                assert saved["candidates"][choice] == policy["chosen_response"]
                assert policy["full_request_success"] == (choice == saved["expected_action"])
        row_check[split] = {"families": len({row["family"] for row in records}),
                            "contexts": len(records), "preference_pairs": len(experiment._pairs(records)),
                            "evaluations_identical_to_fresh_run": True,
                            "all_raw_rows_and_predictions_checked": True,
                            "policy_success": published_eval["policies"]}
    parameter_counts = {"policy": sum(p.numel() for p in FiniteResponsePolicy().parameters()),
                        "reward": sum(p.numel() for p in FiniteRewardModel().parameters()),
                        "critic": sum(p.numel() for p in FiniteValueModel().parameters())}
    summary = {
        "environment": environment,
        "section_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "prerequisite_sha256": {key: hashlib.sha256(bodies[key].encode()).hexdigest()
                                for key in ("13.10", "13.11", "13.12")},
        "executions": executions, "numeric": numeric,
        "formal_source_sha256": digest(formal_path), "original_records_sha256": digest(original_records_path),
        "all_configs": experiment.CONFIG, "row_check": row_check,
        "parameter_counts": parameter_counts,
        "formal_timings": {"elapsed_seconds": formal["elapsed_seconds"],
                           **{stage: formal["results"][stage]["seconds"] for stage in ("sft", "reward", "ppo", "dpo")}},
        "fresh_timings": {"elapsed_seconds": fresh["elapsed_seconds"],
                          **{stage: fresh["results"][stage]["seconds"] for stage in ("sft", "reward", "ppo", "dpo")}},
        "denominators": {"seed": 42, "cpu_threads": 2, "train_contexts": 132,
                         "validation_contexts": 15, "test_contexts": 18, "effective_tokens": 0,
                         "sft_steps": 60, "sft_batch_size": 32, "sft_draws": 1920,
                         "reward_steps": 300, "reward_batch_size": 64, "reward_pair_draws": 19200,
                         "ppo_rollouts": 120, "ppo_rollout_size": 64, "sampled_actions": 7680,
                         "policy_and_value_updates_each": 360, "reused_action_draws": 23040,
                         "dpo_steps": 360, "dpo_pair_draws": 23040,
                         "timing_repetitions": 1, "separate_warmup_runs": 0},
        "time_scope": "One CPU run. Stage timers cover training loops; experiment timer starts after records/features,"
        " covers model setup/training/checkpoints/evaluation; outer elapsed includes run_posttraining and code hashes."
        " No environment installation or GPU timing.",
    }
    save("audit.json", summary)
    print(json.dumps({"section_sha256": summary["section_sha256"], "numeric_entries": len(numeric),
                      "all_165_rows_checked": True, "official_fetch": official,
                      "formal_seconds": summary["formal_timings"], "fresh_seconds": summary["fresh_timings"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
