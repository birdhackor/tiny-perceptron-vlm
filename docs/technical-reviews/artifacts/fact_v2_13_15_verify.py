"""Independent CPU/original-source audit for lesson 13.15; no textbook edits."""

import contextlib
import hashlib
import importlib
import inspect
import io
import json
import platform
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments.common import records_sha256
from scripts.course_experiments.posttraining import CONFIG, build_records, split_records
from tiny_perceptron.posttraining import FiniteResponsePolicy, FiniteRewardModel, FiniteValueModel

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_15"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / f"{PREFIX}_{name}"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(ROOT))


def state_hash(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def official_sources():
    sgd_module = importlib.import_module("torch.optim.sgd")
    objects = {
        "torch/distributions/categorical.py": [torch.distributions.Categorical],
        "torch/optim/sgd.py": [torch.optim.SGD, sgd_module._single_tensor_sgd],
        "torch/optim/optimizer.py": [torch.optim.Optimizer.zero_grad],
        "torch/autograd/grad_mode.py": [torch.no_grad],
        "torch/nn/modules/module.py": [torch.nn.Module.requires_grad_],
        "torch/_tensor.py": [torch.Tensor.backward],
        "torch/nn/functional.py": [torch.nn.functional.cross_entropy],
    }

    def fetch(item):
        relative, targets = item
        url = f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/{relative}"
        remote = urllib.request.urlopen(url, timeout=20).read()
        local = Path(inspect.getsourcefile(inspect.unwrap(targets[0])))
        assert hashlib.sha256(remote).hexdigest() == digest(local), relative
        excerpts = []
        for target in targets:
            lines, start = inspect.getsourcelines(target)
            excerpts.append({"symbol": target.__qualname__, "first_line": start, "last_line": start + len(lines) - 1})
        path = OUT / f"{PREFIX}_torch_{Path(relative).stem}.txt"
        chunks = [f"URL: {url}\nVersion: {torch.__version__}; commit: {torch.version.git_version}\n"]
        for target, entry in zip(targets, excerpts, strict=True):
            chunks.append(f"\n{entry['symbol']}; original lines {entry['first_line']}-{entry['last_line']}\n")
            chunks.extend(f"{i:5}: {line}" for i, line in enumerate(inspect.getsourcelines(target)[0], entry["first_line"]))
        path.write_text("".join(chunks), encoding="utf-8")
        return {
            "url": url,
            "remote_sha256": hashlib.sha256(remote).hexdigest(),
            "local_distribution_sha256": digest(local),
            "local_source": str(local),
            "snapshot": str(path.relative_to(ROOT)),
            "symbols": excerpts,
        }

    with ThreadPoolExecutor(max_workers=7) as pool:
        sources = list(pool.map(fetch, objects.items()))
    tensor_contracts = {
        "gather": torch.gather.__doc__,
        "squeeze": torch.squeeze.__doc__,
        "manual_seed": torch.manual_seed.__doc__,
    }
    save("torch_tensor_contracts.json", tensor_contracts)
    return sources


def original_papers():
    base = ROOT / "outputs/posttrain-design/ppo-reference"
    selections = {"ppo": (1, 5, "1707.06347v2"), "dpo": (3, 4, "2305.18290v3"), "instructgpt": (7, 9, "2203.02155v1")}
    manifest = json.loads((base / "sources.json").read_text())
    results = []
    for name, (first, last, version) in selections.items():
        pdf = base / f"{name}-original.pdf"
        recorded = next(row for row in manifest if row["name"] == pdf.name)
        assert digest(pdf) == recorded["sha256"]
        extracted = subprocess.check_output(["pdftotext", "-f", str(first), "-l", str(last), "-layout", str(pdf), "-"], text=True)
        first_page = subprocess.check_output(["pdftotext", "-f", "1", "-l", "1", str(pdf), "-"], text=True)
        assert version in first_page
        snapshot = OUT / f"{PREFIX}_{name}_original_excerpt.txt"
        snapshot.write_text(f"Original PDF: {pdf.relative_to(ROOT)}\nSHA-256: {digest(pdf)}\nVersion: {version}\n" + extracted, encoding="utf-8")
        results.append({"paper": name, "version": version, "original_url": recorded["url"], "pdf_sha256": digest(pdf), "pages": [first, last], "snapshot": str(snapshot.relative_to(ROOT)), "extraction_command": f"pdftotext -f {first} -l {last} -layout {pdf.relative_to(ROOT)} -"})
    return results


def short_program():
    body = dict(sections(ROOT / "course/chapters/13.md"))["13.15"]
    code = re.search(r"```python\n(.*?)```", body, re.S)[1]
    code_path = OUT / f"{PREFIX}_lesson_program.txt"
    code_path.write_text(code, encoding="utf-8")
    # Execute the exact textbook cell once, then add observation-only instrumentation
    # around the step in a separate run to test every registered parameter.
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, str(code_path), "exec"), {})
    records = []
    for third_feature in (0, 1):
        changed = code.replace("[[0.1, 0.2, 0.0, 0.0]]", f"[[0.1, 0.2, {third_feature}.0, 0.0]]")
        changed = changed.replace("loss.backward()", "snapshots = {name: [p.detach().clone() for p in model.parameters()] for name, model in [('policy', policy), ('critic', critic), ('reward', reward_model), ('reference', reference)]}\nloss.backward()")
        namespace = {}
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(changed, str(code_path), "exec"), namespace)
        models = {"policy": namespace["policy"], "critic": namespace["critic"], "reward": namespace["reward_model"], "reference": namespace["reference"]}
        changed_models = {}
        residuals = {}
        for name, model in models.items():
            before = namespace["snapshots"][name]
            changed_models[name] = any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
            if name in ("policy", "critic"):
                residuals[name] = max(float((new.detach() - (old - 0.1 * new.grad)).abs().max()) for old, new in zip(before, model.parameters(), strict=True))
        assert changed_models == {"policy": True, "critic": True, "reward": False, "reference": False}
        assert max(residuals.values()) < 1e-7
        assert namespace["terms"]["ratio"].item() == 1.0
        assert not namespace["old_log_probability"].requires_grad
        assert not namespace["reward"].requires_grad
        assert not namespace["advantage"].requires_grad
        assert namespace["new_log_probability"].requires_grad
        assert list(namespace["context"].shape) == list(namespace["new_logits"].shape) == [1, 4]
        actual_optimizer = {id(p) for group in namespace["optimizer"].param_groups for p in group["params"]}
        assert actual_optimizer == {id(p) for model in (models["policy"], models["critic"]) for p in model.parameters()}
        records.append({"third_feature": third_feature, "stdout": captured.getvalue(), "changed_models": changed_models, "sgd_max_abs_residual": residuals, "context_dtype": str(namespace["context"].dtype), "action_dtype": str(namespace["action"].dtype), "ratio": namespace["terms"]["ratio"].tolist(), "advantage": namespace["advantage"].tolist(), "kl_before_step": float(namespace["exact_kl"](namespace["new_logits"], models["reference"](namespace["context"])).mean().detach()), "sampled_action": namespace["action"].tolist()})
    return {"exact_cell_stdout": stdout.getvalue(), "code_sha256": digest(code_path), "instrumented_runs": records}


def audit_report(report):
    result = report["results"]
    assert result["config"] == CONFIG
    generated = split_records(build_records(), 42)
    sets = [set(row["family"] for row in generated[name]) for name in ("train", "validation", "test")]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    assert sum(map(len, sets)) == 55
    assert [len(rows) for rows in generated.values()] == [132, 15, 18]
    for name, rows in generated.items():
        assert records_sha256(rows) == result["splits"][name]["sha256"]
        assert len({row["family"] for row in rows}) == len(result["splits"][name]["families"])
        assert sum(len(row["preference_pairs"]) for row in rows) == result["splits"][name]["preference_pairs"]
    assert result["parameters"] == {"policy": 148, "reward_model": 241, "value_model": 97}
    assert result["sft"]["demonstrations"] == 44
    assert result["reward"]["unique_training_pairs"] == 660
    assert result["reward"]["processed_pair_draws"] == 300 * 64 == 19200
    assert result["ppo"]["sampled_actions"] == 120 * 64 == 7680
    assert result["ppo"]["policy_updates"] == result["ppo"]["value_updates"] == 120 * 3 == 360
    assert result["ppo"]["reused_action_draws"] == 120 * 64 * 3 == 23040
    assert result["effective_tokens"] == 0 and result["schedule_completed"]
    trace = result["ppo"]["first_rollout_trace"]
    for field in ("train_context_indices", "sampled_actions", "old_selected_log_probabilities", "old_values", "fixed_advantages", "normalized_rm_rewards", "old_log_probability_sha256", "reference_state_sha256"):
        assert trace[0][field] == trace[1][field] == trace[2][field]
    assert trace[0]["ratios_before_update"] == [1.0] * 64
    for reward, value, advantage in zip(trace[0]["normalized_rm_rewards"], trace[0]["old_values"], trace[0]["fixed_advantages"], strict=True):
        assert abs(reward - value - advantage) < 3e-7
    assert result["reference_state_sha256_before"] == result["reference_state_sha256_after"] == result["ppo"]["initial_state_sha256"]
    per_question = []
    for split, evaluation in result["evaluations"].items():
        for policy_name in ("sft", "ppo", "dpo"):
            count = 0
            for row in evaluation["rows"]:
                policy = row["policies"][policy_name]
                expected = {"number": 0, "explain": 1, "missing": 3}[row["mode"]]
                probs = policy["probabilities"]
                assert len(probs) == 4 and all(0 <= p <= 1 for p in probs) and abs(sum(probs) - 1) < 2e-7
                action = max(range(4), key=probs.__getitem__)
                assert action == policy["chosen_action"]
                assert policy["chosen_response"] == row["candidates"][action]
                assert policy["full_request_success"] == (action == expected)
                count += action == expected
                if split == "test":
                    per_question.append({"family": row["family"], "mode": row["mode"], "policy": policy_name, "expected": expected, "observed": action, "success": action == expected, "probabilities": probs})
            summary = evaluation["policies"][policy_name]["greedy_full_request_success"]
            assert summary["numerator"] == count and summary["denominator"] == len(evaluation["rows"])
    assert result["evaluations"]["test"]["policies"]["sft"]["greedy_full_request_success"]["numerator"] == 6
    assert result["evaluations"]["test"]["policies"]["ppo"]["greedy_full_request_success"]["numerator"] == 12
    return {"all_checks": "passed", "test_question_checks": per_question, "outer_elapsed_seconds": report["elapsed_seconds"], "inner_experiment_seconds": result["experiment_seconds"], "ppo_seconds": result["ppo"]["seconds"]}


def checkpoint_audit(directory, report):
    result = report["results"]
    models = {}
    results = []
    for checkpoint in result["checkpoints"]:
        path = directory / checkpoint["file"]
        data = torch.load(path, map_location="cpu", weights_only=True)
        name = checkpoint["file"].removesuffix(".pt")
        model = FiniteRewardModel() if name == "reward" else FiniteValueModel() if name == "value" else FiniteResponsePolicy()
        model.load_state_dict(data["model"])
        model.eval()
        assert digest(path) == checkpoint["sha256"] and state_hash(model) == checkpoint["state_sha256"]
        models[name] = model
        results.append({"file": str(path.relative_to(ROOT)), "sha256": digest(path), "state_sha256": state_hash(model)})
    max_error = 0.0
    with torch.no_grad():
        for evaluation in result["evaluations"].values():
            x = torch.tensor([row["features"] for row in evaluation["rows"]])
            for name in ("sft", "ppo", "dpo"):
                probs = models[name](x).softmax(-1)
                expected = torch.tensor([row["policies"][name]["probabilities"] for row in evaluation["rows"]])
                max_error = max(max_error, float((probs - expected).abs().max()))
            scores = models["reward"](x)
            saved_scores = torch.tensor([row["reward_model_raw_scores"] for row in evaluation["rows"]])
            max_error = max(max_error, float((scores - saved_scores).abs().max()))
    assert max_error < 1e-6
    return {"checkpoints": results, "maximum_probability_or_reward_error": max_error, "all_checks": "passed"}


def main():
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_commit": torch.version.git_version, "device": "cpu", "platform": platform.platform()}
    originals = original_papers()
    official = official_sources()
    short = short_program()
    original = json.loads((ROOT / "docs/course-experiments/results/posttraining.json").read_text())
    raw_directory = ROOT / "outputs/posttraining-development/fixed"
    assert json.loads((raw_directory / "records.json").read_text()) == split_records(build_records(), 42)
    assert json.loads((raw_directory / "experiment.json").read_text()) == original["results"]
    torch.set_num_threads(2)
    existing_checkpoints = checkpoint_audit(ROOT / "outputs/posttraining-development/fixed", original)
    audit_original = audit_report(original)
    run_root = OUT / f"{PREFIX}_run"
    command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--experiment", "posttraining", "--device", "cpu", "--output", str(run_root)]
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    rerun = json.loads((run_root / "posttraining/result.json").read_text())
    audit_rerun = audit_report(rerun)
    rerun_checkpoints = checkpoint_audit(run_root / "posttraining", rerun)
    original_states = {row["file"]: row["state_sha256"] for row in original["results"]["checkpoints"]}
    rerun_states = {row["file"]: row["state_sha256"] for row in rerun["results"]["checkpoints"]}
    assert original_states == rerun_states
    logits = torch.tensor([[0.2, -0.1, 0.4, -0.7]], dtype=torch.float64)
    smoothed_loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]), label_smoothing=0.2)
    explicit_targets = torch.tensor([[0.85, 0.05, 0.05, 0.05]], dtype=torch.float64)
    explicit_loss = -(explicit_targets * logits.log_softmax(-1)).sum()
    assert abs(float(smoothed_loss - explicit_loss)) < 1e-14
    evidence = {"command": f"PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/{PREFIX}_verify.py", "environment": environment, "papers": originals, "official_sources": official, "short_program": short, "published_report_sha256": digest(ROOT / "docs/course-experiments/results/posttraining.json"), "published_report_audit": audit_original, "original_checkpoint_audit": existing_checkpoints, "original_raw_json": {str(p.relative_to(ROOT)): digest(p) for p in (raw_directory / "result.json", raw_directory / "experiment.json", raw_directory / "records.json")}, "label_smoothing_probe": {"target": explicit_targets.tolist(), "functional_loss": float(smoothed_loss), "explicit_loss": float(explicit_loss), "absolute_difference": abs(float(smoothed_loss - explicit_loss))}, "rerun_command": command, "rerun_stdout": completed.stdout, "rerun_stderr": completed.stderr, "rerun_report_audit": audit_rerun, "rerun_checkpoint_audit": rerun_checkpoints, "same_trained_states_as_original": original_states == rerun_states, "result": "all assertions passed"}
    output = save("audit.json", evidence)
    print(json.dumps({"result": "all assertions passed", "evidence": output, "exact_cell_stdout": short["exact_cell_stdout"], "published_seconds": original["elapsed_seconds"], "rerun_seconds": rerun["elapsed_seconds"], "rerun_ppo_seconds": rerun["results"]["ppo"]["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
