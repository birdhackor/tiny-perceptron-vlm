"""Fresh CPU audit of lesson 13.12; no full experiment training or environment edits."""

import copy
import hashlib
import io
import json
import math
import platform
import re
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments import posttraining as experiment
from tiny_perceptron.posttraining import (
    FiniteResponsePolicy,
    FiniteRewardModel,
    bandit_advantage,
    ppo_clipped_objective,
)

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_12_"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = ART / (PREFIX + name)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def run_snippet(code, name):
    output = io.StringIO()
    namespace = {}
    with redirect_stdout(output):
        exec(compile(code, name, "exec"), namespace)
    save(name, code)
    save(name.replace(".py", "_stdout.txt"), output.getvalue())
    return namespace, output.getvalue()


def close(actual, expected, tolerance=1e-6):
    return torch.allclose(torch.tensor(actual), torch.tensor(expected), atol=tolerance, rtol=0)


def main():
    torch.set_num_threads(2)
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "threads": str(torch.get_num_threads()),
    }
    chapter = dict(sections(ROOT / "course/chapters/13.md"))
    body = chapter["13.12"]
    save("section.md", body)
    save("prerequisites.md", chapter["13.11"] + chapter["13.4"])
    assert not re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", body)
    code, = re.findall(r"```python\n(.*?)```", body, flags=re.S)
    original, stdout = run_snippet(code, "snippet.py")
    assert stdout == "更新前後的機率比 [2.0, 0.5]\n"
    assert original["ratio"].shape == (2,)
    assert original["ratio"].dtype == torch.float32
    assert original["ratio"].device.type == "cpu"
    assert close(original["ratio"].tolist(), [2.0, 0.5])
    exercise_code = code.replace("new_probability = torch.tensor([0.4, 0.1])", "new_probability = torch.tensor([0.3, 0.1])")
    assert exercise_code != code
    exercise, exercise_stdout = run_snippet(exercise_code, "exercise.py")
    assert exercise_stdout == "更新前後的機率比 [1.5, 0.5]\n"
    assert close(exercise["ratio"].tolist(), [1.5, 0.5])
    wrong_ratio = (original["new_log_probability"] - original["new_log_probability"]).exp()
    assert torch.equal(wrong_ratio, torch.ones(2))
    positive_score = (original["ratio"][0] * torch.tensor(0.6)).item()
    assert abs(positive_score - 1.2) < 1e-6

    # Gradient paths are tested with inputs that really require gradients.
    old_probability = torch.tensor([0.2, 0.2], requires_grad=True)
    new_probability = torch.tensor([0.4, 0.1], requires_grad=True)
    old_log_probability = old_probability.log().detach()
    ratio = (new_probability.log() - old_log_probability).exp()
    ratio.sum().backward()
    assert old_probability.grad is None
    assert close(new_probability.grad.tolist(), [5.0, 5.0])
    old_log = torch.tensor([math.log(0.2), math.log(0.2)], requires_grad=True)
    new_log = torch.tensor([math.log(0.4), math.log(0.1)], requires_grad=True)
    advantage = torch.tensor([0.6, -0.4], requires_grad=True)
    terms = ppo_clipped_objective(new_log, old_log, advantage)
    terms["policy_loss"].backward()
    assert old_log.grad is None and advantage.grad is None
    assert close(terms["ratio"].tolist(), [2.0, 0.5])
    aliased = old_log.detach()
    snapshot = old_log.detach().clone()
    with torch.no_grad():
        old_log.add_(1)
    assert torch.equal(aliased, old_log)
    assert not torch.equal(snapshot, old_log)
    assert aliased.data_ptr() == old_log.data_ptr()
    assert snapshot.data_ptr() != old_log.data_ptr()

    # A finite log-softmax avoids probability underflow in this explicit example.
    tiny_old_logits = torch.tensor([[0.0, -1000.0]])
    tiny_new_logits = torch.tensor([[0.0, -999.0]])
    tiny_old_probability = tiny_old_logits.softmax(-1)[0, 1]
    tiny_new_probability = tiny_new_logits.softmax(-1)[0, 1]
    direct = tiny_new_probability / tiny_old_probability
    stable = (tiny_new_logits.log_softmax(-1)[0, 1] - tiny_old_logits.log_softmax(-1)[0, 1]).exp()
    assert tiny_old_probability.item() == tiny_new_probability.item() == 0
    assert torch.isnan(direct)
    assert abs(stable.item() - math.e) < 1e-6
    # Log ratios do not guarantee against overflow of the final exp.
    assert torch.isinf(torch.tensor(1000.0).exp())

    # Three short updates implement the same collection/reuse mechanism, not a quality experiment.
    torch.manual_seed(13012)
    policy = FiniteResponsePolicy()
    reference = copy.deepcopy(policy).requires_grad_(False).eval()
    reference_hash = experiment._state_sha256(reference)
    features = torch.tensor([[0.1, 0.2, 0.0, 0.0], [0.3, 0.4, 1.0, 0.0]])
    actions = torch.tensor([0, 1])
    with torch.no_grad():
        old_selected = policy(features).log_softmax(-1).gather(1, actions[:, None]).squeeze(1).clone()
        fixed_advantage = bandit_advantage(torch.tensor([1.0, 0.0]), torch.tensor([0.4, 0.4]))
    old_hash = hashlib.sha256(old_selected.numpy().tobytes()).hexdigest()
    advantage_hash = hashlib.sha256(fixed_advantage.numpy().tobytes()).hexdigest()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
    update_trace = []
    for epoch in range(3):
        current_selected = policy(features).log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        update_terms = ppo_clipped_objective(current_selected, old_selected, fixed_advantage)
        optimizer.zero_grad(set_to_none=True)
        update_terms["policy_loss"].backward()
        optimizer.step()
        with torch.no_grad():
            after = policy(features).log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        assert hashlib.sha256(old_selected.numpy().tobytes()).hexdigest() == old_hash
        assert hashlib.sha256(fixed_advantage.numpy().tobytes()).hexdigest() == advantage_hash
        assert experiment._state_sha256(reference) == reference_hash
        update_trace.append({
            "epoch": epoch + 1,
            "old_selected": old_selected.tolist(),
            "current_selected_before_update": current_selected.detach().tolist(),
            "current_selected_after_update": after.tolist(),
            "ratio_before_update": update_terms["ratio"].detach().tolist(),
            "fixed_advantage": fixed_advantage.tolist(),
            "old_sha256": old_hash,
            "advantage_sha256": advantage_hash,
            "reference_sha256": reference_hash,
        })
    assert close(update_trace[0]["ratio_before_update"], [1.0, 1.0])
    assert not close(update_trace[1]["ratio_before_update"], [1.0, 1.0])
    with torch.no_grad():
        next_old_selected = policy(features).log_softmax(-1).gather(1, actions[:, None]).squeeze(1).clone()
    assert not torch.equal(next_old_selected, old_selected)
    assert experiment._state_sha256(reference) == reference_hash

    # Original paper extracts are recreated directly from the candidate PDFs.
    original_files = ROOT / "outputs/posttrain-design/ppo-reference"
    pdf_manifest = []
    for name, pages, version in [
        ("ppo", [(1, 5)], "arXiv:1707.06347v2, 28 Aug 2017"),
        ("instructgpt", [(1, 1), (8, 9)], "arXiv:2203.02155v1, 4 Mar 2022"),
        ("dpo", [(1, 1), (3, 5)], "arXiv:2305.18290v3, 29 Jul 2024"),
    ]:
        pdf = original_files / f"{name}-original.pdf"
        chunks = []
        commands = []
        for first, last in pages:
            command = ["pdftotext", "-layout", "-f", str(first), "-l", str(last), str(pdf), "-"]
            commands.append(command)
            extracted = subprocess.run(command, check=True, capture_output=True, text=True)
            chunks.append(f"PDF pages {first}-{last}\n{extracted.stdout}")
        target = save(f"{name}_original.txt", "\n".join(chunks))
        pdf_manifest.append({
            "path": str(pdf.relative_to(ROOT)), "sha256": digest(pdf), "version": version,
            "extract_path": str(target.relative_to(ROOT)), "commands": commands,
        })
    official = original_files / "openai-lm-human-preferences-core.py"
    save("openai_train_policy.txt", official.read_text())
    tensor_path = Path(torch.__file__).parent / "_tensor.py"
    docs_path = Path(torch.__file__).parent / "_torch_docs.py"
    functional_path = Path(torch.__file__).parent / "nn/functional.py"
    torch_snapshots = []
    for path, spans in [(tensor_path, [(798, 817)]), (docs_path, [(4310, 4334), (6289, 6311)]),
                        (functional_path, [(2293, 2324)])]:
        lines = path.read_text().splitlines(keepends=True)
        chunk = "".join(f"{path.name}:{first}-{last}\n" + "".join(lines[first - 1:last]) for first, last in spans)
        target = save(f"torch_{path.name.replace('.py', '')}.txt", chunk)
        torch_snapshots.append({"original_path": str(path), "original_sha256": digest(path),
                                "snapshot": str(target.relative_to(ROOT)), "spans": spans})
    save("original_manifest.json", {"pdfs": pdf_manifest,
        "official_code": {"path": str(official.relative_to(ROOT)), "sha256": digest(official),
                          "commit": "cbfd210bb8b08f6bc5c26878c10984b90f516c66"},
        "torch_sources": torch_snapshots})

    # Inspect the complete saved experiment and independently evaluate existing frozen weights.
    formal_path = ROOT / "docs/course-experiments/results/posttraining.json"
    formal = json.loads(formal_path.read_text())
    saved = formal["results"]
    assert formal["device"] == "cpu" and formal["seed"] == 42
    assert saved["effective_tokens"] == 0
    assert saved["config"] == experiment.CONFIG
    code_hashes = {}
    for relative, expected in formal["code_sha256"].items():
        code_hashes[relative] = digest(ROOT / relative)
        assert code_hashes[relative] == expected
    records = experiment.split_records(experiment.build_records(), 42)
    frozen_directory = ROOT / "outputs/posttraining-development/fixed"
    persisted_records = json.loads((frozen_directory / "records.json").read_text())
    assert persisted_records == records
    policies = {name: FiniteResponsePolicy().eval() for name in ("sft", "ppo", "dpo")}
    reward_model = FiniteRewardModel().eval()
    checkpoint_evidence = []
    for checkpoint in saved["checkpoints"]:
        path = frozen_directory / checkpoint["file"]
        assert digest(path) == checkpoint["sha256"]
        checkpoint_evidence.append({**checkpoint, "path": str(path.relative_to(ROOT))})
    for name, model in {**policies, "reward": reward_model}.items():
        checkpoint = torch.load(frozen_directory / f"{name}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.requires_grad_(False)
    assert experiment._state_sha256(policies["sft"]) == saved["reference_state_sha256_before"]
    assert saved["reference_state_sha256_before"] == saved["reference_state_sha256_after"]
    complete_rows = {}
    split_summary = {}
    for split, rows in records.items():
        assert len(rows) == saved["splits"][split]["contexts"]
        assert experiment.records_sha256(rows) == saved["splits"][split]["sha256"]
        assert len(experiment._pairs(rows)) == saved["splits"][split]["preference_pairs"]
        actual = experiment._evaluate(rows, policies, reward_model, policies["sft"],
                                      saved["reward"]["fixed_normalization_scale_from_train"], "cpu")
        expected = saved["evaluations"][split]
        for old_row, new_row in zip(expected["rows"], actual["rows"], strict=True):
            assert old_row["prompt"] == new_row["prompt"]
            assert old_row["expected_action"] == new_row["expected_action"]
            assert close(old_row["reward_model_raw_scores"], new_row["reward_model_raw_scores"])
            assert close(old_row["reward_model_normalized_scores"], new_row["reward_model_normalized_scores"])
            for name in policies:
                assert close(old_row["policies"][name]["probabilities"], new_row["policies"][name]["probabilities"])
                assert old_row["policies"][name]["chosen_action"] == new_row["policies"][name]["chosen_action"]
                assert old_row["policies"][name]["full_request_success"] == new_row["policies"][name]["full_request_success"]
        complete_rows[split] = actual
        split_summary[split] = {"contexts": len(rows), "preference_pairs": len(experiment._pairs(rows)),
                               "rows_recomputed": len(actual["rows"]), "record_sha256": experiment.records_sha256(rows)}
    save("frozen_all_rows.json", complete_rows)
    trace_audit = []
    first_trace = saved["ppo"]["first_rollout_trace"]
    assert len(first_trace) == 3
    for epoch in first_trace:
        old = torch.tensor(epoch["old_selected_log_probabilities"])
        new = torch.tensor(epoch["log_probabilities_before_update"])
        observed_ratio = (new - old).exp()
        rewards = torch.tensor(epoch["normalized_rm_rewards"])
        old_values = torch.tensor(epoch["old_values"])
        assert len(old) == len(new) == 64
        assert close(observed_ratio.tolist(), epoch["ratios_before_update"])
        assert close((rewards - old_values).tolist(), epoch["fixed_advantages"])
        assert epoch["old_selected_log_probabilities"] == first_trace[0]["old_selected_log_probabilities"]
        assert epoch["fixed_advantages"] == first_trace[0]["fixed_advantages"]
        assert epoch["old_log_probability_sha256"] == hashlib.sha256(old.numpy().tobytes()).hexdigest()
        assert epoch["reference_state_sha256"] == saved["reference_state_sha256_before"]
        trace_audit.append({**epoch, "recomputed_ratio": observed_ratio.tolist(),
                            "max_ratio_abs_error": float((observed_ratio - torch.tensor(epoch["ratios_before_update"])).abs().max())})
    save("formal_first_rollout.json", trace_audit)
    result = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_12_verify.py",
        "environment": environment,
        "section_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "figure_references": [],
        "snippet": {"stdout": stdout, "old_probability": original["old_probability"].tolist(),
                    "new_probability": original["new_probability"].tolist(),
                    "old_log_probability": original["old_log_probability"].tolist(),
                    "new_log_probability": original["new_log_probability"].tolist(),
                    "ratio": original["ratio"].tolist(), "shape": [2], "dtype": "float32",
                    "old_requires_grad": original["old_log_probability"].requires_grad},
        "exercise": {"stdout": exercise_stdout, "ratio": exercise["ratio"].tolist()},
        "fixed_advantage_product": positive_score,
        "wrong_new_denominator": wrong_ratio.tolist(),
        "gradients": {"old_probability_grad": None, "new_probability_grad": new_probability.grad.tolist(),
                      "old_log_grad": None, "advantage_grad": None, "new_log_grad": new_log.grad.tolist()},
        "detach": {"shares_storage": True, "in_place_mutation_visible": True, "clone_keeps_history": True},
        "tiny_probability": {"old_probability": 0.0, "new_probability": 0.0, "direct_ratio": "NaN",
                             "old_log_probability": -1000.0, "new_log_probability": -999.0,
                             "log_ratio": stable.item(), "exp_1000": "infinity"},
        "short_update_scope": "two fixed artificial examples, three SGD updates, seed 13012; no quality or speed result",
        "three_update_trace": update_trace,
        "next_round_old_updated": True,
        "reference_unchanged": True,
        "formal_audit": {"report_path": str(formal_path.relative_to(ROOT)), "report_sha256": digest(formal_path),
                         "source_hashes_match": code_hashes, "config": saved["config"], "splits": split_summary,
                         "effective_tokens": saved["effective_tokens"], "first_rollout_examples": 64,
                         "first_rollout_epochs": 3, "all_first_rollout_values_checked": 192,
                         "sampled_actions": saved["ppo"]["sampled_actions"],
                         "reused_action_draws": saved["ppo"]["reused_action_draws"],
                         "policy_updates": saved["ppo"]["policy_updates"],
                         "saved_ppo_seconds": saved["ppo"]["seconds"],
                         "saved_experiment_seconds": saved["experiment_seconds"],
                         "saved_cli_elapsed_seconds": formal["elapsed_seconds"],
                         "time_scope": "Saved PPO timer surrounds sampling and 3-epoch policy/critic updates, analytic KL, tracing; excludes SFT/RM/DPO, checkpoint saves, final frozen evaluation. Experiment timer starts before policy creation and ends after final evaluations, before experiment.json write. CLI elapsed starts before run_posttraining and ends after its output and code hashing, before final result write. No new timing or speed claim.",
                         "checkpoints": checkpoint_evidence,
                         "limitation": "Frozen CPU re-evaluation and recorded rollout arithmetic only; no full retraining or independent reproduction of saved wall time. Lesson 13.12 contains no empirical quality claim."},
        "result": "All CPU assertions passed; both lesson outputs and exact saved rollout/frozen-evaluation arithmetic agree within 1e-6.",
    }
    save("execution.json", result)
    print(json.dumps({"result": result["result"], "environment": environment,
                      "section_sha256": result["section_sha256"], "rows_recomputed": 165,
                      "rollout_ratios_checked": 192}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
