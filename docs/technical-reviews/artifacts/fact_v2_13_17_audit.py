"""Independent CPU arithmetic, full-row evidence and SVG audit for lesson 13.17."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments.common import records_sha256
from scripts.course_experiments.posttraining import build_records, split_records
from tiny_perceptron.alignment import dpo_loss

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_17_"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_rows(result, records):
    checked = {}
    for split, evaluation in result["evaluations"].items():
        rows = evaluation["rows"]
        assert len(rows) == len(records[split])
        pair_wins = pair_total = 0
        counts = {name: {mode: [0, 0] for mode in ("number", "explain", "missing")} for name in ("sft", "ppo", "dpo")}
        for row, original in zip(rows, records[split], strict=True):
            assert all(row[key] == value for key, value in original.items())
            expected = {"number": 0, "explain": 1, "missing": 3}[row["mode"]]
            assert row["expected_action"] == expected
            for winner, loser in row["preference_pairs"]:
                pair_total += 1
                pair_wins += row["reward_model_raw_scores"][winner] > row["reward_model_raw_scores"][loser]
            for name, policy in row["policies"].items():
                probabilities = policy["probabilities"]
                assert len(probabilities) == 4 and all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                assert math.isclose(sum(probabilities), 1, abs_tol=2e-7)
                chosen = max(range(4), key=lambda index: probabilities[index])
                assert policy["chosen_action"] == chosen
                assert policy["chosen_response"] == row["candidates"][chosen]
                passed = chosen == expected
                assert policy["full_request_success"] == passed
                counts[name][row["mode"]][0] += passed
                counts[name][row["mode"]][1] += 1
        assert evaluation["reward_model_pairwise_accuracy"]["numerator"] == pair_wins
        assert evaluation["reward_model_pairwise_accuracy"]["denominator"] == pair_total
        for name, modes in counts.items():
            metrics = evaluation["policies"][name]
            assert metrics["greedy_full_request_success"]["numerator"] == sum(v[0] for v in modes.values())
            assert metrics["greedy_full_request_success"]["denominator"] == len(rows)
            for mode, (wins, total) in modes.items():
                assert metrics["by_mode"][mode]["numerator"] == wins
                assert metrics["by_mode"][mode]["denominator"] == total
        checked[split] = {"contexts": len(rows), "rm_pairwise": [pair_wins, pair_total], "policy_by_mode": counts}
    return checked


def main():
    section = dict(sections(ROOT / "course/chapters/13.md"))["13.17"]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(re.search(r"```python\n(.*?)\n```", section, re.S)[1], "13.17-original", "exec"), {})
    original_output = buffer.getvalue()
    assert "0.6931" in original_output and "0.3567" in original_output
    reference = torch.tensor([0.5, 0.5], requires_grad=True).log()
    values = []
    for probabilities in ([0.5, 0.5], [0.7, 0.3], [0.3, 0.7]):
        policy = torch.tensor(probabilities, requires_grad=True)
        loss = dpo_loss(policy[:1].log(), policy[1:].log(), reference[:1], reference[1:], beta=1.0)
        expected = -math.log(probabilities[0])
        assert abs(loss.item() - expected) < 1e-6
        values.append({"probabilities": probabilities, "expected_minus_log_chosen": expected, "observed": loss.item()})
    assert round(values[2]["observed"], 4) == 1.204
    pc = torch.tensor([-4.0], requires_grad=True)
    pr = torch.tensor([-3.0], requires_grad=True)
    rc = torch.tensor([-4.0], requires_grad=True)
    rr = torch.tensor([-3.0], requires_grad=True)
    loss = dpo_loss(pc, pr, rc, rr, beta=0.1)
    loss.backward()
    assert abs(pc.grad.item() + 0.05) < 1e-7 and abs(pr.grad.item() - 0.05) < 1e-7
    assert rc.grad is None and rr.grad is None
    published_path = ROOT / "docs/course-experiments/results/posttraining.json"
    fresh_path = OUT / (PREFIX + "cpu_run/result.json")
    published = json.loads(published_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    raw_path = OUT / (PREFIX + "published_raw_experiment.json")
    assert json.loads(raw_path.read_text()) == published["results"]
    records = split_records(build_records(), 42)
    raw_records_path = OUT / (PREFIX + "published_raw_records.json")
    assert json.loads(raw_records_path.read_text()) == records
    families = {key: {r["family"] for r in rows} for key, rows in records.items()}
    assert not families["train"] & families["test"]
    assert not families["train"] & families["validation"]
    assert not families["test"] & families["validation"]
    audits = {}
    for name, report in (("published", published), ("fresh_cpu", fresh)):
        result = report["results"]
        assert report["device"] == "cpu" and report["seed"] == 42
        assert result["effective_tokens"] == 0 and result["schedule_completed"]
        assert result["parameters"]["policy"] == 148
        assert result["config"]["dpo_beta"] == 0.1
        assert result["ppo"]["initial_state_sha256"] == result["dpo"]["initial_state_sha256"]
        assert result["ppo"]["initial_state_sha256"] == result["reference_state_sha256_before"]
        assert result["reference_state_sha256_before"] == result["reference_state_sha256_after"]
        assert result["ppo"]["policy_updates"] == result["dpo"]["policy_updates"] == 120 * 3 == 360
        assert result["ppo"]["value_updates"] == 360
        assert result["ppo"]["sampled_actions"] == 120 * 64 == 7680
        assert result["ppo"]["reused_action_draws"] == result["dpo"]["processed_pair_draws"] == 360 * 64 == 23040
        assert result["reward"]["processed_pair_draws"] == 300 * 64 == 19200
        assert result["sft"]["demonstrations"] == 44
        for split, rows in records.items():
            assert result["splits"][split]["sha256"] == records_sha256(rows)
        for path, expected_sha in report["code_sha256"].items():
            assert sha(ROOT / path) == expected_sha
        audits[name] = {"all_rows": audit_rows(result, records), "stage_seconds": {key: result[key]["seconds"] for key in ("sft", "reward", "ppo", "dpo")}}
    # The prior checkpoint's float32 tensors are preserved losslessly in Git-safe JSON.
    # This closure recheck consumes existing evidence and performs no training.
    sft_state_path = OUT / (PREFIX + "fresh_sft_state.json")
    sft = json.loads(sft_state_path.read_text())
    reconstructed = {}
    for name, item in sft["state"].items():
        assert item["dtype"] == "torch.float32"
        reconstructed[name] = torch.tensor(item["values"], dtype=torch.float32).reshape(item["shape"])
    assert sum(t.numel() for t in reconstructed.values()) == sft["parameter_elements"] == 148
    state_digest = hashlib.sha256()
    for name, tensor in sorted(reconstructed.items()):
        state_digest.update(name.encode())
        state_digest.update(tensor.numpy().tobytes())
    assert state_digest.hexdigest() == sft["state_sha256"] == fresh["results"]["reference_state_sha256_before"]
    assert fresh["results"]["config"] == published["results"]["config"]
    # Separate existing token-language-model DPO evidence from the finite-card run.
    token_path = ROOT / "docs/course-experiments/results/dpo.json"
    token = json.loads(token_path.read_text())
    token_audit = {}
    for name, run in token["results"]["runs"].items():
        assert run["beta"] == (0.1 if name == "model" else 1.0)
        pair_rows = {}
        for split, metrics in run["preference"].items():
            samples = metrics["samples"]
            assert len(samples) == metrics["records"]
            for row in samples:
                assert math.isclose(row["policy_chosen_logp"] - row["policy_rejected_logp"], row["policy_margin"], abs_tol=1e-5)
                assert math.isclose(row["policy_margin"] - row["reference_margin"], row["relative_margin"], abs_tol=1e-5)
            pair_rows[split] = len(samples)
        token_audit[name] = {"beta": run["beta"], "steps": run["training"]["steps"], "effective_answer_tokens_both_sides": run["training"]["effective_answer_tokens_both_sides"], "preference_rows_checked": pair_rows}
    capstone_section = dict(sections(ROOT / "course/chapters/19.md"))["19.8"]
    capstone_buffer = io.StringIO()
    with contextlib.redirect_stdout(capstone_buffer):
        exec(compile(re.search(r"```python\n(.*?)\n```", capstone_section, re.S)[1], "19.8-linked-original", "exec"), {})
    capstone_stdout = capstone_buffer.getvalue()
    assert "0.6931" in capstone_stdout and "False" in capstone_stdout
    figure = ROOT / "course/figures/ppo_dpo_routes.svg"
    xml = ET.parse(figure).getroot()
    assert xml.attrib["viewBox"] == "0 0 820 520"
    text_labels = [element.text for element in xml.iter() if element.tag.endswith("text")]
    assert "偏好對 → 直接更新策略" in text_labels
    assert "偏好對 → 先教評分員" in text_labels
    assert "SFT checkpoint，複製兩份" in text_labels
    receipt = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_17_audit.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "section_sha256": hashlib.sha256(section.encode()).hexdigest(),
        "original_snippet_stdout": original_output,
        "hand_derivation": "margin=log(p_chosen/p_rejected)-log(0.5/0.5); beta=1 and p_chosen+p_rejected=1 give sigmoid(margin)=p_chosen, so loss=-ln(p_chosen). 148=(4*16+16)+(16*4+4); 360=120*3; 23040=360*64; 7680=120*64.",
        "numeric_results": values,
        "numeric_dtype": str(reference.dtype),
        "sft_snapshot_state_sha256": sft["state_sha256"],
        "gradient_probe_beta_0_1": {"loss": loss.item(), "chosen_gradient": pc.grad.item(), "rejected_gradient": pr.grad.item(), "reference_gradients": [None, None]},
        "full_row_audits": audits,
        "token_lm_separate_evidence": {"device": token["device"], "timing_scope": token["timing_scope"], "runs": token_audit, "not_retrained": True},
        "linked_capstone_example": {"section_sha256": hashlib.sha256(capstone_section.encode()).hexdigest(), "stdout": capstone_stdout, "scope": "Random token-language-model DPO forward pass only, no checkpoint load and no training; distinct from finite-card PPO experiment."},
        "svg_xml_labels": text_labels,
        "svg_xml_paths": [element.attrib["d"] for element in xml.iter() if element.tag.endswith("path")],
        "inspected_evidence_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [published_path, fresh_path, raw_path, raw_records_path, sft_state_path, token_path, figure]},
        "result": "All assertions passed; original example, reversed exercise, full 165 contextual rows per run, all 495 policy decisions per run, 825 RM pairs per run and separate token-LM preference rows checked.",
    }
    path = OUT / (PREFIX + "audit_result.json")
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(receipt["result"])
    print(json.dumps({"test_counts": audits["published"]["all_rows"]["test"], "fresh_stage_seconds": audits["fresh_cpu"]["stage_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
