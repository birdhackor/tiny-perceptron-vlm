"""重算相對差距，核對固定參考與教材引用的實驗紀錄。"""

import copy
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def sources():
    specifications = [
        (
            "dpo",
            "https://arxiv.org/html/2305.18290v3",
            "arXiv:2305.18290v3, 2024-07-29",
            [
                ("DPO outline", "we initialize π ref = π SFT \\pi_{\\text{ref}}=\\pi^{\\text{SFT}} whenever available."),
                ("Appendix B, dpo_loss", "losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))"),
            ],
        ),
        (
            "lm_policy",
            "https://arxiv.org/html/1909.08593v2",
            "arXiv:1909.08593v2, 2020-01-08",
            [("Methods: language-model factorization and policy", "fixing the beginning of the sample to x x and generating subsequent tokens using ρ \\rho")],
        ),
        (
            "torch_autograd",
            "https://docs.pytorch.org/docs/2.14/notes/autograd.html",
            "PyTorch 2.14 documentation",
            [
                ("Setting requires_grad", "To freeze parts of your model, simply apply .requires_grad_(False)"),
                ("Evaluation Mode (nn.Module.eval())", "Evaluation mode is not a mechanism to locally disable gradient computation."),
            ],
        ),
        (
            "python_copy",
            "https://docs.python.org/3.13/library/copy.html",
            "Python 3.13.16 documentation",
            [
                ("copy: assignment", "Assignment statements in Python do not copy objects"),
                ("copy: deep copy", "A deep copy constructs a new compound object"),
            ],
        ),
    ]
    receipts = []
    for identifier, url, version, selections in specifications:
        command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", url]
        fetched = subprocess.run(command, capture_output=True, check=True)
        parser = Text()
        parser.feed(fetched.stdout.decode("utf-8"))
        text = " ".join(" ".join(parser.parts).split())
        excerpts = []
        for locator, phrase in selections:
            start = text.index(phrase)
            excerpts.append({"locator": locator, "excerpt": text[start : start + len(phrase)]})
        receipts.append(
            {
                "id": identifier,
                "url": url,
                "version": version,
                "accessed_on": datetime.now(timezone.utc).date().isoformat(),
                "command_argv": command,
                "exit_code": fetched.returncode,
                "original_response_sha256": digest(fetched.stdout),
                "original_response_bytes": len(fetched.stdout),
                "checked_original": True,
                "excerpts": excerpts,
            }
        )
    return {"environment": {"python": sys.version, "transport": "curl HTTPS"}, "sources": receipts}


def checks():
    import torch

    from scripts.course_experiments.behavior import _pair_examples, _state_digest
    from scripts.course_experiments.common import split_records
    from scripts.course_experiments.text import arithmetic_records
    from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
    from tiny_perceptron.data import ByteTokenizer
    from tiny_perceptron.model import ModelConfig, TinyLM

    torch.manual_seed(42)
    torch.set_num_threads(1)
    policy = TinyLM(ModelConfig(width=8))
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    before = reference.embedding.weight.clone()
    reference_hash = _state_digest(reference.state_dict())
    storage_independent = all(a.data_ptr() != b.data_ptr() for a, b in zip(policy.parameters(), reference.parameters()))
    initial_identical = all(torch.equal(a, b) for a, b in zip(policy.parameters(), reference.parameters()))
    with torch.no_grad():
        policy.embedding.weight.add_(0.1)
    scores = reference(torch.tensor([[1, 2]]))["logits"]
    copied = {
        "all_parameter_storages_independent": storage_independent,
        "initial_parameters_identical": initial_identical,
        "reference_embedding_unchanged": torch.equal(before, reference.embedding.weight),
        "reference_state_unchanged": reference_hash == _state_digest(reference.state_dict()),
        "reference_output_requires_grad": scores.requires_grad,
        "policy_still_trainable": all(p.requires_grad for p in policy.parameters()),
        "reference_all_parameters_frozen": all(not p.requires_grad for p in reference.parameters()),
        "reference_eval_mode": not reference.training,
        "output_shape": list(scores.shape),
        "input_dtype": "torch.int64",
        "parameter_dtype": str(reference.embedding.weight.dtype),
    }
    assert storage_independent and initial_identical
    assert copied["reference_embedding_unchanged"] and copied["reference_state_unchanged"]
    assert not copied["reference_output_requires_grad"]
    assert copied["policy_still_trainable"] and copied["reference_all_parameters_frozen"]

    policy = TinyLM(ModelConfig(width=8))
    reference = policy.eval().requires_grad_(False)
    before = reference.embedding.weight.clone()
    with torch.no_grad():
        policy.embedding.weight.add_(0.1)
    alias = {
        "same_object": reference is policy,
        "policy_all_parameters_frozen": all(not p.requires_grad for p in policy.parameters()),
        "reference_embedding_unchanged": torch.equal(before, reference.embedding.weight),
        "output_requires_grad": reference(torch.tensor([[1, 2]]))["logits"].requires_grad,
    }
    assert alias["same_object"] and alias["policy_all_parameters_frozen"]
    assert not alias["reference_embedding_unchanged"]
    evaluated = TinyLM(ModelConfig(width=8)).eval()
    eval_only = {
        "output_requires_grad": evaluated(torch.tensor([[1, 2]]))["logits"].requires_grad,
        "parameters_require_grad": all(p.requires_grad for p in evaluated.parameters()),
    }
    assert eval_only["output_requires_grad"] and eval_only["parameters_require_grad"]

    opening = {"reference_margin": -4 - (-3), "policy_margin": -3 - (-3)}
    opening["relative_margin"] = opening["policy_margin"] - opening["reference_margin"]
    initial_margin = (-4 - (-3)) - (-4 - (-3))
    assert opening == {"reference_margin": -1, "policy_margin": 0, "relative_margin": 1}
    assert initial_margin == 0
    betas = []
    for beta in (0.1, 1.0):
        observed = float(dpo_loss(*(torch.tensor([v], dtype=torch.float64) for v in (-3, -3, -4, -3)), beta=beta))
        expected = math.log1p(math.exp(-beta * opening["relative_margin"]))
        assert abs(observed - expected) < 1e-14
        betas.append({"beta": beta, "observed_loss": observed, "formula_loss": expected})

    result_paths = {name: ROOT / f"docs/course-experiments/results/{name}.json" for name in ("style", "dpo")}
    logs = {name: json.loads(path.read_bytes()) for name, path in result_paths.items()}
    style, dpo = logs["style"]["results"], logs["dpo"]["results"]
    parts = split_records(arithmetic_records(), seed=42)
    split_counts = {name: len(rows) for name, rows in parts.items()}
    assert split_counts == {"train": 49, "validation": 8, "test": 7}
    assert style["content_training"]["records"] == 49
    assert style["content_training"]["checkpoint"] == "content.pt"
    test = style["content_evaluation"]["test"]
    exact = sum(row["generated"] == row["expected"] for row in test["samples"])
    assert exact == test["matches"] == 0
    assert len(test["samples"]) == test["records"] == 7
    sample = next(row for row in dpo["runs"]["model"]["preference"]["validation"]["samples"] if row["prompt"] == "3+5=?")
    recomputed_policy = sample["policy_chosen_logp"] - sample["policy_rejected_logp"]
    recomputed_relative = recomputed_policy - sample["reference_margin"]
    assert recomputed_policy == sample["policy_margin"]
    assert recomputed_relative == sample["relative_margin"]
    for field, rounded in (("policy_margin", -8.44789), ("reference_margin", -10.89346), ("relative_margin", 2.44557)):
        assert abs(sample[field] - rounded) <= 0.000005
    for split in ("validation", "test"):
        assert all(s["relative_margin"] == 0 for s in dpo["before"][split]["samples"])
    reference_test_nll = -sum(row["policy_chosen_logp"] for row in dpo["before"]["test"]["samples"])
    nll_tolerance = 32 * torch.finfo(torch.float32).eps * abs(test["nll_sum"])
    assert abs(reference_test_nll - test["nll_sum"]) < nll_tolerance
    assert dpo["reference_unchanged"] is True
    assert [dpo["runs"][n]["beta"] for n in ("model", "beta1")] == [0.1, 1.0]

    pair = _pair_examples([sample], 128)[0]
    tok = ByteTokenizer()
    masks = []
    for x, y in pair:
        selected = y[y != -100]
        assert len(selected) == 2 and selected[-1] == tok.eos_id
        synthetic = torch.zeros(1, len(x), 264)
        logp = float(sequence_log_probability(synthetic, y[None]))
        assert abs(logp + 2 * math.log(264)) < 1e-6
        masks.append({"input_tokens": len(x), "answer_tokens_including_eos": len(selected), "answer_token_ids": selected.tolist(), "uniform_logp_sum": logp})
    paths = [
        "tiny_perceptron/model.py", "tiny_perceptron/alignment.py", "tiny_perceptron/data.py",
        "scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
    ]
    return {
        "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "seed": "42", "threads": "1"},
        "copied_reference": copied, "aliased_reference": alias, "eval_only": eval_only,
        "opening_arithmetic": opening, "identical_initial_relative_margin": initial_margin, "beta_loss": betas,
        "arithmetic_split_counts": split_counts,
        "recorded_content_training": style["content_training"],
        "recorded_test": test,
        "recomputed_test_matches": exact,
        "recorded_validation_sample": sample,
        "recomputed_policy_margin": recomputed_policy,
        "recomputed_relative_margin": recomputed_relative,
        "absolute_probability_ratio_chosen_over_rejected": math.exp(recomputed_policy),
        "recorded_reference_sha256": dpo["reference_sha256"],
        "recorded_reference_unchanged": dpo["reference_unchanged"],
        "reference_test_nll_recomputed_from_dpo": reference_test_nll,
        "style_test_nll_sum": test["nll_sum"],
        "nll_comparison": {"absolute_difference": abs(reference_test_nll - test["nll_sum"]), "absolute_tolerance": nll_tolerance, "reason": "float32 的逐例與批次 reduction 只作近似一致性核對；用 32 倍 machine epsilon 乘總和尺度，不據此證明 checkpoint 逐位元組相同。"},
        "sequence_mask_checks": masks,
        "experiment_records": {
            name: {"path": str(result_paths[name].relative_to(ROOT)), "sha256": digest(result_paths[name].read_bytes()),
                   **{k: log[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "evidence_status")},
                   "current_code_digest_matches": {p: log["code_sha256"].get(p) == digest((ROOT / p).read_bytes()) for p in paths}}
            for name, log in logs.items()
        },
        "repository_sha256": {p: digest((ROOT / p).read_bytes()) for p in paths},
        "scope": "只重算既有紀錄並實跑短程式；沒有重跑 CUDA 訓練或載入歷史 checkpoint。實驗的 behavior.py 全檔 hash 與現行檔不同，現行程式不能視為原訓練版本的逐位元組重現。",
    }


if __name__ == "__main__":
    print(json.dumps(sources() if "--sources" in sys.argv else checks(), ensure_ascii=False, indent=2))
