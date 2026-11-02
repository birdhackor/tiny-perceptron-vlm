"""Reviewer-owned T.7 CPU mechanism/CLI and preserved-JSON checks; no recipe rerun."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from scripts import train
from scripts.course_experiments.posttraining import build_records, split_records
from scripts.course_experiments.run import experiment_spec, list_assets
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, generate
from tiny_perceptron.training import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args):
    command = [sys.executable, *map(str, args)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(730)
    out = {
        "scope": "Synthetic CPU CLI/mechanism checks, one artificial optimizer update and arithmetic on preserved JSON; no 200-step student recipe, fixed research training, teacher, final or GPU evaluation.",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": torch.version.git_version, "device": "cpu"},
        "cli_help": [run(path, "--help") for path in ["scripts/prepare_data.py", "scripts/train.py", "scripts/infer.py", "scripts/fetch_training_assets.py"]] + [run("-m", "scripts.course_experiments.run", "--help")],
    }
    args = train.parser().parse_args(["--task", "dpo", "--checkpoint", "checkpoints/style.pt", "--data", "data/generated/preference/train.jsonl", "--train", "--steps", "200", "--output", "checkpoints/preferred.pt"])
    out["textbook_training_args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    assert args.task == "dpo" and args.steps == 200 and args.train and not args.resume
    out["fixed_recipe_dispatch"] = {name: {"spec": experiment_spec(name), "assets": list_assets(name)} for name in ["dpo", "posttraining"]}
    assert experiment_spec("dpo")["dependencies"] == ["style"]
    assert experiment_spec("posttraining")["dependencies"] == []
    with tempfile.TemporaryDirectory(prefix="review-T7-") as td:
        temp = Path(td)
        archive = ROOT / "assets/training/ultrafeedback-dpo-v1.tar.gz"
        assert archive.stat().st_size == 267858
        out["local_external_asset_unpack"] = run("scripts/fetch_training_assets.py", "--asset", "ultrafeedback-dpo", "--output", temp / "external-assets")
        rows_file = temp / "external-assets/behavior-initial/ultrafeedback-dpo/train-first-100.jsonl"
        out["external_asset_rows"] = {"records": len(rows_file.read_text().splitlines()), "sha256": digest(rows_file), "archive_sha256": digest(archive), "scope": "Unpacked already-present 267858-byte data archive; no network pull or training."}
        out["prepare"] = run("scripts/prepare_data.py", "--kind", "preference", "--output", temp / "generated")
        splits = {s: [json.loads(line) for line in (temp / "generated/preference" / f"{s}.jsonl").read_text().splitlines()] for s in ["train", "validation", "test"]}
        out["generated_preferences"] = {s: {"records": len(rows), "all_labels_follow_rule": all(r["chosen"] == str(sum(map(int, r["family"].split("+")))) and int(r["rejected"]) == int(r["chosen"]) + 1 for r in rows)} for s, rows in splits.items()}
        out["zero_plus_five"] = [{"split": s, **r} for s, rows in splits.items() for r in rows if r["prompt"] == "0+5=?"]
        assert sum(map(len, splits.values())) == 64
        assert all(x["all_labels_follow_rule"] for x in out["generated_preferences"].values())
        tok = ByteTokenizer()
        tiny = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=64))
        initial = temp / "artificial-initial.pt"
        updated = temp / "artificial-updated.pt"
        save_checkpoint(initial, tiny)
        initial_sha = digest(initial)
        prompt = torch.tensor([[tok.bos_id, tok.user_id] + tok.encode("0+5=?") + [tok.eos_id, tok.assistant_id]])
        tiny.eval()
        with torch.no_grad():
            first_argmax = tiny(prompt)["logits"][:, -1].argmax(-1).item()
        generated = generate(tiny, prompt, max_new_tokens=3, temperature=0, eos_id=tok.eos_id)
        assert generated[0, len(prompt[0])].item() == first_argmax
        out["greedy"] = {"first_direct_argmax": first_argmax, "generated_tail": generated[0, len(prompt[0]):].tolist(), "input_dtype": str(prompt.dtype), "input_shape": list(prompt.shape)}
        i1 = run("scripts/infer.py", initial, "--chat", "--prompt", "0+5=?", "--tokens", "3", "--temperature", "0", "--device", "cpu", "--json")
        i2 = run("scripts/infer.py", initial, "--chat", "--prompt", "0+5=?", "--tokens", "3", "--temperature", "0", "--device", "cpu", "--json")
        assert i1["stdout"] == i2["stdout"]
        out["synthetic_inference"] = {"first": i1, "second_identical": True}
        out["one_synthetic_update"] = run("scripts/train.py", "--task", "dpo", "--checkpoint", initial, "--data", temp / "generated/preference/train.jsonl", "--train", "--steps", "1", "--batch-size", "1", "--device", "cpu", "--output", updated)
        policy, payload = load_checkpoint(updated)
        reference_state = payload["training_state"]["dpo_reference"]["model"]
        same_reference = all(torch.equal(v, reference_state[k]) for k, v in tiny.state_dict().items())
        policy_changed = any(not torch.equal(v, policy.state_dict()[k]) for k, v in tiny.state_dict().items())
        out["synthetic_checkpoint"] = {"step": payload["step"], "initial_file_preserved": digest(initial) == initial_sha, "reference_equals_original_parameters": same_reference, "policy_parameters_changed": policy_changed, "distinct_output_created": updated.is_file()}
        assert same_reference and policy_changed and digest(initial) == initial_sha and payload["step"] == 1
        a = train.parser().parse_args(["--task", "dpo", "--data", str(temp / "generated/preference/train.jsonl")])
        x, y = train.prepare_examples(a, tok)[0][0]
        out["dpo_answer_mask"] = {"input_dtype": str(x.dtype), "label_dtype": str(y.dtype), "x_shape": list(x.shape), "active_label_ids": y[y != -100].tolist(), "prompt_labels_masked": bool((y[:6] == -100).all()), "active_token_count": int((y != -100).sum())}
    chosen = torch.tensor([-4.0], requires_grad=True)
    rejected = torch.tensor([-3.0], requires_grad=True)
    rc = torch.tensor([-4.0], requires_grad=True)
    rr = torch.tensor([-3.0], requires_grad=True)
    loss = dpo_loss(chosen, rejected, rc, rr, beta=0.1)
    loss.backward()
    out["scalar_dpo_gradient"] = {"margin": 0.0, "loss": loss.item(), "chosen_gradient": chosen.grad.item(), "rejected_gradient": rejected.grad.item(), "reference_gradients_none": rc.grad is None and rr.grad is None, "paper_gradient_expected": [-0.05, 0.05]}
    assert abs(chosen.grad.item() + 0.05) < 1e-7 and rc.grad is None and rr.grad is None
    records = build_records()
    finite_splits = split_records(records)
    out["finite_candidate_data"] = {"contexts": len(records), "all_four_prewritten_candidates": all(len(r["candidates"]) == 4 for r in records), "best_action_matches_author_rule": all(r["expected_action"] == {"number": 0, "explain": 1, "missing": 3}[r["mode"]] for r in records), "splits": {k: {"contexts": len(v), "preference_pairs": sum(len(x["preference_pairs"]) for x in v)} for k, v in finite_splits.items()}}
    old = {}
    for name in ["dpo", "posttraining"]:
        path = ROOT / f"docs/course-experiments/results/{name}.json"
        j = json.loads(path.read_text())
        old[name] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "elapsed_seconds": j["elapsed_seconds"], "rounded_one_decimal": round(j["elapsed_seconds"], 1), "device": j["device"], "seed": j["seed"], "recorded_environment": {k: j.get(k) for k in ["torch_version", "python_version", "gpu"]}}
        if name == "dpo":
            old[name]["timing_scope"] = j["timing_scope"]
            old[name]["modal"] = j["modal"]
            old[name]["arithmetic_and_preference_recounts"] = []
            for branch, branch_data in j["results"]["runs"].items():
                for split in ["validation", "test"]:
                    p, a = branch_data["preference"][split], branch_data["arithmetic"][split]
                    verified_counts = {"branch": branch, "split": split, "preference_rows": len(p["samples"]), "preference_higher_recount": sum(r["policy_margin"] > 0 for r in p["samples"]), "arithmetic_rows": len(a["samples"]), "arithmetic_matches_recount": sum(r["generated"] == r["expected"] for r in a["samples"])}
                    assert verified_counts["preference_higher_recount"] == p["chosen_higher_absolute_probability"]
                    assert verified_counts["arithmetic_matches_recount"] == a["matches"]
                    assert len(p["samples"]) == p["records"] and len(a["samples"]) == a["records"]
                    old[name]["arithmetic_and_preference_recounts"].append(verified_counts)
            p = next(r for r in j["results"]["runs"]["model"]["preference"]["test"]["samples"] if r["prompt"] == "4+2=?")
            a = next(r for r in j["results"]["runs"]["model"]["arithmetic"]["test"]["samples"] if r["messages"][0]["content"] == "4+2=?")
            old[name]["ranking_generation_counterexample"] = {"prompt": "4+2=?", "chosen": p["chosen"], "rejected": p["rejected"], "chosen_logp": p["policy_chosen_logp"], "rejected_logp": p["policy_rejected_logp"], "generated": a["generated"], "eos": a["eos"]}
            assert p["policy_chosen_logp"] > p["policy_rejected_logp"] and a["generated"] != a["expected"]
        else:
            old[name]["inner_experiment_seconds"] = j["results"]["experiment_seconds"]
            old[name]["label_source"] = j["results"]["label_source"]
            old[name]["scope"] = j["results"]["scope"]
            old[name]["config"] = j["results"]["config"]
            old[name]["parallel_initializations"] = {k: j["results"][k]["initial_state_sha256"] for k in ["ppo", "dpo"]}
            assert len(set(old[name]["parallel_initializations"].values())) == 1
    out["preserved_results"] = old
    out["all_assertions_passed"] = True
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
