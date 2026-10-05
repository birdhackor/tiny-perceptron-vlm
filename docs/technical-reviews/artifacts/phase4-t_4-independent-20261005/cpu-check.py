"""T.4 independent bounded contract checks; not a model capability experiment."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from scripts.evaluate import answer_sample, evaluate, parse_limit
from scripts.prepare_data import generate_records
from scripts.train import parser, prepare_examples
from scripts.course_experiments.run import experiment_spec, list_assets
from tiny_perceptron.data import ByteTokenizer, IGNORE, load_jsonl, render_chat, shifted
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum
from tiny_perceptron.tokenization import load_tokenizer
from tiny_perceptron.training import load_checkpoint, seed_everything

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
results = {}
commands = []

def run(argv, cwd, name, expected_code=0):
    p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=45)
    (OUT / f"{name}.stdout.txt").write_text(p.stdout)
    (OUT / f"{name}.stderr.txt").write_text(p.stderr)
    commands.append({"command": shlex.join(argv), "cwd": str(cwd), "stdout": f"{name}.stdout.txt", "stderr": f"{name}.stderr.txt", "exit_code": p.returncode, "expected_code": expected_code})
    assert p.returncode == expected_code, (name, p.returncode, p.stderr)

def read_pointers(report_path, pointers):
    raw = report_path.read_bytes()
    report = json.loads(raw)
    values = {}
    for pointer in pointers:
        value = report
        for part in pointer.lstrip("/").split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
        values[pointer] = value
    return {"sha256": hashlib.sha256(raw).hexdigest(), "inspected_pointers": values}

with tempfile.TemporaryDirectory(prefix="phase4-t_4-cpu-") as temp:
    work = Path(temp)
    for name in ["scripts", "tiny_perceptron", ".venv", "pyproject.toml"]:
        (work / name).symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())
    py = ".venv/bin/python"
    for kind in ["toy-text", "attributes-sft"]:
        run([py, "scripts/prepare_data.py", "--kind", kind, "--seed", "42"], work, f"prepare-{kind}")
        directory = work / "data/generated" / kind
        (OUT / "generated" / kind).mkdir(parents=True)
        parts = {split: load_jsonl(directory / f"{split}.jsonl") for split in ["train", "validation", "test"]}
        families = {split: {r["family"] for r in rows} for split, rows in parts.items()}
        assert not families["train"] & families["validation"] and not families["train"] & families["test"] and not families["validation"] & families["test"]
        assert sum(len(f) for f in families.values()) == 12
        counts = {split: len(rows) for split, rows in parts.items()}
        assert counts == ({"train": 9, "validation": 1, "test": 2} if kind == "toy-text" else {"train": 45, "validation": 5, "test": 10})
        if kind == "attributes-sft":
            assert all(sum(r["family"] == f for r in rows) == 5 for rows in parts.values() for f in {r["family"] for r in rows})
        original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
        for p in directory.iterdir():
            shutil.copyfile(p, OUT / "generated" / kind / p.name)
        run([py, "scripts/prepare_data.py", "--kind", kind, "--seed", "42"], work, f"prepare-repeat-{kind}")
        assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
        results[kind] = {"records": counts, "families": {k: len(v) for k, v in families.items()}, "no_family_crosses_splits": True, "same_seed_byte_identical": True, "file_sha256": original}

    tok = ByteTokenizer()
    messages = [{"role": "user", "content": "color=red;shape=circle;pitch=low;shape?"}, {"role": "assistant", "content": "circle"}]
    x, y = render_chat(messages, tok)
    effective = y[y != IGNORE].tolist()
    assert effective == tok.encode("circle") + [tok.eos_id] and len(effective) == 7
    varied = [{"role": "system", "content": "literal <assistant>"}, {"role": "user", "content": "changed question"}, messages[-1]]
    _, varied_y = render_chat(varied, tok)
    assert varied_y[varied_y != IGNORE].tolist() == effective
    results["answer_mask"] = {"effective_ids": effective, "effective_tokens": len(effective), "input_tokens": len(x), "prompt_and_role_targets_ignored": True, "varied_prompt_same_targets": True}

    original_checkpoint = Path("/tmp/phase4-t_4-original-20261005/workspace/checkpoints/start.pt")
    model, payload = load_checkpoint(original_checkpoint, "cpu")
    assert payload["step"] == 0 and payload["optimizer"] is None
    seed_everything(42)
    fresh = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
    assert all(torch.equal(v, fresh.state_dict()[k]) for k, v in model.state_dict().items())
    assert model.embedding.weight.shape == (264, 32) and len(model.blocks) == 1
    assert model.blocks[0].attention.heads == 1 and model.position.num_embeddings == 128
    try:
        model(torch.ones((1, 129), dtype=torch.long))
        raise AssertionError("overlong input accepted")
    except ValueError as error:
        results["context_rejection"] = str(error)
    (work / "checkpoints").mkdir()
    shutil.copyfile(original_checkpoint, work / "checkpoints/start.pt")
    results["original_checkpoint"] = {"bytes": original_checkpoint.stat().st_size, "sha256": hashlib.sha256(original_checkpoint.read_bytes()).hexdigest(), "step": payload["step"], "optimizer": None, "seed_matches_all_parameters": True, "width": 32, "layers": 1, "heads": 1, "max_length": 128}

    for task, kind, tokens in [("text", "toy-text", 32), ("sft", "attributes-sft", 24)]:
        output_name = f"outputs/{task}-before.json"
        run([py, "scripts/evaluate.py", "checkpoints/start.pt", "--data", f"data/generated/{kind}/validation.jsonl", "--mode", task, "--tokens", str(tokens), "--device", "cpu", "--limit", "all", "--output", output_name], work, f"eval-{task}-before")
        pointers = ["/records_read", "/records_selected", "/effective_tokens", "/exact_match", "/completed_exact_match", "/eos_rate", "/metric_denominators", "/skipped", "/samples/0/row", "/samples/0/target", "/samples/0/generated_ids", "/samples/0/generation_status"]
        results[f"evaluation-{task}"] = read_pointers(work / output_name, pointers)
        shutil.copyfile(work / output_name, OUT / f"{task}-before-raw.json")
        run([py, "scripts/train.py", "--task", task, "--data", f"data/generated/{kind}/train.jsonl", "--checkpoint", "checkpoints/start.pt", "--seed", "42", "--device", "cpu", "--train", "--steps", "1", "--output", f"checkpoints/{task}-one-step.pt"], work, f"train-{task}-bounded-one-step")
        updated, state = load_checkpoint(work / f"checkpoints/{task}-one-step.pt")
        assert state["step"] == 1 and state["optimizer"] is not None
        assert any(not torch.equal(v, model.state_dict()[k]) for k, v in updated.state_dict().items())
        results[f"bounded-{task}"] = {"steps": 1, "optimizer_saved": True, "some_parameters_changed": True, "purpose": "CLI contract, not capability or article result"}

    run([py, "scripts/train.py", "--task", "sft", "--data", "data/generated/attributes-sft/train.jsonl", "--checkpoint", "checkpoints/text-one-step.pt", "--seed", "42", "--device", "cpu", "--train", "--steps", "1", "--output", "checkpoints/new-stage.pt"], work, "new-stage-bounded-one-step")
    _, staged = load_checkpoint(work / "checkpoints/new-stage.pt")
    assert staged["step"] == 1 and staged["metadata"]["task"] == "sft"
    run([py, "scripts/train.py", "--task", "sft", "--data", "data/generated/attributes-sft/train.jsonl", "--checkpoint", "checkpoints/start.pt", "--resume", "--seed", "42", "--device", "cpu", "--train", "--steps", "1", "--output", "checkpoints/resume-not-created.pt"], work, "resume-start-rejected", expected_code=1)
    assert "optimizer" in (OUT / "resume-start-rejected.stderr.txt").read_text()
    assert not (work / "checkpoints/resume-not-created.pt").exists()
    results["stage_vs_resume"] = {"new_stage_step": staged["step"], "new_stage_task": staged["metadata"]["task"], "start_resume_rejected_for_missing_optimizer": True}

    samples = {}
    for label, ids in {"content-only": tok.encode("circle"), "content-eos": tok.encode("circle") + [tok.eos_id], "leading-space": tok.encode(" circle") + [tok.eos_id], "invalid-role": [tok.user_id] + tok.encode("circle") + [tok.eos_id]}.items():
        report = answer_sample(tok, ids, "circle")
        samples[label] = {k: report[k] for k in ["exact_match", "completed_exact_match", "eos", "invalid_special_tokens", "generation_status"]}
    assert samples["content-only"]["exact_match"] and not samples["content-only"]["completed_exact_match"]
    assert samples["content-eos"]["completed_exact_match"]
    assert not samples["leading-space"]["exact_match"] and not samples["invalid-role"]["exact_match"]
    results["metric_boundaries"] = samples
    records = [{"messages": messages}, {"messages": [{"role": "user", "content": "a" * 128}, messages[-1]]}]
    report = evaluate(model, records, "sft", max_new_tokens=2)
    assert report["records_selected"] == 2 and report["generation_evaluated_records"] == 1 and report["effective_tokens"] == 7
    assert report["skipped"][0]["row"] == 1 and report["metric_denominators"]["exact_match"] == 1
    results["skipping"] = {k: report[k] for k in ["records_selected", "generation_evaluated_records", "effective_tokens", "metric_denominators", "skipped"]}
    text_report = evaluate(model, [{"text": "a" * 150}], "text", max_new_tokens=2)
    assert text_report["effective_tokens"] == 151 and text_report["exact_match"] is None and text_report["completed_exact_match"] is None
    ids = [tok.bos_id] + tok.encode("a" * 150) + [tok.eos_id]
    totals = []
    with torch.no_grad():
        for start in range(0, len(ids) - 1, 128):
            xx, yy = shifted(ids[start:start + 129]); total, count = loss_sum(model(xx[None])["logits"], yy[None]);totals.append((float(total),int(count)))
    independent_mean = sum(s for s,c in totals)/sum(c for s,c in totals)
    assert abs(text_report["mean_token_nll"] - independent_mean) < 1e-7
    assert len(tok.encode("圓")) == 3 and parse_limit("all") is None
    results["text_units"] = {"150_ascii_bytes_targets_including_eos": 151, "independent_count": sum(c for s,c in totals), "mean_nll_difference": abs(text_report["mean_token_nll"] - independent_mean), "text_matching_null": True, "chinese_character_bytes": 3, "limit_all": None}

    definition = work / "byte-tokenizer.json"
    definition.write_text(json.dumps(tok.state()))
    assert load_tokenizer(definition, 264).vocab_size == 264
    rejected = {}
    for label, args in {"nonbyte-no-file": (None, 512, None), "wrong-hash": (definition, 264, {"tokenizer": {"type": "byte", "sha256": "0"*64}}), "wrong-vocab": (definition, 512, None)}.items():
        try:
            load_tokenizer(*args)
            raise AssertionError(label)
        except ValueError as error:
            rejected[label] = str(error)
    results["tokenizer_binding"] = rejected
    recipes = {}
    for identifier in ["text_foundation", "real_text", "tokenizer", "sft", "sft_ablation"]:
        spec = experiment_spec(identifier)
        recipes[identifier] = {k: spec[k] for k in ["module", "function", "assets"]}
        recipes[identifier]["asset_archive_paths"] = list_assets(identifier)
        run([py, "-m", "scripts.course_experiments.run", "--list-assets", identifier], work, f"list-assets-{identifier}")
    run([py, "scripts/train.py", "--help"], work, "train-help")
    run([py, "scripts/evaluate.py", "--help"], work, "evaluate-help")
    run([py, "scripts/fetch_training_assets.py", "--list"], work, "fetch-assets-list")
    results["fixed_recipe_contracts"] = recipes
    results["cleanup"] = {"temporary_weights_only": True, "temporary_directory_removed_by_context": str(work), "permanent_weights_saved": False}

(OUT / "cpu-result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
(OUT / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
environment = {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "bounded_training": "3 independent one-step synthetic CLI contract checks; no full course training", "offline": "CUDA_VISIBLE_DEVICES empty; HF_HUB_OFFLINE=1; HF_DATASETS_OFFLINE=1; TRANSFORMERS_OFFLINE=1"}
(OUT / "cpu-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print(json.dumps({"all_assertions_passed": True, "contract_commands": len(commands), "weights_retained": False, "verified_groups": list(results)}, ensure_ascii=False))
