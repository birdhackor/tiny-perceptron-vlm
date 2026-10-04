"""Independent bounded CPU checks for lesson 8.9; no parameter training."""

import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments.behavior import _merge_lora, _restore_adapter, _style_record
from scripts.course_experiments.common import evaluate_lm, split_records, text_examples
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.adapters import base_state_sha256, load_lora_adapter
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.training import load_checkpoint

ROOT = Path.cwd()
ART = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_08_09"
torch.set_num_threads(1)
torch.manual_seed(0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    path = ART / f"{PREFIX}_{name}.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path.relative_to(ROOT).as_posix()


environment = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "device": "cpu",
    "dtype": "torch.float32",
    "threads": str(torch.get_num_threads()),
    "platform": platform.platform(),
}
body = dict(sections(ROOT / "course/chapters/08.md"))["8.9"]
source_path = ART / f"{PREFIX}_section.txt"
source_path.write_text(body, encoding="utf-8")
for path, number in [("course/chapters/08.md", "8.8"), ("course/training.md", "T.5")]:
    text = dict(sections(ROOT / path))[number]
    (ART / f"{PREFIX}_prerequisite-{number}.txt").write_text(text, encoding="utf-8")
original = re.search(r"```python\n(.*?)```", body, re.S)[1]
(ART / f"{PREFIX}_original-code.txt").write_text(original, encoding="utf-8")
snippet_checks = {}
for name, code in [("original", original), ("exercise", original.replace('    layer.a.copy_(adapter_a["a"])\n', ""))]:
    stdout = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, f"8.9-{name}", "exec"), namespace)
    layer = namespace["layer"]
    a = namespace["adapter_a"]
    b = namespace["adapter_b"]
    snippet_checks[name] = {
        "stdout": stdout.getvalue(),
        "b_mean_unrounded": layer.b.mean().item(),
        "a_shape": list(layer.a.shape),
        "b_shape": list(layer.b.shape),
        "a_restored": torch.equal(layer.a, a["a"]),
        "b_restored": torch.equal(layer.b, a["b"]),
        "adapters_different_a": not torch.equal(a["a"], b["a"]),
        "adapters_different_b": not torch.equal(a["b"], b["b"]),
        "backup_distinct_storage": a["a"].data_ptr() != layer.a.data_ptr(),
        "backup_requires_grad": a["a"].requires_grad,
        "backup_grad_fn": str(a["a"].grad_fn),
    }
assert snippet_checks["original"]["stdout"] == "切回A的B平均 0.1\nA兩個矩陣都恢復 True\n"
assert snippet_checks["exercise"]["stdout"] == "切回A的B平均 0.1\nA兩個矩陣都恢復 False\n"
with torch.no_grad():
    layer = LoRALinear(torch.nn.Linear(4, 3), rank=2, alpha=2)
    layer.a.fill_(0.2)
    layer.b.fill_(-0.1)
    delta2 = layer.b @ layer.a * (layer.alpha / layer.rank)
    layer.alpha = 4
    delta4 = layer.b @ layer.a * (layer.alpha / layer.rank)
    assert torch.equal(delta4, delta2 * 2)
    alias = layer.a
    backup = layer.a.clone()
    layer.a.fill_(0.3)
    assert torch.equal(alias, layer.a) and not torch.equal(backup, layer.a)
snippet_checks["scaling"] = {"rank": 2, "alpha2_delta": delta2.tolist(), "alpha4_delta": delta4.tolist(), "exactly_double": True}

formal = json.loads((ROOT / "docs/course-experiments/results/lora.json").read_text())
style_report = json.loads((ROOT / "docs/course-experiments/results/style.json").read_text())
fr = formal["results"]
formal_code = {}
for name in ["behavior", "common", "run", "text"]:
    artifact = ART / f"{PREFIX}_formal-{name}.py.txt"
    expected = formal["code_sha256"][f"scripts/course_experiments/{name}.py"]
    formal_code[name] = {"path": artifact.relative_to(ROOT).as_posix(), "sha256": digest(artifact), "matches_report": digest(artifact) == expected}
    assert formal_code[name]["matches_report"]

parts = split_records(arithmetic_records(), seed=42)
families = {split: {row["family"] for row in rows} for split, rows in parts.items()}
assert not families["train"] & families["validation"]
assert not families["train"] & families["test"]
assert not families["validation"] & families["test"]
data_audit = {}
all_rows = {}
for style in ["concise", "vivid"]:
    rows_by_split = {split: [_style_record(row, style, False) for row in rows] for split, rows in parts.items()}
    all_rows[style] = rows_by_split
    entry = {}
    for split, rows in rows_by_split.items():
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path = ART / f"{PREFIX}_{style}-{split}.jsonl"
        path.write_text(text, encoding="utf-8")
        info = fr["runs"][style]["data"][split]
        assert digest(path) == info["sha256"]
        examples = text_examples(rows, mode="sft", max_length=128)
        lengths = [int((y != IGNORE).sum()) for _, y in examples]
        if split != "train":
            assert sum(lengths) == fr["runs"][style]["evaluation"][split]["effective_tokens"]
        entry[split] = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "records": len(rows), "families": len(families[split]), "answer_and_eos_targets": lengths, "total_targets": sum(lengths)}
    sampler = random.Random(42)
    examples = text_examples(rows_by_split["train"], mode="sft", max_length=128)
    count = sum(int((y != IGNORE).sum()) for _ in range(450) for _, y in sampler.choices(examples, k=16))
    assert count == fr["runs"][style]["training"]["effective_tokens"]
    entry["sampled_training_targets_450_steps_batch16"] = count
    data_audit[style] = entry
assert data_audit["vivid"]["sampled_training_targets_450_steps_batch16"] == fr["full_sft"]["training"]["effective_tokens"]

base_path = "checkpoints/course/style/content.pt"
adapter_paths = {style: f"checkpoints/course/lora/adapter-{style}.pt" for style in ["concise", "vivid"]}
base, native = load_checkpoint(base_path, "cpu")
payloads = {style: torch.load(path, map_location="cpu", weights_only=True) for style, path in adapter_paths.items()}
assert base_state_sha256(native["model"]) == fr["base_sha256"]
input_files = [base_path, *adapter_paths.values(), "checkpoints/course/lora/merged-vivid.pt", "checkpoints/course/lora/full-sft.pt"]
input_hashes = {path: digest(path) for path in input_files}
for model_name in ["style", "lora"]:
    manifest = json.loads((ROOT / f"checkpoints/course/{model_name}/download-manifest.json").read_text())
    for record in manifest["files"]:
        path = f"checkpoints/course/{model_name}/{record['output']}"
        if path in input_hashes:
            assert input_hashes[path] == record["sha256"]
switch_model = copy.deepcopy(base).eval()
metadata = load_lora_adapter(switch_model, adapter_paths["concise"], base_state=native["model"])
probe = torch.tensor([[1, 3, 57, 51, 57, 69, 71, 2, 4]], dtype=torch.long)
tok = ByteTokenizer()
assert tok.decode(probe[0].tolist()) == "1+1=?"
with torch.no_grad():
    a = switch_model(probe)["logits"].clone()
    _restore_adapter(switch_model, payloads["vivid"]["adapter"])
    b = switch_model(probe)["logits"].clone()
    merged = _merge_lora(switch_model).eval()
    merge_scores = merged(probe)["logits"]
    saved_merged, _ = load_checkpoint("checkpoints/course/lora/merged-vivid.pt", "cpu")
    saved_merged.eval()
    saved_scores = saved_merged(probe)["logits"]
    _restore_adapter(switch_model, payloads["concise"]["adapter"])
    restored = switch_model(probe)["logits"]
switch_audit = {"prompt_text": tok.decode(probe[0].tolist()), "probe_ids": probe.tolist(), "logits_shape": list(a.shape), "compared_scalar_logits": a.numel(), "a_b_a_max_difference": float((a - restored).abs().max()), "adapter_a_b_logit_difference": float((a - b).abs().max()), "fresh_merge_max_difference": float((b - merge_scores).abs().max()), "saved_merge_max_difference": float((b - saved_scores).abs().max()), "fresh_saved_weight_exact": all(torch.equal(v, saved_merged.state_dict()[k]) for k, v in merged.state_dict().items()), "base_parameters": sum(p.numel() for p in base.parameters()), "adapter_parameters": sum(layer.a.numel() + layer.b.numel() for layer in switch_model.modules() if isinstance(layer, LoRALinear)), "inference_requires_grad_parameters": sum(p.numel() for p in switch_model.parameters() if p.requires_grad), "modules": metadata["modules"], "base_sha256": metadata["base_sha256"]}
assert switch_audit["adapter_parameters"] == 9504
assert switch_audit["a_b_a_max_difference"] == 0
assert switch_audit["adapter_a_b_logit_difference"] > 0
assert switch_audit["fresh_merge_max_difference"] < 1e-4
assert switch_audit["saved_merge_max_difference"] < 1e-4

evaluations = {}
for style in ["concise", "vivid"]:
    model = copy.deepcopy(base).eval()
    load_lora_adapter(model, adapter_paths[style], base_state=native["model"])
    evaluations[style] = {}
    for split in ["validation", "test"]:
        observed = evaluate_lm(model, all_rows[style][split], mode="sft", tokens=96)
        historical = fr["runs"][style]["evaluation"][split]
        comparisons = []
        for cpu_sample, old in zip(observed["samples"], historical["samples"], strict=True):
            comparisons.append({"prompt": old["messages"], "expected": old["expected"], "historical_ids": old["generated_ids"], "cpu_ids": cpu_sample["generated_ids"], "ids_equal": cpu_sample["generated_ids"] == old["generated_ids"], "cpu_generated": cpu_sample["generated"], "historical_generated": old["generated"], "eos": cpu_sample["eos"]})
        evaluations[style][split] = {"records": observed["records"], "effective_tokens": observed["effective_tokens"], "matches": observed["matches"], "eos_rate": observed["eos_rate"], "samples": comparisons}

rejections = {}
with tempfile.TemporaryDirectory(prefix=PREFIX) as scratch:
    broken = copy.deepcopy(payloads["concise"])
    first = next(iter(broken["adapter"]))
    for case in ["wrong_base", "wrong_scaling", "missing_b", "unknown_layer", "bad_shape"]:
        model = copy.deepcopy(base)
        modified = copy.deepcopy(broken)
        if case == "wrong_base":
            with torch.no_grad():
                model.embedding.weight[0, 0].add_(0.01)
        elif case == "wrong_scaling":
            modified["scaling"] = "other"
        elif case == "missing_b":
            del modified["adapter"][first]["b"]
        elif case == "unknown_layer":
            modified["adapter"]["missing.module"] = modified["adapter"].pop(first)
        elif case == "bad_shape":
            modified["adapter"][first]["a"] = modified["adapter"][first]["a"][:, :-1]
        path = Path(scratch) / (case + ".pt")
        torch.save(modified, path)
        try:
            load_lora_adapter(model, path)
        except ValueError as error:
            rejections[case] = {"error": str(error), "no_partial_insertion": not any(isinstance(layer, LoRALinear) for layer in model.modules())}
        else:
            raise AssertionError(f"Malformed adapter accepted: {case}")
    try:
        load_lora_adapter(switch_model, adapter_paths["vivid"])
    except ValueError as error:
        rejections["already_adapted"] = {"error": str(error)}
    try:
        load_lora_adapter(saved_merged, adapter_paths["vivid"])
    except ValueError as error:
        rejections["already_merged"] = {"error": str(error)}

cli_checks = []
for style in ["concise", "vivid"]:
    command = [sys.executable, "scripts/infer.py", base_path, "--adapter", adapter_paths[style], "--chat", "--prompt", "2+2=?", "--tokens", "96", "--device", "cpu", "--json"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    run = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True, timeout=60)
    output = json.loads(run.stdout)
    assert output["adapter"]["base_sha256_verified"]
    assert output["answer"] == ("3" if style == "concise" else "3，像把兩組積木合在一起再數。")
    cli_checks.append({"command": command, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr, "output": output})

result = {"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_08_09_audit.py", "result": "All bounded assertions passed; no training executed", "environment": environment, "source_sha256": digest(source_path), "input_sha256": input_hashes, "formal_code": formal_code, "snippet_checks": snippet_checks, "data_audit": data_audit, "switch_and_merge_cpu": switch_audit, "evaluations_cpu_against_historical": evaluations, "rejections": rejections, "cli_checks": cli_checks, "model_config": native["config"], "historical_training_configuration": {"optimizer": "AdamW", "lr": 0.003, "batch_size": 16, "steps_per_adapter": 450, "seed": 42, "rank": 4, "alpha": 4, "scaling": "alpha/rank", "loss": "masked mean cross entropy over assistant answer bytes and EOS; user/roles/padding ignored", "clip_grad_norm": 1.0, "device": "cuda", "dtype": "FP32", "schedule": "constant", "split": "exchange-addend families before byte tokenization;49/8/7 rows"}, "historical_experiment": {key: formal[key] for key in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "elapsed_seconds", "timing_scope", "step_scale"]}, "historical_switch_merge": {key: fr[key] for key in ["a_b_a_max_difference", "adapter_a_b_logit_difference", "merge_max_difference"]}, "historical_style_content_training": style_report["results"]["content_training"], "scope": "One 9-position 264-vocabulary probe; all 8 validation and 7 test samples per adapter independently regenerated on CPU; no CUDA rerun or training/timing claim."}
save("audit-result", result)
print(json.dumps({"environment": environment, "snippet_checks": snippet_checks, "switch_and_merge_cpu": switch_audit, "all_historical_generated_ids_match": all(sample["ids_equal"] for style in evaluations.values() for split in style.values() for sample in split["samples"]), "cli_answers": [entry["output"]["answer"] for entry in cli_checks], "result": result["result"]}, ensure_ascii=False, indent=2))
