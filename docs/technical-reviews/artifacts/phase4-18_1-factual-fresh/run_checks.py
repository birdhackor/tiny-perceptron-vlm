"""Independent, bounded CPU checks for lesson 18.1; no downloads or model scoring."""

import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

import torch
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import IGNORE, render_chat, pad_batch
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


env = {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__, "torch_git_version": torch.version.git_version, "tokenizers": tokenizers.__version__, "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "platform": platform.platform(), "threads": str(torch.get_num_threads()), "offline": "HF_HUB_OFFLINE=1; HF_DATASETS_OFFLINE=1; TRANSFORMERS_OFFLINE=1; CUDA_VISIBLE_DEVICES=''"}
dump("environment.json", env)

# Execute the exact saved fence, and the exact requested exercise replacement.
fence = HERE / "inputs/fence-1.py"
metadata = json.loads((HERE / "inputs/metadata.json").read_bytes())
assert sha(fence) == metadata["fences"][0]["sha256"]
done = subprocess.run([sys.executable, str(fence)], cwd=ROOT, text=True, capture_output=True, timeout=30)
(HERE / "fence.stdout.txt").write_text(done.stdout)
(HERE / "fence.stderr.txt").write_text(done.stderr)
assert done.returncode == 0
assert len(done.stdout.splitlines()) == 2 and all(x.startswith("回答 4 分布 ") for x in done.stdout.splitlines())
exercise = fence.read_text().replace('print("回答", answer, "分布", prob.tolist())', 'print("回答", answer)')
(HERE / "exercise.py").write_text(exercise)
changed = subprocess.run([sys.executable, str(HERE / "exercise.py")], cwd=ROOT, text=True, capture_output=True, timeout=30)
(HERE / "exercise.stdout.txt").write_text(changed.stdout)
(HERE / "exercise.stderr.txt").write_text(changed.stderr)
assert changed.returncode == 0 and changed.stdout == "回答 4\n回答 4\n"
print("ORIGINAL FENCE:", done.stdout, sep="\n", end="")
print("EXERCISE:", changed.stdout, sep="\n", end="")

# Numeric values are dimensionless; the candidate axis is the last dimension.
rows = torch.tensor([[0.8, 0.15, 0.05], [0.4, 0.35, 0.25]], dtype=torch.float64)
assert torch.allclose(rows.sum(-1), torch.ones(2, dtype=torch.float64), rtol=0, atol=1e-15)
assert rows.argmax(-1).tolist() == [0, 0]
reconstructed = rows.log().softmax(-1)
assert torch.allclose(reconstructed, rows, rtol=0, atol=1e-15)
numbers = {"input_rows": rows.tolist(), "row_sums": rows.sum(-1).tolist(), "argmax_indices": rows.argmax(-1).tolist(), "top1_minus_top2": (rows[:, 0]-rows[:, 1]).tolist(), "softmax_log_probability_max_error": float((reconstructed-rows).abs().max()), "axis": "last dimension = 3 candidate tokens", "units": "dimensionless probabilities; no empirical sample denominator"}

# Same text can be independently encoded with distinct student/teacher vocabularies.
tok_a = Tokenizer(models.WordLevel({"[UNK]": 0, "4": 1, "3": 2, "5": 3}, unk_token="[UNK]"))
tok_b = Tokenizer(models.WordLevel({"4": 0, "5": 1, "[UNK]": 2, "3": 3}, unk_token="[UNK]"))
for tok in (tok_a, tok_b):
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
ids_a, ids_b = tok_a.encode("4").ids, tok_b.encode("4").ids
assert ids_a == [1] and ids_b == [0]
# A column permutation must be undone before direct distribution comparison.
p = rows[0]
q_swapped = p[torch.tensor([1, 0, 2])]
naive_kl = (p * (p.log()-q_swapped.log())).sum()
aligned_kl = (p * (p.log()-q_swapped[torch.tensor([1, 0, 2])].log())).sum()
assert naive_kl > 0 and aligned_kl == 0
numbers.update(teacher_token_ids=ids_a, student_token_ids=ids_b, same_text="4", permuted_column_kl_nats=float(naive_kl), aligned_column_kl_nats=float(aligned_kl))
dump("numeric-checks.json", numbers)
print("NUMERIC/ALIGNMENT:", json.dumps(numbers, ensure_ascii=False))

# Load the exact experiment-era implementation, not a guessed reconstruction.
recorded = HERE / "inputs/compression-recorded.py"
raw_results = json.loads((HERE / "inputs/distillation-original.json").read_bytes())
assert sha(recorded) == raw_results["code_sha256"]["scripts/course_experiments/compression.py"]
name = "scripts.course_experiments.compression_review18_1"
spec = importlib.util.spec_from_file_location(name, recorded)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
actual_tree = ast.parse((ROOT / "scripts/course_experiments/compression.py").read_text())
original_tree = ast.parse(recorded.read_text())
matching = {}
for fname in ("_teacher", "_cache_text", "_fit_text", "_distill_case", "run_distillation"):
    old = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == fname)
    new = next(n for n in actual_tree.body if isinstance(n, ast.FunctionDef) and n.name == fname)
    matching[fname] = ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
assert all(matching.values())

# Exactly two synthetic rows, one forward/backward/update for each objective.
torch.manual_seed(18)
teacher = TinyLM(ModelConfig(width=8, layers=1, heads=2, max_length=32))
torch.manual_seed(19)
initial_student = TinyLM(ModelConfig(width=8, layers=1, heads=2, max_length=32))
records = [{"question": "2+2=?", "answer": "4"}, {"question": "1+2=?", "answer": "3"}]
teacher_before = module._parameter_hash(teacher)
cache, _ = module._cache_text(teacher, records, "cpu")
assert [list(x.shape) for x in cache] == [[2, 264], [2, 264]]
x, labels, valid = pad_batch([render_chat(module._chat(r)) for r in records])
student_logits = initial_student(x, valid=valid)["logits"]
teacher_logits = teacher(x, valid=valid)["logits"]
assert list(student_logits.shape) == [2, 10, 264]
assert int((labels != IGNORE).sum()) == 4
kl = distillation_kl(student_logits, teacher_logits, labels, 2.0)
tp = (teacher_logits.detach().float() / 2).softmax(-1)
lq = (student_logits.float() / 2).log_softmax(-1)
manual = (tp * (tp.log()-lq)).sum(-1)[labels != IGNORE].sum() / 4 * 4
assert torch.allclose(kl, manual, rtol=1e-5, atol=1e-6)
ce = masked_loss(student_logits, labels)
manual_ce = -student_logits.log_softmax(-1)[labels != IGNORE].gather(1, labels[labels != IGNORE].unsqueeze(-1)).mean()
assert torch.allclose(ce, manual_ce, rtol=1e-6, atol=1e-6)
interface = {"records": 2, "input_shape": list(x.shape), "logits_shape": list(student_logits.shape), "logits_dtype": str(student_logits.dtype), "cache_shapes": [list(v.shape) for v in cache], "cache_dtype": str(cache[0].dtype), "cache_first_position_first_five_values": cache[0][0, :5].tolist(), "effective_answer_positions": 4, "vocab_columns": 264, "ce_nats_per_effective_position": float(ce.detach()), "kl_teacher_to_student_T2_scaled": float(kl.detach()), "manual_kl": float(manual.detach()), "denominator": "4 effective answer positions after summing 264 vocabulary columns", "T": 2.0, "T_squared": 4.0, "tolerance": "KL rtol=1e-5 atol=1e-6; CE rtol=1e-6 atol=1e-6", "matching_current_method_AST": matching}
with tempfile.TemporaryDirectory(prefix="review18_1-cpu-") as temp:
    ctx = SimpleNamespace(seed=42, device="cpu", output=Path(temp))
    results = {}
    for method in ("ce", "ce_kl"):
        model, path, train = module._fit_text(ctx, copy.deepcopy(initial_student), records, method, steps=1, teacher_cache=cache if method == "ce_kl" else None)
        assert train["optimizer_updates"] == 1 and train["weights_changed"] and train["first_output_weight_gradient_norm"] > 0
        assert train["effective_supervised_tokens"] == 32
        results[method] = {k: train[k] for k in ["steps", "optimizer_updates", "batch_size", "training_examples", "training_sequence_chunks", "effective_supervised_tokens", "initialization_sha256", "final_sha256", "weights_changed", "first_output_weight_gradient_norm", "objective", "alpha", "temperature", "temperature_squared_applied_once", "batch_plan_sha256", "loss_trace"]}
    assert results["ce"]["initialization_sha256"] == results["ce_kl"]["initialization_sha256"]
    assert results["ce"]["batch_plan_sha256"] == results["ce_kl"]["batch_plan_sha256"]
    saved_receipts = [{"name": p.name, "sha256": sha(p), "bytes": p.stat().st_size, "retained": False} for p in Path(temp).glob("*.pt")]
    assert len(saved_receipts) == 4
assert not Path(temp).exists()
assert module._parameter_hash(teacher) == teacher_before
assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
interface.update(one_step_runs=results, teacher_frozen_and_unchanged=True, temporary_checkpoint_receipts=saved_receipts, all_temporary_weights_deleted=True, scope="Only synthetic CPU interface/gradient/update verification, no teacher/student capability evaluation or reproduction of full training.")
dump("bounded-interface.json", interface)
print("BOUNDED INTERFACE:", json.dumps(interface, ensure_ascii=False))

# Independently cross-link original raw experiment outputs, without reading scores/notes.
inspection = {"raw_distillation_sha256": sha(HERE / "inputs/distillation-original.json"), "original_revision": raw_results["revision"], "tasks": {}, "read_pointers": []}
for task, source_name, artifact_index, expected_steps in [("attributes", "sft", 10, 900), ("style_transfer", "style", 38, 1000), ("moe_to_dense", "moe", 5, 180)]:
    source_file = HERE / "inputs" / (source_name + "-original.json")
    source = json.loads(source_file.read_bytes())
    source_artifact = source["artifacts"][artifact_index]
    provenance = raw_results["results"]["tasks"][task]["teacher_provenance"]
    assert source_artifact["path"] == "model.pt" and source_artifact["sha256"] == provenance["sha256"]
    if source_name == "moe":
        assert source["results"]["teacher_variant"] == "top2_aux0.01"
        training = source["results"]["variants"]["top2_aux0.01"]["training"]
        assert source["results"]["variants"]["top2_aux0.01"]["model"]["config"] == provenance["config"]
        assert provenance["config"]["experts"] == 4 and provenance["config"]["top_k"] == 2
        src_pointer = "/results/variants/top2_aux0.01/training"
    else:
        training = source["results"]["training"]
        src_pointer = "/results/training"
    assert provenance["steps"] == training["steps"] == expected_steps
    assert provenance["metadata"]["effective_tokens"] == training["effective_tokens"]
    ordinary = [key for key in raw_results["results"]["tasks"][task]["runs"] if key.endswith("_ce") or key.endswith("_ce_kl")]
    runs = {}
    for key in ordinary:
        train = raw_results["results"]["tasks"][task]["runs"][key]["training"]
        assert train["steps"] == train["optimizer_updates"] and train["weights_changed"]
        expected_objective = "CE" if key.endswith("_ce") else "(1-alpha)*CE + alpha*KL(teacher||student)*T²"
        assert train["objective"] == expected_objective
        runs[key] = {k: train[k] for k in ["steps", "optimizer_updates", "batch_size", "training_examples", "training_sequence_chunks", "effective_supervised_tokens", "objective", "alpha", "temperature", "temperature_squared_applied_once", "initialization_sha256", "batch_plan_sha256", "weights_changed"]}
        inspection["read_pointers"].append("distillation:/results/tasks/"+task+"/runs/"+key+"/training/{"+",".join(runs[key])+"}")
    inspection["tasks"][task] = {"teacher_provenance": provenance, "source_result_file": str(source_file.relative_to(ROOT)), "source_result_sha256": sha(source_file), "source_revision": source["revision"], "source_model_artifact_pointer": "/artifacts/"+str(artifact_index), "source_model_artifact": source_artifact, "source_training_pointer": src_pointer, "source_steps": training["steps"], "source_effective_tokens": training["effective_tokens"], "source_hash_and_steps_match": True, "formal_runs": runs}
    inspection["read_pointers"].extend(["distillation:/results/tasks/"+task+"/teacher_provenance", source_name+":/artifacts/"+str(artifact_index), source_name+":"+src_pointer+"/steps", source_name+":"+src_pointer+"/effective_tokens"])
    if source_name == "moe":
        inspection["read_pointers"].extend(["moe:/results/teacher_variant", "moe:/results/variants/top2_aux0.01/model/config"])
dump("raw-provenance-checks.json", inspection)
print("ORIGINAL RAW PROVENANCE:", json.dumps(inspection, ensure_ascii=False))
print("ALL BOUNDED CHECKS PASSED")
