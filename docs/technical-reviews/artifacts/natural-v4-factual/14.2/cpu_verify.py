"""Independent bounded verification for 14.2; no model training or GPU benchmark."""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum
from tiny_perceptron.modern import RMSNorm
from scripts.course_experiments.common import text_examples

torch.set_num_threads(1)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

environment = {"python": platform.python_version(), "torch": str(torch.__version__),
               "device": "cpu", "cuda_available": torch.cuda.is_available(),
               "threads": torch.get_num_threads(), "dtype": "torch.float32"}
assert environment["python"] == "3.13.5"
assert environment["torch"] == "2.14.1+cpu"
assert not environment["cuda_available"]
section = (HERE / "original-section.md").read_text()
code = re.search(r"```python\n(.*?)\n```", section, re.S)[1]
(HERE / "main-excerpt.py").write_text(code + "\n")
predictions = {"main_LN": [[0, 0, 0], [-1.2247, 0, 1.2247]],
               "main_RMS": [[1, 1, 1], [0.4629, 0.9258, 1.3887]],
               "shift_by_10": "LN invariant within float rounding; RMS generally changes (second row)",
               "zero_input": "both zero and finite with eps=1e-5",
               "parameters": {"baseline": 141568, "rmsnorm": 141248, "difference": 320},
               "record_targets": {"validation": 39256, "test": 41914, "training_sampled": 452102}}
print("PREDICTIONS", json.dumps(predictions, ensure_ascii=False))
print("ENVIRONMENT", json.dumps(environment))
def execute_excerpt(text):
    values, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(text, "course/chapters/14.md#14.2", "exec"), values)
    return values, output.getvalue()

main, main_stdout = execute_excerpt(code)
shifted, shift_stdout = execute_excerpt(code.replace("eps = 1e-5", "x = x + 10\neps = 1e-5"))
print("EXACT MAIN EXCERPT OUTPUT\n" + main_stdout)
print("EXERCISE x=x+10 OUTPUT\n" + shift_stdout)
ln, rms = main["ln"], main["rms"]
epsilon = 1e-5
ln_denominator = math.sqrt(2/3 + epsilon)
rms_denominator = math.sqrt(14/3 + epsilon)
hand_ln = torch.tensor([[0,0,0], [-1/ln_denominator,0,1/ln_denominator]], dtype=torch.float64)
hand_rms = torch.tensor([[2/math.sqrt(4+epsilon)]*3,
                         [v/rms_denominator for v in [1,2,3]]], dtype=torch.float64)
ln_error = float((ln.double()-hand_ln).abs().max())
rms_error = float((rms.double()-hand_rms).abs().max())
assert ln_error < 2e-7 and rms_error < 2e-7
assert torch.allclose(ln, shifted["ln"], atol=1e-6, rtol=0)
assert not torch.allclose(rms[1], shifted["rms"][1])
assert ln.shape == rms.shape == (2,3)
assert main["x"].square().mean(-1, keepdim=True).shape == (2,1)
zeros = torch.zeros(2,3)
zero_ln = F.layer_norm(zeros, (3,), eps=epsilon)
zero_rms = zeros / torch.sqrt(zeros.square().mean(-1, keepdim=True)+epsilon)
assert bool(torch.isfinite(zero_ln).all() and torch.isfinite(zero_rms).all())
assert torch.equal(zero_ln, zeros) and torch.equal(zero_rms, zeros)
changed = main["x"].clone(); changed[1,0] += 100
assert torch.equal(F.layer_norm(changed, (3,), eps=epsilon)[0], ln[0])
assert torch.equal(RMSNorm(3)(changed)[0], rms[0])
assert torch.allclose(RMSNorm(3)(main["x"]), rms, atol=2e-7, rtol=0)
assert torch.allclose(torch.nn.RMSNorm(3, eps=epsilon)(main["x"]), rms, atol=2e-7, rtol=0)
numeric = {"LN": ln.tolist(), "RMS": rms.tolist(), "shifted_LN": shifted["ln"].tolist(),
           "shifted_RMS": shifted["rms"].tolist(), "LN_max_abs_error_vs_scalar_math": ln_error,
           "RMS_max_abs_error_vs_scalar_math": rms_error,
           "LN_denominator_with_eps": ln_denominator,
           "RMS_without_eps": math.sqrt(14/3), "RMS_denominator_with_eps": rms_denominator,
           "all_zero_finite": True, "per_position_independent": True,
           "affine_defaults": {"F.layer_norm_weight": "None = identity", "bias": "None = zero"}}

record_path = ROOT / "docs/course-experiments/results/modern.json"
record = json.loads(record_path.read_text())
variants = record["results"]["variants"]
models, counts, norm_modules, bounded_backward = {}, {}, {}, {}
for name in ["baseline", "rmsnorm"]:
    torch.manual_seed(42)
    model = TinyLM(ModelConfig(**variants[name]["model"]["config"]))
    models[name] = model
    counts[name] = sum(p.numel() for p in model.parameters())
    assert counts[name] == predictions["parameters"][name]
    norm_modules[name] = [{"name": key, "weight_numel": m.weight.numel(),
                          "bias_numel": 0 if getattr(m,"bias",None) is None else m.bias.numel()}
                         for key,m in model.named_modules()
                         if isinstance(m, (torch.nn.LayerNorm,RMSNorm))]
    assert len(norm_modules[name]) == 5
    ids = torch.tensor([[1,11,12,13],[1,14,15,16]])
    labels = torch.tensor([[11,12,13,2],[14,15,16,2]])
    output = model(ids)["logits"]
    summed, tokens = loss_sum(output, labels)
    loss = summed/tokens
    loss.backward()
    assert bool(torch.isfinite(output).all() and torch.isfinite(loss))
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    bounded_backward[name] = {"input_shape": list(ids.shape), "output_shape": list(output.shape),
                              "effective_targets": int(tokens), "loss": float(loss.detach()),
                              "all_output_and_gradient_finite": True, "optimizer_updates": 0}
assert counts["baseline"] - counts["rmsnorm"] == 5*64 == 320

# Independently reconstruct whole-story family split and target lengths from original 512 records.
data_path = ROOT / "data/training/text-initial/tinystories-train-512.jsonl"
rows = [json.loads(s) for s in data_path.read_text().splitlines() if s.strip()]
assert len(rows) == 512
groups = {}
seen = set()
for row in rows:
    text_sha = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    assert text_sha == row["text_sha256"]
    row["family"] = text_sha
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
    if encoded not in seen:
        seen.add(encoded); groups.setdefault(text_sha,[]).append(row)
keys = sorted(groups)
random.Random(42).shuffle(keys)
a,b = int(len(keys)*0.8),int(len(keys)*0.9)
data = {n:[row for key in selected for row in groups[key]]
        for n,selected in [("train",keys[:a]),("validation",keys[a:b]),("test",keys[b:])]}
split_summary = {}
window_lengths = []
for name, selected in data.items():
    h = hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert h == record["results"]["dataset"][name]["sha256"]
    targets = sum(len(row["text"].encode("utf-8"))+1 for row in selected)
    windows = sum(math.ceil((len(row["text"].encode("utf-8"))+1)/128) for row in selected)
    split_summary[name] = {"stories": len(selected), "records_sha256":h,
                           "effective_next_byte_or_EOS_targets":targets, "windows": windows}
    production_examples = text_examples(selected, max_length=128)
    assert len(production_examples) == windows
    assert sum(len(y) for x,y in production_examples) == targets
    if name == "train":
        for row in selected:
            n = len(row["text"].encode("utf-8"))+1
            window_lengths.extend(min(128,n-start) for start in range(0,n,128))
assert [len(data[n]) for n in ["train","validation","test"]] == [409,51,52]
assert not (set(r["family"] for r in data["train"]) & set(r["family"] for r in data["validation"]))
assert not (set(r["family"] for r in data["train"]) & set(r["family"] for r in data["test"]))
assert not (set(r["family"] for r in data["validation"]) & set(r["family"] for r in data["test"]))
sampler=random.Random(42)
training_targets=sum(sum(sampler.choices(window_lengths,k=16)) for _ in range(240))
assert training_targets == 452102

old_path = ROOT / "outputs/natural-v4/factual-research/14.2/originals/original-architecture.py"
assert digest(old_path) == record["code_sha256"]["scripts/course_experiments/architecture.py"]
def nodes(path):
    return {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
old,current=nodes(old_path),nodes(ROOT/"scripts/course_experiments/architecture.py")
identical=["_sync","_clone_config","_copy_matching","_text_dataset","_nll","_train","_heldout","run_modern"]
assert all(old[n] == current[n] for n in identical)
run_numbers = {}
for name in ["baseline", "rmsnorm"]:
    v = variants[name]
    t = v["training"]
    assert t["steps"] == t["optimizer_updates"] == t["requested_steps"] == 240
    assert t["skipped_updates"] == 0 and t["effective_tokens"] == training_targets
    assert t["batch_size"] == 16 and t["learning_rate"] == 0.003
    assert t["all_requested_attempts_completed"] and not t["scaler_enabled"]
    run_numbers[name] = {}
    for split in ["validation", "test"]:
        entry = v["heldout"][split]
        denom = split_summary[split]["effective_next_byte_or_EOS_targets"]
        assert entry["effective_tokens"] == denom
        assert entry["examples"] == split_summary[split]["windows"]
        assert entry["records"] == split_summary[split]["stories"]
        nll = entry["nll_sum"]/denom
        assert nll == entry["nll"]
        run_numbers[name][split] = {"nll_sum_nats":entry["nll_sum"], "denominator":denom,
                                   "recomputed_nll":nll,"rounded_5dp":f"{nll:.5f}"}
    run_numbers[name]["warm_step_median_ms"] = t["warm_step_median_seconds"]*1000
    run_numbers[name]["rounded_median_ms_3dp"] = f"{t['warm_step_median_seconds']*1000:.3f}"
assert record["seed"] == record["results"]["seed"] == 42
assert record["gpu"] == "NVIDIA L4" and record["torch_version"] == "2.14.1+cu126"
assert record["results"]["runtime"]["cuda_matmul_allow_tf32"] is False
assert [run_numbers[n][s]["rounded_5dp"] for n in ["baseline","rmsnorm"]
        for s in ["validation","test"]] == ["2.26331","2.29163","2.25913","2.28631"]
assert [run_numbers[n]["rounded_median_ms_3dp"] for n in ["baseline","rmsnorm"]] == ["13.062","13.952"]

torch.manual_seed(42); init1 = TinyLM(ModelConfig(width=64,layers=2,heads=4)).embedding.weight.detach().clone()
torch.manual_seed(42); init2 = TinyLM(ModelConfig(width=64,layers=2,heads=4)).embedding.weight.detach().clone()
torch.manual_seed(43); init3 = TinyLM(ModelConfig(width=64,layers=2,heads=4)).embedding.weight.detach().clone()
assert torch.equal(init1,init2) and not torch.equal(init1,init3)
sample1=random.Random(42).choices(range(2789),k=16)
sample2=random.Random(42).choices(range(2789),k=16)
sample3=random.Random(43).choices(range(2789),k=16)
assert sample1 == sample2 and sample1 != sample3

result = {"environment": environment, "source_sha256":digest(HERE/"original-section.md"),
          "predictions_before_checks":predictions, "numeric_results":numeric,
          "parameters":counts,"normalization_modules":norm_modules,"bounded_backward":bounded_backward,
          "record_sha256":digest(record_path), "data_input_sha256":digest(data_path),
          "splits":split_summary,"sampled_training_targets":training_targets,
          "original_architecture_sha256":digest(old_path),"identical_original_current_AST_functions":identical,
          "record_values_recomputed":run_numbers, "same_seed_initialization_and_sampling_repeat":True,
          "changed_seed_initialization_and_sampling_differ":True,
          "limitations":"CPU mechanism and original-record accounting only; no optimizer updates, no GPU reproduction. "
          "Raw per-step timing samples are absent from the original report, so the median algorithm and its saved value/units "
          "were checked, not the median independently rebuilt. Single seed, 240 updates, FP32 L4 small corpus cannot "
          "establish stable quality or speed advantage. Near-duplicate stories are not family-clustered."}
(HERE/"cpu-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print("VERIFIED RESULTS",json.dumps(result,ensure_ascii=False,indent=2))
print("ALL BOUNDED CHECKS PASSED")
