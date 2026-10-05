"""Independent 17.15 checks: original fence, CPU variants, and recorded raw data."""
import ast
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, generate
from tiny_perceptron.quantization import QuantizedLinear, replace_linear_layers
from tiny_perceptron.training import load_checkpoint


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.manual_seed(17)
environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "cwd": str(Path.cwd()),
    "audit_sha256": sha(Path(__file__).read_bytes()),
    "offline": {k: os.environ.get(k, "") for k in ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]},
}
write("environment.json", environment)

# Execute the exact extracted fence as its own Python process, without changing it.
command = [sys.executable, str(OUT / "original-fence.py")]
result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30, check=False)
(OUT / "original.stdout.txt").write_bytes(result.stdout)
(OUT / "original.stderr.txt").write_bytes(result.stderr)
assert result.returncode == 0, result.stderr.decode()
assert result.stdout.decode().splitlines() == ["分數MAE 0.005", "原版選擇 貓", "量化版選擇 <EOS>"]
print("ORIGINAL_FENCE", result.stdout.decode().strip())

# The tensor is one candidate axis, not a batch/token average. Score units are arbitrary.
cases = []
for dtype in [torch.float32, torch.float64]:
    for cat in [1.005, 2.0]:
        a = torch.tensor([1.0, cat], dtype=dtype)
        b = torch.tensor([1.01, cat], dtype=dtype)
        error = (a - b).abs()
        mae = error.mean().item()
        assert abs(mae - 0.005) < 1e-8
        choices = [a.argmax().item(), b.argmax().item()]
        assert choices == ([1, 0] if cat == 1.005 else [1, 1])
        cases.append({"dtype": str(dtype), "shape": list(a.shape), "original": a.tolist(), "modified": b.tolist(), "absolute_errors": error.tolist(), "candidate_denominator": a.numel(), "mae": mae, "argmax": choices})
tie = torch.tensor([1.0, 1.0]).argmax().item()
assert tie == 0
print("NUMERIC_VARIANTS", json.dumps({"cases": cases, "tie_first_index": tie}, ensure_ascii=False))


class Scripted(torch.nn.Module):
    """A parameter-free generator that records every prefix passed into forward."""
    def __init__(self, first, feedback=False):
        super().__init__()
        self.first = first
        self.feedback = feedback
        self.config = SimpleNamespace(max_length=8)
        self.seen = []

    def forward(self, ids, cache=None):
        self.seen.append(ids.tolist())
        scores = torch.full((1, ids.shape[1], 5), -10.0)
        if ids.shape[1] == 1:
            scores[0, -1, :len(self.first)] = torch.tensor(self.first)
        elif self.feedback and ids.shape[1] == 2:
            scores[0, -1, 3 if ids[0, -1].item() == 1 else 4] = 10
        else:
            scores[0, -1, 0] = 10
        return {"logits": scores, "cache": None}


prefix = torch.tensor([[4]])
original, modified = Scripted([1.0, 1.005]), Scripted([1.01, 1.005])
orig_ids = generate(original, prefix, 3, eos_id=0).tolist()
quant_ids = generate(modified, prefix, 3, eos_id=0).tolist()
assert orig_ids == [[4, 1, 0]] and quant_ids == [[4, 0]]
a, b = Scripted([-10, 1.005, 1.0], True), Scripted([-10, 1.0, 1.005], True)
a_ids, b_ids = generate(a, prefix, 3, eos_id=0).tolist(), generate(b, prefix, 3, eos_id=0).tolist()
assert a_ids == [[4, 1, 3, 0]] and b_ids == [[4, 2, 4, 0]]
assert a.seen[1] != b.seen[1]
print("AUTOREGRESSIVE_VARIANTS", json.dumps({"original": orig_ids, "modified": quant_ids, "feedback_a": a_ids, "feedback_b": b_ids, "seen_a": a.seen, "seen_b": b.seen}))
sampling = []
for seed in [3, 17, 42]:
    pairs = []
    for _ in range(2):
        torch.manual_seed(seed)
        pairs.append(torch.multinomial(torch.tensor([0.49, 0.51]), 12, replacement=True).tolist())
    assert pairs[0] == pairs[1]
    sampling.append({"seed": seed, "draws": pairs[0]})
print("PAIRED_SEED_VARIANT", json.dumps(sampling))

# Bounded replacement for T.9 recipes: small untrained CPU model, temporary save/load.
# Nothing here re-evaluates the full model or supports a speed/quality result.
config = ModelConfig(width=4, heads=1, layers=1, max_length=8)
base = TinyLM(config).eval()
storage_check = []
with tempfile.TemporaryDirectory(prefix="factual17_15_") as temp:
    for bits in [4, 8]:
        model = replace_linear_layers(copy.deepcopy(base), bits).eval()
        path = Path(temp) / (str(bits) + ".pt")
        state = model.state_dict()
        payload = {"format_version": "quantized-v1", "config": asdict(model.config), "bits": bits, "model": state, "optimizer": None, "tokenizer": ByteTokenizer().state()}
        torch.save(payload, path)
        reloaded, loaded_payload = load_checkpoint(path, "cpu")
        x = torch.tensor([[1, 15]])
        with torch.no_grad():
            error = (model(x)["logits"] - reloaded(x)["logits"]).abs().max().item()
        assert error == 0
        tensors = [{"name": n, "shape": list(v.shape), "dtype": str(v.dtype), "numel": v.numel(), "element_size": v.element_size(), "bytes": v.numel() * v.element_size()} for n,v in loaded_payload["model"].items()]
        total = sum(v["bytes"] for v in tensors)
        assert total == sum(v.numel()*v.element_size() for v in model.state_dict().values())
        storage_check.append({"bits": bits, "tensor_bytes": total, "temporary_file_bytes": path.stat().st_size, "temporary_file_sha256": sha(path.read_bytes()), "reload_max_logit_difference": error, "model_tensors": tensors})
assert not Path(temp).exists()
print("BOUNDED_SAVE_LOAD", json.dumps(storage_check))

# Recorded data, not model inference. Only named raw measurement/provenance pointers.
dataset_raw = (OUT / "sft-dataset.raw.json").read_bytes()
assert sha(dataset_raw) == "6c93499fc2550d2f6ac221a320f1c7b2646a6bd6b7f77a0e14be7e4d2d8d590a"
dataset = json.loads(dataset_raw)
families = {s: {r["family"] for r in rows} for s,rows in dataset.items()}
assert all(not families[a] & families[b] for a,b in [("train","validation"),("train","test"),("validation","test")])
assert {s:len(rows) for s,rows in dataset.items()} == {"train":45,"validation":5,"test":10}
for rows in dataset.values():
    for row in rows:
        prompt = row["messages"][-2]["content"]
        assert all(s in prompt for s in ["color=", "shape=", "pitch="])
task_counts = {}
for split, rows in dataset.items():
    tasks = {}
    for row in rows:
        task = row["messages"][-2]["content"].split(";")[-1]
        tasks[task] = tasks.get(task, 0) + 1
    task_counts[split] = tasks
print("DATASET_TASKS", json.dumps(task_counts))

# Compile only the raw data-layout helpers from the hash-matched original implementation.
original_raw = (OUT / "compression-original.py").read_bytes()
assert sha(original_raw) == "28d8ce258369672d71a24f7efbcbf68f3e5e8114e073620aba91c4d807043a44"
tree = ast.parse(original_raw)
functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ["_chat","_example","_examples"]]
ns = {"torch":torch,"ByteTokenizer":ByteTokenizer,"IGNORE":IGNORE,"render_chat":render_chat}
exec(compile(ast.Module(body=functions, type_ignores=[]), "compression-original:selected-data-functions", "exec"), ns)
chunks = ns["_examples"](dataset["train"], 128)
plan_checks = {}
for steps in [120,350]:
    rng = random.Random(42)
    plan = [[rng.randrange(len(chunks)) for _ in range(16)] for _ in range(steps)]
    effective = sum(int((chunks[i][1] != IGNORE).sum()) for indices in plan for i in indices)
    plan_checks[steps] = {"batch_plan_sha256":sha(json.dumps(plan).encode()), "effective_supervised_tokens":effective}

raw_audit = {"dataset_sha256":sha(dataset_raw), "family_counts":{s:len(v) for s,v in families.items()}, "plans":plan_checks,"files":{}}
for name in ["quantization","qat"]:
    path = ROOT / "docs/course-experiments/results" / (name + ".json")
    raw = path.read_bytes()
    full = json.loads(raw)
    data = full["results"]
    assert data["data"]["sha256"] == sha(dataset_raw)
    assert data["data"]["counts"] == {s:len(v) for s,v in dataset.items()}
    assert data["data"]["families"] == {s:len(v) for s,v in families.items()}
    assert data["data"]["family_intersections"] == 0
    trains = {"ptq":data["training"]} if name == "quantization" else data["training"]
    checked_train = {}
    for branch,t in trains.items():
        expected_steps = 120 if name == "quantization" else 350
        assert t["steps"] == t["optimizer_updates"] == expected_steps
        assert t["batch_plan_sha256"] == plan_checks[expected_steps]["batch_plan_sha256"]
        assert t["effective_supervised_tokens"] == plan_checks[expected_steps]["effective_supervised_tokens"]
        checked_train[branch] = {k:t[k] for k in ["steps","optimizer_updates","batch_size","learning_rate","initialization_sha256","batch_plan_sha256","effective_supervised_tokens"]}
    if name == "qat":
        assert checked_train["fp_finetuning"] == checked_train["qat"]
    runs = {}
    for branch, run in data["runs"].items():
        checked = {}
        for split in ["validation","test"]:
            evaluation = run[split]
            samples = evaluation["generated_samples"]
            assert len(samples) == len(dataset[split]) == evaluation["examples"]
            target_count = sum(len(ByteTokenizer().encode(r["messages"][-1]["content"]))+1 for r in dataset[split])
            assert evaluation["supervised_tokens"] == target_count
            assert math.isclose(evaluation["answer_nll"], evaluation["nll_sum"]/target_count, abs_tol=1e-12)
            correct = eos = completed = 0
            for row,sample in zip(dataset[split],samples):
                assert sample["question"] == row["messages"][-2]["content"]
                assert sample["expected"] == row["messages"][-1]["content"]
                assert sample["family"] == row["family"]
                new = sample["generated_ids"]
                ended = ByteTokenizer().eos_id in new
                answer_ids = new[:new.index(ByteTokenizer().eos_id)] if ended else new
                exact = answer_ids == ByteTokenizer().encode(sample["expected"])
                assert exact == sample["exact"] and ended == sample["ended_with_eos"]
                assert ByteTokenizer().decode(answer_ids) == sample["generated"]
                correct += exact; eos += ended; completed += exact and ended
            assert correct == evaluation["correct"] and eos == evaluation["eos_count"] and completed == evaluation["completed_correct"]
            assert evaluation["exact_match"] == correct/len(samples)
            assert evaluation["eos_rate"] == eos/len(samples)
            assert evaluation["max_new_tokens"] == 24 and evaluation["generation"] == "greedy, full recompute, no KV cache"
            checked[split] = {"examples":len(samples),"supervised_tokens":target_count,"correct":correct,"eos":eos,"answer_nll":evaluation["answer_nll"],"sample_criteria":"raw answer ID identity; EOS checked independently"}
        storage = run["storage"]
        assert storage["tensor_bytes"] == storage["parameter_tensor_bytes"] + storage["buffer_tensor_bytes"]
        assert storage["buffer_tensor_bytes"] == sum(storage["buffers"].values())
        assert storage["file_overhead_bytes"] == storage["file_bytes"] - storage["tensor_bytes"]
        checked["storage"] = {k:storage[k] for k in ["file_bytes","tensor_bytes","parameter_tensor_bytes","buffer_tensor_bytes","forward"]}
        if "timing" in run:
            timing = run["timing"]
            assert timing["repetitions"] == 3 and timing["decode_tokens"] == 8
            assert math.isclose(timing["decode_seconds_per_token"],timing["decode_seconds"]/8,abs_tol=1e-15)
            assert timing["cuda_additional_peak_bytes"] == timing["cuda_peak_allocated"] - timing["cuda_allocated_before"]
            checked["timing"] = {k:timing[k] for k in ["repetitions","decode_tokens","decode_seconds_per_token","cuda_allocated_before","cuda_peak_allocated","cuda_additional_peak_bytes"]}
        runs[branch] = checked
    pointers = ["/"+k for k in ["revision","device","seed","torch_version","python_version"]]
    pointers += ["/results/data/"+k for k in ["sha256","counts","families","family_intersections"]]
    pointers += ["/results/teacher_provenance/"+k for k in ["experiment","checkpoint","sha256","steps"]]
    for branch in trains:
        prefix = "/results/training/" if name == "quantization" else "/results/training/"+branch+"/"
        pointers += [prefix+k for k in checked_train[branch]]
    for branch,run in data["runs"].items():
        prefix = "/results/runs/"+branch+"/"
        for split in ["validation","test"]:
            pointers += [prefix+split+"/"+k for k in ["nll_sum","supervised_tokens","answer_nll","examples","correct","exact_match","completed_correct","eos_count","eos_rate","generation","max_new_tokens","generated_samples"]]
        pointers += [prefix+"storage/"+k for k in ["file_bytes","tensor_bytes","parameter_tensor_bytes","buffer_tensor_bytes","buffers","file_overhead_bytes","forward"]]
        if name == "quantization":pointers += [prefix+"timing/"+k for k in ["repetitions","decode_tokens","decode_seconds","decode_seconds_per_token","cuda_allocated_before","cuda_peak_allocated","cuda_additional_peak_bytes"]]
    raw_audit["files"][name] = {"path":str(path.relative_to(ROOT)),"sha256":sha(raw),"actual_pointers":pointers,"training":checked_train,"runs":runs,"teacher_checkpoint_input_identity":data["teacher_provenance"]["sha256"]}
write("raw-audit.json",raw_audit)
print("RAW_RECORDED_AUDIT",json.dumps({"files":{k:{"sha256":v["sha256"],"training":v["training"],"runs":v["runs"],"pointer_count":len(v["actual_pointers"])} for k,v in raw_audit["files"].items()},"family_counts":raw_audit["family_counts"],"plans":raw_audit["plans"]},ensure_ascii=False))
write("cpu-values.json",{"scores":cases,"sampling":sampling,"storage":storage_check,"autoregressive":{"orig":orig_ids,"quant":quant_ids,"feedback_a":a_ids,"feedback_b":b_ids}})
print("COMPLETE: original fence and bounded variants passed; recorded JSON checks passed; no full-model evaluation, no training, no retained weights.")
