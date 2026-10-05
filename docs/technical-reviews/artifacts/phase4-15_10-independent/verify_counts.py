"""Independent bounded CPU check of section 15.10; no data or checkpoints."""

import ast
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import MoEFFN

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
torch.manual_seed(23)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

raw_results = (OUT / "original-moe-results.json").read_bytes()
results = json.loads(raw_results)
for filename in ("tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/attention.py"):
    assert sha((ROOT / filename).read_bytes()) == results["code_sha256"][filename]

fence = (OUT / "fence-1.py").read_bytes()
fence_outputs = {}
for name, code, expected in (
    ("original_k2", fence, "expert數 4 總權重 272 每token使用估計 144 FP32權重bytes 1088\nexpert數 8 總權重 544 每token使用估計 160 FP32權重bytes 2176\n"),
    ("exercise_k1", fence.replace(b"width, hidden, k = 4, 8, 2", b"width, hidden, k = 4, 8, 1"), "expert數 4 總權重 272 每token使用估計 80 FP32權重bytes 1088\nexpert數 8 總權重 544 每token使用估計 96 FP32權重bytes 2176\n"),
):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(code, name, "exec"), {})
    assert capture.getvalue() == expected
    fence_outputs[name] = capture.getvalue()
    if name == "exercise_k1":
        (OUT / "exercise-k1.py").write_bytes(code)

# Execute the actual immutable budget function, excluding unrelated result prose.
raw_architecture = (OUT / "original-architecture.py").read_bytes()
assert sha(raw_architecture) == results["code_sha256"]["scripts/course_experiments/architecture.py"]
tree = ast.parse(raw_architecture)
budget_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_parameter_budget")
budget_namespace = {"MoEFFN": MoEFFN}
exec(compile(ast.Module(body=[budget_node], type_ignores=[]), "immutable-architecture-budget", "exec"), budget_namespace)
budget_function = budget_namespace["_parameter_budget"]

variants = {}
for name in ("top1_aux0", "top1_aux0.01", "top2_aux0", "top2_aux0.01"):
    recorded = results["results"]["variants"][name]
    model = TinyLM(ModelConfig(**recorded["model"]["config"]))
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    observed = budget_function(model)
    assert observed == recorded["budget"]
    other = observed["total_parameters"] - observed["expert_parameters"] - observed["router_parameters"]
    assert other == 75392
    assert sum(p.numel() * p.element_size() for p in model.parameters()) == recorded["model"]["parameter_bytes"]
    assert model.embedding.weight.numel() == 264 * 64
    assert model.position.weight.numel() == 128 * 64
    # Lookup contract: one token ID / one position produce one row each.
    token_row = model.embedding(torch.tensor([7], device="cpu"))
    position_row = model.position(torch.tensor([3], device="cpu"))
    assert tuple(token_row.shape) == tuple(position_row.shape) == (1, 64)
    assert sum(p.numel() for p in model.blocks[0].ffn.experts[0].parameters()) == 33088
    assert observed["expert_parameters"] == 2 * 4 * (64 * 256 + 256 + 256 * 64 + 64)
    variants[name] = {"observed_budget": observed, "other_parameters_excluding_router": other, "full_embedding_and_position_parameters": 25088, "lookup_output_elements_for_one_token": token_row.numel() + position_row.numel(), "one_expert_parameters_including_bias": 33088, "observed_fp32_parameter_bytes": recorded["model"]["parameter_bytes"]}

assert torch.empty(0, dtype=torch.float32, device="cpu").element_size() == 4
report = {"fence_outputs": fence_outputs, "variants": variants, "device": "cpu", "no_data_or_weights_loaded": True, "no_model_training_or_quality_evaluation": True, "all_assertions_passed": True}
(OUT / "counts-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
environment = {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "device": "cpu", "threads": str(torch.get_num_threads()), "cwd": str(Path.cwd()), "repo_root": str(ROOT), "official_contract_versions": "PyTorch 2.8 docs / v2.8.0 Adam source; installed runtime separately verified as 2.14.1+cpu"}
(OUT / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
