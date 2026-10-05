"""Bounded CPU checks for 14.2; no training, weight loading, or data download."""
from pathlib import Path
import hashlib
import json
import math
import platform
import random
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import RMSNorm

ART = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
eps = 1e-5
x = torch.tensor([[2., 2., 2.], [1., 2., 3.]])
ln = F.layer_norm(x, (3,), eps=eps)
rms = x / torch.sqrt(x.square().mean(-1, keepdim=True) + eps)
manual_ln = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, correction=0) + eps)
assert torch.allclose(ln, manual_ln, atol=2e-7, rtol=0)
expected_ln = torch.tensor([[0., 0., 0.], [-1.2247, 0., 1.2247]])
expected_rms = torch.tensor([[1., 1., 1.], [0.4629, 0.9258, 1.3887]])
assert torch.allclose(ln, expected_ln, atol=5e-5, rtol=0)
assert torch.allclose(rms, expected_rms, atol=5e-5, rtol=0)
assert torch.allclose(RMSNorm(3, eps)(x), rms, atol=2e-7, rtol=0)
assert torch.allclose(nn.RMSNorm(3, eps=eps)(x), rms, atol=2e-7, rtol=0)
shifted = x + 10
shifted_ln = F.layer_norm(shifted, (3,), eps=eps)
shifted_rms = shifted / (shifted.square().mean(-1, keepdim=True) + eps).sqrt()
assert torch.allclose(shifted_ln, ln, atol=2e-7, rtol=0)
assert not torch.allclose(shifted_rms, rms)
z = torch.zeros_like(x)
assert torch.isfinite(F.layer_norm(z, (3,), eps=eps)).all()
assert torch.isfinite(RMSNorm(3, eps)(z)).all()
perturbed = x.clone()
perturbed[1, 0] += 10
assert torch.equal(F.layer_norm(perturbed, (3,), eps=eps)[0], ln[0])
assert torch.equal(RMSNorm(3, eps)(perturbed)[0], RMSNorm(3, eps)(x)[0])
layer = nn.LayerNorm(3, eps=eps)
assert torch.equal(layer.weight, torch.ones(3)) and torch.equal(layer.bias, torch.zeros(3))
assert tuple(x.square().mean(-1, keepdim=True).shape) == (2, 1)
assert ln.shape == rms.shape == x.shape == (2, 3)
assert not torch.equal((ln @ torch.ones(3)), (rms @ torch.ones(3)))

raw_path = ART / "inputs/modern-raw.json"
raw = json.loads(raw_path.read_text())
rows = {}
pointers = ["/revision", "/device", "/seed", "/torch_version", "/python_version", "/gpu", "/code_sha256", "/results/runtime", "/results/dataset"]
for name in ("baseline", "rmsnorm"):
    variant = raw["results"]["variants"][name]
    config = variant["model"]["config"]
    model = TinyLM(ModelConfig(**config))
    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.LayerNorm, RMSNorm))]
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == variant["model"]["parameters"]
    assert len(modules) == 5 and config["width"] == 64
    training_keys = ("requested_steps", "steps", "optimizer_updates", "skipped_updates", "effective_tokens", "batch_size", "learning_rate", "step_scale", "warm_step_median_seconds")
    training = {k: variant["training"][k] for k in training_keys}
    assert training["steps"] == training["optimizer_updates"] == training["requested_steps"] == 240
    assert training["skipped_updates"] == 0
    heldout = {}
    for split in ("validation", "test"):
        item = variant["heldout"][split]
        recalculated = item["nll_sum"] / item["effective_tokens"]
        assert math.isclose(recalculated, item["nll"], abs_tol=1e-12, rel_tol=0)
        assert item["records"] == raw["results"]["dataset"][split]["records"]
        assert len(item["samples"]) == 8
        assert all(len(s["generated_ids"]) <= 32 for s in item["samples"])
        heldout[split] = {k: item[k] for k in ("nll", "nll_sum", "effective_tokens", "examples", "records")}
        heldout[split].update(recalculated_nll=recalculated, rounded_nll=round(recalculated, 5), illustrative_samples=8, generated_token_counts=[len(s["generated_ids"]) for s in item["samples"]])
        base = f"/results/variants/{name}/heldout/{split}"
        pointers.extend(base + "/" + k for k in ("nll", "nll_sum", "effective_tokens", "examples", "records"))
        pointers.extend(base + f"/samples/{i}/generated_ids" for i in range(8))
    rows[name] = {"config":config, "parameters":parameters, "normalization_modules":[n for n,m in modules], "normalization_bias_parameters":sum(m.bias.numel() for n,m in modules if getattr(m,"bias",None) is not None), "training":training, "heldout":heldout, "warm_step_milliseconds":1000 * training["warm_step_median_seconds"]}
    base = f"/results/variants/{name}"
    pointers += [base + "/model", base + "/changes"]
    pointers.extend(base + "/training/" + k for k in training_keys)
assert rows["baseline"]["parameters"] - rows["rmsnorm"]["parameters"] == 5 * 64 == 320
assert rows["baseline"]["normalization_bias_parameters"] == 320 and rows["rmsnorm"]["normalization_bias_parameters"] == 0
assert rows["baseline"]["training"]["effective_tokens"] == rows["rmsnorm"]["training"]["effective_tokens"] == 452102
assert rows["baseline"]["heldout"]["validation"]["rounded_nll"] == 2.26331
assert rows["rmsnorm"]["heldout"]["validation"]["rounded_nll"] == 2.25913
assert rows["baseline"]["heldout"]["test"]["rounded_nll"] == 2.29163
assert rows["rmsnorm"]["heldout"]["test"]["rounded_nll"] == 2.28631
assert round(rows["baseline"]["warm_step_milliseconds"], 3) == 13.062
assert round(rows["rmsnorm"]["warm_step_milliseconds"], 3) == 13.952
assert rows["rmsnorm"]["warm_step_milliseconds"] > rows["baseline"]["warm_step_milliseconds"]
assert raw["seed"] == raw["results"]["seed"] == 42
torch.manual_seed(42)
init1 = torch.randn(4)
torch.manual_seed(42)
init2 = torch.randn(4)
assert torch.equal(init1, init2)
sampler1, sampler2 = random.Random(42), random.Random(42)
assert sampler1.choices(range(8), k=16) == sampler2.choices(range(8), k=16)
provenance = {}
for name in ("tiny_perceptron/model.py", "tiny_perceptron/modern.py", "scripts/course_experiments/common.py"):
    h = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    assert h == raw["code_sha256"][name]
    provenance[name] = h
measurement_code = ART/"sources/architecture-measurement-revision.py"
assert hashlib.sha256(measurement_code.read_bytes()).hexdigest() == raw["code_sha256"]["scripts/course_experiments/architecture.py"]
provenance["scripts/course_experiments/architecture.py@" + raw["revision"]] = hashlib.sha256(measurement_code.read_bytes()).hexdigest()
result = {"environment":{"python":sys.version,"torch":str(torch.__version__),"platform":platform.platform(),"device":"cpu","threads":str(torch.get_num_threads())}, "norm_outputs":{"ln":ln.tolist(),"rms":rms.tolist(),"shifted_ln":shifted_ln.tolist(),"shifted_rms":shifted_rms.tolist(),"means":x.mean(-1).tolist(),"population_variances":x.var(-1,correction=0).tolist(),"epsilon_free_denominators":[math.sqrt(2/3),math.sqrt(14/3)],"eps":eps,"no_parameter_update":True},"historical_measurement":{"device":raw["device"],"gpu":raw["gpu"],"seed":raw["seed"],"torch":raw["torch_version"],"revision":raw["revision"],"rows":rows,"raw_sha256":hashlib.sha256(raw_path.read_bytes()).hexdigest(),"read_json_pointers":pointers,"verified_provenance":provenance,"rerun_training":False,"limitations":"Raw reported medians and token counts audited; underlying latency samples and full original corpus were not remeasured. Constructor and normalization checks are bounded CPU checks, not model training or evaluation."},"assertions":"all passed"}
(ART/"execution/verification-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
