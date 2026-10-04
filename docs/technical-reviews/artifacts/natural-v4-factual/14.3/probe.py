"""Independent bounded CPU factual review of 14.3; no training or downloads."""
import contextlib
import dataclasses
import hashlib
import inspect
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
torch.manual_seed(42)
EVIDENCE = Path(__file__).resolve().parent
RESEARCH = ROOT / "outputs/natural-v4/factual-research/14.3"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def softmax_python(xs):
    maximum = max(xs)
    exps = [math.exp(x - maximum) for x in xs]
    return [x / sum(exps) for x in exps]

source = (RESEARCH / "14.3.original.md").read_text()
code = re.search(r"```python\n(.*?)```", source, re.S).group(1)
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(code, "course/chapters/14.md#14.3 original code", "exec"))
print("ORIGINAL SECTION CODE OUTPUT")
print(printed.getvalue(), end="")

q = torch.tensor([[1., 1.]])
k = torch.tensor([[1., 0.], [0., 100.]])
unit_q, unit_k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
raw, unit = q @ k.T, unit_q @ unit_k.T
assert raw.tolist() == [[1., 100.]]
assert raw.shape == unit.shape == (1, 2)
torch.testing.assert_close(unit, torch.full((1, 2), math.sqrt(.5)), atol=1e-7, rtol=0)
torch.testing.assert_close(unit.softmax(-1), torch.tensor([[.5, .5]]), atol=0, rtol=0)
expected_raw = softmax_python([1., 100.])
torch.testing.assert_close(raw.double().softmax(-1), torch.tensor([expected_raw], dtype=torch.float64), atol=1e-58, rtol=1e-14)
direction_cos = [sum(a*b for a,b in zip([1.,1.], vec)) / (math.sqrt(2)*math.hypot(*vec)) for vec in ([1.,0.],[0.,100.])]
angles = [math.degrees(math.acos(x)) for x in direction_cos]
assert all(abs(x-45) < 1e-12 for x in angles)
base = torch.tensor([1., 0.]).softmax(-1)
scaled = torch.tensor([1., 0.]).mul(4).softmax(-1)
temperature = (torch.tensor([1., 0.]) / .25).softmax(-1)
assert torch.equal(scaled, temperature)
for observed, prediction in [(base, softmax_python([1.,0.])), (scaled, softmax_python([4.,0.]))]:
    torch.testing.assert_close(observed.double(), torch.tensor(prediction,dtype=torch.float64),atol=5e-8,rtol=0)
exercise_k = torch.tensor([[1., 0.], [0., .1]])
exercise_raw = q @ exercise_k.T
exercise_unit = unit_q @ F.normalize(exercise_k,dim=-1).T
torch.testing.assert_close(exercise_unit.softmax(-1),torch.tensor([[.5,.5]]),atol=0,rtol=0)
torch.testing.assert_close(exercise_raw.double().softmax(-1),torch.tensor([softmax_python([1.,.1])],dtype=torch.float64),atol=4e-10,rtol=0)
wrong_axis = F.normalize(q, dim=0) @ F.normalize(k, dim=0).T
assert not torch.equal(unit,wrong_axis)
zero = F.normalize(torch.zeros(1,2),dim=-1)
tiny = F.normalize(torch.tensor([[1e-14,0.]],dtype=torch.float64),dim=-1)
assert torch.equal(zero,torch.zeros(1,2))
assert tiny.tolist() == [[.01,0.]]
negative_dot_before = sum(a*b for a,b in zip([1.,0.],[-1.,0.]))
negative_dot_after = sum(a*b for a,b in zip([1.,0.],[-100.,0.]))

x = torch.tensor([[2.,2.,2.],[1.,2.,3.]])
eps = 1e-5
ln = F.layer_norm(x,(3,),eps=eps)
rms = x / torch.sqrt(x.square().mean(-1,keepdim=True)+eps)
ln_shift = F.layer_norm(x+10,(3,),eps=eps)
rms_shift = (x+10)/torch.sqrt((x+10).square().mean(-1,keepdim=True)+eps)
torch.testing.assert_close(ln,ln_shift,atol=0,rtol=0)
assert not torch.equal(rms,rms_shift)
assert ln.shape == rms.shape == x.shape
assert ln[0].tolist() == [0.,0.,0.]
torch.testing.assert_close(ln[1],torch.tensor([-1.2247,0,1.2247]),atol=5e-5,rtol=0)
torch.testing.assert_close(rms[1],torch.tensor([.4629,.9258,1.3887]),atol=5e-5,rtol=0)

record_path = ROOT / "docs/course-experiments/results/modern.json"
record = json.loads(record_path.read_text())
variants = record["results"]["variants"]
assert list(variants) == ["baseline","rope","rmsnorm","relu2","swiglu","tied"]
config_fields = [f.name for f in dataclasses.fields(ModelConfig)]
assert not any("qk" in name.lower() for name in config_fields)
model_probe = {}
for name, result in variants.items():
    model = TinyLM(ModelConfig(**result["model"]["config"]))
    count = sum(p.numel() for p in model.parameters())
    assert count == result["model"]["parameters"]
    model_probe[name] = {"parameters":count,"changes":result["changes"],"attention_modules":[n for n,_ in model.blocks[0].attention.named_modules()]}
assert model_probe["baseline"]["parameters"]-model_probe["rmsnorm"]["parameters"] == 5*64 == 320
model = TinyLM(ModelConfig(width=8,layers=1,heads=2,max_length=8))
with torch.no_grad():
    logits = model(torch.tensor([[1,2,3,4]]))["logits"]
assert logits.shape == (1,4,264) and bool(torch.isfinite(logits).all())

record_recalculation = {}
for name in ("baseline","rmsnorm"):
    variant = variants[name]
    training = variant["training"]
    assert training["optimizer_updates"] == training["steps"] == training["requested_steps"] == 240
    assert training["skipped_updates"] == 0 and training["effective_tokens"] == 452102
    assert training["batch_size"] == 16
    row={"heldout":{},"warm_update_median_ms":training["warm_step_median_seconds"]*1000,"updates":240,"training_effective_targets":training["effective_tokens"]}
    for split in ("validation","test"):
        h=variant["heldout"][split]
        value = h["nll_sum"] / h["effective_tokens"]
        assert value == h["nll"]
        assert h["effective_tokens"] == {"validation":39256,"test":41914}[split]
        assert h["records"] == {"validation":51,"test":52}[split]
        row["heldout"][split]={"nll_sum":h["nll_sum"],"effective_targets":h["effective_tokens"],"recomputed_nll":value,"rounded_5dp":f"{value:.5f}","records":h["records"],"windows":h["examples"]}
    record_recalculation[name]=row
assert [record_recalculation[name]["heldout"][split]["rounded_5dp"] for name in ("baseline","rmsnorm") for split in ("validation","test")] == ["2.26331","2.29163","2.25913","2.28631"]
assert [f"{record_recalculation[n]['warm_update_median_ms']:.3f}" for n in ("baseline","rmsnorm")] == ["13.062","13.952"]

local_functional = Path(inspect.getsourcefile(F.normalize))
remote_functional = RESEARCH/"authorities/torch2141-functional.py"
assert sha(local_functional)==sha(remote_functional)
result={
 "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"torch_git_version":torch.version.git_version,"platform":platform.platform(),"device":"cpu","cuda_available":str(torch.cuda.is_available()),"threads":"1","probe_seed":"42"},
 "source_sha256":sha(RESEARCH/"14.3.original.md"),
 "original_code_output":printed.getvalue(),
 "geometry":{"q_norm":math.sqrt(2),"key_norms":[1.,100.],"angles_degrees":angles,"cosines":direction_cos,"unit_q":unit_q.tolist(),"unit_k":unit_k.tolist()},
 "attention":{"raw":raw.tolist(),"raw_weights_float32":raw.softmax(-1).tolist(),"raw_weights_float64":raw.double().softmax(-1).tolist(),"independent_python_weights":expected_raw,"unit_scores":unit.tolist(),"unit_weights":unit.softmax(-1).tolist(),"base_weights":base.tolist(),"scale4_weights":scaled.tolist(),"temperature0_25_weights":temperature.tolist(),"wrong_axis_scores":wrong_axis.tolist()},
 "exercise":{"raw_scores":exercise_raw.tolist(),"raw_weights":exercise_raw.softmax(-1).tolist(),"unit_scores":exercise_unit.tolist(),"unit_weights":exercise_unit.softmax(-1).tolist()},
 "domain_limits":{"normalize_zero":zero.tolist(),"normalize_sub_epsilon":tiny.tolist(),"antiparallel_dot_before":negative_dot_before,"antiparallel_dot_after_100x":negative_dot_after,"interpretation":"Dot product is norm product times cosine; score increase with length requires positive cosine. The lesson uses acute 45-degree vectors. Cosine identity requires nonzero vectors with norms at least eps. Positive temperature is reciprocal positive scale; equal logits stay uniform."},
 "prerequisite_14_2":{"layernorm":ln.tolist(),"rmsnorm":rms.tolist(),"rms_no_epsilon_denominator":math.sqrt(14/3),"rms_shifted":rms_shift.tolist(),"epsilon":eps},
 "project":{"config_fields":config_fields,"variants":model_probe,"tiny_forward_shape":list(logits.shape),"modern_record_sha256":sha(record_path),"record_recalculation":record_recalculation,"record_environment":{"device":record["device"],"gpu":record["gpu"],"torch_version":record["torch_version"],"seed":record["seed"],"splits":record["results"]["dataset"],"runtime":record["results"]["runtime"]}},
 "local_official_functional_byte_match":{"local_sha256":sha(local_functional),"retrieved_v2_14_1_sha256":sha(remote_functional),"equal":True},
 "limits":"Only original snippet, transparent math, edge probes, model construction and a tiny forward were executed on CPU. Saved modern records were read and their ratios/unit conversions recomputed; no GPU training, model-quality, timing or large-dataset replication was performed. Per-step latency arrays are not in modern.json, so its saved median itself is not independently reconstructible here."
}
(EVIDENCE/"probe-result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
print("ALL BOUNDED CPU ASSERTIONS PASSED")
