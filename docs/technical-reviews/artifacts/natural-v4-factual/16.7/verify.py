"""Fresh bounded CPU verification and fixed-record audit; never runs GPU training."""
import ast
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from scripts.prepare_data import generate_records
from scripts.course_experiments.common import split_records, records_sha256, text_examples
from tiny_perceptron.data import IGNORE, pad_batch
from tiny_perceptron.model import TinyLM, ModelConfig, loss_sum

torch.set_num_threads(1)
environment = {"python": platform.python_version(), "torch": str(torch.__version__),
               "device": "cpu", "cuda_available": str(torch.cuda.is_available()),
               "platform": platform.platform(), "threads": str(torch.get_num_threads())}
assert torch.__version__ == "2.14.1+cpu" and not torch.cuda.is_available()
out = {"environment": environment, "scope": "CPU probes and audit of fixed L4 records; no GPU replication"}
out["finfo"] = {}
for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
    f = torch.finfo(dtype)
    out["finfo"][name] = {"bits": f.bits, "bytes": torch.tensor(0, dtype=dtype).element_size(),
                          "max": f.max, "eps": f.eps, "tiny": f.tiny}
assert out["finfo"]["fp16"]["max"] == (2 - 2**-10) * 2**15 == 65504
assert out["finfo"]["bf16"]["max"] == (2 - 2**-7) * 2**127
out["matmul"] = {}
for name, values, scale in (("original", [[1.001,2.002],[3.003,4.004]], 1),
                            ("times10", [[1.001,2.002],[3.003,4.004]], 10),
                            ("integers", [[1.0,2.0],[3.0,4.0]], 1)):
    a = torch.tensor(values) * scale
    b = torch.tensor([[0.5,1.0],[1.5,-1.0]])
    a_before, b_before = a.clone(), b.clone()
    full = a @ b
    with torch.autocast("cpu", dtype=torch.bfloat16):
        low = a @ b
    delta = (full-low.float()).abs()
    out["matmul"][name] = {"input_dtype": str(a.dtype), "output_dtype": str(low.dtype),
                           "input": a.tolist(), "bf16_input": a.bfloat16().float().tolist(),
                           "full": full.tolist(), "full_rounded4": full.round(decimals=4).tolist(),
                           "low": low.float().tolist(), "delta": delta.tolist(),
                           "max_delta": delta.max().item(), "rounded_max_delta": round(delta.max().item(),4),
                           "inputs_unchanged": bool(torch.equal(a,a_before) and torch.equal(b,b_before))}
    assert a.dtype == b.dtype == torch.float32 and low.dtype == torch.bfloat16
    assert torch.equal(a, a_before) and torch.equal(b,b_before)
assert out["matmul"]["original"]["low"] == [[3.5,-1.0],[7.5,-1.0]]
assert out["matmul"]["original"]["rounded_max_delta"] == 0.0075
assert out["matmul"]["times10"]["max_delta"] > out["matmul"]["original"]["max_delta"]
assert out["matmul"]["integers"]["max_delta"] == 0

out["scaling"] = []
for bad in (None, "inf", "nan"):
    p = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([p],lr=0.1)
    scaler = torch.amp.GradScaler("cpu",init_scale=1024)
    loss = (p-3).square()
    scaler.scale(loss).backward()
    scaled_grad = p.grad.item()
    if bad: p.grad.fill_(float(bad))
    scaler.unscale_(opt)
    unscaled_grad = str(p.grad.item()) if bad else p.grad.item()
    scaler.step(opt)
    scaler.update()
    out["scaling"].append({"injected_gradient": bad, "loss":loss.item(),
                           "scaled_gradient":scaled_grad, "unscaled_gradient":unscaled_grad,
                           "parameter_after":p.item(),"scale_after":scaler.get_scale()})
    assert scaled_grad == -4096
    assert p.item() == 1 if bad else abs(p.item()-1.4)<1e-6
    assert scaler.get_scale() == (512 if bad else 1024)
tiny = torch.tensor(2**-30)
out["special_values"] = {"unscaled_fp16":tiny.half().float().item(),
                         "scaled_fp16_back_to_fp32":(tiny*1024).half().float().div(1024).item(),
                         "fp16_overflow_is_inf":bool(torch.isinf(torch.tensor(100000.).half())),
                         "zero_div_zero_is_nan":bool(torch.isnan(torch.tensor(0.)/torch.tensor(0.)))}
assert out["special_values"]["unscaled_fp16"] == 0
assert out["special_values"]["scaled_fp16_back_to_fp32"] == 2**-30

record_path = Path("docs/course-experiments/results/precision.json")
record = json.loads(record_path.read_text())
result = record["results"]
parts = split_records(generate_records("attributes-sft"),seed=42)
examples = {split:text_examples(rows,"sft",128) for split,rows in parts.items()}
out["dataset"] = {}
families = {split:{r["family"] for r in rows} for split,rows in parts.items()}
assert not (families["train"] & families["validation"] or families["train"] & families["test"] or families["validation"] & families["test"])
for split, rows in parts.items():
    out["dataset"][split] = {"records":len(rows), "families":len(families[split]),
                             "effective_targets":sum(int((y!=IGNORE).sum()) for x,y in examples[split]),
                             "sha256":records_sha256(rows)}
    assert len(rows) == result["dataset"][split]["records"]
    assert records_sha256(rows) == result["dataset"][split]["sha256"]
sampler = random.Random(42)
budget = sum(sum(int((y!=IGNORE).sum()) for x,y in sampler.choices(examples["train"],k=16)) for _ in range(200))
assert budget == 22493
out["sampled_training_budget"] = {"seed":42,"attempts":200,"batch_records":16,"effective_targets":budget}

torch.manual_seed(42)
model = TinyLM(ModelConfig(width=64,layers=2,heads=1,max_length=128))
optimizer = torch.optim.AdamW(model.parameters(),lr=0.003)
x,y,valid = pad_batch(examples["validation"])
with torch.autocast("cpu",dtype=torch.bfloat16):
    logits = model(x,valid=valid)["logits"]
    total,count = loss_sum(logits.float(),y)
    loss = total/count
loss.backward()
assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
optimizer.step()
out["one_cpu_bf16_update"] = {"parameters":sum(p.numel() for p in model.parameters()),
                              "parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
                              "weights_dtype":sorted({str(p.dtype) for p in model.parameters()}),
                              "logits_dtype":str(logits.dtype), "loss_dtype":str(loss.dtype),
                              "adam_state_dtypes":sorted({str(v.dtype) for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)}),
                              "loss":loss.item(), "effective_targets":int(count),
                              "all_parameters_finite":all(bool(torch.isfinite(p).all()) for p in model.parameters())}
assert out["one_cpu_bf16_update"]["parameters"] == 141568
assert out["one_cpu_bf16_update"]["parameter_bytes"] == 566272
assert out["one_cpu_bf16_update"]["weights_dtype"] == out["one_cpu_bf16_update"]["adam_state_dtypes"] == ["torch.float32"]

out["fixed_l4_audit"] = {"record_sha256":hashlib.sha256(record_path.read_bytes()).hexdigest(),
                         "record_environment":{k:str(record[k]) for k in ("revision","seed","device","torch_version","python_version","gpu")},
                         "variants":{}}
expected_nll = {"fp32":["0.47748","0.36272"],"bf16":["0.83820","0.32709"],"fp16":["1.17649","0.71923"]}
expected_matches = {"fp32":[2,7],"bf16":[1,6],"fp16":[1,4]}
expected_times = {"fp32":["11.673","2.444"],"bf16":["14.414","2.835"],"fp16":["14.280","2.740"]}
expected_memory = {"fp32":["33.086","79.261","46.175"],"bf16":["65.374","75.882","10.509"],"fp16":["65.232","75.743","10.510"]}
for name, v in result["variants"].items():
    tr=v["training"]; inf=v["inference"]; held=[]
    assert tr["requested_steps"] == tr["steps"] == tr["optimizer_updates"] == 200 and tr["skipped_updates"]==0
    assert tr["effective_tokens"] == budget and tr["batch_size"]==16 and tr["learning_rate"]==0.003
    assert tr["all_parameters_finite"] and v["logits_finite"] and all(math.isfinite(h["loss"]) and h["gradients_finite"] for h in tr["history"])
    assert tr["scaler_enabled"] == (name=="fp16")
    for i,split in enumerate(("validation","test")):
        h=v["heldout"][split]; samples=h["samples"]
        matches=sum(s["generated_ids"][:-1] == [b+8 for b in s["expected"].encode("utf-8")] for s in samples)
        targets=sum(len(s["expected"].encode("utf-8"))+1 for s in samples)
        assert all(s["eos"] and s["generated_ids"][-1]==2 and len(s["generated_ids"])<=32 for s in samples)
        assert all(s["exact"] == (s["generated_ids"][:-1] == [b+8 for b in s["expected"].encode("utf-8")]) for s in samples)
        assert targets == h["effective_tokens"] == out["dataset"][split]["effective_targets"]
        assert matches == h["matches"] == expected_matches[name][i]
        assert len(samples) == h["records"] == h["examples"] == len(parts[split])
        assert abs(h["nll"]-h["nll_sum"]/targets)<1e-12
        assert f'{h["nll"]:.5f}' == expected_nll[name][i]
        held.append({"split":split,"nll_sum":h["nll_sum"],"targets":targets,"recomputed_nll":h["nll_sum"]/targets,
                     "matches":matches,"records":len(samples),"eos":sum(s["eos"] for s in samples),
                     "first_expected":samples[0]["expected"],"first_generated":samples[0]["generated"],
                     "first_generated_ids":samples[0]["generated_ids"]})
    med=statistics.median(inf["samples_seconds"])
    assert med == inf["median_seconds"] and inf["warmup_calls"]==3 and inf["measured_calls"]==len(inf["samples_seconds"])==9 and inf["synchronized"]
    times=[f'{tr["warm_step_median_seconds"]*1000:.3f}',f'{med*1000:.3f}']
    assert times == expected_times[name]
    memory=[tr[k] for k in ("memory_allocated_before_bytes","peak_memory_allocated_bytes","peak_additional_allocated_bytes")]
    assert memory[1]-memory[0]==memory[2]
    assert [f'{b/2**20:.3f}' for b in memory] == expected_memory[name]
    out["fixed_l4_audit"]["variants"][name] = {"heldout":held,"timings_ms":times,"forward_samples_seconds":inf["samples_seconds"],
                                               "memory_bytes":memory,"memory_MiB":[b/2**20 for b in memory],
                                               "updates":tr["optimizer_updates"],"skipped":tr["skipped_updates"],
                                               "observed_logits_dtype":v["observed_logits_dtype"],"weight_dtype":v["weights_dtype"],
                                               "scaler_enabled":tr["scaler_enabled"],"scaler_history":tr["history"]}
out["timing_limits"] = "Forward medians recomputed from nine saved samples; training-step raw latencies not exported, so training median only audited against code and saved aggregate. Single fixed L4 run; no cross-run stability or GPU replication."
old=Path("outputs/natural-v4/factual-research/16.7/architecture-recorded.py").read_text()
current=Path("scripts/course_experiments/architecture.py").read_text()
def functions(s):
    return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
old_f,current_f=functions(old),functions(current)
out["historical_code_comparison"]={}
for name in ("_sync","_amp","_sft_dataset","_description","_train","_benchmark","_heldout","run_precision"):
    assert old_f[name] == current_f[name]
    out["historical_code_comparison"][name] = {"exact_source_equal":True,"sha256":hashlib.sha256(old_f[name].encode()).hexdigest()}
print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
