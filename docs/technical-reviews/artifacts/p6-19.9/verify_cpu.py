"""Bounded CPU mechanism verification; no training or held-out evaluation."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer
from tiny_perceptron.modern import RMSNorm, rope

torch.set_num_threads(2)
receipt = {"environment": {"python": sys.version, "torch": torch.__version__,
    "torch_git": torch.version.git_version, "device": "cpu", "threads": "2"}}
section = (OUT / "section.md").read_text()
fence = re.search(r"```python\n(.*?)```", section, re.S)[1]
receipt["fences"] = []
for split in (3, 2):
    code = fence if split == 3 else fence.replace("split = 3", "split = 2")
    path = OUT / f"fence-split{split}.py"
    path.write_text(code)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    completed = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
        cwd=ROOT, env=env, timeout=30)
    (OUT / f"fence-split{split}-stdout.txt").write_text(completed.stdout)
    (OUT / f"fence-split{split}-stderr.txt").write_text(completed.stderr)
    receipt["fences"].append({"split": split, "command": [sys.executable, str(path)],
        "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    assert completed.returncode == 0 and "True" in completed.stdout

torch.manual_seed(42)
config = SelftrainedConfig(vocab_size=32, width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32, top_k=2)
core = LimitedAssistant(config).eval()
before = {k: v.clone() for k, v in core.state_dict().items()}
ids = torch.tensor([[1, 11, 12, 13, 14]])
splits = []
with torch.no_grad():
    full_result = core(ids)
    full = full_result["logits"][:, -1]
    for split in (1, 2, 3, 4):
        prefix = core(ids[:, :split])
        suffix = core(ids[:, split:], cache=prefix["cache"])
        cached = suffix["logits"][:, -1]
        item = {"split": split, "max_abs_difference": (full-cached).abs().max().item(),
            "allclose_1e-4": torch.allclose(full, cached, atol=1e-4, rtol=1e-4),
            "cache_shape": [list(t.shape) for t in suffix["cache"][0]],
            "logit_shape": list(cached.shape), "requires_grad": cached.requires_grad}
        splits.append(item)
        assert item["allclose_1e-4"] and item["cache_shape"] == [[1,1,5,8],[1,1,5,8]]
    cache = core(ids[:, :3])["cache"]
    wrong = core(ids[:, 3:], positions=torch.arange(2), cache=cache)["logits"][:, -1]
receipt["split_invariants"] = splits
receipt["wrong_position_control"] = {"max_abs_difference": (full-wrong).abs().max().item(),
    "allclose_1e-4": torch.allclose(full, wrong, atol=1e-4, rtol=1e-4)}
assert not receipt["wrong_position_control"]["allclose_1e-4"]
receipt["no_learning"] = {"state_dict_unchanged": all(torch.equal(before[k],v) for k,v in core.state_dict().items()),
    "all_parameter_gradients_none": all(p.grad is None for p in core.parameters()), "training": core.training}
assert receipt["no_learning"]["state_dict_unchanged"] and receipt["no_learning"]["all_parameter_gradients_none"]

torch.manual_seed(17)
x = torch.randn(1,2,5,8)
y = rope(x, torch.arange(5))
norm_error = (x.square().sum(-1)-y.square().sum(-1)).abs().max().item()
norm = RMSNorm(4)
norm.weight.data.copy_(torch.tensor([1.,2.,3.,4.]))
a = torch.tensor([[1.,2.,3.,4.]])
expected = a / torch.sqrt(a.square().mean(-1,keepdim=True)+1e-5) * norm.weight
receipt["norms"] = {"rope_squared_norm_error": norm_error,
    "rms_formula_max_error": (norm(a)-expected).abs().max().item(), "rms_output": norm(a).tolist()}
assert norm_error < 2e-6 and torch.allclose(norm(a),expected,atol=1e-6)

chosen = core(ids, attention_mask=torch.tensor([[True,True,True,False,False]]))["routing"][0]
receipt["routing"] = {"valid_tokens": chosen["valid_tokens"], "counts": chosen["counts"].tolist(),
    "chosen": chosen["chosen"].tolist(), "experts": len(core.lm.blocks[0].ffn.experts)}
assert receipt["routing"]["experts"] == 4 and sum(receipt["routing"]["counts"]) == 6
assert (chosen["chosen"][0,:3] >= 0).all() and (chosen["chosen"][0,3:] == -1).all()

prefix_ids = torch.tensor([[1,7,7]+[9]*16+[11,12]])
payloads = [[{"kind":"image", "values":torch.rand(2,1,32,32),
    "coordinates":torch.tensor([[.25,.5],[.75,.5]])},
    {"kind":"audio", "values":torch.rand(13,40)}]]
counts = {"image":0,"audio":0}
q_lengths = []
def count(kind):
    def hook(module, args, result): counts[kind] += 1
    return hook
hooks = [core.vision_encoder.register_forward_hook(count("image")),
    core.audio_encoder.register_forward_hook(count("audio")),
    core.lm.blocks[0].attention.q.register_forward_pre_hook(lambda module,args:q_lengths.append(args[0].shape[1]))]
multimodal = []
for use_cache in (True,False):
    counts.update(image=0,audio=0); q_lengths.clear()
    kwargs = {} if use_cache else {"use_cache":False}
    generated = core.generate(prefix_ids, modalities=payloads, max_new_tokens=3, eos_id=-1, **kwargs)
    multimodal.append({"default_cache" if use_cache else "cache_disabled":True,
        "encoder_calls":dict(counts),"query_input_lengths":q_lengths.copy(),"generated_ids":generated.tolist()})
for hook in hooks:hook.remove()
receipt["multimodal_generation"] = multimodal
assert multimodal[0]["encoder_calls"] == {"image":1,"audio":1}
assert multimodal[0]["query_input_lengths"] == [21,1,1]
assert multimodal[1]["encoder_calls"] == {"image":3,"audio":3}
assert multimodal[1]["query_input_lengths"] == [21,22,23]
assert multimodal[0]["generated_ids"] == multimodal[1]["generated_ids"]

tokenizer = CharacterTokenizer.build(["甲乙丙丁戊己"])
encoder = RecordEncoder(tokenizer, ROOT, context=512)
records = [{"id":"short","task":"text","messages":[{"role":"user","content":"甲"},{"role":"assistant","content":"乙"}]},
    {"id":"long","task":"text","messages":[{"role":"user","content":"甲乙丙丁"},{"role":"assistant","content":"戊己"}]}]
rows = [encoder.encode(r) for r in records]
batch = encoder.batch(records)
receipt["batching"] = {"shape":list(batch["input_ids"].shape), "row_lengths":[len(r["input_ids"]) for r in rows],
    "input_ids":batch["input_ids"].tolist(),"mask":batch["attention_mask"].tolist(),
    "padding_labels":batch["labels"][~batch["attention_mask"]].tolist(),"keys":list(batch)}
assert len(batch["input_ids"]) == len(records) and "segments" not in batch
for i,row in enumerate(rows):
    n=len(row["input_ids"])
    assert batch["input_ids"][i,:n].tolist() == row["input_ids"]
    assert (batch["input_ids"][i,n:] == tokenizer.pad_id).all()
assert (batch["labels"][~batch["attention_mask"]] == -100).all()

raw = json.loads((ROOT/"docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json").read_text())
final_config = raw["config"]
final_core = LimitedAssistant(SelftrainedConfig(**final_config))
att = final_core.lm.blocks[0].attention
receipt["delivered_config"] = final_config
receipt["delivered_structure"] = {"blocks":len(final_core.lm.blocks),"experts_per_block":[len(b.ffn.experts) for b in final_core.lm.blocks],
    "top_k_per_block":[b.ffn.top_k for b in final_core.lm.blocks],"norms":[type(b.norm1).__name__ for b in final_core.lm.blocks],
    "tied_identity":final_core.lm.embedding.weight is final_core.lm.output.weight,
    "attention_backend":att.backend,"rotary":att.rotary,"q_weight_shape":list(att.q.weight.shape),
    "k_weight_shape":list(att.k.weight.shape),"v_weight_shape":list(att.v.weight.shape),
    "kv_parameter_count":att.k.weight.numel()+att.v.weight.numel(),"mha_kv_parameter_count":2*final_config["width"]**2,
    "kv_cache_elements_per_token_per_layer":2*final_config["kv_heads"]*att.head_dim,
    "mha_cache_elements_per_token_per_layer":2*final_config["heads"]*att.head_dim}
assert receipt["delivered_structure"]["tied_identity"]
assert receipt["delivered_structure"]["kv_parameter_count"]*2 == receipt["delivered_structure"]["mha_kv_parameter_count"]
receipt["all_assertions_passed"] = True
(OUT/"cpu-results.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(receipt,ensure_ascii=False,indent=2))
