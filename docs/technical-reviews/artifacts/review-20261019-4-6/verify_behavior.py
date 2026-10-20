"""核對凍結範例、候選軸與函式庫的可選輸入；只做小型 CPU 計算。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from torch import nn
from tiny_perceptron.model import ModelConfig, TinyLM, Block, generate
from tiny_perceptron.modern import RMSNorm

torch.set_num_threads(1)
torch.set_default_device("cpu")
source = Path("docs/course-review-20261019/freeze-01/sources/4.6.md")
raw = source.read_bytes()
fences = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    for code in fences:
        exec(compile(code, str(source), "exec"), namespace)
model, ids, logits = (namespace[key] for key in ("model", "ids", "logits"))
results = {}


def record(name, condition, **details):
    assert bool(condition), name
    results[name] = {"verified": True, **details}


record("original_execution", namespace["same"] and tuple(logits.shape) == (1, 3, 20),
       stdout=captured.getvalue(), shape=list(logits.shape), seed=42)
record("embedding_tables", list(model.embedding.weight.shape) == [20, 8]
       and list(model.position.weight.shape) == [128, 8],
       token_shape=list(model.embedding.weight.shape), position_shape=list(model.position.weight.shape))
record("output_linear", model.output.bias is None
       and torch.allclose(logits, model.final_norm(model.blocks[0](model.embedding(ids)
                           + model.position(torch.arange(3)))[0]) @ model.output.weight.T),
       weight_shape=list(model.output.weight.shape), bias=str(model.output.bias))
record("arange", torch.equal(torch.arange(ids.shape[1]), torch.tensor([0, 1, 2])),
       positions=torch.arange(ids.shape[1]).tolist())
record("raw_logits", not torch.allclose(logits.sum(-1), torch.ones(1, 3)),
       minimum=float(logits.min().detach()), maximum=float(logits.max().detach()),
       row_sums=logits.detach().sum(-1).tolist())
record("argmax", logits.argmax(-1).tolist() == [[10, 17, 7]],
       ids=logits.argmax(-1).tolist(), shape=list(logits.argmax(-1).shape))
scores = namespace["scores_for_questions"]
answers = namespace["answers_for_questions"]
individual = [math.log(sum(math.exp(float(x)) for x in row)) - float(row[target])
              for row, target in zip(scores.detach(), answers)]
mean = sum(individual) / len(individual)
ce = float(torch.nn.functional.cross_entropy(scores, answers).detach())
record("loss", abs(mean-ce) < 1e-6 and abs(ce-3.336) < 0.0005,
       score_shape=list(scores.shape), answer_shape=list(answers.shape),
       answers=answers.tolist(), individual_costs=individual, mean_manual=mean,
       cross_entropy=ce, denominator=3, dtype=str(logits.dtype))
record("reshape_order", torch.equal(scores, torch.stack([logits[0, i] for i in range(3)]))
       and answers.tolist() == [2, 3, 4])
library = namespace["library"]
result = library(ids)
record("return_keys", list(result) == ["logits", "cache", "auxiliary"], keys=list(result))
record("dense_auxiliary", result["auxiliary"].item() == 0,
       auxiliary=result["auxiliary"].item(), caches=len(result["cache"]),
       cache_shapes=[[list(t.shape) for t in pair] for pair in result["cache"]])
prefix_errors = []
for position in range(3):
    prefix = library(ids[:, :position+1])["logits"][:, -1]
    prefix_errors.append(float((prefix-result["logits"][:, position]).detach().abs().max()))
record("causal_prefixes", max(prefix_errors) < 1e-6, maximum_errors=prefix_errors,
       allowed_contexts=[[1], [1, 2], [1, 2, 3]], targets=[2, 3, 4])
altered = library(torch.tensor([[1, 19, 18]]))["logits"]
record("future_independence", torch.equal(altered[:, 0], result["logits"][:, 0]))
record("generation", torch.equal(generate(library, ids, max_new_tokens=1, eos_id=-1),
                                  torch.cat((ids, logits[:, -1].argmax(-1, keepdim=True)), 1)),
       generated=generate(library, ids, max_new_tokens=1, eos_id=-1).tolist())
record("untied", model.output.weight is not model.embedding.weight
       and library.output.weight is not library.embedding.weight)
torch.manual_seed(42)
wide_config = ModelConfig(vocab_size=25, width=8)
wide = namespace["TinyLM"](wide_config)
wide_library = TinyLM(wide_config)
wide_library.load_state_dict(wide.state_dict())
wide_logits = wide(ids)["logits"]
record("vocabulary_25", tuple(wide_logits.shape) == (1, 3, 25)
       and torch.allclose(wide_logits, wide_library(ids)["logits"]),
       shape=list(wide_logits.shape), argmax=wide_logits.argmax(-1).tolist())
deep = TinyLM(ModelConfig(vocab_size=20, width=8, layers=2))
record("registered_independent_blocks", isinstance(deep.blocks, nn.ModuleList)
       and deep.blocks[0] is not deep.blocks[1]
       and deep.blocks[0].attention.q.weight is not deep.blocks[1].attention.q.weight
       and "blocks.1.attention.q.weight" in deep.state_dict(), blocks=len(deep.blocks))
record("description", deep.description() == {"config": asdict(deep.config),
       "parameters": sum(p.numel() for p in deep.parameters())} and not hasattr(model, "description"),
       description=deep.description())
default = TinyLM()
record("default_config", default.config == ModelConfig(), config=asdict(default.config))
rotary = TinyLM(ModelConfig(vocab_size=20, width=8, rotary=True))
record("rotary", rotary.position is None and rotary.blocks[0].attention.rotary,
       output_shape=list(rotary(ids)["logits"].shape))
rms = TinyLM(ModelConfig(vocab_size=20, width=8, norm="rms"))
record("rms", isinstance(rms.final_norm, RMSNorm)
       and all(isinstance(norm, RMSNorm) for b in rms.blocks for norm in (b.norm1, b.norm2)),
       output_shape=list(rms(ids)["logits"].shape))
tied = TinyLM(ModelConfig(vocab_size=20, width=8, tied=True))
record("tied", tied.output.weight is tied.embedding.weight,
       output_shape=list(tied(ids)["logits"].shape))
prefix = library(ids[:, :2])
cached = library(ids[:, 2:], cache=prefix["cache"])
record("cache_continuation", torch.allclose(cached["logits"], result["logits"][:, 2:], atol=1e-6),
       cached_shape=list(cached["logits"].shape), cache_length=cached["cache"][0][0].shape[2])
embeddings = library.embedding(ids)
record("direct_embeddings", torch.equal(library(embeddings=embeddings)["logits"], result["logits"]))
valid = torch.tensor([[False, True, True, True]])
positions = torch.tensor([[0, 0, 1, 2]])
padded_ids = torch.tensor([[0, 1, 2, 3]])
padded = library(padded_ids, valid=valid, positions=positions)["logits"]
record("left_padding_positions", torch.allclose(padded[:, 1:], result["logits"], atol=1e-6),
       positions=positions.tolist(), valid=valid.tolist())
packed_ids = torch.tensor([[1, 2, 3]])
segments = torch.tensor([[0, 0, 1]])
packed_positions = torch.tensor([[0, 1, 0]])
packed = library(packed_ids, positions=packed_positions, segments=segments)["logits"]
alone = library(torch.tensor([[3]]))["logits"]
record("segments", torch.allclose(packed[:, 2:], alone, atol=1e-6), segments=segments.tolist())
moe = TinyLM(ModelConfig(vocab_size=20, width=8, layers=2, experts=2, top_k=1))
aux_values = []
handles = [b.register_forward_hook(lambda module, args, output: aux_values.append(output[2])) for b in moe.blocks]
moe_result = moe(ids)
for handle in handles:
    handle.remove()
record("auxiliary_sum", torch.equal(sum(aux_values), moe_result["auxiliary"]),
       block_costs=[float(x.detach()) for x in aux_values], total=float(moe_result["auxiliary"].detach()))
errors = {}
for name, call, exception in [
    ("id_out_of_range", lambda: library(torch.tensor([[20]])), IndexError),
    ("invalid_norm", lambda: TinyLM(ModelConfig(norm="other")), ValueError),
    ("invalid_layers", lambda: TinyLM(ModelConfig(layers=0)), ValueError),
    ("invalid_max_length", lambda: TinyLM(ModelConfig(max_length=0)), ValueError),
    ("length_exceeded", lambda: library(torch.ones(1, 129, dtype=torch.long)), ValueError),
]:
    try:
        call()
    except exception as error:
        errors[name] = {"type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError(name)
record("configuration_and_bounds", len(errors) == 5, errors=errors)
payload = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-6/verify_behavior.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu"},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "results": results,
    "verified_checks": len(results),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
