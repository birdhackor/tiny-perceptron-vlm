"""Independent glossary CPU checks; no training run and no network calls."""

import hashlib
import inspect
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.data import ByteTokenizer, SPECIALS, pad_batch, render_chat, shifted
from tiny_perceptron.simple import ContextMLP
from tiny_perceptron.selftrained.tools import ToolCallError, execute_tool_call, parse_tool_call
from tiny_perceptron.model import ModelConfig, TinyLM, generate
from tiny_perceptron.modern import MoEFFN


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
snapshot = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix04/sources/glossary.md"
torch.set_num_threads(1)
torch.manual_seed(1)
result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu"},
    "source_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "checks": {},
}
c = result["checks"]
c["cat_id_example"] = {"text": "貓看狗", "vocabulary": ["狗", "看", "貓"], "ids": [["狗", "看", "貓"].index(x) for x in "貓看狗"], "embedding_row_2": [0.2, -0.1]}
c["utf8"] = {x: {"codepoints": [hex(ord(z)) for z in x], "bytes": list(x.encode("utf-8")), "byte_count": len(x.encode("utf-8"))} for x in ["貓", "👍🏽", "a", "𠀀"]}
c["byte_range"] = {"bits": 8, "distinct_values": 2**8, "minimum": 0, "maximum": 2**8-1}
tok = ByteTokenizer()
c["byte_tokenizer"] = {"specials": list(SPECIALS), "special_ids": [tok.pad_id, tok.bos_id, tok.eos_id, tok.user_id, tok.assistant_id, tok.image_id, tok.audio_id, tok.system_id], "cat_ids": tok.encode("貓"), "roundtrip": tok.decode(tok.encode("貓")), "control_spelling_stays_content": all(x >= 8 for x in tok.encode("<assistant>"))}
logits = torch.tensor([2., 1., 0.], dtype=torch.float64)
c["softmax_example"] = {"logits": logits.tolist(), "probabilities": F.softmax(logits, dim=-1).tolist(), "negative_log_p_third": -F.log_softmax(logits, dim=-1)[2].item()}
ce = F.cross_entropy(logits[None], torch.tensor([2]), reduction="none")
c["crossentropy_reductions"] = {"unbatched": F.cross_entropy(logits, torch.tensor(2)).item(), "none": ce.tolist(), "sum": F.cross_entropy(logits[None], torch.tensor([2]), reduction="sum").item(), "mean": F.cross_entropy(logits[None], torch.tensor([2]), reduction="mean").item()}
x = torch.tensor([[[1., 2., 3.], [-1., 0., 2.]], [[0., 1., 2.], [3., 2., 1.]]])
linear = nn.Linear(3, 2)
linear.weight.data.copy_(torch.tensor([[1., 0., -1.], [2., 1., 0.]]))
linear.bias.data.copy_(torch.tensor([0.5, -0.5]))
c["linear_last_axis"] = {"input_shape": list(x.shape), "output_shape": list(linear(x).shape), "first_input": x[0, 0].tolist(), "first_output": linear(x)[0, 0].tolist(), "bias_false": nn.Linear(3, 2, bias=False).bias is None}
c["relu"] = F.relu(torch.tensor([-2., 0., 3.])).tolist()
c["l2"] = {"nonzero": F.normalize(torch.tensor([[3., 4.]]), dim=-1).tolist(), "zero": F.normalize(torch.zeros(1, 2), dim=-1).tolist(), "tiny": F.normalize(torch.tensor([[1e-14, 0.]]), dim=-1).tolist()}
c["layernorm_constant_input"] = nn.LayerNorm(2)(torch.ones(1, 2)).tolist()
c["all_negative_infinity_softmax_is_nan"] = torch.isnan(F.softmax(torch.tensor([-torch.inf, -torch.inf]), dim=-1)).tolist()
inputs, targets = render_chat([{"role": "user", "content": "問"}, {"role": "assistant", "content": "答"}], tok)
second = (inputs[:-1], targets[:-1])
batch_x, batch_y, valid = pad_batch([(inputs, targets), second])
c["chat_masks"] = {"inputs": inputs.tolist(), "targets": targets.tolist(), "first_input_is_bos": inputs[0].item() == tok.bos_id, "last_target_is_eos": targets[-1].item() == tok.eos_id, "supervised_ids": targets[targets != -100].tolist(), "expected_answer_plus_eos": tok.encode("答") + [tok.eos_id], "padding_id": batch_x[1, -1].item(), "padding_target": batch_y[1, -1].item(), "padding_valid": valid[1, -1].item()}
a, b = shifted([0, 1, 2, 3])
c["shift_T_plus_1"] = {"original": [0, 1, 2, 3], "input": a.tolist(), "target": b.tolist(), "T": len(a)}
p = nn.Parameter(torch.tensor(2.))
optimizer = torch.optim.SGD([p], lr=0.1)
loss = p**2
loss.backward()
c["gradient_sgd"] = {"old": p.item(), "gradient": p.grad.item(), "lr": 0.1}
optimizer.step()
c["gradient_sgd"]["new"] = p.item()
optimizer.zero_grad()
c["gradient_sgd"]["zero_grad_is_none"] = p.grad is None
ap = nn.Parameter(torch.tensor([1., 2.]))
adam = torch.optim.Adam([ap], lr=0.01)
ap.grad = torch.tensor([1., 4.])
adam.step()
c["adam_state"] = {"state_keys": sorted(adam.state[ap]), "m": adam.state[ap]["exp_avg"].tolist(), "v": adam.state[ap]["exp_avg_sq"].tolist(), "step": adam.state[ap]["step"].item(), "param_groups_have_lr": "lr" in adam.state_dict()["param_groups"][0]}
c["context_mlp"] = {"context": 3, "input_shape": [2, 3], "output_shape": list(ContextMLP(4, context=3, width=4)(torch.tensor([[0, 1, 2], [1, 2, 3]])).shape)}
c["ffn_4_16"] = {"feature_width": 4, "hidden_width": 16, "output_shape": list(nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 4))(torch.zeros(2, 3, 4)).shape)}
tool_text = '{"tool":"calculator","operation":"add","a":2,"b":3}'
bad_tool_text = '{"tool":"calculator","operation":"add","a":true,"b":3}'
try:
    parse_tool_call(bad_tool_text)
    invalid_rejected = False
except ToolCallError:
    invalid_rejected = True
c["tool_boundary"] = {"valid_input": tool_text, "parsed": parse_tool_call(tool_text).to_dict(), "result": execute_tool_call(tool_text), "invalid_boolean_rejected": invalid_rejected}
enumerated = " ".join(map(str, range(16)))
clean = enumerated.strip()
parsed = int(clean) if re.fullmatch(r"-?[0-9]+", clean) else None
c["reward_rules"] = {"answer": enumerated, "truth_range": [0, 15], "loose_rewards": [float(str(truth) in enumerated.split()) for truth in range(16)], "strict_parsed": parsed, "strict_rewards": [float(parsed == truth) for truth in range(16)]}
moe = TinyLM(ModelConfig(vocab_size=16, width=8, heads=2, max_length=8, experts=4, top_k=2))
expert_parameters = sum(p.numel() for block in moe.blocks for p in block.ffn.experts.parameters())
total_parameters = sum(p.numel() for p in moe.parameters())
router_parameters = sum(p.numel() for block in moe.blocks for p in block.ffn.router.parameters())
c["moe_parameter_proxy"] = {"total": total_parameters, "all_experts": expert_parameters, "router": router_parameters, "proxy": total_parameters-expert_parameters+expert_parameters*2//4, "embedding_in_nonexpert": moe.embedding.weight.numel(), "position_in_nonexpert": moe.position.weight.numel()}
small = TinyLM(ModelConfig(vocab_size=16, width=4, heads=1, max_length=8, tied=True))
for parameter in small.parameters():
    parameter.data.zero_()
generated = generate(small, torch.tensor([[1, 3]]), max_new_tokens=3, temperature=0.0, eos_id=15)
eos_generated = generate(small, torch.tensor([[1, 3]]), max_new_tokens=3, temperature=0.0, eos_id=0)
c["generation_limit_and_tying"] = {"prompt_length": 2, "max_new_tokens": 3, "returned_shape": list(generated.shape), "same_weight_object": small.embedding.weight is small.output.weight, "eos_id_for_stop_probe": 0, "eos_stop_returned_shape": list(eos_generated.shape)}
repo_files = ["tiny_perceptron/data.py", "tiny_perceptron/simple.py", "tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/attention.py", "tiny_perceptron/selftrained/tools.py", "tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/train.py", "scripts/course_experiments/architecture.py", "scripts/course_experiments/compression.py", "docs/selftrained/v2-manifest.json", "docs/natural-assistant/v4/public-release.json"]
c["repository_source_identity"] = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in repo_files}
c["temperature_rank_reversal"] = {str(T): [F.softmax(torch.tensor(z, dtype=torch.float64) / T, dim=-1).max().item() for z in [[1., 0., 0.], [0.5, 0., -10.]]] for T in [1., 0.1]}
c["choice_concentration"] = {"n": 3, "pmax": 0.8, "formula": (0.8-1/3)/(1-1/3)}
c["snr"] = {"signal": [1., -1.], "reconstruction": [0.9, -0.9], "signal_energy": 2., "error_energy": 0.02, "db": 10*math.log10(2/0.02)}
c["kl_entropy"] = {"p": [0.25, 0.75], "kl_self": sum(p*(math.log(p)-math.log(p)) for p in [0.25, 0.75]), "entropy": -sum(p*math.log(p) for p in [0.25, 0.75])}
c["rounding_frequency"] = {"hertz": 440, "cycles_in_one_second": 440}
body = snapshot.read_text()
links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body)
missing = []
for label, dest in links:
    path, _, fragment = dest.partition("#")
    target = ROOT / "course" / path
    if not target.exists():
        missing.append({"label": label, "destination": dest, "reason": "missing path"})
    elif fragment and not re.search(r"^##\s+" + re.escape(fragment) + r"\s", target.read_text(), re.M):
        missing.append({"label": label, "destination": dest, "reason": "missing numbered heading"})
index = body.split("### 英文與縮寫索引", 1)[1]
indexed = []
sorting = []
for line in index.splitlines():
    if not line.startswith("**"):
        continue
    terms = [m.group(1) for m in re.finditer(r"\[([^\]]+)\]\([^)]+\)", line)]
    if line.startswith("**D**"):
        terms = ["decode", "deep supervision", "Dense", "derivative", "distillation", "DPO"]
    keys = [re.sub("[^a-z0-9]", "", t.lower()) for t in terms]
    if keys != sorted(keys):
        sorting.append(terms)
    indexed += terms
c["link_index_audit"] = {"links_checked": len(links), "missing_targets": missing, "index_entries": len(indexed), "duplicate_terms": sorted({t for t in indexed if indexed.count(t) > 1}), "sort_failures": sorting, "unlisted_in_tables_example": "activation checkpointing", "above_tables_contains_activation_checkpointing": "activation checkpointing" in body.split("### 英文與縮寫索引")[0]}
docs = []
for name, obj in [("Linear", nn.Linear), ("ReLU", nn.ReLU), ("GELU", nn.GELU), ("Embedding", nn.Embedding), ("LayerNorm", nn.LayerNorm), ("normalize", F.normalize), ("softmax", F.softmax), ("cross_entropy", F.cross_entropy), ("Tensor.backward", torch.Tensor.backward), ("Optimizer.zero_grad", inspect.unwrap(torch.optim.Optimizer.zero_grad)), ("Optimizer.state_dict", inspect.unwrap(torch.optim.Optimizer.state_dict))]:
    file = Path(inspect.getfile(obj))
    _, line = inspect.getsourcelines(obj)
    docs.append({"name": name, "installed_path": str(file), "file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "source_start_line": line, "excerpt": (inspect.getdoc(obj) or "")[:1800]})
result["inspected_official_torch_docstrings"] = docs
(OUT / "verification-output.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
