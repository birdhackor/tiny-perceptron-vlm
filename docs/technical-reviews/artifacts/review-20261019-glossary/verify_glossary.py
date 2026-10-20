"""離線核對名詞頁的算例、介面及本課定義；不作訓練或效能推論。"""

import ast
import copy
import hashlib
import inspect
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.alignment import LoRALinear, distillation_kl, dpo_loss
from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import SPECIALS, ByteTokenizer, fingerprint, pad_batch, render_chat, shifted
from tiny_perceptron.efficiency import online_attention
from tiny_perceptron.lesson_products.data import observable_key
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver
from tiny_perceptron.model import ModelConfig, TinyLM, generate, loss_sum, masked_loss
from tiny_perceptron.modern import DenseFFN, MoEFFN, rope
from tiny_perceptron.multimodal import MultiModalLM, log_mel, mel_filter_bank, patchify, tone
from tiny_perceptron.posttraining import bandit_advantage, ppo_clipped_objective
from tiny_perceptron.quantization import quantize_symmetric
from tiny_perceptron.selftrained.tools import execute_tool_call, parse_tool_call
from tiny_perceptron.simple import ContextMLP

BASE = Path(__file__).parent
ROOT = Path.cwd()
torch.set_num_threads(1)
torch.manual_seed(20261020)
out = {"environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "dtype": "float32，softmax 算例另用 float64", "seed": "20261020"}}
out["command"] = "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-glossary/verify_glossary.py"
out["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
freeze_path = Path("docs/course-review-20261019/freeze-01/sources/glossary.md")
body = freeze_path.read_text()
out["freeze"] = {"path": str(freeze_path), "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(), "images": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)}

vocab = {"狗": 0, "看": 1, "貓": 2}
ids = [vocab[c] for c in "貓看狗"]
table = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.2, -0.1]], dtype=torch.float64)
out["id_lookup"] = {"ids": ids, "vector": table[2].tolist()}
assert ids == [2, 1, 0] and table[2].tolist() == [0.2, -0.1]
out["bytes"] = {"bits": 8, "unsigned_range": [0, 2**8-1], "utf8_lengths": {s: len(s.encode()) for s in ["A", "é", "貓", "😀"]}, "graphemes": {s: {"codepoints": [hex(ord(c)) for c in s], "len": len(s), "utf8_bytes": len(s.encode())} for s in ["é", "e\u0301", "🇹🇼", "👨‍👩‍👧‍👦"]}}
tok = ByteTokenizer()
out["tokenizer"] = {"specials": dict(zip(SPECIALS, range(8))), "cat_tokens": tok.encode("貓"), "cat_roundtrip": tok.decode(tok.encode("貓")), "literal_special_is_bytes": tok.encode("<eos>")}
assert len(tok.encode("貓")) == 3 and tok.decode(tok.encode("貓")) == "貓"
x, y = render_chat([{"role": "user", "content": "問"}, {"role": "assistant", "content": "答"}])
xx, yy, valid = pad_batch([(x, y), (x[:-1], y[:-1])])
out["chat_mask"] = {"inputs": x.tolist(), "labels": y.tolist(), "pad_inputs": xx.tolist(), "pad_labels": yy.tolist(), "valid": valid.tolist(), "bos_target_count": int((y == tok.bos_id).sum()), "eos_target_count": int((y == tok.eos_id).sum())}
assert y[-1] == tok.eos_id and not (y == tok.bos_id).any()
assert xx[-1, -1] == 0 and yy[-1, -1] == -100 and not valid[-1, -1]
sx, sy = shifted([2, 1, 0, 2])
out["shift"] = {"input": sx.tolist(), "target": sy.tolist(), "source": [2, 1, 0, 2]}
out["shape_linear_ffn"] = {"shape": list(torch.zeros(2, 3).shape), "linear_shape": list(nn.Linear(4, 5)(torch.ones(2, 3, 4)).shape), "bias_default": nn.Linear(4, 5).bias is not None, "no_bias": nn.Linear(4, 5, bias=False).bias is None, "ffn_up": list(DenseFFN(4).up.weight.shape), "ffn_down": list(DenseFFN(4).down.weight.shape), "activation": DenseFFN(4).activation, "mlp_logits": list(ContextMLP(8)(torch.tensor([[1, 2, 3]])).shape)}
out["relu"] = torch.relu(torch.tensor([-1., 0., 2.])).tolist()
p = torch.tensor([2., 1., 0.], dtype=torch.float64).softmax(0)
out["softmax"] = {"probabilities": p.tolist(), "rounded": [round(float(v), 3) for v in p], "sum": float(p.sum()), "target_last_nll": float(-p[2].log())}
assert out["softmax"]["rounded"] == [0.665, 0.245, 0.09]
logits = torch.tensor([[[2., 1., 0.], [0., 2., 1.], [1., 0., 2.]]])
labels = torch.tensor([[2, -100, 2]])
total, count = loss_sum(logits, labels)
out["nll"] = {"sum": float(total), "count": int(count), "mean": float(masked_loss(logits, labels)), "direct_sum": float(-logits.log_softmax(-1)[0, 0, 2]-logits.log_softmax(-1)[0, 2, 2])}
assert torch.allclose(total/count, masked_loss(logits, labels)) and count == 2
out["nan"] = torch.softmax(torch.full((3,), float("-inf")), 0).isnan().tolist()
assert all(out["nan"])
w = nn.Parameter(torch.tensor([1., 2.]))
optimizer = torch.optim.Adam([w], lr=.01)
(w.square().sum()).backward()
optimizer.step()
state = optimizer.state[w]
out["adam"] = {"m": state["exp_avg"].tolist(), "v": state["exp_avg_sq"].tolist(), "step": float(state["step"]), "optimizer_state_keys": list(optimizer.state_dict()), "param_group_has_lr": "lr" in optimizer.state_dict()["param_groups"][0]}
saved_parameter = w.detach().clone()
saved_state = copy.deepcopy(optimizer.state_dict())
def next_update(parameter, opt):
    opt.zero_grad()
    ((parameter-torch.tensor([2., -1.]))**2).sum().backward()
    opt.step()
next_update(w, optimizer)
restored = nn.Parameter(saved_parameter.clone())
restored_opt = torch.optim.Adam([restored], lr=.01)
restored_opt.load_state_dict(saved_state)
next_update(restored, restored_opt)
assert torch.equal(w, restored)
out["checkpoint"] = {"same_next_update_with_optimizer": torch.equal(w, restored), "scope": "同梯度、同資料、同設定的確定性 CPU Adam；未證明隨機流程只靠參數與 optimizer 就能完全續訓"}
out["causal"] = attention_mask(torch.arange(3), torch.arange(3))[0, 0].tolist()
assert out["causal"] == [[True, False, False], [True, True, False], [True, True, True]]
model = TinyLM(ModelConfig(vocab_size=16, width=8, heads=2, layers=2, max_length=16, tied=True))
model.eval()
with torch.no_grad():
    cached = model(torch.tensor([[1, 3]]))
    next_cached = model(torch.tensor([[4]]), cache=cached["cache"])
    all_logits = model(torch.tensor([[1, 3, 4]]))["logits"][:, -1]
out["kv_cache"] = {"prefill_shapes": [[list(t.shape) for t in pair] for pair in cached["cache"]], "decode_shapes": [[list(t.shape) for t in pair] for pair in next_cached["cache"]], "max_error": float((all_logits-next_cached["logits"][:, -1]).abs().max()), "tied": model.output.weight is model.embedding.weight}
assert torch.allclose(all_logits, next_cached["logits"][:, -1], atol=2e-6)
class Fixed(nn.Module):
    def __init__(self, next_id):
        super().__init__()
        self.config = ModelConfig(vocab_size=8, max_length=20)
        self.next_id = next_id
    def forward(self, ids, cache=None):
        score = torch.zeros(ids.shape[0], ids.shape[1], 8)
        score[..., self.next_id] = 10.
        return {"logits": score, "cache": None}
prefix = torch.tensor([[1, 3, 4]])
out["generation"] = {"eos": generate(Fixed(2), prefix, max_new_tokens=5).tolist(), "limit": generate(Fixed(5), prefix, max_new_tokens=2).tolist(), "prefix_length": 3}
assert out["generation"]["eos"] == [[1, 3, 4, 2]] and len(out["generation"]["limit"][0]) == 5
lora = LoRALinear(nn.Linear(4, 3), rank=2)
features = torch.randn(2, 4)
out["lora"] = {"a_shape": list(lora.a.shape), "b_shape": list(lora.b.shape), "base_frozen": all(not p.requires_grad for p in lora.base.parameters()), "initial_equals_base": torch.equal(lora(features), lora.base(features))}
assert out["lora"]["base_frozen"] and out["lora"]["initial_equals_base"]
moe = MoEFFN(4, experts=4, top_k=2)
value, aux, chosen = moe(torch.randn(2, 3, 4))
out["moe"] = {"chosen_shape": list(chosen.shape), "output_shape": list(value.shape), "stored_experts": len(moe.experts), "selected_experts_per_token": chosen.shape[-1], "auxiliary": float(aux.detach())}
quantized, scale = quantize_symmetric(torch.tensor([-1., .1, 1.]), bits=4)
out["quantization"] = {"codes": quantized.tolist(), "scale": float(scale), "reconstruction": (quantized*scale).tolist(), "error": (quantized*scale-torch.tensor([-1., .1, 1.])).tolist()}
out["normalization"] = {"l2_length": float(nn.functional.normalize(torch.tensor([3., 4.]), dim=0).norm()), "text_same": fingerprint("a  b\n") == fingerprint("a b")}
sample = torch.randn(1, 1, 3, 4)
out["rope"] = {"norm_error": float((sample.norm(dim=-1)-rope(sample, torch.arange(3)).norm(dim=-1)).abs().max())}
q, k, v = torch.randn(3, 4), torch.randn(5, 4), torch.randn(5, 6)
out["online_attention"] = {"max_error": float((online_attention(q, k, v)-(q@k.T/math.sqrt(4)).softmax(-1)@v).abs().max())}
out["audio_image"] = {"patches": list(patchify(torch.zeros(1, 3, 16, 16)).shape), "mel_bank": list(mel_filter_bank().shape), "mel_features": list(log_mel(tone()).shape), "tone_samples": len(tone(frequency=440, seconds=1)), "tone_cycles_per_second": 440}
mm = MultiModalLM(TinyLM(ModelConfig(width=8)))
out["projectors"] = {"image": list(mm.image_projector.weight.shape), "audio": list(mm.audio_projector.weight.shape)}
out["kl_distillation"] = {"identical_kl": float(distillation_kl(logits, logits, labels)), "dpo_reference_detached": "detach" in inspect.getsource(dpo_loss), "advantage": bandit_advantage(torch.tensor([1., 0.]), torch.tensor([.5, .5])).tolist()}
assert out["kl_distillation"]["identical_kl"] < 1e-5
new = torch.tensor([math.log(.9)], requires_grad=True)
ppo = ppo_clipped_objective(new, torch.tensor([math.log(.5)]), torch.tensor([1.]))
ppo["policy_loss"].backward()
out["ppo"] = {"ratio": float(ppo["ratio"].detach()[0]), "clipped": float(ppo["clipped"].detach()[0]), "surrogate": float(ppo["surrogate"].detach()[0]), "gradient": float(new.grad[0]), "note": "合法目標函式可評估超出裁切範圍的策略；它沒有把策略機率投影回邊界"}
namespace = {"re": re}
tree = ast.parse(Path("scripts/course_experiments/applications.py").read_text())
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_policy_reward" or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_POLICY_ACTIONS" for t in n.targets)]
exec(compile(ast.Module(body=selected, type_ignores=[]), "applications.py", "exec"), namespace)
out["reward_rules"] = {"enumeration": namespace["_POLICY_ACTIONS"][-1], "weak": [namespace["_policy_reward"](16, n, True) for n in range(16)], "strict": [namespace["_policy_reward"](16, n, False) for n in range(16)]}
assert out["reward_rules"]["weak"] == [1.]*16 and out["reward_rules"]["strict"] == [0.]*16
out["tool"] = execute_tool_call(parse_tool_call('{"tool":"calculator","operation":"add","a":2,"b":3}'))
out["concentration"] = {"uniform": (1/3-1/3)/(1-1/3), "onehot": (1-1/3)/(1-1/3), "example": (.6-1/3)/(1-1/3)}
loop = LoopedSolver(Config(vocab_size=16, width=8, ff=16, heads=2))
calls = []
hook = loop.shared.register_forward_hook(lambda *args: calls.append(id(loop.shared)))
result = loop(torch.tensor([[1, 3, 4]]), loops=3, all_depths=True)
hook.remove()
out["loop"] = {"calls": len(calls), "same_block": len(set(calls)) == 1, "logits_shape": list(result.shape), "parameter_count": sum(p.numel() for p in loop.parameters())}
assert len(calls) == 3 and len(set(calls)) == 1
scorer = CandidateScorer(Config(vocab_size=16, width=8, ff=16, heads=2))
out["candidate_scorer"] = {"shape": list(scorer(torch.tensor([[1, 3, 4], [1, 4, 3]])).shape)}
out["observable_key"] = {"same_observable_different_family": observable_key({"state":"s", "question":"q", "candidates":["b","a"], "family_id":"first"}, "c") == observable_key({"state":"s", "question":"q", "candidates":["a","b"], "family_id":"second"}, "c")}
links = re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", body)
missing = []
for link in links:
    path, _, anchor = link.partition("#")
    target = Path("course")/path
    if not target.is_file() or anchor and not re.search(r"^#{1,6} " + re.escape(anchor) + r"(?:\s|$)", target.read_text(), re.M):
        missing.append(link)
index = body.split("### 英文與縮寫索引\n", 1)[1]
words = re.findall(r"\[([^\]]+)\]\([^)]+\)", index)
words = [w for w in words if w not in {"解碼", "逐步生成階段"}]
words.append("decode")
duplicates = sorted({w for w in words if words.count(w) > 1})
out["links_index"] = {"link_count": len(links), "missing": missing, "duplicates": duplicates, "index_word_count_including_decode": len(words), "body_headings": re.findall(r"^## (.+)", body, re.M), "repair_markers": re.findall(r".*(?:finding|裁決|審查輪次|TODO).*", body, re.I)}
assert not missing and not duplicates
out["repository_files"] = {}
for name in ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/simple.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py", "tiny_perceptron/alignment.py", "tiny_perceptron/posttraining.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/quantization.py", "tiny_perceptron/training.py", "tiny_perceptron/natural_assistant.py", "tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/data.py", "tiny_perceptron/lesson_products/train.py", "tiny_perceptron/lesson_products/training.py", "tiny_perceptron/selftrained/tools.py", "tiny_perceptron/selftrained/model.py", "scripts/course_experiments/applications.py", "scripts/course_experiments/architecture.py", "scripts/course_experiments/compression.py", "scripts/course_experiments/behavior.py", "docs/selftrained/v2-manifest.json"]:
    out["repository_files"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
out["result"] = "所有離線斷言通過；PPO 與 Unicode 的反例保留為技術問題的證據。"
(BASE/"verification-output.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
