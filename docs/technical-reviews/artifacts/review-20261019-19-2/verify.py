"""核對凍結小節的計數、路由、狀態載入契約與公開紀錄。"""

import hashlib
import html
import inspect
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from tiny_perceptron.modern import DenseFFN
from tiny_perceptron.selftrained.dataset import train_tokenizer
from tiny_perceptron.selftrained.model import LimitedAssistant, PaddingSafeMoE, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, OCR_CHARACTERS, SPECIALS


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    return completed.stdout, {"command": command, "response_sha256": sha(completed.stdout), "bytes": len(completed.stdout)}


def plain(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


torch.manual_seed(19)
torch.set_num_threads(1)
outdir = Path(__file__).resolve().parent
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-2/verify.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "seed": "19"},
    "source_files": {},
}
source_paths = [
    "tiny_perceptron/selftrained/model.py", "tiny_perceptron/selftrained/tokenizer.py",
    "tiny_perceptron/selftrained/dataset.py", "tiny_perceptron/model.py",
    "tiny_perceptron/modern.py", "tiny_perceptron/attention.py", "scripts/selftrained/train.py",
    "docs/selftrained/results/v2-final-public-results.json",
    "docs/course-review-20261019/freeze-01/sources/19.2.md",
    "docs/course-review-20261019/freeze-04.json",
]
for path in source_paths:
    result["source_files"][path] = sha((ROOT / path).read_bytes())

frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/19.2.md").read_text()
fences = re.findall(r"```python\n(.*?)```", frozen, re.S)
namespace = {}
exec(compile(fences[1], "<frozen-class>", "exec"), namespace)
SimpleAssistant = namespace["LimitedAssistant"]
public = json.loads((ROOT / "docs/selftrained/results/v2-final-public-results.json").read_bytes())
result["counts"] = {}
result["structure"] = {}
for architecture in ("moe", "dense"):
    for top_k in (2, 1):
        config = SelftrainedConfig(vocab_size=550, width=256, ffn_hidden=512, top_k=top_k, architecture=architecture)
        model, simple = LimitedAssistant(config), SimpleAssistant(config)
        loaded = simple.load_state_dict(model.state_dict())
        description = model.description()
        name = f"{architecture}_top{top_k}"
        result["counts"][name] = {key: description[key] for key in ["parameters", "language_parameters", "active_language_parameters_per_token", "expert_parameters", "router_parameters", "perception_parameters"]}
        result["counts"][name]["simple_counts"] = simple.description()
        result["counts"][name]["loaded"] = str(loaded)
        assert simple.description() == {key: description[key] for key in simple.description()}
        if top_k == 2:
            assert description == public["architectures"][architecture]["capacity"]
            result["structure"][architecture] = {
                "config": asdict(config),
                "parts": {key: type(value).__name__ for key, value in model.named_children()},
                "blocks": [{"ffn": type(b.ffn).__name__, "norm1": type(b.norm1).__name__, "norm2": type(b.norm2).__name__, "q_shape": list(b.attention.q.weight.shape), "k_shape": list(b.attention.k.weight.shape), "v_shape": list(b.attention.v.weight.shape), "query_heads": b.attention.heads, "kv_heads": b.attention.kv_heads, "rotary": b.attention.rotary} for b in model.lm.blocks],
                "same_embedding_output_parameter": model.lm.embedding.weight is model.lm.output.weight,
                "shared_parameter_occurrences": sum(p is model.lm.embedding.weight for p in model.parameters()),
                "state_keys_equal": set(model.state_dict()) == set(simple.state_dict()),
                "class_name_expression": type(model.lm.blocks[0].ffn).__name__,
                "description_type": type(description).__name__,
            }
        del model, simple

expert = 256 * 512 + 512 + 512 * 256 + 256
shared = 550 * 256 + 4 * (256 * 256 + 2 * 256 * 128 + 256 * 256 + 2 * 256) + 256
result["arithmetic"] = {
    "expert": expert, "all_experts": 4 * 4 * expert, "shared_dense": shared,
    "router": 4 * 256 * 4, "moe_language": shared + 4 * 4 * expert + 4 * 256 * 4,
    "dense_language": shared + 4 * expert,
    "moe_top2_active": 5140224 - 4206592 * 2 // 4,
    "moe_top1_active": 5140224 - 4206592 * 3 // 4,
    "perception": 5447107 - 5140224,
    "top1_minus_dense": 1985280 - 1981184,
    "floor_division_example": 7 // 4,
}
route = PaddingSafeMoE(4, 4, 2, 8)
x = torch.randn(1, 3, 4)
valid = torch.tensor([[True, False, True]])
y, auxiliary, stats = route(x, valid)
active = x[valid]
probabilities = route.router(active).float().softmax(-1)
weights, chosen = probabilities.topk(2, dim=-1)
weights = weights / weights.sum(-1, keepdim=True)
manual = torch.stack([sum(route.experts[int(chosen[r, s])](active[r]) * weights[r, s] for s in range(2)) for r in range(2)])
result["routing"] = {
    "counts": stats["counts"].tolist(), "valid_tokens": stats["valid_tokens"],
    "chosen": stats["chosen"].tolist(), "padding_output": y[~valid].tolist(),
    "weighted_sum_matches": torch.allclose(y[valid], manual, atol=1e-7, rtol=1e-6),
    "max_difference": (y[valid] - manual).abs().max().item(),
    "expert_types": [type(e).__name__ for e in route.experts],
    "expert_parameters_trainable": all(p.requires_grad for e in route.experts for p in e.parameters()),
}
assert result["routing"]["weighted_sum_matches"] and stats["counts"].sum().item() == 4
assert stats["chosen"][0, 1].tolist() == [-1, -1] and y[~valid].count_nonzero().item() == 0

a, b = DenseFFN(2, 4, activation="gelu"), DenseFFN(2, 4, activation="relu2")
with torch.no_grad():
    for p in a.parameters():
        p.fill_(0.5)
loaded = b.load_state_dict(a.state_dict())
sample = torch.tensor([[1.0, 2.0]])
result["state_loading_counterexample"] = {
    "source_activation": a.activation, "destination_activation": b.activation,
    "load_result": str(loaded), "same_tensor_names_and_shapes": all(a.state_dict()[k].shape == b.state_dict()[k].shape for k in a.state_dict()),
    "source_output": a(sample).detach().tolist(), "destination_output": b(sample).detach().tolist(),
    "outputs_equal": torch.equal(a(sample), b(sample)),
}
assert not result["state_loading_counterexample"]["outputs_equal"]
for label, state in [("missing_name", {k: v for k, v in a.state_dict().items() if k != "up.bias"}), ("wrong_shape", {**a.state_dict(), "up.bias": torch.zeros(5)})]:
    try:
        b.load_state_dict(state)
    except RuntimeError as error:
        result.setdefault("state_loading_errors", {})[label] = str(error)
    else:
        raise AssertionError("不相容的狀態未被拒絕")

records = [{"split": split, "messages": [{"content": text}]} for split, text in [("train", "甲"), ("validation", "乙"), ("test", "丙")]]
tokenizer = train_tokenizer(records)
result["tokenizer_probe"] = {"specials": list(SPECIALS), "special_count": len(SPECIALS), "ocr_characters": OCR_CHARACTERS, "ocr_count": len(OCR_CHARACTERS), "all_ascii_printable_present": all(chr(i) in tokenizer.characters for i in range(32, 127)), "train_character_present": "甲" in tokenizer.characters, "validation_character_present": "乙" in tokenizer.characters, "test_character_present": "丙" in tokenizer.characters, "encoded": tokenizer.encode("甲乙丙"), "unknown_id": tokenizer.unk_id}
assert tokenizer.encode("乙丙") == [tokenizer.unk_id] * 2

result["training"] = {}
for arch in ("moe-native", "dense-weighted"):
    base = Path("docs/selftrained/results/training-raw") / arch
    d = json.loads((ROOT / base / "raw/train-receipt.json").read_bytes())
    e = json.loads((ROOT / base / "raw/execution.json").read_bytes())
    for p in [base / "raw/train-receipt.json", base / "raw/execution.json"]:
        result["source_files"][str(p)] = sha((ROOT / p).read_bytes())
    result["training"][arch] = {key: d[key] for key in ["steps", "tokens", "target_tokens", "train_records", "validation_records", "test_used_for_selection", "seed", "config", "selection", "tokenizer_sha256", "tool_loss_weight", "numeric_run_loss_weight", "stage_history", "origin"]}
    result["training"][arch]["native_voice_loss_weight"] = d.get("native_voice_loss_weight", 1.0)
    result["training"][arch]["execution"] = {key: e[key] for key in ["run_id", "stage", "revision", "status", "returncode", "command"]}

snapshots = {"accessed_on": "2026-10-20", "sources": []}
paper_url = "https://arxiv.org/html/2401.04088v1"
raw, receipt = fetch(paper_url)
paras = [plain(p) for p in re.findall(r"<p\b[^>]*>(.*?)</p>", raw.decode(), re.S)]
selected = [p[p.index(needle):] for needle in ["This motivates a distinction", "Note that EP introduces"] for p in paras if needle in p]
assert len(selected) == 2
snapshots["sources"].append({"id": "mixtral", "url": paper_url, "version": "arXiv:2401.04088v1, 2024-01-08", "receipt": receipt, "locator": "Sparse Mixture of Experts", "excerpts": selected, "inspection": "另在原文讀取加權輸出公式、TopK gate 公式與 expert 為標準 FFN 的圖說；僅引用機制與限制，未移用該模型的效能結果。"})

torch_url = f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/torch/nn/modules/module.py"
raw, receipt = fetch(torch_url)
upstream = raw.decode()
local = Path(inspect.getsourcefile(nn.Module)).read_bytes()
snippets = []
for needle in ["    def load_state_dict(", "                if not is_param_lazy and input_param.shape != param.shape:", "    def parameters(", "    def named_parameters("]:
    start = upstream.index(needle)
    snippets.append(upstream[start:start + (900 if "load_state_dict" in needle else 650)])
snapshots["sources"].append({"id": "torch_module", "url": torch_url, "version": torch.__version__, "git_commit": torch.version.git_version, "receipt": receipt, "local_file_sha256": sha(local), "upstream_matches_installed_file": sha(local) == sha(raw), "excerpts": snippets, "inspection": "讀取 load_state_dict、_load_from_state_dict、parameters、named_parameters 與 _named_members；預設 strict 比對狀態鍵，載入時比對張量形狀；未比較 activation 或 forward；parameters 預設去除重複 Parameter。"})
assert sha(local) == sha(raw)

result["public_metadata"] = {}
hfroot = f"https://huggingface.co/{public['hf_repo']}/resolve/{public['hf_revision']}/selftrained/v2"
for arch in ("moe", "dense"):
    result["public_metadata"][arch] = {}
    for filename in ("model-config.json", "inference-manifest.json", "tokenizer.json"):
        url = f"{hfroot}/{arch}-joint/{filename}"
        raw, receipt = fetch(url)
        d = json.loads(raw)
        if filename == "tokenizer.json":
            t = CharacterTokenizer.from_dict(d)
            training_sha = sha(json.dumps(d, ensure_ascii=False, sort_keys=True).encode())
            values = {"vocab_size": t.vocab_size, "ordinary_characters": len(t.characters), "specials": d["specials"], "compact_tokenizer_sha256": t.sha256, "training_tokenizer_sha256": training_sha}
            assert t.vocab_size == 550 and training_sha == result["training"]["moe-native"]["tokenizer_sha256"]
        elif filename == "model-config.json":
            values = d
            assert d == public["architectures"][arch]["capacity"]["config"]
        else:
            keys = ["schema", "stage", "selected_step", "selected_checkpoint_sha256", "files", "origin", "tokenizer_sha256", "selection", "tool_loss_weight", "numeric_run_loss_weight", "native_voice_loss_weight"]
            values = {key: d[key] for key in keys if key in d}
            assert d["selected_step"] == 1000
            assert d["selected_checkpoint_sha256"] == public["architectures"][arch]["selected_source"]["sha256"]
        result["public_metadata"][arch][filename] = {"url": url, "receipt": receipt, "data": values}
    manifest = result["public_metadata"][arch]["inference-manifest.json"]["data"]
    for filename in ["model-config.json", "tokenizer.json"]:
        assert manifest["files"][filename] == result["public_metadata"][arch][filename]["receipt"]["response_sha256"]

(outdir / "original-source-excerpts.json").write_text(json.dumps(snapshots, ensure_ascii=False, indent=2) + "\n")
(outdir / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"counts": result["counts"], "arithmetic": result["arithmetic"], "routing": result["routing"], "state_loading_counterexample": result["state_loading_counterexample"], "tokenizer_probe": result["tokenizer_probe"], "public_metadata_fetched": list(result["public_metadata"]), "upstream_torch_matches_installed": sha(local) == sha(upstream.encode())}, ensure_ascii=False, indent=2))
