"""重算 LoRA 算例、核對訓練收據，並以 curl 讀取原始來源。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.alignment import LoRALinear


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    path = HERE / name
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        assert path.read_text() == raw, "既有證據內容與本次輸出不同"
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(raw)


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = None
        self.parts = []
        self.items = []

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "h2", "h3", "h4"):
            self.active = tag
            self.parts = []
        if self.active and tag == "math":
            self.parts.append(dict(attrs).get("alttext", ""))

    def handle_endtag(self, tag):
        if self.active == tag:
            self.items.append(" ".join(" ".join(self.parts).split()))
            self.active = None

    def handle_data(self, text):
        if self.active:
            self.parts.append(text)


SPECS = [
    {
        "id": "lora",
        "url": "https://arxiv.org/html/2106.09685v2",
        "version": "arXiv:2106.09685v2",
        "needles": ["Low-Rank-Parametrized Update Matrices", "random Gaussian initialization", "During training,", "We still need"],
        "excerpts": ["We use a random Gaussian initialization for", "and zero for", "We then scale"],
        "note": "閱讀 Low-Rank-Parametrized Update Matrices 與 Application to Transformer 的相關段落：W 凍結、A/B 形狀、同輸入兩路相加、B 零初始化、alpha/r 縮放、部署仍需底座。r 是瓶頸寬度，BA 的實際秩至多為 r。",
    },
    {
        "id": "tensor",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_tensor.py",
        "version": "PyTorch v2.14.1",
        "needles": ["def backward(", "detach =", "Returned Tensor shares the same storage"],
        "excerpts": ["Computes the gradient of current tensor wrt graph leaves.", "Returned Tensor shares the same storage with the original one."],
        "note": "閱讀 Tensor.backward 與 Tensor.detach 的完整 docstring；backward 累積葉節點梯度，detach 不追蹤梯度但仍共享儲存。",
    },
    {
        "id": "clone",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_torch_docs.py",
        "version": "PyTorch v2.14.1",
        "needles": ["Returns a copy of :attr:`input`."],
        "excerpts": ["Returns a copy of :attr:`input`."],
        "note": "閱讀 torch.clone 條目；clone 建立副本，本身可微分；先 detach 再 clone 取得不連梯度圖且儲存獨立的舊值。",
    },
    {
        "id": "linear",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/linear.py",
        "version": "PyTorch v2.14.1",
        "needles": ["class Linear", "self.weight = Parameter(", "F.linear(input, self.weight, self.bias)"],
        "excerpts": ["torch.empty((out_features, in_features), **factory_kwargs)", "return F.linear(input, self.weight, self.bias)"],
        "note": "閱讀 Linear 的類別說明、建構子與 forward：weight 為 out_features×in_features；bias 預設存在，實際算例底座為仿射變換，圖的 Wx 是權重路徑的簡寫。",
    },
    {
        "id": "sgd",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/optim/sgd.py",
        "version": "PyTorch v2.14.1",
        "needles": ["def step(", "param.add_(grad, alpha=-lr)"],
        "excerpts": ["param.add_(grad, alpha=-lr)"],
        "note": "閱讀 SGD.step、單 tensor SGD 與預設參數；lr=0.1、momentum=0、weight_decay=0 時以原地加法更新參數。零梯度的 A 首步保持原值。",
    },
    {
        "id": "qwen_config",
        "url": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/config.json",
        "version": "Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203",
        "needles": ["\"num_hidden_layers\": 28", "\"hidden_size\": 2048"],
        "excerpts": [],
        "note": "讀完整官方固定版本 config；取文字核心 hidden_size、head_dim、attention heads、key/value heads 與 layers，重算 q/v 的矩陣尺寸。",
    },
    {
        "id": "qwen_model",
        "url": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
        "version": "Transformers v4.57.6",
        "needles": ["self.q_proj = nn.Linear(", "self.v_proj = nn.Linear("],
        "excerpts": ["config.hidden_size, config.num_attention_heads * self.head_dim", "config.hidden_size, config.num_key_value_heads * self.head_dim"],
        "note": "閱讀 Qwen3VLTextAttention 建構子與 forward；q_proj 輸出為 attention heads×head_dim，v_proj 為 key/value heads×head_dim。固定版本對應 requirements-natural.txt。",
    },
    {
        "id": "peft",
        "url": "https://raw.githubusercontent.com/huggingface/peft/v0.18.1/src/peft/tuners/lora/layer.py",
        "version": "PEFT v0.18.1",
        "needles": ["self.scaling[adapter_name] = lora_alpha / r", "nn.init.zeros_(self.lora_B[adapter_name].weight)"],
        "excerpts": ["self.scaling[adapter_name] = lora_alpha / r", "nn.init.zeros_(self.lora_B[adapter_name].weight)"],
        "note": "閱讀 LoraLayer.update_layer、reset_lora_parameters 及 Linear.forward：標準 LoRA 的 A/B 形狀、B 零初始化與 alpha/r 縮放；不使用 rsLoRA 時適用。",
    },
]


def primary_sources():
    records = []
    fetched = {}
    for spec in SPECS:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", spec["url"]]
        raw = subprocess.check_output(command)
        text = raw.decode("utf-8")
        for needle in spec["needles"] + spec["excerpts"]:
            assert needle in text, (spec["id"], needle)
        fetched[spec["id"]] = text
        excerpts = []
        for needle in spec["excerpts"]:
            at = text.index(needle)
            excerpts.append({"text": needle, "response_line": text[:at].count("\n") + 1})
        record = {
            "id": spec["id"], "url": spec["url"], "version": spec["version"],
            "accessed_on": datetime.now(UTC).date().isoformat(), "command": command,
            "response_sha256": digest(raw), "response_bytes": len(raw),
            "short_original_excerpts": excerpts, "inspection_note": spec["note"],
        }
        records.append(record)
    parser = Paragraphs()
    parser.feed(fetched["lora"])
    for i, text in enumerate(parser.items):
        if i in (29, 30, 31, 35, 36):
            print("PAPER", text)
    for source_id, start, end in (("tensor", "def backward(", "def register_hook("), ("clone", '"clone",', '"contiguous",'), ("qwen_model", "class Qwen3VLTextAttention", "def forward(")):
        text = fetched[source_id]
        at = text.find(start)
        if at >= 0:
            until = text.find(end, at + len(start))
            selected = text[at:until] if until >= 0 else text[at:at + 3000]
            print("ORIGINAL", source_id, selected[:12500])
    config = json.loads(fetched["qwen_config"])["text_config"]
    selected = {key: config[key] for key in ("hidden_size", "head_dim", "num_attention_heads", "num_key_value_heads", "num_hidden_layers", "attention_bias")}
    next(item for item in records if item["id"] == "qwen_config")["selected_original_values"] = selected
    save("primary-sources.json", {"records": records, "policy": "全文只在記憶體中處理；僅保存短摘錄、定位、版本與全文 SHA-256。"})
    return selected


def examples():
    results = []
    for rank in (2, 1):
        torch.manual_seed(42)
        layer = LoRALinear(nn.Linear(4, 3), rank=rank, alpha=rank)
        x = torch.ones(1, 4)
        initial = {name: value.detach().clone() for name, value in layer.named_parameters()}
        optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
        same_output = torch.equal(layer(x), layer.base(x))
        output = layer(x)
        loss = output.square().mean()
        loss.backward()
        grads = {name: None if p.grad is None else p.grad.detach().tolist() for name, p in layer.named_parameters()}
        expected_b_grad = (2 / 3) * (output.detach().T @ (x @ layer.a.detach().T))
        torch.testing.assert_close(layer.b.grad, expected_b_grad, atol=1e-9, rtol=1e-6)
        assert layer.a.grad.count_nonzero().item() == 0
        assert layer.b.grad.count_nonzero().item() > 0
        optimizer.step()
        booleans = [torch.equal(initial["base.weight"], layer.base.weight), torch.equal(initial["a"], layer.a), not torch.equal(initial["b"], layer.b)]
        assert booleans == [True, True, True]
        assert torch.equal(initial["base.bias"], layer.base.bias)
        alias = layer.b
        detached = layer.b.detach()
        independent = layer.b.detach().clone()
        optimizer.zero_grad()
        loss_after = layer(x).square().mean()
        loss_after.backward()
        a_second_gradient_max = float(layer.a.grad.abs().max())
        assert a_second_gradient_max > 0
        a_before_second = layer.a.detach().clone()
        optimizer.step()
        assert not torch.equal(a_before_second, layer.a)
        assert alias is layer.b and torch.equal(alias, layer.b)
        assert detached.data_ptr() == layer.b.data_ptr()
        assert independent.data_ptr() != layer.b.data_ptr()
        assert not torch.equal(independent, layer.b)
        results.append({
            "rank": rank, "alpha": rank, "scale": layer.alpha / layer.rank,
            "shapes": {name: list(p.shape) for name, p in layer.named_parameters()},
            "trainable_names": [name for name, p in layer.named_parameters() if p.requires_grad],
            "trainable_count": sum(p.numel() for p in layer.parameters() if p.requires_grad),
            "initial_output_equals_base": same_output, "initial_output": output.detach().tolist(),
            "loss_before": float(loss.detach()), "loss_after_first_update": float(loss_after.detach()),
            "first_gradients": grads, "expected_b_gradient": expected_b_grad.tolist(),
            "first_update_booleans": booleans, "base_bias_unchanged": True,
            "a_second_gradient_max": a_second_gradient_max, "a_changes_on_second_step": True,
            "alias_shares_updates": True, "detach_shares_storage": True, "clone_has_independent_storage": True,
        })
    return results


def receipts(config):
    base = ROOT / "docs/natural-assistant/evidence/v4-runtime/train-37217452291"
    audit = json.loads((base / "training-audit-summary.json").read_bytes())
    index = json.loads((base / "actual-artifact-index.json").read_bytes())
    independent = audit["independent_audit"]
    sizes = [{"path": f["path"], "bytes": f["bytes"], "sha256": f["sha256"]} for f in index["files"] if f["path"].endswith("adapter_model.safetensors")]
    width = config["hidden_size"]
    q_out = config["num_attention_heads"] * config["head_dim"]
    v_out = config["num_key_value_heads"] * config["head_dim"]
    count = config["num_hidden_layers"] * 8 * ((width + q_out) + (width + v_out))
    assert count == independent["trainable_lora_parameters"] == 1605632
    assert config["num_hidden_layers"] * 2 * 2 == independent["optimizer_name_count"] == independent["changed_adapter_tensors"] == 112
    assert {f["bytes"] for f in sizes} == {6438952}
    assert audit["paired_training_control"]["new_learning_rate"] == 0.00003
    code = (ROOT / "tiny_perceptron/natural_assistant.py").read_text()
    target = re.search(r'^LORA_TARGETS = (.+)$', code, re.M).group(1)
    return {
        "qwen_text_dimensions": config, "q_out": q_out, "v_out": v_out,
        "formula": "28 * 8 * ((2048 + 2048) + (2048 + 1024))",
        "trainable_parameter_count": count, "tensor_count": 112,
        "float32_parameter_payload_bytes": count * 4,
        "indexed_adapter_files": sizes, "decimal_MB": 6438952 / 1000000,
        "recorded_learning_rate": audit["paired_training_control"]["new_learning_rate"],
        "rank_from_source_and_config": 8, "alpha_from_source": 16,
        "target_source": target,
        "audit_original_values": {key: independent[key] for key in ("changed_adapter_tensors", "frozen_samples_unchanged", "frozen_scope", "completed_steps", "trained_rows", "supervised_tokens")},
        "audit_json_root_keys": list(audit),
        "audit_does_not_embed": ["per-update history", "initial_adapter_tensors", "final_adapter_tensors", "optimizer_parameter_names", "adapter_config.json"],
        "full_result_indexed_separately": next(f for f in index["files"] if f["path"] == "review/result.json"),
        "scope": "重算固定配置與直接核對 repo 保存的稽核摘要、檔案索引、稽核器和訓練程式；未讀取被排除的原始 outputs 或私有權重，未重新訓練。",
    }


def main():
    original = ROOT / "outputs/reviewer-tools/runs/20261020T063139204057Z"
    execution = {name: json.loads((original / name).read_bytes()) for name in ("extraction.json", "execution.json", "environment.json")}
    execution.update({name: (original / name).read_text() for name in ("stdout.txt", "stderr.txt", "fence-1.py")})
    execution["invocation"] = 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/20.md#20.6" --execute'
    assert execution["execution.json"]["exit_code"] == 0
    assert execution["extraction.json"]["source_sha256"] == "9b8053b94a2e1abf35fc032411dcaac8dba0e42214819e4de1401d307b8d04c9"
    save("section-execution.json", execution)
    config = primary_sources()
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + Path(__file__).relative_to(ROOT).as_posix(),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "examples": examples(), "receipt_checks": receipts(config),
        "derivation": "h=base(x)+s*B*A*x；L=sum(h_i^2)/3；dL/dA=s*B^T*(2h/3)*x^T，B=0 時為零；dL/dB=s*(2h/3)*(Ax)^T。修正參數量=r*(in+out)，少於 W 的條件為 r*(in+out)<in*out。rank=2 是 2*(4+3)=14>12；rank=1 是 7<12。",
    }
    save("verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
