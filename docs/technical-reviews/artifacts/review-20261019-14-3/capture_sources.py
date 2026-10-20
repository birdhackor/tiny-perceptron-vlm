"""透過 curl 讀取固定版本原始來源，只保存與主張相關的短段落。"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
TORCH = "5c4886908584029761b579af026dcfb627c84070"
QKNORM = "c628cb5d21f1475ba95db779a175748ff9efe940"
GEMMA = "014acb7ac4563a5f77c76d7ff98f31b568c16508"


def fetch(url, version, selections):
    command = ["curl", "-fLsS", "--max-time", "30", url]
    raw = subprocess.check_output(command)
    text = raw.decode("utf-8")
    excerpts = []
    for symbol, begin, end in selections:
        start = text.index(begin)
        finish = text.index(end, start) + len(end)
        excerpts.append({"symbol": symbol, "text": text[start:finish]})
    return {"url": url, "version": version, "accessed_on": datetime.now(UTC).date().isoformat(), "command_argv": command, "download_sha256": hashlib.sha256(raw).hexdigest(), "excerpts": excerpts}


sources = [
    fetch(f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/nn/functional.py", "PyTorch 2.14.1; git " + TORCH, [
        ("normalize", "def normalize(", "return torch.div(input, denom, out=out)"),
        ("softmax formula", "Softmax is defined as:", "lie in the range `[0, 1]` and sum to 1."),
        ("cosine_similarity formula", "cosine_similarity(x1, x2, dim=1", "\\epsilon)}"),
    ]),
    fetch(f"https://raw.githubusercontent.com/CyndxAI/QKNorm/{QKNORM}/QKNorm/layers.py", QKNORM, [
        ("ScaleUp", "class ScaleUp(nn.Module):", "return x * self.scale"),
        ("MultiheadAttention.forward QK normalization", "if self.mha_sup:\n            q = F.normalize", "att_weights = self.mha_scale(torch.bmm(q, k.transpose(1, 2)))"),
        ("MultiheadAttention.forward softmax", "att_weights = F.softmax(att_weights, dim=-1)", "att_weights = F.softmax(att_weights, dim=-1)"),
    ]),
    fetch(f"https://raw.githubusercontent.com/google/gemma_pytorch/{GEMMA}/gemma/model.py", GEMMA, [
        ("RMSNorm._norm", "    def _norm(self, x):", "return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)"),
        ("GemmaAttention QK norm modules", "        self.query_norm = (", "        self.attn_type = attn_type"),
        ("GemmaAttention.forward QK norm application", "        if self.query_norm is not None and self.key_norm is not None:", "            xk = self.key_norm(xk)"),
    ]),
    fetch(f"https://raw.githubusercontent.com/google/gemma_pytorch/{GEMMA}/gemma/config.py", GEMMA, [
        ("get_config_for_1b architecture", "def get_config_for_1b(dtype: str)", "architecture=Architecture.GEMMA_3,"),
        ("get_config_for_1b QK flag", "      tokenizer='tokenizer/gemma3_cleaned_262144_v2.spiece.model',", "      use_qk_norm=True,"),
    ]),
]
target = BASE / "source-excerpts.json"
target.write_text(json.dumps({"sources": sources, "scope": "只保存公式與實作短摘錄；未保存第三方全文。"}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": "原始來源已讀取並節錄", "path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "sources": len(sources)}, ensure_ascii=False))
