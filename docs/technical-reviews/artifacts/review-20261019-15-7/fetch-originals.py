import hashlib
import html
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def fetch(url):
    result = subprocess.run(
        ["curl", "-fsSL", "--max-time", "45", url], capture_output=True, check=True
    )
    return result.stdout


items = []
specs = [
    (
        "autograd",
        "https://docs.pytorch.org/docs/2.14/notes/autograd.html",
        "PyTorch 2.14；How autograd encodes the history 與 Setting requires_grad",
        "Autograd is a reverse automatic differentiation system.",
    ),
    (
        "topk",
        "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html",
        "PyTorch 2.14；torch.topk",
        "Returns the k largest elements of the given input tensor along a given dimension.",
    ),
    (
        "topk_derivative",
        "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/tools/autograd/derivatives.yaml",
        "本機 torch.version.git_version；topk 的導數宣告",
        "output_differentiability: [True, False]",
    ),
    (
        "switch",
        "https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf",
        "JMLR 23(120), 2022，最終出版版；Simplifying Sparse Routing 公式 (1)(2) 與 A Differentiable Load Balancing Loss 公式 (4)–(6)",
        None,
    ),
]
for source_id, url, version, excerpt in specs:
    raw = fetch(url)
    item = {
        "id": source_id,
        "url": url,
        "version": version,
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "fetch_command": "curl -fsSL --max-time 45 " + url,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
    }
    if excerpt:
        text = raw.decode("utf-8")
        text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", html.unescape(text)).strip()
        if excerpt not in text:
            raise ValueError("找不到指定原文摘錄：" + source_id)
        item["excerpt"] = excerpt
    else:
        if not raw.startswith(b"%PDF"):
            raise ValueError("原始論文回應不是 PDF")
        item["inspection_method"] = "curl 取得原檔並計算 SHA-256；以 web PDF 文字檢視核讀正文，未保存全文。"
        item["excerpt"] = "the gate value pi(x) in Equation 2 permits differentiability of the router."
        item["inspection_note"] = "公式 (1) 是全 expert 的 softmax；公式 (2) 以選中 gate 加權 expert 輸出。單一 expert 保留 gate，路由器仍有連續梯度路徑。輔助目標由機率向量提供可微分路徑，另乘係數 alpha 加進訓練目標。"
    items.append(item)

output = Path(__file__).with_name("primary-sources.json")
with output.open("x", encoding="utf-8") as stream:
    json.dump({"sources": items}, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"saved": str(output), "sources": [{"id": i["id"], "response_bytes": i["response_bytes"], "response_sha256": i["response_sha256"]} for i in items]}, ensure_ascii=False))
