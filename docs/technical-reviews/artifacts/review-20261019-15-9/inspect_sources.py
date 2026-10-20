"""以 curl 讀取原始來源；只保存查核收據與短摘錄。"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


specifications = [
    {
        "id": "switch",
        "url": "https://arxiv.org/html/2101.03961v3",
        "version": "arXiv:2101.03961v3, 2022-06-16; HTML rendition of the versioned paper",
        "start": "A Differentiable Load Balancing Loss.",
        "end": "2.3 Putting It All Together",
        "contains": ["f_{i}", "P_{i}", "10^{-2}", "differentiable"],
        "excerpt": "the P-vector is differentiable, but the f-vector is not.",
        "inspection": "逐讀負載均衡段落與公式。原式為 alpha*N*sum(f_i*P_i)，f_i 以 argmax 選擇數除以 T，P_i 是各 token softmax 的平均；每層加入總目標。原文在均勻情形給出內積 1/N，並說明 alpha 的取值不能壓過主要 cross-entropy。本文的 top-2 分母是專案明示的延伸，不冒充原論文的 top-1 原式。原文對全域最小值的敘述不作本報告的數學證據；教材也沒有聲稱此式是完美均衡度量。",
    },
    {
        "id": "torch_ops",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_torch_docs.py",
        "version": "PyTorch v2.14.1 tag",
        "contains": ["torch.argmax,", "torch.bincount,", "torch.mean,", "the indices of the first maximal value are returned"],
        "excerpt": "If there are multiple maximal values then the indices of the first maximal value are returned.",
        "inspection": "讀取 add_docstr 中 argmax(input, dim)、bincount(input, minlength) 及 mean(input, dim) 的原始文件；最後一軸 argmax 回傳各列索引，bincount 計算非負整數出現頻率，minlength 保留空 expert，mean(0) 沿列軸平均得到各欄的值。",
    },
    {
        "id": "torch_detach",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_tensor.py",
        "version": "PyTorch v2.14.1 tag",
        "start": "    detach = _C._add_docstr(",
        "end": "    detach_ = _C._add_docstr(",
        "contains": ["The result will never require gradient."],
        "excerpt": "The result will never require gradient.",
        "inspection": "逐讀 Tensor.detach 文件；detach 切斷目前計算圖並使結果不需要梯度。教材的 fraction 已因整數 argmax/bincount 而無梯度；detach 在這裡是明示固定統計，實際可微路徑仍由 mean_prob 回到 softmax。",
    },
    {
        "id": "torch_randomness",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/docs/source/notes/randomness.md",
        "version": "PyTorch v2.14.1 tag",
        "contains": ["torch.manual_seed()", "same series of random numbers", "individual commits, or different platforms"],
        "excerpt": "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.",
        "inspection": "讀取開頭限制與 Controlling sources of randomness；manual_seed 控制隨機數生成序列，同 seed 仍不是跨版本與平台的品質或完全可重現保證。單 seed 實驗不支持其他 seed 的相同效益。",
    },
    {
        "id": "model_selection",
        "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst",
        "version": "scikit-learn 1.7.2 tag; methodological documentation, no sklearn runtime used",
        "start": "When evaluating different settings",
        "end": "However, by partitioning",
        "contains": ["validation set", "test set", "final evaluation"],
        "excerpt": "A test set should still be held out for final evaluation",
        "inspection": "逐讀開頭與超參數比較段：調整設定會使測試集資訊洩漏，應以 validation 比較設定，保留 test 做最後評估。此文件支持教材的方法規則，不證明專案資料實際分離；實際分離另以資料重建與程式核對。",
    },
]
result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-9/inspect_sources.py",
          "environment": {"python": sys.version, "transport": "curl HTTPS; bodies held in memory, not written to repository"},
          "accessed_on": datetime.now(timezone.utc).date().isoformat(), "sources": []}
for spec in specifications:
    argv = ["curl", "-L", "--fail", "--silent", "--show-error", spec["url"]]
    p = subprocess.run(argv, capture_output=True, check=True)
    raw = p.stdout
    text = raw.decode()
    if spec["id"] == "switch":
        parser = Text()
        parser.feed(text)
        text = " ".join(" ".join(parser.parts).split())
    if "start" in spec:
        start = text.index(spec["start"])
        inspected = text[start:text.index(spec["end"], start)]
    else:
        inspected = text
    for needle in spec["contains"]:
        assert needle in inspected, (spec["id"], needle)
    if spec["id"] != "switch":
        assert " ".join(spec["excerpt"].split()) in " ".join(text.split()), spec["id"]
    result["sources"].append({"id": spec["id"], "url": spec["url"], "version": spec["version"],
                              "command_argv": argv, "exit_code": p.returncode, "body_bytes": len(raw),
                              "body_sha256": hashlib.sha256(raw).hexdigest(), "excerpt": spec["excerpt"],
                              "excerpt_locator": spec.get("start", ", ".join(spec["contains"])),
                              "inspection_note": spec["inspection"], "checked_original": True})
print(json.dumps(result, ensure_ascii=False, indent=2))
