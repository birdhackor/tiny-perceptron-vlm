"""以 curl 親讀官方來源，僅保存短摘錄、定位與本次核對摘要。"""

import hashlib
import json
import re
import subprocess
from datetime import date
from html.parser import HTMLParser


class Article(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []

    def handle_starttag(self, tag, attributes):
        if tag == "article":
            self.active = True

    def handle_endtag(self, tag):
        if tag == "article":
            self.active = False

    def handle_data(self, data):
        if self.active and data.strip():
            self.parts.append(data.strip())


specs = [
    {
        "id": "compile_api",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.compile.html",
        "title": "torch.compile",
        "version": "PyTorch 2.14 官方版本文件",
        "terms": ["guard failure", "fullgraph", "dynamic", "we will always specialize", "fall back to eager"],
        "excerpt": "compile it and cache the compiled result on the code object for future use.",
        "locator": "torch.compile 描述；fullgraph、dynamic、backend 參數",
        "inspection_note": "原文規定逐 frame 捕捉與快取，guard 失效可重新編譯並在上限後回退 eager。fullgraph=True 要求單圖，graph break 會報錯；dynamic=False 固定特化，不生成動態 kernel。",
    },
    {
        "id": "compiler_overview",
        "url": "https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html",
        "title": "torch.compiler",
        "version": "PyTorch 2.14；頁面更新日期 2026-07-20",
        "terms": ["Jul 20, 2026", "Frame Evaluation API", "TorchInductor", "generates fast code for multiple accelerators and backends"],
        "excerpt": "generates fast code for multiple accelerators and backends.",
        "locator": "TorchDynamo、TorchInductor 技術說明與 backend 段落",
        "inspection_note": "Dynamo 捕捉 PyTorch 圖；Inductor 是預設編譯後端，為多種裝置產生程式。文件把產生機器碼與可能的速度收益連結，但沒有保證任意模型都加速。",
    },
    {
        "id": "dynamo_core",
        "url": "https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html",
        "title": "Dynamo Core Concepts",
        "version": "PyTorch 2.14；頁面更新日期 2026-07-27",
        "terms": ["Jul 27, 2026", "Data-dependent if-statements", "TENSOR_MATCH", "Recompilations add to overall compile time"],
        "excerpt": "Guards are run in future calls to the compiled function to determine if we can reuse previously compiled code.",
        "locator": "Summary、Dynamo Tracing、Graph Breaks、Guards、Recompilations、Dynamic Shapes",
        "inspection_note": "逐段親讀圖捕捉、Python 不支援部分的 eager 執行及恢復追蹤；guard 包含 type/device/dtype/shape 等條件，重用不只取決於形狀。資料相依的 Python if 可能切圖，形狀變化可能重編譯，dynamic shapes 可以減少重編譯。",
    },
    {
        "id": "cuda_semantics",
        "url": "https://docs.pytorch.org/docs/2.14/notes/cuda.html",
        "title": "CUDA semantics",
        "version": "PyTorch 2.14 官方版本文件",
        "terms": ["time measurements without synchronizations are not accurate", "max_memory_allocated()", "max_memory_reserved()"],
        "excerpt": "time measurements without synchronizations are not accurate.",
        "locator": "Asynchronous execution；Memory management",
        "inspection_note": "GPU 預設非同步，牆鐘計時要同步或用 CUDA events 並等完成；allocated 統計 tensor 佔用，reserved 統計配置器管理量。搭配獨立程序程式與記錄，可判斷主程序峰值不能代表子程序 GPU 峰值。",
    },
    {
        "id": "linear_api",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html",
        "title": "torch.nn.Linear",
        "version": "PyTorch 2.14 官方版本文件",
        "terms": ["y = xA^T + b", "all but the last dimension", "out_features"],
        "excerpt": "all but the last dimension are the same shape as the input",
        "locator": "Linear 定義與 Shape",
        "inspection_note": "線性層沿最後一維以 in_features→out_features 轉換，保留其餘維度；所以 Linear(4,3) 對 [2,4] 輸入輸出 [2,3]。",
    },
    {
        "id": "eager_source",
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_dynamo/backends/debugging.py",
        "title": "PyTorch eager backend 原始碼",
        "version": "pytorch/pytorch tag v2.14.1",
        "terms": ["def eager(", "return gm.forward"],
        "excerpt": "return gm.forward",
        "locator": "eager",
        "inspection_note": "親讀 eager 函式：預設直接交回 GraphModule.forward；特殊 autograd cache 設定下包成可序列化 callable。這不是 Inductor 的 kernel 產生路徑。",
        "source_code": True,
    },
]
receipt = {"accessed_on": date.today().isoformat(), "command": "curl --fail --silent --show-error --location <各項 url>", "sources": []}
for spec in specs:
    raw = subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--location", spec["url"]])
    content = raw.decode()
    if spec.get("source_code"):
        text = content
    else:
        parser = Article()
        parser.feed(content)
        text = " ".join(parser.parts)
    text = re.sub(r"\s+", " ", text)
    for term in spec["terms"]:
        assert term in text, f"官方原文缺少預期定位：{spec['id']} {term}"
    assert spec["excerpt"] in text, f"摘錄無法在官方原文定位：{spec['id']}"
    saved = {k: v for k, v in spec.items() if k not in ("terms", "source_code")}
    saved.update(checked_original=True, fetched_content_sha256=hashlib.sha256(raw).hexdigest(), checked_terms=spec["terms"])
    if spec.get("source_code"):
        saved["excerpt_line"] = next(i for i, line in enumerate(content.splitlines(), 1) if spec["excerpt"] in line)
    receipt["sources"].append(saved)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
