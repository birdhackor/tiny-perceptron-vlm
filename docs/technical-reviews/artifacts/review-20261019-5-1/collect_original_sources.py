"""以 curl 讀取原始來源，只保存版本收據與支持主張的小段摘錄。"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime

BASE = "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/"
specs = [
    ("s_adamw", BASE + "torch/optim/adamw.py", "PyTorch v2.14.1", "AdamW.__init__、AdamW.__doc__ 的梯度動量公式", [
        "class AdamW(Adam):",
        "m_t           \\leftarrow   \\beta_1 m_{t-1} + (1 - \\beta_1) g_t",
        "v_t           \\leftarrow   \\beta_2 v_{t-1} + (1-\\beta_2) g^2_t",
    ]),
    ("s_zero", BASE + "torch/optim/optimizer.py", "PyTorch v2.14.1", "Optimizer.zero_grad", [
        "Reset the gradients of all optimized :class:`torch.Tensor` s.", "p.grad = None",
    ]),
    ("s_backward", BASE + "torch/_tensor.py", "PyTorch v2.14.1", "Tensor.backward", [
        "The graph is differentiated using the chain rule.",
        "This function accumulates gradients in the leaves",
    ]),
    ("s_parameters", BASE + "torch/nn/modules/module.py", "PyTorch v2.14.1", "Module.parameters", [
        "def parameters(self, recurse: bool = True) -> Iterator[Parameter]:",
        "for _name, param in self.named_parameters(recurse=recurse):", "yield param",
    ]),
    ("s_embedding", BASE + "torch/nn/modules/sparse.py", "PyTorch v2.14.1", "Embedding 類別與 weight 形狀", [
        "A simple lookup table that stores embeddings of a fixed dictionary and size.",
        "torch.empty((num_embeddings, embedding_dim), **factory_kwargs)",
    ]),
    ("s_norm", BASE + "torch/functional.py", "PyTorch v2.14.1", "norm：預設 fro 與向量二範數", [
        "p: float | str | None = \"fro\"", "Number  --              sum(abs(x)**ord)**(1./ord)",
        "return torch.linalg.vector_norm(\n                        input, 2, _dim, keepdim, dtype=dtype\n                    )",
    ]),
    ("s_train_test", "https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/basics/optimization_tutorial.py",
     "PyTorch tutorials main；以當次完整內容 SHA-256 固定讀取版本", "Optimization Loop：訓練與留出評估", [
        "**The Validation/Test Loop** - iterate over the test dataset to check if model performance is improving.",
    ]),
]
records = []
for identifier, url, version, locator, snippets in specs:
    args = ["curl", "-fsSL", "--max-time", "30", url]
    run = subprocess.run(args, capture_output=True, check=True)
    body = run.stdout.decode()
    excerpts = []
    for snippet in snippets:
        offset = body.find(snippet)
        if offset < 0:
            raise ValueError("原始來源未找到預期內容：" + identifier + " " + snippet)
        excerpts.append({"text": snippet, "line_computed_at_read": body[:offset].count("\n") + 1})
    records.append({
        "id": identifier, "url": url, "version": version, "locator": locator,
        "accessed_on": datetime.now(UTC).date().isoformat(), "checked_original": True,
        "command": args, "exit_code": run.returncode, "stderr": run.stderr.decode(),
        "full_response_sha256": hashlib.sha256(run.stdout).hexdigest(),
        "full_response_bytes": len(run.stdout), "excerpts": excerpts,
    })
print(json.dumps({"result": "已讀原始來源；完整第三方內容未保存。", "sources": records}, ensure_ascii=False, indent=2))
