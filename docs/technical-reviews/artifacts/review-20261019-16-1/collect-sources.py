"""以 curl 讀取原始來源，只保存支持主張的小段摘錄與取得收據。"""

import hashlib
import html
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
COMMIT = torch.version.git_version
BASE = "https://raw.githubusercontent.com/pytorch/pytorch/" + COMMIT + "/"


def fetch(identifier, url, version, anchors=None, line_terms=None, local=None):
    command = ["curl", "--location", "--fail", "--silent", "--show-error", url]
    result = subprocess.run(command, capture_output=True, check=True)
    raw = result.stdout
    text = raw.decode("utf-8")
    item = {"id": identifier, "url": url, "version": version, "accessed_at": datetime.now(timezone.utc).isoformat(), "command_argv": command, "exit_code": result.returncode, "original_sha256": hashlib.sha256(raw).hexdigest(), "checked_original": True, "excerpts": []}
    if anchors:
        normalized = html.unescape(re.sub(r"<[^>]*>", " ", text))
        normalized = re.sub(r"\s+", " ", normalized)
        for anchor, before, after in anchors:
            start = normalized.find(anchor)
            if start < 0:
                raise ValueError("找不到來源定位：" + anchor)
            item["excerpts"].append({"locator": anchor, "text": normalized[max(0, start-before):start+after]})
    if line_terms:
        lines = text.splitlines()
        for term, before, after in line_terms:
            indexes = [i for i, line in enumerate(lines) if term in line]
            if not indexes:
                raise ValueError("找不到原始碼定位：" + term)
            for i in indexes:
                lo, hi = max(0, i-before), min(len(lines), i+after)
                item["excerpts"].append({"locator": term, "first_line": lo+1, "last_line": hi, "text": "\n".join(lines[lo:hi])})
    if local:
        installed = Path(local).read_bytes()
        item["installed_source_sha256"] = hashlib.sha256(installed).hexdigest()
        item["matches_installed_source"] = raw == installed
        assert raw == installed
    return item


import torch.autograd.profiler_util
items = [
    fetch("profiler_util", BASE + "torch/autograd/profiler_util.py", COMMIT,
          line_terms=[("def key_averages(", 0, 12), ("key = [", 0, 13), ("].add(evt)", 4, 1),
                      ("def cpu_time(self):", 0, 2), ("return self.cpu_time_total - sum(", 3, 4),
                      ("sum_self_cpu_time_total +=", 2, 1), ("evt.self_cpu_percent =", 0, 8),
                      ("if event_limit == row_limit:", 0, 3),
                      ("key=lambda evt: getattr(", 2, 10)],
          local=torch.autograd.profiler_util.__file__),
    fetch("linear", BASE + "aten/src/ATen/native/Linear.cpp", COMMIT,
          line_terms=[("return at::addmm(bias, input_flattened, weight.t());", 2, 1),
                      ("if (input_dim == 2 && bias->defined())", 0, 4),
                      ("if (is_bias_likely_fusable && !input.is_xla())", 0, 6),
                      ("auto output = at::matmul(input, weight.t());", 0, 2)]),
    fetch("embedding", BASE + "aten/src/ATen/native/Embedding.cpp", COMMIT,
          line_terms=[("if (indices.dim() == 1)", 0, 3),
                      ("return weight.index_select(0, indices.reshape(-1))", 0, 1)]),
    fetch("profiler_api", "https://docs.pytorch.org/docs/2.14/profiler.html", "PyTorch 2.14 documentation",
          anchors=[("list of activity groups (CPU, CUDA)", 0, 360)]),
    fetch("profiler_overhead", "https://docs.pytorch.org/tutorials/beginner/profiler.html", "PyTorch Tutorials 2.14.0+cu130, accessed page",
          anchors=[("Note that using Profiler incurs some overhead", 0, 170)]),
    fetch("cuda", "https://docs.pytorch.org/docs/2.14/notes/cuda.html", "PyTorch 2.14 documentation",
          anchors=[("Asynchronous execution #", 0, 700), ("A consequence of the asynchronous computation", 0, 380),
                   ("memory_allocated() and max_memory_allocated()", 100, 350)]),
    fetch("statistics", "https://docs.python.org/3.13/library/statistics.html", "Python 3.13 documentation",
          anchors=[("statistics. median ( data )", 0, 800), ("statistics. mean ( data )", 0, 340)]),
    fetch("binary_prefix", "https://physics.nist.gov/cuu/Units/binary.html", "NIST binary multiples page; IEC 60027-2 edition 2 (2000-11) table reproduced",
          anchors=[("one mebibyte", 0, 65)]),
    fetch("tensor_docs", BASE + "torch/_tensor_docs.py", COMMIT,
          line_terms=[("Returns the size in bytes of an individual element.", 0, 6),
                      ("Returns a Tensor of size :attr:`size` filled with ``0``.", 0, 5),
                      ("Returns a Tensor of size :attr:`size` filled with uninitialized data.", 0, 4),
                      ("The returned tensor shares the same data", 0, 4)]),
    fetch("torch_docs", BASE + "torch/_torch_docs.py", COMMIT,
          line_terms=[("Returns the total number of elements in the :attr:`input` tensor.", 0, 3),
                      ("Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.", 0, 3),
                      ("Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.", 0, 4),
                      ("The returned tensor shares the same underlying data with this tensor.", 0, 2),
                      ("Create a view of an existing `torch.Tensor`", 0, 5)]),
]
with (OUT / "original-source-excerpts.json").open("x", encoding="utf-8") as stream:
    json.dump({"retrieval_method": "curl HTTPS；全文僅在程序記憶體，檔案只含摘錄與原檔雜湊。", "sources": items}, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"sources_checked": len(items), "profiler_source_matches_installed": items[0]["matches_installed_source"]}, ensure_ascii=False))
