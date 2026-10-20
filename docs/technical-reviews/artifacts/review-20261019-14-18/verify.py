"""核對假設參數算例，取得原始來源並只保存小段摘錄與收據。"""

import hashlib
import io
import json
import platform
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


out = Path(__file__).resolve().parent
root = out.parents[3]
bundle = json.loads((out / "section-execution.json").read_text())
section = bundle["files"]["section.md"]["content"]
code = bundle["files"]["fence-1.py"]["content"]
capture = io.StringIO()
namespace = {}
with redirect_stdout(capture):
    exec(compile(code, "frozen-section.py", "exec"), namespace)

rows = []
for name, copies, rounds in namespace["designs"]:
    rows.append({"name": name, "copies": copies, "rounds": rounds,
                 "parameters": 200_000 + copies * 1_000_000})
expected = [(1_200_000, 1), (1_200_000, 5), (5_200_000, 5)]
assert [(r["parameters"], r["rounds"]) for r in rows] == expected
assert capture.getvalue() == bundle["files"]["stdout.txt"]["content"]
assert 200_000 == 200000
exercise = {"copies": 2, "rounds": 2, "parameters": 200_000 + 2 * 1_000_000}
assert exercise["parameters"] == 2_200_000
matches = [{"name": r["name"], "same_parameters": r["parameters"] == exercise["parameters"],
            "same_block_work": r["rounds"] == exercise["rounds"]} for r in rows]
assert all(not m["same_parameters"] and not m["same_block_work"] for m in matches)

prerequisites = {}
for file_name, heading in [
    ("course/chapters/14.md", "同一個區塊多跑幾次，省下的是什麼？"),
    ("course/chapters/16.md", "怎麼從一種運算的耗時找下一個瓶頸？"),
    ("course/chapters/19.md", "什麼證據才足以把候選版叫成品？"),
]:
    raw = (root / file_name).read_bytes()
    text = raw.decode()
    headings = list(re.finditer(r"(?m)^## [^\n]+", text))
    current = next(i for i, h in enumerate(headings) if heading in h.group())
    start = headings[current].start()
    end = headings[current + 1].start() if current + 1 < len(headings) else len(text)
    body = text[start:end]
    selected = [line for line in body.splitlines() if any(word in line for word in [
        "假設字的embedding", "GPU收到工作", "先同步", "要交付成品", "若新成品沒有達主要門檻",
    ])]
    prerequisites[file_name] = {"file_sha256": digest(raw), "heading": heading,
                               "section_sha256": digest(body.encode()), "selected_lines": selected}

environment = {"python": sys.version, "torch": str(torch.__version__),
               "platform": platform.platform(), "device": "cpu",
               "cuda_available": str(torch.cuda.is_available()),
               "mps_available": str(torch.backends.mps.is_available())}
verification = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(root)),
    "environment": environment,
    "source_sha256": digest(section.encode()),
    "rows": rows,
    "stdout": capture.getvalue(),
    "tuple_unpacking": {"variables": ["name", "copies", "rounds"], "items_per_design": 3},
    "integer_separator_equality": 200_000 == 200000,
    "fixed_parameter_pair": [0, 1],
    "fixed_block_execution_pair": [1, 2],
    "exercise": exercise,
    "exercise_matches": matches,
    "equal_update_count_example": {"updates": 10, "one_round_block_executions": 10,
                                   "five_round_block_executions": 50,
                                   "scope": "只算區塊前向執行次數；不是全部訓練 FLOPs 或秒數。"},
    "prerequisites": prerequisites,
    "result": "原始程式輸出與工具執行一致；參數、執行次數與新增設計的比較均精確相等。",
}
with (out / "verification.json").open("x") as handle:
    json.dump(verification, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

specs = [
    {
        "name": "recurrent-depth",
        "url": "https://arxiv.org/html/2502.05171v2",
        "version": "arXiv:2502.05171v2，2025-02-17",
        "quote": "Recurrent-depth networks perform more FLOPs per parameter than standard transformers",
        "needles": ["The core block is set between", "Training Recurrent Models through Unrolling.",
                    "Due to the weight sharing inherent", "single large scale model run"],
    },
    {
        "name": "benchmark-variance",
        "url": "https://arxiv.org/html/2103.03098v1",
        "version": "arXiv:2103.03098v1，2021-03-01",
        "quote": "uncontrolled factors can impact these measures and dominate the meaningful difference between the methods.",
        "needles": ["The performance measure", "Our recommendations: good benchmarks with a budget",
                    "In cases where models are given", "To pair the learning algorithms"],
    },
    {
        "name": "pytorch-cuda",
        "url": "https://docs.pytorch.org/docs/2.14/notes/cuda.html",
        "version": "PyTorch 2.14 CUDA semantics；本機為 2.14.1",
        "quote": "By default, GPU operations are asynchronous.",
        "needles": ["Asynchronous execution", "time measurements without synchronizations are not accurate"],
    },
    {
        "name": "pytorch-benchmark",
        "url": "https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html",
        "version": "PyTorch Tutorials 2.14.0+cu130；讀取日頁面與完整回應 SHA-256 固定於本收據",
        "quote": "This is why it’s important to do a warm-up run before benchmarking",
        "needles": ["This is why it’s important to do a warm-up run", "The difference in the results between timeit",
                    "takes many measurements", "for different input sizes"],
    },
    {
        "name": "nvidia-performance",
        "url": "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html",
        "version": "GPU Performance Background User’s Guide，Last updated on Feb 1, 2023",
        "quote": "Performance of a function on a given processor is limited by one of the following three factors; memory bandwidth, math bandwidth and latency.",
        "needles": ["Each multiply-add comprises two operations", "Performance of a function on a given processor",
                    "The left-hand side, the ratio"],
    },
]
receipts = []
for spec in specs:
    command = ["curl", "-fsSL", "--max-time", "40", spec["url"]]
    done = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    parser = TextOnly()
    parser.feed(done.stdout.decode())
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    assert spec["quote"] in text, spec["name"]
    quote = spec["quote"]
    assert len(quote.split()) <= 25, spec["name"]
    locators = []
    for needle in spec["needles"]:
        positions = [m.start() for m in re.finditer(re.escape(needle), text)]
        assert positions, (spec["name"], needle)
        position = positions[-1] if len(positions) > 1 else positions[0]
        locators.append({"anchor_text": needle, "text_character_offset": position})
        print(json.dumps({"source": spec["name"], "anchor": needle,
                          "read_text": text[max(0, position - 100):position + 1800]}, ensure_ascii=False))
    receipt = {"name": spec["name"], "url": spec["url"], "version": spec["version"],
               "command": shlex.join(command), "exit_code": done.returncode,
               "accessed_at_utc": datetime.now(UTC).isoformat(),
               "response_sha256": digest(done.stdout), "response_bytes": len(done.stdout),
               "excerpt": quote, "locators": locators,
               "storage": "原文只在記憶體解析；只保存支持主張的小段摘錄與取得收據。"}
    receipts.append(receipt)
with (out / "source-receipts.json").open("x") as handle:
    json.dump(receipts, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"numeric_verification": "pass", "sources_acquired": len(receipts)}, ensure_ascii=False))
