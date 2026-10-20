"""核對導言所指的實作與成本邊界，保存小份原始來源收據。"""

import ast
import hashlib
import html
import inspect
import json
import platform
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch

from tiny_perceptron.attention import attention_mask
from tiny_perceptron.lesson_products import inference
from tiny_perceptron.model import Block, ModelConfig, TinyLM


ROOT = Path.cwd()
ARTIFACT = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(path, heading):
    raw = (ROOT / path).read_text()
    match = re.search(r"^## " + re.escape(heading) + r" .*?$", raw, re.M)
    assert match, "找不到指定標題"
    following = re.search(r"^## ", raw[match.end():], re.M)
    end = match.end() + following.start() if following else len(raw)
    return raw[match.start():end]


PRIMARY = [
    ("flash", "https://arxiv.org/abs/2205.14135v2", "arXiv:2205.14135v2，Abstract",
     "an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes"),
    ("speculative", "https://proceedings.mlr.press/v202/leviathan23a.html", "ICML 2023，PMLR 202，Abstract",
     "by running them in parallel on the outputs of the approximation models, potentially generating several tokens concurrently"),
    ("recurrence", "https://arxiv.org/abs/2502.05171v2", "arXiv:2502.05171v2，Abstract",
     "Our model works by iterating a recurrent block, thereby unrolling to arbitrary depth at test-time."),
    ("checkpoint", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/utils/checkpoint.py", "PyTorch v2.14.1，checkpoint docstring",
     "Activation checkpointing is a technique that trades compute for memory."),
    ("compile", "https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html", "即時官方教學，Demonstrating Speedups；以回應 SHA-256 識別版本",
     "torch.compile takes extra time to compile the model on the first few executions."),
    ("benchmark", "https://docs.pytorch.org/docs/2.14/benchmark_utils.html", "PyTorch 2.14，Timer，Runtime aware",
     "Timer will perform warmups"),
    ("effort", "https://platform.claude.com/docs/en/build-with-claude/effort", "即時官方文件，How effort works；以回應 SHA-256 識別版本",
     "The effort parameter lets you control how many tokens Claude spends when responding to requests."),
    ("sdpa", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html", "PyTorch 2.14，scaled_dot_product_attention",
     "A boolean mask where a value of True indicates that the element should take part in attention."),
]


def fetch(item):
    identifier, url, version, excerpt = item
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "40", url,
               "--write-out", "\n__FETCH_META__%{http_code} %{url_effective}"]
    run = subprocess.run(command, capture_output=True, check=False)
    assert run.returncode == 0, run.stderr.decode()
    raw, meta = run.stdout.rsplit(b"\n__FETCH_META__", 1)
    text = raw.decode(errors="replace")
    if identifier != "checkpoint":
        text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.S)
        text = html.unescape(re.sub(r"<[^>]*>", " ", text))
    text = re.sub(r"\s+", " ", text)
    assert excerpt in text, "下載的原始來源沒有預期摘錄：" + identifier
    return {
        "id": identifier, "url": url, "version": version,
        "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
        "command": shlex.join(command), "exit_code": run.returncode,
        "http_status_and_effective_url": meta.decode(),
        "response_sha256": digest(raw), "response_bytes": len(raw),
        "excerpt": excerpt,
        "storage": "只保存支持主張的短摘錄；第三方全文僅在記憶體讀取，未寫入 repo。",
    }


def verify():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    checks = {}
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/chapter-16.md"
    raw = frozen.read_bytes()
    checks["freeze"] = {
        "source_sha256": digest(raw),
        "freeze_sha256": digest((ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes()),
        "current_intro_equal": (ROOT / "course/chapters/16.md").read_bytes().split(b"## ", 1)[0] == raw,
        "image_references": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", raw.decode()),
    }
    assert checks["freeze"]["source_sha256"] == "33bfbdf79416bc8523c5680eb3863ec658d6db65c132adaba2875f4790379c22"
    assert checks["freeze"]["current_intro_equal"]
    body = section("course/chapters/16.md", "16.5")
    snippets = re.findall(r"```python\n(.*?)```", body, re.S)
    snippet = next(code for code in snippets if "def attention_mask(" in code)
    assert ast.dump(ast.parse(snippet)) == ast.dump(ast.parse(inspect.getsource(attention_mask)))
    positions = torch.arange(4)
    segments = torch.tensor([[0, 0, 1, 1]])
    packed = attention_mask(positions, positions, segments=segments)
    expected = [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
    assert packed[0, 0].int().tolist() == expected
    checks["attention_mask"] = {"excerpt_ast_equal": True, "shape": list(packed.shape), "allowed": expected}
    ids = torch.tensor([[1, 2, 3, 4]])
    caches = {}
    with torch.inference_mode():
        for kv_heads in (4, 1):
            model = TinyLM(ModelConfig(width=16, heads=4, kv_heads=kv_heads)).eval()
            full = model(ids)["logits"][:, -1]
            prefix = model(ids[:, :3])
            step = model(ids[:, 3:], cache=prefix["cache"])
            error = float((full - step["logits"][:, 0]).abs().max())
            assert torch.allclose(full, step["logits"][:, 0], atol=1e-6)
            assert step["cache"][0][0].shape[2] == 4
            assert torch.equal(prefix["cache"][0][0], step["cache"][0][0][:, :, :3])
            caches[str(kv_heads)] = {
                "prefix_k_shape": list(prefix["cache"][0][0].shape),
                "appended_k_shape": list(step["cache"][0][0].shape),
                "prefix_bytes": sum(t.numel() * t.element_size() for pair in prefix["cache"] for t in pair),
                "appended_bytes": sum(t.numel() * t.element_size() for pair in step["cache"] for t in pair),
                "max_abs_logit_error": error, "old_k_preserved": True,
            }
        manual = TinyLM(ModelConfig(width=16, heads=4)).eval()
        sdpa = TinyLM(ModelConfig(width=16, heads=4, backend="sdpa")).eval()
        sdpa.load_state_dict(manual.state_dict())
        original = torch.nn.functional.scaled_dot_product_attention
        with patch("tiny_perceptron.attention.F.scaled_dot_product_attention", wraps=original) as call:
            actual = sdpa(ids)["logits"]
            assert call.call_count == 1
            passed = call.call_args.kwargs
        baseline = manual(ids)["logits"]
        assert torch.allclose(actual, baseline, atol=1e-6)
    checks["cache"] = caches
    checks["sdpa"] = {
        "calls": call.call_count, "boolean_mask_true_means_allowed": passed["attn_mask"].dtype == torch.bool,
        "dropout_p": passed["dropout_p"], "max_abs_logit_error": float((actual - baseline).abs().max()),
        "device": "cpu", "dtype": "torch.float32", "input_ids": ids.tolist(),
    }
    block = Block(ModelConfig(width=8)).eval()
    counts = {}
    with torch.inference_mode():
        for loops in (1, 3, 5):
            called = []
            handle = block.register_forward_hook(lambda *args: called.append(True))
            state = torch.randn(1, 3, 8)
            for _ in range(loops):
                state, _, _ = block(state)
            handle.remove()
            counts[str(loops)] = {"calls": len(called), "parameters": sum(p.numel() for p in block.parameters())}
            assert len(called) == loops
    checks["recurrent_block"] = counts
    clock = SimpleNamespace(now=0.0)
    events = []

    class Solver:
        auto_policy = {"version": 1, "low_max_events": 0, "medium_max_events": 4, "sum_depth": 1}

        def synchronize(self):
            return None

        def solve(self, state, loops):
            events.append(["solve", loops])
            clock.now += 0.5 * loops
            return {"seconds": 0.5 * loops, "loops": loops}

    class Judge:
        def synchronize(self):
            return None

        def judge(self, state, question):
            seconds = 0.125 if question == "choice" else 0.25
            clock.now += seconds
            events.append(["judge", question])
            return {"selected": "boxes" if question == "choice" else "yes", "seconds": seconds}

    state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
    with patch.object(inference, "time", SimpleNamespace(perf_counter=lambda: clock.now)):
        automatic = inference.auto_solve(Solver(), Judge(), state)
        assert automatic["seconds"] == automatic["c_seconds"] + automatic["solution"]["seconds"] == 1.875
        events.clear()
        low = inference.auto_solve(Solver(), Judge(), state, effort="low")
        high = inference.auto_solve(Solver(), Judge(), state, effort="high")
    assert events == [["judge", "choice"], ["judge", "info"], ["solve", 1],
                      ["judge", "choice"], ["judge", "info"], ["solve", 5]]
    checks["auto_timing"] = {
        "method": "真實 auto_solve 控制流程搭配假判斷器、假解題器與可追蹤邏輯時鐘；不是實測 latency。",
        "automatic": automatic, "low": low, "high": high,
        "low_then_high_events": events, "low_then_high_total_seconds": low["seconds"] + high["seconds"],
    }
    links = {"course/chapters/03.md": ["3.7"], "course/chapters/07.md": ["7.23"],
             "course/chapters/14.md": ["14.11"], "course/chapters/19.md": ["19.17"],
             "course/chapters/16.md": ["16.1", "16.3", "16.4", "16.5", "16.6", "16.7", "16.8", "16.9", "16.10", "16.11", "16.12", "16.13", "16.14"]}
    checks["linked_headings"] = {path: {key: section(path, key).splitlines()[0] for key in keys} for path, keys in links.items()}
    paths = list(links) + ["tiny_perceptron/attention.py", "tiny_perceptron/model.py", "tiny_perceptron/lesson_products/inference.py", "tiny_perceptron/lesson_products/evaluation.py"]
    return {
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-16/verify.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "torch.float32", "platform": platform.platform(), "threads": str(torch.get_num_threads()), "seed": "0"},
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "repository_sha256": {path: digest((ROOT / path).read_bytes()) for path in paths},
        "checks": checks,
        "primary_sources": list(ThreadPoolExecutor(max_workers=4).map(fetch, PRIMARY)),
        "result": "所有實作斷言與短摘錄定位通過；未執行訓練、GPU 計時或外部寫入。",
    }


if __name__ == "__main__":
    result = verify()
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (ARTIFACT / "verification.json").write_text(output)
    sys.stdout.write(output)
