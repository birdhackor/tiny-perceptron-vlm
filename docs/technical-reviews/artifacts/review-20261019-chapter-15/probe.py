"""核對 MoE 導言的固定算例、連結與函式庫資料流。"""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.modern import MoEFFN


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def section(chapter, title):
    headings = list(re.finditer(r"^## .+$", chapter, re.MULTILINE))
    for index, heading in enumerate(headings):
        if heading.group().endswith(title):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(chapter)
            return chapter[heading.start():end]
    raise AssertionError("找不到指定標題：" + title)


def first_example(chapter, title):
    body = section(chapter, title)
    code = re.search(r"```python\n(.*?)\n```", body, re.DOTALL).group(1)
    captured = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(captured):
        exec(compile(code, title, "exec"), namespace)
    return {"title": title, "stdout": captured.getvalue()}, namespace


def main():
    torch.set_num_threads(1)
    torch.manual_seed(731)
    chapter_path = "course/chapters/15.md"
    snapshot_path = "docs/course-review-20261019/freeze-01/sources/chapter-15.md"
    freeze_path = "docs/course-review-20261019/freeze-04.json"
    chapter = (ROOT / chapter_path).read_text()
    snapshot = (ROOT / snapshot_path).read_text()
    assert chapter.split("## ", 1)[0] == snapshot, "導言與凍結原文不同"
    freeze = json.loads((ROOT / freeze_path).read_bytes())
    page = next(page for page in freeze["pages"] if page["page_id"] == "chapter-15")
    assert page["source_sha256"] == digest(snapshot_path), "凍結原文雜湊不符"
    assert page["figures_sha256"] == {}, "導言出現未預期的圖"

    links = []
    for target in re.findall(r"\]\(([^)]+)\)", snapshot):
        relative, _, fragment = target.partition("#")
        path = (ROOT / chapter_path).parent / relative
        assert path.is_file(), "連結目標不存在：" + relative
        if fragment:
            assert re.search(r"^## " + re.escape(fragment) + r" ", path.read_text(), re.MULTILINE)
        links.append({"target": target, "exists": True, "fragment_exists": bool(fragment)})

    examples = []
    record, namespace = first_example(chapter, "能只啟動少數 experts 嗎？")
    probability = namespace["prob"]
    selected = {}
    for k, expected in ((2, [[1, 2], [0, 1]]), (1, [[1], [0]])):
        actual = probability.topk(k, dim=-1).indices.tolist()
        assert actual == expected, "固定機率的 expert 選擇不符"
        selected[str(k)] = actual
    record["checked_choices"] = selected
    examples.append(record)

    record, namespace = first_example(chapter, "Token 怎麼送出再排回來？")
    assert namespace["result"].tolist() == [[2.0, 2.0], [0.0, 0.0], [4.0, 4.0]], "合併沒有恢復來源列"
    examples.append(record)

    record, _ = first_example(chapter, "Router 怎麼收到學習訊號？")
    assert "1.3333" in record["stdout"] and "1.185" in record["stdout"], "router 梯度示範不符"
    examples.append(record)

    walkthrough = section(chapter, "少算一些就一定更快嗎？")
    code_blocks = re.findall(r"```python\n(.*?)\n```", walkthrough, re.DOTALL)
    class_code = next(code for code in code_blocks if "class MoEFFN" in code)
    source_class = ast.parse(inspect.getsource(MoEFFN)).body[0]
    documented_class = ast.parse(class_code).body[0]
    assert ast.dump(source_class) == ast.dump(documented_class), "課文節錄與函式庫類別不同"

    layer = MoEFFN(2, experts=5, top_k=2, hidden=3).double()
    with torch.no_grad():
        layer.router.weight.copy_(torch.tensor(
            [[0.0, 0.0], [math.log(7), 0.0], [math.log(2), 0.0], [-4.0, 0.0], [-1.0, 0.0]],
            dtype=torch.float64,
        ))
    x = torch.tensor([[[1.0, 2.0], [-1.0, 2.0], [0.5, -1.0]]], dtype=torch.float64)
    runs = []
    for k in (1, 2):
        layer.top_k = k
        layer.zero_grad(set_to_none=True)
        recorded_inputs = {}

        def capture(index):
            def hook(module, inputs, output):
                recorded_inputs[index] = inputs[0].detach().clone()
            return hook

        hooks = [expert.register_forward_hook(capture(index)) for index, expert in enumerate(layer.experts)]
        output, auxiliary, chosen = layer(x)
        for hook in hooks:
            hook.remove()
        flat = x.reshape(-1, x.shape[-1])
        scores = flat @ layer.router.weight.T
        probabilities = scores.softmax(-1)
        order = scores.argsort(-1, descending=True)[:, :k]
        assert torch.equal(chosen, order), "選路與獨立排序不符"
        reference_rows = []
        for row, token in enumerate(flat):
            weights = probabilities[row, order[row]]
            if k > 1:
                weights = weights / weights.sum()
            reference_rows.append(sum(
                weights[slot] * layer.experts[int(index)](token)
                for slot, index in enumerate(order[row])
            ))
        reference = torch.stack(reference_rows).view_as(x)
        error = (output - reference).abs().max().item()
        assert error < 1e-12, "稀疏結果與逐 token 加權和不符"
        expected_counts = torch.bincount(chosen.flatten(), minlength=len(layer.experts))
        actual_counts = [len(recorded_inputs[index]) if index in recorded_inputs else 0 for index in range(len(layer.experts))]
        assert actual_counts == expected_counts.tolist(), "expert 的實際輸入筆數與選擇不符"
        for index, actual in recorded_inputs.items():
            expected_rows = torch.where(chosen == index)[0]
            assert torch.equal(actual, flat[expected_rows]), "expert 收到不屬於自己的 token"
        load = expected_counts.to(torch.float64) / chosen.numel()
        expected_auxiliary = len(layer.experts) * (load * probabilities.mean(0)).sum()
        assert torch.allclose(auxiliary, expected_auxiliary, atol=1e-7, rtol=0), "輔助代價公式不符"
        output.square().mean().backward()
        task_gradient = layer.router.weight.grad.norm().item()
        assert task_gradient > 0 and math.isfinite(task_gradient), "router 沒有有限的任務梯度"
        layer.zero_grad(set_to_none=True)
        _, auxiliary_only, _ = layer(x)
        auxiliary_only.backward()
        auxiliary_gradient = layer.router.weight.grad.norm().item()
        assert auxiliary_gradient > 0 and math.isfinite(auxiliary_gradient), "router 沒有有限的輔助梯度"
        runs.append({
            "top_k": k,
            "chosen": chosen.tolist(),
            "actual_expert_token_counts": actual_counts,
            "uninvoked_experts": [index for index in range(len(layer.experts)) if index not in recorded_inputs],
            "max_absolute_reference_error": error,
            "auxiliary": auxiliary.item(),
            "independent_auxiliary": expected_auxiliary.item(),
            "task_router_gradient_norm": task_gradient,
            "auxiliary_router_gradient_norm": auxiliary_gradient,
            "output_shape": list(output.shape),
        })

    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-15/probe.py",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "module_dtype": "torch.float64",
            "example_dtype": "torch.float32",
            "threads": str(torch.get_num_threads()),
            "seed": "731",
        },
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": {path: digest(path) for path in (chapter_path, snapshot_path, freeze_path, "tiny_perceptron/modern.py", "course/glossary.md")},
        "snapshot_matches_live_introduction": True,
        "frozen_figure_count": len(page["figures_sha256"]),
        "links": links,
        "examples": examples,
        "documented_class_matches_library_ast": True,
        "moe_runs": runs,
        "result": "全部檢查通過",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
