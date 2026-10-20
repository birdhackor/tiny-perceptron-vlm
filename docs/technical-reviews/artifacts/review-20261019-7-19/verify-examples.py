"""核對凍結算例、練習、圖表位置與連結到的監督安排。"""

import ast
import contextlib
import hashlib
import io
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, render_chat


def execute(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "凍結範例", "exec"), namespace)
    return namespace, output.getvalue()


code = (BASE / "section-fence-1.py").read_text()
namespace, original_output = execute(code)
prompt = namespace["prompt"]
assert len(prompt) == 12
assert prompt[-1] == "〈assistant〉"
assert [len(prompt[:3]), len(prompt[3:6]), len(prompt[6:11]), len(prompt[11:])] == [3, 3, 5, 1]
assert prompt.count("〈eos〉") == 3
assert "〈pad〉" not in prompt
expected_output = (
    "前文長度P 12 還能生成的位置T-P 4\n"
    "G=4：新增3個token ['小小', '書店', '〈eos〉']，停止原因：模型選了EOS\n"
    "G=2：新增2個token ['小小', '書店']，停止原因：用完G\n"
    "G=5：P+G=17 超過 T=16，拒絕這個設定\n"
)
assert original_output == expected_output


def candidates_variant(tokens):
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "candidates" for t in node.targets):
            node.value = ast.List(elts=[ast.Constant(value=t) for t in tokens], ctx=ast.Load())
    ast.fix_missing_locations(tree)
    values, output = execute(tree)
    assert values["prompt"] == prompt
    return output


exercise = candidates_variant(["小小", "〈eos〉"])
assert "G=4：新增2個token ['小小', '〈eos〉']，停止原因：模型選了EOS" in exercise
immediate = candidates_variant(["〈eos〉"])
assert "G=4：新增1個token ['〈eos〉']，停止原因：模型選了EOS" in immediate

tokenizer = ByteTokenizer()
supervision = []
for question, answer in [("問", "答"), ("Q", "A")]:
    x, y = render_chat([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    first = (y != -100).nonzero()[0].item()
    assert x[first].item() == tokenizer.assistant_id
    assert y[first].item() == tokenizer.encode(answer)[0]
    assert y[-1].item() == tokenizer.eos_id
    assert tokenizer.decode(tokenizer.encode(answer) + [tokenizer.eos_id]) == answer
    supervision.append({"question": question, "answer": answer, "x": x.tolist(), "y": y.tolist(), "first_target_position": first})

figure = ROOT / "course/figures/new-7.19-sequence-budget.svg"
svg = ET.fromstring(figure.read_bytes())
ns = {"svg": "http://www.w3.org/2000/svg"}
symbols = [element.text for element in svg.findall("svg:g[@class='symbol']/svg:text", ns)]
assert symbols == ["U", "a", "E", "A", "b", "E", "U", "c", "d", "e", "E", "A", "f", "g", "E"]
input_count = len(svg.findall("svg:g[@class='input']/svg:rect", ns))
output_count = len(svg.findall("svg:g[@class='output']/svg:rect", ns))
assert (input_count, output_count) == (12, 3)
positions = [int(element.text) for element in svg.findall("svg:g[@class='position']/svg:text", ns)]
assert positions == list(range(1, 17))
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
record = next(record for record in receipt["records"] if record["page_id"] == "7.19")
png = ROOT / "outputs/course-review-20261019/figure-renders-04/de421ad84cb7cb72c620ea1dedf4c2e2b86fce8874e807ecc9755b46ebf60e19.png"
png_sha = hashlib.sha256(png.read_bytes()).hexdigest()
assert record["png"]["sha256"] == png_sha

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-19/verify-examples.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
    "original_stdout": original_output,
    "exercise_stdout": exercise,
    "immediate_eos_stdout": immediate,
    "table_lengths": [3, 3, 5, 1],
    "prompt_tokens": prompt,
    "unused_budget": {"G": 4, "generated_including_eos": 3, "unused": 1, "pad_inserted": False},
    "exercise_visible_answer": "小小",
    "exercise_complete_bookstore_name": False,
    "linked_supervision": supervision,
    "figure": {"symbols": symbols, "input_cells": input_count, "output_cells": output_count, "positions": positions, "png": str(png.relative_to(ROOT)), "png_sha256": png_sha, "receipt": record},
    "checks_passed": True,
}
output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
with (BASE / "verification-output.json").open("x", encoding="utf-8") as handle:
    handle.write(output)
print(output, end="")
