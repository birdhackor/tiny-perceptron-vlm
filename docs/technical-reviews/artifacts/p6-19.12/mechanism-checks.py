"""Bounded CPU checks of existing CTC and tool scoring; no neural generation."""
from pathlib import Path
import importlib.util
import json
import platform
import re
import sys
import torch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.selftrained.model import decode_ocr
from tiny_perceptron.selftrained.tools import run_tool_loop
spec = importlib.util.spec_from_file_location("original_evaluator", ROOT / "scripts/selftrained/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

path = [1, 1, 0, 1, 2, 2, 0]
logits = torch.full((len(path), 13), -10.0)
logits[torch.arange(len(path)), torch.tensor(path)] = 10.0
decoded = decode_ocr(logits)
assert decoded == "大大小"
base = {"task": "tool_call", "messages": [{"role": "user", "content": "2+3"}, {"role": "assistant", "content": "結果是5。"}], "supervision": {"expected_call": {"tool": "calculator", "operation": "add", "a": 2, "b": 3}}}
results = []
for b, reply in [(3, "結果是5。"), (4, "結果是6。")]:
    outputs = iter([json.dumps({"tool": "calculator", "operation": "add", "a": 2, "b": b}), reply])
    trace = run_tool_loop(lambda _: next(outputs), base["messages"][:-1])
    result = ev.score_reply(base, trace["final_output"], trace)
    results.append({"b": b, "executed": trace["executed"], "returned_result": trace["tool_result"]["result"], "score": result})
assert results[0]["score"]["tool_roundtrip"] is True
assert results[1]["executed"] is True and results[1]["score"]["tool_final"] is True and results[1]["score"]["tool_roundtrip"] is False
text = (ROOT / "course/chapters/19.md").read_text()
section = text[text.index("## 19.12 "):]
fence = re.search(r"```python\n(.*?)```", section, re.S).group(1)
namespace = {}
exec(compile(fence, "course/chapters/19.md#19.12", "exec"), {}, namespace)
assert namespace["by_question"] == 2 / 3 and namespace["by_task"] == .5
out = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"}, "ctc": {"path": path, "decoded": decoded}, "tool_checks": results, "actual_section_python_fence": fence, "means": {k: namespace[k] for k in ["by_question", "by_task"]}}
(ART / "mechanism-checks-output.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(out,ensure_ascii=False,indent=2))
