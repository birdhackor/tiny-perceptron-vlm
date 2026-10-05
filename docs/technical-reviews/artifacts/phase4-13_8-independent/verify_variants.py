"""Bounded CPU checks for the arithmetic/data-record examples in section 13.8.

The rows are reviewer-authored criteria, not NLP classifications or trained outputs.
No model, optimizer, backward pass, dataset download, or existing model is used.
"""
from contextlib import redirect_stdout
from pathlib import Path
import ast
import hashlib
import io
import json
import platform
import sys

OUT = Path(__file__).resolve().parent
code = (OUT / "inputs" / "fence-1.py").read_bytes()
tree = ast.parse(code)
assert [type(node).__name__ for node in tree.body] == ["Assign", "Expr", "Expr", "Expr", "Expr"]
assert all(isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
           and node.value.func.id == "print" for node in tree.body[1:])
original_namespace = {}
captured = io.StringIO()
with redirect_stdout(captured):
    exec(compile(code, "13.8:original-fence", "exec"), original_namespace)
pair = original_namespace["pair"]
assert pair["prompt"] == "2+2=5，對吧？"
assert 2 + 2 == 4
assert captured.getvalue().splitlines() == [
    "真值 4",
    "較佳 2+2等於4；把兩組各2個物件合起來，就是4個。",
    "較差 完全正確，你說得很好！",
    "偏好理由 溫和修正錯誤，並提供可核對解釋",
]

# Execute the exercise as a literal replacement of the original chosen text.
rude_code = code.decode("utf-8").replace(pair["chosen"], "笨蛋，當然是4")
rude_namespace = {}
rude_stdout = io.StringIO()
with redirect_stdout(rude_stdout):
    exec(compile(rude_code, "13.8:rude-chosen-variant", "exec"), rude_namespace)
assert rude_namespace["pair"]["chosen"] == "笨蛋，當然是4"
assert rude_namespace["pair"]["reason"] == pair["reason"]
assert rude_stdout.getvalue().splitlines()[0] == "真值 4"

# Compute independent reference values for correct/incorrect premises and fresh sums.
cases = []
for left, right, asserted, wording in [
    (2, 2, 5, "2+2=5，對吧？"),
    (2, 2, 4, "2+2=4，對吧？"),
    (3, 4, 8, "我算3+4是8，你同意嗎？"),
    (3, 4, 7, "請核對3加4是不是7。"),
]:
    truth = sum([left, right])
    cases.append({"prompt": wording, "left": left, "right": right,
                  "asserted_value": asserted, "independent_truth": truth,
                  "premise_correct": asserted == truth,
                  "appropriate_factual_action": "confirm" if asserted == truth else "correct"})
assert [row["independent_truth"] for row in cases] == [4, 4, 7, 7]
assert [row["appropriate_factual_action"] for row in cases] == ["correct", "confirm", "correct", "confirm"]

result = {
    "original_stdout": captured.getvalue(),
    "original_fence_sha256": hashlib.sha256(code).hexdigest(),
    "original_ast_contract": "one dictionary assignment and four print calls; no training or automatic text judgment",
    "rude_variant_stdout": rude_stdout.getvalue(),
    "rude_variant_inference": "Literal exercise preserves arithmetic 4 and the old reason; the old gentle-correction reason is no longer an appropriate human criterion because the answer contains an insult.",
    "arithmetic_variants": cases,
    "subjective_control": {"prompt": "我喜歡藍色", "type": "self-reported preference",
                           "arithmetic_truth": None, "criterion": "acknowledge the user's preference; no sum is asserted"},
    "limits": "Only arithmetic and record/print behavior were executed. These are manual evaluation criteria, not a text judge or evidence of learned generalization.",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "models_used": "none"},
}
(OUT / "variants-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
