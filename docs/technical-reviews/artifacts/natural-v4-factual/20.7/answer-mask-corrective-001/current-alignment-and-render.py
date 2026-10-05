"""Current section alignment/figure correction check; bounded CPU only."""
from pathlib import Path
import hashlib
import json
import re
import sys
import torch
from tiny_perceptron.data import render_chat
from playwright.sync_api import sync_playwright

root = Path.cwd()
out = Path(__file__).resolve().parent
section = (out / "current-20.7.md").read_text()
code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
prior = out.parent / "original-example.py"
assert code.encode() == prior.read_bytes()
print(json.dumps({"python": sys.version, "torch": torch.__version__, "device": "cpu",
                  "cuda_available": torch.cuda.is_available(),
                  "current_example_exact_bytes_equal_to_own_prior": True}))
exec(compile(code, "current20.7-example", "exec"))
for answer in ("A", "AB"):
    x, y = render_chat([{"role": "user", "content": "Q"},
                        {"role": "assistant", "content": answer}])
    expected_x = [1, 3, 89, 2, 4, 73] + ([74] if answer == "AB" else [])
    expected_y = [-100] * 4 + [73] + ([74] if answer == "AB" else []) + [2]
    assert x.tolist() == expected_x and y.tolist() == expected_y
    assert x[4].item() == 4 and y[4].item() == 73
    assert int((y != -100).sum()) == len(answer) + 1
    print(json.dumps({"answer": answer, "X": x.tolist(), "Y": y.tolist(),
                      "first_predictor_index": 4, "first_input_id": 4,
                      "first_target_id": 73, "input_question_retained": 89 in x.tolist()}))

notebook = root / "outputs/notebooks/20/20.7.ipynb"
nb = json.loads(notebook.read_text())
match = next(c for c in nb["cells"] if c["cell_type"] == "code" and
             "print(\"計分目標ID\"" in "".join(c["source"]))
assert "".join(match["source"]).strip() == code.strip()
text = "".join("".join(o.get("text", [])) for o in match["outputs"] if o["output_type"] == "stream")
assert text == "輸入位置數 6\n計分目標數 2\n第一個計分目標的位置 4\n計分目標ID [73, 2]\n"
print(json.dumps({"current_native_notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
                  "actual_native_output": text, "execution_count": match["execution_count"]}, ensure_ascii=False))

receipts = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium")
    page = browser.new_page(viewport={"width": 800, "height": 760}, device_scale_factor=1)
    for name in ("natural-v4-answer-mask.svg", "foundations_answer_alignment.svg"):
        source = root / "course/figures" / name
        page.set_content(source.read_text())
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(out / (name + ".png")), full_page=True)
        receipts.append({"source": source.relative_to(root).as_posix(),
                         "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "render": (out / (name + ".png")).relative_to(root).as_posix(),
                         "browser": browser.version, "viewport": {"width": 800, "height": 760}})
    browser.close()
(out / "current-render-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
