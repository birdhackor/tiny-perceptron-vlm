"""Execute the two necessary prerequisite CPU snippets unchanged."""
import contextlib
import io
import json
import platform
import re
from pathlib import Path

import torch

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
outputs = {}
for key in ("3.3", "14.2"):
    raw = (base / f"{key}.prerequisite.original.md").read_text()
    code = re.search(r"```python\n(.*?)```", raw, re.S).group(1)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, f"prerequisite {key} original snippet", "exec"))
    outputs[key] = printed.getvalue()
    print(key, printed.getvalue())
q = torch.tensor([[1., 0.], [0., 1.]])
k = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
assert (q @ k.T).tolist() == [[1., 0., 1.], [0., 1., 1.]]
k = torch.cat((k, torch.tensor([[2., 0.]])), dim=0)
scores = q @ k.T
assert scores.shape == (2, 4)
assert scores[:, -1].tolist() == [2., 0.]
result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": "1"}, "unchanged_snippet_outputs": outputs, "3.3_exercise_shape": list(scores.shape), "3.3_exercise_scores": scores.tolist()}
(base / "prerequisite-probe-result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(result, indent=2, ensure_ascii=False))
print("NECESSARY PREREQUISITE CPU ASSERTIONS PASSED")
