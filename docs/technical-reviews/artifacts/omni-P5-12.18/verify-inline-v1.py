"""12.18 reviewer: exact frozen artificial CPU example and transparent probes.

No model loading, inference, training, tuning, workflow or publishing is performed.
Run from the repository root with the task's CPU Python environment.
"""

from contextlib import redirect_stdout
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import platform
import re

import torch


ROOT = Path.cwd()
BODY_PATH = Path(
    "docs/course-revision-20261029-omni/reviews/P5-actual/"
    "source-freeze-v1/sources/12.18.md"
)
BODY_SHA = "a52a7f88a6a1e789b0e3b316363c716d8da529f82b1007de3d0b67e4ab6c81f4"
body_bytes = BODY_PATH.read_bytes()
assert hashlib.sha256(body_bytes).hexdigest() == BODY_SHA
body = body_bytes.decode("utf-8")
code = re.findall(r"```python\n(.*?)\n```", body, re.DOTALL)
assert len(code) == 1
inline = code[0]
nb_path = Path("notebooks/12/12.18.ipynb")
nb_bytes = nb_path.read_bytes()
notebook = json.loads(nb_bytes)
nb_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
assert nb_cells[-1].rstrip("\n") == inline.rstrip("\n")
print("ENVIRONMENT", json.dumps({
    "python": platform.python_version(), "torch": torch.__version__,
    "torch_git": torch.version.git_version, "device": "cpu",
    "platform": platform.platform(), "default_dtype": str(torch.get_default_dtype()),
}, ensure_ascii=False))
print("SOURCE", json.dumps({
    "snapshot": str(BODY_PATH), "bytes": len(body_bytes), "sha256": BODY_SHA,
    "notebook": str(nb_path), "notebook_sha256": hashlib.sha256(nb_bytes).hexdigest(),
    "inline_equals_notebook_main_cell": True,
}, ensure_ascii=False))


def run_example(label, source):
    namespace = {}
    capture = io.StringIO()
    with redirect_stdout(capture):
        exec(compile(source, str(BODY_PATH) + ":" + label, "exec"), namespace)
    print("BEGIN", label)
    print(capture.getvalue(), end="")
    print("SHAPES", json.dumps({
        "waveform": list(namespace["waveform"].shape),
        "blocks": list(namespace["blocks"].shape),
        "broadcast_difference": list((namespace["blocks"][:, None, :] - namespace["codebook"][None, :, :]).shape),
        "distances": list(namespace["distances"].shape),
        "codes": list(namespace["codes"].shape),
        "reconstruction": list(namespace["reconstruction"].shape),
        "dtype": str(namespace["waveform"].dtype),
        "input_requires_grad": namespace["waveform"].requires_grad,
        "codebook_requires_grad": namespace["codebook"].requires_grad,
    }))
    print("MSE_FULL_PRECISION", repr(namespace["mse"].item()))
    print("END", label)
    return namespace


main = run_example("exact_frozen_inline", inline)
expected_distances = torch.tensor([
    [1.62, 0.02, 3.62], [1.28, 3.28, 0.08], [0.04, 2.04, 2.04]
], dtype=torch.float64)
assert torch.allclose(main["distances"], expected_distances, atol=1e-14, rtol=0)
assert main["codes"].tolist() == [1, 2, 0]
assert main["reconstruction"].tolist() == [1., 0., -1., 0., 0., 1., 0., -1., 0., 0., 0., 0.]
assert abs(main["mse"].item() - 0.14 / 12) < 1e-14
assert len(main["waveform"]) / main["sample_rate"] * 1000 == 0.75

exercise_line = "waveform[:4] = torch.tensor([0.4,0,-0.4,0])\n"
exercise = run_example("exact_exercise_assignment", inline.replace("blocks = waveform.reshape", exercise_line + "blocks = waveform.reshape"))
assert exercise["codes"].tolist() == [0, 2, 0]
assert round(exercise["mse"].item(), 6) == 0.036667
assert torch.allclose(exercise["distances"][0], torch.tensor([.32, .72, 2.32], dtype=torch.float64), atol=2e-8, rtol=0)
print("EXERCISE_FLOAT32_ASSIGNMENT_VALUES", exercise["waveform"][:4].tolist())
print("EXACT_DECIMAL_DERIVATION", json.dumps({
    "main_sse": str(Decimal(".02") + Decimal(".08") + Decimal(".04")),
    "main_mse": str(Decimal(".14") / 12),
    "exercise_sse": str(Decimal(".32") + Decimal(".08") + Decimal(".04")),
    "exercise_mse": str(Decimal(".44") / 12),
    "duration_ms": str(Decimal(12) / 16000 * 1000),
}))

book = main["codebook"]
tie_block = torch.tensor([[.5, 0., -.5, 0.]], dtype=torch.float64)
tie_distances = (tie_block[:, None, :] - book[None, :, :]).square().sum(-1)
assert tie_distances.tolist() == [[.5, .5, 2.5]]
assert tie_distances.argmin(1).tolist() == [0]
print("EXACT_TIE", tie_distances.tolist(), "codes", tie_distances.argmin(1).tolist())
collision_blocks = torch.tensor([[.89, 0., -.89, 0.], [.91, 0., -.91, 0.]], dtype=torch.float64)
collision_ids = (collision_blocks[:, None, :] - book[None, :, :]).square().sum(-1).argmin(1)
assert collision_ids.tolist() == [1, 1]
print("MANY_TO_ONE", collision_blocks.tolist(), collision_ids.tolist(), book[collision_ids].tolist())
try:
    exec(inline.replace("block_size = codebook.shape[1]", "waveform = waveform[:-1]\nblock_size = codebook.shape[1]"), {})
except ValueError as exc:
    assert str(exc) == "本例只接受完整的四點區塊"
    print("INCOMPLETE_BLOCK_GUARD", type(exc).__name__, str(exc))
else:
    raise AssertionError("incomplete waveform did not raise")
print("RESULT all independent artificial CPU checks passed")
