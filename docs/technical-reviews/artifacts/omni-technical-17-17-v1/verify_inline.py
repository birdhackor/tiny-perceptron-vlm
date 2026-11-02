"""執行 17.17 fix11 的完整內嵌程式，再做頁內指定的單值練習。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = "docs/course-revision-20261029-omni/reviews/wave-01/source-freeze/sources/17.17.md"
EXPECTED_SHA = "24a05ae3e95a9d119731d4cb6020dc05cdbf45410eb146ce9b3b12d079878dc6"
raw = (ROOT / SNAPSHOT).read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
blocks = re.findall(r"```python\n(.*?)```", raw.decode("utf-8"), re.S)
assert len(blocks) == 1
code = blocks[0]


def run_block(text, value):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(text, SNAPSHOT + ":python", "exec"), namespace)
    assert namespace["value"].item() == torch.tensor(value).item()
    weights = b"".join(
        tensor.detach().numpy().tobytes()
        for module in [namespace["context"], namespace["heads"], namespace["embeddings"]]
        for tensor in module.state_dict().values()
    )
    with torch.no_grad():
        h = torch.tanh(namespace["context"](torch.tensor([0.3, -0.4])))
        stages = []
        for q in range(len(namespace["books"])):
            scores = namespace["heads"][q](h)
            k = scores.argmax(-1)
            stages.append(
                {
                    "codebook": q,
                    "h_shape": list(h.shape),
                    "scores_shape": list(scores.shape),
                    "chosen_id": k.item(),
                    "tanh_bounds_satisfied": bool(((h >= -1) & (h <= 1)).all()),
                }
            )
            h = torch.tanh(h + namespace["embeddings"][q](k))
    return {
        "value": value,
        "tensor_dtype": str(namespace["value"].dtype),
        "target_ids": namespace["target_ids"],
        "total": namespace["total"].item(),
        "residual": namespace["residual"].item(),
        "condition": [0.3, -0.4],
        "condition_shape": [2],
        "predicted": namespace["predicted"],
        "generated": namespace["generated"],
        "parameter_sha256": hashlib.sha256(weights).hexdigest(),
        "prediction_shapes": stages,
        "stdout": stdout.getvalue(),
    }


base = run_block(code, 1.4)
assert base["target_ids"] == [1, 2, 0]
assert abs(base["total"] - 1.375) < 1e-6
assert abs(base["residual"] - 0.025) < 1e-6
assert base["predicted"] == [2, 1, 0]
assert base["generated"] == 1.875
needle = "value = torch.tensor(1.4)"
assert code.count(needle) == 1
exercise = run_block(code.replace(needle, "value = torch.tensor(1.6)"), 1.6)
assert exercise["target_ids"] == [2, 0, 2]
assert abs(exercise["total"] - 1.625) < 1e-6
assert exercise["predicted"] == base["predicted"]
assert exercise["parameter_sha256"] == base["parameter_sha256"]

# 精確可表示的平手只核 argmin 契約，不冒充本例原本有平手。
tie_distances = (torch.tensor(0.5) - torch.tensor([0.0, 1.0, 2.0])).square()
tie_id = tie_distances.argmin().item()
assert tie_distances.tolist() == [0.25, 0.25, 2.25] and tie_id == 0
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-17-17-v1/verify_inline.py",
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "platform": platform.platform(),
        "device": "cpu",
        "threads": torch.get_num_threads(),
    },
    "snapshot": {"path": SNAPSHOT, "sha256": EXPECTED_SHA},
    "inline_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "base": base,
    "exercise_only_value_changed": exercise,
    "exact_tie_probe": {"squared_distances": tie_distances.tolist(), "argmin": tie_id},
    "result": "All assertions passed. Forward mechanism only; no training or audio-quality evaluation.",
}
out = Path(__file__).with_name("inline-execution.json")
assert not out.exists(), "保留原執行證據，拒絕覆寫"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
