"""獨立執行 12.18 fix08 的原版內嵌程式與明示練習；僅 CPU、無訓練。"""
from __future__ import annotations

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
OUT = Path(__file__).with_name("execution.json")
SOURCE = "docs/course-revision-20261029-omni/reviews/wave-01/source-freeze/sources/12.18.md"
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix08.json"
EXPECTED_SHA = "414b753581940ac40f546619ed1220b169b5a38d362aabd845fefa07252ac0a8"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def execute(code: str) -> dict:
    namespace = {"__name__": "__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "<12.18-inline>", "exec"), namespace)
    w, cb = namespace["waveform"], namespace["codebook"]
    return {
        "stdout": stdout.getvalue(),
        "waveform": w.tolist(),
        "codebook": cb.tolist(),
        "waveform_dtype": str(w.dtype),
        "codebook_dtype": str(cb.dtype),
        "device": str(w.device),
        "requires_grad": {"waveform": w.requires_grad, "codebook": cb.requires_grad},
        "shapes": {
            "blocks": list(namespace["blocks"].shape),
            "blocks_expanded": list(namespace["blocks"][:, None, :].shape),
            "codebook_expanded": list(cb[None, :, :].shape),
            "differences": list((namespace["blocks"][:, None, :] - cb[None, :, :]).shape),
            "distances": list(namespace["distances"].shape),
        },
        "distances": namespace["distances"].tolist(),
        "codes": namespace["codes"].tolist(),
        "reconstruction": namespace["reconstruction"].tolist(),
        "mse": namespace["mse"].item(),
        "squared_error_sum": (w - namespace["reconstruction"]).square().sum().item(),
        "mse_denominator": w.numel(),
        "duration_ms": w.numel() / namespace["sample_rate"] * 1000,
        "codebook_matches_source": cb.tolist() == [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]],
    }


def main() -> None:
    torch.set_num_threads(1)
    raw = (ROOT / SOURCE).read_bytes()
    assert digest(raw) == EXPECTED_SHA
    frozen = next(p for p in json.loads((ROOT / FREEZE).read_bytes())["pages"] if p["page_id"] == "12.18")
    assert frozen["source_sha256"] == EXPECTED_SHA
    program, = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
    notebook = json.loads((ROOT / "notebooks/12/12.18.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert any(c.rstrip() == program.rstrip() for c in cells)
    original = execute(program)
    expected_distances = torch.tensor([[1.62, .02, 3.62], [1.28, 3.28, .08], [.04, 2.04, 2.04]], dtype=torch.float64)
    assert torch.allclose(torch.tensor(original["distances"], dtype=torch.float64), expected_distances, rtol=0, atol=1e-14)
    assert original["codes"] == [1, 2, 0]
    assert original["reconstruction"] == [1., 0., -1., 0., 0., 1., 0., -1., 0., 0., 0., 0.]
    assert round(original["mse"], 6) == .011667
    assert original["duration_ms"] == .75
    insertion = "waveform[:4] = torch.tensor([0.4,0,-0.4,0])\n"
    exercise_program = program.replace("blocks = waveform.reshape", insertion + "blocks = waveform.reshape", 1)
    exercise = execute(exercise_program)
    assert exercise["codes"] == [0, 2, 0]
    assert round(exercise["mse"], 6) == .036667
    assert exercise["codebook"] == original["codebook"]
    # 平手屬於 API 契約；另以最簡單的第一個分界作實際核對。
    tie = execute(program.replace("blocks = waveform.reshape", "waveform[:4] = torch.tensor([0.5,0,-0.5,0])\nblocks = waveform.reshape", 1))
    assert tie["distances"][0][:2] == [.5, .5] and tie["codes"][0] == 0
    invalid = program.replace("blocks = waveform.reshape", "blocks = waveform.reshape")
    invalid = invalid.replace("block_size = codebook.shape[1]", "waveform = waveform[:-1]\nblock_size = codebook.shape[1]", 1)
    try:
        execute(invalid)
    except ValueError as exc:
        guard = {"sample_count": 11, "exception_type": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError("11-point input did not trigger explicit guard")
    record = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": "/root/omni_technical_coordinator/technical_12_18",
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-12-18-v1/verify_inline.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "platform": platform.platform(), "device": "cpu", "threads": "1", "default_dtype": str(torch.get_default_dtype())},
        "source": SOURCE, "source_sha256": digest(raw),
        "freeze": FREEZE, "freeze_sha256": digest((ROOT / FREEZE).read_bytes()),
        "inline_program_sha256": digest(program.encode()),
        "notebook_inline_match": True,
        "original": original, "exercise": exercise,
        "tie_check": {"distances_first_block": tie["distances"][0], "selected_id": tie["codes"][0]},
        "nonmultiple_guard": guard,
        "result": "All assertions passed. Original and exercise execute the frozen inline body, with only the lesson-prescribed insertion for the exercise. No training, GPU, download or quality test occurred.",
        "precision_note": "Exercise RHS is float32 by the stated code and is assigned into float64 waveform. Its 0.4 becomes 0.4000000059604645; printed rounded distances and MSE still agree. Exact-decimal arithmetic is separately labeled hand calculation.",
    }
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"result": "passed", "original_codes": original["codes"], "original_mse": round(original["mse"], 6), "exercise_codes": exercise["codes"], "exercise_mse": round(exercise["mse"], 6), "output": str(OUT.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
