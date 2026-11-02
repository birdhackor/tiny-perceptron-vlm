"""CPU check of the exact frozen 12.17 inline code and its stated exercise."""

import contextlib
import hashlib
import io
import json
import platform
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
INLINE = HERE / "inline.py"
source = ROOT / "docs/course-revision-20261029-omni/reviews/wave-01/source-freeze/sources/12.17.md"
inline_stdout = io.StringIO()
with contextlib.redirect_stdout(inline_stdout):
    ns = runpy.run_path(str(INLINE))

wave, frames = ns["wave"], ns["frames"]
local_tokens, answer = ns["local_tokens"], ns["answer"]
a, b = ns["a"], ns["b"]


def rejected(call):
    try:
        call()
    except ValueError as exc:
        return {"exception": type(exc).__name__, "message": str(exc)}
    raise AssertionError("Expected ValueError")


shifted_a, shifted_b = local_tokens(wave, 1500.0), local_tokens(wave, 2500.0)
energy = [float(token["content"][-1]) for token in a]
raw = wave[3200:6400]
normalized = raw / raw.square().mean().sqrt().clamp(min=1e-6)
power = torch.stft(normalized, n_fft=400, hop_length=160,
                   window=torch.hann_window(400, dtype=wave.dtype),
                   center=False, return_complex=True).abs().square()
outside = wave.clone()
outside[:3200] = 0.2
outside[6400:] = -0.4
boundary_indices = [int(torch.searchsorted(ns["frame_ms"], torch.tensor(t), right=True)) - 1
                    for t in [0.0, 1999.0, 2000.0, 3999.0]]
results = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-12-17-v1/probe.py",
    "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                    "torch_git_version": torch.version.git_version, "platform": platform.platform(),
                    "device": "cpu", "threads": str(torch.get_num_threads()), "wave_dtype": str(wave.dtype)},
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "inline_sha256": hashlib.sha256(INLINE.read_bytes()).hexdigest(),
    "inline_stdout": inline_stdout.getvalue().splitlines(),
    "inputs": {"frame_shape": list(frames.shape), "wave_shape": list(wave.shape),
               "sample_rate": ns["rate"], "chunk_samples": ns["chunk_samples"],
               "bank_shape": list(ns["bank"].shape), "power_shape_per_full_chunk": list(power.shape),
               "content_shapes": [list(token["content"].shape) for token in a],
               "all_content_finite": all(bool(torch.isfinite(token["content"]).all()) for token in a),
               "energies": energy, "bank_min": float(ns["bank"].min())},
    "base": {"supports_a": [[t["start"],t["end"]] for t in a],
             "supports_b": [[t["start"],t["end"]] for t in b],
             "answers": ns["predictions"], "same_content": ns["same_content"],
             "active_chunk": [i for i,e in enumerate(energy) if e > 0.01],
             "boundary_query_indices": boundary_indices,
             "sequence_kinds": [k for k,v in ns["sequence_a"]]},
    "exercise": {"offsets": [1500.0,2500.0],
                 "anchors_a": [t["anchor"] for t in shifted_a],
                 "anchors_b": [t["anchor"] for t in shifted_b],
                 "answers": [answer(frames,wave,o) for o in [1500.0,2500.0]],
                 "content_unchanged": all(torch.equal(x["content"],y["content"])
                                           for x,y in zip(a,shifted_a,strict=True)),
                 "offset_3500": rejected(lambda: local_tokens(wave,3500.0))},
    "locality": {"all_other_chunks_changed_active_content_equal": torch.equal(
                     a[1]["content"],local_tokens(outside,1000.0)[1]["content"]),
                 "short_support": [ns["short"]["start"],ns["short"]["end"]],
                 "short_anchor": ns["short"]["anchor"]},
    "input_rejections": {
        "empty": rejected(lambda: local_tokens(wave[:0],1000.0)),
        "nonfinite_wave": rejected(lambda: local_tokens(torch.tensor([float("nan")]),1000.0)),
        "negative_start": rejected(lambda: local_tokens(wave,-1.0)),
        "nonfinite_start": rejected(lambda: local_tokens(wave,float("inf"))),
        "image_shape": rejected(lambda: answer(frames[:15],wave,1000.0))},
    "missing_or_ambiguous": {"no_images": answer(None,wave,1000.0),
                             "no_wave": answer(frames,None,1000.0),
                             "silent": answer(frames,torch.zeros_like(wave),1000.0),
                             "two_active": answer(frames,ns["changed"],1000.0)}
}
assert results["base"]["answers"] == ["圓","方"]
assert results["exercise"]["answers"] == ["圓","方"]
assert results["exercise"]["content_unchanged"]
assert results["locality"]["all_other_chunks_changed_active_content_equal"]
assert boundary_indices == [0,7,8,15]
assert all(value == "不可判定" for value in results["missing_or_ambiguous"].values())
print(json.dumps(results,ensure_ascii=False,indent=2))
