"""Check the disclosed audio omission and that returned labels fit the loss API; no training."""

import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys

import torch
import torch.nn.functional as F

from tiny_perceptron.model import masked_loss
from tiny_perceptron.multimodal import tone


raw = Path("docs/course-review-20261019/freeze-01/sources/10.8.md").read_bytes()
namespace = {}
with contextlib.redirect_stdout(io.StringIO()):
    for block in re.findall(r"```python\n(.*?)\n```", raw.decode(), re.S):
        exec(compile(block, "frozen-10.8-supplement", "exec"), namespace)
library = namespace["library"]
ours = namespace["ours"]
waveform = tone()
audio_ids = torch.tensor([1, 6, 4, 73, 2])
combined_ids = torch.tensor([1, 5, 6, 4, 73, 2])
with torch.no_grad():
    audio = library(audio_ids, waveform=waveform)
    combined = library(combined_ids, image=namespace["scene"](), waveform=waveform)
    omitted_audio_check = ours(torch.tensor([1, 6, 4]))
    try:
        library(torch.tensor([1, 6, 4]))
    except ValueError as error:
        library_audio_guard = f"{type(error).__name__}: {error}"
    else:
        raise AssertionError("Library should reject missing audio")
    labelled = ours(namespace["sequence"], torch.tensor([-100, -100, -100, 73, 2]), image=namespace["scene"]())
    loss = masked_loss(labelled["logits"], labelled["labels"])
    selected_logits = labelled["logits"][labelled["labels"] != -100]
    selected_labels = labelled["labels"][labelled["labels"] != -100]
    reference_loss = F.cross_entropy(selected_logits, selected_labels)
    assert torch.equal(loss, reference_loss) and torch.isfinite(loss)
    assert selected_labels.tolist() == [73, 2]
    assert audio["logits"].shape == (1, 14, 264)
    assert combined["logits"].shape == (1, 30, 264)
    assert omitted_audio_check["logits"].shape == (1, 3, 264)
    assert library_audio_guard == "ValueError: placeholder 缺少配對圖片／聲音"

print(json.dumps({
    "created_at": datetime.now(timezone.utc).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_10_8-v1/verify_supplement.py",
    "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
    "audio_input_ids": audio_ids.tolist(), "waveform_shape": list(waveform.shape),
    "audio_logits_shape": list(audio["logits"].shape),
    "combined_input_ids": combined_ids.tolist(), "combined_logits_shape": list(combined["logits"].shape),
    "disclosed_audio_omission": {"simplified_accepts_audio_id_as_ordinary_embedding": True, "library_error": library_audio_guard},
    "loss_check": {"effective_targets": selected_labels.tolist(), "denominator": selected_labels.numel(), "masked_loss": loss.item(), "selected_targets_reference": reference_loss.item(), "exact_equal": torch.equal(loss, reference_loss)},
    "scope": "Forward/loss computation only; no backward, optimiser update or training. Audio is checked only as the library API premise, not its signal-processing theory.",
    "result": "all assertions passed",
}, ensure_ascii=False, indent=2))
