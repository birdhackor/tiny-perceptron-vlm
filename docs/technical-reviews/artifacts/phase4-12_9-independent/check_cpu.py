"""Independent bounded checks; no optimizer, checkpoint, model load or data download."""
import contextlib
import hashlib
import io
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.multimodal import log_mel, tone

torch.set_num_threads(1)
ART = Path(__file__).resolve().parent
code = (ART / "extracted/fence-1.py").read_text()
results = []
for frequency, answer in [(440, "high"), (200, "low")]:
    variant = code.replace('tok.encode("high")', f'tok.encode("{answer}")').replace("tone(440)", f"tone({frequency})")
    variant = variant.replace("loss.backward()", "before = {n:p.detach().clone() for n,p in model.named_parameters()}\nloss.backward()")
    name = f"bounded-{frequency}-{answer}.py"
    (ART / name).write_text(variant)
    namespace = {"__name__": "__main__"}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(variant, name, "exec"), namespace)
    model, tok, out = namespace["model"], namespace["tok"], namespace["out"]
    wave = tone(frequency)
    with torch.no_grad():
        features = model.audio(wave[None])
        projected = model.audio_projector(features)
    labels = out["labels"]
    effective = torch.nonzero(labels[0] != -100).flatten().tolist()
    count = int((labels != -100).sum())
    expected_answer = tok.encode(answer) + [tok.eos_id]
    assert labels[0, effective].tolist() == expected_answer
    assert effective == list(range(14, 14 + len(expected_answer)))
    assert tuple(out["logits"].shape) == (1, 14 + len(expected_answer), 264)
    assert tuple(features.shape) == (1, 11, 16)
    assert tuple(projected.shape) == (1, 11, 8)
    grad = float(model.audio_projector.weight.grad.norm())
    assert grad > 0
    changed = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), namespace["before"][n])]
    assert changed == []
    gathered = out["logits"][0, effective].log_softmax(-1)
    manual = -gathered[torch.arange(count), torch.tensor(expected_answer)].mean()
    assert torch.allclose(namespace["loss"], manual, atol=1e-6, rtol=0)
    # Changing only a future answer input must not affect the first-answer scores.
    altered = namespace["ids"].clone()
    altered[-2] = tok.encode("z")[0]
    with torch.no_grad():
        future_changed = model(altered, namespace["labels"], waveform=wave)["logits"]
    difference = float((future_changed[0, 14] - out["logits"][0, 14]).abs().max().detach())
    assert difference == 0.0
    results.append({
        "frequency_hz": frequency, "answer": answer, "original_ids": namespace["ids"].tolist(),
        "waveform_samples": len(wave), "duration_seconds": len(wave) / 16000,
        "log_mel_shape": list(log_mel(wave[None]).shape), "audio_feature_shape": list(features.shape),
        "projected_shape": list(projected.shape), "logit_shape": list(out["logits"].shape),
        "expanded_unshifted_length": len(namespace["ids"]) - 1 + len(projected[0]),
        "shifted_labels": labels.tolist(), "effective_positions": effective, "effective_targets": count,
        "loss": float(namespace["loss"].detach()), "manual_ce_mean": float(manual.detach()),
        "audio_projector_gradient_norm": grad, "changed_parameter_names": changed,
        "future_answer_change_first_answer_logit_max_abs_difference": difference,
        "captured_stdout": stream.getvalue(), "code_path": name,
        "code_sha256": hashlib.sha256(variant.encode()).hexdigest(),
    })
report = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
    "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda),
    "threads": str(torch.get_num_threads())}, "cases": results,
    "scope": "Short random-weight forward/backward checks only. No optimizer step, no trained model evaluation."}
(ART / "bounded-cpu.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
