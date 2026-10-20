"""獨立核對聲音入口的確定性算例與實驗紀錄；不進行訓練。"""

import ast
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.multimodal import AudioEncoder, VisionEncoder, log_mel, tone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition):
    assert condition, "核對結果不符合預期"


torch.set_num_threads(1)
torch.manual_seed(0)
wave = tone()[None]
spectrogram = log_mel(wave)
encoder = AudioEncoder(bands=16, width=8)
features = encoder(wave)
transposed = spectrogram.transpose(-1, -2)
projected = encoder.projection(transposed)
first_frame = spectrogram[0, :, 0]
alone = encoder.feature(encoder.projection(first_frame))
permutation = torch.arange(features.shape[1] - 1, -1, -1)
reordered = encoder.feature(encoder.projection(transposed[:, permutation]))
check(wave.shape == (1, 1600))
check(spectrogram.shape == (1, 16, 11))
check(transposed.shape == (1, 11, 16))
check(projected.shape == features.shape == (1, 11, 8))
check(alone.shape == (8,))
check(torch.allclose(alone, features[0, 0]))
check(torch.allclose(reordered, features[:, permutation]))
check(torch.allclose(features.mean(1), features[:, permutation].mean(1)))
check(not torch.allclose(features, features[:, permutation]))

wide = AudioEncoder(bands=16, width=12)
more_bands = AudioEncoder(bands=32, width=8)
check(wide(wave).shape == (1, 11, 12))
check(more_bands(wave).shape == (1, 11, 8))
check(more_bands.projection.weight.shape == (8, 32))
check(spectrogram.shape == (1, 16, 11))
try:
    more_bands.feature(more_bands.projection(first_frame))
except RuntimeError as error:
    mismatch = str(error)
else:
    raise AssertionError("不同帶數應造成維度錯誤")
check(more_bands.feature(more_bands.projection(log_mel(wave, bands=32)[0, :, 0])).shape == (8,))

boundaries = []
for length in (0, 1, 199, 200, 201, 202):
    short = torch.arange(length, dtype=torch.float32)[None]
    padded = F.pad(short, (0, 201 - length)) if length <= 200 else short
    window = torch.hann_window(400, dtype=short.dtype, device=short.device)
    direct = torch.stft(padded, 400, 160, window=window, return_complex=True)
    actual = log_mel(short)
    check(actual.shape == (1, 16, 2))
    check(torch.isfinite(actual).all())
    boundaries.append({"input_samples": length, "samples_after_guard": padded.shape[-1],
                       "stft_shape": list(direct.shape), "log_mel_shape": list(actual.shape),
                       "finite": bool(torch.isfinite(actual).all())})
reflection = F.pad(torch.arange(201.0)[None, None], (200, 200), mode="reflect")[0, 0]
check(torch.equal(reflection[:200], torch.arange(200.0, 0.0, -1.0)))
check(torch.equal(reflection[-200:], torch.arange(199.0, -1.0, -1.0)))
try:
    torch.stft(torch.arange(200.0)[None], 400, 160, window=torch.hann_window(400), return_complex=True)
except RuntimeError as error:
    reflection_failure = str(error)
else:
    raise AssertionError("反射補點長度必須小於原始長度")

double_wave = wave.double()
double_window = torch.hann_window(400, device=double_wave.device, dtype=double_wave.dtype)
check(double_window.dtype == double_wave.dtype)
check(log_mel(double_wave).dtype == torch.float64)

raw = (ROOT / "docs/course-review-20261019/freeze-01/sources/12.8.md").read_text()
blocks = raw.split("```python\n")[1:]
excerpt = blocks[1].split("```")[0]
excerpt_tree = ast.parse(excerpt)
module_tree = ast.parse((ROOT / "tiny_perceptron/multimodal.py").read_text())
definition_matches = {}
for name in ("log_mel", "AudioEncoder"):
    a = next(node for node in excerpt_tree.body if getattr(node, "name", None) == name)
    b = next(node for node in module_tree.body if getattr(node, "name", None) == name)
    definition_matches[name] = ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    check(definition_matches[name])

from scripts.course_experiments import modalities
from tiny_perceptron.model import ModelConfig, TinyLM

dependency_calls = []
saved_encoders = {"vision": VisionEncoder(), "audio": AudioEncoder()}
for model in saved_encoders.values():
    for parameter in model.parameters():
        nn.init.constant_(parameter, 0.03125)


class Context:
    device = "cpu"

    def dependency(self, experiment, filename=None):
        dependency_calls.append([experiment, filename])
        return (experiment, filename)


def mock_load(reference, **kwargs):
    check(kwargs == {"map_location": "cpu", "weights_only": True})
    check(reference[0] == "encoders")
    return {"encoder": saved_encoders[reference[1].removesuffix(".pt")].state_dict()}


with patch.object(modalities, "load_checkpoint", return_value=(TinyLM(ModelConfig(width=8)), {})), patch.object(
    modalities.torch, "load", side_effect=mock_load
):
    loaded = modalities._modal(Context(), "audio")
for name, saved in saved_encoders.items():
    actual = getattr(loaded, name).state_dict()
    check(all(torch.equal(actual[k], v) for k, v in saved.state_dict().items()))
entrypoint_calls = {}
for function in (modalities.run_audio, modalities.run_joint):
    tree = ast.parse(inspect.getsource(function))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_modal"]
    check(len(calls) == 1)
    check(not any(keyword.arg == "pretrained" for keyword in calls[0].keywords))
    entrypoint_calls[function.__name__] = ast.unparse(calls[0])

reports = {}
for name in ("encoders", "audio", "joint"):
    path = ROOT / f"docs/course-experiments/results/{name}.json"
    report = json.loads(path.read_text())
    code = {p: {"recorded": report["code_sha256"][p], "current": sha(ROOT / p)}
            for p in ("tiny_perceptron/multimodal.py", "scripts/course_experiments/modalities.py")}
    for hashes in code.values():
        check(hashes["recorded"] == hashes["current"])
    reports[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                     "revision": report["revision"], "device": report["device"],
                     "seed": report["seed"], "torch": report["torch_version"],
                     "code_fingerprints": code}
    if name == "encoders":
        audio = report["results"]["audio"]
        splits = audio["data"]["splits"]
        for split in splits.values():
            check(split["count"] == len(split["records"]))
            check(all(row["answer"] == ("high" if row["frequency"] > 300 else "low")
                      for row in split["records"]))
        test = audio["test"]
        check(test["count"] == len(test["samples"]) == splits["test"]["count"])
        check(test["correct"] == sum(row["correct"] for row in test["samples"]))
        check(test["accuracy"] == test["correct"] / test["count"])
        reports[name]["audio"] = {
            "steps": audio["training"]["steps"], "initial_loss": audio["training"]["initial_loss"],
            "final_loss": audio["training"]["final_loss"], "config": audio["config"],
            "split_frequencies": {k: sorted({r["frequency"] for r in v["records"]}) for k, v in splits.items()},
            "split_counts": {k: v["count"] for k, v in splits.items()}, "test": test,
            "scope": audio["scope"],
        }

figure = ROOT / "course/figures/rewrite-12-frame-features.svg"
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_text())
record = next(r for r in receipt["records"] if r["page_id"] == "12.8")
check(sha(figure) == record["frozen_svg"]["sha256"])
check(sha(ROOT / record["png"]["path"]) == record["png"]["sha256"])

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-8/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform()},
    "seed": 0,
    "base": {"wave_shape": list(wave.shape), "seconds": 1600 / 16000,
             "frame_count_derivation": "1 + floor((1600 + 2*200 - 400)/160) = 11",
             "spectrogram_shape": list(spectrogram.shape), "transposed_shape": list(transposed.shape),
             "projected_shape": list(projected.shape), "features_shape": list(features.shape),
             "alone_shape": list(alone.shape), "alone_allclose": bool(torch.allclose(alone, features[0, 0])),
             "alone_max_abs_difference": float((alone - features[0, 0]).abs().max().detach()),
             "tolerance": {"rtol": 1e-5, "atol": 1e-8},
             "parameter_count": sum(p.numel() for p in encoder.parameters()),
             "layernorm_normalized_shape": list(encoder.feature[0].normalized_shape)},
    "ordering": {"pointwise_map_commutes_with_permutation": bool(torch.allclose(reordered, features[:, permutation])),
                 "sequence_changes_after_permutation": not bool(torch.allclose(features, features[:, permutation])),
                 "temporal_mean_unchanged": bool(torch.allclose(features.mean(1), features[:, permutation].mean(1))),
                 "derivation": "逐框函式 g 作用在每個列向量：g(PX)=Pg(X)。時間平均是各列之和除以列數；交換加總次序不改變精確實數的平均，浮點運算使用 allclose。保留列序並不保證 16 維到 8 維映射可逆，也不表示入口已學會聲音語義。"},
    "exercises": {"width_12_features": list(wide(wave).shape),
                  "bands_32_features": list(more_bands(wave).shape),
                  "bands_32_projection_weight": list(more_bands.projection.weight.shape),
                  "separate_spectrogram": list(spectrogram.shape), "mismatch_error": mismatch,
                  "matching_bands_first_frame_shape": [8]},
    "short_waveforms": boundaries,
    "reflection": {"left_first_last": [float(reflection[0]), float(reflection[199])],
                   "right_first_last": [float(reflection[-200]), float(reflection[-1])],
                   "without_guard_length_200_error": reflection_failure},
    "float64": {"window_dtype": str(double_window.dtype), "log_mel_dtype": str(log_mel(double_wave).dtype)},
    "excerpt_definition_matches": definition_matches,
    "dependency_loading": {"mocked_checkpoint_io": True, "actual_model_and_strict_state_loading": True,
                           "calls": dependency_calls, "entrypoints": entrypoint_calls,
                           "scope": "只驗證現行入口確實呼叫預訓練載入路徑；以記憶體中的固定權重替代磁碟 checkpoint，未驗證遠端備份權重內容。"},
    "experiment_receipts": reports,
    "figure_receipt": record,
    "result": "全部確定性檢查符合預期；既有訓練只核對保存的報告與程式指紋，未重新訓練或下載權重。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
