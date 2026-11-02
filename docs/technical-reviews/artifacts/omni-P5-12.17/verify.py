"""Independent CPU checks for frozen canonical 12.17; no models or fixed families."""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"
page = next(p for p in json.loads(FREEZE.read_text())["pages"] if p["page_id"] == "12.17")
raw = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
source = (ROOT / page["source"]).read_text()
current = re.search(r"^## 12\.17 .*?(?=^## |\Z)", source, re.M | re.S)[0]
assert current.encode() == raw
code = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(code) == 1
notebook = json.loads((ROOT / page["notebook"]).read_text())
assert hashlib.sha256((ROOT / page["notebook"]).read_bytes()).hexdigest() == page["notebook_sha256"]
assert "".join(notebook["cells"][5]["source"]) == code[0].rstrip("\n")

stdout = io.StringIO()
scope = {}
with contextlib.redirect_stdout(stdout):
    exec(compile(code[0], page["snapshot"] + ":inline-python", "exec"), scope)

tokens, answer = scope["local_tokens"], scope["answer"]
wave, frames = scope["wave"], scope["frames"]
a, b = scope["a"], scope["b"]
assert scope["same_content"] and len(scope["sequence_a"]) == 24
assert scope["sequence_a"][20][0] == "audio"
assert torch.equal(scope["sequence_a"][20][1], a[1]["content"])
assert scope["predictions"] == ["圓", "方"]
assert [x["anchor"] for x in a] == [1100, 1300, 1500, 1700]
assert [x["anchor"] for x in b] == [2100, 2300, 2500, 2700]
assert answer(None, wave, 1000.0) == answer(frames, None, 1000.0) == "不可判定"
assert all(torch.equal(x["content"], y["content"]) for x, y in zip(a, b, strict=True))
assert scope["bank"].shape == (16, 201)
assert [list(x["content"].shape) for x in a] == [[17]] * 4
power = torch.stft(
    wave[3200:6400], n_fft=400, hop_length=160,
    window=torch.hann_window(400, dtype=wave.dtype), center=False, return_complex=True,
).abs().square()
assert power.shape == (201, 18)
changed = wave.clone()
changed[9600:] = 0.3
changed_tokens = tokens(changed, 1000.0)
assert all(torch.equal(a[i]["content"], changed_tokens[i]["content"]) for i in (0, 1, 2))
assert not torch.equal(a[3]["content"], changed_tokens[3]["content"])
short = tokens(wave[:100], 1000.0)[0]
assert [short[k] for k in ("start", "end", "anchor")] == [1000.0, 1006.25, 1003.125]
exercise = [answer(frames, wave, offset) for offset in (1500.0, 2500.0)]
assert exercise == ["圓", "方"]
try:
    tokens(wave, 3500.0)
except ValueError as error:
    overrun = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("3500 + 800 ms must fail")
lookup = [(t, int(torch.searchsorted(scope["frame_ms"], torch.tensor(t), right=True)) - 1)
          for t in (1300.0, 1800.0, 1999.0, 2000.0, 2300.0, 2800.0)]
assert lookup == [(1300.0, 5), (1800.0, 7), (1999.0, 7), (2000.0, 8), (2300.0, 9), (2800.0, 11)]

# Only deterministic preprocessing/collation of the above artificial data.
# No AVModel, trained weights, fixed train/dev/final families, or evaluator is run.
from tiny_perceptron.omni_products.av import collate_av, local_audio
from tiny_perceptron.omni_products.data import AVInput

features, starts, ends, anchors = local_audio(wave.float())
different, *_ = local_audio(changed.float())
assert features.shape == (4, 1, 24, 37)
assert torch.equal(features[:3], different[:3])
assert not torch.equal(features[3], different[3])
assert starts.tolist() == [0, 200, 400, 600]
assert ends.tolist() == [200, 400, 600, 800]
assert anchors.tolist() == [100, 300, 500, 700]
partial = local_audio(wave[:3301].float())
assert partial[0].shape == (2, 1, 24, 37)
assert partial[2].tolist() == [200.0, 206.3125]
assert partial[3].tolist() == [100.0, 203.15625]
item = AVInput(frames, scope["frame_ms"].float(), wave.float(), 1000.0, 4000).validate()
missing = replace(item, waveform=wave[:3301].float(), audio_present=False)
batch = collate_av([(item, 0), (missing, 2)])
assert batch["frames"].shape == (2, 16, 1, 32, 32)
assert batch["audio"].shape == (2, 4, 1, 24, 37)
assert batch["mask"].shape == (2, 20)
assert batch["mask"][0].all() and batch["mask"][1, :16].all() and not batch["mask"][1, 16:].any()
assert batch["presence"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
assert batch["audio_support_end_ms"][1].tolist() == [1200.0, 1206.3125, 0.0, 0.0]

# Preserve the actual formal pair-scoring expression, without running inference.
import ast
tree = ast.parse((ROOT / "scripts/omni_revision/core_evaluation.py").read_text())
pair_rule = [value for node in ast.walk(tree) if isinstance(node, ast.Dict)
             for key, value in zip(node.keys, node.values)
             if isinstance(key, ast.Constant) and key.value == "paired_correct"]
assert len(pair_rule) == 1
rule = compile(ast.Expression(pair_rule[0]), "preserved-pair-expression", "eval")
pair_results = [eval(rule, {"answers": x, "row": {"labels": [0, 1]}})
                for x in ([0, 1], [0, 0], [1, 1], [1, 0])]
assert pair_results == [True, False, False, False]

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_12_17",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "torch_git": torch.version.git_version, "device": "CPU", "threads": str(torch.get_num_threads())},
    "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
    "source_sha256": page["source_sha256"],
    "inline_stdout": stdout.getvalue(),
    "dimensions": {"frames": list(frames.shape), "waveform": list(wave.shape), "mel_bank": list(scope["bank"].shape),
                   "full_chunk_power": list(power.shape), "token": list(a[0]["content"].shape)},
    "energies": [x["content"][-1].item() for x in a],
    "supports_A": [[x["start"], x["end"], x["anchor"]] for x in a],
    "supports_B": [[x["start"], x["end"], x["anchor"]] for x in b],
    "active_interval_switch_clearances_ms": [700, 300],
    "perturbation": "first three tokens unchanged; changed fourth differs",
    "short": {k: short[k] for k in ("start", "end", "anchor")},
    "exercise": {"offsets_ms": [1500, 2500], "centers_ms": [1800, 2800], "answers": exercise, "overrun": overrun},
    "sample_hold_lookup": lookup,
    "actual_preprocessing": {"shape": list(features.shape), "starts_ms": starts.tolist(), "ends_ms": ends.tolist(),
                             "anchors_ms": anchors.tolist(), "partial_end_ms": partial[2].tolist(),
                             "partial_anchor_ms": partial[3].tolist(),
                             "batch_audio_shape": list(batch["audio"].shape), "mask": batch["mask"].tolist(),
                             "presence": batch["presence"].tolist(),
                             "support_end_ms": batch["audio_support_end_ms"].tolist()},
    "formal_pair_score_truth_table": {"gold": [0, 1], "answer_rows": [[0, 1], [0, 0], [1, 1], [1, 0]], "pair_correct": pair_results},
    "result": "all assertions passed",
    "not_run": ["training", "neural inference", "fixed-family evaluation", "weights", "Modal", "workflow dispatch", "publication"],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
