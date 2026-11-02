"""Reviewer-owned CPU check of the exact frozen artificial teaching examples."""

import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import tempfile
import wave
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
FROZEN = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v3-final/sources/19.22.md"
raw = FROZEN.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "28a160e345070c8dcdd795a965636496246fcbd3faac1a7c80f316a7153af96c"
blocks = re.findall(r"```python\n(.*?)\n```", raw.decode(), re.S)
assert len(blocks) == 2
result = {
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "threads": 1},
    "frozen_sha256": hashlib.sha256(raw).hexdigest(),
    "executed_block_sha256": [hashlib.sha256(code.encode()).hexdigest() for code in blocks],
    "scope": "Exact inline artificial one-update fixture and optional new random-parameter shape example. No saved model loaded, no final NN inference, no formal training or tuning.",
}
old_cwd = Path.cwd()
with tempfile.TemporaryDirectory(prefix="technical-19.22-") as temporary:
    os.chdir(temporary)
    ns = {"__name__": "__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(blocks[0], str(FROZEN) + ":inline", "exec"), ns)
    result["inline_stdout"] = stdout.getvalue().splitlines()
    assert abs(ns["loss"].item() - 1.371999) < 0.0000005
    assert ns["targets"]["circle"].tolist() == [1, 1, 0, 2, 2, 3]
    assert ns["targets"]["square"].tolist() == [2, 2, 0, 1, 1, 3]
    assert torch.equal(ns["frozen_table"], ns["codec"].table)
    assert ns["rows"][0]["gold"]["codes"] == ns["rows"][0]["chain"]["codes"] == [2]
    assert ns["rows"][0]["gold"]["stop"] == ns["rows"][0]["chain"]["stop"] == "EOS"
    with wave.open("demo-sound.wav", "rb") as wav:
        result["wav"] = {"channels": wav.getnchannels(), "sample_width_bytes": wav.getsampwidth(), "rate_hz": wav.getframerate(), "samples": wav.getnframes(), "compression": wav.getcomptype(), "sha256": hashlib.sha256(Path("demo-sound.wav").read_bytes()).hexdigest()}
    result["exercise"] = {}
    for semantic, expected in [("unknown", "transfer"), ("unsupported", "parse_error")]:
        row = ns["run_case"]({"gold": "square", "a_result": {"semantic": semantic}})
        assert row["chain"]["status"] == expected and "waveform" not in row["chain"]
        assert row["gold"]["codes"] == [2] and row["gold"]["stop"] == "EOS"
        result["exercise"][semantic] = {"status": row["chain"]["status"], "gold_codes": row["gold"]["codes"], "gold_stop": row["gold"]["stop"], "denominator": 1}
    formal = {"__name__": "__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(blocks[1], str(FROZEN) + ":optional", "exec"), formal)
    result["optional_stdout"] = stdout.getvalue().splitlines()
    result["parameter_counts"] = {"codec": sum(p.numel() for p in formal["formal_codec"].parameters()), "talker": sum(p.numel() for p in formal["formal_talker"].parameters())}
    assert result["parameter_counts"] == {"codec": 37232, "talker": 109314}
    assert tuple(formal["formal_scores"].shape) == (1, 51, 258)
    assert formal["formal_schedule"].shape == (51, 3)
    assert torch.equal(formal["whole"], formal["chunked"]) and len(formal["whole"]) == 4000
    changed = formal["example_codes"].clone()
    changed[25:] = (changed[25:] + 1) % 256
    with torch.no_grad():
        changed_wave = formal["formal_codec"].decode_codes(changed)
    result["artificial_prefix_equal"] = torch.equal(formal["whole"][:2000], changed_wave[:2000])
    assert result["artificial_prefix_equal"]
    os.chdir(old_cwd)

def definitions(text):
    return {node.name: node for node in ast.parse(text).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}

optional = definitions(blocks[1])
source = definitions((ROOT / "tiny_perceptron/omni_products/sound.py").read_text())
result["ast_matches_repository"] = {}
for name in ["CodecConfig", "SoundCodec", "program_schedule", "ProgramTalker"]:
    node = source[name]
    if name == "SoundCodec":
        node.body = [item for item in node.body if getattr(item, "name", "") != "reset_unused_training_codes"]
    same = ast.dump(optional[name], include_attributes=False) == ast.dump(node, include_attributes=False)
    result["ast_matches_repository"][name] = same
    assert same, name
data = definitions((ROOT / "tiny_perceptron/omni_products/data.py").read_text())
for name in ["Event", "validate_program"]:
    same = ast.dump(optional[name], include_attributes=False) == ast.dump(data[name], include_attributes=False)
    result["ast_matches_repository"][name] = same
    assert same, name

from tiny_perceptron.omni_products.data import program_id, sound_splits
from tiny_perceptron.omni_products.inference import SOUND_MAP

splits = sound_splits()
ids = {name: {program_id(program) for program in values} for name, values in splits.items()}
result["family_splits"] = {name: len(values) for name, values in ids.items()}
result["family_overlap"] = {"train_dev": len(ids["train"] & ids["dev"]), "train_final": len(ids["train"] & ids["final"]), "dev_final": len(ids["dev"] & ids["final"])}
result["semantic_programs_in_train"] = {name: program_id(program) in ids["train"] for name, program in SOUND_MAP.items()}
assert result["family_splits"] == {"train": 2560, "dev": 256, "final": 200}
assert not any(result["family_overlap"].values()) and all(result["semantic_programs_in_train"].values())
result["passed"] = True
print(json.dumps(result, ensure_ascii=False, indent=2))
