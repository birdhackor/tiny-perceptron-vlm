"""Bounded CPU checks of the lesson's shift, exercises, and real helper."""
import hashlib
import json
import platform
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
import torch
from tiny_perceptron.data import shifted

torch.set_num_threads(1)
out = Path(__file__).resolve().parent
checks = []
for text in ("貓看狗。", "鳥飛。", "貓看看狗。"):
    s = list(text)
    saved = s.copy()
    x, y = s[:-1], s[1:]
    pairs = list(zip(x, y, strict=True))
    expected = [(s[t], s[t + 1]) for t in range(len(s) - 1)]
    assert pairs == expected
    assert len(x) == len(y) == len(s) - 1
    assert s == saved
    ids = [ord(c) for c in text]
    helper_x, helper_y = shifted(ids)
    assert helper_x.device.type == helper_y.device.type == "cpu"
    assert helper_x.dtype == helper_y.dtype == torch.long
    assert helper_x.tolist() == ids[:-1]
    assert helper_y.tolist() == ids[1:]
    decoded = [(chr(a), chr(b)) for a, b in zip(helper_x.tolist(), helper_y.tolist(), strict=True)]
    assert decoded == expected
    check = {"text": text, "input": x, "target": y, "pairs": pairs, "pair_count": len(pairs), "helper_pairs": decoded, "original_unchanged": s == saved}
    checks.append(check)
    print(json.dumps(check, ensure_ascii=False))

s = list("貓看狗。")
wrong_x, wrong_y = s[:-1], s[:-1]
assert len(wrong_x) == len(wrong_y) == 3
wrong_pairs = list(zip(wrong_x, wrong_y, strict=True))
assert wrong_pairs == [("貓", "貓"), ("看", "看"), ("狗", "狗")]
assert wrong_pairs != list(zip(s[:-1], s[1:], strict=True))
wrong = {"wrong_target_same_length": True, "pairs": wrong_pairs, "length_assertion_passes": True, "position_alignment_correct": False}
print(json.dumps(wrong, ensure_ascii=False))

try:
    list(zip(["貓", "看", "狗"], ["看", "狗"], strict=True))
except ValueError as error:
    strict_error = str(error)
else:
    raise AssertionError("zip(strict=True) failed to reject unequal lengths")

short_errors = []
for ids in ([], [ord("貓")]):
    try:
        shifted(ids)
    except ValueError as error:
        short_errors.append({"ids": ids, "error": str(error)})
    else:
        raise AssertionError("shifted must reject inputs with no next-token pair")
print(json.dumps({"strict_length_error": strict_error, "short_input_errors": short_errors}, ensure_ascii=False))

environment = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "device": "CPU",
    "threads": str(torch.get_num_threads()),
    "repository_helper": "tiny_perceptron/data.py:46-51",
    "repository_helper_file_sha256": hashlib.sha256((root / "tiny_perceptron/data.py").read_bytes()).hexdigest(),
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(out / "probe-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(out / "probe-results.json").write_text(json.dumps({"checks": checks, "wrong_target": wrong, "strict_length_error": strict_error, "short_input_errors": short_errors, "conclusion_scope": "List slices, positions, exercise pairs, equal-length failure mode, and shifted helper contract only; no model training or performance evidence."}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("PASS: positional alignment and deliberate misalignment checks completed")
