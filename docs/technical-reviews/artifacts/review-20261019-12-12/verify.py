import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from html import unescape

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
from tiny_perceptron.multimodal import log_mel, scene, tone

torch.set_num_threads(1)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_bytes())


def write(name, data):
    with (ART / name).open("x", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def fetch(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "40", url]
    p = subprocess.run(command, capture_output=True, check=True)
    return p.stdout.decode(), {"url": url, "command_argv": command, "exit_code": p.returncode,
                               "response_sha256": sha(p.stdout), "bytes": len(p.stdout),
                               "accessed_on": "2026-10-20"}


tensor_text, tensor_receipt = fetch("https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_tensor.py")
tensor_tree = ast.parse(tensor_text)
backward = next(n for c in tensor_tree.body if isinstance(c, ast.ClassDef) and c.name == "Tensor"
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "backward")
backward_doc = ast.get_docstring(backward)
tensor_receipt.update(version="v2.14.1", locator="Tensor.backward",
                      excerpt=backward_doc[:backward_doc.index("If the tensor")].strip(),
                      inspection="完整讀取 Tensor.backward 的 docstring 與函式體；鏈式法則計算並累積梯度，函式呼叫 torch.autograd.backward。")
paper_text, paper_receipt = fetch("https://arxiv.org/html/2004.07780v5")


def paper_section(ident):
    match = re.search(r'<section\b[^>]*id="' + re.escape(ident) + r'"', paper_text)
    assert match is not None
    end = re.search(r'<section\b', paper_text[match.end():])
    raw = paper_text[match.start():match.end() + end.start()] if end else paper_text[match.start():]
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", raw))).strip()


inspected = {key: paper_section(key) for key in
             ("S3", "S4.SS1", "S6.SS1.SSS0.Px1", "S6.SS1.SSS0.Px3", "S6.SS2.SSS0.Px2")}
paper_receipt.update(version="arXiv:2004.07780v5, 2023-11-21",
                     inspected_locators=list(inspected), excerpts=[
                         {"locator": "S3", "text": "When location is controlled for, network performance deteriorates to random guessing (red)."},
                         {"locator": "S4.SS1", "text": "This example highlights how a systematic relationship between object and background or context can easily create a shortcut opportunity."}],
                     inspection="已讀形狀與位置的玩具反例、資料相關性造成捷徑、區分資料集成績與能力、強基線及設計分布外測試的原文。其一般警示不能代替本專案的逐題結果。")
for excerpt in paper_receipt["excerpts"]:
    assert excerpt["text"] in inspected[excerpt["locator"]]
write("official-excerpts.json", {"sources": [tensor_receipt, paper_receipt]})

inputs = ["docs/course-review-20261019/freeze-01/sources/12.12.md",
          "docs/course-review-20261019/freeze-04.json", "course/figures/rewrite-12-joint-clues.svg",
          "docs/course-experiments/results/joint.json", "docs/course-experiments/results/encoders.json",
          "scripts/course_experiments/modalities.py", "tiny_perceptron/multimodal.py",
          "tiny_perceptron/model.py", "tiny_perceptron/data.py", "docs/review-tools/section_facts.py"]
input_hashes = {path: sha((ROOT / path).read_bytes()) for path in inputs}
assert input_hashes[inputs[0]] == "3a7347c88fbef9511033e09deca56aaeb62148d122437ecec0ee91edfa086fda"
assert input_hashes[inputs[1]] == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
assert input_hashes[inputs[2]] == "39dcb282ad3315efe5092daffe0415308fa2d730f6a74b6e910c4ac750e5ce2c"
run = ROOT / "outputs/reviewer-tools/runs/20261020T062316843671Z"
section_copy = {name: (json.loads((run / name).read_bytes()) if name.endswith(".json") else (run / name).read_text())
                for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py")}
assert section_copy["extraction.json"]["source_sha256"] == input_hashes[inputs[0]]
assert section_copy["execution.json"]["exit_code"] == 0
assert section_copy["stdout.txt"] == "圖路徑梯度 True\n音路徑梯度 True\n"

code = section_copy["fence-1.py"]
tree = ast.parse(code)
split = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "out" for t in node.targets))
ns = {"__name__": "__main__"}
exec(compile(ast.Module(body=tree.body[:split], type_ignores=[]), "frozen-fence-initialization", "exec"), ns)
model = ns["model"]
before = {name: p.detach().clone() for name, p in model.named_parameters()}
assert all(p.grad is None for p in model.parameters())
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(ast.Module(body=tree.body[split:], type_ignores=[]), "frozen-fence-forward-backward", "exec"), ns)
unchanged = all(torch.equal(before[name], p.detach()) for name, p in model.named_parameters())
assert unchanged
gradients = {name: {"norm": p.grad.norm().item(), "finite": bool(torch.isfinite(p.grad).all())}
             for name, p in model.named_parameters() if name in ("image_projector.weight", "audio_projector.weight")}
assert all(value["norm"] > 0 and value["finite"] for value in gradients.values())
with torch.no_grad():
    image_features = model.vision(scene("red", "circle")[None])
    audio_features = model.audio(tone(440)[None])
    mel = log_mel(tone(440))
assert list(image_features.shape) == [1, 16, 16]
assert list(audio_features.shape) == [1, 11, 16]
assert list(ns["out"]["logits"].shape) == [1, 42, 264]
assert int((ns["out"]["labels"] != -100).sum()) == 12

svg = (ROOT / inputs[2]).read_text()
import xml.etree.ElementTree as ET
svg_root = ET.fromstring(svg)
rects = [n for n in svg_root if n.tag.endswith("rect") and n.get("width") == "12" and n.get("height") == "12"]
assert len(rects) == 256
pixels = scene("red", "circle")
for rect in rects:
    x, y = (int(rect.get("x")) - 58) // 12, (int(rect.get("y")) - 86) // 12
    assert rect.get("fill") == ("#ff0000" if pixels[0, y, x].item() == 1 else "#000000")
receipt = load("docs/course-review-20261019/figure-renders-04.json")

j = load("docs/course-experiments/results/joint.json")
enc = load("docs/course-experiments/results/encoders.json")
r = j["results"]
for record in (j, enc):
    assert record["code_sha256"]["scripts/course_experiments/modalities.py"] == input_hashes["scripts/course_experiments/modalities.py"]
    assert record["code_sha256"]["tiny_perceptron/multimodal.py"] == input_hashes["tiny_perceptron/multimodal.py"]
splits = r["data"]["splits"]
split_summary = {}
for name, entry in splits.items():
    rows = entry["records"]
    assert len(rows) == entry["count"]
    assert sha(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()) == entry["sha256"]
    expected_n = 24 if name == "train" else 12
    assert len(rows) == expected_n
    combos = {(row["shape"], row["answer"].split(",")[1]) for row in rows}
    assert len(combos) == 4
    split_summary[name] = {"count": len(rows), "offsets": sorted({row["offset"] for row in rows}),
                           "frequencies": sorted({row["frequency"] for row in rows}),
                           "colors": sorted({row["color"] for row in rows}),
                           "shape_pitch_counts": {f"{a},{b}": sum(row["answer"] == f"{a},{b}" for row in rows)
                                                  for a, b in sorted(combos)}}
tok = ns["tok"]
counts, records = {}, {}
for name, evaluation in {**r["evaluations"], "validation": r["validation"]}.items():
    data = splits["validation" if name == "validation" else "test"]["records"]
    assert len(evaluation["samples"]) == evaluation["examples"] == len(data) == 12
    samples = []
    for row, sample in zip(data, evaluation["samples"], strict=True):
        shape, pitch = row["shape"], "high" if row["frequency"] > 300 else "low"
        if name == "swap_image":
            shape = "square" if shape == "circle" else "circle"
        if name == "swap_audio":
            pitch = "high" if pitch == "low" else "low"
        expected = f"{shape},{pitch}"
        assert sample["target"] == expected
        ids = sample["generated_ids"]
        assert tok.eos_id in ids
        raw = ids[:ids.index(tok.eos_id)]
        assert all(token >= 8 for token in raw)
        assert tok.decode(raw) == sample["generated"]
        assert sample["exact_match"] == (raw == tok.encode(expected))
        samples.append({"family": row["family"], "target": expected, "generated": sample["generated"]})
    measured = {"examples": len(samples), "correct": sum(s["generated"] == s["target"] for s in samples),
                "shape_correct": sum(s["generated"].split(",")[0] == s["target"].split(",")[0] for s in samples),
                "pitch_correct": sum(s["generated"].split(",")[1] == s["target"].split(",")[1] for s in samples)}
    assert all(evaluation[key] == val for key, val in measured.items())
    assert evaluation["effective_tokens"] == sum(len(tok.encode(s["target"])) + 1 for s in samples) == 138
    counts[name], records[name] = measured, samples
expected_counts = {"none": (12, 12, 12), "blank_image": (6, 12, 6), "blank_audio": (12, 6, 6),
                   "blank": (6, 6, 3), "swap_image": (12, 12, 12), "swap_audio": (12, 12, 12),
                   "validation": (8, 12, 8)}
for name, expected in expected_counts.items():
    assert tuple(counts[name][key] for key in ("shape_correct", "pitch_correct", "correct")) == expected
assert {s["generated"].split(",")[0] for s in records["blank_image"]} == {"circle"}
assert {s["generated"].split(",")[1] for s in records["blank_audio"]} == {"high"}
assert {s["generated"] for s in records["blank"]} == {"circle,high"}
errors = [s for s in records["validation"] if s["generated"] != s["target"]]
assert len(errors) == 4 and {s["family"].split(":")[0] for s in errors} == {"red", "blue"}
assert all(s["target"].startswith("circle,") and s["generated"].startswith("square,") for s in errors)
audio_train = enc["results"]["audio"]["data"]["splits"]["train"]["records"]
freqs = sorted({row["frequency"] for row in audio_train})
assert 260 in freqs and 340 in freqs
ambiguous_image = {shape: sorted({row["answer"] for row in splits["test"]["records"] if row["shape"] == shape})
                   for shape in ("circle", "square")}
ambiguous_audio = {str(f): sorted({row["answer"] for row in splits["test"]["records"] if row["frequency"] == f})
                   for f in (260, 340)}
assert all(len(v) == 2 for v in list(ambiguous_image.values()) + list(ambiguous_audio.values()))
historical = {}
for name, document in (("joint", j), ("encoders", enc)):
    p = subprocess.run(["git", "show", document["revision"] + ":scripts/course_experiments/modalities.py"], capture_output=True)
    historical[name] = {"revision": document["revision"], "exit_code": p.returncode,
                        "stderr": p.stderr.decode(), "current_matches_recorded_code_hash": True}

result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-12/verify.py",
          "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "dtype": "float32"},
          "input_sha256": input_hashes, "section_execution_copy": section_copy,
          "runtime": {"stdout": stdout.getvalue(), "parameters_unchanged": unchanged, "gradient_weights": gradients,
                      "ids_dtype": str(ns["ids"].dtype), "labels_dtype": str(ns["labels"].dtype),
                      "image_features_shape": list(image_features.shape), "audio_features_shape": list(audio_features.shape),
                      "mel_shape": list(mel.shape), "waveform_samples": len(tone(440)),
                      "expanded_logits_shape": list(ns["out"]["logits"].shape), "supervised_targets": 12},
          "figure": {"svg_pixels_match_scene": True, "pixel_count": len(rects),
                     "render_receipt_sha256": sha((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes()),
                     "scope": "SVG 像素與 scene(red, circle) 逐點比對；PNG 已另用 view_image 目視核對。"},
          "experiment_environment": {key: j[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu")},
          "training": {key: value for key, value in r["training"].items() if key != "history"},
          "splits": split_summary, "recounted": counts, "sample_comparisons": records,
          "validation_errors": errors, "encoder_audio_train_frequencies": freqs,
          "single_modality_ambiguity": {"shape_only": ambiguous_image, "frequency_only": ambiguous_audio},
          "constant_guess": {"shape": "6/12", "pitch": "6/12", "joint": "3/12"},
          "historical_git_object_probes": historical,
          "scope": "重新執行短程式並重算已保存逐題結果；未重跑訓練或載入歷史 GPU checkpoint，不能外推到其他資料、種子與平台。",
          "result": "全部斷言通過。"}
write("verification.json", result)
print(json.dumps({"result": result["result"], "runtime": result["runtime"], "recounted": counts}, ensure_ascii=False))
