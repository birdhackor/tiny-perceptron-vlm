"""12.12 獨立技術核對：原封不動執行凍結課文內嵌程式，另重算保存紀錄。

不啟動訓練、不下載權重；既有 GPU 成績只核逐題紀錄及其來源程式。
"""

import contextlib
import hashlib
import inspect
import io
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("execution.json")
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def section(path, number):
    raw = (ROOT / path).read_text()
    match = re.search(r"^## " + re.escape(number) + r"\b.*?(?=^## |\Z)", raw, re.M | re.S)
    assert match
    return match.group()


def score(block):
    rows = block["samples"]
    pairs = [(s["target"].split(","), s["generated"].split(",")) for s in rows]
    counts = {
        "examples": len(rows),
        "correct": sum(s["target"] == s["generated"] for s in rows),
        "shape_correct": sum(len(b) == 2 and a[0] == b[0] for a, b in pairs),
        "pitch_correct": sum(len(b) == 2 and a[1] == b[1] for a, b in pairs),
    }
    assert all(block[key] == val for key, val in counts.items())
    assert block["skipped"] == [] and block["generation_errors"] == 0
    assert block["invalid_special_tokens"] == 0 and block["eos_rate"] == 1.0
    return {**counts, "generated_counts": dict(Counter(s["generated"] for s in rows))}


def main():
    started = datetime.now(UTC).isoformat()
    # 一筆極小 CPU 計算；執行的課文程式本身未作任何改寫。
    torch.set_num_threads(1)
    freeze = json.loads((ROOT / FREEZE).read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "12.12")
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert digest(raw) == page["source_sha256"]
    assert section(page["source"], "12.12").encode() == raw
    body = raw.decode()
    blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(blocks) == 1
    code = blocks[0]
    notebook_path = ROOT / page["notebook"]
    assert digest(notebook_path.read_bytes()) == page["notebook_sha256"]
    notebook = json.loads(notebook_path.read_bytes())
    assert any(c["cell_type"] == "code" and "".join(c["source"]).strip() == code.strip()
               for c in notebook["cells"])
    captured, ns, initial = io.StringIO(), {}, {}
    marker = code.splitlines().index('out = model(ids, labels, image=scene("red", "circle"), waveform=tone(440))') + 1
    filename = "<exact-frozen-12.12-code-block>"

    def trace(frame, event, arg):
        if frame.f_code.co_filename == filename and event == "line" and frame.f_lineno == marker:
            initial.update({k: v.detach().clone() for k, v in frame.f_globals["model"].state_dict().items()})
        return trace

    sys.settrace(trace)
    try:
        with contextlib.redirect_stdout(captured):
            exec(compile(code, filename, "exec"), ns)
    finally:
        sys.settrace(None)
    stdout = captured.getvalue()
    assert stdout == "圖路徑梯度 True\n音路徑梯度 True\n"
    model, tok, output = ns["model"], ns["tok"], ns["out"]
    unchanged = all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())
    assert initial and unchanged
    with torch.no_grad():
        image, waveform = ns["scene"]("red", "circle"), ns["tone"](440)
        image_features = model.image_projector(model.vision(image[None]))
        audio_features = model.audio_projector(model.audio(waveform[None]))
        loss = ns["masked_loss"](output["logits"], output["labels"])
    valid = output["labels"] != -100
    assert image_features.shape == (1, 16, 8) and audio_features.shape == (1, 11, 8)
    assert output["logits"].shape == (1, 42, 264)
    assert output["labels"][valid].tolist() == ns["answer"]
    assert int(valid.sum()) == 12 and output["labels"][0, :30].eq(-100).all()
    assert torch.isfinite(output["logits"]).all() and torch.isfinite(loss)
    grads = {name: float(getattr(model, name).weight.grad.norm())
             for name in ("image_projector", "audio_projector")}
    assert all(x > 0 for x in grads.values())
    inline = {
        "code_sha256": digest(code.encode()), "stdout": stdout,
        "seed": 0, "updates": 0, "all_state_dict_entries_unchanged": unchanged,
        "input_ids": ns["ids"].tolist(), "input_labels": ns["labels"].tolist(),
        "input_dtypes": {k: str(ns[k].dtype) for k in ("ids", "labels")},
        "image_shape": list(image.shape), "waveform_shape": list(waveform.shape),
        "feature_dtype": str(image.dtype), "image_feature_shape": list(image_features.shape),
        "audio_feature_shape": list(audio_features.shape),
        "logits_shape": list(output["logits"].shape), "shifted_labels": output["labels"].tolist(),
        "effective_targets": int(valid.sum()), "masked_loss": float(loss),
        "projector_grad_norms": grads,
        "scope": "Exact frozen lesson code, CPU gradient connectivity only; no optimizer or generation accuracy test.",
    }

    joint_path = "docs/course-experiments/results/joint.json"
    enc_path = "docs/course-experiments/results/encoders.json"
    joint = json.loads((ROOT / joint_path).read_bytes())
    enc = json.loads((ROOT / enc_path).read_bytes())
    result = joint["results"]
    # 核對當次紀錄與目前真正閱讀的主要執行程式完全相同。
    paths = ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/multimodal.py",
             "scripts/course_experiments/modalities.py"]
    code_matches = {}
    for path in paths:
        sha = digest((ROOT / path).read_bytes())
        assert joint["code_sha256"][path] == sha == enc["code_sha256"][path]
        code_matches[path] = sha
    expected = {"none": (12, 12, 12), "blank_image": (6, 12, 6),
                "blank_audio": (12, 6, 6), "blank": (6, 6, 3),
                "swap_image": (12, 12, 12), "swap_audio": (12, 12, 12)}
    evaluations = {}
    for name, block in result["evaluations"].items():
        values = score(block)
        assert values["examples"] == 12
        assert tuple(values[k] for k in ("shape_correct", "pitch_correct", "correct")) == expected[name]
        evaluations[name] = values
    validation = score(result["validation"])
    assert (validation["shape_correct"], validation["pitch_correct"], validation["correct"]) == (8, 12, 8)
    errors = [{k: s[k] for k in ("family", "target", "generated")}
              for s in result["validation"]["samples"] if s["target"] != s["generated"]]
    assert len(errors) == 4 and all(s["family"].startswith(("red:circle:", "blue:circle:")) for s in errors)

    split_summaries = {}
    for name, split in result["data"]["splits"].items():
        rows = split["records"]
        assert len(rows) == split["count"]
        computed_hash = digest(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode())
        assert computed_hash == split["sha256"]
        assert all(r["answer"] == f"{r['shape']},{'high' if r['frequency'] > 300 else 'low'}" for r in rows)
        split_summaries[name] = {
            "count": len(rows), "offsets": sorted({r["offset"] for r in rows}),
            "frequencies": sorted({r["frequency"] for r in rows}),
            "colors": sorted({r["color"] for r in rows}),
            "shape_pitch_counts": dict(Counter(r["answer"] for r in rows)),
            "records_sha256": computed_hash,
        }
    rows = result["data"]["splits"]["test"]["records"]
    originals = result["evaluations"]["none"]["samples"]
    swap_consistency = {}
    for name, part in (("swap_image", 0), ("swap_audio", 1)):
        modified = result["evaluations"][name]["samples"]
        assert len(modified) == len(rows)
        checks = []
        for r, a, b in zip(rows, originals, modified, strict=True):
            expected_shape = ("square" if r["shape"] == "circle" else "circle") if part == 0 else r["shape"]
            freq = (340 if r["frequency"] <= 300 else 260) if part == 1 else r["frequency"]
            target = f"{expected_shape},{'high' if freq > 300 else 'low'}"
            assert b["target"] == target == b["generated"]
            before, after = a["generated"].split(","), b["generated"].split(",")
            checks.append(before[part] != after[part] and before[1-part] == after[1-part])
        assert all(checks)
        swap_consistency[name] = {"pairs": len(checks), "only_expected_part_changed_and_correct": sum(checks)}
    audio_train = enc["results"]["audio"]["data"]["splits"]["train"]["records"]
    audio_freqs = sorted({r["frequency"] for r in audio_train})
    assert {260, 340}.issubset(audio_freqs)
    assert not ({260, 340} & set(split_summaries["train"]["frequencies"]))
    assert all(s["generated"].startswith("circle,") for s in result["evaluations"]["blank_image"]["samples"])
    assert all(s["generated"].endswith(",high") for s in result["evaluations"]["blank_audio"]["samples"])
    assert all(s["generated"] == "circle,high" for s in result["evaluations"]["blank"]["samples"])
    t = result["training"]
    assert joint["seed"] == 42 and t["steps"] == 400 and not t["cpu_smoke"]
    assert t["parameters"] == t["trainable_parameters"] == 145664
    historical = {
        "report_paths_and_sha256": {p: digest((ROOT / p).read_bytes()) for p in (joint_path, enc_path)},
        "scope": "Recomputed saved per-example generations and split metadata, read exact producing code; did not rerun GPU training or generation.",
        "original_environment": {k: joint[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status")},
        "training": {k: t[k] for k in ("steps", "parameters", "trainable_parameters", "config", "modal_config", "effective_targets", "cpu_smoke")},
        "code_sha256_matches_both_reports": code_matches,
        "splits": split_summaries, "evaluations": evaluations,
        "validation": validation, "validation_errors": errors,
        "swap_pair_checks": swap_consistency, "encoder_audio_train_frequencies": audio_freqs,
    }

    figure_path = "course/figures/rewrite-12-joint-clues.svg"
    figure_raw = (ROOT / figure_path).read_bytes()
    assert digest(figure_raw) == page["figures_sha256"][figure_path]
    root = ET.fromstring(figure_raw)
    cells = [n for n in root if n.tag.endswith("rect") and n.attrib.get("width") == "12"]
    assert len(cells) == 256
    red = 0
    for cell in cells:
        x = (int(cell.attrib["x"]) - 58) // 12
        y = (int(cell.attrib["y"]) - 86) // 12
        expected_color = "#ff0000" if image[0, y, x] == 1 else "#000000"
        assert cell.attrib["fill"] == expected_color
        red += int(expected_color == "#ff0000")
    assert red == 49
    figure = {"svg_sha256": digest(figure_raw), "grid": [16, 16], "red_pixels": red,
              "pixel_fills_exactly_match_scene_red_circle": True,
              "text_labels": [n.text for n in root if n.tag.endswith("text")]}

    official = []
    for obj in (torch.Tensor.backward, torch.optim.Optimizer.step):
        lines, begin = inspect.getsourcelines(obj)
        path = Path(inspect.getsourcefile(obj))
        official.append({"api": obj.__qualname__, "installed_path": str(path),
                         "file_sha256": digest(path.read_bytes()), "line_start": begin,
                         "line_end": begin + len(lines) - 1,
                         "source_excerpt": "".join(lines[:15])})
    evidence = {
        "started_at": started, "completed_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-12-12-v1/verify_12_12.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_git_version": torch.version.git_version, "device": "cpu",
                        "threads": str(torch.get_num_threads()), "platform": platform.platform()},
        "source_freeze": FREEZE, "source_snapshot": page["snapshot"], "source_sha256": digest(raw),
        "exact_inline": inline, "historical_report_recomputation": historical,
        "figure_data_check": figure, "installed_official_source_inspection": official,
        "result": "PASS: exact inline stdout, shapes, labels, gradients, no update; every saved empirical count and swap pair; code versions; figure pixel fills.",
    }
    with OUT.open("x") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "sha256": digest(OUT.read_bytes()),
                      "inline_stdout": stdout, "result": evidence["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
