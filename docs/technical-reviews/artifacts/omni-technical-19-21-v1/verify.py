"""Independently execute the exact frozen 19.21 examples; CPU only, no updates."""
import contextlib
import copy
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
OUT = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def execute(code, name):
    namespace = {"__name__": "__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, name, "exec"), namespace)
    return namespace, stdout.getvalue()


def section(raw, lesson):
    match = re.search(r"^## " + re.escape(lesson) + r" .*(?:\n|$)", raw, re.M)
    tail = raw[match.start():]
    following = re.search(r"^## ", tail[3:], re.M)
    return tail if following is None else tail[:following.start() + 3]


freeze = json.loads(FREEZE.read_text())
inputs = {"source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE.read_bytes())}, "pages": []}
for lesson in ["19.21", "12.17", "14.19", "4.5", "3.6", "10.5"]:
    page = next(p for p in freeze["pages"] if p["page_id"] == lesson)
    data = (ROOT / page["snapshot"]).read_bytes()
    current = section((ROOT / page["source"]).read_text(), lesson).encode()
    inputs["pages"].append({"lesson_id": lesson, "snapshot": page["snapshot"], "sha256": sha(data), "freeze_sha256": page["source_sha256"], "current_slice_sha256": sha(current), "snapshot_matches_freeze": sha(data) == page["source_sha256"], "current_slice_matches_snapshot": current == data})

page = next(p for p in freeze["pages"] if p["page_id"] == "19.21")
text = (ROOT / page["snapshot"]).read_text()
blocks = re.findall(r"```python\n(.*?)```", text, re.S)
notebook = json.loads((ROOT / page["notebook"]).read_text())
notebook_code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
inputs["embedded_blocks"] = [{"index": i, "sha256": sha(code.encode()), "lines": len(code.splitlines()), "found_exactly_in_notebook": any(code.strip() == c.strip() for c in notebook_code)} for i, code in enumerate(blocks)]
inputs["notebook"] = {"path": page["notebook"], "sha256": sha((ROOT / page["notebook"]).read_bytes()), "freeze_sha256": page["notebook_sha256"]}
inputs["figures"] = [{"path": p, "sha256": sha((ROOT / p).read_bytes()), "freeze_sha256": h} for p, h in page["figures_sha256"].items()]

small, small_stdout = execute(blocks[0], page["snapshot"] + ":block0")
main, main_stdout = execute(blocks[1], page["snapshot"] + ":block1")
exercise_code = blocks[0].replace("local_demo = F.avg_pool2d", "pixel_demo[0, 0, 3, 3] = 32\nlocal_demo = F.avg_pool2d")
small_ex, small_ex_stdout = execute(exercise_code, page["snapshot"] + ":exercise-pixel32")
exercise_code = blocks[1].replace("audio_valid[2] = False", "audio_valid[2] = False\nframe_valid[2] = False")
missing, missing_stdout = execute(exercise_code, page["snapshot"] + ":exercise-missing-both")

model, args = main["model"], main["args"]
weights, logits = main["weights"].detach(), main["logits"].detach()
before = [p.detach().clone() for p in model.parameters()]
model.zero_grad()
with torch.enable_grad():
    repeated, _ = model(*args)
    torch.nn.functional.cross_entropy(repeated, torch.tensor([0, 1, 2])).backward()
no_parameter_update = all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
gradient_names = ["vision.0.weight", "audio.weight", "layers.0.q.weight", "layers.1.q.weight", "head.weight"]
gradient_norms = {name: float(dict(model.named_parameters())[name].grad.norm()) for name in gradient_names}

checks = []
for label, edit in [
    ("wrong_image_shape", lambda a: a.__setitem__(0, a[0][:, :, :, :31])),
    ("wrong_audio_width", lambda a: a.__setitem__(2, a[2][:, :, :16])),
    ("wrong_time_shape", lambda a: a.__setitem__(1, a[1][:, :15])),
    ("nonbool_mask", lambda a: a.__setitem__(4, a[4].float())),
    ("nan_features", lambda a: a[2].__setitem__((0, 0, 0), float("nan"))),
    ("duration_zero", lambda a: a[6].__setitem__(0, 0)),
    ("duration_4001", lambda a: a[6].__setitem__(0, 4001)),
    ("interior_padding", lambda a: a[4].__setitem__((0, 1), False)),
    ("frame_negative", lambda a: a[1].__setitem__((0, 0), -1)),
    ("first_frame_not_zero", lambda a: a[1].__setitem__((0, 0), 1)),
    ("frame_at_end", lambda a: a[1].__setitem__((0, 15), 4000)),
    ("frame_not_increasing", lambda a: a[1].__setitem__((0, 1), 0)),
    ("audio_negative", lambda a: a[3].__setitem__((0, 0, 0), -1)),
    ("audio_over_end", lambda a: a[3].__setitem__((0, 3, 1), 4001)),
    ("audio_zero_duration", lambda a: a[3].__setitem__((0, 0, 1), 1000)),
    ("audio_over_200ms", lambda a: a[3].__setitem__((0, 0, 1), 1201)),
    ("audio_overlap", lambda a: a[3].__setitem__((0, 1, 0), 1199)),
]:
    candidate = [a.clone() for a in args]
    edit(candidate)
    try:
        model(*candidate)
        checks.append({"case": label, "rejected": False})
    except ValueError as exc:
        checks.append({"case": label, "rejected": True, "message": str(exc)})
for label, images, features in [
    ("over16frames", torch.rand(3, 17, 1, 32, 32), args[2]),
    ("over20chunks", args[0], torch.rand(3, 21, 17)),
    ("zero_frames", torch.rand(3, 0, 1, 32, 32), args[2]),
    ("zero_chunks", args[0], torch.rand(3, 0, 17)),
]:
    candidate = list(args)
    candidate[0], candidate[2] = images, features
    try:
        model(*candidate)
        checks.append({"case": label, "rejected": False})
    except ValueError as exc:
        checks.append({"case": label, "rejected": True, "message": str(exc)})

prereq_page = next(p for p in freeze["pages"] if p["page_id"] == "12.17")
prereq_code = re.findall(r"```python\n(.*?)```", (ROOT / prereq_page["snapshot"]).read_text(), re.S)[0]
prereq, prereq_stdout = execute(prereq_code, prereq_page["snapshot"] + ":block0")
rope_page = next(p for p in freeze["pages"] if p["page_id"] == "14.19")
rope_code = re.findall(r"```python\n(.*?)```", (ROOT / rope_page["snapshot"]).read_text(), re.S)[0]
rope, rope_stdout = execute(rope_code, rope_page["snapshot"] + ":block0")
scope = {
    "main_example": {"batch": 3, "frames": 16, "chunks": 4, "width": 12, "layers": 2, "seed": 7, "device": "cpu", "dtype": str(logits.dtype), "training_updates": 0},
    "first_pooling": {"response_first_row": small["response_demo"][0, 0, 0].tolist(), "local": small["local_demo"][0, 0].tolist(), "global": small["global_demo"].item()},
    "pooling_exercise": {"local": small_ex["local_demo"][0, 0].tolist(), "global": small_ex["global_demo"].item()},
    "main": {"logits_shape": list(logits.shape), "weights_shape": list(weights.shape), "logits": logits.tolist(), "centers": main["support"].mean(-1)[:2, 1].tolist(), "path1_mass": float(weights[0, 0, 20, 1:17].sum()), "path2_mass": float(weights[0, 1, 24, 19:23].sum()), "physical_future_frame_mass": float(weights[0, 0, 20, 16]), "future_sequence_audio_weight": float(weights[0, 0, 20, 21]), "masked_key_max_weight": float(weights[2, :, :, 19:23].abs().max()), "masked_query_max_weight": float(weights[2, :, 19:23, :].abs().max()), "index_shift_exact_equal": torch.equal(main["index_logits"][0], main["index_logits"][1]), "masked_audio_perturbation_equal": torch.equal(logits[2], main["changed_logits"][2]), "no_parameter_update_after_backward": no_parameter_update, "gradient_norms": gradient_norms, "record": main["record"], "talker_condition": main["talker_condition"]},
    "representation_comparison": {"time_pair_indices": [0, 3], "frequency": model.frequency.tolist(), "parameter_counts": {mode: sum(p.numel() for p in main[name].parameters()) for mode, name in [("time_rope", "model"), ("additive", "additive_model"), ("index", "index_model")]}, "same_initial_state": all(torch.equal(model.state_dict()[k], main["additive_model"].state_dict()[k]) for k in model.state_dict()), "additive_logits_shape": list(main["additive_logits"].shape)},
    "both_missing_exercise": {"logits_shape": list(missing["logits"].shape), "weights_shape": list(missing["weights"].shape), "third_row_readable_key_indices": torch.where(missing["weights"][2, 1, 24] > 0)[0].tolist(), "third_row_prediction": ["circle", "square", "unknown"][int(missing["logits"][2].argmax())]},
    "input_validation": checks,
    "prerequisite_12_17": {"code_sha256": sha(prereq_code.encode()), "feature_shape": list(prereq["a"][1]["content"].shape), "event_support_A": [prereq["a"][1]["start"], prereq["a"][1]["end"]], "event_support_B": [prereq["b"][1]["start"], prereq["b"][1]["end"]], "predictions": prereq["predictions"], "short_support_anchor": [prereq["short"][k] for k in ["start", "end", "anchor"]]},
    "prerequisite_14_19": {"code_sha256": sha(rope_code.encode()), "scores": [float(q @ rope["key"]) for q in [rope["q_a"], rope["q_b"]]]},
    "planned_threshold_arithmetic": {"main": 170 / 200, "panel_count_ceiling": 38, "panel_at_38": 38 / 50, "panel_at_37": 37 / 50, "missing": 180 / 200, "downstream_runs": 2 * 3},
    "planned_product_file_exists": (ROOT / "tiny_perceptron/omni_products/av.py").exists(),
    "stdout": {"19.21_block0": small_stdout, "19.21_block1": main_stdout, "pixel_exercise": small_ex_stdout, "both_missing_exercise": missing_stdout, "12.17_block0": prereq_stdout, "14.19_block0": rope_stdout},
}
receipt = {"created_at": datetime.now(UTC).isoformat(), "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-21-v1/verify.py", "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "platform": platform.platform()}, "inputs": inputs, "result": scope}
path = OUT / "execution-receipt.json"
with path.open("x") as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"receipt": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes()), "main_stdout": main_stdout, "validation_cases": len(checks), "all_rejected": all(c["rejected"] for c in checks), "source_pins_match": all(p["snapshot_matches_freeze"] and p["current_slice_matches_snapshot"] for p in inputs["pages"])}, ensure_ascii=False, indent=2))
