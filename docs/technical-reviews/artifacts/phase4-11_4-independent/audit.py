"""Independent bounded CPU checks for 11.4; never load or train a checkpoint."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, scene
from scripts.course_experiments.modalities import _sequence, _vision_records

torch.set_num_threads(1)
assert torch.version.cuda is None
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def emit(key, value):
    print(json.dumps({key: value}, ensure_ascii=False, sort_keys=True))

emit("environment", {"python": platform.python_version(), "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version), "device": "cpu", "threads": 1,
    "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
code = (HERE / "fence-1.py").read_bytes()
namespace = {"__name__": "__main__"}
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(code, "original-11.4-fence-1", "exec"), namespace)
assert capture.getvalue() == ('red_circle 什麼顏色？ → 紅\nred_circle 什麼形狀？ → 圓形\n同圖 True\n答案不同 True\n')
emit("original_fence", {"sha256": sha(code), "stdout": capture.getvalue(), "records": namespace["records"]})
records = namespace["records"]
changed = records + [{"image": "blue_circle", "question": "什麼顏色？", "answer": "藍"}]
color_rows = [r for r in changed if r["question"] == "什麼顏色？"]
assert len({r["answer"] for r in color_rows}) == 2
emit("bounded_variations", {"red_blue_color_targets": [r["answer"] for r in color_rows],
    "constant_red_lookup_correct": sum(r["answer"] == "紅" for r in color_rows),
    "constant_red_lookup_denominator": len(color_rows),
    "joint_question_added": {"image": "red_circle", "question": "同時給顏色與形狀", "answer": "紅色圓形"},
    "extra_content_exact_match": "紅色圓形" == "紅"})

raw_path = ROOT / "docs/course-experiments/results/vqa.json"
raw = raw_path.read_bytes()
original = json.loads(raw)
# Only these explicit raw measurement/configuration/provenance/sample pointers are read.
# Do not inspect notes, review, scope, or any *_scope_correction.
provenance_keys = ["revision", "device", "seed", "torch_version", "python_version", "step_scale"]
pointers = ["/" + k for k in provenance_keys]
emit("original_provenance", {k: original[k] for k in provenance_keys})
emit("original_file", {"path": str(raw_path.relative_to(ROOT)), "sha256": sha(raw), "bytes": len(raw)})
snap = HERE / "vqa-original.json"
snap.write_bytes(raw)
for name in ["scripts/course_experiments/modalities.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/model.py"]:
    pointers.append("/code_sha256/" + name.replace("~", "~0").replace("/", "~1"))
    observed = sha((ROOT / name).read_bytes())
    assert observed == original["code_sha256"][name]
    emit("original_code_match", {"path": name, "sha256": observed})

data = original["results"]["data"]
pointers.append("/results/data/seed")
assert data["seed"] == original["seed"]
expected = _vision_records(("shape?", "color?"))
families, pixels = {}, {}
for split in ["train", "validation", "test"]:
    p = "/results/data/splits/" + split
    pointers.extend([p + "/count", p + "/sha256", p + "/records"])
    item = data["splits"][split]
    rows = item["records"]
    assert rows == expected[split]
    assert len(rows) == item["count"]
    assert sha(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()) == item["sha256"]
    families[split] = {r["family"] for r in rows}
    pixels[split] = set()
    for family in families[split]:
        pair = [r for r in rows if r["family"] == family]
        assert len(pair) == 2 and {r["question"] for r in pair} == {"shape?", "color?"}
        assert pair[0]["answer"] != pair[1]["answer"]
        images = [scene(r["color"], r["shape"], offset=r["offset"]) for r in pair]
        assert torch.equal(images[0], images[1])
        pixels[split].add(sha(images[0].numpy().tobytes()))
    effective = sum(len(ByteTokenizer().encode(r["answer"])) + 1 for r in rows)
    emit("split", {"name": split, "questions": len(rows), "image_families": len(families[split]),
        "pixel_identities": len(pixels[split]), "effective_answer_byte_plus_EOS_targets": effective,
        "question_counts": {q: sum(r["question"] == q for r in rows) for q in ["shape?", "color?"]},
        "example_same_image_pair": rows[:2]})
for a,b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not (families[a] & families[b])
    assert not (pixels[a] & pixels[b])
emit("split_intersections", {"all_family_and_pixel_intersections": "empty"})

tok = ByteTokenizer()
for name in ["projector_only", "partial", "all", "all_replay", "direct_vqa"]:
    variant = original["results"]["variants"][name]
    p = "/results/variants/" + name
    training_keys = ["steps", "trainable_parameters", "effective_tokens", "weights_changed", "nonzero_gradient_seen", "config", "modal_config", "history"]
    training = {k: variant["training"][k] for k in training_keys}
    pointers.extend(p + "/training/" + k for k in training_keys)
    assert training["steps"] == len(training["history"])
    assert training["effective_tokens"] == sum(h["effective_targets"] for h in training["history"])
    assert training["weights_changed"] and training["nonzero_gradient_seen"]
    emit("historical_training", {"variant": name, **{k:v for k,v in training.items() if k != "history"}})
    for split in ["validation", "test"]:
        keys = ["examples", "correct", "exact_match", "effective_tokens", "eos_rate", "samples", "groups"]
        measure = {k:variant[split][k] for k in keys}
        pointers.extend(p + "/" + split + "/" + k for k in keys)
        rows = expected[split]
        assert measure["examples"] == len(rows) == len(measure["samples"]) == 12
        matches = []
        for row, sample in zip(rows, measure["samples"], strict=True):
            assert (sample["family"], sample["question"], sample["target"]) == (row["family"], row["question"], row["answer"])
            ids = sample["generated_ids"]
            predicted = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            matches.append(predicted == tok.encode(row["answer"]))
            assert sample["exact_match"] == matches[-1]
            assert sample["eos"] == (tok.eos_id in ids)
        assert sum(matches) == measure["correct"]
        assert measure["exact_match"] == sum(matches)/len(rows)
        assert measure["effective_tokens"] == sum(len(tok.encode(r["answer"])) + 1 for r in rows) == 72
        assert measure["eos_rate"] == sum(s["eos"] for s in measure["samples"])/len(rows)
        for q,group in measure["groups"].items():
            chosen = [s for s in measure["samples"] if s["question"] == q]
            assert group["count"] == len(chosen) == 6
            assert group["correct"] == sum(s["exact_match"] for s in chosen)
        emit("historical_evaluation", {"variant": name, "split": split,
            **{k:v for k,v in measure.items() if k != "samples"}, "samples_inspected": len(measure["samples"]),
            "red_circle_two_questions": measure["samples"][:2]})

# A fresh tiny untrained forward checks the representation and answer mask only.
torch.manual_seed(0)
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
context = SimpleNamespace(device="cpu")
with torch.no_grad():
    for row in expected["train"][:2]:
        ids, labels, count = _sequence(row, context)
        image = scene(row["color"], row["shape"], offset=row["offset"])
        out = model(ids, labels, image=image)
        effective = out["labels"][out["labels"] != -100].tolist()
        assert effective == tok.encode(row["answer"]) + [tok.eos_id]
        assert len(effective) == count
        emit("fresh_forward_representation", {"question": row["question"], "answer": row["answer"],
            "valid_answer_plus_EOS_targets": count, "valid_label_ids": effective,
            "expanded_shifted_labels_shape": list(out["labels"].shape),
            "scope": "no backward, optimizer, checkpoint or learned-answer test"})

# Compare every pictured pixel with the actual deterministic generator.
svg = ROOT / "course/figures/rewrite-11-same-image-questions.svg"
tree = ET.fromstring(svg.read_bytes())
grid = [r for r in tree.findall(".//{http://www.w3.org/2000/svg}rect") if r.attrib.get("width") == "13"]
assert len(grid) == 256
image = scene("red", "circle")
for r in grid:
    col = (int(r.attrib["x"]) - 210)//13
    row = (int(r.attrib["y"]) - 82)//13
    expected_fill = "#ff0000" if image[0,row,col] == 1 else "#000000"
    assert r.attrib["fill"] == expected_fill
emit("figure_pixels", {"svg_sha256": sha(svg.read_bytes()), "grid_shape": [16,16],
    "matching_pixels": 256, "red_pixels": int(image[0].sum()),
    "black_pixels": int((image[0] == 0).sum()), "same_generator": "scene(red,circle,size=16,offset=0)",
    "text_labels": [" ".join(x.itertext()) for x in tree.findall(".//{http://www.w3.org/2000/svg}text")]})
(HERE / "json-pointer-inspection.json").write_text(json.dumps({"original_path": str(raw_path.relative_to(ROOT)),
    "original_sha256": sha(raw), "inspected_pointers": pointers,
    "excluded": ["notes", "review", "scope", "*_scope_correction"],
    "scope": "raw data/config/provenance/samples; no previous reviewer or author judgments inspected"}, ensure_ascii=False, indent=2)+"\n")
emit("result", "all bounded assertions passed")
