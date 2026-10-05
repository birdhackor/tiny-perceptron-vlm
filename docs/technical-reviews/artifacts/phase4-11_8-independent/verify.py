"""Bounded independent CPU audit; no training, checkpoint load, model save or downloads."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
import ast
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import modalities as m
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.multimodal import scene
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
BASE = Path(__file__).parent
digest = lambda b: hashlib.sha256(b).hexdigest()
raw = (BASE / "inputs/vision_ablation.original.json").read_bytes()
obj = json.loads(raw)
pointers = []
def read(pointer):
    parts = pointer.strip("/").split("/")
    assert all(p not in {"notes", "review", "crop_note", "patch_budget"} and not p.endswith("scope_correction") for p in parts)
    value = obj
    for p in parts:
        p = p.replace("~1", "/").replace("~0", "~")
        value = value[int(p)] if isinstance(value, list) else value[p]
    pointers.append(pointer)
    return value

provenance = {k: read("/" + k) for k in ["schema_version", "experiment_id", "revision", "device", "seed", "torch_version", "python_version", "step_scale"]}
training = {k: read("/results/training/" + k) for k in ["config", "modal_config", "steps", "effective_tokens", "effective_targets", "weights_changed", "nonzero_gradient_seen", "cpu_smoke"]}
splits = m._vision_records(("shape?", "color?"))
split_check = {}
for name, rows in splits.items():
    records = read(f"/results/data/splits/{name}/records")
    count = read(f"/results/data/splits/{name}/count")
    expected_sha = read(f"/results/data/splits/{name}/sha256")
    sha = digest(json.dumps(records, ensure_ascii=False, sort_keys=True).encode())
    assert records == rows and len(records) == count and sha == expected_sha
    split_check[name] = {"count": count, "records_sha256": sha, "families": len({r["family"] for r in records})}
families = [{r["family"] for r in rows} for rows in splits.values()]
assert all(not a & b for i, a in enumerate(families) for b in families[i + 1:])
data_metadata = {k: read("/results/data/" + k) for k in ["seed", "split_policy"]}

# Execute the original shape-swap list expression, without invoking its training runner.
tree = ast.parse((ROOT / "scripts/course_experiments/modalities.py").read_bytes())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_vision_ablation")
swap_node = next(n.value for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "swapped" for t in n.targets))
swapped = eval(compile(ast.Expression(swap_node), "original:run_vision_ablation:swapped", "eval"), {"splits": splits})
assert len(swapped) == len(splits["test"]) == 12
for a, b in zip(splits["test"], swapped, strict=True):
    assert a["color"] == b["color"] and a["offset"] == b["offset"] and a["question"] == b["question"]
    assert a["shape"] != b["shape"]
    assert b["answer"] == (b["shape"] if b["question"] == "shape?" else b["color"])

tok = ByteTokenizer()
evaluation = {}
sample_fields = ["row", "family", "question", "target", "generated", "generated_ids", "exact_match", "eos", "generation_error", "invalid_special_tokens", "donor_row"]
metrics = ["examples", "correct", "exact_match", "effective_tokens", "eos_rate", "generation_errors", "invalid_special_tokens", "ablation", "skipped", "groups"]
for name in ["none", "blank", "shuffle", "shape_swap_relabelled"]:
    prefix = "/results/interventions/" + name
    scalars = {k: read(prefix + "/" + k) for k in metrics}
    samples = read(prefix + "/samples")
    selected = []
    expected_records = swapped if name == "shape_swap_relabelled" else splits["test"]
    for index, s in enumerate(samples):
        selected.append({k: s[k] for k in sample_fields})
        ids = s["generated_ids"]
        answer_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        assert s["row"] == index and s["family"] == expected_records[index]["family"]
        assert s["target"] == expected_records[index]["answer"]
        assert s["question"] == expected_records[index]["question"]
        assert s["generated"] == tok.decode(answer_ids)
        assert s["exact_match"] == (answer_ids == tok.encode(s["target"]))
        assert s["eos"] == (tok.eos_id in ids)
        assert s["invalid_special_tokens"] == sum(t < 8 for t in answer_ids)
        assert s["generation_error"] is None
        assert s["donor_row"] == ((index + 6) % 12 if name == "shuffle" else None)
    correct = sum(s["exact_match"] for s in samples)
    assert len(samples) == scalars["examples"] == 12 and correct == scalars["correct"]
    assert scalars["exact_match"] == correct / len(samples)
    assert scalars["effective_tokens"] == sum(len(tok.encode(s["target"])) + 1 for s in samples) == 72
    assert scalars["skipped"] == [] and scalars["eos_rate"] == 1.0
    groups = {q: {"correct": sum(s["exact_match"] for s in samples if s["question"] == q), "count": sum(s["question"] == q for s in samples)} for q in ["shape?", "color?"]}
    assert groups == scalars["groups"]
    evaluation[name] = {"metrics": scalars, "samples": selected}
original = evaluation["none"]["samples"]
replacement = evaluation["shape_swap_relabelled"]["samples"]
shape_pairs = [{"row": s["row"], "original_target": s["target"], "replacement_target": t["target"], "original_generated": s["generated"], "replacement_generated": t["generated"], "both_correct": s["exact_match"] and t["exact_match"]} for s, t in zip(original, replacement, strict=True) if s["question"] == "shape?"]
assert len(shape_pairs) == 6 and sum(p["both_correct"] for p in shape_pairs) == 0
assert all(p["original_generated"] == p["replacement_generated"] == "circle" for p in shape_pairs)
shuffle_samples = evaluation["shuffle"]["samples"]
color_relabelled = [{"row": s["row"], "donor_row": s["donor_row"], "original_target": s["target"], "donor_target": splits["test"][s["donor_row"]]["color"], "generated": s["generated"]} for s in shuffle_samples if s["question"] == "color?"]
assert all(s["generated"] == s["donor_target"] for s in color_relabelled)
assert all(s["generated"] != s["original_target"] for s in color_relabelled)

# A parameterless recording double exercises the original evaluator's pixel and target contracts.
# Its constant answer is deliberately not evidence of any trained model's ability.
class RecordingDouble:
    def __init__(self): self.images = []
    def eval(self): return self
    def __call__(self, ids, labels, image=None, waveform=None):
        self.images.append(image.clone())
        return {"logits": torch.zeros(1, len(ids), 264), "labels": labels[None]}
saved_generate = m.generate_modal
def constant_generation(model, prefix, image, waveform, tokens):
    return torch.cat([prefix, torch.tensor(tok.encode("circle") + [tok.eos_id])])
m.generate_modal = constant_generation
contract_rows = [splits["test"][0], splits["test"][6]]
contract = {}
try:
    for mode in ["none", "blank", "shuffle"]:
        double = RecordingDouble()
        score = m._evaluate(double, contract_rows, SimpleNamespace(device="cpu"), mode)
        assert [s["target"] for s in score["samples"]] == ["circle", "square"]
        if mode == "blank": assert all(not torch.count_nonzero(i) for i in double.images)
        else:
            expected_images = [m._media(r, SimpleNamespace(device="cpu"))[0] for r in (contract_rows[::-1] if mode == "shuffle" else contract_rows)]
            assert all(torch.equal(a,b) for a,b in zip(double.images, expected_images, strict=True))
        contract[mode] = {"unchanged_targets": [s["target"] for s in score["samples"]], "donor_rows": [s["donor_row"] for s in score["samples"]], "image_nonzero_scalars": [int(torch.count_nonzero(i)) for i in double.images]}
finally:
    m.generate_modal = saved_generate

# Test the displayed schematic's 16x16 pixel maps against the actual scene rules at offset 0.
svg = ET.fromstring((ROOT / "course/figures/rewrite-11-shape-pairs.svg").read_bytes())
figure = {}
for origin, shape in [(54, "circle"), (373, "square")]:
    pixels = torch.zeros(3, 16, 16)
    count = 0
    for e in svg.iter("{http://www.w3.org/2000/svg}rect"):
        if e.get("width") != "13" or e.get("height") != "13": continue
        x, y = int(e.get("x")), int(e.get("y"))
        if not origin <= x < origin + 16 * 13: continue
        assert y >= 127 and (x-origin)%13 == (y-127)%13 == 0
        assert e.get("fill") in ["#000000", "#ff0000"]
        if e.get("fill") == "#ff0000": pixels[0,(y-127)//13,(x-origin)//13] = 1
        count += 1
    assert count == 256 and torch.equal(pixels, scene("red", shape, offset=0))
    figure[shape] = {"pixels": count, "red_pixels": int(pixels[0].sum()), "schematic_offset": 0, "historical_test_offset": 2}

baseline = {}
for name, truth, prediction in [("original_balanced", ["circle","square","circle","square"], "circle"), ("nine_to_one", ["circle"]*9+["square"], "circle"), ("permuted_balanced", ["square","circle","square","circle"], "circle"), ("opposite_constant", ["circle","square","circle","square"], "square"), ("only_square", ["square"]*4, "circle")]:
    correct = sum(a == b for a,b in zip(truth, [prediction]*len(truth), strict=True))
    baseline[name] = {"correct": correct, "count": len(truth), "accuracy": correct/len(truth)}
assert [x["accuracy"] for x in baseline.values()] == [.5,.9,.5,.5,0]
unchanged_shape = {"question": "shape?", "original": "red circle", "replacement": "blue circle", "original_target": "circle", "replacement_target": "circle"}
code_versions = {}
for p in ["scripts/course_experiments/modalities.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/data.py"]:
    sha = digest((ROOT/p).read_bytes())
    assert sha == read("/code_sha256/"+p.replace("/","~1"))
    code_versions[p] = sha
result = {"scope": "Recompute existing raw measurements and exercise parameterless evaluator/data/pixel contracts; no trained-model inference or training", "environment": {"python":sys.version, "python_executable":sys.executable, "torch":str(torch.__version__), "torch_git_version":str(torch.version.git_version), "cuda_build":str(torch.version.cuda), "cuda_available":str(torch.cuda.is_available()), "device":"cpu", "threads":torch.get_num_threads()}, "original_json_sha256":digest(raw), "provenance":provenance, "training_raw_configuration":training, "data_metadata":data_metadata, "split_checks":split_check, "interventions":evaluation, "shape_pairs":shape_pairs, "paired_shape_correct":0, "paired_shape_count":6, "unique_undirected_shape_pairs":3, "color_donor_checks":color_relabelled, "color_donor_correct":6, "color_donor_count":6, "evaluator_recording_double":contract, "figure_pixel_check":figure, "baseline_variations":baseline, "question_relevance_example":unchanged_shape, "code_sha256":code_versions, "inspected_json_pointers":sorted(set(pointers))}
(BASE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"raw_measurements_verified":True,"shape_original_correct":3,"shape_replacement_correct":3,"shape_pairs_both_correct":"0/6","original_total":"9/12","blank_original_target":"5/12","shuffle_original_target":"3/12","shuffle_donor_color":"6/6","baseline_variations":baseline,"figure_red_pixels":figure,"training_or_trained_model_inference_run":False},ensure_ascii=False,indent=2))
