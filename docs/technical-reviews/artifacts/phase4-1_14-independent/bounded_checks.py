"""1.14 CPU audit: exact fence, local variations, historical records and inference.

No optimizer, backward, model/data download or training is performed.
"""
import contextlib
import hashlib
import io
import json
import platform
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from scripts.course_experiments.text import _simple_examples, _simple_sample
from scripts.infer_simple import generate_simple, load_simple_checkpoint
from tiny_perceptron.data import split_documents, toy_documents

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

code = (OUT / "original/fence-1.py").read_text()

def execute_fence(source):
    namespace = {"__name__": "__main__"}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(source, "1.14 exact or explicitly modified fence", "exec"), namespace)
    return namespace, capture.getvalue().strip()

base, base_text = execute_fence(code)
repeat, repeat_text = execute_fence(code)
seven, seven_text = execute_fence(code.replace("manual_seed(42)", "manual_seed(7)"))
short, short_text = execute_fence(code.replace("range(12)", "range(3)").replace("len(output) == 13", "len(output) == 4"))
forced, forced_text = execute_fence(code.replace("g = torch.Generator()", "p = torch.zeros_like(p)\np[:, 3] = 1.0\ng = torch.Generator()"))
assert base_text == repeat_text
assert len(base["output"]) == len(seven["output"]) == 13
assert len(short["output"]) == 4
assert forced_text == "貓" + "。" * 12
p = base["p"]
assert p.shape == (4, 4) and p.dtype == torch.float32
assert bool(torch.isfinite(p).all()) and bool((p > 0).all())
assert torch.allclose(p.sum(dim=1), torch.ones(4), atol=1e-7, rtol=0)
assert abs(float(p[0, 1]) - 0.85) <= 1e-7
assert not p.requires_grad

g = torch.Generator().manual_seed(42)
current = 0
trace, ids = [], [0]
for step in range(12):
    old = current
    sampled = torch.multinomial(p[current], 1, generator=g)
    assert sampled.shape == (1,) and sampled.dtype == torch.int64
    current = sampled.item()
    assert type(current) is int and 0 <= current < 4
    ids.append(current)
    trace.append({"step": step + 1, "input_id": old, "selected_id": current, "chosen_probability": float(p[old, current]), "output_length": len(ids)})
assert "".join(base["chars"][i] for i in ids) == base_text
assert all(trace[i + 1]["input_id"] == trace[i]["selected_id"] for i in range(11))

historical = json.loads((OUT / "simple_models.raw.json").read_text())
assert historical["revision"] == "26f34ebb5d1e237611567697d2b3ea4d64669331"
assert historical["device"] == "cpu" and historical["seed"] == 42
assert historical["evidence_status"] == "complete_run" and historical["step_scale"] == 1
assert historical["results"]["runs"]["bigram"]["steps"] == 200
source_names = ["tiny_perceptron/simple.py", "tiny_perceptron/data.py", "scripts/course_experiments/text.py", "scripts/infer_simple.py", "scripts/course_release.py", "scripts/build_course.py", "docs/review-tools/section_facts.py"]
snapshots = OUT / "code"
for name in source_names:
    target = snapshots / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)
for name in source_names[:3]:
    assert sha(ROOT / name) == historical["code_sha256"][name]

documents = toy_documents()
parts = split_documents(documents, 42)
assert len(documents) == len(set(documents)) == 12
assert {split: len(texts) for split, texts in parts.items()} == {"train": 9, "validation": 1, "test": 2}
data_facts = {}
data_dir = OUT / "historical-data"
data_dir.mkdir(exist_ok=True)
for split, texts in parts.items():
    original_path = ROOT / "outputs/course-experiments/course-v1/simple_models/data" / (split + ".jsonl")
    assert sha(original_path) == historical["results"]["data"][split]["sha256"]
    raw_rows = [json.loads(line) for line in original_path.read_text().splitlines()]
    assert [row["text"] for row in raw_rows] == texts
    shutil.copyfile(original_path, data_dir / original_path.name)
    data_facts[split] = {"documents": len(texts), "sha256": sha(original_path), "character_targets_including_one_eos_per_document": sum(len(t) + 1 for t in texts)}

public_release = ROOT / "docs/course-experiments/public-releases/simple_models.json"
export_manifest = ROOT / "checkpoints/course/simple_models/export-manifest.json"
shutil.copyfile(public_release, OUT / "public-release.raw.json")
shutil.copyfile(export_manifest, OUT / "export-manifest.raw.json")
release = json.loads(public_release.read_text())
manifest = json.loads(export_manifest.read_text())
public_weight = ROOT / "checkpoints/course/simple_models/bigram.pt"
published = next(item for item in release["release"]["public_manifest"]["files"] if item["output"] == "bigram.pt")
exported = next(item for item in manifest["files"] if item["output"] == "bigram.pt")
original_weight = next(item for item in historical["artifacts"] if item["path"] == "bigram.pt")
assert sha(public_weight) == published["sha256"] == exported["sha256"]
assert exported["source_sha256"] == original_weight["sha256"]
assert manifest["provenance"]["revision"] == historical["revision"]
assert release["approval"]["private_source"]["revision"] == historical["hf"]["revision"]
shutil.copyfile(public_weight, OUT / "public-bigram.pt")
model, saved, vocabulary = load_simple_checkpoint(OUT / "public-bigram.pt", "cpu")
assert saved["context"] == 1 and saved["kind"] == "bigram"
assert saved["metadata"]["revision"] == historical["revision"]
assert vocabulary == {c: i + 2 for i, c in enumerate(sorted(set("".join(parts["train"]))))}
params_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
prompts = ["顏色=", "顏色=紅；形狀="]
samples = [_simple_sample(model, vocabulary, 1, prompt, "cpu", count=24) for prompt in prompts]
assert samples == historical["results"]["runs"]["bigram"]["samples"]
inference = [generate_simple(model, vocabulary, 1, prompt, tokens=24) for prompt in prompts]
assert all(item["answer"] == "三角。" and item["generated_ids"] == [4, 13, 3, 0] and item["eos"] for item in inference)
first_scores = [model(torch.tensor([[vocabulary[prompt[-1]]]])) for prompt in prompts]
assert torch.equal(*first_scores)
two_histories = torch.tensor([[vocabulary["色"], vocabulary["="]], [vocabulary["狀"], vocabulary["="]]])
assert torch.equal(model(two_histories)[0], model(two_histories)[1])
with torch.no_grad():
    post_nll = {}
    for split, texts in parts.items():
        x, y = _simple_examples(texts, vocabulary, 1)
        post_nll[split] = float(F.cross_entropy(model(x), y))
        expected = historical["results"]["runs"]["bigram"]["after_nll_same_post_update_time"][split]
        assert abs(post_nll[split] - expected) <= 1e-6
assert all(torch.equal(params_before[k], model.state_dict()[k]) for k in params_before)
before_nll = historical["results"]["runs"]["bigram"]["before_nll"]

result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "threads": "1", "cuda_build": str(torch.version.cuda), "platform": platform.platform()},
    "fence": {"original_sha256": sha(OUT / "original/fence-1.py"), "base_seed42": base_text, "repeat_seed42": repeat_text, "seed7": seven_text, "three_draws_seed42": short_text, "forced_punctuation_12_draws": forced_text, "base_length": len(base["output"]), "seed7_length": len(seven["output"]), "short_length": len(short["output"]), "row_sums": p.sum(1).tolist(), "shape_axes": "row=current ID; column=next ID", "float_tolerance": "absolute 1e-7, relative 0", "trace": trace, "parameter_update": "none; probability tensor has requires_grad=False; no optimizer/backward"},
    "numeric_derivation": {"cat_look_dog_probability": 0.85 * 0.5, "initial_cat_cat_probability": 0.05, "initial_cat_period_probability": 0.05, "length_formula": "1 opening character + N draws; 12 -> 13, 3 -> 4"},
    "historical": {"revision": historical["revision"], "run_id": historical["modal"]["run_id"], "original_environment": {"python": historical["python_version"], "torch": historical["torch_version"], "device": historical["device"]}, "data": data_facts, "total_documents": len(documents), "optimization_steps_recorded": 200, "original_samples": historical["results"]["runs"]["bigram"]["samples"], "before_nll_recorded": before_nll, "after_nll_recorded": historical["results"]["runs"]["bigram"]["after_nll_same_post_update_time"], "post_nll_recomputed_from_public_export": post_nll, "nll_absolute_tolerance": "1e-6; mean natural-log cross-entropy per character/EOS target", "nll_deltas_recorded": {split: post_nll[split] - before_nll[split] for split in parts}, "retrained": False},
    "public_export_inference": {"public_revision": release["release"]["revision"], "public_checkpoint_sha256": sha(public_weight), "private_source_checkpoint_sha256_in_export_manifest": exported["source_sha256"], "source_link_is_documentary": True, "public_export_metadata_revision": saved["metadata"]["revision"], "samples_via_original_helper": samples, "samples_via_generate_simple": inference, "identical_logits_for_equal_last_ids": True, "equal_last_id": vocabulary["="], "greedy_first_id": int(first_scores[0].argmax(-1).item()), "parameters_unchanged": True, "scope": "No retraining or download; hash-verified public inference export plus hash-verified original result/data/code. Private original checkpoint bytes are not locally rerun."},
    "source_code_sha256": {name: sha(ROOT / name) for name in source_names},
    "all_assertions_passed": True,
}
(OUT / "bounded.results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
