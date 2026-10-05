"""Bounded CPU review: raw evidence accounting, original fence changes, no training.

Only named raw measurement/config/provenance pointers are inspected in vqa.json.
No checkpoint is loaded, no optimizer is run, no weights are saved.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import torch
from scripts.course_experiments.common import split_records
from scripts.course_experiments.modalities import _freeze, _hash, _sequence, _vision_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
raw_path = OUT / "inputs/docs/course-experiments/results/vqa.json"
original_bytes = raw_path.read_bytes()
raw = json.loads(original_bytes)
inspected = []

def pointer(path):
    assert all(k not in {"notes", "review", "scope"} and not k.endswith("scope_correction")
               for k in path.strip("/").split("/")), path
    inspected.append(path)
    obj = raw
    for key in path.strip("/").split("/"):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj

result = {
    "scope": "Reaccounting saved evidence, not retraining or reevaluating saved models. New random model only checks requires_grad flags; no optimizer or backward.",
    "raw_json_sha256": hashlib.sha256(original_bytes).hexdigest(),
    "environment": {"python": sys.version, "python_executable": sys.executable,
                    "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
                    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
                    "device": "cpu", "threads": torch.get_num_threads()},
}
provenance_fields = ["schema_version", "experiment_id", "revision", "device", "seed", "torch_version", "python_version", "step_scale"]
result["original_run_provenance"] = {k: pointer("/" + k) for k in provenance_fields}
seed = result["original_run_provenance"]["seed"]
revision = result["original_run_provenance"]["revision"]
result["code_version_checks"] = []
for name in ["scripts/course_experiments/modalities.py", "scripts/course_experiments/common.py",
             "scripts/course_experiments/text.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py",
             "tiny_perceptron/multimodal.py"]:
    digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    # JSON pointer escape is explicit; no access to unrelated code provenance values.
    escaped = name.replace("~", "~0").replace("/", "~1")
    inspected.append("/code_sha256/" + escaped)
    recorded = raw["code_sha256"][name]
    historical = subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)
    historical_digest = hashlib.sha256(historical).hexdigest()
    assert digest == recorded == historical_digest, name
    result["code_version_checks"].append({"path": name, "current_sha256": digest,
                                          "recorded_sha256": recorded, "revision_sha256": historical_digest})
prepared = generate_records("attributes-sft")
text_splits = split_records(prepared, seed=seed)
vqa_splits = _vision_records(("shape?", "color?"))
result["split_checks"] = {}
for key in ["train", "validation", "test"]:
    source_records = pointer("/results/data/splits/" + key + "/records")
    source_count = pointer("/results/data/splits/" + key + "/count")
    source_hash = pointer("/results/data/splits/" + key + "/sha256")
    text_count = pointer("/results/text_data/" + key + "/count")
    text_hash = pointer("/results/text_data/" + key + "/sha256")
    assert source_records == vqa_splits[key]
    assert len(source_records) == source_count and _hash(source_records) == source_hash
    assert len(text_splits[key]) == text_count and _hash(text_splits[key]) == text_hash
    result["split_checks"][key] = {"vqa_count": source_count, "vqa_sha256": source_hash,
                                   "text_count": text_count, "text_sha256": text_hash,
                                   "vqa_families": len({r["family"] for r in source_records}),
                                   "text_families": len({r["family"] for r in text_splits[key]})}
for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not {r["family"] for r in text_splits[first]} & {r["family"] for r in text_splits[second]}
    assert not {r["family"] for r in vqa_splits[first]} & {r["family"] for r in vqa_splits[second]}
tok = ByteTokenizer()

def raw_exact(sample, expected, exact_key):
    generated = sample["generated_ids"]
    answer_ids = generated[:generated.index(tok.eos_id)] if tok.eos_id in generated else generated
    exact = answer_ids == tok.encode(expected)
    assert exact == sample[exact_key]
    assert sample["generated"] == tok.decode(answer_ids)
    assert sample["eos"] == (tok.eos_id in generated)
    return exact

variants = ["projector_only", "partial", "all", "all_replay"]
result["variants"] = {}
baseline = None
baseline_identity = None
for name in variants:
    base = "/results/variants/" + name
    before = pointer(base + "/text_before/samples")
    after = pointer(base + "/text_after/samples")
    vqa_before = pointer(base + "/before/samples")
    vqa_after = pointer(base + "/test/samples")
    assert len(before) == len(after) == len(text_splits["test"]) == 10
    assert len(vqa_before) == len(vqa_after) == len(vqa_splits["test"]) == 12
    identities = [(s["messages"], s["expected"]) for s in before]
    assert identities == [(s["messages"], s["expected"]) for s in after]
    assert identities == [(r["messages"][:-1], r["messages"][-1]["content"]) for r in text_splits["test"]]
    if baseline is None:
        baseline = before
        baseline_identity = identities
    assert before == baseline and identities == baseline_identity
    booleans_before = [raw_exact(s, s["expected"], "exact") for s in before]
    booleans_after = [raw_exact(s, s["expected"], "exact") for s in after]
    for samples in [vqa_before, vqa_after]:
        for i, (s, record) in enumerate(zip(samples, vqa_splits["test"], strict=True)):
            assert (s["row"],s["family"],s["question"],s["target"]) == (i,record["family"],record["question"],record["answer"])
            raw_exact(s,s["target"],"exact_match")
    counts = {"text_before":sum(booleans_before),"text_after":sum(booleans_after),
              "vqa_before":sum(s["exact_match"] for s in vqa_before),
              "vqa_after":sum(s["exact_match"] for s in vqa_after)}
    for metric, phase in [("text_before","text_before"),("text_after","text_after")]:
        assert pointer(base + "/" + phase + "/matches") == counts[metric]
        assert pointer(base + "/" + phase + "/records") == len(before)
        assert pointer(base + "/" + phase + "/exact_match") == counts[metric]/len(before)
    assert pointer(base + "/test/correct") == counts["vqa_after"]
    assert pointer(base + "/test/examples") == len(vqa_after)
    assert pointer(base + "/test/exact_match") == counts["vqa_after"]/len(vqa_after)
    training_fields = ["config", "modal_config", "trainable_parameters", "parameters", "steps",
                       "effective_targets", "effective_tokens", "weights_changed", "nonzero_gradient_seen"]
    training = {k:pointer(base+"/training/"+k) for k in training_fields}
    history = pointer(base + "/training/history")
    assert len(history) == training["steps"] == 160
    assert [h["step"] for h in history] == list(range(1,training["steps"]+1))
    assert sum(h["effective_targets"] for h in history) == training["effective_tokens"] == training["effective_targets"]
    replay_probability = pointer(base+"/replay_probability_per_example")
    scope = pointer(base+"/freeze_scope")
    # Reproduce the recorded sampler and answer-position count without model calls.
    n_text,n_vqa,positions_text,positions_vqa = 0,0,0,0
    per_step_counts = []
    for step in range(training["steps"]):
        rng = random.Random(seed + step)
        step_count = 0
        for _ in range(4):
            if text_splits["train"] and rng.random() < replay_probability:
                record = rng.choice(text_splits["train"])
                x,y = render_chat(record["messages"])
                count = int((y != -100).sum())
                n_text += 1;positions_text += count
            else:
                record = rng.choice(vqa_splits["train"])
                ids,labels,count = _sequence(record,SimpleNamespace(device="cpu"))
                assert int((labels != -100).sum()) == count
                n_vqa += 1;positions_vqa += count
            step_count += count
        per_step_counts.append(step_count)
    assert per_step_counts == [h["effective_targets"] for h in history]
    result["variants"][name] = {
        "counts":counts,"vqa_denominator":12,"text_denominator":10,"freeze_scope":scope,
        "replay_probability_per_example":replay_probability,"training":training,
        "reconstructed_sampling":{"text_examples":n_text,"vqa_examples":n_vqa,
                                  "text_effective_positions":positions_text,"vqa_effective_positions":positions_vqa,
                                  "text_example_share":n_text/(n_text+n_vqa),
                                  "text_effective_position_share":positions_text/(positions_text+positions_vqa)},
        "text_transitions": [{"row":i,"prompt":b["messages"][0]["content"],"expected":b["expected"],
                              "before":b["generated"],"after":a["generated"],"before_exact":bb,"after_exact":aa}
                             for i,(b,a,bb,aa) in enumerate(zip(before,after,booleans_before,booleans_after,strict=True))],
        "eos_text_before":sum(s["eos"] for s in before),"eos_text_after":sum(s["eos"] for s in after),
        "eos_vqa_after":sum(s["eos"] for s in vqa_after),
        "lost_text_rows":[i for i,(b,a) in enumerate(zip(booleans_before,booleans_after)) if b and not a],
        "gained_text_rows":[i for i,(b,a) in enumerate(zip(booleans_before,booleans_after)) if a and not b],
    }
expected_table = {"projector_only":(3,5),"partial":(12,2),"all":(9,1),"all_replay":(9,4)}
for name,(v,t) in expected_table.items():
    assert result["variants"][name]["counts"]["vqa_after"] == v
    assert result["variants"][name]["counts"]["text_after"] == t
    assert result["variants"][name]["counts"]["text_before"] == 5
result["scope_flag_probe"] = {}
for scope in ["projector", "partial", "all"]:
    torch.manual_seed(0)
    model=MultiModalLM(TinyLM(ModelConfig(width=8,layers=2)))
    _freeze(model,scope)
    enabled=[n for n,p in model.named_parameters() if p.requires_grad]
    if scope == "projector": assert all(n.startswith("image_projector.") for n in enabled)
    if scope == "partial": assert all(n.startswith(("image_projector.","language.blocks.1.")) for n in enabled)
    if scope == "all": assert len(enabled) == len(list(model.named_parameters()))
    result["scope_flag_probe"][scope] = {"trainable_names":enabled,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)}
# Execute the original fence and a source-preserving bounded recipe change.
code = (OUT/"fence-1.py").read_bytes()
def run_fence(data):
    stream=io.StringIO();ns={"__name__":"__main__"}
    with contextlib.redirect_stdout(stream):exec(compile(data,"11.7-original-or-recipe-only-variation","exec"),ns)
    return {"stdout":stream.getvalue(),"ratios":ns["ratios"],"recorded":ns["recorded"],"total_positions":ns["total_positions"]}
original=run_fence(code)
changed=code.replace('750, "圖文有效位置": 250'.encode(), '500, "圖文有效位置": 500'.encode())
assert changed != code
(OUT/"fence-1-500-500.py").write_bytes(changed)
variation=run_fence(changed)
assert original["ratios"] == {"文字有效位置":0.75,"圖文有效位置":0.25}
assert variation["ratios"] == {"文字有效位置":0.5,"圖文有效位置":0.5}
assert original["recorded"] == variation["recorded"]
assert original["total_positions"] == variation["total_positions"] == 1000
result["fence_checks"]={"original":original,"recipe_500_500":variation,"score_table_unchanged":True}
# Weighted average loss can fall solely by changing the weight of an easier group.
result["mixture_loss_counterexample"]={"fixed_text_mean_loss":1.0,"fixed_vqa_mean_loss":4.0,
                                      "text_share_0_25":0.25*1+0.75*4,
                                      "text_share_0_75":0.75*1+0.25*4,
                                      "interpretation":"3.25 -> 1.75 with unchanged per-group loss; a numerical possibility, not an observed model result"}
result["inspected_json_pointers"] = list(dict.fromkeys(inspected))
result["checks"] = "All assertions passed. No model training, saved-model reevaluation, GPU, or .pt output."
(OUT/"independent_cpu_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
print(json.dumps({"checks":result["checks"],"environment":result["environment"],
                  "raw_json_sha256":result["raw_json_sha256"],"fence_checks":result["fence_checks"],
                  "variants":{n:{k:v[k] for k in ["counts","training","reconstructed_sampling","lost_text_rows","gained_text_rows"]} for n,v in result["variants"].items()},
                  "inspected_pointer_count":len(result["inspected_json_pointers"])},ensure_ascii=False,indent=2))
