import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import re
import sys

B = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(B / "freeze/implementation"))
import numpy as np
import soundfile as sf
import torch

from tiny_perceptron.data import render_chat
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM, mel_filter_bank
from tiny_perceptron.alignment import sequence_log_probability
from scripts.audio_utils import resample_waveform

torch.set_num_threads(1)
ev = Path(__file__).parent
report = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
          "numpy": np.__version__, "soundfile": sf.__version__, "device": "cpu"},
          "command": "/workspace/tiny-perceptron-vlm/.venv/bin/python evidence/technical-modalities/run_checks.py",
          "no_training": True, "checks": {}}

for page in ["10.11", "11.5", "11.6", "12.2", "12.6", "13.3"]:
    source = B / "freeze/sources" / (page + ".md")
    outputs = []
    for code in re.findall(r"```python\n(.*?)```", source.read_text(), re.S):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, str(source), "exec"), {})
        outputs.append(stream.getvalue())
    report["checks"]["snippet-" + page] = {"stdout": outputs,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}

s = torch.tensor([[2., .1], [.4, 1.8]], requires_grad=True)
i = torch.nn.functional.cross_entropy(s, torch.arange(2))
t = torch.nn.functional.cross_entropy(s.T, torch.arange(2))
w = (2*i+t)/3
w.backward()
report["checks"]["weighted-10.11"] = {"i2t": i.item(), "t2i": t.item(), "mean": ((i+t)/2).item(),
        "weighted": w.item(), "gradient_shape": list(s.grad.shape), "gradient": s.grad.tolist()}

counts = {}
for layers in [1, 2]:
    model = MultiModalLM(TinyLM(ModelConfig(width=8, layers=layers)))
    rows = {}
    for mode in ["projector", "partial", "all"]:
        model.requires_grad_(mode == "all")
        model.image_projector.requires_grad_(True)
        if mode == "partial": model.language.blocks[-1].requires_grad_(True)
        rows[mode] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts[str(layers)] = rows
report["checks"]["layers-exercise-11.5"] = counts

report["checks"]["mel-12.6"] = {
    "frequencies_hz": [0, 700, 2100], "mels": [2595*math.log10(1+f/700) for f in [0,700,2100]],
    "32_bands_shape": list(mel_filter_bank(bands=32).shape),
    "32_output_shape": list((mel_filter_bank(bands=32) @ torch.ones(201,3)).shape),
    "toy_band_output": (torch.tensor([[1,.5,0],[0,.5,1]]) @ torch.tensor([4.,2.,1.])).tolist(),
    "16_bands_rank": int(torch.linalg.matrix_rank(mel_filter_bank())),
    "16_overlap_frequency_count": int(((mel_filter_bank()>0).sum(0)>1).sum()),
    "16_each_band_nonempty": bool((mel_filter_bank().sum(-1)>0).all())}

logits = torch.tensor([[[0.,2.,0.],[2.,0.,0.],[0.,0.,2.]]])
full_score = sequence_log_probability(logits, torch.tensor([[1,0,2]]))
report["checks"]["extra-label-13.3"] = {"sum":full_score.item(), "mean":full_score.item()/3,
        "probability":full_score.exp().item()}

# Transparent alias derivation: cos(2*pi*(Fs-f)*n/Fs) == cos(2*pi*f*n/Fs).
n = torch.arange(1600, dtype=torch.float64)
lower = torch.cos(2*math.pi*1000*n/8000)
upper = torch.cos(2*math.pi*7000*n/8000)
report["checks"]["alias-12.2"] = {"rate":8000, "frequencies":[1000,7000],
       "max_absolute_error":(lower-upper).abs().max().item(),
       "duration_source":4591/8000, "duration_resampled":9182/16000}

source_audio = Path("/workspace/tiny-perceptron-vlm/data/training/fsdd-initial/recordings/0_jackson_5.wav")
if source_audio.exists():
    samples, rate = sf.read(source_audio, dtype="float32")
    code_path = B / "freeze/implementation/scripts/course_experiments/modalities.py"
    tree = ast.parse(code_path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name=="_resample_8_to_16")
    namespace = {"torch":torch, "math":math}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(code_path),"exec"),namespace)
    converted = namespace["_resample_8_to_16"](samples)
    buffer = io.BytesIO()
    sf.write(buffer, converted, 16000, subtype="FLOAT", format="WAV")
    report["checks"]["fsdd-example-12.2"] = {
        "path":str(source_audio), "source_sha256":hashlib.sha256(source_audio.read_bytes()).hexdigest(),
        "source_rate":rate,"source_samples":len(samples), "target_rate":16000,"target_samples":len(converted),
        "duration":len(converted)/16000, "regenerated_wav_sha256":hashlib.sha256(buffer.getvalue()).hexdigest(),
        "current_resampler_max_error":float(np.max(np.abs(converted-resample_waveform(samples,rate,16000))))}

# Reconstruct only data and sampler counts from the recorded original DPO source, with no model updates.
historic = ev / "historical/dpo/scripts/course_experiments/behavior.py"
tree = ast.parse(historic.read_text())
parts_function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="_preference_parts")
namespace = {}
exec(compile(ast.Module(body=[parts_function], type_ignores=[]),str(historic),"exec"),namespace)
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records
parts = namespace["_preference_parts"](split_records(arithmetic_records(),seed=42))
raw_dpo = json.loads((B/"freeze/technical-data/docs/course-experiments/results/dpo.json").read_text())
identities = {}
for split, rows in parts.items():
    raw = "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows).encode()
    sha = hashlib.sha256(raw).hexdigest()
    identities[split] = {"rows":len(rows),"sha256":sha,
                "matches_record":sha==raw_dpo["results"]["data"][split]["sha256"]}
counts_by_example = [sum(int((render_chat([{"role":"user","content":r["prompt"]},
        {"role":"assistant","content":r[side]}])[1]!=-100).sum()) for side in ["chosen","rejected"])
        for r in parts["train"]]
sampler = random.Random(42)
exposure = sum(sum(sampler.choices(counts_by_example,k=8)) for _ in range(250))
report["checks"]["dpo-denominator-13.3"] = {"data_identities":identities,"updates":250,
       "sampled_pairs_per_update":8,"exposure_both_answer_sides_including_eos":exposure,
       "answer_examples":{answer:int((render_chat([{"role":"user","content":"2+2=?"},
                 {"role":"assistant","content":answer}])[1]!=-100).sum()) for answer in ["4","10"]}}

# Recount the published samples and exposure logs; no new model inference.
empirical = {}
for name in ["contrastive", "vqa"]:
    file = B/"freeze/technical-data/docs/course-experiments/results"/(name+".json")
    data = json.loads(file.read_text())
    res = data["results"]
    section = {"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"recorded_revision":data["revision"],
            "seed":data["seed"],"device":data["device"],"variants":{},"splits":{}}
    for split, entry in res["data"]["splits"].items():
        rows=entry["records"]
        section["splits"][split] = {"rows":len(rows),"offsets":sorted({r["offset"] for r in rows}),
          "families":len({r["family"] for r in rows}), "known_colour_shapes":sorted({r["color"]+":"+r["shape"] for r in rows})}
    families={s:{r["family"] for r in e["records"]} for s,e in res["data"]["splits"].items()}
    section["split_family_disjoint"] = all(not families[a]&families[b] for a,b in [("train","validation"),("train","test"),("validation","test")])
    for variant, entry in res["variants"].items():
        training=entry["training"]
        v = {"trainable_parameters":training["trainable_parameters"],"steps":training["steps"],
             "training_exposure":training["effective_targets"],
             "history_recount":sum(row["effective_targets"] for row in training["history"]),"evaluations":{}}
        for split in ["validation","test"]:
            result=entry[split]
            if name=="contrastive":
                v["evaluations"][split]={direction:{"correct":sum(row["correct"] for row in result[direction]),
                         "denominator":len(result[direction])} for direction in ["image_samples","text_samples"]}
            else:
                v["evaluations"][split]={"correct":sum(row["exact_match"] for row in result["samples"]),
                    "denominator":len(result["samples"]), "valid_answer_targets":result["effective_tokens"],
                    "eos_successes":sum(row["eos"] for row in result["samples"])}
        if name=="vqa" and variant in ["projector_only","partial","all","direct_vqa"]:
            total=0
            for step in range(training["steps"]):
                rng=random.Random(data["seed"]+step)
                for _ in range(4):
                    # run_vqa passes a nonempty replay pool even when its probability is 0.
                    # The existing implementation still consumes rng.random() in that condition.
                    if variant != "direct_vqa": rng.random()
                    total+=len(rng.choice(res["data"]["splits"]["train"]["records"])["answer"].encode())+1
            v["sampler_recount_with_eos"] = total
        section["variants"][variant]=v
    if name=="vqa":
        align=res["two_stage_alignment_training"]
        section["two_stage"]={"alignment_targets":align["effective_targets"],
            "alignment_history_recount":sum(row["effective_targets"] for row in align["history"]),
            "qa_targets":res["variants"]["all"]["training"]["effective_targets"],
            "total":align["effective_targets"]+res["variants"]["all"]["training"]["effective_targets"],
            "direct_has_text_after": "text_after" in res["variants"]["direct_vqa"]}
    empirical[name]=section
report["checks"]["historical-recounts"] = empirical

manifest=json.loads((B/"manifest.json").read_text())
imported={}
for name,module in list(sys.modules.items()):
    path=getattr(module,"__file__",None)
    if path and str(B/"freeze/implementation") in path and path.endswith(".py"):
        relative=str(Path(path).relative_to(B/"freeze/implementation"))
        actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        expected=manifest["implementation_sha256"].get(relative)
        imported[relative]={"sha256":actual,"manifest_sha256":expected,"matches":actual==expected}
report["imported_frozen_source_identities"]=imported

(ev / "execution.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(report,ensure_ascii=False,indent=2))
