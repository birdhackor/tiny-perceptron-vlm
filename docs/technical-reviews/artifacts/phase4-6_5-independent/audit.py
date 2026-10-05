"""Bounded CPU audit: historical input/provenance, arithmetic, masks, no models or training."""
import ast
import hashlib
import json
import math
import platform
import random
import shutil
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
import tokenizers

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
HIST = OUT / "historical"
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def exact_functions(path, names, ns):
    """Execute unchanged AST nodes from the preserved historical files, excluding long entrypoints."""
    text = path.read_text()
    tree = ast.parse(text)
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    return {n.name: {"path": str(path.relative_to(ROOT)), "line": n.lineno, "end_line": n.end_lineno} for n in selected}


report = json.loads((OUT / "inputs/docs/course-experiments/results/tokenizer.json").read_text())
ns = {"torch": torch, "F": F, "hashlib": hashlib, "json": json, "random": random}
# data.py consists solely of imports, constants, definitions; no training entrypoint.
exec(compile((HIST / "tiny_perceptron/data.py").read_bytes(), str(HIST / "tiny_perceptron/data.py"), "exec"), ns)
contracts = {}
contracts.update(exact_functions(HIST / "scripts/course_experiments/text.py", ["_json_bytes", "_digest", "_deduplicate_text", "_utf8_prefix", "_BPE"], ns))
contracts.update(exact_functions(HIST / "scripts/course_experiments/common.py", ["split_records", "text_examples"], ns))
contracts.update(exact_functions(HIST / "tiny_perceptron/model.py", ["loss_sum", "masked_loss"], ns))
write("executed-contracts.json", contracts)

# Candidate files are accepted only on a byte hash match with the original experiment.
inputs = []
def preserve(candidate, expected, relative):
    candidate = ROOT / candidate
    observed = sha(candidate)
    assert expected == observed, (candidate, expected, observed)
    dest = OUT / "inputs" / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, dest)
    inputs.append({"candidate_path": str(candidate.relative_to(ROOT)), "expected_sha256": expected,
                   "observed_sha256": observed, "match": True, "snapshot": str(dest.relative_to(ROOT)),
                   "bytes": dest.stat().st_size})
    return dest

source_files = {}
for asset in report["assets"]:
    basename = "tinystories-train-512.jsonl" if asset["id"] == "tinystories" else "chinese-classical-train-365.jsonl"
    entry = next(f for f in asset["files"] if f["path"].endswith(basename))
    source_files[asset["id"]] = preserve("data/training/" + entry["path"], entry["sha256"], "raw/" + basename)
artifact_hash = {a["path"]: a["sha256"] for a in report["artifacts"]}
base = "outputs/text-behavior-interface-check/tokenizer/"
split_files = {s: preserve(base + "data/" + s + ".jsonl", report["results"]["data"][s]["sha256"], "split/" + s + ".jsonl")
               for s in ("train", "validation", "test")}
bpe_path = preserve(base + "tokenizer-bpe512.json", artifact_hash["tokenizer-bpe512.json"], "tokenizer-bpe512.json")
byte_path = preserve(base + "tokenizer-byte.json", artifact_hash["tokenizer-byte.json"], "tokenizer-byte.json")
write("input-provenance.json", {"result_revision": report["revision"], "result_sha256": sha(OUT / "inputs/docs/course-experiments/results/tokenizer.json"),
      "policy": "Use only hash-matching candidates, snapshot every required input under docs; no weights copied or evaluated", "files": inputs})
load_rows = ns["load_jsonl"]
raw = load_rows(source_files["tinystories"])[:96] + load_rows(source_files["chinese-poetry"])[:96]
parts_complete = ns["split_records"](ns["_deduplicate_text"](raw), seed=report["seed"])
parts = {s: [{**row, "text": ns["_utf8_prefix"](row["text"], 256), "complete_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()}
             for row in rows] for s, rows in parts_complete.items()}
for s, rows in parts.items():
    raw_split = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    assert raw_split == split_files[s].read_bytes(), s
assert not (set(r["family"] for r in parts["train"]) & set(r["family"] for r in parts["validation"]))
assert not (set(r["family"] for r in parts["train"]) & set(r["family"] for r in parts["test"]))
assert not (set(r["family"] for r in parts["validation"]) & set(r["family"] for r in parts["test"]))
tok_byte = ns["ByteTokenizer"]()
tok_bpe = ns["_BPE"](Tokenizer.from_file(str(bpe_path)))
assert json.loads(byte_path.read_text()) == tok_byte.state()
for s, rows in parts.items():
    for row in rows:
        for tok in [tok_byte, tok_bpe]:
            assert tok.decode(tok.encode(row["text"])) == row["text"]

# Sampler is created once, never reseeded between batches.
sampler = random.Random(report["seed"])
schedule = [sampler.choices(range(len(parts["train"])), k=8) for _ in range(400)]
schedule_hash = ns["_digest"](schedule)
assert schedule_hash == report["results"]["raw_document_schedule_sha256"]
exposure = sum(len(parts["train"][i]["text"].encode()) for batch in schedule for i in batch)
assert exposure == 664759
metrics = {}
for name, tok in [("byte256", tok_byte), ("bpe512", tok_bpe)]:
    run = report["results"]["runs"][name]
    examples = ns["text_examples"](parts["train"], "text", 384, tok)
    assert len(examples) == 153
    assert run["training"]["steps"] == 400 and run["training_raw_utf8_bytes_exposed"] == exposure
    train_targets = sum(len(examples[i][1]) for batch in schedule for i in batch)
    train_padded = sum(max(len(examples[i][0]) for i in batch) * len(batch) for batch in schedule)
    split_metrics = {}
    for s in ["validation", "test"]:
        rows = parts[s]
        byte_count = sum(len(r["text"].encode()) for r in rows)
        count = sum(len(tok.encode(r["text"])) + 1 for r in rows)
        phases = {}
        for phase in ["before", "after"]:
            result = run[phase][s]
            assert result["records"] == len(rows) and result["raw_utf8_bytes"] == byte_count
            assert result["effective_tokens"] == count
            assert result["skipped"] == [] and len(result["samples"]) == len(rows)
            assert [v["row"] for v in result["samples"]] == list(range(len(rows)))
            assert [v["prompt"] for v in result["samples"]] == [r["text"][:4] for r in rows]
            nll_sum = result["mean_token_nll"] * count
            bpb = nll_sum / (byte_count * math.log(2))
            err = abs(bpb - result["bpb_including_eos_boundary_targets"])
            assert err < 2e-14
            phases[phase] = {"nll_sum_reconstructed_from_mean": nll_sum, "bpb_recomputed": bpb, "bpb_recorded": result["bpb_including_eos_boundary_targets"], "error": err,
                             "mean_token_nll": result["mean_token_nll"]}
        split_metrics[s] = {"records": len(rows), "utf8_bytes": byte_count, "text_tokens": count-len(rows), "eos_targets": len(rows), "effective_targets": count, "phases": phases}
    metrics[name] = {"vocab_size": tok.vocab_size, "parameters_recorded": run["parameters"], "training_supervision_targets": train_targets,
                     "training_padded_input_positions": train_padded, "raw_bytes_exposed": exposure, "updates": 400, "batch_size": 8, "splits": split_metrics}
assert metrics["byte256"]["training_supervision_targets"] != metrics["bpe512"]["training_supervision_targets"]
assert metrics["byte256"]["training_padded_input_positions"] != metrics["bpe512"]["training_padded_input_positions"]
assert 57696-41824 == 2*(512-264)*32

# Execute original loss contracts on synthetic logits: masked context/PAD has zero contribution, EOS remains a target.
prob = torch.tensor([[[.7,.2,.1],[.1,.3,.6],[.2,.4,.4]]],dtype=torch.float64)
labels = torch.tensor([[0,2,-100]])
total,count = ns["loss_sum"](prob.log(), labels)
mean = ns["masked_loss"](prob.log(),labels)
expected = -math.log(.7)-math.log(.6)
assert int(count)==2 and abs(float(total)-expected)<1e-14
assert abs(float(mean)*int(count)-float(total))<1e-14
prob[:,2] = torch.tensor([.99,.005,.005])
assert float(ns["loss_sum"](prob.log(),labels)[0]) == float(total)
example = ns["text_examples"]([{"text":"ABCD"}], "text", 3, tok_byte)
scored = [int(y) for x, ys in example for y in ys]
assert scored == tok_byte.encode("ABCD") + [tok_byte.eos_id]
x,y,valid = ns["pad_batch"]([ns["shifted"]([1,8,2]), ns["shifted"]([1,9,10,2])])
assert y.tolist()==[[8,2,-100],[9,10,2]] and not bool(valid[0,-1])
window_events = ["B","C"] + ["D"]
assert len(window_events)==len(set(window_events))==3
toy = {"bytes":len("貓🙂".encode()),"unicode":[f"U+{ord(c):04X}" for c in "貓🙂"],"individual_bytes":[len(c.encode()) for c in "貓🙂"],
       "a":5/(7*math.log(2)),"b":4/(7*math.log(2)),"math_log_half":-math.log(.5),"one_bit":-math.log(.5)/math.log(2),
       "token_mean_a":.6,"token_mean_b":1.,"total_a":10*.6,"total_b":5*1.,"chain_rule_sum":-math.log(.4)-math.log(.5),"chain_rule_product":-math.log(.4*.5)}
assert toy["bytes"]==7 and toy["individual_bytes"]==[3,4] and toy["one_bit"]==1.
assert abs(toy["chain_rule_sum"]-toy["chain_rule_product"])<1e-15
toy["double_nll"]=[(v*2)/(7*math.log(2)) for v in [5.,4.]]
toy["double_text_same_nll"]=[v/(len("貓🙂貓🙂".encode())*math.log(2)) for v in [5.,4.]]
assert toy["double_nll"]==[2*toy["a"],2*toy["b"]]
assert toy["double_text_same_nll"]==[toy["a"]/2,toy["b"]/2]
wrong_id = tok_byte.encode("A")[0]
id_mapping = {"text":"A","byte_ids":tok_byte.encode("A"),"bpe_ids":tok_bpe.encode("A"),"byte_id_as_bpe":tok_bpe.decode([wrong_id]),"byte_id_in_bpe_range":wrong_id<tok_bpe.vocab_size}
assert id_mapping["byte_ids"]!=id_mapping["bpe_ids"] and id_mapping["byte_id_as_bpe"]!="A"

# Figure bars use an implicit linear 80 pixels per numeric unit from x=140. Check nearest-pixel rounding.
svg = ET.parse(OUT / "inputs/course/figures/tokenizer_common_scale.svg").getroot()
bars = [e for e in svg.iter() if e.tag.endswith('rect') and e.get('x')=='140']
assert len(bars)==4
values=[metrics['byte256']['splits']['test']['phases']['after']['mean_token_nll'],metrics['bpe512']['splits']['test']['phases']['after']['mean_token_nll'],
        metrics['byte256']['splits']['test']['phases']['after']['bpb_recomputed'],metrics['bpe512']['splits']['test']['phases']['after']['bpb_recomputed']]
figure = [{"y":float(e.get('y')),"width":float(e.get('width')),"value":v,"expected_width_80px_per_unit":80*v,"error_px":abs(float(e.get('width'))-80*v)} for e,v in zip(bars,values,strict=True)]
assert all(v['error_px']<.5 for v in figure)
byte_samples=report['results']['runs']['byte256']['after']['test']['samples']
bpe_samples=report['results']['runs']['bpe512']['after']['test']['samples']
generation=[{'row':a['row'],'byte_chars':len(a['generated']),'bpe_chars':len(b['generated']),'byte_decoded_utf8_bytes':len(a['generated'].encode()),'bpe_decoded_utf8_bytes':len(b['generated'].encode())} for a,b in zip(byte_samples,bpe_samples,strict=True)]
per_row_fields=sorted(byte_samples[0])
results={"scope":"CPU arithmetic/contract/data-provenance audit of historical measurement. No training, checkpoint loads, model inference, downloads of training data or weights.",
         "historical_revision":report['revision'],"raw_inputs_reconstructed_splits":True,"schedule_sha256":schedule_hash,"raw_bytes_exposed":exposure,"metrics":metrics,
         "toy":toy,"masked_loss":{"total":float(total),"count":int(count),"mean":float(mean),"expected":expected,"padding_labels":y.tolist(),"chunk_scored_ids":scored,"window_scored_events":window_events},
         "id_mapping":id_mapping,"figure_bars":figure,"generation_lengths_from_saved_decoded_strings":generation,
         "generation_bpe_longer_chars_count":sum(g['bpe_chars']>g['byte_chars'] for g in generation),"per_row_fields_in_linked_result":per_row_fields,
         "per_document_nll_available_in_linked_result":any('nll' in k or 'loss' in k or 'bpb' in k for k in per_row_fields),
         "long_recipe_inspection":"training.md T.5 points to CUDA tokenizer entrypoint; historical run_tokenizer uses 400 updates, 8 excerpts/batch, width32/layers1/max384/lr0.003/seed42. Inspected, not run; replaced with hash reconstruction and synthetic loss checks."}
write('audit-results.json',results)
write('audit-environment.json',{'python':sys.version,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'tokenizers':tokenizers.__version__,'device':'cpu','cuda_build':str(torch.version.cuda),'platform':platform.platform(),'python_executable':sys.executable})
print(json.dumps(results,ensure_ascii=False,indent=2))
