"""Independent bounded CPU checks; no model load, neural training, or downloads."""
import ast
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.data import IGNORE, SPECIALS, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import masked_loss

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert not torch.cuda.is_available() and torch.version.cuda is None
environment = {
    "python": platform.python_version(), "python_executable": sys.executable,
    "torch": str(torch.__version__), "torch_git_version": torch.version.git_version,
    "device": "cpu", "cuda_available": str(torch.cuda.is_available()),
    "cuda_build": str(torch.version.cuda), "threads": str(torch.get_num_threads()),
    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "unset"),
}
(BASE / "probe-environment.json").write_text(json.dumps(environment, indent=2) + "\n")

tok = ByteTokenizer()
question = "2+2=?"
answers = ["4", "4，就像兩雙筷子共有四根。", "4，共有四個", "", "4🙂"]
counts = []
for answer in answers:
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    inputs, labels = render_chat(messages)
    # Independent byte accounting: BOS, user role, question, user EOS,
    # assistant role, answer, assistant EOS, followed by one position shift.
    assert len(inputs) == 5 + len(question.encode("utf-8")) + len(answer.encode("utf-8")) - 1
    assert int((labels != IGNORE).sum()) == len(answer.encode("utf-8")) + 1
    assert labels[7].item() == IGNORE
    assert labels[8].item() == (tok.encode(answer)[0] if answer else tok.eos_id)
    assert labels[-1].item() == tok.eos_id
    assert inputs[-1].item() == (tok.encode(answer)[-1] if answer else tok.assistant_id)
    counts.append({"answer": answer, "answer_utf8_bytes": len(answer.encode("utf-8")),
                   "input_shape": list(inputs.shape), "target_count": int((labels != IGNORE).sum()),
                   "inputs": inputs.tolist(), "labels": labels.tolist()})
assert [(r["input_shape"][0], r["target_count"]) for r in counts] == [(10,2),(46,38),(25,17),(9,1),(14,6)]
try:
    render_chat([{"role":"user", "content":question}])
except ValueError as e:
    missing_assistant = str(e)
else:
    raise AssertionError("Missing assistant must fail")
batch = [render_chat([{"role":"user", "content":question}, {"role":"assistant", "content":a}]) for a in answers[:3]]
inputs, labels, valid = pad_batch(batch)
assert list(inputs.shape) == [3,46]
assert (labels != IGNORE).sum(1).tolist() == [2,38,17]
assert (labels != IGNORE).sum().item() == 57

# Check the actual loss contract using deterministic direct logits, not a language model.
logits = torch.zeros((3,46,264), dtype=torch.float64, requires_grad=True)
loss = masked_loss(logits, labels)
assert abs(loss.item() - math.log(264)) < 1e-12
loss.backward()
mask = labels != IGNORE
assert torch.count_nonzero(logits.grad[~mask]).item() == 0
assert torch.count_nonzero(logits.grad[mask]).item() > 0
target_gradient = logits.grad[mask].gather(1, labels[mask].unsqueeze(1))
assert torch.allclose(target_gradient, torch.full_like(target_gradient,(1/264-1)/57),atol=1e-14,rtol=0)
changed = logits.detach() - 0.1 * logits.grad
before_prob = logits.detach().softmax(-1)[mask].gather(1,labels[mask,None])
after_prob = changed.softmax(-1)[mask].gather(1,labels[mask,None])
assert (after_prob > before_prob).all().item()
after_loss = masked_loss(changed, labels).item()
assert after_loss < loss.item()
try:
    masked_loss(torch.zeros(1,2,264), torch.full((1,2), IGNORE))
except ValueError as e:
    all_ignored = str(e)
else:
    raise AssertionError("All-ignored labels must fail")

# Load precisely selected definitions from the original experiment version.
# No historical train/evaluate function is executed.
namespace = {"random":random, "json":json, "hashlib":hashlib, "ByteTokenizer":ByteTokenizer,
             "render_chat":render_chat, "shifted":None, "pad_batch":pad_batch, "IGNORE":IGNORE}
def definitions(path, names):
    original = BASE / "inputs/historical" / path
    tree = ast.parse(original.read_text())
    chosen = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in chosen} == set(names)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(original),"exec"),namespace)

definitions("scripts/course_experiments/common.py", ["split_records","records_sha256","text_examples"])
definitions("scripts/course_experiments/text.py", ["arithmetic_records"])
definitions("scripts/course_experiments/behavior.py", ["_conversation","_style_record","_style_metrics"])
raw_report = BASE / "inputs/current/docs/course-experiments/results/style.json"
report = json.loads(raw_report.read_text())
results = report["results"]
assert report["revision"] == "ae7bbbf95537d228a44810041d2a9e978360d369"
parts = namespace["split_records"](namespace["arithmetic_records"](), seed=report["seed"])
assert [len(parts[k]) for k in ("train","validation","test")] == [49,8,7]
families = [{r["family"] for r in parts[k]} for k in ("train","validation","test")]
assert all(not (a & b) for i,a in enumerate(families) for b in families[i+1:])
assert question not in [r["messages"][0]["content"] for r in parts["train"]]
assert question in [r["messages"][0]["content"] for r in parts["test"]]
dataset_fingerprints = []
def save_verify(prefix, values, manifest):
    for split, rows in values.items():
        encoded = "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        assert digest == manifest[split]["sha256"]
        output = BASE / "reconstructed-original-inputs" / prefix / (split + ".jsonl")
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_bytes(encoded)
        dataset_fingerprints.append({"path":str(output.relative_to(ROOT)),"sha256":digest,"records":len(rows)})
save_verify("arithmetic",parts,results["arithmetic_data"])
summaries = {}
for style in ("concise","vivid"):
    style_parts = {split:[namespace["_style_record"](row,style,False) for row in rows] for split,rows in parts.items()}
    run = results["default_style_runs"][style]
    save_verify("default-"+style+"-data",style_parts,run["data"])
    training = run["training"]
    assert training["records"] == 49 and training["steps"] == 450
    assert namespace["records_sha256"](style_parts["train"]) == training["records_sha256"]
    examples = namespace["text_examples"](style_parts["train"], mode="sft", max_length=128)
    sampler = random.Random(report["seed"])
    effective = sum(int((y != IGNORE).sum()) for _ in range(450) for _,y in sampler.choices(examples,k=16))
    assert effective == training["effective_tokens"] == {"concise":16591,"vivid":318991}[style]
    comparisons = {}
    for phase in ("before","after"):
        evaluation = copy.deepcopy(run[phase]["test"])
        checked = namespace["_style_metrics"](evaluation,style_parts["test"])
        assert checked["rubric"] == run[phase]["test"]["rubric"]
        independent_content = independent_style = 0
        for row,sample in zip(style_parts["test"],evaluation["samples"],strict=True):
            ids = sample["generated_ids"]
            assert ids[-1] == tok.eos_id and all(x >= 8 for x in ids[:-1])
            assert tok.decode(ids) == sample["generated"]
            content = sample["generated"].split("，",1)[0].strip() == str(row["a"]+row["b"])
            styled = sample["generated"].strip().isdigit() if style=="concise" else "像把兩組積木合在一起再數" in sample["generated"]
            independent_content += content
            independent_style += styled
        assert independent_content == checked["rubric"][style]["content_correct"]
        assert independent_style == checked["rubric"][style]["style_correct"]
        comparisons[phase] = {"content_correct":independent_content,"style_correct":independent_style,"records":7}
    assert comparisons["after"] == {"content_correct":0,"style_correct":7,"records":7}
    sample = next(s for s in run["after"]["test"]["samples"] if s["messages"][0]["content"] == question)
    assert sample["generated"] == ("3" if style=="concise" else "3，像把兩組積木合在一起再數。")
    summaries[style] = {"steps":450,"batch_size":16,"sampled_examples":7200,"effective_targets":effective,
                        "comparison":comparisons,"2+2_generated":sample["generated"]}
assert results["content_evaluation"]["test"]["matches"] == 0
assert all(s["generated"] != s["expected"] for s in results["content_evaluation"]["test"]["samples"])
assert summaries["vivid"]["effective_targets"] - summaries["concise"]["effective_targets"] == 7200*42
output = {"counts":counts,"no_assistant_error":missing_assistant,"batch_shape":[3,46],
          "effective_target_count":57,"loss":loss.item(),"expected_uniform_loss":math.log(264),
          "after_direct_gradient_update_loss":after_loss,"all_ignored_error":all_ignored,
          "loss_gradient_mask_exact_zero":True,"gradient_tolerance":"absolute 1e-14; rtol=0",
          "split_records":{"train":49,"validation":8,"test":7},"split_families":[len(s) for s in families],
          "base_test_matches":0,"base_test_records":7,"historical_style_runs":summaries,
          "dataset_fingerprints":dataset_fingerprints,
          "scope":"Rechecked existing GPU report, recreated deterministic inputs/token accounting only; no historical model generation or retraining."}
(BASE / "probe-results.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in output.items() if k not in ['counts','dataset_fingerprints']},ensure_ascii=False,indent=2))
