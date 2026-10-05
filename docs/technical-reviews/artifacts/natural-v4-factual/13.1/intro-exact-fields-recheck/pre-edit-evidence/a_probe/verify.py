"""Independent bounded factual probe for section 13.1; no full training replay."""
from pathlib import Path
import contextlib
import copy
import hashlib
import io
import json
import platform
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import behavior
from scripts.course_experiments.common import Context, split_records
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.posttraining import build_records
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.posttraining import FiniteResponsePolicy

OUT = Path(__file__).resolve().parent
WORK = ROOT / "outputs/natural-v4/factual-research/13.1/cpu-probe"
WORK.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
environment = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
               "cuda_available": str(torch.cuda.is_available()), "cpu_threads": str(torch.get_num_threads())}
print("ENVIRONMENT", json.dumps(environment, ensure_ascii=False))
assert torch.__version__ == "2.14.1+cpu"

raw = (ROOT / "course/chapters/13.md").read_bytes()
matches = list(re.finditer(rb"^## ", raw, re.M))
start = next(m.start() for m in matches if raw[m.start():].startswith(b"## 13.1 "))
end = next((m.start() for m in matches if m.start() > start), len(raw))
section = raw[start:end]
intro = raw[:matches[0].start()]
assert section == (OUT / "initial-section.md").read_bytes()
assert intro == (OUT / "initial-introduction.md").read_bytes()
code = re.search(r"```python\n(.*?)\n```", section.decode("utf-8"), re.S).group(1)
expected = "條件 只回數字：2+2=?\n比較 4 ／ 4，就像兩雙筷子共有四根。\n理由 先正確，再遵循只回數字 選擇 A\n"
namespace, buffer = {}, io.StringIO()
with contextlib.redirect_stdout(buffer):
    exec(compile(code, "course/chapters/13.md#13.1", "exec"), namespace)
actual = buffer.getvalue()
assert actual == expected
print("EXACT SECTION SNIPPET OUTPUT\n" + actual, end="")
changed = code.replace("只回數字：2+2=?", "2+2=?")
changed_namespace, changed_buffer = {}, io.StringIO()
with contextlib.redirect_stdout(changed_buffer):
    exec(compile(changed, "13.1 prompt-only exercise", "exec"), changed_namespace)
assert changed_namespace["chosen"] == "A"
print("PROMPT-ONLY EXERCISE chosen remains", changed_namespace["chosen"])

# Independently construct unordered groups and split assignments, then compare repo output.
independent = {}
for a in range(8):
    for b in range(8):
        key = f"{min(a,b)}+{max(a,b)}"
        independent.setdefault(key, []).append((a, b))
assert sum(map(len, independent.values())) == 8 * 8 == 64
assert len(independent) == 8 * 9 // 2 == 36
keys = sorted(independent)
random.Random(42).shuffle(keys)
families = {"train": keys[:28], "validation": keys[28:32], "test": keys[32:]}
parts = split_records(arithmetic_records(), seed=42)
pairs = behavior._preference_parts(parts)
dpo = json.loads((ROOT / "docs/course-experiments/results/dpo.json").read_bytes())
style = json.loads((ROOT / "docs/course-experiments/results/style.json").read_bytes())
summary = {}
for name in families:
    expected_operands = [ab for family in families[name] for ab in independent[family]]
    assert [(r["a"],r["b"]) for r in parts[name]] == expected_operands
    assert set(r["family"] for r in parts[name]) == set(families[name])
    assert all(r["chosen"] == str(a+b) and r["rejected"] == str(a+b+1)
               for r,(a,b) in zip(pairs[name], expected_operands, strict=True))
    pair_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs[name]).encode()
    arithmetic_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in parts[name]).encode()
    pair_sha = hashlib.sha256(pair_bytes).hexdigest()
    arithmetic_sha = hashlib.sha256(arithmetic_bytes).hexdigest()
    assert pair_sha == dpo["results"]["data"][name]["sha256"]
    assert arithmetic_sha == style["results"]["arithmetic_data"][name]["sha256"]
    assert len(pairs[name]) == dpo["results"]["data"][name]["records"]
    local_file = ROOT / f"outputs/text-behavior-interface-check/dpo/data/{name}.jsonl"
    assert local_file.read_bytes() == pair_bytes
    summary[name] = {"records": len(pairs[name]), "families": len(families[name]),
                     "family_ids": families[name], "pair_jsonl_sha256": pair_sha,
                     "arithmetic_jsonl_sha256": arithmetic_sha}
assert [summary[n]["records"] for n in ("train", "validation", "test")] == [49,8,7]
assert not any(set(families[a]) & set(families[b]) for a,b in (("train","validation"),("train","test"),("validation","test")))
print("INDEPENDENT SPLIT", json.dumps(summary, ensure_ascii=False))

results = dpo["results"]
for split,rows in pairs.items():
    format_rows = [{**r,"rejected":r["chosen"] + "; answer complete"} for r in rows]
    format_bytes = "".join(json.dumps(r,ensure_ascii=False) + "\n" for r in format_rows).encode()
    assert hashlib.sha256(format_bytes).hexdigest() == results["format_only"]["data"][split]["sha256"]
print("SEPARATE FORMAT BRANCH all 64 pairs retain the same correct numeral; fixed suffix changes format")
for split in ("validation", "test"):
    expected_pair_rows = pairs[split]
    for evaluation in [results["before"][split]] + [r["preference"][split] for r in results["runs"].values()]:
        assert evaluation["records"] == len(expected_pair_rows) == len(evaluation["samples"])
        assert [{k:r[k] for k in ("family","prompt","chosen","rejected")} for r in evaluation["samples"]] == expected_pair_rows
assert [(name,r["beta"],r["training"]["steps"]) for name,r in results["runs"].items()] == [("model",0.1,250),("beta1",1.0,250)]
historical_sampler = random.Random(42)
historical_draws = [r for _ in range(250) for r in historical_sampler.choices(pairs["train"], k=8)]
historical_token_count = sum(len(r[side].encode("utf-8")) + 1 for r in historical_draws for side in ("chosen","rejected"))
assert historical_token_count == 9448
assert all(r["training"]["effective_answer_tokens_both_sides"] == historical_token_count for r in results["runs"].values())
historical_source = ROOT / "outputs/natural-v4/factual-research/13.1/behavior-run-original.py"
assert hashlib.sha256(historical_source.read_bytes()).hexdigest() == dpo["code_sha256"]["scripts/course_experiments/behavior.py"]
print("SAVED RUN RECORD VALIDATION",json.dumps({"seed":dpo["seed"],"device":dpo["device"],
    "torch":dpo["torch_version"],"step_scale":dpo["step_scale"],"runs":{n:{"beta":r["beta"],"updates":r["training"]["steps"],"pair_draws":250*8,"answer_tokens_both_sides":r["training"]["effective_answer_tokens_both_sides"]} for n,r in results["runs"].items()},
    "before_test_records":results["before"]["test"]["records"],"after_test_records":{n:r["preference"]["test"]["records"] for n,r in results["runs"].items()},
    "original_source_sha_matches_record":True,"timing_scope":dpo["timing_scope"]},ensure_ascii=False))

# Actual two-update CPU execution of the current training function using train records only.
torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8,layers=1,max_length=32))
reference = copy.deepcopy(model).eval().requires_grad_(False)
initial = behavior._state_digest(model.state_dict())
reference_initial = behavior._state_digest(reference.state_dict())
ctx = Context(device="cpu",output=WORK,dependencies=WORK,assets=WORK,seed=42)
before = behavior._preference_evaluate(model,reference,pairs["test"])
training = behavior._dpo_train(model,reference,pairs["train"],ctx,"bounded",beta=0.1,count=2)
after = behavior._preference_evaluate(model,reference,pairs["test"])
assert behavior._state_digest(model.state_dict()) != initial
assert behavior._state_digest(reference.state_dict()) == reference_initial
assert before["records"] == after["records"] == 7
assert training["steps"] == 2
independent_sampler = random.Random(42)
independent_draws = [r for _ in range(2) for r in independent_sampler.choices(pairs["train"], k=8)]
expected_effective = sum(len(r[side].encode("utf-8")) + 1 for r in independent_draws for side in ("chosen","rejected"))
assert training["effective_answer_tokens_both_sides"] == expected_effective
print("BOUNDED ACTUAL CPU TRAINING",json.dumps({"seed":42,"width":8,"layers":1,"max_length":32,
    "batch_pairs":8,"updates":2,"pair_draws":16,"beta":0.1,"learning_rate":0.001,
    "train_records":49,"test_records_before_after":7,"answer_tokens_both_sides":training["effective_answer_tokens_both_sides"],
    "losses_before_update":training["history"],"policy_changed":True,"reference_unchanged":True,
    "scope":"two updates from a random small CPU model; no canonical GPU training or quality replication"},ensure_ascii=False))

torch.manual_seed(42)
candidate_rows = build_records()
policy = FiniteResponsePolicy()
x = torch.tensor([r["features"] for r in candidate_rows[:3]],dtype=torch.float32)
with torch.no_grad():
    logits = policy(x)
    actions = torch.multinomial(logits.softmax(-1),1,generator=torch.Generator().manual_seed(44)).squeeze(1)
assert tuple(logits.shape) == (3,4)
selected_texts = [r["candidates"][i] for r,i in zip(candidate_rows[:3],actions.tolist(),strict=True)]
print("FINITE CANDIDATE ACTUAL CPU SELECTION",json.dumps({"seed":42,"sampling_seed":44,"logit_shape":list(logits.shape),"actions":actions.tolist(),"selected_full_strings":selected_texts,"autoregressive_generation":False},ensure_ascii=False))
print("ALL ASSERTIONS PASSED")
