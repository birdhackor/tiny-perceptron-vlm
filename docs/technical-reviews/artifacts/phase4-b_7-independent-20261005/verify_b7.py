"""Independent bounded CPU audit of frozen raw B.7 inputs. No fitting or generation."""
import hashlib
import json
import math
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments.tool_choice import build_records, parse_action, split_records, summarize

def read(name):
    return json.loads((HERE / "frozen-input" / name).read_bytes())

def checksum(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def action_from_raw(sample):
    ids = sample["generated_ids"]
    eos = bool(ids and ids[-1] == 2)
    raw = ids[:-1] if eos else ids
    illegal = [i for i in raw if i < 8]
    decoded = bytes(i - 8 for i in raw if i >= 8).decode("utf-8", errors="replace")
    assert sample["eos"] == eos
    assert sample["invalid_special_tokens"] == illegal
    assert sample["generated"] == decoded
    assert sample["generated_tokens"] == len(ids)
    chosen = decoded if eos and not illegal and decoded in ("DIRECT", "TOOL", "ASK") else None
    assert parse_action(sample) == chosen
    return chosen

def independent_metrics(rows, actions):
    needed = [i for i,r in enumerate(rows) if r["expected_action"] == "TOOL"]
    not_needed = [i for i,r in enumerate(rows) if r["expected_action"] != "TOOL"]
    unavailable = [i for i,r in enumerate(rows) if not r["calculator_available"]]
    return {
        "accuracy": [sum(actions[i] == r["expected_action"] for i,r in enumerate(rows)), len(rows)],
        "needed_tool_selected": [sum(actions[i] == "TOOL" for i in needed), len(needed)],
        "needed_tool_missed": [sum(actions[i] != "TOOL" for i in needed), len(needed)],
        "unnecessary_tool_selected": [sum(actions[i] == "TOOL" for i in not_needed), len(not_needed)],
        "unavailable_tool_selected": [sum(actions[i] == "TOOL" for i in unavailable), len(unavailable)],
    }

d = read("docs/course-experiments/results/tool_choice.json")
raw = d["results"]
splits = split_records(build_records(), d["seed"])
test_families = {r["family"] for r in splits["test"]}
splits["paraphrase_diagnostic"] = [r for r in build_records(paraphrase=True) if r["family"] in test_families]
owners = {}
prompts = {}
for name in ("train", "validation", "test"):
    for r in splits[name]:
        owners.setdefault(r["family"], set()).add(name)
        prompts.setdefault(tuple(m["content"] for m in r["messages"][:-1]), set()).add(name)
assert all(len(x) == 1 for x in owners.values())
assert all(len(x) == 1 for x in prompts.values())
assert owners["pair:1:2"] == {"test"}
assert len(owners) == 55
assert {v for r in splits["train"] for v in r["operands"]} == set(range(10))
train_questions = {r["messages"][1]["content"] for r in splits["train"]}
assert not train_questions.intersection(r["messages"][1]["content"] for r in splits["paraphrase_diagnostic"])
split_results = {}
for name, rows in splits.items():
    recorded = raw["data"][name]
    observed = {"records":len(rows), "families":sorted({r["family"] for r in rows}),
                "action_counts":dict(Counter(r["expected_action"] for r in rows)), "sha256":checksum(rows)}
    assert observed == recorded
    for r in rows:
        wanted = ("TOOL" if r["calculator_available"] else "ASK") if r["intent"] == "numerical_addition" else "ASK" if r["intent"] == "missing_quantity" else "DIRECT"
        assert r["expected_action"] == wanted
        assert r["messages"][-1] == {"role":"assistant", "content":wanted}
        assert r["messages"][0]["content"] == ("計算器可用。" if r["calculator_available"] else "計算器停用。")
        assert not any(marker in r["messages"][1]["content"] for marker in ("CALC", "COPY", "TOOL", "DIRECT", "ASK"))
        if wanted == "ASK":
            assert r["ask_reason"] in ("missing_quantity", "calculator_unavailable")
            assert r["expected_human_step"] not in r["messages"][-1]["content"]
    split_results[name] = observed

scores = {}
for name in ("test", "paraphrase_diagnostic"):
    samples = raw[name]["samples"]
    rows = [s["record"] for s in samples]
    assert rows == splits[name]
    actions = []
    for s in samples:
        generation = s["generation"]
        assert generation["messages"] == s["record"]["messages"][:-1]
        prompt = [1]
        for m in generation["messages"]:
            prompt += [{"system":7, "user":3}[m["role"]]] + [b+8 for b in m["content"].encode()] + [2]
        prompt += [4]
        assert generation["input_ids"] == prompt
        assert generation["input_tokens"] == len(prompt)
        assert generation["candidate_count"] == 1
        assert generation["temperature"] == 0.0 and generation["max_new_tokens"] == 8
        a = action_from_raw(generation["samples"][0])
        assert a == s["chosen_action"]
        actions.append(a)
    measured = independent_metrics(rows, actions)
    helper = summarize(rows, actions)
    for key, pair in measured.items():
        assert pair == [helper[key]["numerator"], helper[key]["denominator"]]
        assert helper[key] == raw[name]["metrics"][key]
    assert raw[name]["records_sha256"] == checksum(rows)
    for base, label in [("always_direct", "DIRECT"), ("always_tool", "TOOL")]:
        baseline = independent_metrics(rows, [label]*len(rows))["accuracy"]
        saved = raw[name]["baselines"][base]["accuracy"]
        assert baseline == [saved["numerator"],saved["denominator"]]
        measured[base] = baseline
    missed = [{"family":r["family"], "question":r["messages"][1]["content"], "generated":a}
              for r,a in zip(rows,actions,strict=True) if r["expected_action"] == "TOOL" and a != "TOOL"]
    paired = {}
    for r,a in zip(rows,actions,strict=True):
        paired.setdefault(r["messages"][1]["content"],{})[r["calculator_available"]] = a
    example = {q:{str(k):v for k,v in values.items()} for q,values in paired.items()
               if q in ("1 加 2 等於多少？", "幫我把 1 與 2 相加，給我答案。")}
    scores[name] = {"metrics":measured, "raw_sample_count":len(samples), "valid_with_eos":sum(a is not None for a in actions),
                    "missed_tool_samples":missed, "availability_example":example,
                    "by_intent":helper["by_intent"], "by_availability":helper["by_availability"]}
assert scores["test"]["metrics"]["accuracy"] == [96,96]
assert scores["paraphrase_diagnostic"]["metrics"]["accuracy"] == [42,48]
assert scores["paraphrase_diagnostic"]["metrics"]["needed_tool_selected"] == [0,6]
assert {s["generated"] for s in scores["paraphrase_diagnostic"]["missed_tool_samples"]} == {"ASK"}

# Fixed schedule token denominator is reconstructed without running any model.
sampler = random.Random(d["seed"])
effective_tokens = 0
for _ in range(900):
    for r in sampler.choices(splits["train"], k=24):
        effective_tokens += len(r["expected_action"].encode()) + 1
training = raw["training"]
assert training["steps"] == 900
assert training["records"] == 704
assert training["records_sha256"] == checksum(splits["train"])
assert training["effective_tokens"] == effective_tokens
assert training["parameters"] == training["trainable_parameters"] == raw["model"]["parameters"]
original_code = HERE/"frozen-input/scripts/course_experiments/tool_choice.py"
code_sha = hashlib.sha256(original_code.read_bytes()).hexdigest()
assert d["code_sha256"]["scripts/course_experiments/tool_choice.py"] == code_sha == raw["reproducibility"]["script_sha256"]
assert raw["frozen_evaluation"]["unchanged"] is True

# Meaningful bounded changes: abstention, gratuitous use, malformed label and empty tool subset.
gold = ["TOOL", "TOOL", "DIRECT", "DIRECT", "ASK", "ASK"]
toyrows = [{"expected_action":a,"intent":"numerical_addition" if a == "TOOL" else "copy" if a=="DIRECT" else "missing_quantity","calculator_available":True} for a in gold]
toy = independent_metrics(toyrows,["TOOL","DIRECT","TOOL","DIRECT","ASK","DIRECT"])
assert toy["accuracy"] == [3,6] and toy["needed_tool_missed"] == [1,2] and toy["unnecessary_tool_selected"] == [1,4]
variants = {"always_direct":independent_metrics(toyrows,["DIRECT"]*6),"always_tool":independent_metrics(toyrows,["TOOL"]*6)}
assert variants["always_tool"]["accuracy"] == [2,6]
assert variants["always_tool"]["unnecessary_tool_selected"] == [4,4]
assert variants["always_direct"]["needed_tool_missed"] == [2,2]
pure_tool = [dict(toyrows[0]) for _ in range(6)]
assert independent_metrics(pure_tool,["TOOL"]*6)["accuracy"] == [6,6]
bad_labels = [("TOOL ",True,[]),("TOOL",False,[]),("TOOL",True,[4]),("TOOL\n",True,[])]
assert all(parse_action({"generated":g,"eos":e,"invalid_special_tokens":s}) is None for g,e,s in bad_labels)
zero = summarize([toyrows[2]],["DIRECT"])["needed_tool_selected"]
assert zero == {"numerator":0,"denominator":0,"rate":None}
try:
    sum(a==b for a,b in zip(gold,gold[:-1],strict=True))
except ValueError:
    unequal_rejected = True
else:
    raise AssertionError("strict zip did not reject mismatched sizes")

# Compare the separate old JSON model only at the scope B.7 cites.
other = read("docs/course-experiments/results/tools.json")["results"]
episode_results = []
for s in other["samples"]:
    calls = [e for e in s["trace"] if e.get("executed")]
    final = None
    for e in s["trace"]:
        if "generation" not in e:
            continue
        generated = e["generation"]["samples"][0]["generated"]
        try:
            a = json.loads(generated)
        except json.JSONDecodeError:
            continue
        if isinstance(a,dict) and set(a)=={"done","answer"} and a["done"] is True:
            final = a["answer"]
    if s["operation"] == "copy":
        expected = int(s["question"].split(":")[1])
        required = not calls
    else:
        match = re.fullmatch(r"CALC:(add|multiply)\((\d+),(\d+)\)",s["question"])
        assert match
        op,a,b = match.group(1),int(match.group(2)),int(match.group(3))
        expected = a+b if op=="add" else a*b
        for e in calls:
            req = e["parsed"]
            assert req == {"name":op,"arguments":{"a":a,"b":b}}
            assert e["tool_result"] == expected and math.isfinite(e["tool_result"])
        required = bool(calls and final==calls[-1]["tool_result"])
    good_answer = s["status"] == "done" and type(final) in (int,float) and final == expected
    good = good_answer and required
    assert good == s["correct"]
    assert expected == s["expected"]
    episode_results.append(good)
assert [sum(episode_results),len(episode_results)] == [19,20]
assert raw["model"]["config"] != other["model"]["config"]

result = {
    "environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","training_or_model_generation":"none"},
    "input_raw_sha256":hashlib.sha256((HERE/"frozen-input/docs/course-experiments/results/tool_choice.json").read_bytes()).hexdigest(),
    "splits":split_results,"scores":scores,"six_row_example":toy,"bounded_variants":variants,
    "pure_tool_baseline":[6,6],"strict_zip_rejects_unequal":unequal_rejected,"invalid_label_cases_rejected":len(bad_labels),"zero_tool_subset":zero,
    "training_provenance":{"fixed_steps":900,"batch_size":24,"records":704,"effective_tokens_recomputed":effective_tokens,"all_parameters_trainable":training["trainable_parameters"],"original_script_matches_raw_hash":True,"reported_frozen_weights_unchanged":True,"checkpoint_not_loaded":True},
    "different_json_model":{"correct":[sum(episode_results),len(episode_results)],"arithmetic":sum(s["operation"]!="copy" for s in other["samples"]),"copy":sum(s["operation"]=="copy" for s in other["samples"]),"distinct_config":True,"selection_max_length":raw["model"]["config"]["max_length"],"json_max_length":other["model"]["config"]["max_length"]},
    "inspected_json_pointers":["/seed","/revision","/device","/torch_version","/python_version","/code_sha256","/results/data","/results/model/config","/results/model/parameters","/results/training/steps","/results/training/records","/results/training/records_sha256","/results/training/effective_tokens","/results/training/parameters","/results/training/trainable_parameters","/results/test/samples","/results/test/metrics","/results/test/baselines","/results/test/records_sha256","/results/paraphrase_diagnostic/samples","/results/paraphrase_diagnostic/metrics","/results/paraphrase_diagnostic/baselines","/results/paraphrase_diagnostic/records_sha256","/results/frozen_evaluation","/results/reproducibility"],
    "tools_inspected_json_pointers":["/results/samples","/results/model/config","/results/split","/results/training/checkpoint","/results/protocol/max_steps_including_done"],
    "scope":"Recount of retained raw evidence and short CPU demonstrations; no retraining, checkpoint loading, mature-model generation, or end-to-end deployment verification."
}
(HERE/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"test":scores["test"]["metrics"],"paraphrase":scores["paraphrase_diagnostic"]["metrics"],"six_row":toy,"effective_tokens":effective_tokens,"separate_json_model":[19,20],"all_assertions":"passed"},ensure_ascii=False,indent=2))
