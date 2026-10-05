"""Independent B.2 CPU protocol checks; injected IDs, never trained model scores."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import applications as app
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.retrieval import call_tool

OUT = Path(__file__).resolve().parent
tok = ByteTokenizer()
ctx = SimpleNamespace(device=torch.device("cpu"))
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {
    "python": sys.version,
    "torch": str(torch.__version__),
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu",
    "fixture": "hand-specified output IDs through the actual _sample; no learned weights",
}


class IdSequence:
    def __init__(self, ids):
        self.ids = iter(ids)
        self.training = True
        self.config = SimpleNamespace(max_length=2000)

    def eval(self):
        self.training = False

    def train(self, training):
        self.training = training

    def __call__(self, current, cache=None):
        chosen = next(self.ids)
        scores = torch.full((current.shape[0], 1, tok.vocab_size), -float("inf"))
        scores[:, 0, chosen] = 0
        return {"logits": scores, "cache": None}


request = '{"name":"add","arguments":{"a":2,"b":3}}'
record = {
    "family": "pair:2:3", "operation": "add", "a": 2, "b": 3, "answer": 5,
    "messages": [{"role": "system", "content": app._TOOLS_SYSTEM},
                 {"role": "user", "content": "CALC:add(2,3)"}],
}
real_parser, real_tool = app._parse_json_action, app.call_tool


def episode_from_ids(ids, max_steps=1):
    counts = {"parser": 0, "tool": 0}

    def parse(text):
        counts["parser"] += 1
        return real_parser(text)

    def tool(action):
        counts["tool"] += 1
        return real_tool(action)

    app._parse_json_action, app.call_tool = parse, tool
    try:
        episode = app._tool_episode(IdSequence(ids), copy.deepcopy(record), ctx, max_steps=max_steps)
    finally:
        app._parse_json_action, app.call_tool = real_parser, real_tool
    json.dumps(episode, allow_nan=False)
    return {"calls": counts, "episode": episode}


checks = {}
checks["minimal_schema"] = {"valid_add": call_tool(request), "valid_multiply": call_tool(
    {"name": "multiply", "arguments": {"a": 2, "b": 3}}),
    "legal_wrong_parameter_result": call_tool('{"name":"add","arguments":{"a":2,"b":4}}'),
    "bool_is_int_subclass": isinstance(True, int), "bool_exact_type_is_int": type(True) is int}
assert checks["minimal_schema"]["valid_add"] == 5
assert checks["minimal_schema"]["valid_multiply"] == 6
assert checks["minimal_schema"]["legal_wrong_parameter_result"] == 6
bad = [
    '{"name":"delete_all","arguments":{"a":2,"b":3}}',
    '{"name":"add","arguments":{"a":true,"b":3}}',
    '{"name":"add","arguments":{"a":"2","b":3}}',
    '{"name":"add","arguments":{"a":2,"b":3},"extra":0}',
    '{"name":"add","arguments":{"a":2,"b":3,"extra":0}}',
]
checks["minimal_schema"]["rejections"] = []
for text in bad:
    try:
        call_tool(text)
        raise AssertionError("unexpected schema acceptance")
    except ValueError as error:
        checks["minimal_schema"]["rejections"].append({"input": text, "error": str(error)})

checks["strict_parser"] = []
for text in [
    '{"name":"add","arguments":{"a":NaN,"b":3}}',
    '{"name":"add","arguments":{"a":Infinity,"b":3}}',
    '{"name":"add","arguments":{"a":-Infinity,"b":3}}',
    '{"name":"add","arguments":{"a":1e999,"b":3}}',
    '{"name":"add","arguments":{"a":-1e999,"b":3}}',
    '{"name":"add","name":"multiply","arguments":{"a":2,"b":3}}',
    '{"name":"add","arguments":{"a":2,"a":3,"b":3}}',
    r'{"name":"add","arguments":{"a":2,"\u0061":3,"b":3}}',
    '{"x":{"y":{"v":1,"v":2}}}',
]:
    result = episode_from_ids(tok.encode(text) + [tok.eos_id])
    assert result["calls"] == {"parser": 1, "tool": 0}
    assert result["episode"]["status"] == "invalid_request"
    checks["strict_parser"].append({"input": text, "error": result["episode"]["trace"][0]["error"]})
checks["minimal_not_strict"] = {
    "NaN_is_accepted": math.isnan(call_tool('{"name":"add","arguments":{"a":NaN,"b":3}}')),
    "Infinity_is_accepted": math.isinf(call_tool('{"name":"add","arguments":{"a":Infinity,"b":3}}')),
    "1e999_is_accepted": math.isinf(call_tool('{"name":"add","arguments":{"a":1e999,"b":3}}')),
    "duplicate_name_uses_last": call_tool('{"name":"add","name":"multiply","arguments":{"a":2,"b":3}}'),
}
assert all(checks["minimal_not_strict"][k] for k in ["NaN_is_accepted", "Infinity_is_accepted", "1e999_is_accepted"])
assert checks["minimal_not_strict"]["duplicate_name_uses_last"] == 6

ids = tok.encode(request)
insert_at = len(ids) // 2
checks["control_id_group"] = []
for control in [tok.pad_id, tok.bos_id, tok.user_id, tok.assistant_id, tok.image_id, tok.audio_id, tok.system_id]:
    result = episode_from_ids(ids[:insert_at] + [control] + ids[insert_at:] + [tok.eos_id])
    event = result["episode"]["trace"][0]
    sample = event["generation"]["samples"][0]
    assert sample["generated"] == request and sample["invalid_special_tokens"] == [control]
    assert sample["generated_ids"][-1] == tok.eos_id and sample["eos"]
    assert result["calls"] == {"parser": 0, "tool": 0}
    assert "parsed" not in event and event["executed"] is False
    assert result["episode"]["status"] == "invalid_request"
    checks["control_id_group"].append({"control_id": control, "calls": result["calls"],
        "generated": sample["generated"], "invalid_special_tokens": sample["invalid_special_tokens"],
        "generated_ids": sample["generated_ids"], "eos": sample["eos"], "error": event["error"]})

normal = app._sample(IdSequence(ids + [tok.eos_id, tok.user_id]), record["messages"], ctx, tokens=64)
sample = normal["samples"][0]
assert sample["generated"] == request and sample["invalid_special_tokens"] == []
assert sample["generated_ids"] == ids + [tok.eos_id]
assert sample["generated_tokens"] == len(ids) + 1 and sample["eos"]
early = episode_from_ids(ids[:10] + [tok.eos_id, tok.user_id])
assert early["calls"] == {"parser": 1, "tool": 0}
assert early["episode"]["trace"][0]["generation"]["samples"][0]["invalid_special_tokens"] == []
assert early["episode"]["status"] == "invalid_request"
without_eos = app._sample(IdSequence(ids), record["messages"], ctx, tokens=len(ids))["samples"][0]
assert not without_eos["eos"] and without_eos["generated"] == request
checks["eos"] = {"normal_sample": sample, "early_stop": early,
    "budget_stop_sample": without_eos, "scope": "EOS is excluded only as terminal stop; absence of EOS itself is not a _tool_episode parsing gate."}

array = episode_from_ids(tok.encode("[]") + [tok.eos_id])
assert array["calls"] == {"parser": 1, "tool": 0}
assert array["episode"]["status"] == "invalid_request" and "parsed" not in array["episode"]["trace"][0]
checks["top_level_array"] = array
overflow = episode_from_ids(tok.encode('{"name":"multiply","arguments":{"a":1e308,"b":1e308}}') + [tok.eos_id])
event = overflow["episode"]["trace"][0]
assert overflow["calls"] == {"parser": 1, "tool": 1}
assert overflow["episode"]["status"] == "invalid_tool_result" and overflow["episode"]["actual_tool_calls"] == 1
assert event["executed"] and not event["finite_result"] and event["tool_result"] == "inf"
json.dumps(overflow, allow_nan=False)
checks["overflow"] = overflow
checks["float_values"] = {"1e308_finite": math.isfinite(float("1e308")),
    "1e999_is_inf": math.isinf(float("1e999")), "product_repr": repr(1e308 * 1e308)}

raw_path = OUT / "inputs/docs/course-experiments/results/tools.json"
raw = json.loads(raw_path.read_bytes())
injections = raw["results"]["injected_protocol_checks_not_model_scores"]
assert len(injections) == 7 and all(row["rejected"] for row in injections)
rechecked = []
for row in injections:
    try:
        real_tool(real_parser(row["input"]))
        raise AssertionError("stored rejected input accepted")
    except (ValueError, TypeError, KeyError) as error:
        assert str(error) == row["error"]
        rechecked.append({"input": row["input"], "rejected": True, "error": str(error)})
episodes = raw["results"]["samples"]
assert len(episodes) == raw["results"]["test"]["examples"] == 20
actual = 0
for episode in episodes:
    for event in episode["trace"]:
        if "generation" in event:
            for sample in event["generation"]["samples"]:
                output = sample["generated_ids"]
                plain = output[:-1] if output and output[-1] == tok.eos_id else output
                assert sample["generated"] == tok.decode(plain)
                assert sample["eos"] == bool(output and output[-1] == tok.eos_id)
                assert sample["invalid_special_tokens"] == [x for x in plain if x < 8]
        if event.get("executed"):
            actual += 1
            action = event["parsed"]
            assert all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 9 for x in action["arguments"].values())
            expected_result = real_tool(action)
            assert event["finite_result"] is True and expected_result == event["tool_result"]
    assert episode["status"] != "invalid_tool_result"
assert not any(x["input"] == request for x in injections)
provenance = {}
for filename in ["tiny_perceptron/data.py", "tiny_perceptron/retrieval.py", "scripts/course_experiments/applications.py"]:
    digest = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    assert raw["code_sha256"][filename] == digest
    provenance[filename] = digest
checks["existing_raw_measurements"] = {"raw_file_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    "revision": raw["revision"], "device": raw["device"], "seed": raw["seed"],
    "python_version": raw["python_version"], "torch_version": raw["torch_version"],
    "fault_injections": rechecked, "fault_injection_count": len(injections),
    "model_episode_count": len(episodes), "actual_calls_inspected_for_finite_result": actual,
    "model_overflow_event_count": 0, "provenance_code_hashes_matched": provenance,
    "scope": "Stored raw records inspected and recomputed; no model training or new ability score."}
(OUT / "verification-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
(OUT / "verification-results.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"cpu_checks": "passed", "minimal_schema": "3 arithmetic results + 5 rejections",
    "strict_parser": "9 rejected inputs; parser=1/tool=0", "control_id_group": "7 rejected non-EOS special IDs; parser=0/tool=0; includes actual user ID 3",
    "eos": "terminal EOS retained in raw IDs, excluded from invalid list, halts before subsequent user ID; early and budget stops checked",
    "top_level_array": "[] parser=1/tool=0", "overflow": "actual call=1, executed=true, finite_result=false, result string inf; strict serialization passed",
    "raw_measurements": checks["existing_raw_measurements"]}, ensure_ascii=False, indent=2, allow_nan=False))
