"""Independent bounded CPU audit for C.3; no training or downloads.

The existing formal GPU report is input evidence. Scores are recomputed from
its raw token IDs rather than accepted from its stored scoring flags.
"""
from pathlib import Path
import contextlib
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import platform
import random
import re
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs/technical-reviews/artifacts/natural-v4-factual/C.3"
import torch
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from scripts.course_experiments.applications import _reasoning_records, _reasoning_sft, _sample
from scripts.course_experiments.common import Context, records_sha256, split_records, text_examples

torch.set_num_threads(1)
environment = {"python": sys.version, "torch": str(torch.__version__),
               "device": "cpu", "cuda_available": str(torch.cuda.is_available()),
               "platform": platform.platform(), "threads": str(torch.get_num_threads()),
               "scope": "CPU arithmetic, raw-record audit, initial random weights and four short inference calls; zero training updates"}
(OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print("ENVIRONMENT", json.dumps(environment))

section = (OUT / "source-initial.md").read_text()
snippet = re.search(r"```python\n(.*?)```", section, re.S).group(1)
print("TEXTBOOK_SNIPPET")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(snippet, "C.3 textbook snippet", "exec"), {})
print(buf.getvalue(), end="")
assert buf.getvalue().splitlines() == [
    "候選數 1 獨立假設下包含正解 0.3", "候選數 2 獨立假設下包含正解 0.51",
    "候選數 4 獨立假設下包含正解 0.7599", "候選數 8 獨立假設下包含正解 0.942352",
    "本組候選 [3, 4, 3, 5] 包含真值4 True"]
exercise = snippet.replace("candidates = [3, 4, 3, 5]", "candidates = [3,3,3,3]")
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    exec(compile(exercise, "C.3 exercise", "exec"), {})
print("EXERCISE"); print(buf2.getvalue(), end="")
assert buf2.getvalue().splitlines()[:4] == buf.getvalue().splitlines()[:4]
assert buf2.getvalue().splitlines()[-1] == "本組候選 [3, 3, 3, 3] 包含真值4 False"

p = Fraction(3, 10)
exact = {n: 1 - (1 - p) ** n for n in (1, 2, 4, 8)}
assert exact == {1: Fraction(3, 10), 2: Fraction(51, 100),
                 4: Fraction(7599, 10000), 8: Fraction(94235199, 100000000)}
print("EXACT_COVERAGE", {n: str(v) for n, v in exact.items()})
def estimator(n, c, k):
    return Fraction(1) - Fraction(math.comb(n-c, k) if n-c >= k else 0, math.comb(n, k))
for n in (1, 2, 4, 8):
    for c in range(n+1):
        assert estimator(n, c, n) == int(c > 0)
print("PASS_AT_K_N_EQUALS_K", "all 19 cases n=k in [1,2,4,8], c=0..n exactly equal indicator(c>0)")
true_mixed_coverage = ((1-(1-Fraction(0))**2)+(1-(1-Fraction(3,5))**2))/2
using_mean = 1-(1-Fraction(3,10))**2
assert true_mixed_coverage == Fraction(21,50) and using_mean == Fraction(51,100)
print("HETEROGENEOUS_P_COUNTEREXAMPLE", "p=[0,0.6], mean p=0.3; mean true n=2 coverage=0.42; plugging mean gives 0.51")

record_path = ROOT / "docs/course-experiments/results/reasoning.json"
record_bytes = record_path.read_bytes()
r = json.loads(record_bytes)
assert r["seed"] == r["results"]["seed"] == 42
assert r["step_scale"] == 1.0 and r["status"] == "completed"
assert r["unfinished_schedules"] == []
summary = {"record_path": str(record_path.relative_to(ROOT)),
           "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
           "formal_environment": {k:r[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "timing_scope", "step_scale", "evidence_status")},
           "review_environment": environment, "branches": {},
           "scope": "Independent re-scoring of existing GPU records, not my own GPU training replication; no model checkpoints downloaded or loaded"}
splits = split_records(_reasoning_records(), 42)
families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
assert all(families[a].isdisjoint(families[b]) for a,b in [("train","validation"),("train","test"),("validation","test")])
for name, rows in splits.items():
    assert {"records":len(rows), "families":len(families[name]), "sha256":records_sha256(rows)} == r["results"]["split"][name]
test_keys = {(row["a"],row["b"],row["c"]) for row in splits["test"]}
assert len(test_keys) == 24
summary["reconstructed_splits"] = r["results"]["split"]

torch.manual_seed(42)
base = TinyLM(ModelConfig(**r["results"]["model_config"]))
state_hash = lambda m: hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
base_hash = state_hash(base)
assert base_hash == "d34b57a18a2051e64735e377620cd73a08bb605ef59199b97957d83b0d7c8a4b"
assert sum(p.numel() for p in base.parameters()) == 141568
summary["reconstructed_initial_state_sha256"] = base_hash
tokenizer = ByteTokenizer()
assert _reasoning_sft([{"a":1,"b":2,"c":3,"truth":6,"question":"(1+2)+3=?"}], "direct")[0]["messages"][-1]["content"] == "6"
assert _reasoning_sft([{"a":1,"b":2,"c":3,"truth":6,"question":"(1+2)+3=?"}], "steps")[0]["messages"][-1]["content"] == "1+2=3;3+3=6;answer=6"

expected = {"direct": [1,1,2,1], "steps": [11,13,13,14]}
predictions = []
for mode, branch in r["results"]["comparison"].items():
    assert branch["base_state_sha256"] == base_hash
    assert branch["training"]["steps"] == branch["training"]["planned_steps"] == 900
    assert branch["training"]["step_scale"] == 1.0
    train_records = _reasoning_sft(splits["train"], mode)
    assert records_sha256(train_records) == branch["training"]["records_sha256"]
    examples = text_examples(train_records, mode="sft", max_length=128)
    sampler = random.Random(42)
    target_count = sum(int((y != IGNORE).sum()) for _ in range(900) for _, y in sampler.choices(examples, k=24))
    assert target_count == branch["training"]["effective_tokens"]
    total = 0
    budgets = []
    for b_index, budget in enumerate(branch["budgets"]):
        k = budget["candidate_count"]
        assert k == (1,2,4,8)[b_index]
        rows = budget["samples"]
        assert len(rows) == 24
        assert {(s["a"],s["b"],s["c"]) for s in rows} == test_keys
        hits = 0; tokens = 0; correct_candidates = 0
        for q_index, row in enumerate(rows):
            operands = tuple(map(int, re.fullmatch(r"\(([0-5])\+([0-5])\)\+([0-5])=\?", row["question"]).groups()))
            truth = sum(operands)
            assert truth == row["truth"]
            assert operands == (row["a"],row["b"],row["c"])
            assert row["family"] == ":".join(map(str, sorted(operands)))
            assert len(row["candidates"]) == k
            gen = row["generation"]
            assert gen["temperature"] == 0.7 and gen["candidate_count"] == k
            assert gen["max_new_tokens"] == (8 if mode == "direct" else 48)
            expected_prompt = [1,3] + [b+8 for b in row["question"].encode()] + [2,4]
            assert gen["input_ids"] == expected_prompt and gen["input_tokens"] == len(expected_prompt)
            row_successes = 0; row_tokens = 0
            for c_index, candidate in enumerate(row["candidates"]):
                ids = candidate["generated_ids"]
                assert 0 < len(ids) <= gen["max_new_tokens"]
                assert all(isinstance(i,int) and 0 <= i < 264 for i in ids)
                eos = ids[-1] == 2
                raw = ids[:-1] if eos else ids
                invalid = [i for i in raw if i < 8]
                text = bytes(i-8 for i in raw if i>=8).decode("utf-8",errors="replace")
                assert text == candidate["generated"]
                assert invalid == candidate["invalid_special_tokens"]
                assert eos == candidate["eos"]
                assert len(ids) == candidate["generated_tokens"]
                clean = not invalid
                final = None
                stripped = text.strip()
                if clean and mode == "direct" and re.fullmatch(r"-?[0-9]+", stripped):
                    final = int(stripped)
                if clean and mode == "steps":
                    final_match = re.search(r";answer=(-?[0-9]+)$", stripped)
                    if final_match:
                        final = int(final_match[1])
                correct = final == truth
                assert final == candidate["final_answer"] and correct == candidate["final_correct"]
                row_successes += int(correct)
                row_tokens += len(ids)
                predictions.append([mode,k,q_index,c_index,row["question"],truth,final,int(correct),len(ids),hashlib.sha256(json.dumps(ids).encode()).hexdigest()])
            hit = row_successes > 0
            assert hit == row["oracle_coverage"]
            assert estimator(k,row_successes,k) == int(hit)
            assert row_tokens == gen["generated_tokens"]
            hits += int(hit); tokens += row_tokens; correct_candidates += row_successes
        assert hits == expected[mode][b_index]
        assert budget["oracle_coverage"] == {"numerator":hits,"denominator":24,"rate":hits/24}
        assert budget["generated_tokens"] == tokens
        assert budget["candidate_final_accuracy"] == {"numerator":correct_candidates,"denominator":24*k,"rate":correct_candidates/(24*k)}
        total += 24*k
        budgets.append({"k":k,"hits":hits,"questions":24,"candidates":24*k,"rate":hits/24,"generated_tokens_including_EOS":tokens,
                        "correct_candidates":correct_candidates,"max_new_tokens":8 if mode=="direct" else 48,"temperature":0.7,
                        "mean_n_equals_k_estimator":str(Fraction(hits,24))})
    assert total == 360
    summary["branches"][mode] = {"budgets":budgets,"candidates":total,"unique_test_questions":24,
                                 "training_updates":900,"sampled_training_examples":900*24,
                                 "recomputed_effective_answer_targets_including_EOS":target_count,
                                 "base_state_sha256":branch["base_state_sha256"]}

assert len(predictions) == 720
with (OUT/"predictions.csv").open("w",newline="") as f:
    writer=csv.writer(f);writer.writerow(["branch","k","question_index","candidate_index","question","truth","parsed_final","correct","generated_token_count","generated_ids_sha256"]);writer.writerows(predictions)
summary["predictions_rows"] = len(predictions)
summary["training_token_ratio"] = Fraction(466322,48803).__str__()
print("RAW_RECORD_RECOMPUTATION", json.dumps(summary, indent=2))

# Actual current CPU software call, deliberately bounded to four tokens per candidate.
# This is a mechanics probe of random initial weights, not a quality experiment.
ctx = Context("cpu", OUT, ROOT, ROOT, seed=42)
torch.manual_seed(7303)
probe=[]
for k in (1,2,4,8):
    before=state_hash(base)
    sampled=_sample(base,[{"role":"user","content":"(1+2)+3=?"}],ctx,count=k,tokens=4,temperature=0.7)
    assert state_hash(base)==before==base_hash and base.training
    assert len(sampled["samples"])==k and sampled["candidate_count"]==k
    probe.append({"k":k,"temperature":sampled["temperature"],"max_new_tokens":4,
                  "generated_ids":[s["generated_ids"] for s in sampled["samples"]],
                  "generated_tokens":sampled["generated_tokens"],"unchanged_weights":True,"training_mode_restored":base.training})
summary["current_CPU_sampler_mechanics_probe"] = {"seed":7303,"training_updates":0,"probe":probe,
                                                "scope":"four-token calls verify current batch sampling and fixed weights; no quality, independence proof, or latency benchmark"}
print("CURRENT_CPU_SAMPLER_PROBE", json.dumps(summary["current_CPU_sampler_mechanics_probe"]))
(OUT/"audit-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print("ALL_ASSERTIONS_PASSED; 720 candidates independently decoded and rescored; no training performed")
