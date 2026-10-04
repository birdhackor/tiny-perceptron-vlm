"""Fresh bounded CPU checks; saved CUDA results are recomputed, not rerun."""
import contextlib
import hashlib
import io
import json
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat, IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM
from scripts.course_experiments import applications as apps

torch.set_num_threads(1)
torch.manual_seed(0)
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": "1"}}
assert torch.__version__ == "2.14.1+cpu" and not torch.cuda.is_available()

# Execute the unchanged section code and its single-variable exercise.
section = (ROOT / "course/chapters/0C.md").read_text().split("## C.5 ", 1)[1].split("## C.6 ", 1)[0]
code = section.split("```python\n", 1)[1].split("```", 1)[0]
out["planning"] = {}
for tokens, expected in [(5, [85, 100]), (0, [80, 80])]:
    namespace = {}; stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(code if tokens == 5 else code.replace("verify_tokens_each = 5", "verify_tokens_each = 0"), namespace)
    lines = stdout.getvalue().splitlines()
    totals = [int(line.rsplit(" ", 1)[1]) for line in lines]
    assert totals == expected
    out["planning"][str(tokens)] = {"stdout": lines, "totals": totals}

# Independently reconstruct the grouped data split without the repository split helper.
rows = [{"family": ":".join(map(str, sorted((a,b,c)))), "a":a, "b":b, "c":c,
         "question":f"({a}+{b})+{c}=?", "truth":a+b+c}
        for a in range(6) for b in range(6) for c in range(6)]
groups = {}
for row in rows: groups.setdefault(row["family"], []).append(row)
keys = sorted(groups); random.Random(42).shuffle(keys)
splits = {name: [row for key in selected for row in groups[key]]
          for name, selected in [("train",keys[:44]),("validation",keys[44:50]),("test",keys[50:]) ]}
record = json.loads((ROOT / "docs/course-experiments/results/reasoning.json").read_text())
results = record["results"]
out["original_run"] = {k: record[k] for k in ["revision", "device", "gpu", "torch_version", "python_version", "seed", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status"]}
out["record_sha256"] = digest(ROOT / "docs/course-experiments/results/reasoning.json")
out["splits"] = {}
for name, items in splits.items():
    value = {"records":len(items), "families":len({r["family"] for r in items}),
             "sha256":hashlib.sha256(json.dumps(items,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    assert value == results["split"][name]
    out["splits"][name] = value
assert all(set(r["family"] for r in splits[a]).isdisjoint(r["family"] for r in splits[b])
           for a,b in [("train","validation"),("train","test"),("validation","test")])

out["source_hashes"] = {}
for path in ["scripts/course_experiments/applications.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py"]:
    current = digest(ROOT / path)
    out["source_hashes"][path] = {"current":current,"recorded":record["code_sha256"][path],"same":current==record["code_sha256"][path]}
    assert current == record["code_sha256"][path]

out["budgets"] = []; out["training_counts"] = {}
for mode, model in results["comparison"].items():
    tok = ByteTokenizer(); examples=[]
    for row in splits["train"]:
        a,b,c = row["a"],row["b"],row["c"]
        answer = str(a+b+c) if mode=="direct" else f"{a}+{b}={a+b};{a+b}+{c}={a+b+c};answer={a+b+c}"
        _, labels = render_chat([{"role":"user","content":row["question"]},{"role":"assistant","content":answer}],tok)
        count = int((labels != IGNORE).sum())
        assert count == len(answer.encode()) + 1  # Answer bytes plus EOS; role/prompt targets masked.
        examples.append(count)
    rng=random.Random(42)
    effective=sum(sum(rng.choices(examples,k=24)) for _ in range(900))
    assert effective==model["training"]["effective_tokens"] and model["training"]["steps"]==900
    out["training_counts"][mode]={"updates":900,"batch_size":24,"sampled_examples":900*24,"effective_answer_targets_including_eos":effective}
    for budget in model["budgets"]:
        n=budget["candidate_count"]; cap=8 if mode=="direct" else 48
        counts={"coverage":0,"majority":0,"verifier":0,"selection_failure":0}
        token_total=0; seconds=0.; eos=0; forward=0
        assert len(budget["samples"])==24
        for sample, row in zip(budget["samples"],splits["test"],strict=True):
            assert {key:sample[key] for key in row}==row
            g=sample["generation"]
            assert (g["candidate_count"],g["max_new_tokens"],g["temperature"])==(n,cap,0.7)
            assert len(sample["candidates"])==n
            assert g["input_ids"]==[1,3]+[b+8 for b in row["question"].encode()]+[2,4]
            answers=[]; correct=[]; verified=[]; lengths=[]
            for candidate in sample["candidates"]:
                ids=candidate["generated_ids"]; lengths.append(len(ids)); eos+=int(ids[-1]==2)
                assert 1<=len(ids)<=cap and candidate["generated_tokens"]==len(ids)
                assert 2 not in ids[:-1] and candidate["eos"]==(ids[-1]==2)
                raw=ids[:-1] if ids[-1]==2 else ids
                text=bytes(i-8 for i in raw if i>=8).decode("utf-8",errors="replace").strip()
                assert candidate["generated"].strip()==text
                clean=not [i for i in raw if i<8]
                assert candidate["invalid_special_tokens"]==[i for i in raw if i<8]
                full=re.fullmatch(r"(-?\d+)\+(-?\d+)=(-?\d+);(-?\d+)\+(-?\d+)=(-?\d+);answer=(-?\d+)",text)
                if mode=="direct":
                    final=int(text) if clean and re.fullmatch(r"-?[0-9]+",text) else None
                    valid=final==row["truth"]
                else:
                    ending=re.search(r";answer=(-?[0-9]+)$",text) if clean else None
                    final=int(ending[1]) if ending else None
                    if full and clean:
                        a,b,subtotal,previous,c,total,last=map(int,full.groups())
                        valid=(a+b==subtotal and previous+c==total and previous==subtotal and total==last
                               and (a,b,c)==(row["a"],row["b"],row["c"]) and last==row["truth"])
                    else: valid=False
                assert candidate["final_answer"]==final
                assert candidate["final_correct"]==(final==row["truth"])
                assert candidate["fully_verified"]==valid
                if final is not None: answers.append(final)
                correct.append(final==row["truth"])
                if valid: verified.append(final)
            majority=Counter(answers).most_common(1)[0][0] if answers else None
            verifier=verified[0] if verified else None
            coverage=any(correct); right=majority==row["truth"]
            assert sample["oracle_coverage"]==coverage and sample["majority_answer"]==majority
            assert sample["majority_correct"]==right and sample["verifier_correct"]==(verifier==row["truth"])
            assert sample["verified_answer"]==verifier
            counts["coverage"]+=coverage; counts["majority"]+=right; counts["verifier"]+=verifier==row["truth"]
            counts["selection_failure"]+=coverage and not right
            assert sum(lengths)==g["generated_tokens"]
            expected_forward=n*(len(g["input_ids"])+max(lengths)-1)
            assert g["forward_input_tokens"]==expected_forward
            token_total+=sum(lengths); seconds+=g["seconds"]; forward+=expected_forward
        assert token_total==budget["generated_tokens"] and abs(seconds-budget["generation_seconds"])<=1e-12
        for key, field in [("coverage","oracle_coverage"),("majority","majority_accuracy"),("verifier","verifier_accuracy")]:
            assert budget[field]["numerator"]==counts[key] and budget[field]["denominator"]==24
        out["budgets"].append({"mode":mode,"k":n,"questions":24,"candidates":24*n,"generated_tokens":token_total,
                              "eos_tokens":eos,"forward_input_tokens":forward,"batch_generation_seconds_sum":seconds,
                              "seconds_rounded_3dp":f"{seconds:.3f}",**counts})
out["step_budget_change"]={"generated_token_ratio_k8_to_k1":4132/512,"coverage_gain_questions":14-11,
                          "majority_gain_questions":12-11,"denominator":24}
assert f"{record['elapsed_seconds']:.2f}"=="45.71"

# Meaningful actual CPU dataflow: attention sees all cached keys while only projecting new positions.
torch.manual_seed(0); lm=TinyLM(ModelConfig(width=8)).eval(); ids=torch.tensor([[1,2,3,4]])
with torch.no_grad():
    full=lm(ids)["logits"][:,-1]; prefix=lm(ids[:,:3]); step=lm(ids[:,3:],cache=prefix["cache"])
error=(full-step["logits"][:,0]).abs().max().item()
assert torch.allclose(full,step["logits"][:,0],atol=1e-6)
out["cache"]={"full_logits_shape":list(full.shape),"cached_logits_shape":list(step["logits"][:,0].shape),
              "prefix_K_shape":list(prefix["cache"][0][0].shape),"step_K_shape":list(step["cache"][0][0].shape),
              "max_abs_error":error,"absolute_tolerance":1e-6}

# Actual sample execution and ordered event capture; no fake clock or simulated timings.
events=[]; original_prompt=apps._prompt_ids; original_decode=ByteTokenizer.decode; original_clock=apps.time.perf_counter
def prompt(*args,**kwargs): events.append({"event":"prompt_ids"}); return original_prompt(*args,**kwargs)
def decode(*args,**kwargs): events.append({"event":"decode_text"}); return original_decode(*args,**kwargs)
def clock(): events.append({"event":"perf_counter"}); return original_clock()
def hook(module,args,kwargs): events.append({"event":"model_forward","ids_shape":list(args[0].shape),"grad_enabled":torch.is_grad_enabled()})
lm.train(); before={k:v.clone() for k,v in lm.state_dict().items()}
h=lm.register_forward_pre_hook(hook,with_kwargs=True)
with patch.object(apps,"_prompt_ids",prompt),patch.object(ByteTokenizer,"decode",decode),patch.object(apps.time,"perf_counter",clock):
    sampled=apps._sample(lm,[{"role":"user","content":"(0+3)+3=?"}],SimpleNamespace(device="cpu"),count=4,tokens=4,temperature=0.7)
h.remove()
assert all(torch.equal(before[k],v) for k,v in lm.state_dict().items()) and lm.training
timers=[i for i,e in enumerate(events) if e["event"]=="perf_counter"]
assert len(timers)==2 and events[0]["event"]=="prompt_ids"
assert all(timers[0]<i<timers[1] for i,e in enumerate(events) if e["event"]=="model_forward")
assert all(i>timers[1] for i,e in enumerate(events) if e["event"]=="decode_text")
assert all(not e["grad_enabled"] for e in events if e["event"]=="model_forward")
out["actual_cpu_sample"]={"events":events,"generated_ids":[s["generated_ids"] for s in sampled["samples"]],
                          "generated_tokens":sampled["generated_tokens"],"forward_input_tokens":sampled["forward_input_tokens"],
                          "seconds":sampled["seconds"],"weights_unchanged":True,"training_mode_restored":True,
                          "scope":"Operation/timing-boundary check on random tiny CPU weights; no quality or GPU-speed replication."}
out["status"]="all assertions passed"
print(json.dumps(out,ensure_ascii=False,indent=2))
