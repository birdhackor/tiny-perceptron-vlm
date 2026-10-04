"""Fresh bounded CPU audit of C.7; no language model or full policy training."""
import contextlib
import hashlib
import io
import json
import math
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import applications as app
from scripts.course_experiments.common import records_sha256, split_records, fit, seed
from tiny_perceptron.training import load_checkpoint

torch.set_num_threads(1)
EVIDENCE = Path(__file__).parent
RESEARCH = ROOT / "outputs/natural-v4/factual-research/C.7"
env = {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": "1"}
assert str(torch.__version__) == "2.14.1+cpu"
proof = {"environment": env, "scope": "Fresh toy executions, deterministic full fixed-record audit, existing exported-checkpoint forward checks, and one update of each finite policy; no reproduction of the L4 training run."}

section = (EVIDENCE / "source-original.md").read_text()
toy = re.search(r"```python\n(.*?)```", section, re.S)[1]
proof["toy"] = {}
for label, code, expected_grad, expected_prob in [
    ("reward1", toy, [0.25, -0.25], [0.487503, 0.512497]),
    ("exercise_reward0", toy.replace("reward, baseline = 1.0, 0.5", "reward, baseline = 0.0, 0.5"), [-0.25, 0.25], [0.512497, 0.487503]),
]:
    buf, namespace = io.StringIO(), {}
    with contextlib.redirect_stdout(buf):
        exec(compile(code, label, "exec"), namespace)
    assert namespace["logits"].grad.tolist() == expected_grad
    assert [round(p, 6) for p in namespace["after"].tolist()] == expected_prob
    assert not namespace["before"].requires_grad and not namespace["updated"].requires_grad
    assert namespace["logits"].tolist() == [0, 0]
    exact_prob = [1/(1+math.exp(.05)), 1/(1+math.exp(-.05))]
    if label != "reward1":
        exact_prob.reverse()
    proof["toy"][label] = {"stdout": buf.getvalue(), "dtype": str(namespace["logits"].dtype), "max_error_against_double_formula": max(abs(p - q) for p, q in zip(namespace["after"].tolist(), exact_prob))}

proof["baseline_estimator"] = {}
for baseline in (0.0, 0.5):
    updates = []
    for action, reward in [(0, 0.0), (1, 1.0)]:
        logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        objective = (reward-baseline) * logits.log_softmax(0)[action]
        updates.append(torch.autograd.grad(objective, logits)[0])
    samples = torch.stack(updates)
    assert torch.equal(samples.mean(0), torch.tensor([-.25, .25], dtype=torch.float64))
    proof["baseline_estimator"][str(baseline)] = {"sample_gradients": samples.tolist(), "exact_expectation": samples.mean(0).tolist(), "component_variance": samples.var(0, unbiased=False).tolist()}

ctx = SimpleNamespace(device="cpu", seed=42)
splits = split_records(app._reasoning_records(), 42)
record_path = ROOT / "docs/course-experiments/results/reasoning.json"
record = json.loads(record_path.read_text())
assert record["device"] == "cuda" and record["gpu"] == "NVIDIA L4" and record["seed"] == 42
assert record["step_scale"] == 1 and record["evidence_status"] == "complete_run" and not record["unfinished_schedules"]
split_summary = {k: {"records": len(v), "families": len({r["family"] for r in v}), "sha256": records_sha256(v)} for k, v in splits.items()}
assert split_summary == record["results"]["split"]
family_sets = {k: {r["family"] for r in rows} for k, rows in splits.items()}
assert all(not family_sets[a] & family_sets[b] for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")])
proof["record"] = {"path": str(record_path.relative_to(ROOT)), "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(), "seed": record["seed"], "device": record["device"], "gpu": record["gpu"], "python": record["python_version"], "torch": record["torch_version"], "elapsed_seconds": record["elapsed_seconds"], "timing_scope": record["timing_scope"], "splits": split_summary, "disjoint_operand_families": True, "branch_training": {}}
for path in ["scripts/course_experiments/applications.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py"]:
    observed = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    assert observed == record["code_sha256"][path]

traces = record["results"]["reinforce"]
assert traces["strict"]["before"] == traces["weak_proxy"]["before"]
expected = {("strict", "before"): [1,23,62,39], ("weak_proxy", "before"): [1,23,62,39], ("strict", "after"): [0,0,0,0], ("weak_proxy", "after"): [0,0,384,384]}
proof["table_recomputed"] = {}
for branch in ["strict", "weak_proxy"]:
    training = traces[branch]["training"]
    assert training["steps"] == training["planned_steps"] == 1200
    assert training["sampled_actions"] == 1200 * 64 == 76800 and training["effective_tokens"] == 0
    assert all(abs(row["sampled_mean_reward"]*64 - round(row["sampled_mean_reward"]*64)) < 1e-12 for row in training["reward_history"])
    proof["record"]["branch_training"][branch] = {k: training[k] for k in ["steps", "planned_steps", "step_scale", "sampled_actions", "effective_tokens", "seconds", "checkpoint"]}
    proof["record"]["branch_training"][branch]["last_logged_reward"] = training["reward_history"][-1]
    for when in ["before", "after"]:
        evaluation = traces[branch][when]
        rows = evaluation["samples"]
        assert len(rows) == 24
        assert [{k:r[k] for k in ["family","a","b","c","question","truth"]} for r in rows] == splits["test"]
        greedy = strict = weak = enum = 0
        for row in rows:
            assert row["truth"] == row["a"]+row["b"]+row["c"] and len(row["samples"]) == 16
            probs = row["probabilities"]
            assert len(probs) == 17 and all(0 <= p <= 1 for p in probs) and abs(sum(probs)-1)<2e-7
            chosen = max(range(17), key=lambda i: probs[i])
            assert app._POLICY_ACTIONS[chosen] == row["greedy_action"]
            g = app._policy_reward(chosen, row["truth"])
            assert bool(g) == row["greedy_strict_correct"]
            greedy += int(g)
            for sample in row["samples"]:
                action=sample["action_id"]
                assert app._POLICY_ACTIONS[action] == sample["generated"]
                sr = app._policy_reward(action,row["truth"])
                pr = app._policy_reward(action,row["truth"],True)
                assert (sr,pr) == (sample["strict_reward"], sample["proxy_reward"])
                strict += int(sr); weak += int(pr); enum += int(action == 16)
        counts=[greedy,strict,weak,enum]
        assert counts == expected[(branch,when)]
        for key,count,denominator in zip(["greedy_accuracy","sample_accuracy","sample_proxy_reward","enumeration_action_rate"],counts,[24,384,384,384]):
            assert evaluation[key] == {"numerator": count,"denominator":denominator,"rate":count/denominator}
        proof["table_recomputed"][branch+"_"+when]={"greedy": [greedy,24], "sample_strict": [strict,384], "sample_proxy": [weak,384], "enumeration": [enum,384]}
assert traces["strict"]["training"]["reward_history"][-1]["sampled_mean_reward"] == 39/64
proof["example"]={}
for branch in ["strict","weak_proxy"]:
    row=traces[branch]["after"]["samples"][0]
    assert row["question"] == "(0+3)+3=?" and row["truth"] == 6
    assert all(s["action_id"]==(7 if branch=="strict" else 16) for s in row["samples"])
    proof["example"][branch]={"question":row["question"],"truth":row["truth"],"greedy":row["greedy_action"],"sampled_ids":[s["action_id"] for s in row["samples"]]}

proof["checkpoint_forwards"] = {}
seed(42); initial=app._FinitePolicy()
initial_probs=initial(app._policy_features(splits["test"],ctx)).softmax(-1).detach()
initial_record=torch.tensor([r["probabilities"] for r in traces["strict"]["before"]["samples"]])
assert torch.allclose(initial_probs,initial_record,atol=2e-7,rtol=0)
proof["initial_probability_max_cpu_l4_error"]=(initial_probs-initial_record).abs().max().item()
for branch in ["strict","weak_proxy"]:
    path=ROOT/"checkpoints/course/reasoning"/f"policy-{branch}.pt"
    payload=torch.load(path,map_location="cpu",weights_only=True)
    model=app._FinitePolicy();model.load_state_dict(payload["model"],strict=True)
    probs=model(app._policy_features(splits["test"],ctx)).softmax(-1).detach()
    target=torch.tensor([r["probabilities"] for r in traces[branch]["after"]["samples"]])
    assert torch.allclose(probs,target,atol=2e-6,rtol=0)
    assert any(not torch.equal(initial.state_dict()[key],value) for key,value in payload["model"].items())
    try: load_checkpoint(path,"cpu")
    except ValueError as error: loader_result=str(error)
    else: raise AssertionError("Finite policy must not load as TinyLM")
    proof["checkpoint_forwards"][branch]={"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"format_version":payload["format_version"],"architecture":payload["architecture"],"max_cpu_l4_probability_error":(probs-target).abs().max().item(),"weights_changed_from_seed42":True,"TinyLM_loader_rejection":loader_result}

features=app._policy_features([{"a":2,"b":3,"c":5}],ctx)
assert features.shape==(1,18) and features.sum().item()==3 and features[0,:6].tolist()==[0,0,1,0,0,0]
uniform=torch.distributions.Categorical(logits=torch.zeros(17,dtype=torch.float64))
peak=torch.distributions.Categorical(probs=torch.tensor([1.]+[0.]*16,dtype=torch.float64))
assert abs(uniform.entropy().item()-math.log(17))<1e-14 and abs(peak.entropy().item())<1e-14
proof["features_entropy"]={"features":features.tolist(),"uniform_entropy":uniform.entropy().item(),"concentrated_entropy_float_roundoff":peak.entropy().item(),"uniform_entropy_bonus":-.005*uniform.entropy().item()}

# Execute the actual _reinforce loss closure and actual fit optimizer/checkpoint path,
# but bound each branch to exactly one update. Preserve its full-schedule metadata.
original_fit=app._fit
updates=[]
def one_update(model,loss_fn,ctx,*,steps,lr,name,metadata):
    assert steps==1200 and lr==.003
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    result=fit(model,loss_fn,ctx,steps=1,lr=lr,name=name,metadata=metadata)
    checkpoint=torch.load(ctx.output/(name+".pt"),map_location="cpu",weights_only=True)
    assert checkpoint["step"]==1 and checkpoint["optimizer"]["param_groups"][0]["lr"]==.003
    assert checkpoint["optimizer"]["param_groups"][0]["weight_decay"]==.01
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
    updates.append({"name":name,"executed_updates":1,"requested_full_updates":steps,"optimizer":"AdamW","lr":lr,"weight_decay":.01,"checkpoint_keys":list(checkpoint),"weights_changed":True})
    return {**result,"planned_steps":steps,"step_scale":"bounded one-update probe"}
try:
    app._fit=one_update
    bounded=app._reinforce(splits,SimpleNamespace(device="cpu",seed=42,output=RESEARCH/"one-update"))
finally:app._fit=original_fit
for branch in ["strict","weak_proxy"]:
    t=bounded[branch]["training"]
    assert t["sampled_actions"]==64 and t["effective_tokens"]==0
    first=t["reward_history"][0]
    assert math.isclose(first["moving_baseline"],.05*first["sampled_mean_reward"],abs_tol=1e-15)
    updates[["strict","weak_proxy"].index(branch)]["first_reward_baseline"]=first
proof["one_update_actual_repo_closure"]=updates

proof["entrypoint"]={}
for args in [["--help"],["--list-assets","reasoning"]]:
    command=[sys.executable,"scripts/course_experiments/run.py",*args]
    proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode==0
    if "--list-assets" in args: assert proc.stdout=="assets/training/gsm8k-v1.tar.gz\n"
    proof["entrypoint"][" ".join(args)]={"command":command,"returncode":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr}

proof["all_assertions_passed"]=True
(EVIDENCE/"cpu-audit.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(proof,ensure_ascii=False,indent=2))
