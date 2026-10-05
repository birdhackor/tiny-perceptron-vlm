"""Fresh T.7 checks. No full training, asset download, or retained weights."""
import ast
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from scripts import train
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.behavior import _preference_parts, _pair_examples
from scripts.course_experiments.posttraining import build_records, split_records as card_splits
from scripts.course_experiments.run import experiment_spec, list_assets
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.posttraining import (FiniteResponsePolicy, FiniteRewardModel,
    FiniteValueModel, preference_loss, bandit_advantage, ppo_clipped_objective)
from tiny_perceptron.training import save_checkpoint

torch.set_num_threads(1)
torch.manual_seed(42)
results = {"environment": {"python": platform.python_version(), "executable": sys.executable,
    "torch": str(torch.__version__), "torch_git_version": torch.version.git_version,
    "device": "cpu", "cuda_available": str(torch.cuda.is_available()), "cuda_build": str(torch.version.cuda)}}
assert not torch.cuda.is_available() and torch.version.cuda is None

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def original_parser(path, argv):
    """Execute the original argparse-only prefix of main, without invoking its workload."""
    tree = ast.parse((ROOT / path).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    prefix = []
    for stmt in main.body:
        prefix.append(stmt)
        if isinstance(stmt, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "args" for x in stmt.targets):
            break
    namespace = {"argparse": __import__("argparse"), "Path": Path, "ROOT": ROOT,
                 "__doc__": "Original parser bounded check"}
    old = sys.argv
    try:
        sys.argv = [path] + argv
        exec(compile(ast.Module(body=prefix, type_ignores=[]), path, "exec"), namespace)
    finally:
        sys.argv = old
    return vars(namespace["args"])

original_argv = ["--task", "dpo", "--checkpoint", "checkpoints/style.pt", "--data",
    "data/generated/preference/train.jsonl", "--train", "--steps", "200", "--output", "checkpoints/preferred.pt"]
args = train.parser().parse_args(original_argv)
assert args.task == "dpo" and args.train and args.steps == 200
inference = original_parser("scripts/infer.py", ["checkpoints/style.pt", "--chat", "--prompt", "0+5=?", "--tokens", "32", "--temperature", "0"])
assert inference["chat"] and inference["tokens"] == 32 and inference["temperature"] == 0
fetch = original_parser("scripts/fetch_training_assets.py", ["--asset", "ultrafeedback-dpo"])
assert fetch["asset"] == ["ultrafeedback-dpo"]
for name, device in [("dpo", "cuda"), ("posttraining", "cpu")]:
    p = original_parser("scripts/course_experiments/run.py", ["--experiment", name, "--device", device])
    assert p["experiment"] == name and p["device"] == device
results["recipe_contracts"] = {"local_dpo_args": original_argv, "inference": {k: str(v) for k,v in inference.items()},
    "fixed_dpo_spec": experiment_spec("dpo"), "posttraining_spec": experiment_spec("posttraining"),
    "dpo_asset_archives": list_assets("dpo"),
    "execution_scope": "Original parsers checked; full DPO, asset fetching, and fixed card training deliberately not invoked."}

env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
commands = []
with tempfile.TemporaryDirectory(prefix="t7-bounded-") as name:
    temp = Path(name)
    for entry in (".venv", "scripts", "tiny_perceptron"):
        (temp / entry).symlink_to(ROOT / entry, target_is_directory=True)
    def run(label, argv):
        completed = subprocess.run(argv, cwd=temp, env=env, text=True, capture_output=True, timeout=45)
        (OUT / (label + ".stdout.txt")).write_text(completed.stdout)
        (OUT / (label + ".stderr.txt")).write_text(completed.stderr)
        commands.append({"label": label, "argv": argv, "cwd": str(temp), "returncode": completed.returncode,
                         "stdout": label + ".stdout.txt", "stderr": label + ".stderr.txt"})
        assert completed.returncode == 0, completed.stderr
        return completed
    run("prepare-original", [".venv/bin/python", "scripts/prepare_data.py", "--kind", "preference"])
    parts = {s:[json.loads(x) for x in (temp / f"data/generated/preference/{s}.jsonl").read_text().splitlines()]
             for s in ("train", "validation", "test")}
    families = {s:{r["family"] for r in rows} for s, rows in parts.items()}
    assert [len(parts[s]) for s in parts] == [51,6,7]
    assert all(not families[a] & families[b] for a,b in [("train","validation"),("train","test"),("validation","test")])
    for rows in parts.values():
        for r in rows:
            a,b = map(int,r["family"].split("+"))
            assert r["chosen"] == str(a+b) and r["rejected"] == str(a+b+1)
    zero_five = next(r for rows in parts.values() for r in rows if r["prompt"] == "0+5=?")
    results["local_data"] = {"counts": {s:len(rows) for s,rows in parts.items()}, "zero_five": zero_five,
        "family_contract": "ordered operands in local generator; all records retain their named family"}
    tiny = TinyLM(ModelConfig(width=8,layers=1,heads=1,max_length=256))
    original_state = copy.deepcopy(tiny.state_dict())
    save_checkpoint(temp / "checkpoints/style.pt", tiny)
    bounded_argv = [".venv/bin/python", "scripts/train.py"] + original_argv + ["--stop-after", "2", "--device", "cpu"]
    run("train-bounded-two-updates", bounded_argv)
    payload = torch.load(temp / "checkpoints/preferred.pt", weights_only=True)
    assert payload["step"] == 2 and payload["metadata"]["schedule_steps"] == 200
    policy_changed = any(not torch.equal(v,payload["model"][k]) for k,v in original_state.items())
    reference_equal = all(torch.equal(v,payload["training_state"]["dpo_reference"]["model"][k]) for k,v in original_state.items())
    assert policy_changed and reference_equal
    for checkpoint,label in [("style.pt","infer-before"),("preferred.pt","infer-after")]:
        run(label,[".venv/bin/python","scripts/infer.py","checkpoints/"+checkpoint,"--chat","--prompt","0+5=?","--tokens","32","--temperature","0"])
    results["two_updates"] = {"temporary_start": "random 8-width model; no capability claim", "policy_changed":policy_changed,
        "reference_exactly_equal_to_start":reference_equal,"completed_updates":2,"configured_schedule_steps":200,
        "checkpoint_handling":"temporary native save/load only; all new weights deleted with TemporaryDirectory"}

chosen = torch.tensor([-4.0], requires_grad=True)
rejected = torch.tensor([-3.0], requires_grad=True)
refc = torch.tensor([-4.0], requires_grad=True)
refr = torch.tensor([-3.0], requires_grad=True)
loss = dpo_loss(chosen,rejected,refc,refr,beta=.1)
loss.backward()
assert abs(float(loss.detach())-math.log(2)) < 1e-6
assert abs(float(chosen.grad)+.05) < 1e-7 and abs(float(rejected.grad)-.05) < 1e-7
assert refc.grad is None and refr.grad is None
better = float(dpo_loss(torch.tensor([-3.0]),rejected.detach(),refc,refr,beta=.1))
assert better < float(loss.detach())
results["dpo_math"] = {"equal_margin_loss":float(loss.detach()),"chosen_gradient":float(chosen.grad),
    "rejected_gradient":float(rejected.grad),"reference_gradients":None,"improved_margin_loss":better,
    "tolerance":"absolute 1e-6 loss; 1e-7 score gradients"}

# A pair can rank 5 above 6 while greedy decoding chooses another answer 7.
probs = torch.tensor([.35,.25,.40])
assert probs[0] > probs[1] and int(probs.argmax()) == 2
results["ranking_counterexample"] = {"candidate_labels":["5","6","7"],"probabilities":probs.tolist(),
    "pair_ranks_correctly":True,"greedy_answer":"7","scope":"hand-set distribution, no model score"}

fixed = _preference_parts(split_records(arithmetic_records(),seed=42))
fixed_families = {s:{r["family"] for r in rows} for s,rows in fixed.items()}
assert [len(fixed[s]) for s in fixed] == [49,8,7]
assert all(not fixed_families[a] & fixed_families[b] for a,b in [("train","validation"),("train","test"),("validation","test")])
for rows in fixed.values():
    for r in rows:
        a,b = map(int,r["prompt"].removesuffix("=?").split("+"))
        assert r["chosen"] == str(a+b) and r["rejected"] == str(a+b+1)
        pair = _pair_examples([r],256)[0]
        assert all(int((y != -100).sum()) == len(r[side].encode())+1 for (_,y),side in zip(pair,("chosen","rejected")))
raw = json.loads((OUT / "inspected-raw-pointers.json").read_text())["raw_values"]
for split, rows in fixed.items():
    raw_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    manifest = raw["/results/data"][split]
    assert manifest["records"] == len(rows) and manifest["families"] == len(fixed_families[split])
    assert manifest["sha256"] == hashlib.sha256(raw_bytes).hexdigest()
verified_counts = {}
tok = ByteTokenizer()
for branch in ("model","beta1"):
    for split in ("validation","test"):
        base = f"/results/runs/{branch}"
        pref = raw[base+f"/preference/{split}/samples"]
        n = raw[base+f"/preference/{split}/records"]
        assert n == len(pref) == len(fixed[split])
        for row in pref:
            assert abs(row["policy_margin"]-(row["policy_chosen_logp"]-row["policy_rejected_logp"])) < 1e-9
            assert abs(row["relative_margin"]-(row["policy_margin"]-row["reference_margin"])) < 1e-9
            assert row["chosen_answer_tokens"] == len(row["chosen"].encode())+1
            assert row["rejected_answer_tokens"] == len(row["rejected"].encode())+1
        absolute = sum(x["policy_chosen_logp"] > x["policy_rejected_logp"] for x in pref)
        relative = sum(x["relative_margin"] > 0 for x in pref)
        assert absolute == raw[base+f"/preference/{split}/chosen_higher_absolute_probability"]
        assert relative == raw[base+f"/preference/{split}/relative_preference_improved"]
        assert abs(sum(x["relative_margin"] for x in pref)/n-raw[base+f"/preference/{split}/mean_relative_margin"]) < 1e-9
        samples = raw[base+f"/arithmetic/{split}/samples"]
        assert n == raw[base+f"/arithmetic/{split}/records"] == len(samples)
        exact = ended = 0
        for sample in samples:
            ids = sample["generated_ids"]
            bare = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            is_exact = bare == tok.encode(sample["expected"])
            is_eos = tok.eos_id in ids
            assert sample["exact"] == is_exact and sample["eos"] == is_eos
            exact += is_exact; ended += is_eos
        assert exact == raw[base+f"/arithmetic/{split}/matches"]
        assert abs(exact/n-raw[base+f"/arithmetic/{split}/exact_match"]) < 1e-12
        assert abs(ended/n-raw[base+f"/arithmetic/{split}/eos_rate"]) < 1e-12
        verified_counts[branch+":"+split] = {"denominator":n,"absolute_pair_wins":absolute,
            "relative_pair_improvements":relative,"generated_exact_matches":exact,"eos":ended}
results["existing_report_recalculation"] = {"counts":verified_counts,"split_counts":{s:len(r) for s,r in fixed.items()},
    "scope":"Recomputed existing raw sample fields and split recipe; no historical retraining or model capability score."}

cards = card_splits(build_records())
record = cards["train"][0]
features = torch.tensor([record["features"]])
assert len(record["candidates"]) == 4 and len(record["features"]) == 4
assert all(row["expected_action"] == {"number":0,"explain":1,"missing":3}[row["mode"]] for rows in cards.values() for row in rows)
policy, rm, value = FiniteResponsePolicy(), FiniteRewardModel(), FiniteValueModel()
reference = copy.deepcopy(policy).eval().requires_grad_(False)
original = copy.deepcopy(reference.state_dict())
rm_scores = rm(features)
winner, loser = record["preference_pairs"][0]
rm_loss = preference_loss(rm_scores[:,winner],rm_scores[:,loser])
rm_loss.backward()
old = policy(features).log_softmax(-1).detach()
action = torch.tensor([winner])
rewards = rm_scores[:,winner].detach()
old_values = value(features).detach()
advantage = bandit_advantage(rewards,old_values)
terms = ppo_clipped_objective(policy(features).log_softmax(-1).gather(1,action[:,None]).squeeze(1),
    old.gather(1,action[:,None]).squeeze(1),advantage)
optimizer = torch.optim.SGD(policy.parameters(),lr=.01)
optimizer.zero_grad();terms["policy_loss"].backward();optimizer.step()
assert all(torch.equal(v,reference.state_dict()[k]) for k,v in original.items())
assert any(not torch.equal(v,policy.state_dict()[k]) for k,v in original.items())
results["finite_card_bounded_update"] = {"candidate_count":4,"input_shape":list(features.shape),
    "policy_output_shape":list(policy(features).shape),"reward_output_shape":list(rm_scores.shape),
    "value_output_shape":list(value(features).shape),"one_policy_update":True,"reference_unchanged":True,
    "split_counts":{s:len(r) for s,r in cards.items()},"label_checks":"all 165 records follow author rules; answers are template strings",
    "scope":"One CPU update checks machinery only; fixed run_posttraining refuses step_scale != 1, so full recipe was not run."}
dump("commands.json",commands)
dump("verification-results.json",results)
print(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False))
