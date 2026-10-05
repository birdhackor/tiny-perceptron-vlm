"""Independent T.10 checks: stored evidence arithmetic and bounded CPU contracts.

No downloads, full schedules, new quality measurements, or retained new weights.
Only selected raw JSON pointers are inspected; authored result prose is not read.
"""
from pathlib import Path
import ast
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import torch
from tiny_perceptron.alignment import distillation_kl, distillation_loss
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.training import load_checkpoint, seed_everything
from scripts.course_experiments import compression as c
from scripts.course_experiments.run import experiment_spec

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, obj):
    (ART / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

env = {"python": platform.python_version(), "python_executable": sys.executable, "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "cuda_build": str(torch.version.cuda), "device": "cpu", "threads": str(torch.get_num_threads()), "platform": platform.platform()}
assert env["torch_git_version"] == "5c4886908584029761b579af026dcfb627c84070"
for mod, snap in [(torch.nn.functional, "pytorch-functional-installed.py"), (torch.nn.modules.loss, "pytorch-loss-installed.py")]:
    assert sha(inspect.getfile(mod)) == sha(ART / "sources" / snap)
env["official_source_same_as_installed"] = "functional.py and nn/modules/loss.py byte-identical to pinned upstream snapshots"
save("environment.json", env)

raw = json.loads((ART / "raw/distillation.json").read_text())
tok = ByteTokenizer()
checked_pointers = ["/revision", "/device", "/seed", "/torch_version", "/step_scale", "/evidence_status", "/unfinished_schedules", "/code_sha256", "/artifacts"]
cache_path = ART / "raw/original-sft-teacher-logits.raw-snapshot"
cache_sha = next(a["sha256"] for a in raw["artifacts"] if a["path"] == "sft-teacher-logits.pt")
assert sha(cache_path) == cache_sha
cache = torch.load(cache_path, map_location="cpu", weights_only=True)
assert set(cache) == {"logits", "teacher_sha256", "alignment", "temperature"}
assert cache["teacher_sha256"] == raw["results"]["tasks"]["attributes"]["teacher_provenance"]["sha256"]
assert cache["temperature"] == 2.0
assert len(cache["logits"]) == 45
assert all(t.ndim == 2 and t.shape[1] == 264 and not t.requires_grad and torch.isfinite(t).all() for t in cache["logits"])
cache_facts = {"source_sha256": cache_sha, "teacher_sha256": cache["teacher_sha256"], "temperature": cache["temperature"], "tensor_rows": [len(t) for t in cache["logits"]], "vocab_columns": 264, "records": len(cache["logits"]), "saved_values": "raw pre-softmax logits; no new targets generated", "payload_keys_types": {k:type(v).__name__ for k,v in cache.items()}}

def rescore(ev, text_task):
    samples = ev["generated_samples"]
    assert len(samples) == ev["examples"]
    hits, completed, eos = [], [], []
    for s in samples:
        ids = s["generated_ids"]
        ended = tok.eos_id in ids
        answer_ids = ids[:ids.index(tok.eos_id)] if ended else ids
        hit = answer_ids == tok.encode(s["expected"])
        assert hit == s["exact"] and ended == s["ended_with_eos"]
        hits.append(hit); completed.append(hit and ended); eos.append(ended)
    assert sum(hits) == ev["correct"] and sum(completed) == ev["completed_correct"] and sum(eos) == ev["eos_count"]
    for n,d,f in [(sum(hits),len(samples),"exact_match"),(sum(completed),len(samples),"completed_exact_match"),(sum(eos),len(samples),"eos_rate"),(ev["nll_sum"],ev["supervised_tokens"],"answer_nll")]:
        assert math.isclose(n/d, ev[f], rel_tol=0, abs_tol=1e-12)
    if not text_task:
        count = sum(len(tok.encode(s["expected"])) + 1 for s in samples)
        assert count == ev["supervised_tokens"]
        assert sum(len(s["expected"].encode()) for s in samples) == ev["answer_bytes"]
    return {"correct":sum(hits),"completed_correct":sum(completed),"eos_count":sum(eos),"examples":len(samples),"effective_nll_positions":ev["supervised_tokens"],"exact_match":ev["exact_match"],"completed_exact_match":ev["completed_exact_match"],"nll_arithmetic_checked":True,"nll_position_recounted_from_gold_samples":not text_task}, hits

tasks = {}
for name, t in raw["results"]["tasks"].items():
    base = "/results/tasks/" + name
    checked_pointers += [base+"/teacher_provenance/{experiment,checkpoint,sha256,format_version,steps,config}", base+"/teacher_storage/{parameter_count,tensor_bytes,file_bytes}", base+"/data/{source,sha256,counts,families,student_training_records}", base+"/teacher_cache/{file,file_bytes,seconds}", base+"/teacher_test/{counts,generated_samples}"]
    cfg = t["teacher_provenance"]["config"]
    teacher_count = sum(p.numel() for p in TinyLM(ModelConfig(**cfg)).parameters())
    assert teacher_count == t["teacher_storage"]["parameter_count"]
    student_count = sum(p.numel() for p in TinyLM(ModelConfig(width=32,layers=1,heads=2,max_length=cfg["max_length"])).parameters())
    teacher_eval, _ = rescore(t["teacher_test"], name == "moe_to_dense")
    runs, hit_lists = {}, {}
    for key in ("w32_ce", "w32_ce_kl"):
        r = t["runs"][key]; tr = r["training"]
        checked_pointers += [base+"/runs/"+key+"/training/{steps,optimizer_updates,batch_size,initialization_sha256,batch_plan_sha256,training_examples,training_sequence_chunks,effective_supervised_tokens,alpha,temperature,temperature_squared_applied_once,objective,final_sha256}", base+"/runs/"+key+"/storage/{parameter_count,tensor_bytes,file_bytes}", base+"/runs/"+key+"/test/{counts,nll_sum,supervised_tokens,generated_samples}"]
        assert student_count == r["storage"]["parameter_count"] < teacher_count
        assert r["storage"]["parameter_tensor_bytes"] == student_count * 4
        assert r["storage"]["file_bytes"] >= r["storage"]["tensor_bytes"]
        assert tr["optimizer_updates"] == tr["steps"]
        rng = random.Random(raw["seed"])
        plan = [[rng.randrange(tr["training_sequence_chunks"]) for _ in range(tr["batch_size"])] for _ in range(tr["steps"])]
        plan_sha = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
        assert plan_sha == tr["batch_plan_sha256"]
        if name == "attributes":
            effective = sum(len(cache["logits"][i]) for batch in plan for i in batch)
            assert effective == tr["effective_supervised_tokens"]
        counts, hits = rescore(r["test"], name == "moe_to_dense")
        hit_lists[key] = hits
        for s, ts in zip(r["test"]["generated_samples"],t["teacher_test"]["generated_samples"],strict=True):
            assert (s["family"],s["question"],s["expected"]) == (ts["family"],ts["question"],ts["expected"])
        agree = sum(s["generated_ids"] == ts["generated_ids"] for s,ts in zip(r["test"]["generated_samples"],t["teacher_test"]["generated_samples"],strict=True))
        assert agree/len(t["teacher_test"]["generated_samples"]) == r["teacher_agreement"]
        runs[key] = {"steps":tr["steps"],"batch_size":tr["batch_size"],"training_records":tr["training_examples"],"training_chunks":tr["training_sequence_chunks"],"effective_positions":tr["effective_supervised_tokens"],"alpha":tr["alpha"],"temperature":tr["temperature"],"parameters":student_count,"parameter_bytes":student_count*4,"native_checkpoint_bytes":r["storage"]["file_bytes"],"test":counts,"agreement_numerator":agree,"agreement_denominator":len(t["teacher_test"]["generated_samples"])}
    a,b=(t["runs"][k]["training"] for k in ("w32_ce","w32_ce_kl"))
    for k in ("initialization_sha256","batch_plan_sha256","steps","batch_size","training_examples","training_sequence_chunks","effective_supervised_tokens"):
        assert a[k] == b[k]
    assert a["alpha"] == 0 and b["alpha"] == 0.5 and b["temperature_squared_applied_once"] is True
    tasks[name] = {"teacher_experiment":t["teacher_provenance"]["experiment"],"teacher_prior_steps":t["teacher_provenance"]["steps"],"teacher_parameters":teacher_count,"data_counts":t["data"]["counts"],"data_sha256":t["data"]["sha256"],"runs":runs,"paired_input_start_schedule_verified":True,"old_stored_sample_comparison":{"improved":sum(not x and y for x,y in zip(hit_lists['w32_ce'],hit_lists['w32_ce_kl'],strict=True)),"regressed":sum(x and not y for x,y in zip(hit_lists['w32_ce'],hit_lists['w32_ce_kl'],strict=True))}}

# Recount original limited GSM8K selection, without downloading or running generation.
gsm = [json.loads(line) for line in (ART / "raw/gsm8k-train-first200.jsonl").read_text().splitlines() if line.strip()]
selection = raw["results"]["tasks"]["attributes"]["out_of_domain_gsm8k"]["selection"]
assert sha(ART / "raw/gsm8k-train-first200.jsonl") == selection["source_file_sha256"]
eligible, skipped = [], []
for i,r in enumerate(gsm):
    gold = r["answer"].split("####")[-1].strip().replace(",", "")
    prefix = len(tok.encode(r["question"])) + 4
    if prefix + selection["reserved_generation_tokens"] <= selection["max_length"] and len(tok.encode(gold))+1 <= selection["reserved_generation_tokens"]:
        eligible.append(i)
    else: skipped.append(i)
assert len(gsm)==selection["source_rows"]==200
assert len(eligible)==selection["eligible_rows"] and len(skipped)==selection["skipped_rows"]
assert eligible[:6]==selection["selected_source_rows"] and len(eligible[:6])==selection["selected_rows"]
checked_pointers.append("/results/tasks/attributes/out_of_domain_gsm8k/selection/{source_rows,source_file_sha256,eligible_rows,selected_rows,skipped_rows,selected_source_rows,max_length,reserved_generation_tokens}")
save("raw-verification.json", {"raw_source_sha256":sha(ART/"raw/distillation.json"),"original_revision":raw["revision"],"original_environment":{"device":raw["device"],"torch":raw["torch_version"],"seed":raw["seed"],"step_scale":raw["step_scale"]},"tasks":tasks,"cache":cache_facts,"gsm8k":{"source_rows":len(gsm),"eligible":len(eligible),"skipped":len(skipped),"selected":eligible[:6],"scope":"recount of original restricted selection, no generation/benchmark"},"inspected_pointers":checked_pointers,"limits":"Stored results were recounted, not retrained or rescored by fresh generation. MoE NLL denominator and file-size metadata are original observations; full story bytes are unavailable locally, so only their recorded arithmetic is checked."})

# Test exactly the current original inference fence in a fixture working directory.
# Released native-file packaging differs, but model state is identical to the original run.
state_identity = {}
for key in ("ce", "ce_kl"):
    path = ART / "raw" / f"original-export-sft-w32-{key}.raw-snapshot"
    model, payload = load_checkpoint(path,"cpu")
    expected = raw["results"]["tasks"]["attributes"]["runs"]["w32_"+key]["training"]["final_sha256"]
    assert c._parameter_hash(model) == expected
    state_identity[key] = {"input_path":str(path.relative_to(ROOT)),"input_file_sha256":sha(path),"model_state_sha256":expected,"config":payload["config"],"original_artifact_file_sha256":next(a["sha256"] for a in raw["artifacts"] if a["path"]==f"sft-w32-{key}.pt"),"same_model_state":True,"packaging":"existing student export; original run weights, not newly trained"}
section = (ART / "frozen/T.10.md").read_bytes()
fences = []
opened=None
for line in section.splitlines(keepends=True):
    if line.startswith(b"```bash"):
        opened=[]
    elif line.startswith(b"```") and opened is not None:
        fences.append(b"".join(opened));opened=None
    elif opened is not None: opened.append(line)
assert len(fences) == 2
(ART/"original-long-recipe.sh").write_bytes(fences[0])
(ART/"original-inference-fence.sh").write_bytes(fences[1])
commands = fences[1].decode().splitlines()
cli=[]
with tempfile.TemporaryDirectory(prefix="t10-inference-") as work:
    work=Path(work)
    for name in (".venv","scripts","tiny_perceptron"):
        (work/name).symlink_to(ROOT/name,target_is_directory=True)
    target=work/"outputs/course-experiments/course-v1/distillation";target.mkdir(parents=True)
    for key in ("ce","ce_kl"):
        (target/f"sft-w32-{key}.pt").symlink_to(ART/"raw"/f"original-export-sft-w32-{key}.raw-snapshot")
    for i, command in enumerate(commands,1):
        r=subprocess.run(["bash","-c",command],cwd=work,capture_output=True,text=True,timeout=30,env=os.environ.copy())
        (ART/f"inference-{i}.stdout.txt").write_text(r.stdout)
        (ART/f"inference-{i}.stderr.txt").write_text(r.stderr)
        assert r.returncode==0, r.stderr
        data=json.loads(r.stdout)
        assert set(data)=={"answer","generated_ids","eos","invalid_special_tokens","valid_answer_tokens","generation_status"}
        assert len(data["generated_ids"]) <= 24
        cli.append({"command":command,"exit_code":r.returncode,"stdout_file":f"inference-{i}.stdout.txt","stderr_file":f"inference-{i}.stderr.txt","generated_schema_keys":list(data),"generation_units":len(data["generated_ids"]),"result":data,"cwd_fixture":"temporary isolated layout mirroring documented output paths with symlinks to existing original-state exported checkpoints"})
save("original-inference-execution.json", {"fence_sha256":sha(ART/"original-inference-fence.sh"),"environment":env,"inputs":state_identity,"commands":cli,"scope":"The original short commands executed unchanged in the fixture layout; no model download, training, or quality benchmark."})

# A two-row, one-update contract exercise preserves the true fitting function.
# Only saving is intercepted; all fitting computations and default objective are real.
records=[{"question":"color=red;shape=square;pitch=high;joint?","answer":"square,high"},{"question":"color=blue;shape=circle;pitch=low;shape?","answer":"circle"}]
seed_everything(42)
teacher=TinyLM(ModelConfig(width=64,layers=2,heads=1,max_length=128))
teacher_before=c._parameter_hash(teacher)
teacher_cache,_=c._cache_text(teacher,records,"cpu")
seed_everything(42)
initial=TinyLM(ModelConfig(width=16,layers=1,heads=2,max_length=128))
old_save,old_float=c.save_checkpoint,c._save_float
saved=[]
c.save_checkpoint=lambda *args,**kwargs:saved.append(str(args[0]))
c._save_float=lambda ctx,model,name,steps,metadata:Path("not-written")/f"{name}.pt"
try:
    ctx=SimpleNamespace(seed=42,device="cpu",output=ART)
    runs={}
    for key,targets in (("ce",None),("ce_kl",teacher_cache)):
        fitted,path,report=c._fit_text(ctx,copy.deepcopy(initial),records,"bounded-"+key,steps=1,teacher_cache=targets)
        assert report["weights_changed"] and report["first_output_weight_gradient_norm"]>0
        runs[key]={k:report[k] for k in ["steps","batch_size","initialization_sha256","final_sha256","batch_plan_sha256","effective_supervised_tokens","alpha","temperature","temperature_squared_applied_once","loss_trace"]}
    for k in ("initialization_sha256","batch_plan_sha256","steps","effective_supervised_tokens"):
        assert runs["ce"][k]==runs["ce_kl"][k]
    assert c._parameter_hash(teacher)==teacher_before
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
finally:
    c.save_checkpoint,c._save_float=old_save,old_float

# Independently calculate masked KL at actual recipe temperature, and meaningful variants.
s=torch.tensor([[[0.,0.],[1.,-1.],[-0.5,0.5]]],requires_grad=True)
t=torch.tensor([[[0.2,-0.2],[2.,-2.],[0.,0.]]],requires_grad=True)
labels=torch.tensor([[-100,0,1]])
temperature=2.
p=(t.detach()/temperature).softmax(-1);q=(s/temperature).softmax(-1)
manual=(p*(p.log()-q.log())).sum(-1)[labels!=IGNORE].mean()*temperature**2
k=distillation_kl(s,t,labels,temperature)
assert torch.allclose(k,manual,atol=1e-7,rtol=0)
ce=masked_loss(s,labels)
mix=distillation_loss(s,t,labels,alpha=.5,temperature=2)
assert torch.allclose(mix,.5*ce+.5*manual,atol=1e-7,rtol=0)
mix.backward();assert t.grad is None and torch.equal(s.grad[:,0],torch.zeros_like(s.grad[:,0]))
changed=t.detach().clone();changed[:,1]=changed[:,1].flip(-1)
assert not torch.allclose(distillation_kl(s.detach(),changed,labels,2),k.detach())
try:
    distillation_kl(torch.zeros(1,3,2),torch.zeros(1,2,2),labels,2)
except ValueError: pass
else: raise AssertionError("different answer row counts must be rejected")
same=distillation_kl(t.detach(),t.detach(),labels,2)
assert abs(float(same))<1e-6
save("bounded-cpu-contract.json", {"environment":env,"fixture":"two artificial attribute questions; random untrained teacher and student; one update per matched student","runs":runs,"teacher_parameter_hash_unchanged":True,"teacher_gradients_absent":True,"checkpoint_save_interceptions":saved,"weights_retained":False,"kl":{"temperature":2,"manual":float(manual.detach()),"observed":float(k.detach()),"mixed_objective":float(mix.detach()),"gold_ce":float(ce.detach()),"mask_first_row_zero_gradient":True,"teacher_distribution_change_changes_loss":True,"misaligned_lengths_rejected":True,"identical_distributions_zero":float(same)},"registry":{"distillation":{k:experiment_spec('distillation')[k] for k in ['module','function','dependencies','assets']},"multimodal_distillation":{k:experiment_spec('multimodal_distillation')[k] for k in ['module','function','dependencies','assets']}},"scope":"method contract only; not a shortened published training run or new model score"})
print(json.dumps({"status":"all_assertions_passed","environment":env,"stored_tasks":{k:{'teacher_parameters':v['teacher_parameters'],'data_counts':v['data_counts'],'runs':v['runs']} for k,v in tasks.items()},"gsm8k":{"original_rows":len(gsm),"eligible":len(eligible),"selected":len(eligible[:6])},"original_inference":cli,"bounded_method_contract":"one update per two artificial matched students; all loss/gradient/freeze/shape assertions passed; saves intercepted"},ensure_ascii=False,indent=2))
