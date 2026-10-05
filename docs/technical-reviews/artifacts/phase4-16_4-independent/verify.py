from pathlib import Path
import ast, hashlib, json, math, platform, statistics, sys
import torch
from tiny_perceptron.model import TinyLM, ModelConfig, loss_sum
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat, pad_batch
root=Path.cwd()
proof=root/"docs/technical-reviews/artifacts/phase4-16_4-independent"
torch.set_num_threads(1)
torch.manual_seed(42)
print("ENV",json.dumps({"python":sys.version,"torch":str(torch.__version__),"torch_git":torch.version.git_version,"device":"cpu","cuda_build":str(torch.version.cuda),"cuda_available":torch.cuda.is_available(),"default_dtype":str(torch.get_default_dtype())}))
print("ORIGINAL_FENCE_BEGIN")
exec(compile((proof/"original-fence.py").read_bytes(),"original-fence.py","exec"),{})
print("ORIGINAL_FENCE_END")
# Bounded shape/storage checks. No training or model-quality evaluation.
for length in (3,6):
 for kv_heads in (4,2,1):
  model=TinyLM(ModelConfig(width=16,heads=4,kv_heads=kv_heads)).eval()
  q_shapes=[]
  h=model.blocks[0].attention.q.register_forward_hook(lambda m,i,o:q_shapes.append(list(o.shape)))
  with torch.no_grad(): cache=model(torch.arange(1,length+1).reshape(1,-1))["cache"]
  h.remove()
  shape=(1,kv_heads,length,4)
  assert all(tuple(t.shape)==shape and t.dtype==torch.float32 and not t.requires_grad for pair in cache for t in pair)
  actual=sum(t.numel()*t.element_size() for pair in cache for t in pair)
  expected=2*1*kv_heads*length*4*4
  assert actual==expected
  print("CACHE_VARIANT",json.dumps({"length":length,"kv_heads":kv_heads,"K_shape":list(cache[0][0].shape),"Q_projection_shape":q_shapes,"bytes":actual,"expected_bytes":expected,"layers":len(cache),"parameter_count":sum(t.numel() for t in model.parameters())}))
try: TinyLM(ModelConfig(width=16,heads=4,kv_heads=3))
except ValueError as e: print("INVALID_HEADS_REJECTED",str(e))
else: raise AssertionError("nondivisible KV heads accepted")
# General batch/layer storage axis formula, not a claim of new measured model scores.
model=TinyLM(ModelConfig(width=16,layers=2,heads=4,kv_heads=1))
with torch.no_grad():cache=model(torch.tensor([[1,2,3],[4,5,6]]))["cache"]
assert sum(t.numel()*t.element_size() for pair in cache for t in pair)==2*2*2*1*3*4*4
print("AXES",json.dumps({"batch":2,"layers":2,"kv_heads":1,"positions":3,"head_dim":4,"element_bytes":4,"cache_bytes":384}))
# Hand example and actual ignore/denominator behavior.
value=-math.log(0.5)
logits=torch.tensor([[0.,0.],[0.,0.],[0.,0.],[0.,0.]])
labels=torch.tensor([0,1,0,IGNORE])
loss,count=loss_sum(logits,labels)
assert int(count)==3 and abs(float(loss/count)-value)<1e-7
print("NLL_EXAMPLE",json.dumps({"negative_log_half":value,"sum":float(loss),"denominator":int(count),"mean":float(loss/count),"tolerance":1e-7}))
# Read raw measurements and raw samples from the saved, unmodified result.
j=json.loads((proof/"inputs/efficiency-result.json").read_text());tok=ByteTokenizer()
for name in ("mha","gqa"):
 v=j["results"]["models"][name]
 config=ModelConfig(**v["model"]["config"]);model=TinyLM(config)
 assert sum(x.numel() for x in model.parameters())==v["model"]["parameters"]
 with torch.no_grad():cache=model(torch.arange(25).reshape(1,-1))["cache"]
 cb=sum(t.numel()*t.element_size() for pair in cache for t in pair)
 assert cb==v["cache"]["prefill_cache_bytes"]==2*config.layers*config.kv_heads*25*(config.width//config.heads)*4
 print("RAW_MODEL",json.dumps({"name":name,"config":v["model"]["config"],"parameters":v["model"]["parameters"],"prefill_cache_bytes":cb,"raw_result_revision":j["revision"]}))
 for split in ("validation","test"):
  e=v["heldout"][split];observed=[]
  for index,s in enumerate(e["samples"]):
   generated=s["generated_ids"];has_eos=tok.eos_id in generated
   raw=generated[:generated.index(tok.eos_id)] if has_eos else generated
   exact=raw==tok.encode(s["expected"])
   assert exact==s["exact"] and has_eos==s["eos"] and len(generated)<=32
   # Rebuild only labels from actual sample messages/standard answers; no learned model evaluation.
   x,y=render_chat(s["messages"]+[{"role":"assistant","content":s["expected"]}])
   targets=int((y!=IGNORE).sum())
   assert targets==len(tok.encode(s["expected"]))+1
   observed.append({"index":index,"answer_bytes":len(tok.encode(s["expected"])),"targets_with_eos":targets,"generated_ids":generated,"exact":exact,"eos":has_eos})
  matches=sum(x["exact"] for x in observed);ended=sum(x["eos"] for x in observed);count=sum(x["targets_with_eos"] for x in observed)
  assert matches==e["matches"] and len(observed)==e["records"] and count==e["effective_tokens"] and ended/len(observed)==e["eos_rate"]
  assert e["nll_sum"]/count==e["nll"]
  print("RAW_HELDOUT",json.dumps({"name":name,"split":split,"nll_sum":e["nll_sum"],"effective_tokens":count,"recomputed_mean":e["nll_sum"]/count,"rounded_5dp":format(e["nll"],".5f"),"matches":matches,"records":len(observed),"eos_count":ended,"samples":observed}))
 tr=v["training"]
 assert tr["requested_steps"]==tr["steps"]==tr["optimizer_updates"]==100 and tr["skipped_updates"]==0 and tr["effective_tokens"]==11278
 print("RAW_TRAINING",json.dumps({"name":name,"requested_steps":tr["requested_steps"],"completed_steps":tr["steps"],"optimizer_updates":tr["optimizer_updates"],"effective_tokens":tr["effective_tokens"],"history_recorded_steps":[x["step"] for x in tr["history"]],"scope":"Read raw training aggregate only; per-step selected training examples and per-target loss values are not present, not invented or retrained."}))
 for field in ("prefill","cached_decode"):
  timing=v["cache"][field];samples=timing["samples_seconds"]
  median=statistics.median(samples);assert median==timing["median_seconds"] and len(samples)==timing["measured_calls"]==9
  print("RAW_TIMING",json.dumps({"name":name,"field":field,"samples_seconds":samples,"median_seconds":median,"median_ms":median*1000,"rounded_3dp_ms":format(median*1000,".3f"),"warmup_calls":timing["warmup_calls"],"measured_calls":len(samples),"unit":"seconds converted to milliseconds by *1000"}))
# Execute original conversion helper only on randomly initialized bounded models.
f=proof/"code/original/scripts/course_experiments/architecture.py";tree=ast.parse(f.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_copy_matching")
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(f),"exec"),ns)
source=TinyLM(ModelConfig(width=64,layers=2,heads=1));torch.manual_seed(42);target=TinyLM(ModelConfig(width=64,layers=2,heads=4,kv_heads=1))
initial_k=target.blocks[0].attention.k.weight.detach().clone()
copied=ns["_copy_matching"](source,target)
assert "blocks.0.attention.k.weight" not in copied and torch.equal(initial_k,target.blocks[0].attention.k.weight)
assert torch.equal(source.embedding.weight,target.embedding.weight)
assert copied==j["results"]["models"]["gqa"]["copied_initial_tables"]
print("CONVERSION",json.dumps({"copied_names":copied,"source_K_shape":list(source.blocks[0].attention.k.weight.shape),"target_K_shape":list(target.blocks[0].attention.k.weight.shape),"target_K_retains_random_initialization":True,"not_quality_test":True}))
print("ALL_ASSERTIONS_PASSED")
