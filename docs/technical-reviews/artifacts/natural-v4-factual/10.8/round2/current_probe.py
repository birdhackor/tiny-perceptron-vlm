from pathlib import Path
import hashlib,json,re,sys,time
import torch
from tiny_perceptron.data import ByteTokenizer

start=time.perf_counter();torch.set_num_threads(1)
r=Path("docs/technical-reviews/artifacts/natural-v4-factual/10.8/round2")
a=r.parent
current=(r/"source-current.raw.md").read_text()
old=(a/"source-first.raw.md").read_text()
current_code=re.findall(r"```python\n(.*?)```",current,re.S)[0]
old_code=re.findall(r"```python\n(.*?)```",old,re.S)[0]
assert current_code==old_code
scope={}
exec(compile(current_code,"course/chapters/10.md#10.8-current-snippet","exec"),scope)
lm,wrapper,ids,original,wrapped=[scope[k] for k in ("lm","multimodal","ids","a","b")]
assert original.shape==(1,3,264) and torch.equal(original,wrapped)
seen=[]
def observe(module,args,kwargs):
    seen.append({"shape":list(args[0].shape),"ids":args[0].tolist(),"kwargs":list(kwargs)})
h=lm.register_forward_pre_hook(observe,with_kwargs=True)
wrapper(ids)
h.remove()
assert seen==[{"shape":[1,3],"ids":[[1,20,30]],"kwargs":[]}]
exercise_ids=torch.tensor([1,40,50]);e1=lm(exercise_ids[None])["logits"];e2=wrapper(exercise_ids)["logits"]
assert e1.shape==(1,3,264) and torch.equal(e1,e2)
errors=[]
for marker in (ByteTokenizer.image_id,ByteTokenizer.audio_id):
    try:wrapper(torch.tensor([1,marker,4]))
    except ValueError as exc:
        errors.append({"marker":marker,"type":type(exc).__name__,"message":str(exc)})
        assert str(exc)=="placeholder 缺少配對圖片／聲音"
    else:raise AssertionError("missing modality was not rejected")
# Execute only the CPU attention-contract example from the newly read link.
linked_code=re.findall(r"```python\n(.*?)```",(r/"prerequisite-16.8.raw.md").read_text(),re.S)[0]
attention_scope={}
exec(compile(linked_code,"course/chapters/16.md#16.8-cpu-snippet","exec"),attention_scope)
manual,optimized,allowed=[attention_scope[k] for k in ("manual","optimized","allowed")]
expected_allowed=torch.arange(4)[None,:]<=torch.arange(4)[:,None]
assert torch.equal(allowed[0,0],expected_allowed)
# The original precision result is reused, not rerun or overwritten.
precision_path=a/"precision-results.json";precision=json.loads(precision_path.read_text())
assert precision["manual_vs_sdpa"]["number_different"]==369
record={"environment":{"python":sys.version.split()[0],"torch":torch.__version__,"torch_git_version":torch.version.git_version,"device":"cpu","dtype":"torch.float32","threads":str(torch.get_num_threads())},"current_section_sha256":hashlib.sha256(current.encode()).hexdigest(),"current_code_equals_first_code":current_code==old_code,"current_code_sha256":hashlib.sha256(current_code.encode()).hexdigest(),"original":{"ids":ids.tolist(),"shape":list(original.shape),"finite":bool(torch.isfinite(original).all()),"logits_compared":original.numel(),"equal":bool(torch.equal(original,wrapped)),"max_absolute_difference":float((original-wrapped).abs().max().detach())},"route":{"same_language_object":wrapper.language is lm,"all_modules_eval":all(not m.training for m in wrapper.modules()),"forward_call":seen,"default_positions":[0,1,2]},"exercise":{"ids":exercise_ids.tolist(),"shape":list(e1.shape),"logits_compared":e1.numel(),"equal":bool(torch.equal(e1,e2)),"max_absolute_difference":float((e1-e2).abs().max().detach())},"missing_modality_errors":errors,"linked_16_8_cpu":{"shape":list(optimized.shape),"finite":bool(torch.isfinite(optimized).all()),"max_absolute_difference":float((manual-optimized).abs().max()),"torch_equal":bool(torch.equal(manual,optimized)),"torch_allclose_atol_1e_6_default_rtol_1e_5":bool(torch.allclose(manual,optimized,atol=1e-6)),"causal_mask":allowed[0,0].tolist(),"compared_output_elements":optimized.numel()},"original_precision_evidence_reused":{"path":str(precision_path),"sha256":hashlib.sha256(precision_path.read_bytes()).hexdigest(),"original_result":precision["manual_vs_sdpa"],"rerun":False},"denominators":{"seed":0,"wrapper_sequences":2,"positions_each":3,"vocabulary":264,"wrapper_logits_compared_each":792,"training_updates":0,"split":"none; initialized model and contract probes","linked_attention_batch":1,"linked_attention_heads":1,"linked_attention_positions":4,"linked_attention_features":3,"linked_attention_outputs":12},"elapsed_seconds_after_imports":time.perf_counter()-start,"timing_scope":"Current source snippets and short contract checks; no benchmark/GPU/training."}
(r/"current-probe.results.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(record,ensure_ascii=False,indent=2))
