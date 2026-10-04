import copy, hashlib, json, platform, sys, time
from pathlib import Path
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.data import ByteTokenizer

start=time.perf_counter()
torch.set_num_threads(1)
# Execute the lesson's actual statements with no hook or optimizer.
torch.manual_seed(0)
lm = TinyLM(ModelConfig(width=8)).eval()
multimodal = MultiModalLM(lm).eval()
ids = torch.tensor([1, 20, 30])
a = lm(ids[None])["logits"]
b = multimodal(ids)["logits"]
print("分數形狀", tuple(a.shape))
print("純文字入口相同", torch.equal(a, b))
assert tuple(a.shape)==(1,3,264) and torch.equal(a,b)
original={"ids":ids.tolist(),"ids_dtype":str(ids.dtype),"shape":list(a.shape),"output_dtype":str(a.dtype),"finite":bool(torch.isfinite(a).all()),"compared_logits":a.numel(),"equal":bool(torch.equal(a,b)),"max_absolute_difference":float((a-b).abs().max().detach())}
state_before={name:value.detach().clone() for name,value in lm.state_dict().items()}
# The exercise is another sequence, not a trained-language quality test.
exercise_ids=torch.tensor([1,40,50])
e1=lm(exercise_ids[None])["logits"]
e2=multimodal(exercise_ids)["logits"]
assert e1.shape==(1,3,264) and torch.equal(e1,e2)
exercise={"ids":exercise_ids.tolist(),"shape":list(e1.shape),"compared_logits":e1.numel(),"equal":bool(torch.equal(e1,e2)),"max_absolute_difference":float((e1-e2).abs().max().detach())}
# Observe the actual forwarded call, including default position creation.
seen=[]
def capture(module,args,kwargs):
    seen.append({"shape":list(args[0].shape),"ids":args[0].tolist(),"kwargs":list(kwargs)})
hook=lm.register_forward_pre_hook(capture,with_kwargs=True)
multimodal(ids)
hook.remove()
identity={"same_language_object":multimodal.language is lm,"same_parameter_objects":all(dict(multimodal.language.named_parameters())[name] is value for name,value in lm.named_parameters()),"all_modules_eval":all(not m.training for m in multimodal.modules()),"forwarded":seen,"positions_by_forward_default":torch.arange(ids.numel()).tolist(),"state_unchanged":all(torch.equal(state_before[name],value) for name,value in lm.state_dict().items())}
assert identity["same_language_object"] and identity["same_parameter_objects"] and identity["all_modules_eval"] and identity["state_unchanged"]
assert seen==[{"shape":[1,3],"ids":[[1,20,30]],"kwargs":[]}]
errors=[]
for marker in (ByteTokenizer.image_id,ByteTokenizer.audio_id):
    bad_ids=torch.tensor([1,marker,4])
    try: multimodal(bad_ids)
    except ValueError as exc:
        errors.append({"ids":bad_ids.tolist(),"type":type(exc).__name__,"message":str(exc)})
        assert str(exc)=="placeholder 缺少配對圖片／聲音"
    else: raise AssertionError("Missing-modality input unexpectedly accepted")
# Fixed seed reproduces this model initialization in this same CPU runtime.
torch.manual_seed(0)
repeat=TinyLM(ModelConfig(width=8)).eval()
repeat_scores=repeat(ids[None])["logits"]
assert torch.equal(a,repeat_scores)
seed_check={"seed":0,"reinitialized_state_equal":all(torch.equal(state_before[name],value) for name,value in repeat.state_dict().items()),"logits_equal_same_runtime":bool(torch.equal(a,repeat_scores))}
# Show the declared limitation: changed scores can preserve selected answers.
changed=copy.deepcopy(lm)
with torch.no_grad(): changed.output.weight[:,0].add_(0.1)
changed_scores=changed(ids[None])["logits"]
limitation={"intervention":"Add 0.1 to column 0 of every output weight row in a deepcopy, no training", "logits_equal":bool(torch.equal(a,changed_scores)),"same_argmax_at_all_positions":bool(torch.equal(a.argmax(-1),changed_scores.argmax(-1))),"max_absolute_difference":float((a-changed_scores).abs().max().detach()),"selected_ids_before":a.argmax(-1).tolist(),"selected_ids_after":changed_scores.argmax(-1).tolist()}
assert not limitation["logits_equal"] and limitation["same_argmax_at_all_positions"]
# eval only changes affected modules; demonstrate one such module directly.
dropout=torch.nn.Dropout(p=0.5)
ones=torch.ones(32)
training_dropout=dropout(ones)
eval_dropout=dropout.eval()(ones)
dropout_check={"training_output_differs_from_input":not bool(torch.equal(training_dropout,ones)),"eval_returns_input_values":bool(torch.equal(eval_dropout,ones)),"TinyLM_has_dropout":any(isinstance(m,torch.nn.Dropout) for m in lm.modules())}
assert dropout_check["eval_returns_input_values"]
record={"environment":{"python":sys.version.split()[0],"torch":torch.__version__,"torch_git_version":torch.version.git_version,"device":"cpu","dtype":"torch.float32","threads":str(torch.get_num_threads()),"platform":platform.platform()},"original":original,"exercise":exercise,"identity_and_forwarding":identity,"missing_modality_errors":errors,"seed_check":seed_check,"changed_weights_limitation":limitation,"eval_api_scope":dropout_check,"denominators":{"seed":0,"sequences":2,"positions_per_sequence":3,"vocabulary_candidates_per_position":264,"logits_compared_per_sequence":792,"training_updates":0,"train_split":"none; initialized model","heldout_split":"none; wrapper smoke check"},"elapsed_seconds":time.perf_counter()-start,"timing_scope":"Entire probe after imports, including model construction, CPU checks and small demonstrations; not a benchmark."}
Path("docs/technical-reviews/artifacts/natural-v4-factual/10.8/probe-results.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(record,ensure_ascii=False,indent=2))
