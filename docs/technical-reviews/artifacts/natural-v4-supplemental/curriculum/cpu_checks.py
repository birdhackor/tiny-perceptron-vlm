"""Bounded independent CPU mechanism checks and recalculation of original fixed records."""
import ast, copy, hashlib, importlib.metadata, json, math, platform, re, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
RAW=ROOT/'outputs/natural-v4/factual-research/curriculum'
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.attention import manual_attention, attention_mask
from tiny_perceptron.alignment import LoRALinear, dpo_loss, distillation_kl
from tiny_perceptron.modern import MoEFFN, rope
from tiny_perceptron.quantization import pack_int4, unpack_int4, fake_quantize, QuantizedLinear
from tiny_perceptron.posttraining import ppo_clipped_objective, bandit_advantage
from tiny_perceptron.capstone import CapstoneModel, parse_action, save_capstone, load_capstone
from tempfile import TemporaryDirectory
torch.set_num_threads(1);torch.manual_seed(7301)
out={'environment':{'python':platform.python_version(),'torch':torch.__version__,'torch_git_version':torch.version.git_version,'device':'cpu','dtype':'float64 unless explicitly float32','threads':'1','seed':'7301'},'checks':{}}
def emit(k,**v):out['checks'][k]=v
def original_defs(number,names,ns=None):
 p=RAW/f'original-{number:02d}.txt'; tree=ast.parse(p.read_text()); nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert len(nodes)==len(names)
 namespace={'torch':torch,'nn':nn,'F':F,'math':math,'Dataset':torch.utils.data.Dataset,**(ns or {})}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace)
 emit('original_ast_'+str(number),names=names,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),lines={n.name:[n.lineno,n.end_lineno] for n in nodes})
 return namespace

# Transparent scalar mathematics, finite differences, and parameter update.
z=torch.tensor([1.,2.,3.],dtype=torch.float64,requires_grad=True)
loss=F.cross_entropy(z[None],torch.tensor([2]));loss.backward()
expected=math.log(math.exp(1)+math.exp(2)+math.exp(3))-3
h=1e-6;fd=(F.cross_entropy((z.detach()+torch.tensor([0,0,h]))[None],torch.tensor([2]))-F.cross_entropy((z.detach()-torch.tensor([0,0,h]))[None],torch.tensor([2])))/(2*h)
assert abs(loss.item()-expected)<1e-12 and abs(fd.item()-z.grad[2].item())<1e-8
updated=z.detach()-0.1*z.grad
assert F.cross_entropy(updated[None],torch.tensor([2]))<loss
emit('softmax_ce_gradient',probabilities=z.detach().softmax(-1).tolist(),loss=loss.item(),expected=expected,gradient=z.grad.tolist(),finite_difference=fd.item(),updated_loss=F.cross_entropy(updated[None],torch.tensor([2])).item(),tolerance='CE 1e-12; finite difference 1e-8')
x=torch.tensor([[1.,2.,3.]],dtype=torch.float64);layer=nn.Linear(3,2).double();assert torch.equal(layer(x),x@layer.weight.T+layer.bias)
ln=nn.LayerNorm(3).double();a=torch.tensor([[[1.,3.,8.],[4.,5.,9.]]],dtype=torch.float64);ref=(a-a.mean(-1,keepdim=True))/torch.sqrt(a.var(-1,unbiased=False,keepdim=True)+ln.eps)
assert torch.allclose(ln(a),ref,atol=1e-12)
emit('linear_layernorm',weight_shape=list(layer.weight.shape),input_shape=list(x.shape),linear_exact=True,layernorm_feature_axis_max_error=(ln(a)-ref).abs().max().item())
p=nn.Parameter(torch.tensor(2.));(p*p).backward();first=p.grad.item();(p*p).backward();second=p.grad.item();assert (first,second)==(4.,8.)
m=nn.Linear(2,1);m.eval();grad_enabled=m(torch.ones(1,2)).grad_fn is not None
with torch.no_grad(): no_grad=m(torch.ones(1,2)).grad_fn is None
assert grad_enabled and no_grad
emit('gradient_lifecycle',first_grad=first,without_zero_second_grad=second,eval_grad_fn_exists=grad_enabled,no_grad_disables_recording=no_grad)

# Complete actual TinyLM forward, ignored-context gradients, future isolation, cache, SDPA.
c=ModelConfig(vocab_size=11,width=8,layers=1,heads=2,max_length=16)
lm=TinyLM(c).double().eval();ids=torch.tensor([[1,2,3,4]]);labels=torch.tensor([[-100,-100,4,5]])
logits=lm(ids)['logits'];logits.retain_grad();masked_loss(logits,labels).backward()
assert logits.grad[0,:2].abs().max()==0 and lm.embedding.weight.grad[2].abs().sum()>0
other=ids.clone();other[0,-1]=7
assert torch.allclose(lm(ids)['logits'][:,:3],lm(other)['logits'][:,:3],atol=1e-12)
state=None;cached=[]
for j in range(4):
 r=lm(ids[:,j:j+1],cache=state);state=r['cache'];cached.append(r['logits'])
cacheerr=(torch.cat(cached,1)-lm(ids)['logits']).abs().max().item();assert cacheerr<1e-12
fast=TinyLM(copy.deepcopy(c)).double().eval();fast.load_state_dict(lm.state_dict());fast.blocks[0].attention.backend='sdpa'
sdpaerr=(fast(ids)['logits']-lm(ids)['logits']).abs().max().item();assert sdpaerr<1e-12
emit('tinylm_attention_mask_cache',logit_shape=list(logits.shape),ignored_logits_direct_gradient=logits.grad[0,:2].abs().max().item(),context_embedding_gradient=lm.embedding.weight.grad[2].abs().sum().item(),future_isolation=True,cache_max_error=cacheerr,sdpa_max_error=sdpaerr,tolerance='float64 absolute 1e-12; no dropout, same weights and positions')
raw_nan=F.cross_entropy(torch.randn(2,3),torch.tensor([-100,-100])).isnan().item()
try:masked_loss(torch.randn(1,2,3),torch.full((1,2),-100));guard=False
except ValueError:guard=True
assert raw_nan and guard
emit('empty_supervision',pytorch_mean_nan=raw_nan,project_rejects_empty_targets=guard)
# Unequal supervised lengths: common denominator versus averaging micro means.
base=nn.Linear(3,5).double();large=copy.deepcopy(base);micro=copy.deepcopy(base)
xx=torch.randn(2,3,dtype=torch.float64);yy=torch.tensor([1,4]);F.cross_entropy(large(xx),yy).backward()
for j in range(2): (F.cross_entropy(micro(xx[j:j+1]),yy[j:j+1],reduction='sum')/2).backward()
err=max((p.grad-q.grad).abs().max().item() for p,q in zip(large.parameters(),micro.parameters()));assert err<1e-12
emit('effective_token_accumulation',effective_targets=2,gradient_max_error=err,tolerance='float64 absolute 1e-12; shared sum denominator, one optimizer step')

# Execute exact inspected upstream functions, with synthetic inputs and no upstream imports.
up=original_defs(17,['dpo_loss']);policy=torch.tensor([[-1.,-2.],[-2.,-3.]],requires_grad=True);reference=torch.tensor([[-1.5,-2.5],[-2.,-2.]])
up_loss=up['dpo_loss'](reference,policy,torch.ones_like(policy),0.1);ours=dpo_loss(torch.tensor([-3.]),torch.tensor([-5.]),torch.tensor([-4.]),torch.tensor([-4.]),beta=0.1)
assert torch.allclose(up_loss,ours)
emit('upstream_dpo',observed=up_loss.item(),expected=-math.log(1/(1+math.exp(-0.2))),reference_fixed=True,tolerance='float32 absolute 1e-7')
up=original_defs(19,['distillation_loss']);s=torch.tensor([[1.,2.,3.],[3.,1.,2.]],requires_grad=True);t=torch.tensor([[2.,1.,3.],[1.,2.,3.]])
up_k=up['distillation_loss'](s,t,temperature=2.);pteacher=(t/2).softmax(-1);manual=(pteacher*(pteacher.log()-(s/2).log_softmax(-1))).sum(-1).mean()*4
ours=distillation_kl(s[None],t[None],torch.tensor([[1,2]]),temperature=2.)
assert torch.allclose(up_k,manual,atol=1e-6) and torch.allclose(ours,manual,atol=1e-6)
emit('distillation_direction',upstream_kl=up_k.item(),teacher_to_student_kl_T2=manual.item(),project_kl=ours.item(),effective_positions=2,temperature=2.,tolerance='float32 1e-6')
up=original_defs(11,['VLMDataset']);dummy=object.__new__(up['VLMDataset']);dummy.bos_id=[8,9];dummy.eos_id=[10];dummy.max_length=8
labels_up=dummy.generate_labels([1,2,8,9,3,4,10,0]);assert labels_up==[-100,-100,-100,-100,3,4,10,-100]
emit('upstream_assistant_labels',input_ids=[1,2,8,9,3,4,10,0],labels=labels_up,first_answer_prediction_position=3,shift_once=True)
up=original_defs(6,['Linear','MLP']);mlp=up['MLP'](SimpleNamespace(n_embd=3));xx=torch.randn(2,3);assert torch.equal(mlp(xx),mlp.c_proj(F.relu(mlp.c_fc(xx)).square()))
emit('upstream_relu2_mlp',input_shape=list(xx.shape),exact_forward_matches=True)

# PPO sign-dependent clipping: no hard projection of all probabilities.
ratios=torch.tensor([1.4,0.6,1.4,0.6]);adv=torch.tensor([1.,1.,-1.,-1.]);res=ppo_clipped_objective(ratios.log(),torch.zeros(4),adv)
assert torch.allclose(res['surrogate'],torch.tensor([1.2,.6,-1.4,-.8]))
emit('ppo_clipping',ratios=ratios.tolist(),advantages=adv.tolist(),surrogate=res['surrogate'].tolist(),expected=[1.2,.6,-1.4,-.8],tolerance='float32 1e-6')
lora=LoRALinear(nn.Linear(3,2).double(),rank=2,alpha=4).double();lora.b.data.fill_(.5);xx=torch.randn(2,3,dtype=torch.float64)
assert torch.allclose(lora(xx),F.linear(xx,lora.merged_weight(),lora.base.bias),atol=1e-12)
emit('lora',adapter_parameters=lora.a.numel()+lora.b.numel(),expected_adapter_parameters=2*(3+2),scale=lora.alpha/lora.rank,merged_equivalent=True,base_frozen=all(not p.requires_grad for p in lora.base.parameters()))
moe=MoEFFN(4,experts=3,top_k=1).double();xx=torch.randn(2,3,4,dtype=torch.float64);y,aux,chosen=moe(xx);y.square().sum().backward()
ref=torch.stack([moe.experts[e](row)*moe.router(row).softmax(-1)[e] for row,e in zip(xx.reshape(-1,4),chosen.flatten())]).reshape_as(xx)
assert torch.allclose(y,ref,atol=1e-12) and moe.router.weight.grad.abs().sum()>0
g=torch.tensor([1.,2.,3.],requires_grad=True);w=g.softmax(-1).topk(1).values;(w/w.sum()).sum().backward();assert g.grad.abs().sum()==0
emit('moe_dispatch_and_top1',shape=list(y.shape),selected_experts=chosen.tolist(),dispatch_combine_max_error=(y-ref).abs().max().item(),raw_top1_router_gradient=moe.router.weight.grad.abs().sum().item(),renormalized_top1_gradient=g.grad.tolist(),all_experts_stored=len(moe.experts))
ints=torch.arange(-8,8,dtype=torch.int8);packed=pack_int4(ints);assert torch.equal(unpack_int4(packed,ints.shape),ints)
fq=torch.tensor([.1,.3,1.],requires_grad=True);fake_quantize(fq,4).sum().backward();assert torch.equal(fq.grad,torch.ones_like(fq))
emit('quantization_packing',signed_values=ints.numel(),packed_bytes=packed.numel(),expected_bytes=8,roundtrip_exact=True,fake_quant_dtype=str(fake_quantize(fq).dtype),straight_through_gradient=fq.grad.tolist())
# Toy STFT and non-injective average summaries, not natural speech or photo accuracy.
sr=8000;n=8000;wave=torch.sin(2*math.pi*440*torch.arange(n)/sr);spec=torch.stft(wave,n_fft=400,hop_length=160,window=torch.hann_window(400),return_complex=True)
peak=spec.abs().mean(-1).argmax().item()*sr/400;assert peak==440
emit('signal_stft',sample_rate=sr,samples=n,seconds=n/sr,stft_shape=list(spec.shape),peak_hz=peak,bin_width_hz=20.,scope='synthetic stationary sine, explicit Hann window; no natural speech claim')

# Independent recalculation of current document counts and original per-item GPU records.
chapter_paths=sorted((ROOT/'course/chapters').glob('*.md'));headings=[]
for p in chapter_paths:headings.extend(re.findall(r'^## ([\w]+\.\d+) ',p.read_text(),flags=re.M))
front=[]
for p in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']:front.extend(re.findall(r'^## ([\w]+\.\d+) ',(ROOT/p).read_text(),flags=re.M))
nb=list((ROOT/'notebooks').glob('*/*.ipynb'));assert len(headings)==266 and len(nb)==266 and len(headings)+len(front)==292
emit('inventory',chapters=len([p for p in chapter_paths if p.stem.isdecimal()]),branches=len([p for p in chapter_paths if not p.stem.isdecimal()]),chapter_sections=len(headings),notebooks=len(nb),front_sections=len(front),all_numbered_sections=len(headings)+len(front))
evaluations={}
for identity,path in {'joint':'deployment/test-joint.json','dpo':'deployment/test-dpo.json','student_ce':'student/test-ce.json','student_kd':'student/test-kd.json','student_kd_int4':'student/test-kd-ptq4.json'}.items():
 p=ROOT/'docs/course-experiments/capstone-evidence'/path;d=json.loads(p.read_text());rows=d['records'];by={}
 for r in rows:
  action=r['action_trace']['eos'] and r['action_trace']['raw']==r['expected_action'];final=action and r['answer']==r['expected_final']
  assert action==r['action_correct'] and final==r['end_to_end_correct']
  a=by.setdefault(r['task'],{'count':0,'correct':0});a['count']+=1;a['correct']+=int(final)
 correct=sum(a['correct'] for a in by.values());assert correct==d['end_to_end_correct'] and len(rows)==d['count']==90
 evaluations[identity]={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'samples':len(rows),'correct':correct,'by_task':by,'protocol':d['protocol']}
assert [v['correct'] for v in evaluations.values()]==[78,78,62,61,61]
emit('fixed_gpu_record_recalculation',evaluations=evaluations,scope='Own CPU recomputation of original fixed GPU item records; no new model inference, training, or GPU replication')
for name in ['capstone_deployment','capstone_student','flash_probe','posttraining']:
 p=ROOT/f'docs/course-experiments/results/{name}.json';d=json.loads(p.read_text());r=d['results']
 emit('record_'+name,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),original_device=d.get('device','see nested runtime'),seed=d.get('seed',r.get('seed')),original_torch=d.get('torch_version','see record'),results_keys=list(r),steps={k:v.get('steps') for k,v in r.get('branches',{}).items()},effective_tokens={k:v.get('effective_tokens') for k,v in r.get('branches',{}).items()})
public=json.loads((ROOT/'docs/course-experiments/public-models.json').read_text());cap=json.loads((ROOT/'docs/course-experiments/capstone-public.json').read_text());natural=json.loads((ROOT/'docs/natural-assistant/v4/public-release.json').read_text())
assert len(public['models'])==30 and sum(f['path'].endswith('.pt') for m in public['models'] for f in m['files'])==120
assert len(cap['models'])==11 and sum(m['id'].startswith('student') for m in cap['models'])==3
assert natural['selected_variant']=='base' and natural['adapter_parameters']==0
emit('release_manifests',original_experiments=len(public['models']),original_weights=120,capstone_weights=len(cap['models']),dense_students=3,same_repo=all(m['repo']==cap['repo'] for m in public['models']),natural_variant=natural['selected_variant'],base_model=natural['base_model'],asr_model=natural['asr_model'],adapter_parameters=natural['adapter_parameters'])
with TemporaryDirectory() as directory:
 model=CapstoneModel();path=Path(directory)/'example.pt';save_capstone(path,model,stage='sft',step=0,inference_only=True);loaded,metadata=load_capstone(path)
 assert metadata['inference_only'] and metadata['step']==0 and 'optimizer' not in metadata
 emit('inference_checkpoint',format=metadata['format_version'],inference_only=metadata['inference_only'],step=metadata['step'],optimizer_present='optimizer' in metadata,parameter_description_equal=model.description()==loaded.description())
(HERE/'cpu.results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
