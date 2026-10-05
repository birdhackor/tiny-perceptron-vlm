from pathlib import Path
import math,json,hashlib,sys,platform,os,copy
import torch
from torch.nn import functional as F
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.course_experiments.modalities import _Contrastive, _vision_records
from tiny_perceptron.multimodal import scene
out=ROOT/'docs/technical-reviews/artifacts/phase4-10_10-independent'
torch.set_num_threads(1);torch.manual_seed(29);torch.set_default_device('cpu')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result={'environment':{'python':sys.version,'torch':torch.__version__,'torch_git_version':torch.version.git_version,'torch_cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu','threads':str(torch.get_num_threads()),'platform':platform.platform()},'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-10_10-independent/execution/verify_numeric_and_raw.py','source_sha256':json.loads((out/'inputs/extraction.json').read_bytes())['source_sha256']}
assert torch.version.cuda is None and not torch.cuda.is_available()
S=torch.tensor([[2.,0.],[0.,2.]],dtype=torch.float64,requires_grad=True);T=.5;targets=torch.arange(2)
old=S.detach().clone();loss=F.cross_entropy(S/T,targets);loss.backward();p=(S/T).softmax(1).detach();analytic=(p-F.one_hot(targets,2))/(2*T)
expected=math.log1p(math.exp(-4));assert math.isclose(loss.item(),expected,abs_tol=1e-14);assert torch.allclose(S.grad,analytic,atol=1e-14,rtol=0)
assert torch.equal(S.detach(),old)
finite=torch.zeros_like(S);eps=1e-6
for i in range(2):
 for j in range(2):
  plus=old.clone();minus=old.clone();plus[i,j]+=eps;minus[i,j]-=eps
  finite[i,j]=(F.cross_entropy(plus/T,targets)-F.cross_entropy(minus/T,targets))/(2*eps)
assert torch.allclose(S.grad,finite,atol=2e-10,rtol=0)
proposed=old-.1*S.grad
result['original_math']={'score_shape':list(S.shape),'axes':{'0':'image query row N=2','1':'text candidate class C=2'},'targets':targets.tolist(),'target_dtype':str(targets.dtype),'temperature':T,'probabilities':p.tolist(),'loss_mean_nats':loss.item(),'mean_denominator':2,'gradient_wrt_unscaled_scores':S.grad.tolist(),'analytic_gradient_formula':'(softmax(scores/T)-one_hot(target))/(N*T)','analytic_gradient':analytic.tolist(),'finite_difference_gradient':finite.tolist(),'finite_difference_max_abs_error':float((S.grad-finite).abs().max()),'backward_scores_unchanged':True,'hypothetical_score_update_lr':.1,'hypothetical_scores':proposed.tolist(),'loss_unit':'nat per image query; temperature and scores dimensionless'}
reversed_scores=old.clone().requires_grad_();reversed_targets=torch.tensor([1,0]);reversed_loss=F.cross_entropy(reversed_scores/T,reversed_targets);reversed_loss.backward();assert math.isclose(reversed_loss.item(),4+expected,abs_tol=1e-14);assert bool((reversed_scores.grad.diag()>0).all());assert reversed_scores.grad[0,1]<0 and reversed_scores.grad[1,0]<0
unscaled=old.clone().requires_grad_();loss_t1=F.cross_entropy(unscaled,targets);loss_t1.backward()
result['variations']={'reversed_targets':{'targets':reversed_targets.tolist(),'loss_mean_nats':reversed_loss.item(),'gradient':reversed_scores.grad.tolist(),'off_diagonal_negative':True},'temperature_1':{'loss_mean_nats':loss_t1.item(),'gradient':unscaled.grad.tolist(),'ordering_unchanged':bool(torch.equal(old.argmax(-1),(old/T).argmax(-1)))}}
# Forward/backward only; do not call optimizer, _fit, evaluate, checkpoint load/save.
model=_Contrastive();params={n:p.detach().clone() for n,p in model.named_parameters()}
labels=[f'{c} {s}' for c in ['red','green','blue'] for s in ['circle','square']]
images=torch.stack([scene(*label.split()) for label in labels]);scores=model.scores(images,labels);score_snapshot=scores.detach().clone();scores.retain_grad();encoder_loss=F.cross_entropy(scores,torch.arange(6));encoder_loss.backward()
vision_norm=float(model.vision.projection.weight.grad.norm());text_norm=float(model.text.weight.grad.norm());assert vision_norm>0 and text_norm>0
assert all(torch.equal(params[n],p.detach()) for n,p in model.named_parameters());assert torch.equal(score_snapshot,scores.detach())
result['repository_forward_backward']={'batch_labels':labels,'image_shape':list(images.shape),'score_shape':list(scores.shape),'temperature':.1,'vision_projection_grad_norm':vision_norm,'text_embedding_grad_norm':text_norm,'all_model_parameters_unchanged':True,'scores_unchanged':True,'no_optimizer_or_training':True}
with torch.no_grad(): duplicate_scores=model.scores(torch.stack([scene('red','circle'),scene('red','circle')]),['red circle','red circle'])
duplicate_loss=F.cross_entropy(duplicate_scores,torch.arange(2));assert torch.allclose(duplicate_scores[:,0],duplicate_scores[:,1]);assert math.isclose(duplicate_loss.item(),math.log(2),abs_tol=1e-7)
result['duplicate_positive_counterexample']={'labels':['red circle','red circle'],'scores':duplicate_scores.tolist(),'probabilities':duplicate_scores.softmax(-1).tolist(),'diagonal_only_loss_nats':duplicate_loss.item(),'expected_log2':math.log(2),'both_columns_semantically_valid':True,'implication':'single-index identity labeling counts another identical valid caption as a negative; multi-positive labels/objective or distinct descriptions are needed'}
rawpath=ROOT/'docs/course-experiments/results/contrastive.json';raw=rawpath.read_bytes();data=json.loads(raw);pointers=[]
def get(pointer):
 assert not any(k in ['notes','review'] or k.endswith('_scope_correction') for k in pointer[1:].split('/'))
 x=data
 for k in pointer[1:].split('/'):x=x[int(k)] if isinstance(x,list) else x[k]
 pointers.append(pointer);return x
selected={}
for key in ['schema_version','experiment_id','revision','device','seed','torch_version','python_version','step_scale','code_sha256']:
 selected['/'+key]=get('/'+key)
for key in ['image_size','patch_size','width','temperature','text_encoder']:selected['/results/config/'+key]=get('/results/config/'+key)
selected['/results/data/seed']=get('/results/data/seed');selected['/results/data/split_policy']=get('/results/data/split_policy')
checks={};fresh_splits=_vision_records()
for split in ['train','validation','test']:
 base='/results/data/splits/'+split
 count=get(base+'/count');sha=get(base+'/sha256');records=get(base+'/records')
 selected[base+'/count']=count;selected[base+'/sha256']=sha;selected[base+'/records']=records
 computed=hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True).encode()).hexdigest();assert count==len(records);assert sha==computed;assert records==fresh_splits[split]
 checks[split]={'declared_count':count,'observed_count':len(records),'hash_match':True,'records_match_original_helper':True,'offsets':sorted({r['offset'] for r in records}),'unique_answers':len({r['answer'] for r in records})}
sets={s:{r['family'] for r in fresh_splits[s]} for s in fresh_splits};assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
variants={}
for variant in ['one_way','two_way']:
 base='/results/variants/'+variant;training={}
 for key in ['parameters','trainable_parameters','initial_loss','final_loss','history','steps','effective_targets','effective_tokens','weights_changed','nonzero_gradient_seen','cpu_smoke']:
  ptr=base+'/training/'+key;training[key]=get(ptr);selected[ptr]=training[key]
 assert len(training['history'])==training['steps'];assert sum(r['effective_targets'] for r in training['history'])==training['effective_targets'];assert [r['step'] for r in training['history']]==list(range(1,training['steps']+1))
 measurements={}
 for split in ['before','validation','test']:
  t={}
  for key in ['image_queries','text_queries','image_to_text_accuracy','text_to_image_accuracy','image_samples','text_samples']:
   ptr=base+'/'+split+'/'+key;t[key]=get(ptr);selected[ptr]=t[key]
  assert t['image_queries']==len(t['image_samples']);assert t['text_queries']==len(t['text_samples'])
  for mode,raws in [('image',t['image_samples']),('text',t['text_samples'])]:
   correct=sum(x['correct'] for x in raws);denominator=t[mode+'_queries'];field='image_to_text_accuracy' if mode=='image' else 'text_to_image_accuracy';assert math.isclose(correct/denominator,t[field],abs_tol=1e-14)
   for row in raws:assert row['correct']==(row['predicted']==row['target'] if mode=='image' else row['retrieved']==row['target'])
  measurements[split]={'image_correct':sum(x['correct'] for x in t['image_samples']),'image_denominator':t['image_queries'],'image_accuracy':t['image_to_text_accuracy'],'text_correct':sum(x['correct'] for x in t['text_samples']),'text_denominator':t['text_queries'],'text_accuracy':t['text_to_image_accuracy']}
 variants[variant]={'steps':training['steps'],'history_length':len(training['history']),'effective_targets':training['effective_targets'],'initial_loss':training['initial_loss'],'final_loss':training['final_loss'],'measurements':measurements}
result['historical_raw_audit']={'path':str(rawpath.relative_to(ROOT)),'complete_file_sha256':hashlib.sha256(raw).hexdigest(),'actually_read_pointers':pointers,'split_checks':checks,'disjoint_families':True,'variants':variants,'execution_boundary':'JSON arithmetic and original data helper only; no existing model evaluation, training, model/data download, GPU or checkpoint load/save','provenance_code_hash_comparison':{p:{'historical_sha256':h,'current_sha256':digest(ROOT/p),'match':h==digest(ROOT/p)} for p,h in selected['/code_sha256'].items() if (ROOT/p).is_file()}}
(out/'inputs/contrastive-original.json').write_bytes(raw)
(out/'inputs/contrastive-selected-pointers.json').write_text(json.dumps({'complete_file_sha256':hashlib.sha256(raw).hexdigest(),'pointers':pointers,'values':selected},ensure_ascii=False,indent=2)+'\n')
(out/'execution/verification-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
