from pathlib import Path
import copy,hashlib,json,platform,statistics,sys
ROOT=Path(__file__).resolve().parents[5];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
import tokenizers
from tiny_perceptron.multimodal import log_mel
torch.set_num_threads(1);torch.manual_seed(7302)
out={'environment':{'python':platform.python_version(),'torch':torch.__version__,'tokenizers':tokenizers.__version__,'device':'CPU','threads':'1','seed':'7302'}}
base=nn.Linear(3,5).double();large=copy.deepcopy(base);micro=copy.deepcopy(base);wrong=copy.deepcopy(base)
x=torch.randn(3,3,dtype=torch.float64);y=torch.tensor([1,4,2]);F.cross_entropy(large(x),y).backward()
for lo,hi in [(0,1),(1,3)]:
 (F.cross_entropy(micro(x[lo:hi]),y[lo:hi],reduction='sum')/3).backward()
 (F.cross_entropy(wrong(x[lo:hi]),y[lo:hi])/2).backward()
err=max((p.grad-q.grad).abs().max().item() for p,q in zip(large.parameters(),micro.parameters()))
bad=max((p.grad-q.grad).abs().max().item() for p,q in zip(large.parameters(),wrong.parameters()));assert err<1e-12 and bad>1e-4
out['unequal_accumulation']={'micro_targets':[1,2],'effective_targets':3,'common_denominator_max_error':err,'mean_of_means_gradient_error':bad,'tolerance':'absolute 1e-12 float64; same samples/weights, no per-micro optimizer step'}
t=Tokenizer(models.BPE(unk_token='[UNK]'));t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);t.decoder=decoders.ByteLevel()
t.train_from_iterator(['cats cats 中文 中文'],trainers.BpeTrainer(vocab_size=280,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['[UNK]']))
sent='繁體🐱 café\n未見';e=t.encode(sent);restored=t.decode(e.ids);assert restored==sent
single=[t.decode([i]) for i in e.ids];assert any('�' in s for s in single)
out['bytelevel']={'input':sent,'tokens':e.tokens,'ids':e.ids,'decoded':restored,'separate_token_decodes':single,'normalizer':'None','alphabet':256}
out['sources']={}
def load(path):
 p=ROOT/path;out['sources'][path]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
stages={};previous=None
for n in ['capstone_pretrain','capstone_sft','capstone_joint','capstone_preference']:
 d=load('docs/course-experiments/results/'+n+'.json');r=d['results'];assert r['steps']==r['requested_steps'] and r['schedule_completed'] and r['parent_checkpoint_sha256']==previous
 stages[n]={k:r[k] for k in ['steps','effective_tokens','parent_checkpoint_sha256','inference_export']};stages[n]['runtime']={k:d[k] for k in ['device','seed','torch_version','python_version','gpu']};previous=r['inference_export']['sha256']
out['stage_chain']=stages
flash=load('docs/course-experiments/results/flash_probe.json');r=flash['results'];audit={}
for name,route in r['routes'].items():
 entry={'correctness':{},'measurements':{}}
 for key,c in [('output',route['correctness']['output']),*route['correctness']['gradients'].items()]:
  assert c['finite'] and c['max_absolute_error']<=c['atol'];entry['correctness'][key]={k:c[k] for k in ['max_absolute_error','atol','rtol']}
 for phase,ways in route['measurements'].items():
  entry['measurements'][phase]={}
  for way,m in ways.items():
   median=statistics.median(m['samples_seconds']);additional=m['peak_allocated_bytes']-m['allocated_before_bytes'];assert median==m['median_seconds'] and additional==m['additional_peak_allocated_bytes'] and len(m['samples_seconds'])==m['measured_calls']==9
   entry['measurements'][phase][way]={'recomputed_median_seconds':median,'recomputed_additional_peak_allocated_bytes':additional,'warmup_calls':m['warmup_calls'],'measured_calls':m['measured_calls']}
  profile=route['profiles'][phase];assert any('flash_fwd_kernel' in n for n in profile['cuda_kernel_names'])
  if phase=='forward_backward':assert any('flash_bwd_' in n for n in profile['cuda_kernel_names'])
 audit[name]=entry
out['flash_fixed_record_audit']={'configuration':r['configuration'],'routes':audit,'scope':'Own CPU arithmetic/profiler-name audit of fixed original NVIDIA L4 records; no GPU execution, no task quality claim'}
selection=load('docs/natural-assistant/v4/selection.json');release=load('docs/natural-assistant/v4/public-release.json');protocol=load('docs/natural-assistant/v4/validation-protocol-lower-lr.json')
result_path='outputs/natural-v4/modal-runs/validation-37219466611/natural-natural-v4-validation-37219466611-1/review/result.json';d=load(result_path)
assert out['sources'][result_path]==selection['validation_result_sha256']
assert selection['selected_variant']==release['selected_variant']=='base' and release['adapter_parameters']==0
assert selection['base_model']['model']==release['base_model']['repo'] and selection['base_model']['revision']==release['base_model']['revision'] and d['model_revision']==release['base_model']['revision']
out['natural_release_provenance']={'selected_variant':selection['selected_variant'],'adapter_parameters':release['adapter_parameters'],'source_result_sha_equal':True,'base_model':release['base_model'],'asr_model':release['asr_model'],'original_validation_runtime':{k:d[k] for k in ['device','dtype','gpu_name','versions']},'declared_denominators':protocol['validation_denominators'],'limit':'Selection record/public manifest/original run fingerprint audit, not a new semantic grading or inference run'}
scores=load('outputs/natural-v4/validation-review/gha-37219466611-1/scored/scores.json')
assert out['sources']['outputs/natural-v4/validation-review/gha-37219466611-1/scored/scores.json']==selection['validation_scoring_sha256']
nums={}
for name,v in scores['variants'].items():
 n=sum(v['correct_counts'][g]*w for g,w in scores['integer_weights'].items());assert n==v['primary_numerator'];nums[name]=n
assert nums['base']>max(nums[n] for n in nums if n!='base')
assert all(not v['eligible'] for n,v in scores['variants'].items() if n!='base')
out['natural_selection_arithmetic']={'primary_numerators':nums,'primary_denominator':75600,'denominators':scores['denominators'],'selected_variant':scores['selected_variant'],'limit':'Recompute frozen counts and declared eligibility, accept original semantic grades as observations; no regrading or new generation'}
(HERE/'extra.results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
