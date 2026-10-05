"""Independent CPU audit of original placeholders and historical raw result aggregates.
Does not train, download data/models, or save neural weights. Original functions
are compiled from the hash-verified historical snapshots, preserving their AST.
"""
import ast, contextlib, hashlib, io, json, math, os, platform, random, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
A=Path('/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-7_15-independent')
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
provenance=[]
def save(name,x):
 p=A/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load_nodes(path,names,namespace):
 raw=path.read_bytes();tree=ast.parse(raw);selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
 assert {node.name for node in selected}==set(names)
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
 for node in selected:provenance.append({'path':str(path.relative_to(A)),'sha256':hashlib.sha256(raw).hexdigest(),'name':node.name,'lines':[node.lineno,node.end_lineno]})
ns={'torch':torch,'F':F,'json':json,'random':random,'hashlib':hashlib,'IGNORE':-100,'SPECIALS':('<pad>','<bos>','<eos>','<user>','<assistant>','<image>','<audio>','<system>')}
load_nodes(A/'historical/tiny_perceptron/data.py',['ByteTokenizer','shifted','render_chat','pad_batch'],ns)
load_nodes(A/'historical/tiny_perceptron/model.py',['generate','loss_sum','masked_loss'],ns)
load_nodes(A/'historical/scripts/course_experiments/common.py',['split_records','records_sha256','text_examples','_nll','evaluate_lm'],ns)
load_nodes(A/'historical/scripts/course_experiments/text.py',['arithmetic_records'],ns)
load_nodes(A/'historical/scripts/prepare_data.py',['conversation','generate_records'],ns)
tok=ns['ByteTokenizer']()
# Original fence, then the exact meaningful extension requested by the lesson.
namespace={};stream=io.StringIO()
with contextlib.redirect_stdout(stream):exec(compile((A/'original/fence-1.py').read_bytes(),'original/fence-1.py','exec'),namespace)
assert stream.getvalue()==(A/'original/stdout.txt').read_text()
table_before=json.dumps(namespace['measurements'],ensure_ascii=False,sort_keys=True)
b_before=list(namespace['b_questions']);a_before=list(namespace['a_questions'])
namespace['a_questions'].append('綠色物體是圓。請接續：綠色物體是')
assert len(namespace['a_questions'])==3 and namespace['a_questions'][:2]==a_before
assert namespace['b_questions']==b_before
assert all(v is None for row in namespace['measurements'].values() for v in row.values())
assert json.dumps(namespace['measurements'],ensure_ascii=False,sort_keys=True)==table_before
placeholder={'original_stdout_exact_match':True,'original_lengths':{'A':2,'B':2},'extension_lengths':{'A':3,'B':2},'all_four_measurements_still_None':True,'table_unchanged':True,'meaning':'Only the held-out question list changed; no model, gradients, optimizer or scores exist in the original fence.'}
# Historical generator/splitter reconstruction and byte-level manifest validation.
r=json.loads((A/'inputs/docs/course-experiments/results/sft_ablation.json').read_text());rr=r['results']
sft=json.loads((A/'inputs/docs/course-experiments/results/sft.json').read_text())
parts={'attributes':ns['split_records'](ns['generate_records']('attributes-sft'),seed=r['seed']),'arithmetic':ns['split_records'](ns['arithmetic_records'](),seed=r['seed'])}
splits={}
for task,part in parts.items():
 families={s:{x['family'] for x in rows} for s,rows in part.items()}
 assert all(not(families[a]&families[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
 splits[task]={}
 for split,rows in part.items():
  original_jsonl=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows).encode()
  h=hashlib.sha256(original_jsonl).hexdigest();manifest=rr['data'][task][split]
  assert h==manifest['sha256'] and len(rows)==manifest['records'] and len(families[split])==manifest['families']
  p=A/'reconstructed-inputs'/task/(split+'.jsonl');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(original_jsonl)
  examples=ns['text_examples'](rows,mode='sft',max_length=128)
  count=sum(int((y != -100).sum()) for _,y in examples)
  splits[task][split]={'records':len(rows),'families':len(families[split]),'sha256':h,'matches_original_manifest':True,'supervised_answer_plus_EOS_tokens':count,'families_list':sorted(families[split])}
assert len(ns['arithmetic_records']())==64
pair_rows=[x for x in parts['arithmetic']['train'] if x['family']=='2+3']
assert {x['messages'][0]['content'] for x in pair_rows}=={'2+3=?','3+2=?'}
assert rr['before']['A_attributes']['test']==sft['results']['after']['test']
# Recompute counts, text decoding, EOS and NLL from original stored raw IDs/sums.
summaries={}
for phase,root in [('before',rr['before']),('b-only',rr['runs']['b-only'])]:
 summaries[phase]={}
 for task,key in [('attributes','A_attributes'),('arithmetic','B_arithmetic')]:
  summaries[phase][task]={}
  for split in ('validation','test'):
   x=root[key][split];rows=parts[task][split];samples=x['samples'];assert len(samples)==len(rows)
   audit=[];matches=ended=0
   for row,sample in zip(rows,samples):
    assert sample['messages']==row['messages'][:-1] and sample['expected']==row['messages'][-1]['content']
    ids=sample['generated_ids'];raw=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
    exact=raw==tok.encode(sample['expected']);eos=tok.eos_id in ids
    assert sample['generated']==tok.decode(raw) and sample['exact']==exact and sample['eos']==eos
    assert len(ids)<=32
    matches+=int(exact);ended+=int(eos)
    audit.append({'prompt':sample['messages'][0]['content'],'expected':sample['expected'],'generated':tok.decode(raw),'generated_ids':ids,'raw_ids_before_EOS':raw,'exact_recomputed':exact,'eos_recomputed':eos})
   tokens=splits[task][split]['supervised_answer_plus_EOS_tokens']
   assert tokens==x['effective_tokens'] and len(rows)==x['records']==x['examples']
   assert matches==x['matches'] and matches/len(rows)==x['exact_match'] and ended/len(rows)==x['eos_rate']
   mean=x['nll_sum']/tokens
   assert math.isclose(mean,x['nll'],abs_tol=1e-12,rel_tol=0)
   summaries[phase][task][split]={'matches':matches,'records':len(rows),'families':splits[task][split]['families'],'eos_count':ended,'answer_plus_EOS_tokens':tokens,'nll_sum':x['nll_sum'],'nll_recomputed':mean,'nll_unit':'natural-log negative log likelihood / supervised token, includes EOS; teacher forced, not accuracy','raw_sample_audit':audit}
assert summaries['before']['attributes']['test']['matches']==5 and summaries['b-only']['attributes']['test']['matches']==0
assert summaries['before']['arithmetic']['test']['matches']==summaries['b-only']['arithmetic']['test']['matches']==0
for phase,printed in [('before',7.05820),('b-only',4.57756)]:assert abs(summaries[phase]['arithmetic']['test']['nll_recomputed']-printed)<=0.000005
assert summaries['b-only']['attributes']['test']['raw_sample_audit'][0]['generated_ids']==[65,2]
# 500 means 500 updates; independently reconstruct only the recorded sampler token budget.
training=rr['runs']['b-only']['training'];assert training['steps']==500 and training['records']==49
assert rr['runs']['b-only']['source_records']=={'A':0,'B':49}
assert ns['records_sha256'](parts['arithmetic']['train'])==training['records_sha256']
examples=ns['text_examples'](parts['arithmetic']['train'],mode='sft',max_length=128)
sampler=random.Random(r['seed']);count=0
for _ in range(500):
 for _,y in sampler.choices(examples,k=16):count+=int((y != -100).sum())
assert count==training['effective_tokens']==18453
assert training['final_loss']<training['initial_loss']
# Run the exact historical evaluator/generator on a scripted CPU finite-output stub.
# This checks raw-ID vs decoder vs EOS contracts, not learned capability.
class Scripted(nn.Module):
 def __init__(self,invalid=False):
  super().__init__();self.anchor=nn.Parameter(torch.zeros(()));self.config=SimpleNamespace(max_length=128);self.invalid=invalid
 def forward(self,ids,valid=None,cache=None):
  logits=torch.full((*ids.shape,264),-5.0)+self.anchor*0
  chosen=torch.full_like(ids,65)
  if self.invalid:
   chosen=torch.where(ids==tok.assistant_id,torch.full_like(ids,tok.user_id),chosen)
  chosen=torch.where(ids==65,torch.full_like(ids,tok.eos_id),chosen)
  logits.scatter_(-1,chosen[...,None],5.0)
  return {'logits':logits,'cache':None}
records=[{'messages':[{'role':'user','content':'probe'},{'role':'assistant','content':'9'}]}]
normal=ns['evaluate_lm'](Scripted(),records,mode='sft',tokens=32)
invalid=ns['evaluate_lm'](Scripted(invalid=True),records,mode='sft',tokens=32)
no_eos=ns['evaluate_lm'](Scripted(),records,mode='sft',tokens=1)
assert normal['samples'][0]['generated_ids']==[65,2] and normal['matches']==1 and normal['eos_rate']==1
assert invalid['samples'][0]['generated_ids']==[3,65,2] and invalid['samples'][0]['generated']=='9' and invalid['matches']==0
assert no_eos['matches']==1 and no_eos['eos_rate']==0
contract={'normal':normal,'invalid_special_hidden_by_decoder':invalid,'one_token_limit_content_correct_without_EOS':no_eos,'scope':'Scripted logits only; establishes raw-ID exact_match and EOS separation, does not train or estimate model capability.'}
save('environment.json',{'python':platform.python_version(),'python_executable':sys.executable,'torch':torch.__version__,'torch_git_version':torch.version.git_version,'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'threads':'1','original_experiment':{k:r[k] for k in ['revision','seed','device','torch_version','python_version','step_scale','evidence_status']}})
save('verification.json',{'placeholder':placeholder,'historical_function_provenance':provenance,'split_reconstruction':splits,'original_aggregates_recomputed':summaries,'training_budget':{'steps':500,'batch_size':16,'sampled_record_presentations':8000,'unique_training_B_records':49,'A_records':0,'sampled_supervised_tokens_recomputed':count,'original_recorded_supervised_tokens':training['effective_tokens'],'initial_training_NLL':training['initial_loss'],'final_training_NLL':training['final_loss'],'training_NLL_denominator':splits['arithmetic']['train']['supervised_answer_plus_EOS_tokens']},'base_test_matches_original_sft_after_test':True,'commutative_pair_2_plus_3_in_same_train_family':True,'scripted_contract_probe':contract,'limitations':['No retraining or original neural weight inference was performed. NLL mean recomputed from original recorded nll_sum and independently reconstructed supervised-token counts.','Original one-seed small data result demonstrates lost exact-answer performance on these held-out families, not inevitable universal forgetting or newly learned arithmetic.','A actual test includes shape, color, pitch and joint queries across two held-out families, unlike the simplified two-prompt placeholder.']})
print(json.dumps({'original_fence_and_extension':'pass','historical_splits_all_hash_match':True,'A_test_matches':'5/10 -> 0/10','B_test_matches':'0/7 -> 0/7','A_test_answer_plus_EOS_tokens':69,'B_test_answer_plus_EOS_tokens':14,'B_test_NLL':[summaries[p]['arithmetic']['test']['nll_recomputed'] for p in ['before','b-only']],'sampler_supervised_tokens':count,'scripted_raw_ID_EOS_probe':'pass','neural_training':'not executed'},indent=2))
