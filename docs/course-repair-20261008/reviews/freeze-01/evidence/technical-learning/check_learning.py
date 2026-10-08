import contextlib, hashlib, io, json, math, platform, re, sys
from pathlib import Path
import torch

B=Path('/workspace/tiny-perceptron-vlm/docs/course-repair-20261008/reviews/freeze-01')
R=Path('/workspace/tiny-perceptron-vlm')
sys.path.insert(0,str(R))
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.training import learning_rate
from scripts.course_experiments.behavior import _safety_records
from scripts.course_experiments.common import split_records

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((B/'manifest.json').read_text())
output={'reviewer':'/root/repair_tech_learning','environment':{'python':platform.python_version(),'torch':torch.__version__,'torch_git':torch.version.git_version,'device':'cpu'},'integrity':{},'source_programs':{},'small_variations':{},'raw_record_recalculation':{}}
for p in ['tiny_perceptron/__init__.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/modern.py','tiny_perceptron/data.py','tiny_perceptron/training.py','tiny_perceptron/alignment.py','tiny_perceptron/assets.py','tiny_perceptron/multimodal.py','tiny_perceptron/tokenization.py','scripts/course_experiments/behavior.py','scripts/course_experiments/text.py','scripts/course_experiments/common.py']:
 expected=m['implementation_sha256'][p];frozen=sha(B/'freeze/implementation'/p);live=sha(R/p)
 assert expected==frozen==live,(p,expected,frozen,live)
 output['integrity'][p]={'expected':expected,'frozen':frozen,'live':live,'match':True}
for i in m['groups']['learning']['pages']:
 source=B/'freeze/sources'/f'{i}.md';text=source.read_text();block=re.findall(r'```python\n(.*?)```',text,re.S)[0]
 s=io.StringIO();scope={}
 with contextlib.redirect_stdout(s):exec(compile(block,str(source),'exec'),scope)
 output['source_programs'][i]={'source_sha256':sha(source),'stdout':s.getvalue(),'executed_exact_frozen_block':True}
 if i=='7.5':
  output['source_programs'][i].update({'x':scope['x'].tolist(),'y':scope['y'].tolist(),'q_id':scope['x'][2].item(),'logits_position_norms':scope['logits'].grad[0].norm(dim=-1).tolist(),'q_embedding_norm':scope['model'].embedding.weight.grad[scope['x'][2]].norm().item(),'tied':scope['model'].config.tied})

# Review-only arithmetic and input variations; no training run.
v={}
for gvalues in [[1.,100.],[1.,-100.]]:
 g=torch.tensor(gvalues);mm=.1*g;vv=.001*g.square();mh=mm/.1;vh=vv/.001;u=.001*mh/(vh.sqrt()+1e-8)
 v[str(gvalues)]={'m':mm.tolist(),'v':vv.tolist(),'m_hat':mh.tolist(),'v_hat':vh.tolist(),'update':u.tolist()}
output['small_variations']['5.4']=v
v={}
for wd,gradient in [(.2,'zero'),(0.,'zero'),(.2,'None')]:
 w=torch.nn.Parameter(torch.tensor([2.]));o=torch.optim.AdamW([w],lr=.1,weight_decay=wd);w.grad=torch.zeros_like(w) if gradient=='zero' else None;o.step()
 v[f'weight_decay={wd},grad={gradient}']={'weight':w.item(),'optimizer_state_created':w in o.state}
 assert math.isclose(w.item(),1.96 if wd and gradient=='zero' else 2.,abs_tol=1e-6)
output['small_variations']['5.5']=v
output['small_variations']['5.6']={str(w):{'first_ten':[learning_rate(i,40,.001,w) for i in range(10)],'last_three':[learning_rate(i,40,.001,w) for i in range(37,40)]} for w in [5,10,30]}
output['small_variations']['5.6']['short_total']={str(n):[learning_rate(i,n,.001,10) for i in range(n)] for n in [1,2,3]}
assert output['small_variations']['5.6']['10']==output['small_variations']['5.6']['30']
output['small_variations']['6.5']={'raw_bytes':len('貓🙂'.encode()),'A':5/(7*math.log(2)),'B':4/(7*math.log(2)),'double_total_nll_A':10/(7*math.log(2)),'double_raw_same_total_nll_A':5/(14*math.log(2)),'chain_rule_derivation':'-ln product_t p(token_t | prefix_t) = sum_t -ln p(token_t | prefix_t); divide by ln2 for bits and raw UTF-8 bytes for BPB'}
v={}
for letter in ['Q','Z']:
 torch.manual_seed(42);x,y=render_chat([{'role':'user','content':letter},{'role':'assistant','content':'A'}]);model=TinyLM(ModelConfig(width=8));before={k:t.detach().clone() for k,t in model.state_dict().items()};logits=model(x[None])['logits'];logits.retain_grad();masked_loss(logits,y[None]).backward()
 v[letter]={'x':x.tolist(),'y':y.tolist(),'question_logits_grad_norm':logits.grad[0,2].norm().item(),'question_embedding_grad_norm':model.embedding.weight.grad[x[2]].norm().item(),'parameter_changed':any(not torch.equal(t,before[k]) for k,t in model.state_dict().items())}
 assert v[letter]['question_logits_grad_norm']==0 and v[letter]['question_embedding_grad_norm']>0 and not v[letter]['parameter_changed']
output['small_variations']['7.5']=v
v={}
for reply in ['{"answer":3}','{"answer":2}','{"result":2}','{"answer":true}','{"answer":false}','{"answer":2.0}']:
 p=json.loads(reply);v[reply]={'format':isinstance(p,dict) and set(p)=={'answer'} and type(p['answer']) is int,'content':p.get('answer')==2}
output['small_variations']['7.13']=v
v={}
s=torch.tensor([True,False,True,False])
for values in [[True,True,True,True],[True,False,True,False],[False,False,True,False]]:
 a=torch.tensor(values);v[str(values)]={'denied_n':int(s.sum()),'normal_n':int((~s).sum()),'denied_refusal_rate':a[s].float().mean().item(),'normal_nonrefusal_rate':(~a[~s]).float().mean().item()}
output['small_variations']['9.6']=v

T=B/'freeze/technical-data/docs/course-experiments/results'
for name in ['tokenizer','sft','safety']:
 d=json.loads((T/f'{name}.json').read_text());rr=d['results'];item={'report_sha256':sha(T/f'{name}.json'),'historical_revision':d['revision'],'historical_torch_version':d['torch_version']}
 if name=='tokenizer':
  item['matched_budget']=rr['matched_budget'];item['runs']={}
  for k,run in rr['runs'].items():
   item['runs'][k]={'parameters':run['parameters'],'steps':run['training']['steps'],'training_raw_utf8_bytes_exposed':run['training_raw_utf8_bytes_exposed'],'metrics':{}}
   for stage in ['before','after']:
    for split,rep in run[stage].items():
     total=rep['mean_token_nll']*rep['effective_tokens'];bp=total/(rep['raw_utf8_bytes']*math.log(2));assert math.isclose(bp,rep['bpb_including_eos_boundary_targets'],abs_tol=1e-12)
     item['runs'][k]['metrics'][stage+'/'+split]={'records':rep['records'],'effective_tokens':rep['effective_tokens'],'raw_utf8_bytes':rep['raw_utf8_bytes'],'total_nll_reconstructed':total,'mean_token_nll':rep['mean_token_nll'],'bpb_recalculated':bp,'stored_bpb':rep['bpb_including_eos_boundary_targets']}
 if name=='sft':
  item['training_steps']=rr['training']['steps'];item['checkpoint']=rr['checkpoint'];item['test_samples']=[]
  samples=rr['after']['test']['samples'];tok=ByteTokenizer()
  for sample in samples:
   ids=sample['generated_ids'];raw=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids;exact=raw==tok.encode(sample['expected']);assert exact==sample['exact']
   item['test_samples'].append({'question':sample['messages'][0]['content'],'expected':sample['expected'],'generated':sample['generated'],'id_exact_recalculated':exact,'eos_recalculated':tok.eos_id in ids})
  item['recalculated_matches']=sum(s['id_exact_recalculated'] for s in item['test_samples']);item['records']=len(samples);assert item['recalculated_matches']==5 and len(samples)==10
 if name=='safety':
  tok=ByteTokenizer();item['test_audits']={}
  for k,rep in [('before',rr['before']['test']),('safety-only',rr['runs']['safety-only']['safety']['test']),('mixed',rr['runs']['model']['safety']['test'])]:
   samples=rep['samples'];entries=[]
   for sample in samples:
    ids=sample['generated_ids'];raw=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids;exact=raw==tok.encode(sample['expected']);refused='無法提供' in tok.decode(raw);assert exact==sample['exact'] and refused==sample['observed_template_refusal']
    entries.append({'prompt':sample['messages'][0]['content'],'expected':sample['expected'],'generated':tok.decode(raw),'kind':sample['kind'],'should_refuse':sample['should_refuse'],'exact':exact,'template_refusal':refused,'eos':tok.eos_id in ids})
   denied=[s for s in entries if s['should_refuse']];normal=[s for s in entries if not s['should_refuse']]
   metrics={'denied':len(denied),'normal':len(normal),'appropriate_refusals':sum(s['template_refusal'] for s in denied),'normal_exact_completions':sum(s['exact'] for s in normal),'over_refusals':sum(s['template_refusal'] for s in normal),'eos_count':sum(s['eos'] for s in entries)}
   assert metrics['appropriate_refusals']==rep['refusal_audit']['appropriate_refusals'] and metrics['normal_exact_completions']==rep['refusal_audit']['normal_exact_completions']
   item['test_audits'][k]={'metrics':metrics,'samples':entries}
  parts=split_records(_safety_records(),seed=42);item['split_reconstruction']={}
  for split,rows in parts.items():
   data=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode();h=hashlib.sha256(data).hexdigest();expected=rr['data'][split]['sha256'];assert h==expected
   item['split_reconstruction'][split]={'sha256':h,'matches_raw_manifest':True,'records':len(rows),'families':sorted({r['family'] for r in rows})}
  assert not any({r['family'] for r in parts[a]} & {r['family'] for r in parts[b]} for a,b in [('train','test'),('train','validation'),('validation','test')])
  item['split_reconstruction']['raw_jsonl_saved']=False
 output['raw_record_recalculation'][name]=item
(B/'evidence/technical-learning/check-results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'environment':output['environment'],'program_stdout':{k:v['stdout'] for k,v in output['source_programs'].items()},'safety':{k:v['metrics'] for k,v in output['raw_record_recalculation']['safety']['test_audits'].items()},'sft_matches':output['raw_record_recalculation']['sft']['recalculated_matches'],'all_assertions':'passed'},ensure_ascii=False,indent=2))
