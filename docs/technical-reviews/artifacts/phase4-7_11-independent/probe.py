import ast, copy, hashlib, json, math, pathlib, random, sys
ROOT=pathlib.Path('/workspace/tiny-perceptron-vlm');sys.path.insert(0,str(ROOT));A=ROOT/'docs/technical-reviews/artifacts/phase4-7_11-independent'
import torch
from tiny_perceptron.data import ByteTokenizer,IGNORE,pad_batch,render_chat,shifted,toy_conversations
from tiny_perceptron.model import ModelConfig,TinyLM,masked_loss,loss_sum
from torch.nn import functional as F
torch.set_num_threads(1);torch.set_default_device('cpu');assert torch.version.cuda is None and not torch.cuda.is_available()
# Execute saved original fences verbatim; snapshot weights in memory only.
ns={'__name__':'__main__'}
for fence in ('fence-1.py','fence-2.py'):
 exec(compile((A/'original'/fence).read_bytes(),'original/'+fence,'exec'),ns)
x,y,valid,model,loss=(ns[k] for k in ('x','y','valid','model','loss'))
assert x.shape==y.shape==valid.shape==(2,10)
assert y.tolist()==[[-100]*8+[56,2],[-100]*8+[57,2]]
assert x.tolist()==[[1,3,56,51,56,69,71,2,4,56],[1,3,56,51,57,69,71,2,4,57]]
assert int((y!=IGNORE).sum())==4 and bool(valid.all())
assert ns['exercise_y'][ns['exercise_y']!=IGNORE].tolist()==[59,2]
old={n:p.detach().clone() for n,p in model.named_parameters()}
model.zero_grad(set_to_none=True);logits=model(x,valid=valid)['logits'];logits.retain_grad();computed=masked_loss(logits,y);computed.backward()
assert logits.shape==(2,10,264)
manual=(-logits.log_softmax(-1)[y!=IGNORE].gather(1,y[y!=IGNORE,None]).sum()/4).detach()
assert abs(float(computed.detach()-manual))<=1e-6
assert all(torch.equal(old[n],p) for n,p in model.named_parameters())
assert bool((logits.grad[y==IGNORE]==0).all())
assert float(model.embedding.weight.grad[51].abs().sum())>0 # '+' is only a question token.
gradients={n:{'finite':bool(torch.isfinite(p.grad).all()),'nonzero':int(torch.count_nonzero(p.grad))} for n,p in model.named_parameters()}
assert all(v['finite'] and v['nonzero']>0 for v in gradients.values())
# One explicit variant step shows the difference between backward and updating parameters; no save.
optimizer=torch.optim.AdamW(model.parameters(),lr=0.003);optimizer.step()
changed=[n for n,p in model.named_parameters() if not torch.equal(old[n],p)]
assert len(changed)==len(old)
# Shallow-copy isolation checks the same retained original list, rather than regenerating it.
original=toy_conversations()[0];before=copy.deepcopy(original)
messages=[m.copy() for m in original];messages[1]['content']='3'
assert original==before and original[1]['content']=='0'
assert all(m is not o for m,o in zip(messages,original))
ex,ey=render_chat(messages);assert ey[ey!=IGNORE].tolist()==[59,2]
assert torch.equal(ex[:-1],x[0,:-1]) and int(ex[-1])==59
messages[1]['content']='0';assert messages==original
variants={}
for answer in ('','12','三'):
 changed_messages=[m.copy() for m in original];changed_messages[1]['content']=answer
 vx,vy=render_chat(changed_messages);active=vy[vy!=IGNORE].tolist();assert active==ByteTokenizer().encode(answer)+[2]
 assert len(vx)==len(vy)==9+len(answer.encode());variants[answer]={'length':len(vx),'targets':active}
question=[m.copy() for m in original];question[0]['content']='0+2=?';qx,qy=render_chat(question);assert torch.equal(qy,render_chat(original)[1])
noanswer=[m.copy() for m in original];noanswer[1]['role']='user'
try:render_chat(noanswer);raise AssertionError('must reject missing assistant')
except ValueError as e:variants['missing_assistant_error']=str(e)
try:pad_batch([render_chat(original)],max_length=8);raise AssertionError('must reject empty target after crop')
except ValueError as e:variants['crop8_error']=str(e)
cx,cy,cv=pad_batch([render_chat(original)],max_length=9);assert cy[cy!=IGNORE].tolist()==[56]
variants['crop9']={'targets':cy[cy!=IGNORE].tolist(),'meaning':'one answer byte retained; EOS lost by explicit crop'}
# Unequal sequence lengths distinguish attention/PAD mask from assistant target mask.
long=[{'role':'user','content':'0+0=? longer'},{'role':'assistant','content':'12'}]
px,py,pv=pad_batch([render_chat(original),render_chat(long)])
assert pv[0].sum()==10 and not bool(pv[0,10:].any()) and bool((py[0,10:]==IGNORE).all())
with torch.no_grad():
 single=model(x[:1],valid=valid[:1])['logits'];padded=model(px,valid=pv)['logits'][:1,:10]
difference=float((single-padded).abs().max());assert difference<=2e-6
ignored=logits.detach().clone();ignored[y==IGNORE]=1000;assert torch.equal(masked_loss(ignored,y),computed.detach())
# Original historical implementation: parse only pure data/tokenization functions from git-show bytes.
hns={'hashlib':hashlib,'json':json,'random':random,'torch':torch,'IGNORE':IGNORE,'SPECIALS':tuple(range(8))}
def historical_nodes(path,names):
 tree=ast.parse((A/'historical'/path).read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'historical/'+path,'exec'),hns)
historical_nodes('tiny_perceptron/data.py',['ByteTokenizer','shifted','render_chat'])
historical_nodes('scripts/prepare_data.py',['conversation','generate_records'])
historical_nodes('scripts/course_experiments/common.py',['split_records','records_sha256','text_examples'])
parts=hns['split_records'](hns['generate_records']('attributes-sft'),seed=42)
raw=json.loads((A/'inputs/original-sft.json').read_text());r=raw['results'];historical={}
for split,rows in parts.items():
 b=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode();h=hashlib.sha256(b).hexdigest();assert h==r['data'][split]['sha256']
 dest=A/'historical-data'/f'{split}.jsonl';dest.parent.mkdir(exist_ok=True);dest.write_bytes(b)
 historical[split]={'records':len(rows),'families':len({row['family'] for row in rows}),'sha256':h,'effective_tokens':sum(int((yy!=IGNORE).sum()) for xx,yy in hns['text_examples'](rows,mode='sft'))}
assert [historical[s]['records'] for s in ('train','validation','test')]==[45,5,10]
assert [historical[s]['families'] for s in ('train','validation','test')]==[9,1,2]
text_records=[{'text':row['messages'][0]['content']+row['messages'][1]['content'],'family':row['family']} for row in parts['train']]
(A/'historical-data/pretrain-text.json').write_text(json.dumps(text_records,ensure_ascii=False,indent=2)+'\n')
for name,rows,mode,steps,result in [
 ('direct_sft',parts['train'],'sft',900,r['training']),
 ('pretraining',text_records,'text',250,r['pretrain_then_sft']['pretraining']),
 ('continued_sft',parts['train'],'sft',900,r['pretrain_then_sft']['sft'])]:
 examples=hns['text_examples'](rows,mode=mode,max_length=128);sampler=random.Random(42)
 tokens=sum(sum(int((yy!=IGNORE).sum()) for xx,yy in sampler.choices(examples,k=16)) for _ in range(steps))
 assert tokens==result['effective_tokens'] and steps==result['steps'] and len(rows)==result['records']
 assert hns['records_sha256'](rows)==result['records_sha256']
 historical[name]={'steps':steps,'records':len(rows),'batch_size':16,'lr':0.003,'effective_tokens':tokens,'records_sha256':hns['records_sha256'](rows)}
# Each original held-out evaluation uses the same generated messages/answers, all samples included.
heldout={}
for branch,evaluations in [('direct_before',r['before']),('direct_after',r['after']),('pretrain_before_sft',r['pretrain_then_sft']['before_sft']),('pretrain_after_sft',r['pretrain_then_sft']['after_sft'])]:
 heldout[branch]={}
 for split,ev in evaluations.items():
  assert [(row['messages'][:-1],row['messages'][-1]['content']) for row in parts[split]]==[(sample['messages'],sample['expected']) for sample in ev['samples']]
  matches=sum(s['generated']==s['expected'] for s in ev['samples']);eos=sum(2 in s['generated_ids'] for s in ev['samples'])
  assert len(ev['samples'])==len(parts[split])==ev['records']==ev['examples'];assert matches==ev['matches'];assert ev['exact_match']==matches/len(parts[split]);assert ev['eos_rate']==eos/len(parts[split])
  assert ev['effective_tokens']==historical[split]['effective_tokens'];assert abs(ev['nll']-ev['nll_sum']/ev['effective_tokens'])<1e-12
  heldout[branch][split]={'records':len(ev['samples']),'matches':matches,'eos_count':eos,'effective_tokens':ev['effective_tokens']}
config_model=TinyLM(ModelConfig(width=64,layers=2));assert sum(p.numel() for p in config_model.parameters())==141568
assert r['training']['parameters']==r['training']['trainable_parameters']==141568
assert historical['direct_sft']['steps']+250==historical['pretraining']['steps']+historical['continued_sft']['steps']==1150
results={'original_loss':float(loss.detach()),'manual_mean_loss':float(manual),'shapes':{'input':list(x.shape),'logits':list(logits.shape)},'valid_inputs':int(valid.sum()),'answer_tokens':int((y!=IGNORE).sum()),'original_targets':y.tolist(),'exercise_targets':ey[ey!=IGNORE].tolist(),'original_data_preserved':True,'backward_weights_unchanged':True,'question_plus_embedding_gradient_l1':float(model.embedding.weight.grad[51].abs().sum()),'gradients':gradients,'explicit_single_optimizer_step_changed_tensors':changed,'variants':variants,'padding_max_logit_difference':difference,'historical':historical,'heldout_recomputed':heldout,'weight_artifacts_written':False,'training_rerun':False}
(A/'probe-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False))
