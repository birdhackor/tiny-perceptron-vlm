from pathlib import Path
import sys,json,hashlib,platform,math
root=Path(__file__).resolve().parents[5];sys.path.insert(0,str(root))
import torch
from tiny_perceptron.posttraining import FiniteResponsePolicy,FiniteRewardModel,FiniteValueModel,ppo_clipped_objective,bandit_advantage,preference_loss
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.retrieval import retrieve,call_tool
from scripts.course_experiments.posttraining import build_records,split_records
from scripts.course_experiments.common import records_sha256
out=Path(__file__).resolve().parent
result={'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','cuda_available':str(torch.cuda.is_available())},'training_updates_executed':0}
torch.manual_seed(0)
model=TinyLM(ModelConfig(width=8)).eval();ids=torch.tensor([[1,2,3,4]])
with torch.no_grad():
 full=model(ids)['logits'][:,-1];prefix=model(ids[:,:3]);step=model(ids[:,3:],cache=prefix['cache']);cached=step['logits'][:,0]
assert torch.allclose(full,cached,atol=1e-6,rtol=0)
result['cache']={'seed':0,'input_ids':ids.tolist(),'shape':list(full.shape),'maximum_logit_error':float((full-cached).abs().max()),'prefix_length':prefix['cache'][0][0].shape[2],'appended_length':step['cache'][0][0].shape[2],'absolute_tolerance':1e-6}
ratio=torch.tensor([.7,1,1.3,.7,1,1.3],requires_grad=True);adv=torch.tensor([1.,1,1,-1,-1,-1])
t=ppo_clipped_objective(ratio.log(),torch.zeros(6),adv);(-t['surrogate'].sum()).backward()
assert torch.allclose(t['surrogate'],torch.tensor([.7,1,1.2,-.8,-1,-1.3]),atol=1e-6)
assert torch.allclose(ratio.grad,torch.tensor([-1.,-1,0,0,1,1]),atol=1e-6)
result['ppo']={'ratio_inputs':ratio.detach().tolist(),'advantages':adv.tolist(),'surrogate':t['surrogate'].tolist(),'negative_objective_ratio_gradients':ratio.grad.tolist(),'advantage_example':bandit_advantage(torch.tensor([1.,0]),torch.tensor([.4,.4])).tolist(),'comparison_loss_at_zero_gap':float(preference_loss(torch.zeros(1),torch.zeros(1)))}
records=build_records();splits=split_records(records)
official=json.loads((root/'docs/course-experiments/results/posttraining.json').read_text())['results']
checks={}
for name,rows in splits.items():
 sha=records_sha256(rows);actual_families=sorted({r['family'] for r in rows});expected=official['splits'][name]
 assert sha==expected['sha256'] and actual_families==expected['families'] and len(rows)==expected['contexts']
 checks[name]={'contexts':len(rows),'families':len(actual_families),'records_sha256':sha,'official_match':True}
assert len(records)==165 and all(len(r['candidates'])==4 for r in records)
assert all(r['candidates'][0]==str(sum(r['operands'])) and r['candidates'][2]==str(sum(r['operands'])+1) for r in records)
assert all(r['preference_pairs']==[[3,0],[3,1],[3,2]] for r in records if r['mode']=='missing')
x=torch.tensor([records[0]['features']],dtype=torch.float32)
policy=FiniteResponsePolicy();reward=FiniteRewardModel();critic=FiniteValueModel()
assert list(policy(x).shape)==[1,4] and list(reward(x).shape)==[1,4] and list(critic(x).shape)==[1]
result['project_demo']={'generated_contexts':len(records),'candidates_per_context':4,'candidate_texts_from_first_record':records[0]['candidates'],'label_source_values':sorted({r['label_source'] for r in records}),'mode_counts':{m:sum(r['mode']==m for r in records) for m in ['number','explain','missing']},'splits_independently_rebuilt':checks,'policy_input_feature_count':len(records[0]['features']),'policy_output_shape':list(policy(x).shape),'reward_output_shape':list(reward(x).shape),'value_output_shape':list(critic(x).shape),'human_rating_records':0,'autoregressive_generation_calls':0}
scores=torch.tensor([0.,math.log(2),math.log(4)]);values=torch.tensor([10.,20,30]);m=torch.tensor(-float('inf'));den=torch.tensor(0.);num=torch.tensor(0.);stats=[]
for start in range(0,3,2):
 b=scores[start:start+2];new_m=torch.maximum(m,b.max());factor=torch.exp(m-new_m);w=torch.exp(b-new_m);den=den*factor+w.sum();num=num*factor+(w*values[start:start+2]).sum();m=new_m;stats.append([float(den),float(num)])
assert abs(float(num/den)-170/7)<1e-5
result['online_softmax']={'scores':scores.tolist(),'values':values.tolist(),'block_size':2,'denominator_numerator_after_each_block':stats,'blocked_result':float(num/den),'whole_result':float((scores.softmax(0)*values).sum()),'exact_hand_result':'170 / 7 = 24.285714285714285','GPU_backend_executed':False}
xq=torch.tensor([-.7,.2,.7,1.2]);q=(xq/.5).round().clamp(-127,127).to(torch.int8)
assert q.tolist()==[-1,0,1,2]
result['quantization']={'input':xq.tolist(),'scale':.5,'q':q.tolist(),'restored':(q.float()*.5).tolist(),'original_tensor_payload_bytes':xq.numel()*xq.element_size(),'quantized_tensor_payload_bytes':q.numel()*q.element_size(),'scope':'Payload only; scale, metadata, serialization, quality and speed are not measured.'}
ref=torch.tensor([.5,.5]).log();dpos=[]
for p in [[.5,.5],[.7,.3],[.3,.7]]:
 log=torch.tensor(p).log();dpos.append(float(dpo_loss(log[:1],log[1:],ref[:1],ref[1:],beta=1)))
assert all(abs(a-b)<1e-6 for a,b in zip(dpos,[-math.log(.5),-math.log(.7),-math.log(.3)]))
result['dpo']={'beta':1,'fixed_reference':[.5,.5],'preferred_rejected_probabilities':[[.5,.5],[.7,.3],[.3,.7]],'losses':dpos}
docs=[{'id':'d1','text':'貓喜歡曬太陽'},{'id':'d2','text':'書店地址是青街8號'}]
assert retrieve('書店地址',docs,k=1)==[docs[1]]
result['retrieval']={'query':'書店地址','hits':retrieve('書店地址',docs,k=1),'paraphrase_hits':retrieve('營業處在哪',docs,k=1),'generation_executed':False}
requests=['{"name":"add","arguments":{"a":2,"b":3}}','{"name":"multiply","arguments":{"a":123,"b":45}}','{"name":"delete_all","arguments":{"a":2,"b":3}}','{"name":"add","arguments":{"a":true,"b":3}}'];rows=[]
for request in requests:
 try:rows.append({'request':request,'result':call_tool(request)})
 except ValueError as error:rows.append({'request':request,'rejected':str(error)})
assert rows[0]['result']==5 and rows[1]['result']==5535 and all('rejected' in row for row in rows[2:])
result['tool_calls']=rows
(out/'cpu-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
