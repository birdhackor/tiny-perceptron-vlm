import json,torch
from tiny_perceptron.data import ByteTokenizer
print('torch',torch.__version__)
x=torch.tensor([1.,2.,3.]);w=torch.tensor(1.,requires_grad=True)
for p in (x[:2],x[2:]):((w*p).square().mean()/2).backward()
print('wrong variation2/1',w.grad.item());assert w.grad.item()==11.5
r=json.load(open('docs/course-experiments/results/efficiency.json'))['results'];p=r['accumulation'];assert sum(p['effective_token_counts'])==p['shared_denominator']==20;print('mechanism',p)
t=ByteTokenizer()
for name in ['ordinary','accumulated']:
 d=r['update_variants'][name];q=d['training'];assert q['steps']==q['optimizer_updates']==40 and q['effective_tokens']==2328
 print(name,'steps',q['steps'],'targets',q['effective_tokens'],'micro',q['micro_batch_sizes'],'ms',q['warm_step_median_seconds']*1000)
 b=q['memory_allocated_before_bytes'];z=q['peak_memory_allocated_bytes'];a=q['peak_additional_allocated_bytes'];assert z-b==a
 print(name,'before/peak/add MiB',b/2**20,z/2**20,a/2**20)
 h=d['heldout']['test'];c=0
 for s in h['samples']:
  ids=s['generated_ids'];ids=ids[:ids.index(2)] if 2 in ids else ids;ok=ids==t.encode(s['expected']);assert ok==s['exact'];c+=ok
 assert c==h['matches']==5;assert h['effective_tokens']==69
 print(name,'test sum/denom/NLL',h['nll_sum'],69,h['nll_sum']/69,'matches',c,'/',len(h['samples']))
print('weight_error',r['update_variants']['accumulated']['weight_max_error_from_ordinary_after_updates'])
