import torch,json
from tiny_perceptron.data import ByteTokenizer
print('torch',torch.__version__);t=ByteTokenizer();r=json.load(open('docs/course-experiments/results/precision.json'))['results']['variants']
for n,d in r.items():
 q=d['training'];b=q['memory_allocated_before_bytes'];p=q['peak_memory_allocated_bytes'];a=q['peak_additional_allocated_bytes'];assert p-b==a;print(n,'memorybytes',b,p,a,'MiB3',[round(x/2**20,3) for x in [b,p,a]])
 for sp,h in d['heldout'].items():assert all(2 in s['generated_ids'] and len(s['generated_ids'])<=32 for s in h['samples']);print(n,sp,'EOSall',len(h['samples']))
 s=d['heldout']['test']['samples'][0];print(n,'first',s['expected'],t.decode(s['generated_ids']));assert t.decode(s['generated_ids'])==s['generated']
