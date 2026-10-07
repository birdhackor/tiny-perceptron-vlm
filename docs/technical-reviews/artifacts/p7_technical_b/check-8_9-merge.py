import torch,json
from torch import nn
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.tokenization import ByteTokenizer
r=json.load(open('docs/technical-reviews/artifacts/p7_technical_b/sources/lora-result.json'))['results']
assert r['a_b_a_max_difference']==0 and r['adapter_a_b_logit_difference']>0 and r['merge_max_difference']<=.0001
ids=[1,3,57,51,57,69,71,2,4];t=ByteTokenizer();assert t.decode([57,51,57,69,71])=='1+1=?'
out={'raw_switch_difference':r['a_b_a_max_difference'],'raw_A_B_difference':r['adapter_a_b_logit_difference'],'raw_merge_difference':r['merge_max_difference'],'raw_merge_tolerance':.0001,'probe_ids':ids,'probe_question':'1+1=?'}
torch.manual_seed(0);l=LoRALinear(nn.Linear(4,3),rank=2,alpha=2);x=torch.randn(2,4)
with torch.no_grad():
 l.b.fill_(.1);original=l(x);base=l.base(x);update=original-base
 l.alpha=4;double=l(x)-base;assert torch.allclose(double,2*update,atol=1e-6)
 l.alpha=2;merged=nn.Linear(4,3);merged.weight.copy_(l.merged_weight());merged.bias.copy_(l.base.bias)
 md=(merged(x)-original).abs().max().item();assert md<1e-6
 duplicate=merged(x)+(x@l.a.T@l.b.T)*(l.alpha/l.rank);dd=(duplicate-original).abs().max().item();assert dd>0
out['single_layer']={'alpha2_to4_double_update':True,'merged_max_difference':md,'duplicate_apply_max_difference':dd}
print(json.dumps(out,indent=2))
