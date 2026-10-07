import torch,json
from tiny_perceptron.model import TinyLM,ModelConfig
torch.manual_seed(0);m=TinyLM(ModelConfig(width=8)).eval()
with torch.no_grad():
 for ids in [torch.tensor([[1,2,3,5]]),torch.tensor([[6,2,3,5]])]:
  p=m(ids[:,:3]);full=m(ids)['logits'][:,-1];cached=m(ids[:,3:],cache=p['cache'])['logits'][:,0];print('IDs',ids.tolist(),'max_error',(full-cached).abs().max().item());assert torch.allclose(full,cached,atol=1e-6)
 r=json.load(open('docs/course-experiments/results/efficiency.json'))['results']['models']
 for n in ['mha','gqa']:
  c=r[n]['cache'];e=max(c['per_step_logit_max_error']);same=c['generated_ids_full']==c['generated_ids_cached'];print('archive',n,'Qheads',r[n]['model']['config']['heads'],'KVheads',r[n]['model']['config']['kv_heads'],'max_error',e,'IDs_equal',same,'steps',len(c['generated_ids_full']));assert e<1e-5 and same
