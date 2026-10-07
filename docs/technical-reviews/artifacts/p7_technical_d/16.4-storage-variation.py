import torch,json
from tiny_perceptron.model import TinyLM,ModelConfig
with torch.no_grad():
 for kv in [4,1]:
  m=TinyLM(ModelConfig(width=16,heads=4,kv_heads=kv)).eval();c=m(torch.arange(1,7)[None])['cache'];print('toy6',kv,'shape',tuple(c[0][0].shape),'bytes',sum(t.numel()*t.element_size() for p in c for t in p))
 r=json.load(open('docs/course-experiments/results/efficiency.json'))['results']['models']
 for n in ['mha','gqa']:
  d=r[n];m=TinyLM(ModelConfig(**d['model']['config'])).eval();c=m(torch.arange(1,26)[None])['cache'];size=sum(t.numel()*t.element_size() for p in c for t in p);print('rawconfig',n,d['model']['config'],'actualbytes',size,'rawbytes',d['cache']['prefill_cache_bytes']);assert size==d['cache']['prefill_cache_bytes']
