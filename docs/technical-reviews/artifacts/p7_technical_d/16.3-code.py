import torch
from tiny_perceptron.model import TinyLM,ModelConfig
torch.manual_seed(0);model=TinyLM(ModelConfig(width=8)).eval();ids=torch.tensor([[1,2,3,4]])
with torch.no_grad():
 full=model(ids)['logits'][:,-1];prefix=model(ids[:,:3]);step=model(ids[:,3:],cache=prefix['cache']);cached=step['logits'][:,0]
error=(full-cached).abs().max().item();print('兩路分數形狀',tuple(full.shape),tuple(cached.shape));print('最大差異',error);assert torch.allclose(full,cached,atol=1e-6)
