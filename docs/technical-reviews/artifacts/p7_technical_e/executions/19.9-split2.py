import torch
from tiny_perceptron.selftrained.model import LimitedAssistant,SelftrainedConfig
torch.set_num_threads(2);torch.manual_seed(42)
core=LimitedAssistant(SelftrainedConfig(vocab_size=32,width=16,layers=1,heads=2,kv_heads=1,ffn_hidden=32,top_k=2)).eval()
ids=torch.tensor([[1,11,12,13,14]]);split=2
with torch.no_grad():
 full=core(ids)["logits"][:,-1]
 cache=core(ids[:,:split])["cache"]
 cached=core(ids[:,split:],cache=cache)["logits"][:,-1]
print("split2 maxdiff",(full-cached).abs().max().item(),"allclose",torch.allclose(full,cached,atol=1e-4,rtol=1e-4))
assert torch.allclose(full,cached,atol=1e-4,rtol=1e-4)
