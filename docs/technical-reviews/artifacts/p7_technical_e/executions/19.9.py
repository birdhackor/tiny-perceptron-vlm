import torch,json
from tiny_perceptron.selftrained.model import LimitedAssistant,SelftrainedConfig
torch.set_num_threads(2);torch.manual_seed(42)
config=SelftrainedConfig(vocab_size=32,width=16,layers=1,heads=2,kv_heads=1,ffn_hidden=32,top_k=2)
core=LimitedAssistant(config).eval();ids=torch.tensor([[1,11,12,13,14]]);split=3
with torch.no_grad():
 full=core(ids)["logits"][:,-1];cache=core(ids[:,:split])["cache"];cached=core(ids[:,split:],cache=cache)["logits"][:,-1]
print("最大分數差",(full-cached).abs().max().item());print("同一計算",torch.allclose(full,cached,atol=1e-4,rtol=1e-4))
assert torch.allclose(full,cached,atol=1e-4,rtol=1e-4)
print("cache shapes",[[list(t.shape) for t in layer] for layer in cache])
with torch.no_grad():
 wrong=core(ids[:,split:],cache=cache,positions=torch.arange(2))["logits"][:,-1]
print("reset position wrong",(full-wrong).abs().max().item(),torch.allclose(full,wrong,atol=1e-4,rtol=1e-4))
assert not torch.allclose(full,wrong,atol=1e-4,rtol=1e-4)
for b in core.lm.blocks:b.attention.backend="manual"
with torch.no_grad():manual=core(ids)["logits"][:,-1]
print("same weights manual vs sdpa",(full-manual).abs().max().item(),torch.allclose(full,manual,atol=1e-4,rtol=1e-4))
assert torch.allclose(full,manual,atol=1e-4,rtol=1e-4)
