import torch, json, hashlib
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
torch.set_num_threads(2)
torch.manual_seed(42)
core=LimitedAssistant(SelftrainedConfig(vocab_size=32,width=16,layers=1,heads=2,kv_heads=1,ffn_hidden=32,top_k=2))
ids=torch.tensor([[1,11,12,13,14]])
def digest():
 h=hashlib.sha256()
 for key,value in core.state_dict().items():
  h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
before=digest()
core.train()
train=core(ids)["logits"]
assert train.requires_grad
core.eval()
eval_graph=core(ids)["logits"]
assert eval_graph.requires_grad
with torch.no_grad():
 eval_no_grad=core(ids)["logits"]
 assert not eval_no_grad.requires_grad
 full=eval_no_grad[:,-1]
 cache2=core(ids[:,:2])["cache"]
 split2=core(ids[:,2:],cache=cache2)["logits"][:,-1]
 cache3=core(ids[:,:3])["cache"]
 wrong=core(ids[:,3:],cache=cache3,positions=torch.arange(2))["logits"][:,-1]
assert torch.equal(train,eval_graph)
assert torch.equal(eval_graph,eval_no_grad)
assert torch.allclose(full,split2,atol=1e-4,rtol=1e-4)
assert not torch.allclose(full,wrong,atol=1e-4,rtol=1e-4)
for b in core.lm.blocks:b.attention.backend="manual"
with torch.no_grad():manual=core(ids)["logits"][:,-1]
assert torch.allclose(full,manual,atol=1e-4,rtol=1e-4)
after=digest();assert before==after
mode_layers=[(name,type(m).__name__) for name,m in core.named_modules() if isinstance(m,(torch.nn.modules.dropout._DropoutNd,torch.nn.modules.batchnorm._BatchNorm))]
assert mode_layers==[]
print(json.dumps({"train_eval_max_diff":(train-eval_graph).abs().max().item(),"eval_grad_no_grad_max_diff":(eval_graph-eval_no_grad).abs().max().item(),"requires_grad":[train.requires_grad,eval_graph.requires_grad,eval_no_grad.requires_grad],"all_modules_eval":all(not m.training for m in core.modules()),"dropout_batchnorm_modules":mode_layers,"split2_diff":(full-split2).abs().max().item(),"wrong_reset_position_diff":(full-wrong).abs().max().item(),"manual_sdpa_diff":(full-manual).abs().max().item(),"parameters_before":before,"parameters_after":after,"cache_shapes":[[list(t.shape) for t in layer] for layer in cache3]},indent=2))
