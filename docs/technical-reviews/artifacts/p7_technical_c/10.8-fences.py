import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

torch.manual_seed(0)
lm = TinyLM(ModelConfig(width=8)).eval()
multimodal = MultiModalLM(lm).eval()
ids = torch.tensor([1, 20, 30])
a = lm(ids[None])["logits"]
b = multimodal(ids)["logits"]
print("分數形狀", tuple(a.shape))
print("純文字入口相同", torch.equal(a, b))


# Owner supplied proportional check
other_ids=torch.tensor([1,40,50])
print('other ids shape/equal',tuple(lm(other_ids[None])['logits'].shape),torch.equal(lm(other_ids[None])['logits'],multimodal(other_ids)['logits']))
try:
    multimodal(torch.tensor([1,5,4]))
except ValueError as e:
    print('missing image ValueError',str(e))
print('TinyLM module types',sorted({type(m).__name__ for m in lm.modules()}))
lm.train()
print('TinyLM repeated train-mode forward equal',torch.equal(lm(ids[None])['logits'],lm(ids[None])['logits']))

