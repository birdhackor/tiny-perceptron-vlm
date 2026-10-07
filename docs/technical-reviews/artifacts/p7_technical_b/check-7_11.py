import json,platform,torch
from tiny_perceptron.data import toy_conversations,render_chat,pad_batch
from tiny_perceptron.model import TinyLM,ModelConfig,masked_loss
torch.manual_seed(42);examples=[render_chat(m) for m in toy_conversations()[:2]];x,y,valid=pad_batch(examples);m=TinyLM(ModelConfig(width=8));loss=masked_loss(m(x,valid=valid)['logits'],y);before={n:p.detach().clone() for n,p in m.named_parameters()};loss.backward()
assert list(x.shape)==[2,10] and int((y!=-100).sum())==4 and all(torch.equal(p,before[n]) for n,p in m.named_parameters())
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','seed':42,'shape':list(x.shape),'active':int((y!=-100).sum()),'loss':float(loss.detach()),'all_parameters_unchanged_without_step':True,'gradient_parameters_nonzero':sum(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters()),'two_messages':toy_conversations()[:2]},ensure_ascii=False,indent=2))
