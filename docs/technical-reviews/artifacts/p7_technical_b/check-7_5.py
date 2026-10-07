import json,platform,torch
from pathlib import Path
from tiny_perceptron.data import render_chat
from tiny_perceptron.model import TinyLM,ModelConfig,masked_loss
torch.manual_seed(42)
x,y=render_chat([{'role':'user','content':'Q'},{'role':'assistant','content':'A'}])
model=TinyLM(ModelConfig(width=8))
logits=model(x[None])['logits'];logits.retain_grad();masked_loss(logits,y[None]).backward()
q_logits=float(logits.grad[0,2].norm());q_embedding=float(model.embedding.weight.grad[x[2]].norm())
assert q_logits==0 and q_embedding>0
out={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','seed':42,'x':x.tolist(),'y':y.tolist(),'question_logits_gradient_norm':q_logits,'question_embedding_gradient_norm':q_embedding,'unseen_B_embedding_gradient_norm':float(model.embedding.weight.grad[74].norm()),'logits_is_leaf':logits.is_leaf,'embedding_is_leaf':model.embedding.weight.is_leaf,'retain_grad_original_docstring':torch.Tensor.retain_grad.__doc__}
print(json.dumps(out,ensure_ascii=False,indent=2))
