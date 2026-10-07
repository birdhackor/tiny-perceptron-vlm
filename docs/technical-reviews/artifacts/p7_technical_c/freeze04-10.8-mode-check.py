import json,sys,torch
from tiny_perceptron.model import TinyLM,ModelConfig
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'CPU'}))
torch.manual_seed(0)
lm=TinyLM(ModelConfig(width=8))
ids=torch.tensor([[1,20,30]])
lm.eval()
eval_output=lm(ids)['logits']
lm.train()
train_output=lm(ids)['logits']
print('eval versus train exact equal',torch.equal(eval_output,train_output))
print('shape',tuple(train_output.shape))
