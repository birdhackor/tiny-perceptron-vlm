import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene

torch.manual_seed(0)
tok = ByteTokenizer()
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
model.requires_grad_(False)
model.image_projector.requires_grad_(True)
answer = tok.encode("red square") + [tok.eos_id]
prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.eos_id, tok.assistant_id]
ids = torch.tensor(prefix + answer)
labels = torch.tensor([-100] * len(prefix) + answer)
out = model(ids, labels, image=scene())
loss = masked_loss(out["logits"], out["labels"])
loss.backward()
print("接頭收到梯度", model.image_projector.weight.grad.norm().item() > 0)
print("語言權重未累積梯度", model.language.embedding.weight.grad is None)


# Owner supplied proportional check
import pathlib,hashlib
print('targets',int((out['labels']!=-100).sum()),'shapes',tuple(out['logits'].shape),tuple(out['labels'].shape))
frozen=MultiModalLM(TinyLM(ModelConfig(width=8))); frozen.requires_grad_(False)
o=frozen(ids,labels,image=scene()); l=masked_loss(o['logits'],o['labels']); print('all frozen',sum(p.numel() for p in frozen.parameters() if p.requires_grad),l.requires_grad)
try:l.backward()
except RuntimeError as e:print('expected backward error',str(e))
late=MultiModalLM(TinyLM(ModelConfig(width=8)));late.requires_grad_(False);late.image_projector.requires_grad_(True)
o=late(ids,labels,image=scene());late.requires_grad_(False);l=masked_loss(o['logits'],o['labels']);print('late loss grad',l.requires_grad);l.backward();print('late backward completed')
p=pathlib.Path('docs/course-experiments/results/projector.json');raw=p.read_bytes();pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/projector-raw.json').write_bytes(raw);r=json.loads(raw)['results'];print('projector rawsha',hashlib.sha256(raw).hexdigest());print('history train',{k:r['training'][k] for k in ['steps','weights_changed','nonzero_gradient_seen','initial_loss','final_loss']})
s=r['test']['samples'];print('test recount',sum(x['generated']==x['target'] for x in s),len(s),'eos',sum(x['eos'] for x in s));print('targets/generated',[(x['target'],x['generated']) for x in s])

