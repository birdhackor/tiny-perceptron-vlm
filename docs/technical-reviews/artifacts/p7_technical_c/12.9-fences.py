import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, tone

torch.manual_seed(0)
tok = ByteTokenizer()
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
prefix = [tok.bos_id, tok.user_id, tok.audio_id, tok.eos_id, tok.assistant_id]
answer = tok.encode("high") + [tok.eos_id]
ids = torch.tensor(prefix + answer)
labels = torch.tensor([-100] * len(prefix) + answer)
out = model(ids, labels, waveform=tone(440))
loss = masked_loss(out["logits"], out["labels"])
loss.backward()
print("分數形狀", tuple(out["logits"].shape))
print("聲音接頭收到梯度", model.audio_projector.weight.grad.norm().item() > 0)


# Owner supplied proportional check
import json
from pathlib import Path
print('mask-counts',int((out['labels']!=-100).sum()),int((out['labels']==-100).sum()),'gradnorm',float(model.audio_projector.weight.grad.norm()))
a=tok.encode('low')+[tok.eos_id];z=torch.tensor(prefix+a);y=torch.tensor([-100]*len(prefix)+a);o=model(z,y,waveform=tone(200));print('low-shape-targets',tuple(o['logits'].shape),int((o['labels']!=-100).sum()))
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_c/originals/audio-raw.json').read_text())['results'];t=r['test'];c=sum(x['target']==x['generated'] for x in t['samples']);print('historical-test',c,len(t['samples']),'eos',sum(x['eos'] for x in t['samples']));print('wrong-families',[x['family'] for x in t['samples'] if x['target']!=x['generated']]);assert c==11 and len(t['samples'])==14
h=r['training']['history'];print('historical-updates',len(h),'effective-targets',sum(x['effective_targets'] for x in h),'nonzero-grad-steps',sum(x['grad_norm']>0 for x in h),'weights-changed',r['training']['weights_changed'])

