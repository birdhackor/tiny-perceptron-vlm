import json
from pathlib import Path
print('mask-counts',int((out['labels']!=-100).sum()),int((out['labels']==-100).sum()),'gradnorm',float(model.audio_projector.weight.grad.norm()))
a=tok.encode('low')+[tok.eos_id];z=torch.tensor(prefix+a);y=torch.tensor([-100]*len(prefix)+a);o=model(z,y,waveform=tone(200));print('low-shape-targets',tuple(o['logits'].shape),int((o['labels']!=-100).sum()))
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_c/originals/audio-raw.json').read_text())['results'];t=r['test'];c=sum(x['target']==x['generated'] for x in t['samples']);print('historical-test',c,len(t['samples']),'eos',sum(x['eos'] for x in t['samples']));print('wrong-families',[x['family'] for x in t['samples'] if x['target']!=x['generated']]);assert c==11 and len(t['samples'])==14
h=r['training']['history'];print('historical-updates',len(h),'effective-targets',sum(x['effective_targets'] for x in h),'nonzero-grad-steps',sum(x['grad_norm']>0 for x in h),'weights-changed',r['training']['weights_changed'])
