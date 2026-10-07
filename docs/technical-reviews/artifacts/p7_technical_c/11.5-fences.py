import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

model = MultiModalLM(TinyLM(ModelConfig(width=8)))
for mode in ["projector", "partial", "all"]:
    model.requires_grad_(mode == "all")
    model.image_projector.requires_grad_(True)
    if mode == "partial":
        model.language.blocks[-1].requires_grad_(True)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(mode, count)


# Owner supplied proportional check
import pathlib,hashlib
m2=MultiModalLM(TinyLM(ModelConfig(width=8,layers=2)))
for mode in ['projector','partial','all']:
 m2.requires_grad_(mode=='all');m2.image_projector.requires_grad_(True)
 if mode=='partial':m2.language.blocks[-1].requires_grad_(True)
 print('layers2',mode,sum(p.numel() for p in m2.parameters() if p.requires_grad))
p=pathlib.Path('docs/course-experiments/results/vqa.json');raw=p.read_bytes();pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vqa-raw.json').write_bytes(raw);print('vqa rawsha',hashlib.sha256(raw).hexdigest());r=json.loads(raw)['results']
for n in ['projector_only','partial','all']:
 v=r['variants'][n];print('historical',n,'trainable',v['training']['trainable_parameters'],'effective_targets',v['training']['effective_targets'])
 for split in ['validation','test']:
  x=v[split];c=sum(s['generated']==s['target'] for s in x['samples']);print('recount',n,split,c,len(x['samples']));assert c==x['correct']

