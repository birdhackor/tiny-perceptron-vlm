import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
height = 16
text_tokens = 20
for patch_size in [8, 4, 2]:
    visual_tokens = (height // patch_size) ** 2
    total = text_tokens + visual_tokens
    print("邊長", patch_size, "視覺位置", visual_tokens, "總長度", total, "注意力格數", total**2)


# Owner supplied proportional check
import pathlib
for p in [8,4,2]:
 v=(32//p)**2;t=20+v;print('edge32',p,v,t,t*t)
r=json.loads(pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vision-ablation-raw.json').read_text())['results']['patch_variants']
for n,v in r.items():
 s=v['test']['samples'];print('historical patch',n,'tokens',v['visual_tokens'],'parameters',v['training']['parameters'],'correct',sum(x['generated']==x['target'] for x in s),'den',len(s))

