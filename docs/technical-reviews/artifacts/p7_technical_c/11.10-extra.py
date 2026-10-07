import pathlib
for p in [8,4,2]:
 v=(32//p)**2;t=20+v;print('edge32',p,v,t,t*t)
r=json.loads(pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vision-ablation-raw.json').read_text())['results']['patch_variants']
for n,v in r.items():
 s=v['test']['samples'];print('historical patch',n,'tokens',v['visual_tokens'],'parameters',v['training']['parameters'],'correct',sum(x['generated']==x['target'] for x in s),'den',len(s))
