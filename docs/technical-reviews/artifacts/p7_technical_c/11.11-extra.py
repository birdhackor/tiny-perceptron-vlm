import pathlib
print('rare20',90/(90+20),(1+0)/2,'other model',(81+9)/(90+10),(81/90+9/10)/2)
s=json.loads(pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vision-ablation-raw.json').read_text())['results']['interventions']['none']['samples']
for target in ['circle','square']:
 t=[x for x in s if x['question']=='shape?' and x['target']==target];print('shape subgroup',target,sum(x['generated']==x['target'] for x in t),len(t))
