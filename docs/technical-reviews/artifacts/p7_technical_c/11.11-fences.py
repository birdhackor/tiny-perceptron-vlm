import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
groups = {"常見": {"correct": 90, "count": 90}, "稀少": {"correct": 0, "count": 10}}
micro = sum(g["correct"] for g in groups.values()) / sum(g["count"] for g in groups.values())
macro = sum(g["correct"] / g["count"] for g in groups.values()) / len(groups)
for name, group in groups.items():
    print(name, "答對/總數", group["correct"], group["count"])
print("按題平均", micro, "按組平均", macro)


# Owner supplied proportional check
import pathlib
print('rare20',90/(90+20),(1+0)/2,'other model',(81+9)/(90+10),(81/90+9/10)/2)
s=json.loads(pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vision-ablation-raw.json').read_text())['results']['interventions']['none']['samples']
for target in ['circle','square']:
 t=[x for x in s if x['question']=='shape?' and x['target']==target];print('shape subgroup',target,sum(x['generated']==x['target'] for x in t),len(t))

