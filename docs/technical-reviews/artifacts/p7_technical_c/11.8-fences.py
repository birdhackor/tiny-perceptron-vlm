import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
truth = ["circle", "square", "circle", "square"]
text_only = ["circle"] * len(truth)
accuracy = sum(a == b for a, b in zip(truth, text_only)) / len(truth)
print("平衡資料的固定猜測", accuracy)
imbalanced = ["circle"] * 9 + ["square"]
print("九比一資料的固定猜測", imbalanced.count("circle") / len(imbalanced))


# Owner supplied proportional check
import pathlib,hashlib
p=pathlib.Path('docs/course-experiments/results/vision_ablation.json');raw=p.read_bytes();pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vision-ablation-raw.json').write_bytes(raw);print('rawsha',hashlib.sha256(raw).hexdigest());r=json.loads(raw)['results']['interventions'];base=r['none']['samples'];swap=r['shape_swap_relabelled']['samples']
shape=[i for i,x in enumerate(base) if x['question']=='shape?'];print('shape pairs',sum(base[i]['generated']==base[i]['target'] for i in shape),sum(swap[i]['generated']==swap[i]['target'] for i in shape),sum(base[i]['generated']==base[i]['target'] and swap[i]['generated']==swap[i]['target'] for i in shape),len(shape))
for n in ['none','blank','shuffle']:
 s=r[n]['samples'];print('recount',n,sum(x['generated']==x['target'] for x in s),len(s))
s=[x for x in r['shuffle']['samples'] if x['question']=='color?'];print('color donor new truth',[(x['target'],base[x['donor_row']]['target'],x['generated']) for x in s],sum(x['generated']==base[x['donor_row']]['target'] for x in s),len(s))

