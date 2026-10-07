import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
from tiny_perceptron.multimodal import scene

image = scene("red", "square", offset=7)
crop = image[:, 4:12, 4:12]
print("原圖形狀", tuple(image.shape), "裁切形狀", tuple(crop.shape))
print("原圖紅色格", int((image[0] > 0).sum()))
print("裁切紅色格", int((crop[0] > 0).sum()))


# Owner supplied proportional check
import pathlib
center=scene('red','square',offset=0);print('offset0',int((center[0]>0).sum()),int((center[:,4:12,4:12][0]>0).sum()))
print('wider offset7',int((image[:,2:14,2:14][0]>0).sum()))
r=json.loads(pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vision-ablation-raw.json').read_text())['results']['interventions']
for n in ['none','edge_cropped']:
 s=r[n]['samples'];print('history crop',n,sum(x['generated']==x['target'] for x in s),len(s),'shape outputs',[x['generated'] for x in s if x['question']=='shape?'])

