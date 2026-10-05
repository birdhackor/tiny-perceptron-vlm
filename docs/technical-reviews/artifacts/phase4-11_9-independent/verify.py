"""Bounded CPU checks for lesson 11.9; no model loading or training."""
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import PIL
import torch
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.multimodal import scene
from tiny_perceptron.data import ByteTokenizer
from scripts.course_experiments.modalities import _media

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

geometry = []
for offset in (7, 0, 12):
    image = scene('red', 'square', offset=offset)
    crop = image[:, 4:12, 4:12]
    positive = torch.nonzero(image[0] > 0)
    geometry.append(dict(offset=offset, original_shape=list(image.shape),
                         crop_shape=list(crop.shape), original_red=int((image[0]>0).sum()),
                         crop_red=int((crop[0]>0).sum()),
                         original_positive_coordinates=positive.tolist(),
                         crop_positive_coordinates=torch.nonzero(crop[0]>0).tolist()))
assert [(g['original_red'],g['crop_red']) for g in geometry] == [(45,8),(81,64),(0,0)]
assert list(range(16))[4:12] == list(range(4,12))

# Different originals with the same crop cannot be distinguished by padding or resizing.
a = scene('red','square',offset=7)
b = a.clone()
b[:,0,0] = torch.tensor([0.,0.,1.])
ca,cb = a[:,4:12,4:12],b[:,4:12,4:12]
assert not torch.equal(a,b) and torch.equal(ca,cb)
assert torch.equal(torch.nn.functional.pad(ca,(4,4,4,4)),torch.nn.functional.pad(cb,(4,4,4,4)))
def pil(tensor):
    return Image.fromarray((tensor.permute(1,2,0).numpy()*255).astype('uint8'))
assert np.array_equal(np.array(pil(ca).resize((16,16))),np.array(pil(cb).resize((16,16))))
small=np.zeros((2,2,3),dtype=np.uint8)
small[0,0,0]=255
shrunk=np.array(Image.fromarray(small).resize((1,1),Image.Resampling.BOX))
assert shrunk[0,0].tolist() == [64,0,0]

path=ROOT/'docs/course-experiments/results/vision_ablation.json'
raw=path.read_bytes(); data=json.loads(raw)
# Never inspect notes, crop_note, review, or author scope-correction fields.
pointers=['/revision','/device','/seed','/torch_version','/python_version',
          '/code_sha256/tiny_perceptron~1multimodal.py',
          '/code_sha256/scripts~1course_experiments~1modalities.py',
          '/results/data/seed','/results/data/split_policy','/results/data/splits/test']
selected={}
def pointer(ptr):
    v=data
    for part in ptr.lstrip('/').split('/'):
        part=part.replace('~1','/').replace('~0','~')
        v=v[int(part)] if isinstance(v,list) else v[part]
    return v
for ptr in pointers:selected[ptr]=pointer(ptr)
for name in ('none','edge_cropped'):
    for key in ('examples','correct','exact_match','effective_tokens','eos_rate',
                'generation_errors','invalid_special_tokens','ablation','skipped','samples','groups'):
        ptr=f'/results/interventions/{name}/{key}'
        pointers.append(ptr); selected[ptr]=pointer(ptr)
rows=pointer('/results/data/splits/test/records')
tok=ByteTokenizer()
for name in ('none','edge_cropped'):
    samples=pointer(f'/results/interventions/{name}/samples')
    count=0; groups={}
    for row,s in zip(rows,samples,strict=True):
        ids=s['generated_ids']
        answer=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        exact=answer==tok.encode(row['answer'])
        assert s['target']==row['answer'] and s['exact_match']==exact
        assert s['generated']==tok.decode(answer)
        count+=exact
        group=groups.setdefault(row['question'],dict(correct=0,count=0))
        group['count']+=1;group['correct']+=exact
    assert count==9 and groups=={'shape?':{'correct':3,'count':6},'color?':{'correct':6,'count':6}}
    assert all(s['generated']=='circle' for s in samples if s['question']=='shape?')
    assert len(samples)==pointer(f'/results/interventions/{name}/examples')==12
    assert count/len(samples)==pointer(f'/results/interventions/{name}/exact_match')==.75
    assert sum(len(tok.encode(r['answer']))+1 for r in rows)==72

crop_receipts=[]
for i,row in enumerate(rows):
    image,_=_media(dict(row,offset=7),SimpleNamespace(device='cpu'))
    crop=image[:,4:12,4:12]
    target=OUT/f'recreated-crop-{i}.png'
    pil(crop).resize((16,16)).save(target)
    loaded,_=_media(dict(row,image=str(target)),SimpleNamespace(device='cpu'))
    artifact=next((a for a in data['artifacts'] if a['path']==f'crop-{i}.png'))
    ptr=f'/artifacts/{data["artifacts"].index(artifact)}'
    pointers.append(ptr);selected[ptr]=artifact
    crop_receipts.append(dict(row=i,color=row['color'],shape=row['shape'],
        active_before=int((image>0).any(dim=0).sum()),active_crop=int((crop>0).any(dim=0).sum()),
        loaded_shape=list(loaded.shape),pixel_max=loaded.amax(dim=(1,2)).tolist(),
        png_sha256=digest(target.read_bytes()),historical_sha256=artifact['sha256'],
        exact_historical_png_match=digest(target.read_bytes())==artifact['sha256']))
assert all(c['active_crop']>0 for c in crop_receipts)

# Independently compare all colored grid cells and crop rectangle to actual tensors.
svg=ET.fromstring((OUT/'crop-evidence.svg').read_bytes())
rects=svg.findall('{http://www.w3.org/2000/svg}rect')
grid=[]
for origin,scale,size,tensor in [((53,129),14,16,a),((398,183),22,8,ca)]:
    cells=[r for r in rects if r.get('width')==str(scale) and r.get('height')==str(scale)]
    assert len(cells)==size*size
    for r in cells:
        x=(float(r.get('x'))-origin[0])/scale;y=(float(r.get('y'))-origin[1])/scale
        assert x==int(x) and y==int(y) and 0<=x<size and 0<=y<size
        expected='#ff0000' if tensor[0,int(y),int(x)]>0 else '#000000'
        assert r.get('fill')==expected
    grid.append(dict(size=size,cells=len(cells),red=sum(r.get('fill')=='#ff0000' for r in cells)))
frame=next(r for r in rects if r.get('fill')=='none')
assert {k:frame.get(k) for k in ('x','y','width','height')}=={'x':'109','y':'185','width':'112','height':'112'}

# Exact prose/code comparison with the served page, excluding heading permalink/editor UI.
soup=BeautifulSoup((OUT/'page-11_9.html').read_text(),'html.parser')
article=soup.find('article')
for el in article.select('.headerlink, .md-content__button'):el.decompose()
paragraphs=[p.get_text(' ',strip=True) for p in article.find_all('p') if not p.find('img')]
source=(OUT/'section.md').read_text()
expected_paragraphs=[]
inside=False
for block in re.sub(r'```[\s\S]*?```', '', source).split('\n\n'):
    if block.startswith('```'):continue
    if block.startswith(('##','![','<','可回顧：','原實驗')):continue
    if not block.strip():continue
    expected_paragraphs.append(block.replace('`','').replace('\n',' '))
normalized=lambda s:re.sub(r'\s+','',s)
actual_normalized=normalized(' '.join(paragraphs))
assert all(normalized(p) in actual_normalized for p in expected_paragraphs)
assert article.find('code').get_text().strip()==(OUT/'fence-1.py').read_text().strip()
assert article.find('img')['alt']=='scene位移7的原始紅方塊與中心裁切，紅像素由45剩8'

write('raw-pointers.json',dict(original_path=str(path.relative_to(ROOT)),original_sha256=digest(raw),
    inspected_pointers=pointers,selected_raw_values=selected,
    exclusions='No notes/review/crop_note/patch_budget or scope-correction values were inspected.'))
write('verification-results.json',dict(environment=dict(python=sys.version,torch=torch.__version__,
    torch_git=torch.version.git_version,cuda_build=torch.version.cuda,device='cpu',pillow=PIL.__version__),
    geometry=geometry,noninjective_crop=dict(originals_different=True,crops_equal=True,padded_equal=True,resized_equal=True),
    downsample_BOX=dict(input_red_pixel=255,input_pixels=4,output_red_pixel=64),
    recorded_scores=dict(none='9/12',edge_cropped='9/12',shape='3/6, all circle',color='6/6'),
    crop_receipts=crop_receipts,svg_grids=grid,served_page_prose_and_fence_match=True,
    scope='Existing measurement validation and reconstruction of preprocessing inputs only. No model evaluation, training, downloads of model/data, GPU use, or neural checkpoints.'))
print('CPU geometry: offset7 45->8, offset0 81->64, offset12 0->0; C,H,W and [4,12) axes checked')
print('Crop non-injective pair: equal crop, equal padded image, equal resized image; BOX 2x2->1x1 red 255->64')
print('Raw samples: none/edge_cropped each 9/12; color 6/6; shape 3/6 with all circle; 72 effective answer+EOS targets')
print('Historical PNG hashes matched',sum(c['exact_historical_png_match'] for c in crop_receipts),'of',len(crop_receipts),'; all reconstructed inputs retain color')
print('SVG cell matrices and yellow [4:12,4:12] frame match; served page prose/fence match frozen 11.9')
