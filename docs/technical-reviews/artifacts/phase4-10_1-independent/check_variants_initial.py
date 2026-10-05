"""Bounded CPU checks of 10.1 claims, executed independently of original fence."""
from pathlib import Path
import sys, json, hashlib, xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from tiny_perceptron.multimodal import scene
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
red=scene('red','square');blue=scene('blue','square');before=red.clone()
assert red.shape==blue.shape==(3,16,16)
assert red[:,8,8].tolist()==[1.,0.,0.] and red[:,0,0].tolist()==[0.,0.,0.]
assert blue[:,8,8].tolist()==[0.,0.,1.]
vals=red[:,8,8].tolist();vals[0]=0.
assert torch.equal(red,before)
batch=red.unsqueeze(0)
assert batch.shape==(1,3,16,16) and batch.numel()==red.numel()==768
assert batch.data_ptr()==red.data_ptr()
# An asymmetric sample makes row/column and channel permutations observable.
chw=torch.arange(3)[:,None,None]*100+torch.arange(2)[None,:,None]*10+torch.arange(4)[None,None,:]
assert chw[:,1,3].tolist()==[13,113,213]
hwc=chw.permute(1,2,0)
recovered=hwc.permute(2,0,1)
assert torch.equal(chw,recovered)
wrong=hwc.reshape(3,2,4)
assert not torch.equal(chw,wrong)
# Check every uint8 value, including the exact synthetic endpoint values.
u8=torch.arange(256,dtype=torch.uint8)
scaled=u8.to(torch.float32)/255
expected=torch.arange(256,dtype=torch.float64)/255
error=(scaled.double()-expected).abs().max().item()
assert error<1e-7
assert scaled[0].item()==0 and scaled[255].item()==1
assert torch.equal((red*255).to(torch.uint8).float()/255,red)
assert (red/255).max().item()<0.004
# Pixel-for-pixel comparison of original SVG fill colours to original scene.
svg=ROOT/'course/figures/rewrite-10-pixels.svg'
root=ET.fromstring(svg.read_bytes());ns={'s':'http://www.w3.org/2000/svg'}
rects=[r for r in root.findall('s:rect',ns) if r.attrib.get('width')=='14' and r.attrib.get('height')=='14']
assert len(rects)==256
coords=[];red_coords=[]
for rect in rects:
 a=rect.attrib;x=(int(a['x'])-64)//14;y=(int(a['y'])-82)//14
 assert int(a['x'])==64+14*x and int(a['y'])==82+14*y
 assert 0<=x<16 and 0<=y<16
 coords.append((y,x))
 rgb=red[:,y,x].tolist();expected_fill='#'+''.join(f'{round(v*255):02x}' for v in rgb)
 assert a['fill']==expected_fill,(y,x,a['fill'],expected_fill)
 if rgb==[1.,0.,0.]:red_coords.append((y,x))
assert len(set(coords))==256 and len(red_coords)==81
assert set(red_coords)=={(y,x) for y in range(4,13) for x in range(4,13)}
labels=[''.join(n.itertext()) for n in root.findall('s:text',ns)]
assert '左上 (0,0)：[0,0,0]' in labels and '中央 (8,8)：[1,0,0]' in labels
highlights=[r.attrib for r in root.findall('s:rect',ns) if r.attrib.get('stroke')=='#ffcf00']
# Some original SVG highlights use path; retain all raw coordinates for inspection.
marks=[n.attrib for n in root if n.attrib.get('stroke') in ['#ffcf00','#ffd447','#ffdc00','#ffdf00','#ffd23f']]
arr=(red.permute(1,2,0).numpy()*255).astype('uint8')
Image.fromarray(arr).resize((320,320),resample=Image.Resampling.NEAREST).save(OUT/'scene-render.png')
result={
 'device':'cpu','shape':list(red.shape),'red_center':red[:,8,8].tolist(),'red_corner':red[:,0,0].tolist(),
 'blue_shape':list(blue.shape),'blue_center':blue[:,8,8].tolist(),
 'tolist_nonmutating':torch.equal(red,before),'batch_shape':list(batch.shape),'element_count_before_after':[red.numel(),batch.numel()],
 'unsqueeze_shared_storage':batch.data_ptr()==red.data_ptr(),
 'asymmetric_chw_shape':list(chw.shape),'asymmetric_selection_c_y1_x3':chw[:,1,3].tolist(),
 'hwc_shape':list(hwc.shape),'permutation_recovers_all_elements':torch.equal(chw,recovered),
 'reshape_wrong_element_count':int((chw!=wrong).sum()),
 'uint8_denominator':255,'all_256_uint8_max_float_error':error,'scene_values':[red.min().item(),red.max().item()],
 'incorrect_double_division_max':(red/255).max().item(),
 'svg_grid_pixels':len(rects),'svg_scene_exact_matches':len(rects),'red_pixels':len(red_coords),'black_pixels':256-len(red_coords),
 'red_bounds_inclusive':{'row':[4,12],'column':[4,12]},'svg_labels':labels,'highlight_attributes':highlights,'mark_attributes':marks,
 'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git':str(torch.version.git_version),'cuda_build':str(torch.version.cuda)},
 'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'tiny_perceptron/multimodal.py',svg]},
 'scope':'Data generation, indexing, axis semantics, RGB encoding, numeric rescaling and original SVG match only; no training or inference.'
}
(OUT/'variant-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
