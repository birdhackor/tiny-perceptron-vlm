"""Independent CPU bookkeeping and interface checks, not retraining or model evaluation."""
import ast
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import modalities as m
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM, scene

torch.set_num_threads(1)
torch.manual_seed(73)
OUT = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
def jsonhash(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

original = ROOT/'docs/course-experiments/results/vision_ablation.json'
all_data = json.loads(original.read_bytes())
# Read only named original measurement/config/provenance pointers. Extra strings remain opaque.
pointers = ['/revision','/device','/seed','/torch_version','/python_version','/step_scale','/code_sha256']
selected = {p:all_data[p[1:]] for p in pointers}
for key in ['parameters','trainable_parameters','config','modal_config','steps','effective_tokens',
            'effective_targets','weights_changed','nonzero_gradient_seen','history']:
    p='/results/training/'+key; pointers.append(p); selected[p]=all_data['results']['training'][key]
for p in ['/results/data/seed','/results/data/split_policy','/results/data/splits']:
    v=all_data
    for key in p.split('/')[1:]: v=v[key]
    pointers.append(p);selected[p]=v
for name in ['none','blank','shuffle','shape_swap_relabelled']:
    for key in ['examples','correct','exact_match','effective_tokens','groups','samples']:
        p='/results/interventions/'+name+'/'+key
        pointers.append(p);selected[p]=all_data['results']['interventions'][name][key]
put('selected-raw-measurements.json', selected)

code_ranges = {
 'scripts/course_experiments/modalities.py': [(1,25),(37,38),(41,53),(73,133),(171,192),(219,283),(287,364),(367,384),(772,821)],
 'tiny_perceptron/multimodal.py': [(12,41),(84,174),(177,190)],
 'tiny_perceptron/model.py': [(15,28),(92,105)],
 'tiny_perceptron/data.py': [(14,28)],
}
inspected=[]; excerpts=[]
for file,ranges in code_ranges.items():
    raw=(ROOT/file).read_bytes(); lines=raw.decode().splitlines()
    assert sha(ROOT/file)==all_data['code_sha256'][file]
    tree=ast.parse(raw)
    inspected.append({'path':file,'sha256':sha(ROOT/file),'ranges':ranges,
                      'historical_code_sha256_matches':True,
                      'ast_function_locations':[
                          {'name':n.name,'first_line':n.lineno,'last_line':n.end_lineno}
                          for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]})
    for first,last in ranges:
        excerpts.append(f'FILE {file} lines {first}-{last}\n'+
                        '\n'.join(f'{i}: {lines[i-1]}' for i in range(first,last+1)))
(OUT/'inspected-code.txt').write_text('\n\n'.join(excerpts)+'\n')
put('inspection-receipt.json',{'source':str(original.relative_to(ROOT)),
    'original_json_sha256':sha(original),'json_pointers':pointers,'code_inspections':inspected,
    'unread_extra_result_explanations':'Opaque bytes retained; not used in inspection or verdict.'})

tok=ByteTokenizer(); ctx=SimpleNamespace(seed=42,device=torch.device('cpu'))
splits=m._vision_records(('shape?','color?'))
split_summary={}
for name,rows in splits.items():
    stored=all_data['results']['data']['splits'][name]
    assert rows==stored['records'] and len(rows)==stored['count']
    assert m._hash(rows)==stored['sha256']
    split_summary[name]={'questions':len(rows),'image_families':len({r['family'] for r in rows}),
                         'offsets':sorted({r['offset'] for r in rows}),
                         'questions_per_image':dict(Counter(r['family'] for r in rows))}
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not {r['family'] for r in splits[a]} & {r['family'] for r in splits[b]}
hist=selected['/results/training/history']
assert len(hist)==100 and [r['step'] for r in hist]==list(range(1,101))
assert sum(r['effective_targets'] for r in hist)==2370

truth=['circle','square','circle','square']
def accuracy(t,p): return sum(a==b for a,b in zip(t,p,strict=True))/len(t)
baseline={'balanced':accuracy(truth,['circle']*4),
          'nine_to_one':accuracy(['circle']*9+['square'],['circle']*10),
          'reversed_labels':accuracy(truth,['square']*4),
          'all_wrong':accuracy(['square']*4,['circle']*4)}
assert baseline=={'balanced':.5,'nine_to_one':.9,'reversed_labels':.5,'all_wrong':0.0}

summary={}; samples={}
for name in ['none','blank','shuffle','shape_swap_relabelled']:
    measurement=all_data['results']['interventions'][name]; ss=measurement['samples']; samples[name]=ss
    count=0; groups={}; answer_tokens=0
    for s in ss:
        ids=s['generated_ids']; raw=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        exact=raw==tok.encode(s['target'])
        assert exact==s['exact_match'] and tok.decode(raw)==s['generated']
        assert (tok.eos_id in ids)==s['eos']
        g=groups.setdefault(s['question'],{'correct':0,'count':0})
        g['count']+=1;g['correct']+=int(exact);count+=int(exact)
        answer_tokens+=len(tok.encode(s['target']))+1
    assert len(ss)==measurement['examples']==12 and count==measurement['correct']
    assert count/len(ss)==measurement['exact_match'] and groups==measurement['groups']
    assert answer_tokens==measurement['effective_tokens']==72
    summary[name]={'correct':count,'questions':len(ss),'groups':groups,'answer_bytes_plus_eos':answer_tokens}
original_shapes=[s for s in samples['none'] if s['question']=='shape?']
changed_shapes=[s for s in samples['shape_swap_relabelled'] if s['question']=='shape?']
shape_pairs=[]
for a,b in zip(original_shapes,changed_shapes,strict=True):
    assert a['row']==b['row'] and a['target']!=b['target']
    shape_pairs.append({'row':a['row'],'original_target':a['target'],'swapped_target':b['target'],
                        'original_prediction':a['generated'],'swapped_prediction':b['generated'],
                        'both_correct':a['exact_match'] and b['exact_match']})
assert sum(p['both_correct'] for p in shape_pairs)==0 and len(shape_pairs)==6
color_pairs=[]
for s in samples['shuffle']:
    row=splits['test'][s['row']]; donor=splits['test'][s['donor_row']]
    assert s['donor_row']==(s['row']+6)%12 and s['target']==row['answer']
    if s['question']=='color?':
        original_sample=samples['none'][s['row']]
        color_pairs.append({'row':s['row'],'donor_row':s['donor_row'],'original_target':s['target'],
                            'new_image_target':donor['color'],'prediction':s['generated'],
                            'correct_to_original':s['exact_match'],
                            'correct_to_new_image':s['generated']==donor['color'],
                            'both_original_and_new_correct':original_sample['exact_match'] and s['generated']==donor['color']})
assert sum(p['correct_to_original'] for p in color_pairs)==0
assert sum(p['correct_to_new_image'] for p in color_pairs)==6
assert sum(p['both_original_and_new_correct'] for p in color_pairs)==6

rows=splits['test']; swapped=[dict(r,shape='square' if r['shape']=='circle' else 'circle',
    answer=('square' if r['shape']=='circle' else 'circle') if r['question']=='shape?' else r['answer']) for r in rows]
for r,s in zip(rows,swapped,strict=True):
    assert r['question']==s['question'] and r['color']==s['color'] and r['offset']==s['offset']
    assert not torch.equal(m._media(r,ctx)[0],m._media(s,ctx)[0])
prefixes=[m._sequence(r,ctx,False)[0] for r in rows]
for i,r in enumerate(rows):
    assert torch.equal(prefixes[i],prefixes[0 if r['question']=='shape?' else 1])
    ids,labels,count=m._sequence(r,ctx)
    assert count==len(tok.encode(r['answer']))+1
    assert labels[labels!=-100].tolist()==tok.encode(r['answer'])+[tok.eos_id]
same_shape_other_color=dict(rows[0],color='blue')
assert same_shape_other_color['answer']==rows[0]['answer']=='circle'
assert not torch.equal(m._media(rows[0],ctx)[0],m._media(same_shape_other_color,ctx)[0])

# Real random model: forward/gradient only. This is not a trained capability test.
model=MultiModalLM(TinyLM(ModelConfig(width=16,layers=1,heads=1,max_length=128))).cpu()
ids,labels,_=m._sequence(rows[0],ctx)
first=m._media(rows[0],ctx)[0];second=m._media(swapped[0],ctx)[0]
result=model(ids,labels,image=first)
loss=m.masked_loss(result['logits'],result['labels']);loss.backward()
grad=float(model.image_projector.weight.grad.norm())
delta=float((result['logits'].detach()-model(ids,labels,image=second)['logits'].detach()).norm())
assert grad>0 and delta>0

# Run the original evaluator with an explicit renderer oracle for generated strings.
# This tests pixel intervention and target bookkeeping; the oracle has no learned weights.
original_generate=m.generate_modal;seen=[]
def renderer_oracle(model,prefix,image,waveform,max_new_tokens):
    seen.append(image.clone())
    question=tok.decode(prefix.tolist())
    active=(image.abs().sum(0)>0); area=int(active.sum())
    color=('red','green','blue')[int(image.sum((1,2)).argmax())]
    answer=('square' if area==81 else 'circle') if 'shape?' in question else color
    return torch.cat((prefix,prefix.new_tensor(tok.encode(answer)+[tok.eos_id])))
m.generate_modal=renderer_oracle
try:
    evaluator={}
    for mode in ['none','blank','shuffle']:
        seen.clear();v=m._evaluate(model,rows,ctx,mode)
        assert len(seen)==12
        for i,img in enumerate(seen):
            expected=torch.zeros_like(first) if mode=='blank' else m._media(rows[(i+6)%12] if mode=='shuffle' else rows[i],ctx)[0]
            assert torch.equal(img,expected)
        evaluator[mode]={'correct':v['correct'],'groups':v['groups']}
    v=m._evaluate(model,swapped,ctx)
    evaluator['shape_swap_relabelled']={'correct':v['correct'],'groups':v['groups']}
    assert evaluator['none']['correct']==12 and evaluator['shape_swap_relabelled']['correct']==12
    assert evaluator['blank']['correct']==5 and evaluator['shuffle']['correct']==0
finally:m.generate_modal=original_generate

# Original SVG uses representative offset=0 scenes; compare all 512 tile colors.
svg=ET.fromstring((ROOT/'course/figures/rewrite-11-shape-pairs.svg').read_bytes())
tiles=[n for n in svg if n.tag.endswith('rect') and n.get('width')=='13' and n.get('height')=='13']
assert len(tiles)==512
pixels={};svgs=[]
for origin,shape in [(54,'circle'),(373,'square')]:
    image=scene('red',shape,offset=0); count=0
    for n in tiles:
        x,y=int(n.get('x')),int(n.get('y'))
        if origin<=x<origin+208:
            col=(x-origin)//13;row=(y-127)//13
            expected='#ff0000' if float(image[:,row,col].sum()) else '#000000'
            assert n.get('fill')==expected
            count+=int(expected=='#ff0000')
    svgs.append({'shape':shape,'active_pixels':count,'tile_size_px':13,'grid_width_pixels':16,'offset':0})
put('verification-result.json',{
    'baseline_accuracy':baseline,'split_summary':split_summary,
    'historical_training':{'steps':len(hist),'effective_answer_targets':sum(r['effective_targets'] for r in hist),
       'device':selected['/device'],'seed':selected['/seed'],'reexecuted':False},
    'raw_measurement_recount':summary,'shape_directed_pairs':shape_pairs,'color_shuffle_pairs':color_pairs,
    'prefix_no_attribute_text':True,'unrelated_color_change_keeps_shape_target':True,
    'random_model_forward_only':{'logit_shape':list(result['logits'].shape),'projector_gradient_norm':grad,'shape_logit_delta_norm':delta,'optimizer_steps':0},
    'original_evaluator_with_renderer_oracle':evaluator,'svg_tiles_match_original_scene_rule':svgs,
    'limit':'No original model inference, checkpoint load/save, dataset download, optimization, GPU or capability certification.'})
put('probe-environment.json',{'python':sys.version,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),
    'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu','threads':torch.get_num_threads(),
    'command':'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-11_8-clean/verify.py',
    'probe_sha256':sha(Path(__file__))})
print(json.dumps({'status':'all_assertions_passed','baseline_accuracy':baseline,'historical_recounts':summary,
                  'shape_pairs_both_correct':'0/6','color_pairs_both_correct':'6/6',
                  'random_model_gradient_norm':grad,'random_model_logit_delta':delta,
                  'renderer_oracle_bookkeeping':evaluator,'svg_pixels':svgs},ensure_ascii=False))
