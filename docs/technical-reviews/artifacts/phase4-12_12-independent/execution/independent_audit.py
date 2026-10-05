"""Bounded CPU routing checks and arithmetic over immutable raw measurements.

Does not load a checkpoint, run generation on a trained model, or train anything.
Only explicitly named raw JSON fields are consumed; author scope prose is excluded.
"""
import collections
import hashlib
import json
import pathlib
import platform
import sys
import xml.etree.ElementTree as ET

import torch
from torch.nn import functional as F
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene, tone

ROOT = pathlib.Path(__file__).resolve().parents[5]
OUT = pathlib.Path(__file__).parent
torch.set_num_threads(1)
torch.set_default_device('cpu')
tok = ByteTokenizer()
environment = {'python': platform.python_version(), 'python_executable': sys.executable,
               'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version),
               'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()),
               'cuda_build': str(torch.version.cuda), 'num_threads': '1'}
assert torch.version.cuda is None and not torch.cuda.is_available()

routes = []
for detach in ((), ('image',), ('audio',), ('image', 'audio')):
    torch.manual_seed(0)
    model = MultiModalLM(TinyLM(ModelConfig(width=8)))
    before = {name: p.detach().clone() for name,p in model.named_parameters()}
    hooks = [getattr(model, name+'_projector').register_forward_hook(lambda mod,args,out:out.detach())
             for name in detach]
    prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.audio_id, tok.eos_id, tok.assistant_id]
    answer = tok.encode('circle,high') + [tok.eos_id]
    ids = torch.tensor(prefix+answer)
    labels = torch.tensor([-100]*len(prefix)+answer)
    out = model(ids,labels,image=scene('red','circle'),waveform=tone(440))
    valid = out['labels'] != -100
    manual = F.cross_entropy(out['logits'][valid],out['labels'][valid],reduction='sum')/valid.sum()
    loss = masked_loss(out['logits'],out['labels'])
    # Masked-vs-full reduction may differ by one float32 rounding unit.
    assert torch.allclose(loss,manual,atol=1e-6,rtol=0)
    loss.backward()
    grads = {name: None if getattr(model,name+'_projector').weight.grad is None else
             float(getattr(model,name+'_projector').weight.grad.norm()) for name in ('image','audio')}
    assert all((grads[name] is None) if name in detach else grads[name]>0 for name in grads)
    unchanged = all(torch.equal(before[name],p.detach()) for name,p in model.named_parameters())
    assert unchanged
    assert tuple(out['logits'].shape)==(1,42,264) and int(valid.sum())==12
    assert out['labels'][valid].tolist()==answer
    routes.append({'detached_branches':list(detach),'projector_gradient_norms':grads,
                   'parameters_unchanged':unchanged,'logits_shape':list(out['logits'].shape),
                   'effective_target_tokens':int(valid.sum()),'loss':float(loss.detach()),
                   'manual_loss_error':float((loss-manual).abs().detach())})
    for h in hooks:h.remove()

image = scene('red','circle')
wave = tone(440)
peak = float(torch.fft.rfftfreq(len(wave),1/16000)[torch.fft.rfft(wave).abs().argmax()])
assert peak==440 and len(wave)==1600
ns = {'s':'http://www.w3.org/2000/svg'}
svg = ET.parse(ROOT/'course/figures/rewrite-12-joint-clues.svg').getroot()
pixels = [r for r in svg.findall('s:rect',ns) if r.attrib.get('width')=='12']
assert len(pixels)==256
for r in pixels:
    x=(int(r.attrib['x'])-58)//12;y=(int(r.attrib['y'])-86)//12
    assert r.attrib['fill']==('#ff0000' if image[0,y,x] else '#000000')

raw_path = ROOT/'docs/course-experiments/results/joint.json'
j = json.loads(raw_path.read_bytes())
splits = j['results']['data']['splits']
balance = {}
for split,obj in splits.items():
    rows=obj['records'];assert len(rows)==obj['count']
    assert hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==obj['sha256']
    pairs=collections.Counter((r['shape'],'high' if r['frequency']>300 else 'low') for r in rows)
    assert len(pairs)==4 and len(set(pairs.values()))==1
    assert all(r['answer']==r['shape']+','+('high' if r['frequency']>300 else 'low') for r in rows)
    balance[split]={'count':len(rows),'shape_pitch_counts':{','.join(k):v for k,v in pairs.items()},
                    'frequencies_hz':sorted({r['frequency'] for r in rows}),
                    'offsets':sorted({r['offset'] for r in rows})}

measurements = {}
for name,ev in [('validation',j['results']['validation']),*j['results']['evaluations'].items()]:
    rows = ev['samples'];den=len(rows);assert den==12==ev['examples']
    shape=pitch=exact=eos=0
    for s in rows:
        ids=s['generated_ids'];content=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        assert tok.decode(content)==s['generated']
        same=content==tok.encode(s['target']);assert same==s['exact_match'];exact+=same
        parts=s['generated'].split(',');targets=s['target'].split(',')
        shape+=len(parts)==2 and parts[0]==targets[0]
        pitch+=len(parts)==2 and parts[1]==targets[1]
        eos+=tok.eos_id in ids
        assert s['generation_error'] is None and s['invalid_special_tokens']==0
    effective=sum(len(tok.encode(s['target']))+1 for s in rows)
    assert (shape,pitch,exact,effective)==(ev['shape_correct'],ev['pitch_correct'],ev['correct'],ev['effective_tokens'])
    assert exact/den==ev['exact_match'] and eos/den==ev['eos_rate']
    measurements[name]={'shape':shape,'pitch':pitch,'exact':exact,'denominator':den,
                        'effective_target_tokens':effective,'ended_eos':eos}
assert [(measurements[n]['shape'],measurements[n]['pitch'],measurements[n]['exact'])
        for n in ('none','blank_image','blank_audio','blank')]==[(12,12,12),(6,12,6),(12,6,6),(6,6,3)]
base=j['results']['evaluations']['none']['samples']
pair_checks={}
for name,changed_part in [('swap_image',0),('swap_audio',1)]:
    changed=j['results']['evaluations'][name]['samples'];count=0
    for original,swapped in zip(base,changed,strict=True):
        a=original['generated'].split(',');b=swapped['generated'].split(',')
        assert original['row']==swapped['row'] and original['family']==swapped['family']
        assert a[changed_part]!=b[changed_part] and a[1-changed_part]==b[1-changed_part]
        assert swapped['generated']==swapped['target'] and original['generated']==original['target']
        count+=1
    pair_checks[name]={'paired_endpoints_correct':count,'changed_part_only':count,'pairs':12}

encoder_path=ROOT/'docs/course-experiments/results/encoders.json'
e=json.loads(encoder_path.read_bytes())
pretrain_freq={r['frequency'] for r in e['results']['audio']['data']['splits']['train']['records']}
overlap={name: sorted(pretrain_freq & {r['frequency'] for r in obj['records']}) for name,obj in splits.items()}
assert overlap['test']==[260,340] and measurements['validation']['exact']==8
code_matches={fn:hashlib.sha256((ROOT/fn).read_bytes()).hexdigest()==j['code_sha256'][fn]
              for fn in ['tiny_perceptron/multimodal.py','scripts/course_experiments/modalities.py']}
assert all(code_matches.values())
result={'environment':environment,'routing_variants':routes,'material_check':{'image_shape':list(image.shape),
        'red_pixels':int((image[0]>0).sum()),'svg_pixels_match_scene':True,
        'waveform_samples':len(wave),'sample_rate_hz':16000,'duration_seconds':0.1,'fft_peak_hz':peak},
        'data_balance':balance,'measurements':measurements,'swap_pair_checks':pair_checks,
        'encoder_training_frequency_overlap_hz':overlap,'original_code_hashes_match':code_matches,
        'raw_files_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw_path,encoder_path)},
        'limits':'Fresh random-model gradient checks and arithmetic over recorded samples only; no training or checkpoint reevaluation.'}
(OUT/'independent-audit-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
