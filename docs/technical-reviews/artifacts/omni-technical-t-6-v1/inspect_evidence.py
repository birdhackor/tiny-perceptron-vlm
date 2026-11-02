"""Inspect original historical evidence and official source contracts, without long experiments."""
from datetime import datetime, UTC
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess

import PIL.Image
import soundfile
import torch

from scripts.course_experiments import run

ROOT=Path(__file__).resolve().parents[4]
ART=Path(__file__).resolve().parent
CACHE=Path('/workspace/work/reference/omni-technical-t-6-v1')
def sha(data): return hashlib.sha256(data).hexdigest()
report={'created_at':datetime.now(UTC).isoformat(),'historical_runs':{},'official_source_runtime_comparison':{},
        'fixed_cpu_evidence_status_probes':{},'source_freeze':{}}

for name in ('encoders','projector','vqa'):
    path=ROOT/f'docs/course-experiments/results/{name}.json'
    d=json.loads(path.read_bytes())
    row={k:d[k] for k in ('experiment_id','revision','device','seed','torch_version','python_version','gpu',
                          'elapsed_seconds','timing_scope','step_scale','evidence_status','modal')}
    row['raw_report']={'path':str(path.relative_to(ROOT)),'sha256':sha(path.read_bytes())}
    row['round_1_decimal']=round(d['elapsed_seconds'],1)
    row['source_match']={}
    for p in ('scripts/course_experiments/modalities.py','scripts/course_experiments/run.py',
              'tiny_perceptron/multimodal.py','tiny_perceptron/training.py'):
        original=subprocess.check_output(['git','show',f"{d['revision']}:{p}"],cwd=ROOT)
        row['source_match'][p]={'historical_commit_sha256':sha(original),'recorded_sha256':d['code_sha256'][p],
                                'equals_recorded':sha(original)==d['code_sha256'][p],
                                'current_sha256':sha((ROOT/p).read_bytes())}
        assert row['source_match'][p]['equals_recorded']
    if name=='encoders':
        row['denominators']={}
        for modality,v in d['results'].items():
            row['denominators'][modality]={
                'split_counts':{k:x['count'] for k,x in v['data']['splits'].items()},
                'steps':v['training']['steps'],'effective_targets':v['training']['effective_targets'],
                'test_correct':v['test']['correct'],'test_count':v['test']['count'],
                'test_accuracy':v['test']['accuracy']}
    elif name=='projector':
        v=d['results']
        row['configuration']=v['training']['config']
        row['denominators']={'steps':v['training']['steps'],'effective_targets':v['training']['effective_targets'],
                             'split_counts':{k:x['count'] for k,x in v['data']['splits'].items()},
                             'test_correct':v['test']['correct'],'test_count':v['test']['examples']}
    else:
        v=d['results']
        row['configuration']=v['variants']['partial']['training']['config']
        row['denominators']={'split_counts':{k:x['count'] for k,x in v['data']['splits'].items()},
                             'text_split_counts':{k:x['count'] for k,x in v['text_data'].items()}}
        row['variants']={k:{'steps':x['training']['steps'],'effective_targets':x['training']['effective_targets'],
                            'test_correct':x['test']['correct'],'test_count':x['test']['examples'],
                            'text_before_matches':x.get('text_before',{}).get('matches'),
                            'text_after_matches':x.get('text_after',{}).get('matches'),
                            'text_count':x.get('text_before',{}).get('examples'),
                            'freeze_scope':x.get('freeze_scope')}
                         for k,x in v['variants'].items()}
    report['historical_runs'][name]=row

for cached,module in [('torch-module.py',torch.nn.modules.module),('torch-adam.py',importlib.import_module('torch.optim.adam')),
                       ('torch-adamw.py',importlib.import_module('torch.optim.adamw')),
                       ('torch-optimizer.py',importlib.import_module('torch.optim.optimizer')),
                       ('pillow-Image.py',PIL.Image),('soundfile.py',soundfile)]:
    local=Path(inspect.getfile(module))
    report['official_source_runtime_comparison'][cached]={
        'downloaded_sha256':sha((CACHE/cached).read_bytes()),'installed_sha256':sha(local.read_bytes()),
        'equal':(CACHE/cached).read_bytes()==local.read_bytes(),'installed_path':str(local)}

# Execute the runner's actual status selection with the experiment function
# replaced in memory by a no-training stub. This is exclusively a status-gate
# check and does not create purported empirical training evidence.
module=importlib.import_module('scripts.course_experiments.modalities')
for name in ('encoders','projector','vqa'):
    spec=run.experiment_spec(name)
    fn=getattr(module,spec['function'])
    setattr(module,spec['function'],lambda ctx:{'probe_only':'No training; test evidence_status gate only.'})
    try:
        d=run.execute(name,'cpu',CACHE/f'fixed-status-probe-{name}',CACHE/'unused-dependencies',ROOT/'assets/training')
    finally:
        setattr(module,spec['function'],fn)
    report['fixed_cpu_evidence_status_probes'][name]={
        'result':d['evidence_status'],'module':spec['module'],'dependencies':spec['dependencies'],
        'input':{'device':'cpu','step_scale':1.0,'experiment_function':'in-memory no-training stub'},
        'scope':'Only the real runner evidence-status condition was executed. No fixed experiment quality or timing was measured.'}
    assert d['evidence_status']=='interface_smoke_only'

freeze=ROOT/'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json'
section=ROOT/'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.6.md'
f=json.loads(freeze.read_bytes())
p=next(p for p in f['pages'] if p['page_id']=='T.6')
assert sha(section.read_bytes())==p['source_sha256']
report['source_freeze']={'path':str(freeze.relative_to(ROOT)),'sha256':sha(freeze.read_bytes()),'page':p}
report['scope']='Historical raw reports and their original git commit read; no historical GPU timing reproduced. Current local teaching CPU run is separate.'
(ART/'evidence-inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'historical_rounding':{k:v['round_1_decimal'] for k,v in report['historical_runs'].items()},
                  'historical_code_matches':True,'cpu_status_probes':report['fixed_cpu_evidence_status_probes'],
                  'official_equal':{k:v['equal'] for k,v in report['official_source_runtime_comparison'].items()}},ensure_ascii=False))
