import hashlib
import json
import sys
from pathlib import Path

p=Path('docs/technical-reviews/artifacts/p7_technical_c/originals/encoders-raw.json')
d=json.loads(p.read_text());v=d['results']['vision']
print('python',sys.version.split()[0],'device CPU parsing-only')
print('raw_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
print('historical_revision',d['revision'],'historical_device',d['device'],'historical_torch',d['torch_version'])
print('config',json.dumps(v['config']))
classes=v['classes'];print('classes',json.dumps(classes),'count',len(classes))
print('class Cartesian complete',set(classes)=={f'{c} {s}' for c in ('red','green','blue') for s in ('square','circle')})
print('training',json.dumps({k:v['training'][k] for k in ('steps','weights_changed','nonzero_gradient_seen','initial_loss','final_loss','effective_targets','trainable_parameters')}))
print('splits',json.dumps({k:{'count_field':x['count'],'actual_records':len(x['records']),'offsets':sorted({r['offset'] for r in x['records']})} for k,x in v['data']['splits'].items()}))
