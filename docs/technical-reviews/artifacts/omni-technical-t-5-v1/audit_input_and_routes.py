"""Check pinned PKU inputs, literal T.5 command routing and published local inventories; no training."""
from collections import Counter
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import platform
import re
import shlex
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.course_experiments.behavior import _conversation
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import _digest, _utf8_prefix
from scripts.train import parser

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
source=Path('/workspace/work/reference/omni-technical-t-5-v1/pku-first100-response.json')
response=json.loads(source.read_text())
raw=[x['row'] for x in response['rows']]
original_jsonl=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw).encode()
records=[]
for index,row in enumerate(raw):
    safer=int(row['safer_response_id'])
    if not row[f'is_response_{safer}_safe']:continue
    records.append(_conversation(_utf8_prefix(row['prompt'],120),_utf8_prefix(row[f'response_{safer}'],120),
        _digest(row['prompt']),source_row=index,source_record_sha256=_digest(row),safer_response_id=safer,
        better_response_id=int(row['better_response_id'])))
parts=split_records(records,seed=42)
historical=json.loads((ROOT/'docs/course-experiments/results/safety.json').read_text())['results']['pku_pilot']
splits={}
for name,rows in parts.items():
    encoded=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode()
    digest=hashlib.sha256(encoded).hexdigest()
    splits[name]={'records':len(rows),'families':len({r['family'] for r in rows}),
       'sha256':digest,'matches_historical':digest==historical['data'][name]['sha256'],
       'source_indices':[r['source_row'] for r in rows],
       'max_prompt_bytes':max(len(r['messages'][0]['content'].encode()) for r in rows),
       'max_answer_bytes':max(len(r['messages'][1]['content'].encode()) for r in rows)}
families={name:{r['family'] for r in rows} for name,rows in parts.items()}
snapshot=ROOT/'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.5.md'
text=snapshot.read_text()
literal=[]
for block in re.findall(r'```bash\n(.*?)\n```',text,re.S):
    for line in block.splitlines():
        args=shlex.split(line)
        item={'literal':line,'argv':args}
        if args[1]=='scripts/train.py':
            parsed=parser().parse_args(args[2:])
            item['parsed']={k:str(v) if isinstance(v,Path) else v for k,v in vars(parsed).items()}
            item['execution_boundary']='One-update CPU variant in execution.json; complete 500-step route not run.'
        elif '--experiment' in args:
            item['execution_boundary']='CLI help/list-assets and source only; no fixed recipe run on CPU or CUDA.'
            name=args[args.index('--experiment')+1]
            item['function']={'style':'run_style','safety':'run_safety','lora':'run_lora'}[name]
            item['dependencies']={'style':[], 'safety':['style/content.pt','PKU archive'], 'lora':['style/content.pt']}[name]
        elif args[1]=='scripts/fetch_training_assets.py':
            item['execution_boundary']='CLI help and LFS/unpack source inspection; LFS pull and unpack not run.'
        else:
            item['execution_boundary']='CPU counterpart in execution.json; output paths isolated in /workspace/work.'
        literal.append(item)
public=json.loads((ROOT/'docs/course-experiments/public-models.json').read_text())
safety=next(x for x in public['models'] if x['id']=='safety')
code=['scripts/ingest_course_result.py','scripts/course_release.py','tiny_perceptron/assets.py',
      'scripts/course_experiments/compression.py']
result={'created_at':datetime.now(UTC).isoformat(),
 'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},
 'pku':{'input':str(source),'response_sha256':sha(source),'revision':'9421ffafec3fa40a1f1a7d567b4d525079477ecb',
    'rows':len(raw),'raw_jsonl_sha256':hashlib.sha256(original_jsonl).hexdigest(),
    'raw_jsonl_matches_manifest':hashlib.sha256(original_jsonl).hexdigest()=='c4a88d08ef7456669766f1a3b908f82d918be5625a24794d9843d7f6df1d552b',
    'both_unsafe_pairs':sum(not r['is_response_0_safe'] and not r['is_response_1_safe'] for r in raw),
    'selected_safe_records':len(records),'split_audit':splits,
    'family_disjoint':all(not families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')]),
    'upstream_labels_apply_to_full_answers':'No semantic relabeling of 120-byte excerpts was performed; the lesson explicitly warns about context loss.'},
 'literal_commands':literal,
 'public_safety_registry':safety,
 'modal_recorded':{name:json.loads((ROOT/f'docs/course-experiments/results/{name}.json').read_text())['modal']
                   for name in ['style','safety','lora']},
 'source_sha256':{p:sha(ROOT/p) for p in code},
 'short_results_scope':'Input/routing/inventory audit only; does not establish trained quality, GPU reproducibility, remote-host availability or legal advice.'}
with (HERE/'input-and-route-execution.json').open('x',encoding='utf-8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps({'artifact':'input-and-route-execution.json','sha256':sha(HERE/'input-and-route-execution.json'),
 'pku':result['pku'],'literal_command_count':len(literal)},ensure_ascii=False,indent=2))
