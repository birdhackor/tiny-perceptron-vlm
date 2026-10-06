from pathlib import Path
import hashlib
import json
import re
import sys
import platform

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
RAW = ROOT / 'docs/selftrained/results/training-raw'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save_json(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

archive_index = json.loads((RAW / 'index.json').read_bytes())
provenance = []
for r in archive_index['files']:
    original, archived = Path(r['original']), ROOT / r['copy']
    assert original.read_bytes() == archived.read_bytes()
    assert sha(original) == sha(archived) == r['sha256']
    target = BASE / 'raw' / r['stage_key'] / ('receipt.json' if archived.name == 'receipt.json' else 'raw/' + archived.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(archived.read_bytes())
    assert sha(target) == sha(original)
    provenance.append({'original': str(original), 'archive': str(archived.relative_to(ROOT)), 'saved': str(target.relative_to(ROOT)), 'sha256': sha(target), 'bytes': target.stat().st_size})

stages = {}
key_inventory = []
for stage in sorted(p.name for p in RAW.iterdir() if p.is_dir()):
    t = json.loads((RAW / stage / 'raw/train-receipt.json').read_bytes())
    e = json.loads((RAW / stage / 'raw/execution.json').read_bytes())
    r = json.loads((RAW / stage / 'receipt.json').read_bytes())
    entries = {x['path']: x for x in r['files']}
    assert len(entries) == len(r['files'])
    assert all(r[k] == v for k, v in e.items())
    assert e['status'] == 'completed' and e['returncode'] == 0
    assert t['completed_requested_steps'] is True and t['interrupted'] is False
    assert t['steps'] == e['job']['steps']
    for name in ['execution.json','train-receipt.json']:
        p = RAW / stage / 'raw' / name
        assert sha(p) == entries[name]['sha256'] and p.stat().st_size == entries[name]['bytes']
    location = next(Path(x['original']).parent for x in archive_index['files'] if x['stage_key'] == stage and x['copy'].endswith('raw/train-receipt.json'))
    for name in ['inference-manifest.json'] + (['metrics.jsonl'] if stage == 'moe-native' else []):
        original = location / name
        assert sha(original) == entries[name]['sha256'] and original.stat().st_size == entries[name]['bytes']
        target = BASE / 'raw' / stage / 'raw' / name
        target.write_bytes(original.read_bytes())
        assert sha(target) == sha(original)
        provenance.append({'original': str(original), 'saved': str(target.relative_to(ROOT)), 'sha256': sha(target), 'bytes': target.stat().st_size})
    inference = json.loads((location / 'inference-manifest.json').read_bytes())
    key_inventory.append({'stage': stage, 'inference_keys': {k:type(v).__name__ for k,v in inference.items()}})
    assert inference['selected_checkpoint_sha256'] == entries['best.pt']['sha256']
    assert inference['origin'] == t['origin']
    assert t['origin']['kind'] == 'all-neural-weights-random'
    assert t['test_used_for_selection'] is False
    assert inference['selection'] == 'validation_loss'
    assert inference['stage'] == t['stage'] == e['job']['stage']
    for key in ['data_sha256','asset_sha256','tokenizer_sha256']:
        assert inference[key] == t[key]
    assert t['architecture'] == e['job']['architecture'] == t['config']['architecture']
    stages[stage] = {'training':t, 'execution':e, 'receipt':r, 'inference':inference, 'files':entries}

rows, edges = [], []
for architecture, order in [('moe',['pretrain','sft','vision','ocr','audio','joint','weighted','native']), ('dense',['pretrain','sft','vision','ocr','audio','joint','weighted'])]:
    previous = None
    for suffix in order:
        name = architecture + '-' + suffix
        s = stages[name]
        t,e,i = (s[k] for k in ['training','execution','inference'])
        job = e['job']
        rows.append({'stage_key':name,'completed_steps':t['steps'],'selected_step':i['selected_step'],'train_records':t['train_records'],'validation_records':t['validation_records'],'tokens':t['tokens'],'target_tokens':t['target_tokens'],'batch_size':job['batch_size'],'seed':t['seed'],'origin_kind':t['origin']['kind'],'best_sha256':s['files']['best.pt']['sha256'],'freeze_perception_backbones':t['freeze_perception_backbones'],'sampling_mode':t['sampling_mode'],'tool_loss_weight':t.get('tool_loss_weight',1),'numeric_run_loss_weight':t.get('numeric_run_loss_weight',1),'native_voice_loss_weight':t.get('native_voice_loss_weight',1)})
        if previous:
            parent=stages[previous]
            last=t['stage_history'][-1]
            assert last['checkpoint_sha256'] == parent['files']['best.pt']['sha256']
            assert last['step'] == parent['inference']['selected_step']
            assert last['selection'] == 'validation_loss'
            assert Path(last['checkpoint']).name == 'best.pt'
            # A resumed execution may name latest.pt; its preserved history binds the new-stage arrow.
            if 'init_checkpoint' in job:
                assert job['init_checkpoint']['sha256'] == parent['files']['best.pt']['sha256']
            edges.append({'from':previous,'to':name,'loaded_step':last['step'],'loaded_sha256':last['checkpoint_sha256'],'parent_completed':parent['training']['completed_requested_steps'],'parent_completed_steps':parent['training']['steps']})
        else:
            assert t['stage_history'] == [] and 'init_checkpoint' not in job and 'resume' not in job
        previous=name

for a in ['moe','dense']:
    base = stages[a+'-joint']['training']
    weighted = stages[a+'-weighted']['training']
    assert base['data_sha256']==weighted['data_sha256'] and base['asset_sha256']==weighted['asset_sha256'] and base['config']==weighted['config']
native = stages['moe-native']['training']
weighted = stages['moe-weighted']['training']
assert native['data_sha256']==weighted['data_sha256'] and native['asset_sha256']==weighted['asset_sha256'] and native['config']==weighted['config']
metrics=[json.loads(line) for line in (BASE/'raw/moe-native/raw/metrics.jsonl').read_text().splitlines()]
validation=[r for r in metrics if r['event']=='validation']
key_inventory.append({'native_metric_row_keys':{k:type(v).__name__ for k,v in validation[0].items()}})
best=min(validation,key=lambda r:r['validation_loss'])
assert best['step']==stages['moe-native']['inference']['selected_step']==1000
assert native['steps']==4000

chapter=(BASE/'inputs/19.4.md').read_text()
fence=re.findall(r'```python\n(.*?)```',chapter,re.S)[0]
print('DOCUMENTED OFFLINE FENCE:')
exec(compile(fence,'19.4-python-fence','exec'),{'__name__':'__main__'})
print('RAW RECORDS:')
for row in rows: print(row['stage_key'],'completed',row['completed_steps'],'selected',row['selected_step'])
print('All 15 completed executions and 13 new-stage parent edges matched. No weights loaded, no training or heldout evaluation performed.')
output={'environment':{'python':sys.version.split()[0],'platform':platform.platform(),'device':'cpu/read-only'},'stage_rows':rows,'lineage_edges':edges,'native_validation':[{'step':r['step'],'validation_loss':r['validation_loss'],'selected':r['selected']} for r in validation], 'inspected_pointers':{'execution':['/status','/returncode','/job','/revision','/command','/manifest_sha256'],'train-receipt':['/stage','/architecture','/steps','/tokens','/target_tokens','/train_records','/validation_records','/seed','/test_used_for_selection','/completed_requested_steps','/interrupted','/origin/kind','/origin/new_joint_initialization','/stage_history','/config','/data_sha256','/asset_sha256','/tokenizer_sha256','/freeze_perception_backbones','/sampling_mode','/tool_loss_weight','/numeric_run_loss_weight','/native_voice_loss_weight','/language_objective_policy'],'outer-receipt':['/files'],'inference-manifest':['/selected_step','/selected_checkpoint_sha256','/origin','/selection','/stage','/data_sha256','/asset_sha256','/tokenizer_sha256'],'metrics':['/event','/step','/validation_loss','/selected'],'stage-index':['/stages/*/stage_key','/stages/*/completed_steps','/stages/*/selected_step','/stages/*/receipt_path'],'final-source-selection':['/moe_joint/completed_requested_steps','/moe_joint/selected_step']}}
save_json(BASE/'raw-audit.json',output)
save_json(BASE/'raw-provenance.json',provenance)
save_json(BASE/'raw-key-inventory.json',key_inventory)
