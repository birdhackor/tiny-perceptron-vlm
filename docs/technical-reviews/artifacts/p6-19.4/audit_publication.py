from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
location=next((ROOT/'outputs/selftrained-v2/operations/release-final-four-exports-attempt-1/monitor/artifact').glob('*/*'))
originals=[location/'receipt.json',location/'raw/execution.json',location/'raw/hf-receipt.json']
provenance=[]
for original in originals:
    target=BASE/'publication'/original.relative_to(location)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(original.read_bytes())
    assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(original.read_bytes()).digest()
    provenance.append({'original':str(original),'saved':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
e=json.loads(originals[1].read_bytes());outer=json.loads(originals[0].read_bytes());hf=json.loads(originals[2].read_bytes())
assert e['status']=='completed' and all(outer[k]==v for k,v in e.items())
entries={r['path']:r for r in outer['files']}
for original in originals[1:]:
    assert hashlib.sha256(original.read_bytes()).hexdigest()==entries[original.name]['sha256']
assert hf['private'] is False and hf['repo_type']=='model'
assert len(hf['commit_sha'])==40
selected={r['name']:r for r in hf['exports']}
output=[]
for name,stage in [('moe-joint','moe-native'),('dense-joint','dense-weighted')]:
    raw=json.loads((BASE/'raw'/stage/'receipt.json').read_bytes())
    entries={r['path']:r for r in raw['files']}
    exp=selected[name]
    assert exp['source']['path']=='best.pt'
    assert exp['source']['sha256']==entries['best.pt']['sha256']
    assert exp['source']['run_id']==raw['run_id']
    assert exp['execution_sha256']==entries['execution.json']['sha256']
    assert exp['train_receipt_sha256']==entries['train-receipt.json']['sha256']
    inference=json.loads((BASE/'raw'/stage/'raw/inference-manifest.json').read_bytes())
    output.append({'export':name,'source_stage':stage,'source_run_id':raw['run_id'],'source_best_sha256':exp['source']['sha256'],'selected_step':inference['selected_step']})
result={'repo_id':hf['repo_id'],'commit_sha':hf['commit_sha'],'commit_url':hf['commit_url'],'prefix':hf['prefix'],'verified_exports':output,'originals':provenance,'inspected_pointers':{'execution':['/status','/revision','/job/release/exports'],'outer':['/files'],'hf-receipt':['/repo_id','/repo_type','/private','/commit_sha','/commit_url','/prefix','/exports/*/name','/exports/*/source','/exports/*/execution_sha256','/exports/*/train_receipt_sha256']}}
(BASE/'publication-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['originals','inspected_pointers']},indent=2))
