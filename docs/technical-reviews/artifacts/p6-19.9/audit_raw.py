"""Check selected original receipt pointers without reading author result commentary."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check_file(name, pointers):
    p = ROOT / name
    raw = json.loads(p.read_text())
    top = {k:type(v).__name__ for k,v in raw.items()}
    values = {}
    for ptr in pointers:
        v = raw
        for key in ptr.split('/')[1:]:
            v = v[int(key)] if isinstance(v,list) else v[key]
        values[ptr] = v
    dest = OUT / 'raw' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p,dest)
    assert digest(p) == digest(dest)
    return {'source':name,'saved':str(dest.relative_to(ROOT)),'sha256':digest(p),
        'top_keys_and_types':top,'checked_pointers':values}

checks = [check_file('docs/selftrained/results/v2-final-public-results.json',
    ['/architectures/moe/capacity/config','/architectures/moe/selected_source','/architectures/moe/safe_weights/sha256']),
    check_file('docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json',['/config','/stage','/architecture']),
    check_file('docs/selftrained/results/training-raw/moe-native/raw/execution.json',['/command','/revision','/returncode'])]
cfg = checks[0]['checked_pointers']['/architectures/moe/capacity/config']
weight_hash = checks[0]['checked_pointers']['/architectures/moe/safe_weights/sha256']
assert cfg == checks[1]['checked_pointers']['/config']
public = []
for task in ['text','vision_relation','voice_qa']:
    prefix=f'docs/selftrained/results/public-cpu-raw/{task}-1'
    argv = check_file(prefix+'-argv.json',['/argv'])
    result = check_file(prefix+'-result.json',['/returncode','/model/config','/model/files/model.safetensors','/stdout_sha256','/stderr_sha256'])
    stdout = check_file(prefix+'-stdout.txt',['/model/config','/model/files/model.safetensors','/generations/0/modality_kinds','/generations/0/prompt_ids','/generations/0/generated_ids'])
    stderr_path=ROOT/(prefix+'-stderr.txt')
    saved=OUT/'raw'/(prefix+'-stderr.txt');shutil.copyfile(stderr_path,saved)
    args=argv['checked_pointers']['/argv']; r=result['checked_pointers']; s=stdout['checked_pointers']
    assert args[args.index('--device')+1]=='cpu'
    assert r['/returncode']==0 and cfg==r['/model/config']==s['/model/config']
    assert weight_hash==r['/model/files/model.safetensors']==s['/model/files/model.safetensors']
    assert r['/stdout_sha256']==stdout['sha256']
    assert r['/stderr_sha256']==digest(stderr_path)==digest(saved)
    checks.extend([argv,result,stdout])
    public.append({'task':task,'device':'cpu','returncode':r['/returncode'],
        'prompt_length':len(s['/generations/0/prompt_ids']),
        'generated_length':len(s['/generations/0/generated_ids']),
        'modality_kinds':s['/generations/0/modality_kinds'],
        'stderr_bytes':stderr_path.stat().st_size})
receipt={'all_assertions_passed':True,'selected_configuration_matches_training_and_public':True,
    'public_original_runs':public,'files_and_pointers':checks,
    'scope':'Only archived original argv/stdout/stderr/result provenance and configuration; no quality rescoring, weight inference, GPU use or training.'}
(OUT/'raw-audit-results.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files_and_pointers'},ensure_ascii=False,indent=2))
