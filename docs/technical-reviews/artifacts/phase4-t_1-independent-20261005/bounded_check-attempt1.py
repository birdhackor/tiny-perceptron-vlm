"""Independent bounded CPU method checks for training.md#T.1; no downloads or training."""
import ast
import hashlib
import io
import json
import platform
import tarfile
import tempfile
from pathlib import Path
import torch
from scripts.prepare_data import generate_records, main as prepare_main
from tiny_perceptron.assets import unpack_asset
from tiny_perceptron.data import shifted
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def output(label, obj):
    print(json.dumps({'check':label, 'result':obj}, ensure_ascii=False))

output('environment', {'python':sys.version,'torch':str(torch.__version__),'torch_git_version':torch.version.git_version,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device':'cpu','platform':platform.platform(),'script_sha256':sha(Path(__file__).read_bytes())})
assert torch.version.cuda is None and not torch.cuda.is_available()

records = generate_records('attributes-sft')
lookup = {r['messages'][0]['content']:r for r in records}
queries = [('color=red;shape=square;pitch=low;shape?', 'square', 'red:square:low'),('color=red;shape=square;pitch=low;describe', 'square', 'red:square:low'),('color=blue;shape=circle;pitch=high;shape?', 'circle', 'blue:circle:high')]
for q,a,f in queries:
    assert lookup[q]['messages'][1]['content'] == a and lookup[q]['family'] == f
output('ABC-and-change-question', {'examples':[{'question':q,'answer':a,'family':f} for q,a,f in queries], 'changed_question_answer':lookup['color=red;shape=square;pitch=low;color?']['messages'][1]['content']})
assert lookup['color=red;shape=square;pitch=low;color?']['messages'][1]['content']=='red'

original_argv = sys.argv
try:
    sys.argv=['scripts/prepare_data.py','--kind','attributes-sft','--seed','42','--output',str(HERE/'generated')]
    # Actual generator CLI; small 60-record synthetic data, no optimizer/model.
    prepare_main()
finally:
    sys.argv=original_argv
sets = {}
for split in ('train','validation','test'):
    p=HERE/'generated/attributes-sft'/f'{split}.jsonl'
    rows=[json.loads(l) for l in p.read_text().splitlines()]
    sets[split]={r['family'] for r in rows}
    assert all(len([r for r in rows if r['family']==f])==5 for f in sets[split])
assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['test'] & sets['validation'])
output('family-disjointness', {s:sorted(v) for s,v in sets.items()})

x,y=shifted([11,12,13,14]);assert x.tolist()==[11,12,13] and y.tolist()==[12,13,14]
output('next-token-target', {'original':[11,12,13,14],'x':x.tolist(),'y':y.tolist()})

manifest_path=ROOT/'assets/training/manifest.json';manifest=json.loads(manifest_path.read_text())
listed=[json.loads(l) for l in (HERE/'list-stdout.jsonl').read_text().splitlines()]
assert len(listed)==len(manifest['assets'])==8
assert listed==[{k:a[k] for k in ('id','training_records','archive_bytes','license')} for a in manifest['assets']]
provenance=[]
for a in manifest['assets']:
    source_path=ROOT/a['source_metadata'];metadata=json.loads(source_path.read_text())
    inspected={k:metadata.get(k) for k in ('source_url','source_revision','license','files')}
    assert inspected['source_url'].startswith('https://') and inspected['source_revision'] and inspected['license']
    assert a['files'] and all(len(f['sha256'])==64 for f in a['files'])
    provenance.append({'id':a['id'],'manifest_fields':{k:a[k] for k in ('archive','archive_bytes','archive_sha256','license','source_metadata','target')},'source_sha256':sha(source_path.read_bytes()),'inspected_pointers':['/source_url','/source_revision','/license','/files'],'source_values':inspected})
    snapshot=HERE/'inputs'/a['source_metadata'];snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(source_path.read_bytes())
output('manifest-list-provenance',provenance)

asset=next(a for a in manifest['assets'] if a['id']=='tinystories')
archive=ROOT/asset['archive']
assert archive.stat().st_size == asset['archive_bytes']
assert sha(archive.read_bytes()) == asset['archive_sha256']
file_checks=[]
with tarfile.open(archive,'r:gz') as package:
    for member in package:
        row=next(r for r in asset['files'] if r['path']==member.name)
        raw=package.extractfile(member).read()
        assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
        file_checks.append({'path':member.name,'bytes':len(raw),'sha256':sha(raw)})
        if member.name.endswith('tinystories-train-512.jsonl'):
            lines=raw.splitlines(keepends=True)
            assert len(lines)==512
            samples=b''.join(lines[:3]);(HERE/'tinystories-first-three-original.jsonl').write_bytes(samples)
            parsed=[json.loads(l) for l in lines[:3]]
            assert all(isinstance(r.get('text'),str) and r['text'] for r in parsed)
            output('tiny-stories-original-rows',{'full_raw_file_sha256':sha(raw),'full_rows':len(lines),'saved_original_line_range':'1–3','sample_sha256':sha(samples),'row_keys':[list(r) for r in parsed],'text_characters':[len(r['text']) for r in parsed]})
output('existing-local-archive-integrity',{'archive_sha256':sha(archive.read_bytes()),'files':file_checks,'download_performed':False,'full_unpack_performed':False})

# Exact original unpack helper on a two-record archive; necessary changes test tampering and no-overwrite.
with tempfile.TemporaryDirectory(dir=HERE) as temp:
    base=Path(temp);payload=b'{"text":"a"}\n{"text":"b"}\n';name='tiny/sample.jsonl';arc=base/'mini.tar.gz'
    with tarfile.open(arc,'w:gz') as package:
        info=tarfile.TarInfo(name);info.size=len(payload);package.addfile(info,io.BytesIO(payload))
    config={'id':'tiny','archive_bytes':arc.stat().st_size,'archive_sha256':sha(arc.read_bytes()),'files':[{'path':name,'bytes':len(payload),'sha256':sha(payload)}]}
    first=unpack_asset(arc,config,base/'destination');second=unpack_asset(arc,config,base/'destination')
    assert first['files_verified']==1 and first['files_written']==1 and second['files_written']==0
    (base/'destination'/name).write_bytes(b'changed')
    try:unpack_asset(arc,config,base/'destination')
    except ValueError as e: different=str(e)
    else:raise AssertionError('must reject changed local target')
    tampered=dict(config,archive_sha256='0'*64)
    try:unpack_asset(arc,tampered,base/'untouched')
    except ValueError as e: wrong_sha=str(e)
    else:raise AssertionError('must reject wrong archive hash')
    assert not (base/'untouched').exists()
    output('bounded-original-unpack-and-variations',{'first':first,'second':second,'different_destination_error':different,'wrong_archive_hash_error':wrong_sha,'untouched_destination_exists':False})

fetch_tree=ast.parse((ROOT/'scripts/fetch_training_assets.py').read_text())
imports=[ast.unparse(n) for n in fetch_tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
assert all('training' not in n and 'model' not in n for n in imports)
output('fetch-contract',{'imports':imports,'default_output':'data/training','selected_asset':'tinystories','no_model_or_optimizer_calls_in_inspected_main':True,'download_command_not_executed':True})
output('complete',{'assertions':'all passed','model_created':False,'weights_updated':False,'network_calls':0})
