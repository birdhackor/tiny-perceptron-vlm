"""Fresh curriculum primary: deterministic source facts and seven bounded CPU predictions."""

import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load

ROOT = Path('/tmp/tpb').resolve()
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(data), 'sha256': sha(data)}


torch.set_num_threads(2)
result = {
    'schema': 'fresh-curriculum-primary-probe-v1',
    'created_at': datetime.now(UTC).isoformat(),
    'scope': 'Source identities, source-derived inventory, notebook generation check, API separation and seven CPU predictions only. No training, final evaluation or anonymous download.',
    'environment': {'python': sys.version, 'torch': torch.__version__, 'torch_git_version': torch.version.git_version, 'platform': platform.platform(), 'device': 'cpu', 'threads': str(torch.get_num_threads())},
}

# Inspect authored headings rather than importing an earlier review inventory.
heading = re.compile(r'^## ([A-Z\d]+\.\d+) (.+)$', re.M)
chapters = {}
chapter_digests = {}
for path in sorted((ROOT/'course/chapters').glob('*.md')):
    rows = heading.findall(path.read_text())
    chapters[path.name] = [pid for pid, _ in rows]
    chapter_digests[path.relative_to(ROOT).as_posix()] = sha(path.read_bytes())
front = {}
for name in ['README.md', 'first-steps.md', 'training.md', 'glossary.md']:
    path = ROOT/'course'/name
    front[name] = [pid for pid, _ in heading.findall(path.read_text())]
index = json.loads((ROOT/'course/lesson-index.json').read_text())
indexed = [row['id'] for row in index]
authored = [pid for rows in chapters.values() for pid in rows]
result['inventory'] = {
    'numeric_chapter_files': [name for name in chapters if 1 <= int(name[:-3]) <= 20] if all(name[:-3].isdigit() for name in chapters) else [name for name in chapters if name[:-3].isdigit() and 1 <= int(name[:-3]) <= 20],
    'chapter_sections': {key: len(value) for key, value in chapters.items()},
    'front_sections': {key: len(value) for key, value in front.items()},
    'chapter_total': len(authored), 'numbered_total': len(authored) + sum(map(len, front.values())),
    'source_hashes': chapter_digests,
    'index_count': len(index), 'index_unique': len(indexed) == len(set(indexed)), 'index_ids_equal_authored': set(indexed) == set(authored),
    'chapter_14_ids': [pid for pid in indexed if pid.startswith('14.')],
    'chapter_19_ids': [pid for pid in indexed if pid.startswith('19.')],
    'c_index_positions': {pid:indexed.index(pid) for pid in ['7.25','C.1','C.5','8.1','13.14','C.6','C.7','14.1'] if pid in indexed},
}
assert len(authored) == 332
assert result['inventory']['numbered_total'] == 358
assert len(result['inventory']['numeric_chapter_files']) == 20
assert len(indexed) == len(set(indexed)) and set(indexed) == set(authored)
assert result['inventory']['chapter_14_ids'] == [f'14.{i}' for i in range(1,20)]
assert result['inventory']['chapter_19_ids'] == [f'19.{i}' for i in range(1,26)]

# Local links are navigation assertions; this does not evaluate the linked research.
body = (ROOT/'docs/curriculum.md').read_text()
links = []
for target in re.findall(r'\]\(([^)]+)\)', body):
    if '://' in target:
        continue
    pathpart, _, fragment = target.partition('#')
    path = (ROOT/'docs'/pathpart).resolve()
    exists = path.is_file()
    anchor = True
    if fragment and exists:
        anchor = fragment in [pid for pid,_ in heading.findall(path.read_text())]
    links.append({'target': target, 'exists': exists, 'numbered_anchor_exists': anchor})
result['local_links'] = links
assert all(row['exists'] and row['numbered_anchor_exists'] for row in links)
project = tomllib.loads((ROOT/'pyproject.toml').read_text())
result['notebook_dependencies'] = project['dependency-groups']['notebook']
assert all(any(dep.split('>=')[0] == name for dep in result['notebook_dependencies']) for name in ['ipykernel','matplotlib','jupyterlab','nbclient'])
generation = subprocess.run([sys.executable,'scripts/build_course.py','--check'], cwd=ROOT, capture_output=True, text=True)
result['notebook_generation_check'] = {'command':[sys.executable,'scripts/build_course.py','--check'], 'exit_code':generation.returncode, 'stdout':generation.stdout, 'stderr':generation.stderr}
assert generation.returncode == 0, generation.stderr

# Exact local ZIP/member identity; nothing is extracted or written to a bundle.
descriptor = json.loads((ROOT/'docs/decision-product/v3/public-release.json').read_text())
archive_data = ARCHIVE.read_bytes()
assert len(archive_data) == descriptor['bytes'] == 6168350
assert sha(archive_data) == descriptor['sha256'] == '552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
result['fixed_zip'] = {'path':str(ARCHIVE), 'bytes':len(archive_data), 'sha256':sha(archive_data), 'transport':'existing local fixed ZIP, no network download', 'bundle_members':{}}
payload={}
with zipfile.ZipFile(ARCHIVE) as source:
    for name,digest in descriptor['bundle'].items():
        member = 'tiny-perceptron-c-v3/bundle/' + name
        assert source.namelist().count(member) == 1
        data=source.read(member)
        assert sha(data) == digest
        payload[name]=data
        result['fixed_zip']['bundle_members'][name]={'bytes':len(data),'sha256':sha(data)}
result['pinned_implementation'] = {}
for relative,digest in descriptor['implementation'].items():
    actual=sha((ROOT/relative).read_bytes())
    assert actual==digest
    result['pinned_implementation'][relative]=actual

from scripts.course_product import dispatch_request
from tiny_perceptron.decision_product.model import Tokenizer
from tiny_perceptron.decision_product.runtime import create_model, encode_for_model, candidate_eligibility
from tiny_perceptron.decision_product.service import DecisionEngine
from tiny_perceptron.decision_product.data import Example

manifest=json.loads(payload['bundle.json'])
policy=json.loads(payload['policy.json'])
model=create_model(manifest['config'])
model.load_state_dict(load(payload['weights.safetensors']))
tokenizer=Tokenizer(json.loads(payload['tokenizer.json']))
engine=DecisionEngine._for_development(model,tokenizer,policy['temperatures'],policy['threshold'])
result['cpu_configuration']={'manifest_config':manifest['config'],'policy':policy,'parameter_count':sum(p.numel() for p in model.parameters()),'weight_sha256':sha(payload['weights.safetensors']), 'construction':'in-memory exact ZIP member weights through _for_development; no on-disk extraction and no new product qualification'}
result['cpu_examples']=[]
for name in ['record-ready.json','record-unknown.json','answer-absent.json']:
    path=ROOT/'examples/decision_product'/name
    incoming=json.loads(path.read_text())
    answer=dispatch_request(engine,incoming)
    result['cpu_examples'].append({'request':file_record(path),'input':incoming,'output':answer})
assert result['cpu_examples'][0]['output']['results']['choice']['selected']=='record'
assert result['cpu_examples'][1]['output']['results']['choice']['status']=='abstained'
assert result['cpu_examples'][2]['output']['results']['choice']['status']=='abstained'
result['text_auto_rejected']=[]
for incoming in [{'text':'add three numbers'},[],None]:
    try:
        dispatch_request(engine,incoming)
    except ValueError as error:
        result['text_auto_rejected'].append({'input':incoming,'exception':type(error).__name__,'message':str(error)})
    else:
        raise AssertionError('legacy text input was accepted')

# Directly inspect ID-only target selection with truth labels absent and changed fields.
state1='裝置A：record；許可是；就緒是。裝置B：play；許可否；就緒否。裝置C：save；許可未知；就緒未知。'
state2='裝置A：save；許可未知；就緒否。裝置B：record；許可是；就緒是。裝置C：play；許可否；就緒未知。'
identity=[]
for state in [state1,state2]:
    example=Example(state,'選擇裝置B的動作。',(('x','裝置B：wait'),('y','裝置A：record')),'','choice','','service','')
    row=encode_for_model(example,tokenizer,model)
    identity.append({'target_row':row['target_row'],'label':row['label'],'eligibility':candidate_eligibility(example)})
assert identity==[{'target_row':1,'label':-100,'eligibility':[True,False,False]}]*2
result['id_only_probe']={'scope':'two synthetic state changes; index and target eligibility only, not neural semantic accuracy','observed':identity}

# Read original official PyTorch source shipped by this installed wheel, preserving
# only relevant method/doc excerpts with exact line numbers and a pinned URL.
pieces=[]
for symbol,path,ranges in [
    (torch.Tensor.backward,'torch/_tensor.py',[(1,65)]),
    (torch.optim.Optimizer.zero_grad,'torch/optim/optimizer.py',[(1,65)]),
    (F.cross_entropy,'torch/nn/functional.py',[(1,105)]),
    (torch.nn.MultiheadAttention.forward,'torch/nn/modules/activation.py',[(1,125)]),
]:
    lines,start=inspect.getsourcelines(symbol)
    pieces.append(f'URL https://github.com/pytorch/pytorch/blob/{torch.version.git_version}/{path}\nInstalled path: {inspect.getsourcefile(symbol)}\nSymbol: {symbol.__qualname__}\n')
    for begin,end in ranges:
        pieces.extend(f'{start+i-1}: {line}' for i,line in enumerate(lines,1) if begin<=i<=end)
    pieces.append('\n')
(OUT/'pytorch-original-excerpts.txt').write_text(''.join(pieces))
result['official_source_excerpts']=file_record(OUT/'pytorch-original-excerpts.txt')
result['result']='all explicit assertions passed; historical empirical/research retention not evaluated'
with (OUT/'primary-probe.json').open('x',encoding='utf-8') as target:
    target.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'result':result['result'],'inventory':{'chapter_total':len(authored),'numbered_total':result['inventory']['numbered_total']},'cpu_example_count':3,'prediction_count':7,'archive_sha256':result['fixed_zip']['sha256'],'environment':result['environment']},ensure_ascii=False,indent=2))
