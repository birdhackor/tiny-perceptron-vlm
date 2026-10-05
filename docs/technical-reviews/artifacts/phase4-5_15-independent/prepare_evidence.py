"""Own evidence acquisition for section 5.15; no old review is read."""
from pathlib import Path
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
def sha(raw): return hashlib.sha256(raw).hexdigest()
def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

temp = Path('/tmp/phase4-5_15-independent-original')
original = BASE / 'original'
original.mkdir(exist_ok=True)
for path in temp.iterdir():
    if path.is_file(): shutil.copyfile(path, original / path.name)
inputs = BASE / 'inputs'
paths = ['docs/review-tools/factual-reviewer-instructions.md',
    'docs/review-tools/section_facts.py', 'scripts/check_technical_reviews.py',
    '.agents/skills/clear-tutorial/references/review-protocol.md',
    'scripts/build_course.py', 'tiny_perceptron/model.py',
    'tiny_perceptron/attention.py', 'tiny_perceptron/modern.py',
    'tiny_perceptron/data.py', 'tiny_perceptron/training.py', 'pyproject.toml']
provenance = []
for name in paths:
    raw = (ROOT / name).read_bytes()
    destination = inputs / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    provenance.append({'input':name,'sha256':sha(raw),'snapshot':destination.relative_to(ROOT).as_posix()})
save_json(BASE / 'input-provenance.json', {
    'scope':'5.15 only; read 3.5 only for the standard-deviation cross-reference; no chapter introduction, no old reviews, no benchmark results, no weights',
    'section':json.loads((original / 'extraction.json').read_text()),
    'inputs':provenance,
    'original_extraction_command':'.venv/bin/python docs/review-tools/section_facts.py course/chapters/05.md#5.15 --output /tmp/phase4-5_15-independent-original --execute --timeout 30',
    'temporary_outputs_copied_as_bytes':True,
    'existing_empirical_json':'Not applicable: section has hypothetical scores, not an existing measured-results citation.',
    'figure':'No referenced image, SVG, or visual asset in section; no render claimed.'})

sources = BASE / 'sources'
sources.mkdir(exist_ok=True)
urls = {
 'pytorch-v2.8.0-random.py':'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/random.py',
 'pytorch-v2.8.0-sparse.py':'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/modules/sparse.py',
 'pytorch-v2.8.0-init.py':'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/init.py',
 'pytorch-v2.8.0-randomness.rst':'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/notes/randomness.rst',
 'pytorch-2.8-std.html':'https://docs.pytorch.org/docs/2.8/generated/torch.std.html',
 'pytorch-2.8-equal.html':'https://docs.pytorch.org/docs/2.8/generated/torch.equal.html',
 'bouthillier21a.pdf':'https://proceedings.mlr.press/v139/bouthillier21a/bouthillier21a.pdf',
 'bouthillier21a.html':'https://proceedings.mlr.press/v139/bouthillier21a.html',
}
def fetch(item):
    name, url = item
    request = urllib.request.Request(url, headers={'User-Agent':'technical-review/5.15'})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            raw=response.read()
            resolved=response.url
        path = sources / name
        path.write_bytes(raw)
        return {'name':name,'url':url,'resolved_url':resolved,'accessed_on':'2026-10-05','sha256':sha(raw),'bytes':len(raw),'saved':True}
    except Exception as e:
        return {'name':name,'url':url,'saved':False,'error':str(e)}
with ThreadPoolExecutor(max_workers=5) as pool:
    receipts=list(pool.map(fetch, urls.items()))
save_json(BASE / 'acquisition.json', receipts)
from bs4 import BeautifulSoup
for receipt in receipts:
    if receipt['saved'] and receipt['name'].endswith('.html'):
        raw=(sources / receipt['name']).read_bytes()
        soup=BeautifulSoup(raw,'html.parser')
        main=soup.find('article') or soup.find('main') or soup
        for element in main.find_all(['script','style']): element.decompose()
        text=main.get_text('\n',strip=True)
        (sources / receipt['name'].replace('.html','.txt')).write_text(text+'\n')
if (sources / 'bouthillier21a.pdf').is_file():
    result = subprocess.run(['pdftotext','-layout',str(sources/'bouthillier21a.pdf'),str(sources/'bouthillier21a.txt')],capture_output=True,text=True,timeout=20)
    save_json(BASE / 'pdf-conversion.json',{'command':result.args,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})

import torch
installed=BASE/'installed'
installed.mkdir(exist_ok=True)
contracts=[]
for name, object_ in [('Embedding',torch.nn.Embedding),('manual_seed',torch.manual_seed),('normal_',torch.nn.init.normal_)]:
    text=inspect.getsource(object_)
    file=installed/(name+'.py')
    file.write_text(text)
    contracts.append({'name':name,'installed_file':inspect.getsourcefile(object_),'installed_start_line':inspect.getsourcelines(object_)[1],'snapshot':file.relative_to(ROOT).as_posix(),'sha256':sha(file.read_bytes())})
save_json(BASE/'installed-environment.json',{'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),'torch_git_version':torch.version.git_version,'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu','contracts':contracts,'official_source_comparison':'Pinned PyTorch v2.8.0 upstream source and 2.8 API docs; runtime is 2.14.1+cpu. This is a version-aware contract check, not a claim that the versions are identical.'})
print(json.dumps({'acquisition':receipts,'input_snapshots':len(provenance),'original_fence':json.loads((original/'execution.json').read_text())['exit_code']},ensure_ascii=False,indent=2))
