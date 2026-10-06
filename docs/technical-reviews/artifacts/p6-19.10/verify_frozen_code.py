"""Check raw frozen code provenance against the actual implementation reviewed."""
from pathlib import Path
import hashlib
import json
import platform

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[3]
paths=['tiny_perceptron/selftrained/model.py','tiny_perceptron/selftrained/inference.py',
       'tiny_perceptron/model.py','tiny_perceptron/modern.py','tiny_perceptron/attention.py',
       'scripts/selftrained/train.py']
checks=[]
for arch in ['moe','dense']:
    freeze=REPO/f'docs/selftrained/results/public-raw/{arch}/freeze/frozen.json'
    raw=freeze.read_bytes()
    frozen=json.loads(raw)
    for name in paths:
        actual=hashlib.sha256((REPO/name).read_bytes()).hexdigest()
        assert actual == frozen['code_sha256'][name]
        checks.append({'freeze_path':str(freeze.relative_to(REPO)),
                       'freeze_sha256':hashlib.sha256(raw).hexdigest(),
                       'pointer':'/code_sha256/'+name,'code_path':name,'code_sha256':actual})
result={'environment':{'python':platform.python_version(),'device':'cpu'},
        'command':'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.10/verify_frozen_code.py',
        'checks':checks,'result':'all twelve original freeze/code bindings equal'}
(BASE/'frozen-code-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['result'])
