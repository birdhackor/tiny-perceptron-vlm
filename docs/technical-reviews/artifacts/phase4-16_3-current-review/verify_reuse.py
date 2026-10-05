import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[4]
P=Path(__file__).resolve().parent
OLD=ROOT/'docs/technical-reviews/artifacts/phase4-16_3-independent'
H=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
current=(P/'input/section.md').read_bytes()
assert H(P/'input/section.md')=='180fd63a4635b089541e5084a0520f00595242ac39ebf9483672bc797edec5af'
assert (P/'input/fence-1.py').read_bytes()==(OLD/'input/fence-1.py').read_bytes()
assert (P/'input/bootstrap.py').read_bytes()==(OLD/'input/bootstrap.py').read_bytes()
print(json.dumps({'environment':{'python':platform.python_version(),'torch_distribution':importlib.metadata.version('torch'),'device_used_for_prior_inference':'cpu','new_model_forward_calls':0,'new_training_updates':0},'fence_sha256':H(P/'input/fence-1.py'),'unchanged_fence':True,'unchanged_bootstrap':True}))
for filename in ['model.py','attention.py','modern.py','data.py']:
    current_path=ROOT/'tiny_perceptron'/filename
    assert current_path.read_bytes()==(OLD/'code'/filename).read_bytes()
    print(json.dumps({'current_code':str(current_path.relative_to(ROOT)),'sha256':H(current_path),'unchanged_from_prior_personally_inspected_code':True}))
with urlopen('http://127.0.0.1:8765/figures/rewrite-16-cache-append.svg',timeout=10) as response:
    served=response.read()
figure=ROOT/'course/figures/rewrite-16-cache-append.svg'
assert served==figure.read_bytes()==(OLD/'input/rewrite-16-cache-append.svg').read_bytes()
(P/'input/current-cache-append.svg').write_bytes(served)
print(json.dumps({'current_figure_sha256':H(figure),'served_figure_matches_current_canonical':True,'original_figure_unchanged':True}))
raw_path=ROOT/'docs/course-experiments/results/efficiency.json'
assert raw_path.read_bytes()==(OLD/'primary/efficiency-original.json').read_bytes()
d=json.loads(raw_path.read_bytes())
for name in ['mha','gqa']:
    c=d['results']['models'][name]['cache']
    assert c['prompt_tokens']==25 and c['generated_tokens']==12
    assert len(c['per_step_logit_max_error'])==len(c['generated_ids_full'])==len(c['generated_ids_cached'])==12
    assert c['generated_ids_full']==c['generated_ids_cached']
    assert max(c['per_step_logit_max_error'])==1.9073486328125e-6
    assert max(c['per_step_logit_max_error'])<1e-5
    raw_bytes=bytes(i-8 for i in c['generated_ids_full'] if i>=8)
    assert raw_bytes.decode()==c['generated_text']
    assert c['generated_ids_full'].count(2)>1
    print(json.dumps({'raw_model':name,'prompt_positions':25,'generation_steps':12,'maximum_absolute_error':max(c['per_step_logit_max_error']),'rounded_display':format(max(c['per_step_logit_max_error']),'.5e'),'all_12_ID_positions_equal':True,'EOS_count':c['generated_ids_full'].count(2),'decoded_text':c['generated_text']}))
for name in ['scripts/course_experiments/architecture.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/data.py']:
    raw=subprocess.run(['git','show',d['revision']+':'+name],cwd=ROOT,check=True,capture_output=True).stdout
    assert hashlib.sha256(raw).hexdigest()==d['code_sha256'][name]
    old_file=OLD/'code'/('architecture-at-raw-revision.py' if name.startswith('scripts/') else Path(name).name)
    assert raw==old_file.read_bytes()
print(json.dumps({'original_measurement_sha256':H(raw_path),'original_git_revision':d['revision'],'original_measurement_and_method_unchanged':True}))
manifest=json.loads((P/'history/prior-manifest.json').read_bytes())
for item in manifest['original_evidence_files']:
    assert H(ROOT/item['path'])==item['sha256']
print(json.dumps({'complete_prior_evidence_manifest_verified':True,'preserved_file_count':len(manifest['original_evidence_files'])}))
