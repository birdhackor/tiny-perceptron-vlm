"""Small summaries of original empirical records and bounded setup/kernel checks."""
import ast
import gzip
import hashlib
import inspect
import json
import os
import platform
import subprocess
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig

ROOT = Path('/workspace/tiny-perceptron-vlm')
HERE = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/omni-technical-readme-en-v1/supplement')
CACHE.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads((ROOT/p).read_bytes())
def ref(p): return {'path':p, 'sha256':sha(ROOT/p)}
result = {'created_at':datetime.now(UTC).isoformat(),'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','threads':'2'},'commands':[]}
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(ROOT),JUPYTER_PATH=str(CACHE/'kernel/share/jupyter'))
def run(name, cmd, timeout=45):
 p=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,timeout=timeout)
 raw=CACHE/(name+'.txt');raw.write_text(p.stdout+'\n[stderr]\n'+p.stderr)
 result['commands'].append({'name':name,'command':cmd,'returncode':p.returncode,'timeout_seconds':timeout,'raw_path':str(raw),'sha256':sha(raw),'key_output':(p.stdout+'\n'+p.stderr).strip()[:900]})
 return p
run('sync-dry-run',['uv','sync','--frozen','--extra','cpu','--group','notebook','--dry-run'])
run('kernel-install',['uv','run','--extra','cpu','python','-m','ipykernel','install','--prefix',str(CACHE/'kernel'),'--name','tiny-perceptron','--display-name','Tiny Perceptron'])
run('notebook-3.6',['uv','run','--extra','cpu','python','scripts/check_notebooks.py','--mode','kernel','--lesson','3.6','--workers','1','--timeout','30','--output',str(CACHE/'executed')])
run('build-course-check',['uv','run','--extra','cpu','python','scripts/build_course.py','--check'])

manifest=load('docs/selftrained/v2-manifest.json')
archive=ROOT/manifest['package']['path']
splits=Counter();verified=0
with tarfile.open(archive,'r:gz') as tar:
 members={m.name.removeprefix('./'):m for m in tar.getmembers() if m.isfile()}
 for item in manifest['records']+manifest['assets']:
  data=tar.extractfile(members[item['path']]).read()
  assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
  verified+=1
  if item in manifest['records']:
   split=next(s for s in ['train','validation','test'] if '-'+s+'.jsonl' in item['path'])
   splits[split]+=sum(bool(line.strip()) for line in data.splitlines())
result['v2_archive']={'source':ref('docs/selftrained/v2-manifest.json'),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'expected_archive_sha256':manifest['package']['sha256'],'verified_files':verified,'record_counts':dict(splits)}
final=load('docs/selftrained/results/v2-final-public-results.json')
result['v2_original_metrics']={}
result['parameter_counts']={}
for arch in ['moe','dense']:
 p=f'docs/selftrained/results/public-raw/{arch}/test/metrics.json';m=load(p)
 expected=final['architectures'][arch]['final_test']['metrics_sha256']
 result['v2_original_metrics'][arch]={'source':ref(p),'hash_matches_selected_final_metrics':sha(ROOT/p)==expected,'count':m['count'],'expected_count':m['expected_count'],'evaluation_complete':m['evaluation_complete'],'limited_smoke':m['limited_smoke'],'teacher_forcing_used_for_generation':m['teacher_forcing_used_for_generation'],'thresholds':m['thresholds'],'test_receipt':ref(f'docs/selftrained/results/public-raw/{arch}/test/receipt.json'),'generation_examples_not_general_quality':True}
 config=final['architectures'][arch]['capacity']['config']
 model=LimitedAssistant(SelftrainedConfig(**config))
 result['parameter_counts'][arch]={'config':config,'actual_parameters':sum(p.numel() for p in model.parameters()),'expected_parameters':final['architectures'][arch]['capacity']['parameters'],'initialization_only_no_training':True}
 del model

result['original_experiment_environments']=[]
for p in sorted((ROOT/'docs/course-experiments/results').glob('*.json')):
 x=json.loads(p.read_bytes())
 if 'torch_version' in x:
  result['original_experiment_environments'].append({'source':ref(str(p.relative_to(ROOT))),'device':x.get('device'),'gpu':x.get('gpu'),'torch':x['torch_version']})

result['external_originals']=json.loads(Path('/workspace/work/omni-technical-readme-en-v1/originals/receipts.json').read_bytes())
def fetch(name, url):
 p=CACHE/name
 with urlopen(Request(url,headers={'User-Agent':'technical-readme-en-independent-review'}),timeout=40) as response: data=response.read()
 p.write_bytes(data)
 result.setdefault('additional_originals',[]).append({'url':url,'path':str(p),'bytes':len(data),'sha256':sha(p),'accessed_at':datetime.now(UTC).isoformat()})
 return p
base='https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/'
for product in ['b','c']:
 p=fetch(product+'-training-receipt.json',base+product+'/training-receipt.json')
 x=json.loads(p.read_bytes())
 result.setdefault('product_training_receipts',{})[product]={k:v for k,v in x.items() if k in ['device','gpu','torch','torch_version','steps','actual_steps','selected_step','best_step','environment','last_step','seed']}
 result['product_training_receipts'][product]['available_keys']=list(x)
raw=fetch('product-predictions.jsonl.gz',base+'raw/final-report-v1.predictions.jsonl.gz')
expected=load('docs/lesson-products/results/publication-package-v1.json')['files']['raw/final-report-v1.predictions.jsonl.gz']
assert raw.stat().st_size==expected['bytes'] and sha(raw)==expected['sha256']
stats=Counter();first=[];uncompressed=hashlib.sha256()
with gzip.open(raw,'rb') as stream:
 for line in stream:
  uncompressed.update(line)
  row=json.loads(line)
  if len(first)<2:first.append({k:v for k,v in row.items() if k in ['product','id','family_id','task','gold','split','c_choice','c_info']})
  stats[str((row.get('group'),row.get('product'),row.get('stage'),row.get('depth'),row.get('question')))]+=1
result['product_raw_predictions']={'sha256':sha(raw),'decompressed_sha256':uncompressed.hexdigest(),'expected_decompressed_sha256':load('docs/lesson-products/results/final-report-v1.json')['predictions']['sha256'],'row_count':sum(stats.values()),'first_two_rows':first,'group_counts':dict(stats)}

source_files=['scripts/build_visuals.py','scripts/format_course_code.py','scripts/build_course.py','scripts/check_course_reviews.py','scripts/check_technical_reviews.py','docs/review-tools/check_review_round.py','scripts/check_notebooks.py','scripts/export_course.py','scripts/stage_mathjax.py','scripts/fetch_training_assets.py','scripts/fetch_course_models.py','scripts/check_course_models.py','tiny_perceptron/lesson_products/inference.py']
result['source_inspection']=[{**ref(p),'ast_syntax_ok':isinstance(ast.parse((ROOT/p).read_text()),ast.Module)} for p in source_files]
result['kernel_install_scope']='README --sys-prefix flag inspected in installed IPythonKernelSpecApp; actual install uses --prefix outside repo to protect shared environment.'
from ipykernel.kernelspec import InstallIPythonKernelSpecApp
src=inspect.getsource(InstallIPythonKernelSpecApp)
result['ipykernel_sys_prefix_supported']='sys-prefix' in src
p=HERE/'supplement-summary.json'
with p.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'artifact':str(p),'sha256':sha(p),'commands':[(x['name'],x['returncode']) for x in result['commands']],'archive':result['v2_archive'],'parameters':{k:v['actual_parameters'] for k,v in result['parameter_counts'].items()},'raw_prediction_rows':sum(stats.values())},ensure_ascii=False))
