"""Fresh README reviewer: small offline checks; no training or downloads."""
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
SCRATCH = Path('/workspace/work/tokenizer-dynamic-20261027/readme-callback')
DEST = ROOT / 'docs/technical-reviews/artifacts/tokenizer-20261027/readme-callback/probe-readme-01.json'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig

torch.set_num_threads(1)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads((ROOT / path).read_bytes())
def run(argv):
    p = subprocess.run(argv, cwd=ROOT, env={**os.environ, 'HF_HUB_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}, capture_output=True, text=True)
    return {'argv':argv, 'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}

freeze = read('docs/course-review-20261027-tokenizer/freeze-02.json')
page = next(p for p in freeze['pages'] if p['page_id']=='readme')
chapter_files = sorted((ROOT / 'course/chapters').glob('*.md'))
chapter_sections = sum(len(re.findall(r'^## [0-9ABC]+\.[0-9]+ ', p.read_text(), re.M)) for p in chapter_files)
reference_counts = {p:len(re.findall(r'^## [0-9A-Z]+\.[0-9]+ ', (ROOT/p).read_text(), re.M)) for p in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']}
notebooks=list((ROOT/'notebooks').glob('**/*.ipynb'))
manifest=read('docs/selftrained/v2-manifest.json')
v2=read('docs/selftrained/results/v2-final-public-results.json')
final=read('docs/lesson-products/results/final-report-v1.json')
settings=read('docs/lesson-products/results/settings-freeze-v1.json')
started=read('docs/lesson-products/results/settings-freeze-v1.test-started.json')
models=read('docs/course-experiments/public-models.json')['models']
capstone=read('docs/course-experiments/capstone-public.json')['models']
parameters={}
for architecture, result in v2['architectures'].items():
    model=LimitedAssistant(SelftrainedConfig(**result['capacity']['config']))
    parameters[architecture]={'current_code':sum(p.numel() for p in model.parameters()), 'saved_result':result['capacity']['parameters'], 'matches':sum(p.numel() for p in model.parameters())==result['capacity']['parameters']}
    del model

commands=[]
commands.append(run([sys.executable,'scripts/prepare_data.py','--kind','toy-text','--output',str(SCRATCH/'generated')]))
commands.append(run([sys.executable,'scripts/train.py','--task','text','--data',str(SCRATCH/'generated/toy-text/train.jsonl'),'--device','cpu','--output',str(SCRATCH/'no-weights.pt')]))
dry=json.loads(commands[-1]['stdout'].splitlines()[-1])
commands.append(run([sys.executable,'scripts/fetch_training_assets.py','--list']))
commands.append(run([sys.executable,'scripts/fetch_course_models.py','--list']))
weight_files=[f for m in models for f in m['files'] if Path(f['path']).suffix in {'.pt','.pth','.safetensors','.ckpt'}]
artifact_files=sum(len(m['files']) for m in models)
ci=read('docs/validation-artifacts/release-compatibility-ci.json')
report={
 'schema_version':1,'owner':'/root/tech_readme_callback_20261027','created_at':datetime.now(timezone.utc).isoformat(),
 'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','platform':platform.platform()},
 'scope':'Independent fresh README checks before any historical issue lookup. Saved historical GPU data inspected, not retrained. Data output paths redirected to reviewer scratch; algorithms/flags otherwise unchanged.',
 'source_sha256':sha(ROOT/'README.md'),'freeze_match':sha(ROOT/'README.md')==page['source_sha256'],'snapshot_match':sha(ROOT/page['snapshot'])==page['source_sha256'],
 'counts':{'chapter_files':len(chapter_files),'chapter_sections':chapter_sections,'notebooks':len(notebooks),'reference_sections':reference_counts,'total_numbered':chapter_sections+sum(reference_counts.values())},
 'v2_parameters':parameters,'v2_package':{'bytes':manifest['package']['bytes'],'files':len(manifest['records'])+len(manifest['assets']),'sha256':manifest['package']['sha256']},
 'v2_data_scope':v2['data_scope'],
 'v2_acceptance':{k:{'test_count':v['final_test']['count'],'acceptance':v['capability_acceptance_passed'],'tool_roundtrip':v['final_test']['per_task_final_reply']['tool_call']['tool_roundtrip']} for k,v in v2['architectures'].items()},
 'product_results':{'b5':{k:final['b']['5'][k] for k in ['boxes','sum']},'c':{k:v['after'] for k,v in final['c'].items()},'c_wording':{k:v['after'] for k,v in final['shifts']['c_wording'].items()},'gates':final['gates'],'settings_hash_matches':sha(ROOT/'docs/lesson-products/results/settings-freeze-v1.json')==final['settings_freeze_sha256'],'settings_recorded_at':settings['recorded_at'],'first_test_started':started,'reported_at':final['recorded_at'],'test_outcomes_observed_at_freeze':settings['test_outcomes_observed']},
 'public_counts':{'experiments':len(models),'weight_files':len(weight_files),'all_paired_files':artifact_files,'capstone_models':len(capstone),'capstone_weight_files':sum(sum(Path(f['path']).suffix in {'.pt','.pth','.safetensors','.ckpt'} for f in m['files']) for m in capstone)},
 'dry_run':{'mode':dry['mode'],'history':dry['history'],'no_checkpoint_created':not (SCRATCH/'no-weights.pt').exists(),'finite_loss':math.isfinite(dry['history'][0]['loss']),'finite_positive_grad':math.isfinite(dry['history'][0]['grad_norm']) and dry['history'][0]['grad_norm']>0},
 'historical_ci':{'revision':ci['revision'],'status':ci['status'],'jobs':[{'job':j['job'],'summary':j['summary'],'environment_check':j['environment_check']} for j in ci['jobs']]},
 'commands':commands,
 'assertions':{'all_commands_passed':all(c['returncode']==0 for c in commands),'counts_match':chapter_sections==313 and len(notebooks)==313 and chapter_sections+sum(reference_counts.values())==339,'model_parameters_match':all(v['matches'] for v in parameters.values()),'dry_run_contract':dry['mode']=='dry-run-no-weight-update' and math.isfinite(dry['history'][0]['loss']) and dry['history'][0]['grad_norm']>0 and not (SCRATCH/'no-weights.pt').exists()}
}
DEST.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'evidence':str(DEST),'assertions':report['assertions'],'dry_run':report['dry_run'],'counts':report['counts'],'public_counts':report['public_counts']},ensure_ascii=False))
