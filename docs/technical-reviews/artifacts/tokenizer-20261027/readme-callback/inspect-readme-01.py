"""Additional original-source inspection, with limited direct CPU API checks."""
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=Path('/workspace/work/tokenizer-dynamic-20261027/readme-callback')
DEST=ROOT/'docs/technical-reviews/artifacts/tokenizer-20261027/readme-callback/inspect-readme-01.json'
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.lesson_products.inference import auto_solve, parse_state, rule_depth

def load(p): return json.loads((ROOT/p).read_bytes())
def digest(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def command(argv):
    p=subprocess.run(argv,cwd=ROOT,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True)
    return {'argv':argv,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

package=load('docs/selftrained/v2-manifest.json')
selection=load('docs/selftrained/results/v2-final-training-source-selection.json')
exports=load('docs/selftrained/v2-jobs/release-final-four-exports.json')['release']['exports']
gpu=[]
for model in load('docs/course-experiments/public-models.json')['models']:
    path=f'docs/course-experiments/results/{model["id"]}.json'
    if (ROOT/path).exists():
        item=load(path)
        gpu.append({'id':model['id'],'source':path,'sha256':digest(path),'device':item.get('device'),'torch_version':item.get('torch_version'),'gpu':item.get('gpu')})
initial={}
for key in ['b','c']:
    p=f'docs/lesson-products/results/initial-{key}-execution.json'
    d=load(p)
    initial[key]={'source':p,'sha256':digest(p),'status':d['status'],'gpu':d['gpu_name'],'training_keys':list(d['training'])}
    for k,v in d['training'].items():
        if not isinstance(v,(dict,list)) or k in ['environment','receipt','config']: initial[key][k]=v

sources={
 'pyproject.toml':'CPU/cu126/cu130 extras conflict; notebook/site groups; platform wheel selection delegated to uv.lock.',
 'uv.lock':'Read torch package entries/platform wheels with TOML parser; no dependency sync performed.',
 'scripts/train.py':'parser defaults, loss.backward, optional optimizer.step and train-only save paths; minimum flow executed separately.',
 'scripts/prepare_data.py':'12 toy rule families; deterministic family splits and output manifest; minimum data command executed separately.',
 'scripts/fetch_training_assets.py':'--list returns before LFS subprocess/unpack; executed --list separately.',
 'scripts/fetch_course_models.py':'pinned 40 hex HF revision, token=False, sha/byte check, only selected model; --list executed separately.',
 'scripts/check_course_models.py':'check_model fetches selected spec, validates folder, CPU probe and optional inference, scope excludes quality benchmark.',
 'scripts/build_course.py':'BOOTSTRAP detects google.colab, clone/install, local root discovery; canonical Notebook generation kernelspec.',
 'scripts/check_notebooks.py':'run_python fresh namespace; run_kernel new NotebookClient per Notebook with tiny-perceptron kernel.',
 'scripts/export_course.py':'canonical readme mapping, executed outputs, Colab commit URL, Zensical export with bundled MathJax.',
 'scripts/check_env.py':'pick_device and 256x256 forward/backward; no full model platform guarantee.',
 'tests/test_env.py':'accelerator test compares 32x32 product with CPU; historical CI receipt is the execution observation.',
 '.github/workflows/ci.yml':'Linux/Windows cpu and macOS default matrix, check_env then tests.',
 'tiny_perceptron/selftrained/model.py':'SelftrainedConfig, all encoders/LM constructed locally; runtime parameter sums checked separately.',
 'tiny_perceptron/natural_assistant.py':'MODEL_ID Qwen3-VL; Qwen3VLForConditionalGeneration.from_pretrained; separate Whisper ASR then text/vision route.',
 'tiny_perceptron/lesson_products/inference.py':'parse_state finite grammar; rule_depth; C choice/info then unsupported/clarification/budget branch in auto_solve.',
 'scripts/selftrained/train_local_stage.py':'Local stage wrapper cli, own --init-checkpoint/--resume paths, offline train subprocess and promotion of own checkpoint.',
 'docs/selftrained/TRAINING.md':'Full local stage commands and own checkpoint distinction from inference-only exports.',
 'zensical.toml':'Search, nav, copy, palette modes; site output paths; mathjax scripts.',
 'LICENSE':'MIT applies to repository material; external data explicitly separate.',
}
bindings=[{'path':p,'sha256':digest(p),'inspection_note':n} for p,n in sources.items()]
locked=tomllib.loads((ROOT/'uv.lock').read_text())
torch_entries=[{'version':p['version'],'source':p['source'],'wheel_count':len(p.get('wheels',[])),'macos_wheels':[w['url'].rsplit('/',1)[-1] for w in p.get('wheels',[]) if 'macosx' in w['url']]} for p in locked['package'] if p['name']=='torch']
nb=load('notebooks/01/1.1.ipynb')
bootstrap=''.join(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
current=ast.parse((ROOT/'scripts/build_course.py').read_text())
generator=next(ast.literal_eval(n.value) for n in current.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BOOTSTRAP' for t in n.targets))
auto_probes={'parsed_state':parse_state('兩盒追蹤;A=2;B=6;事件=交換;查詢=B'),'rule_depths':[rule_depth({'version':1,'sum_depth':3,'low_max_events':1,'medium_max_events':3},'boxes',i) for i in [1,2,4]]}
class B:
    auto_policy={'version':1,'sum_depth':3,'low_max_events':1,'medium_max_events':3}
    def synchronize(self): pass
    def solve(self,state,loops): return {'answer':'stub','loops':loops,'seconds':0.0}
class C:
    def __init__(self,choice='boxes',info='yes'): self.choice,self.info=choice,info
    def synchronize(self): pass
    def judge(self,state,question): return {'selected':self.choice if question=='choice' else self.info,'seconds':0.0}
for label,c,effort,maxloops in [('accepted',C(),'auto',5),('missing_info',C(info='no'),'auto',5),('unsupported',C(choice='card'),'auto',5),('limited',C(),'high',2)]:
    auto_probes[label]=auto_solve(B(),c,'兩盒追蹤;A=2;B=6;事件=交換;查詢=B',effort,maxloops)
commands=[command(['uv','--version']),command(['git','--version']),command(['uv','sync','--frozen','--extra','cpu','--group','notebook','--dry-run','--offline']),command([sys.executable,'-m','ipykernel','install','--prefix',str(OUT/'kernel-check'),'--name','tiny-perceptron','--display-name','Tiny Perceptron'])]
report={'owner':'/root/tech_readme_callback_20261027','created_at':datetime.now(timezone.utc).isoformat(),'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},'scope':'Current-byte source inspection plus four CPU control-flow stub probes. Stubs validate orchestration only and do not claim product quality; no pretrained model loaded, GPU training, weights download, or external publication performed. uv sync was --dry-run --offline; kernel registration used reviewer scratch prefix.',
 'current_sources':bindings,'torch_lock_entries':torch_entries,'bootstrap_matches_notebook':bootstrap.strip()==generator.strip(),'notebook_kernelspec':nb['metadata']['kernelspec'],'v2_selection':{'moe_joint':{'selected_step':selection['moe_joint']['selected_step'],'completed_steps':selection['moe_joint']['completed_requested_steps'],'source_run':selection['moe_joint']['source_training_run'],'source_sha256':selection['moe_joint']['sha256']},'dense_joint':{'selected_step':selection['dense_joint']['selected_step'],'completed_steps':selection['dense_joint']['completed_steps'],'source_run':selection['dense_joint']['source_run_id'],'source_sha256':selection['dense_joint']['best_sha256']}},'v2_exports':{'count':len(exports),'paired_files':sum(len(e['files']) for e in exports),'names':[e['name'] for e in exports],'licenses':sorted({f['license'] for e in exports for f in e['files']})},'historical_gpu_records':gpu,'historical_product_training':initial,'auto_probes':auto_probes,'commands':commands}
DEST.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'evidence':str(DEST),'source_count':len(bindings),'torch_entries':torch_entries,'bootstrap_match':report['bootstrap_matches_notebook'],'selection':report['v2_selection'],'exports':report['v2_exports'],'gpu_records':len(gpu),'gpu_versions':sorted({str(r['torch_version']) for r in gpu}),'command_exit_codes':[c['returncode'] for c in commands],'auto':auto_probes},ensure_ascii=False))
