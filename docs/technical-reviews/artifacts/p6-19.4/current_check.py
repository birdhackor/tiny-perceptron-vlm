from pathlib import Path
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
start = json.loads((BASE/'active-current-source-callback.json').read_text())
CURRENT = ROOT/start['callback_dir']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
old = (BASE/'inputs/19.4.md').read_bytes()
new = (CURRENT/'19.4.md').read_bytes()
oldf = re.findall(rb'```([^\n]*)\n(.*?)```',old,re.S)
newf = re.findall(rb'```([^\n]*)\n(.*?)```',new,re.S)
assert len(oldf)==len(newf)==3
assert re.sub(rb'```python\n.*?```',b'__FENCE__',old,flags=re.S)==re.sub(rb'```python\n.*?```',b'__FENCE__',new,flags=re.S)
for (lang,body),(nlang,nbody) in zip(oldf,newf):
    assert lang==nlang
    if lang==b'python':assert ast.dump(ast.parse(body),include_attributes=False)==ast.dump(ast.parse(nbody),include_attributes=False)
    else:assert body==nbody
diff=''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='initial-19.4',tofile='current-19.4'))
(CURRENT/'section-diff.patch').write_text(diff)
dependencies=[]
for path in ['scripts/selftrained/train.py','scripts/selftrained/train_local_stage.py','scripts/selftrained/modal_runner.py','tiny_perceptron/selftrained/model.py','tiny_perceptron/selftrained/dataset.py','tiny_perceptron/selftrained/tokenizer.py','tiny_perceptron/model.py','course/figures/p6-19-start-lineage.svg','docs/selftrained/TRAINING.md','docs/selftrained/v2-manifest.json','docs/selftrained/v2-training-stage-index.json','docs/selftrained/results/v2-final-training-source-selection.json']:
    initial=BASE/'inputs'/path
    if not initial.exists():initial=BASE/'inputs'/Path(path).name
    assert (ROOT/path).read_bytes()==initial.read_bytes(),path
    dependencies.append({'path':path,'sha256':sha(ROOT/path),'same_as_original_verified_source':True})

expected='moe-pretrain 完成 1000 選定 250\nmoe-sft 完成 8000 選定 8000\nmoe-vision 完成 1200 選定 1200\nmoe-ocr 完成 3000 選定 3000\nmoe-audio 完成 1000 選定 200\nmoe-joint 完成 6000 選定 2000\nmoe-native 完成 4000 選定 1000\n'
executions=[]
for cwd in [ROOT,ROOT/'notebooks/19']:
    command=[sys.executable,'-c',newf[0][1].decode()]
    completed=subprocess.run(command,cwd=cwd,capture_output=True,text=True,timeout=10)
    assert completed.returncode==0 and completed.stdout==expected
    executions.append({'command':command,'cwd':str(cwd),'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr})

raw_rows={}
for path in sorted((BASE/'raw').glob('*/raw/train-receipt.json')):
    name=path.parents[1].name
    train=json.loads(path.read_bytes())
    original=ROOT/'docs/selftrained/results/training-raw'/name/'raw/train-receipt.json'
    assert sha(original)==sha(path)
    execution=json.loads((path.parent/'execution.json').read_bytes())
    receipt=json.loads((path.parent.parent/'receipt.json').read_bytes())
    inference=json.loads((path.parent/'inference-manifest.json').read_bytes())
    files={r['path']:r for r in receipt['files']}
    assert execution['status']=='completed' and execution['returncode']==0
    assert train['completed_requested_steps'] is True and train['interrupted'] is False
    assert train['steps']==execution['job']['steps']
    assert inference['selected_checkpoint_sha256']==files['best.pt']['sha256']
    assert train['origin']['kind']=='all-neural-weights-random' and train['test_used_for_selection'] is False
    raw_rows[name]={'completed_steps':train['steps'],'selected_step':inference['selected_step'],'best_sha256':files['best.pt']['sha256'],'stage_history':[{k:h[k] for k in ['step','checkpoint_sha256','selection']} for h in train['stage_history']]}
    for fname in ['execution.json','train-receipt.json','inference-manifest.json']:
        assert sha(path.parent/fname)==files[fname]['sha256']
edges=[]
for arch,suffixes in [('moe',['pretrain','sft','vision','ocr','audio','joint','weighted','native']),('dense',['pretrain','sft','vision','ocr','audio','joint','weighted'])]:
    for parent,child in zip(suffixes,suffixes[1:]):
        p,c=raw_rows[arch+'-'+parent],raw_rows[arch+'-'+child]
        h=c['stage_history'][-1]
        assert h['checkpoint_sha256']==p['best_sha256'] and h['step']==p['selected_step'] and h['selection']=='validation_loss'
        edges.append({'parent':arch+'-'+parent,'child':arch+'-'+child,'sha_matches':True,'loaded_selected_step':h['step']})
metrics=[json.loads(line) for line in (BASE/'raw/moe-native/raw/metrics.jsonl').read_text().splitlines()]
validation=[r for r in metrics if r['event']=='validation']
assert min(validation,key=lambda r:r['validation_loss'])['step']==raw_rows['moe-native']['selected_step']==1000
assert raw_rows['moe-native']['completed_steps']==4000
for p in [BASE/'external/saving_loading_models.py',BASE/'external/autograd-v2.11.0.rst']:
    assert p.is_file()
    dependencies.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'reuse':'same saved official original; personally reread relevant passages during callback'})
render_source=ROOT/'course/figures/p6-19-start-lineage.svg'
render_copy=CURRENT/'inputs/course/figures/p6-19-start-lineage.svg'
render_copy.parent.mkdir(parents=True,exist_ok=True);render_copy.write_bytes(render_source.read_bytes())
(CURRENT/'render_svg.py').write_bytes((BASE/'render_svg.py').read_bytes())
result={'reviewer_task':'/root/p6_fact_19_4','checked_at':datetime.now(timezone.utc).isoformat(),'environment':{'python':sys.version.split()[0],'device':'cpu / JSON and AST only','platform':platform.platform()},'original_source_sha256':hashlib.sha256(old).hexdigest(),'current_source_sha256':sha(CURRENT/'19.4.md'),'only_change':'Ruff folds repo=next(...) across two physical lines into one. AST equal; all non-Python-fence bytes and both bash fences equal.','python_ast_equal':True,'other_section_bytes_equal':True,'executions':executions,'raw_stages':raw_rows,'raw_edges':edges,'native_validation_min_step':1000,'dependencies':dependencies,'read_scope':'Current whole 19.4 lines166-255, necessary current19.3 context and TRAINING.md lines1-140; prior original evidence and exact required current implementation branches.','preserved_evidence':'Original bounded CPU optimizer/RNG/sampler exact-resume checks retained because trainer, wrapper, model, dataset, tokenizer and source methods are byte-identical. No training or heldout rerun.','pointer_scope':{'train':['/steps','/completed_requested_steps','/interrupted','/origin/kind','/stage_history','/test_used_for_selection'],'execution':['/status','/returncode','/job/steps'],'inference':['/selected_step','/selected_checkpoint_sha256'],'outer':['/files'],'metrics':['/event','/step','/validation_loss']}}
(CURRENT/'current-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Current Python AST is identical; other section bytes and two shell fences unchanged.')
print('Current Python fence executed from repository and notebooks/19: both match all seven expected rows.')
print('Rechecked 15 completed original stages and 13 parent selected-best SHA edges; native completed4000 selected1000.')
print('No training, paid work, heldout evaluation, model download, or upload executed.')
