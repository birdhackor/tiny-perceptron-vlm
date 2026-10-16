"""T.11 independent checks: no training, model download or inference."""
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices
import torch
from scripts.course_experiments.applications import _sample


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def run(*args):
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


out = {"scope": "Independent T.11 record/API audit; historical reports inspected, not rerun training or model inference.",
       "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                       "platform": platform.platform(), "device": "cpu", "git": run('git', '--version')}}
body = lesson_slices((ROOT / 'course/training.md').read_bytes().decode())['T.11']
freeze = read('docs/course-revision-20261016/freeze-08.json')
page = next(x for x in freeze['pages'] if x['page_id'] == 'training')
frozen_body = lesson_slices((ROOT / page['snapshot']).read_bytes().decode())['T.11']
assert body == frozen_body
out['section'] = {"source": "course/training.md#T.11", "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                  "freeze": "freeze-08.json", "frozen_section_matches": True, "figure_references": 0}
old_size, new_size, old_correct, new_correct, total = 80000, 60000, 4, 3, 5
assert new_size < old_size and new_correct < old_correct
out['numeric'] = {"size_change_bytes": new_size-old_size, "size_ratio": new_size/old_size,
                  "old_score": old_correct/total, "new_score": new_correct/total,
                  "size_criterion": True, "score_retention_criterion": False,
                  "same_aggregate_different_errors": {"a": [1,1,1,1,0], "b": [0,1,1,1,1]},
                  "dropping_one_wrong_item_changes_rate": {"full": 4/5, "selected": 4/4},
                  "scope": "Transparent assumed arithmetic only, not quantization results."}
assert sum(out['numeric']['same_aggregate_different_errors']['a']) == sum(out['numeric']['same_aggregate_different_errors']['b']) == 4
commit = run('git', 'rev-parse', 'HEAD')
committed = subprocess.run(['git','show',f'{commit}:scripts/selftrained/train_local_stage.py'], cwd=ROOT, capture_output=True, check=True).stdout
out['identity'] = {"git_commit": commit, "retrieved_tracked_file": "scripts/selftrained/train_local_stage.py",
                   "retrieved_matches_worktree": committed == (ROOT/'scripts/selftrained/train_local_stage.py').read_bytes(),
                   "same_bytes_same_sha256": hashlib.sha256(b'answer=wrong').hexdigest() == hashlib.sha256(b'answer=wrong').hexdigest(),
                   "meaning_note": "A digest identifies bytes; even deliberately wrong content has a reproducible digest."}
spec = importlib.util.spec_from_file_location('t11_local_stage', ROOT/'scripts/selftrained/train_local_stage.py')
wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
help_text = run(sys.executable, 'scripts/selftrained/train_local_stage.py', '--help')
assert all(x in help_text for x in ('--manifest', '--manifest-sha256', '--data-root'))
with tempfile.TemporaryDirectory(prefix='T.11-', dir='/workspace/work/curriculum-20261016') as temp:
    p=Path(temp)/'record.txt'; p.write_bytes(b'fixture, no model checkpoint')
    item={"path":p.name,"bytes":p.stat().st_size,"sha256":digest(p)}
    wrapper.verified_file(Path(temp),item)
    p.write_bytes(b'altered')
    try:wrapper.verified_file(Path(temp),item)
    except ValueError: rejected=True
    else: rejected=False
    assert rejected
source_revision, source_hashes = wrapper.source_identity(ROOT)
module_ast=ast.parse((ROOT/'scripts/selftrained/train_local_stage.py').read_text())
imports=[ast.unparse(n) for n in ast.walk(module_ast) if isinstance(n,(ast.Import,ast.ImportFrom))]
assert not any('modal' in x.lower() for x in imports)
out['local_wrapper']={"help_exit":0,"verified_file_accepts_exact_bytes":True,"modified_file_rejected":rejected,
                      "source_identity_revision":source_revision,"source_identity_files":len(source_hashes),
                      "modal_imports":[],"scope":"Verified helper/API/source contracts only; no stage or checkpoint load run."}
manifest=read('docs/selftrained/v2-manifest.json')
lfs_attribute=run('git','check-attr','filter','--','assets/training/selftrained-v2.tar.gz')
assert manifest['package']['kind']=='git-lfs' and lfs_attribute.endswith('filter: lfs')
assert manifest['initialization']=='random' and len(manifest['records'])==12 and len(manifest['assets'])==8950
out['fixed_data_contract']={"manifest_sha256":digest(ROOT/'docs/selftrained/v2-manifest.json'),
                            "package":manifest['package'],"records":12,"assets":8950,
                            "git_attribute":lfs_attribute,"archive_downloaded_or_unpacked":False}
training=read('docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json')
execution=read('docs/selftrained/results/training-raw/moe-native/raw/execution.json')
selection=read('docs/selftrained/results/v2-final-training-source-selection.json')
final=read('docs/selftrained/results/v2-final-public-results.json')
assert training['steps']==4000 and training['completed_requested_steps'] and not training['interrupted']
assert not training['test_used_for_selection'] and training['inference_exported']
assert execution['returncode']==0 and selection['moe_joint']['selected_step']==1000
out['v2_historical']={"training": {"completed_steps":4000,"selected_step":1000,"train_records":training['train_records'],
                                  "validation_records":training['validation_records'],"tokens":training['tokens'],
                                  "target_tokens":training['target_tokens'],"seed":training['seed'],
                                  "batch_size":execution['job']['batch_size'],"context":execution['job']['context'],
                                  "test_used_for_selection":False},"tests":{}}
for arch, candidate in final['architectures'].items():
    path=f'docs/selftrained/results/public-raw/{arch}/test/metrics.json'; metrics=read(path)
    assert digest(ROOT/path)==candidate['final_test']['metrics_sha256']
    assert metrics['evaluation_complete'] and not metrics['limited_smoke'] and not metrics['interrupted']
    assert metrics['count']==metrics['expected_count']==candidate['final_test']['count']==3734
    assert sum(x['count'] for x in metrics['per_task_final_reply'].values())==3734
    assert metrics['safe_weights_sha256']==candidate['safe_weights']['sha256']
    assert not candidate['capability_acceptance_passed']
    out['v2_historical']['tests'][arch]={"count":3734,"metrics_sha_matches":True,
        "per_task_counts":{k:v['count'] for k,v in metrics['per_task_final_reply'].items()},
        "safe_weights_sha256":metrics['safe_weights_sha256'],"capability_acceptance_passed":False}
# Public HEAD/GET only: do not download weights or execute a model.
base='https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-joint/'
def public_file(name):
    method='HEAD' if name=='model.safetensors' else 'GET'
    with urlopen(Request(base+name, method=method, headers={'User-Agent':'T11-independent-review'}),timeout=30) as response:
        result={"file":name,"method":method,"http_status":response.status,"anonymous":True}
        if method=='GET':
            raw=response.read();result.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        return result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    out['public_release']=list(executor.map(public_file, ['model.safetensors','model-config.json','tokenizer.json','inference-manifest.json']))
assert next(x for x in out['public_release'] if x['file']=='inference-manifest.json')['sha256']=='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e'
# The mature route is a source-level contract, not a new quality measurement.
natural=ast.parse((ROOT/'tiny_perceptron/natural_assistant.py').read_text())
constants={}
for node in natural.body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
        if node.targets[0].id in ('MODEL_ID','MODEL_REVISION'):
            constants[node.targets[0].id]=ast.literal_eval(node.value)
functions={n.name:ast.unparse(n) for n in natural.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
assert 'Qwen3VLForConditionalGeneration.from_pretrained' in functions['load_core']
assert 'WhisperForConditionalGeneration.from_pretrained' in functions['load_asr']
out['mature_route']={**constants,"core_loader":"Qwen3VLForConditionalGeneration.from_pretrained",
                     "asr_loader":"WhisperForConditionalGeneration.from_pretrained","models_executed":False}
reports={}
for name in ('rag','tools','reasoning'):
    d=read(f'docs/course-experiments/results/{name}.json');r=d['results']
    assert r['split']['test']['records']>0
    entry={"historical_revision":d['revision'],"torch_version_recorded":d['torch_version'],"device_recorded":d['device'],
           "seed":d['seed'],"test_records":r['split']['test']['records'],"test_families":r['split']['test']['families']}
    if name=='rag':
        assert all('input_ids' in x['generation'] and x['generation']['samples'] for x in r['samples'])
        entry['saved_generation_samples']=len(r['samples'])
    elif name=='tools':
        assert len(r['samples'])==20 and len(r['one_step_cap_samples'])==18
        entry.update(normal_episodes=20,one_step_cap_episodes=18,score=r['accuracy'])
    else:
        entry['branches']={}
        for branch,b in r['comparison'].items():
            for budget in b['budgets']:
                assert len(budget['samples'])==24
                assert all(x['generation']['input_ids'] and x['candidates'] for x in budget['samples'])
                assert budget['majority_accuracy']['denominator']==24
            entry['branches'][branch]=[{"candidate_count":x['candidate_count'],"samples":len(x['samples'])} for x in b['budgets']]
    reports[name]=entry
out['historical_reports']=reports
old_path=ROOT/'docs/course-experiments/results/tools.json';old_digest=digest(old_path)
reconstruction=run(sys.executable,'docs/course-repair-20261008/evidence/reconstruct_tool_snapshots.py')
assert digest(old_path)==old_digest=='3befd7fcb9a1a48d850761ab6d0917ac42388dc3bbfe9f011f0b98c3752d7140'
out['tool_reconstruction']=json.loads(reconstruction)
old_code=run('git','show','910aebc6419c9fc6217279a27fde9851c5cfad30:scripts/course_experiments/applications.py')
assert '"messages": messages,' in old_code
# A deterministic fake model checks snapshot retention without learned weights.
class Fake:
    def __init__(self):self.training=True;self.config=SimpleNamespace(max_length=256)
    def eval(self):self.training=False
    def train(self,mode):self.training=mode
    def __call__(self,current,cache=None):
        logits=torch.zeros(current.shape[0],current.shape[1],264);logits[:,:,-262]=10
        return {"logits":logits,"cache":None}
messages=[{"role":"user","content":"fixture"}]
saved=_sample(Fake(),messages,SimpleNamespace(device='cpu'),tokens=1)
messages.append({"role":"assistant","content":"later"})
assert len(saved['messages'])==1 and saved['samples'][0]['eos']
out['current_snapshot_contract']={"saved_message_count_after_input_append":len(saved['messages']),
                                 "fake_model_emitted_eos":True,"training_or_quality_evidence":False,
                                 "historical_alias_return_found":True}
output=Path(__file__).with_name('T.11-execution.json')
output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({"checks":"passed","output":output.relative_to(ROOT).as_posix(),"section_sha256":out['section']['body_sha256'],
                  "public_files_checked":len(out['public_release']),"tool_snapshots":out['tool_reconstruction']['records_verified']},ensure_ascii=False))
