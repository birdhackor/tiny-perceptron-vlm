"""Primary-source receipt, historical-run audit, and tiny deterministic T.3 fixtures."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
CACHE=Path('/workspace/work/reference/omni-technical-t-3-originals-v1')
WORK=Path('/workspace/work/omni-technical-t-3-v1')
sys.path.insert(0,str(ROOT))
from tiny_perceptron.data import split_documents,toy_documents
from tiny_perceptron.model import generate
from tiny_perceptron.tokenization import generation_report
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.simple import BigramLM

def sha(raw):return hashlib.sha256(raw).hexdigest()
receipt={'created_at':datetime.now(UTC).isoformat(),
    'command':'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-t-3-v1/audit_t3.py',
    'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},
    'primary_source_receipts':json.loads((CACHE/'fetch-receipts.json').read_text())+json.loads((CACHE/'extra-fetch-receipts.json').read_text()),
    'installed_torch_originals':json.loads((CACHE/'torch-original-receipts.json').read_text())}
paths=['scripts/train_simple.py','tiny_perceptron/simple.py','tiny_perceptron/data.py','tiny_perceptron/model.py',
       'tiny_perceptron/tokenization.py','tiny_perceptron/training.py','scripts/infer.py','scripts/fetch_course_models.py',
       'scripts/course_experiments/text.py','scripts/course_experiments/common.py','scripts/course_experiments/run.py',
       'scripts/course_release.py','scripts/prepare_data.py','scripts/ingest_course_result.py',
       'docs/course-experiments/results/simple_models.json','docs/course-experiments/public-models.json',
       'docs/course-experiments/plan.json']
receipt['repository_files']={path:sha((ROOT/path).read_bytes()) for path in paths}
rawpath=ROOT/'docs/course-experiments/results/simple_models.json'
record=json.loads(rawpath.read_text())
historical={
    'primary_record_path':str(rawpath.relative_to(ROOT)),'primary_record_sha256':sha(rawpath.read_bytes()),
    'revision':record['revision'],'device':record['device'],'python':record['python_version'],'torch':record['torch_version'],
    'seed':record['seed'],'elapsed_seconds':record['elapsed_seconds'],
    'rounded_to_one_decimal':round(record['elapsed_seconds'],1),'timing_scope':record['timing_scope'],
    'modal':record['modal'],'step_scale':record['step_scale'],'evidence_status':record['evidence_status'],
    'recorded_raw_core_results_sha256':sha(json.dumps(record['results'],sort_keys=True).encode()),
    'same_run_raw_locator':'/results/runs contains pre/post loss and per-step raw history for all four configurations; this is the public primary run record, not an author summary or a new local run.',
    'limitations':'Historical author measurement is verified as the preserved primary run record, with the pinned timing code and denominators. This review did not witness/re-run Modal or authenticate the private HF archive (anonymous retrieval returned 401). A new CPU execution is separate and is not used to verify the 4.3-second wall-clock observation.',
    'code_hash_checks':{},'data_hash_checks':{},'historical_record_checks':[]}
for path in ['tiny_perceptron/data.py','tiny_perceptron/simple.py','tiny_perceptron/training.py',
             'scripts/course_experiments/text.py','scripts/course_experiments/common.py','scripts/course_experiments/run.py']:
    old=subprocess.check_output(['git','show',record['revision']+':'+path],cwd=ROOT)
    historical['code_hash_checks'][path]={'recorded':record['code_sha256'][path],'git_object':sha(old),
        'matches':sha(old)==record['code_sha256'][path], 'cache_path':str(CACHE/('historical-'+path.replace('/','_')))}
    assert sha(old)==record['code_sha256'][path]
parts=split_documents(toy_documents(),record['seed'])
for split,texts in parts.items():
    encoded=''.join(json.dumps({'text':text,'family':text},ensure_ascii=False)+'\n' for text in texts).encode()
    historical['data_hash_checks'][split]={'recorded':record['results']['data'][split]['sha256'],
        'reconstructed':sha(encoded),'matches':sha(encoded)==record['results']['data'][split]['sha256'],
        'documents':len(texts),'next_character_targets_including_document_end':sum(len(text)+1 for text in texts)}
    assert sha(encoded)==record['results']['data'][split]['sha256']
for pin in ['c95a79ac','b209f7eb']:
    original=subprocess.check_output(['git','show',pin+':docs/course-experiments/results/simple_models.json'],cwd=ROOT)
    obj=json.loads(original)
    historical['historical_record_checks'].append({'git_revision':subprocess.check_output(['git','rev-parse',pin],cwd=ROOT,text=True).strip(),
       'original_record_sha256':sha(original),'elapsed_seconds':obj['elapsed_seconds'],
       'same_run_id':obj['modal']['run_id']==record['modal']['run_id'],
       'raw_core_results_unchanged':obj['results']==record['results']})
historical['denominators']={'training_documents':9,'validation_documents':1,'test_documents':2,
    'train_targets_per_full_batch':103,'validation_targets':11,'test_targets':22,
    'configurations':4,'updates_per_configuration':200,'training_target_exposures_per_configuration':20600,
    'seed':42,'historical_timed_complete_experiment_trials':1,'warmup_trials':0,
    'timer_scope':'Includes all four toy trainings, full-batch evaluations, local saves and data preparation inside run_simple_models. Excludes image build/startup/HF upload; function_seconds is a different broader interval.'}
receipt['historical_run_audit']=historical

class ForcedIDs(nn.Module):
    def __init__(self,values):super().__init__();self.values=values;self.calls=0;self.config=SimpleNamespace(max_length=128)
    def forward(self,ids,cache=None):
        logits=torch.full((ids.shape[0],ids.shape[1],264),-100.0)
        logits[:,-1,self.values[min(self.calls,len(self.values)-1)]]=100.0
        self.calls+=1
        return {'logits':logits,'cache':None}
tok=ByteTokenizer()
receipt['generation_fixtures']={}
for name,forced,budget,expected in [('eos_before_budget',[73,2,74],32,[73,2]),('budget_without_eos',[73],2,[73,73])]:
    model=ForcedIDs(forced)
    output=generate(model,torch.tensor([[1]]),budget,0.0,eos_id=tok.eos_id)
    new=output[0,1:].tolist()
    assert new==expected
    receipt['generation_fixtures'][name]={'forced_ids':forced,'budget':budget,'new_ids':new,'calls':model.calls,
        'report':generation_report(tok,new)}
receipt['same_visible_text_different_stop']={'raw_without_eos':generation_report(tok,[73]),'raw_with_eos':generation_report(tok,[73,2])}
torch.manual_seed(42)
bigram=BigramLM(17)
with torch.no_grad():
    receipt['bigram_ignores_older_positions']=torch.equal(bigram(torch.tensor([[2,3,4]])),bigram(torch.tensor([[6,8,4]])))
assert receipt['bigram_ignores_older_positions']

# Inspect actual T.3 simple student exports; this downloads tiny public files, never publishes.
args=[sys.executable,str(ROOT/'scripts/fetch_course_models.py'),'--model','simple_models','--output',str(WORK/'checkpoints/course')]
result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=90)
assert result.returncode==0,result.stderr
receipt['simple_public_fetch']={'command':args,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'checkpoints':{}}
for name in ['bigram','mlp1','mlp3','mlp5']:
    file=WORK/f'checkpoints/course/simple_models/{name}.pt'
    obj=torch.load(file,map_location='cpu',weights_only=True)
    receipt['simple_public_fetch']['checkpoints'][name]={'sha256':sha(file.read_bytes()),'keys':sorted(obj),'format_version':obj['format_version'],
        'optimizer_or_rng_present':any(key in obj for key in ['optimizer','torch_rng','python_rng','cuda_rng','mps_rng','training_state'])}
    assert not receipt['simple_public_fetch']['checkpoints'][name]['optimizer_or_rng_present']
receipt['powershell_scope']='Official PowerShell 5.1 New-Item syntax, -Path String[] and existing-directory -Force example inspected. No Windows/PowerShell runtime is installed or claimed; Bash mkdir was executed in execution.json.'
receipt['completed_at']=datetime.now(UTC).isoformat()
(HERE/'source-audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'completed','receipt':str(HERE/'source-audit.json'),'sha256':sha((HERE/'source-audit.json').read_bytes()),'historical_elapsed_seconds':record['elapsed_seconds']},ensure_ascii=False))
