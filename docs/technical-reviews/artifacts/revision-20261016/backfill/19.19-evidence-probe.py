"""Read-only source, initialization and existing raw-score checks; no training."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver, Tokenizer
from tiny_perceptron.lesson_products.training import state_digest

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/backfill'
WORK = Path('/workspace/work/curriculum-20261016/technical-19.19/official')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
result = {'reviewer_task': '/root/backfill_tech_19_19', 'environment': {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu', 'platform': platform.platform()}, 'scope': 'Read-only existing results/checkpoints and reconstruction of random initialization without optimizer updates. No heldout model evaluation or training.'}
frozen = json.loads((ROOT/'docs/course-revision-20261016/freeze-14.json').read_text())
entry = next(p for p in frozen['pages'] if p['page_id']=='19.19')
body = lesson_slices((ROOT/'course/chapters/19.md').read_text())['19.19']
result['frozen_source'] = {'freeze': 'docs/course-revision-20261016/freeze-14.json', 'freeze_sha256': sha(ROOT/'docs/course-revision-20261016/freeze-14.json'), 'snapshot':entry['snapshot'], 'snapshot_sha256':sha(ROOT/entry['snapshot']), 'current_slice_sha256':hashlib.sha256(body.encode()).hexdigest(), 'exact_snapshot_match':body==(ROOT/entry['snapshot']).read_text(), 'figures':entry['figures_sha256']}
result['random_initialization'] = {}
torch.set_num_threads(2)
for product,seed,width,ff,cls in [('b',7421,128,256,LoopedSolver),('c',9041,64,128,CandidateScorer)]:
    torch.manual_seed(seed)
    obj=cls(Config(len(Tokenizer()),width=width,ff=ff))
    expected=json.loads((ROOT/f'outputs/lesson-products/products/{product}/training-receipt.json').read_text())['initial_weights_sha256']
    actual=state_digest(obj.state_dict())
    result['random_initialization'][product]={'seed':seed,'observed_initial_weights_sha256':actual,'primary_receipt_initial_weights_sha256':expected,'matches':actual==expected,'training_updates_executed':0}

old=ROOT/'docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json'
receipt=json.loads(old.read_text())
result['old_a_primary_cpu_receipt']={'path':str(old.relative_to(ROOT)),'sha256':sha(old),'HF_commit':receipt['HF_commit'],'chat_invocations':[{'task':x['task'],'returncode':x['returncode'],'actual_answer':x.get('actual_answer'),'revision':x['argv'][x['argv'].index('--revision')+1]} for x in receipt['invocations'] if x.get('chat')], 'scope':'Existing primary execution receipt, not personally rerun. Only A historical public CPU route and version checked; no heldout score claims assessed.'}

raw=ROOT/'outputs/lesson-products/final-report-v1.predictions.jsonl'
examples=json.loads((ROOT/'docs/lesson-products/results/observed-failure-examples-v1.json').read_text())
targets={e['id']:e for e in examples['examples']}
found={}
first_b_error=None
for line in raw.open():
    row=json.loads(line)
    if first_b_error is None and row['product']=='b' and row['answers_by_depth']['5']!=row['gold']:
        scores=row['scores_by_depth']['5']; predicted=max(range(len(scores)),key=scores.__getitem__)
        first_b_error={'id':row['id'],'task':row['task'],'split':row.get('split'),'gold':row['gold'],'recorded_answer':row['answers_by_depth']['5'],'score_argmax':predicted,'scores':scores}
    if row['product']=='c' and row['id'] in targets and row.get('question')=='choice':
        expected=targets[row['id']]
        found[row['id']]={'raw_row':row,'excerpt_state':expected['state'],'excerpt_scores_match':row['raw_scores']==expected['raw_scores'],'excerpt_gold_matches':row['gold']==expected['gold'],'recomputed_predicted':row['candidates'][max(range(len(row['raw_scores'])),key=row['raw_scores'].__getitem__)]}
    if len(found)==len(targets) and first_b_error is not None:
        break
result['existing_error_checks']={'raw_path':str(raw.relative_to(ROOT)),'raw_sha256':sha(raw),'expected_raw_sha256':examples['raw_predictions_sha256'],'b_one_observed_error':first_b_error,'c_selected_errors':list(found.values()),'denominator_scope':'One B error and three C selected wording-shift errors checked; these selected examples do not estimate error rates.'}

result['training_cli'] = []
for args in [
    ['--help'],
    ['--job','docs/lesson-products/jobs/train-b-v1.json','--output','/workspace/work/curriculum-20261016/technical-19.19','--device','cpu'],
]:
    cmd=[str(ROOT/'.venv/bin/python'),'-m','scripts.lesson_products.train_local',*args]
    proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=15)
    result['training_cli'].append({'command':cmd,'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'training_started':False})
result['additional_source_sha256']={p:sha(ROOT/p) for p in ['tiny_perceptron/lesson_products/evaluation.py','docs/lesson-products/results/b-auto-policy.json','docs/lesson-products/results/observed-failure-examples-v1.json','docs/lesson-products/results/publication-v1.json','docs/lesson-products/results/publication-package-v1.json','docs/lesson-products/jobs/train-b-v1.json','docs/lesson-products/jobs/train-c-v1.json']}
(BASE/'19.19-existing-evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'frozen_source_exact':result['frozen_source']['exact_snapshot_match'],'random_initializations':result['random_initialization'],'C_selected_examples_checked':len(found),'B_error_checked':first_b_error is not None,'training_cli_exit_codes':[x['exit_code'] for x in result['training_cli']]},ensure_ascii=False))
