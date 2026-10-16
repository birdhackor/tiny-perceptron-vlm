"""Read-only bounded CPU checks for frozen 19.13, no training or formal test rerun."""
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices, source_sha256
from tiny_perceptron.lesson_products.data import answer
from tiny_perceptron.lesson_products.inference import EFFORT, Product, auto_solve, parse_state
from tiny_perceptron.lesson_products.model import CandidateScorer, LoopedSolver
from tiny_perceptron.lesson_products.training import state_digest

ROOT = Path.cwd()
OUT = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/backfill/19.13-execution.json'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_bytes())

torch.set_num_threads(1)
freeze = read('docs/course-revision-20261016/freeze-15.json')
entry = next(p for p in freeze['pages'] if p['page_id'] == '19.13')
body = lesson_slices(Path('course/chapters/19.md').read_text())['19.13']
result = {'command': 'uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/backfill/19.13-probe.py',
          'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'platform': platform.platform()},
          'scope': 'Reviewer executed only CPU inference, source/receipt identity inspection, and a read-only recount of existing predictions. Did not train, update parameters, or restart formal evaluation.',
          'source': {'freeze': 'docs/course-revision-20261016/freeze-15.json', 'snapshot': entry['snapshot'], 'slice_sha256': source_sha256(body),
                     'current_equals_snapshot': body == Path(entry['snapshot']).read_text(), 'freeze_sha_matches': source_sha256(body) == entry['source_sha256']}}
products = {p: Product(f'outputs/lesson-products/products/{p}', p, 'cpu') for p in ('b', 'c')}
identity = {}
settings = read('docs/lesson-products/results/settings-freeze-v1.json')
for p, product in products.items():
    receipt = read(product.directory / 'training-receipt.json')
    runner_path = f'docs/lesson-products/results/initial-{p}-execution.json'
    runner = read(runner_path)
    torch.manual_seed(receipt['seed'])
    fresh = (LoopedSolver if p == 'b' else CandidateScorer)(product.config)
    initial_digest = state_digest(fresh.state_dict())
    selected_digest = state_digest(product.model.state_dict())
    best = torch.load(product.directory / 'best.pt', map_location='cpu', weights_only=True)
    identity[p] = {'model_type': type(product.model).__name__, 'parameters': sum(t.numel() for t in product.model.parameters()),
                   'training': {k: receipt[k] for k in ('run_identity','product','mode','random_initialization','resume_source','steps_this_attempt','last_step','selected_step','seed','initial_weights_sha256','selected_weights_sha256','weights_changed','environment')},
                   'runner_receipt_path': runner_path, 'runner_receipt_sha256': sha(runner_path),
                   'runner_completed': runner['status'] == 'completed', 'runner_training_matches_local_receipt': runner['training'] == receipt,
                   'reconstructed_seed_initial_digest': initial_digest, 'matches_recorded_initial': initial_digest == receipt['initial_weights_sha256'],
                   'loaded_selected_digest': selected_digest, 'matches_recorded_selected': selected_digest == receipt['selected_weights_sha256'],
                   'best_checkpoint_matches_selected': state_digest(best['model']) == selected_digest,
                   'every_frozen_product_file_matches': all(sha(product.directory / k) == v for k,v in settings['products'][p].items())}
result['identity'] = identity
state = '兩盒追蹤;A=2;B=6;事件=A設5,A設1;查詢=A'
world = {'task': 'boxes','a':2,'b':6,'events': [('A',5),('A',1)],'query':'A'}
result['b_example'] = {'state': state, 'world_gold': answer(world), 'answers': {str(n): products['b'].solve(state,n) for n in (1,3,5)}}
counts=[]
handle = products['b'].model.shared.register_forward_hook(lambda *args: counts.append(1))
products['b'].solve(state,5)
handle.remove()
result['b_example']['observed_shared_block_calls_at_5'] = len(counts)
result['b_sum_input'] = {'state':'三數加法;x=1;y=2;z=3', 'gold':6, 'output':products['b'].solve('三數加法;x=1;y=2;z=3',5)}
result['c_card_missing'] = {q: products['c'].judge('卡片進度;申請編號=?',q) for q in ('choice','info','rubric')}
result['c_info_output_keys'] = sorted(result['c_card_missing']['info'])
result['c_info_has_clarification_or_rationale'] = any(k in result['c_info_output_keys'] for k in ('clarification','clarification_needed','reason','rationale','evidence','suggestion'))
result['auto_example'] = auto_solve(products['b'],products['c'],state,effort='medium')
result['auto_missing_boxes'] = auto_solve(products['b'],products['c'],'兩盒追蹤;A=?;B=6;事件=交換;查詢=B')
result['auto_card_missing'] = auto_solve(products['b'],products['c'],'卡片進度;申請編號=?')
result['effort'] = EFFORT
result['paraphrases_are_not_literal_api_inputs'] = {}
for text in ('起始狀態2；設成5；設成1','查卡片，沒給申請編號'):
    try: result['paraphrases_are_not_literal_api_inputs'][text] = parse_state(text)
    except ValueError as exc: result['paraphrases_are_not_literal_api_inputs'][text] = str(exc)
help_run = subprocess.run([str(ROOT/'.venv/bin/python'),'-m','tiny_perceptron.lesson_products','--help'],check=True,text=True,capture_output=True)
result['common_cli_help'] = help_run.stdout
result['common_http_routes'] = {'source':'tiny_perceptron/lesson_products/delivery.py:serve', 'health_products':['b','c'], 'post_routes':['/solve','/judge','/auto'], 'a_selector_found':False}
primary = read('docs/lesson-products/results/final-report-v1.json')
raw = Path('outputs/lesson-products/final-report-v1.predictions.jsonl')
counts = Counter(); correct=Counter(); row_keys={}
with raw.open() as stream:
    for line in stream:
        row=json.loads(line)
        p=row['product']
        row_keys.setdefault(p, sorted(row))
        if p == 'b' and row['split'] == 'test':
            for depth, pred in row['answers_by_depth'].items():
                key=f'b/{depth}/{row["task"]}'; counts[key]+=1;correct[key]+=pred==row['gold']
        elif p == 'c' and row['split'] == 'test':
            key=f'c/{row["question"]}';counts[key]+=1;correct[key]+=row['selected']==row['gold']
result['existing_predictions']={'path':str(raw), 'sha256':sha(raw), 'matches_report_sha':sha(raw)==primary['predictions']['sha256'],
                                'row_keys_by_product':row_keys,'recount':{k:{'count':counts[k],'correct':correct[k]} for k in sorted(counts)},
                                'settings_sha_matches':sha('docs/lesson-products/results/settings-freeze-v1.json')==primary['settings_freeze_sha256'],
                                'report_gates':primary['gates'], 'report_scope':primary['scope']}
for depth in map(str, range(1,7)):
    for task in ('boxes','sum'):
        key=f'b/{depth}/{task}'; expected=primary['b'][depth][task]
        assert counts[key]==expected['count'] and correct[key]==expected['correct'],key
for q in ('choice','info','rubric'):
    key=f'c/{q}';expected=primary['c'][q]['after']
    assert counts[key]==expected['count'] and correct[key]==expected['correct'],(key,primary['c'][q])
a_path='docs/selftrained/results/public-raw/moe/test/metrics.json';a=read(a_path)
summary=read('docs/selftrained/results/v2-final-public-results.json')
moe=summary['architectures']['moe']
result['a_existing_primary']={'path':a_path,'sha256':sha(a_path),'matches_published_metrics_sha':sha(a_path)==moe['final_test']['metrics_sha256'],
 'architecture':moe['capacity']['config']['architecture'],'initialization':moe['capacity']['initialization'],
 'selected_source':moe['selected_source'],'safe_weights_sha256':moe['safe_weights']['sha256'],'raw_safe_weights_sha256':a['safe_weights_sha256'],
 'count':a['count'],'thresholds':a['thresholds'],
 'voice_qa':a['per_task_final_reply']['voice_qa'],'tool_call':a['per_task_final_reply']['tool_call'],
 'capability_acceptance_passed':moe['capability_acceptance_passed'],'data_scope':summary['data_scope'],
 'scope':'Read existing primary metric receipt and compare its file hash to public version. No rerun of A or re-evaluation of all criteria.'}
inputs=['docs/course-revision-20261016/freeze-10/sources/19.13.md','tiny_perceptron/lesson_products/model.py','tiny_perceptron/lesson_products/inference.py','tiny_perceptron/lesson_products/data.py','tiny_perceptron/lesson_products/train.py','tiny_perceptron/lesson_products/training.py','tiny_perceptron/lesson_products/delivery.py','tiny_perceptron/lesson_products/__main__.py','docs/lesson-products/results/final-report-v1.json','docs/lesson-products/results/settings-freeze-v1.json','docs/lesson-products/results/pretest-freeze-record-v1.json','docs/lesson-products/results/final-report-v1.started.json','docs/selftrained/results/v2-final-public-results.json',a_path]
result['checked_input_sha256']={p:sha(p) for p in inputs}
result['source_files_sha256']={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'saved':str(OUT.relative_to(ROOT)),'identity':{p:{k:v for k,v in d.items() if isinstance(v,bool)} for p,d in identity.items()},'b_example':result['b_example'],'c_info':result['c_card_missing']['info'],'raw_recount':result['existing_predictions']['recount']},ensure_ascii=False,indent=2))
