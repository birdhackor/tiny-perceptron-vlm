"""T.11 independent checks: no training and no formal evaluation rerun."""

import collections
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from scripts.reading_time import lesson_slices
from tiny_perceptron.lesson_products.delivery import download_product
from tiny_perceptron.lesson_products.inference import Product

DEST = Path('/workspace/work/curriculum-20261016/T.11')
OUT = Path(__file__).with_name('T.11-check-output.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def serialize(messages):
    result = [1]
    for message in messages:
        result += [{'system': 7, 'user': 3, 'assistant': 4}[message['role']]]
        result += [b + 8 for b in message['content'].encode()]
        result += [2]
    return result + [4]


torch.set_num_threads(2)
result = {'reviewer_task': '/root/backfill_tech_t_11',
          'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                          'platform': platform.platform(), 'device': 'cpu'},
          'scope': 'Own CPU download/inference, deterministic receipt and raw-score checks; historical GPU/final tests are retained primary evidence, not rerun.'}
freeze_path = 'docs/course-revision-20261016/freeze-11/sources/training.md'
body = lesson_slices((ROOT / freeze_path).read_text())['T.11']
assert body == lesson_slices((ROOT / 'course/training.md').read_text())['T.11']
result['section_sha256'] = hashlib.sha256(body.encode()).hexdigest()
result['hand_example'] = {'old_bytes': 80000, 'new_bytes': 60000, 'smaller': 60000 < 80000,
                          'old_correct': 4, 'new_correct': 3, 'same_denominator': 5,
                          'accuracy_non_decrease': 3 >= 4,
                          'same_aggregate_different_errors': {'old_correct_ids': [1,2,3,4], 'new_correct_ids': [1,2,3,5]}}

# Recalculate only the T.11 claims from the already saved original predictions.
raw = ROOT / 'outputs/lesson-products/final-report-v1.predictions.jsonl'
report = load('docs/lesson-products/results/final-report-v1.json')
assert raw.stat().st_size == report['predictions']['bytes']
assert digest(raw) == report['predictions']['sha256']
counts = collections.Counter()
correct = collections.Counter()
families = collections.defaultdict(set)
high_errors = collections.Counter()
temperatures = load('docs/lesson-products/results/settings-freeze-v1.json')['calibration']['temperatures']
for line in raw.open():
    row = json.loads(line)
    if row['product'] == 'b' and row['split'] == 'test':
        key = 'b_' + row['task']
        counts[key] += 1
        correct[key] += row['answers_by_depth']['5'] == row['gold']
        families[key].add(row['family_id'])
    elif row['product'] == 'c':
        key = 'c_' + row['split'] + '_' + row['question']
        scores = row['raw_scores']
        selected = row['candidates'][max(range(len(scores)), key=lambda i: scores[i])]
        counts[key] += 1
        correct[key] += selected == row['gold']
        if row['split'] == 'test':
            assert selected == row['selected']
        elif selected != row['gold']:
            z = [x / temperatures[row['question']] for x in scores]
            z = [math.exp(x-max(z)) for x in z]
            pmax = max(z) / sum(z)
            concentration = (len(z)*pmax-1)/(len(z)-1)
            high_errors[key] += concentration >= 0.9
for task in ('boxes','sum'):
    expected = report['b']['5'][task]
    assert (counts['b_'+task],correct['b_'+task]) == (expected['count'],expected['correct'])
for question in ('choice','info','rubric'):
    for split, expected in [('test',report['c'][question]['after']),
                            ('wording_shift',report['shifts']['c_wording'][question]['after'])]:
        key = 'c_'+split+'_'+question
        assert (counts[key],correct[key]) == (expected['count'],expected['correct'])
result['raw_scores'] = {'path': str(raw.relative_to(ROOT)), 'sha256': digest(raw),
                        'counts': dict(counts), 'correct': dict(correct),
                        'b_family_counts': {k:len(v) for k,v in families.items()},
                        'c_wording_errors_concentration_ge_0_9': dict(high_errors),
                        'temperature': temperatures,
                        'gate_protocol': load('docs/lesson-products/results/settings-freeze-v1.json')['protocol'],
                        'reported_gates': report['gates']}

result['training_receipts'] = {}
for product in ('b','c'):
    path = f'docs/lesson-products/results/initial-{product}-execution.json'
    execution = load(path)
    receipt_path = f'outputs/lesson-products/products/{product}/training-receipt.json'
    receipt = load(receipt_path)
    assert execution['status'] == 'completed'
    assert execution['gpu_name'] == 'NVIDIA L4'
    assert receipt == execution['training']
    assert receipt['random_initialization'] and receipt['resume_source'] is None and receipt['weights_changed']
    result['training_receipts'][product] = {'execution_sha256':digest(ROOT/path),
         'receipt_sha256':digest(ROOT/receipt_path),
         **{key:receipt[key] for key in ('random_initialization','resume_source','last_step','selected_step','seed','weights_changed','initial_weights_sha256','selected_weights_sha256','environment')}}

# Read public immutable package with urllib only; this helper never sends Authorization.
release = load('tiny_perceptron/lesson_products/release.json')
result['own_public_cpu'] = {}
for product in ('b','c'):
    entry = release['products'][product]
    directory = DEST / ('public-'+product)
    download = download_product(release['repo'],release['revision'],entry['prefix'],entry['manifest_sha256'],directory,product)
    obj = Product(directory,product,'cpu')
    observation = obj.solve('兩盒追蹤;A=2;B=6;事件=交換;查詢=B',3) if product == 'b' else obj.judge('卡片進度;申請編號=?','info')
    assert observation['answer'] == 2 if product == 'b' else observation['selected'] == 'no'
    result['own_public_cpu'][product] = {'download':download,'example_output':observation}

# Rebuild every original tool prompt; keep original report unchanged.
tools_path = 'docs/course-experiments/results/tools.json'
tools_before = digest(ROOT/tools_path)
tools = load(tools_path)['results']
seen = mismatches = total_tokens = finite_executions = 0
for key in ('samples','one_step_cap_samples'):
    for episode in tools[key]:
        first = next(event['generation'] for event in episode['trace'] if 'generation' in event)
        current = copy.deepcopy(first['messages'][:2])
        assert current[1]['content'] == episode['question']
        for event in episode['trace']:
            if 'generation' not in event:
                continue
            generation = event['generation']
            ids = serialize(current)
            assert ids == generation['input_ids'] and len(ids) == generation['input_tokens']
            seen += 1
            total_tokens += len(ids)
            mismatches += current != generation['messages']
            if event.get('executed') and event.get('finite_result'):
                action = event['parsed']; args = action['arguments'];a,b=args['a'],args['b']
                expected = {'add':lambda:a+b,'subtract':lambda:a-b,'multiply':lambda:a*b,'divide':lambda:a/b}[action['name']]()
                assert expected == event['tool_result']
                finite_executions += 1
                current += [{'role':'assistant','content':generation['samples'][0]['generated']},
                            {'role':'user','content':f"TOOL_RESULT:{event['tool_result']}"}]
assert (seen,mismatches) == (56,36)
assert digest(ROOT/tools_path) == tools_before
legacy_source = subprocess.run(['git','show','910aebc6419c9fc6217279a27fde9851c5cfad30:scripts/course_experiments/applications.py'],cwd=ROOT,capture_output=True,check=True).stdout
assert hashlib.sha256(legacy_source).hexdigest() == load(tools_path)['code_sha256']['scripts/course_experiments/applications.py']
legacy_text = legacy_source.decode()
assert '"messages": messages,' in legacy_text and 'messages += [' in legacy_text
result['historical_tools'] = {'original_sha256':tools_before,'rebuilt_generations':seen,
     'changed_message_metadata':mismatches,'input_tokens_checked':total_tokens,
     'finite_tool_results_recalculated':finite_executions,
     'correct_normal':sum(x['correct'] for x in tools['samples']), 'normal_count':len(tools['samples']),
     'historical_source_revision':'910aebc6419c9fc6217279a27fde9851c5cfad30',
     'historical_source_sha256':hashlib.sha256(legacy_source).hexdigest(),
     'mutation_locator':'_sample returns messages itself; _tool_episode appends executed request and finite result with messages +=; input_ids is a new list.'}

rag = load('docs/course-experiments/results/rag.json')['results']
assert all('input_ids' in x['generation'] and x['generation']['samples'] and 'expected' in x for x in rag['samples'])
reason = load('docs/course-experiments/results/reasoning.json')['results']
reasoning_counts = {}
for mode, group in reason['comparison'].items():
    reasoning_counts[mode] = []
    for budget in group['budgets']:
        assert len(budget['samples']) == budget['majority_accuracy']['denominator']
        assert all('input_ids' in x['generation'] and x['candidates'] and 'truth' in x for x in budget['samples'])
        reasoning_counts[mode].append({'candidate_count':budget['candidate_count'],'samples':len(budget['samples'])})
result['historical_records'] = {'rag_saved_samples':len(rag['samples']),
                                'rag_groups':dict(collections.Counter(x['mode'] for x in rag['samples'])),
                                'reasoning_saved_samples':reasoning_counts}

v2 = load('docs/selftrained/results/v2-final-public-results.json')
v2test = load('docs/selftrained/results/public-raw/moe/test/evaluation-receipt.json')
v2cpu = load('docs/selftrained/results/public-cpu-raw/tool_call-1-result.json')
manifest = load('docs/selftrained/v2-manifest.json')
v2train = load('docs/selftrained/results/training-raw/moe-pretrain/raw/train-receipt.json')
assert v2test['completed_count'] == v2test['expected_count'] == 3734 and v2test['status']=='complete'
assert v2cpu['returncode']==0 and v2cpu['public_source']['authentication']=='disabled'
assert v2train['origin']['kind']=='all-neural-weights-random' and v2train['completed_requested_steps']
assert manifest['package']['kind']=='git-lfs' and manifest['initialization']=='random'
result['v2_existing_primary_receipts'] = {'hf_revision':v2['hf_revision'],
    'moe_fixed_final_count':v2test['completed_count'],'final_checkpoint_sha256':v2test['checkpoint_sha256'],
    'inference_export_sha256':v2['architectures']['moe']['safe_weights']['sha256'],
    'pretrain_origin':v2train['origin'],'lfs_package':manifest['package'],
    'local_record_files':len(manifest['records']),'local_assets':len(manifest['assets']),
    'cpu_anonymous_example':{'returncode':v2cpu['returncode'],'answer':v2cpu['actual_answer'],'public_source':v2cpu['public_source']},
    'scope':'Read historical primary receipts, did not rerun old production training/test/inference.'}
natural = load('docs/natural-assistant/v4/public-release.json')
result['mature_extension'] = {key:natural[key] for key in ('selected_variant','base_model','asr_model','adapter_parameters')}

files = [freeze_path,'docs/lesson-products/results/final-report-v1.json',
 'docs/lesson-products/results/settings-freeze-v1.json','docs/lesson-products/results/pretest-freeze-record-v1.json',
 'docs/lesson-products/results/initial-b-execution.json','docs/lesson-products/results/initial-c-execution.json',
 'docs/lesson-products/results/public-cpu-acceptance-v1.json','tiny_perceptron/lesson_products/release.json',
 'tiny_perceptron/lesson_products/__main__.py','tiny_perceptron/lesson_products/delivery.py',
 'tiny_perceptron/lesson_products/model.py','tiny_perceptron/lesson_products/train.py',
 'tiny_perceptron/lesson_products/evaluation.py','scripts/lesson_products/train_local.py',
 'docs/lesson-products/README.md','docs/course-experiments/results/rag.json',
 'docs/course-experiments/results/reasoning.json',tools_path,
 'docs/course-repair-20261008/tool-trace-errata.md','docs/selftrained/TRAINING.md',
 'scripts/selftrained/train_local_stage.py','docs/selftrained/v2-manifest.json',
 'docs/selftrained/results/v2-final-public-results.json',
 'docs/selftrained/results/public-raw/moe/test/evaluation-receipt.json',
 'docs/selftrained/results/public-cpu-raw/tool_call-1-result.json',
 'docs/selftrained/results/training-raw/moe-pretrain/raw/train-receipt.json',
 'docs/natural-assistant/v4/public-release.json','docs/natural-assistant/v4/TRAINING.md']
result['source_sha256'] = {path:digest(ROOT/path) for path in files}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(OUT.relative_to(ROOT)),'sha256':digest(OUT),'raw_counts':dict(counts),
                   'correct':dict(correct),'high_concentration_errors':dict(high_errors),
                   'tools':result['historical_tools'],'all_assertions':'passed'},ensure_ascii=False,indent=2))
