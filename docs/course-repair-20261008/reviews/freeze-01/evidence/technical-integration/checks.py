import collections
import hashlib
import importlib.util
import json
import platform
import sys
import unicodedata
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'docs/course-repair-20261008/reviews/freeze-01'
EVIDENCE = BASE / 'evidence/technical-integration'
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((BASE / 'manifest.json').read_text())
paths = [
    'tiny_perceptron/natural_concepts.py', 'tiny_perceptron/multimodal.py',
    'tiny_perceptron/selftrained/model.py', 'tiny_perceptron/selftrained/dataset.py',
    'tiny_perceptron/selftrained/tools.py', 'tiny_perceptron/selftrained/tokenizer.py',
    'scripts/selftrained/evaluate.py', 'scripts/selftrained/prepare_vision_ocr.py',
    'scripts/selftrained/prepare_text_tools.py', 'scripts/selftrained/augment_ocr_v2.py',
    'scripts/selftrained/prepare_voice.py',
]
fingerprints = {}
for rel in paths:
    current = sha(ROOT / rel)
    frozen = sha(BASE / 'freeze/implementation' / rel)
    expected = manifest['implementation_sha256'][rel]
    assert current == frozen == expected, rel
    fingerprints[rel] = {'sha256': current, 'matches_frozen_and_manifest': True}

import torch
from tiny_perceptron.natural_concepts import text_error_report
from tiny_perceptron.selftrained.dataset import decode_ocr
from tiny_perceptron.selftrained.tools import run_tool_loop
from scripts.selftrained.evaluate import score_reply, summarize
from scripts.selftrained.prepare_text_tools import text_records, tool_records

cer = [
    text_error_report('今天去臺北', '今天去台北'),
    text_error_report('今天去臺北', '今天去臺'),
    text_error_report('今天去臺北', '今天去臺北'),
    text_error_report('', '今天去臺北'),
    text_error_report('', ''),
]
assert cer[0]['edits'] == cer[1]['edits'] == 1
assert cer[0]['cer'] == cer[1]['cer'] == .2
assert cer[0]['reference_characters'] == cer[1]['reference_characters'] == 5
assert not cer[0]['exact'] and cer[2]['exact']
assert cer[3]['cer'] is None and cer[4]['cer'] is None
nfkc = {s: unicodedata.normalize('NFKC', s) for s in ['Ａ', 'A', '臺', '台']}
assert nfkc == {'Ａ': 'A', 'A': 'A', '臺': '臺', '台': '台'}

ctc = []
for symbols, expected in [([1, 1, 0, 2, 2], '大小'), ([1, 1, 0, 1, 1], '大大')]:
    logits = torch.full((len(symbols), 13), -10.)
    logits[torch.arange(len(symbols)), symbols] = 10.
    observed = decode_ocr(logits)
    assert observed == expected
    ctc.append({'symbols': symbols, 'observed': observed, 'expected': expected})

correct_call = {'tool': 'calculator', 'operation': 'add', 'a': 3, 'b': 8}
record = {
    'task': 'tool_call', 'group_id': 'toy',
    'messages': [{'role': 'user', 'content': '請算3加8。'}, {'role': 'assistant', 'content': '11'}],
    'supervision': {'expected_call': correct_call},
}
cases = []
for call, final, expected in [
    (correct_call, '11', True),
    ({**correct_call, 'b': 9}, '12', False),
    (correct_call, '12', False),
]:
    replies = iter([json.dumps(call), final])
    history = []
    def generate(messages):
        history.append(messages)
        return next(replies)
    trace = run_tool_loop(generate, record['messages'][:-1])
    score = score_reply(record, trace['final_output'], trace)
    assert score['tool_roundtrip'] is expected
    cases.append({'trace': trace, 'score': score, 'second_generation_messages': history[1]})

summary_groups = []
for flags, expected_all, expected_changed in [
    ([True, True], 1, 1), ([True, False], 0, 1),
]:
    rows = []
    for index, okay in enumerate(flags):
        rows.append({
            'record': {'task': 'text', 'group_id': 'toy',
                       'messages': [{'role': 'assistant', 'content': ['答案甲', '答案乙'][index]}],
                       'supervision': {'pair_id': 'toy', 'control_type': 'history_choice'}},
            'score': {'semantic': okay, 'exact': okay, 'format': True},
            'trace': {'final_output': ['答案甲', '錯答乙'][index]}, 'perception': [],
        })
    observed = summarize(rows)['controls']['paired']['history_choice']
    assert observed['all_correct'] == expected_all
    assert observed['output_changed_when_gold_changed'] == expected_changed
    summary_groups.append({'semantic_flags': flags, 'observed': observed})

inputs = json.loads((BASE / 'checks/technical-inputs-manifest.json').read_text())['files_sha256']
metrics_check = {}
expected_scores = {
    'moe': {'text': (464, 560), 'ocr': (252, 324), 'vision_clothing': (357, 360),
            'vision_relation': (1427, 1440), 'voice_qa': (42, 90),
            'voice_topic_continuation': (26, 60), 'tool_call': (0, 276), 'tool_reply': (551, 552),
            'tool_concept': (6, 8), 'tool_missing': (5, 8), 'tool_unavailable': (29, 48), 'tool_unsupported': (4, 8)},
    'dense': {'text': (429, 560), 'ocr': (281, 324), 'vision_clothing': (356, 360),
              'vision_relation': (1426, 1440), 'voice_qa': (37, 90),
              'voice_topic_continuation': (24, 60), 'tool_call': (7, 276), 'tool_reply': (468, 552),
              'tool_concept': (3, 8), 'tool_missing': (8, 8), 'tool_unavailable': (24, 48), 'tool_unsupported': (4, 8)},
}
expected_controls = {
    'moe': {'swap': (720, 707, 715), 'roi': (150, 86, 150), 'history_topic': (40, 31, 40),
            'history_format': (140, 103, 138), 'history_choice': (80, 63, 66)},
    'dense': {'swap': (720, 706, 714), 'roi': (150, 110, 150), 'history_topic': (40, 33, 40),
              'history_format': (140, 113, 140), 'history_choice': (80, 0, 69)},
}
raw_fingerprints = {}
for arch in ['moe', 'dense']:
    rel = f'docs/selftrained/results/public-raw/{arch}/test/metrics.json'
    path = BASE / 'freeze/technical-data' / rel
    assert sha(path) == inputs[rel]
    raw_fingerprints[rel] = sha(path)
    data = json.loads(path.read_text())
    observed_scores = {}
    for task, expected in expected_scores[arch].items():
        metric = data['per_task_final_reply'][task]['semantic']
        observed = (metric['numerator'], metric['denominator'])
        assert observed == expected, (arch, task)
        observed_scores[task] = observed
    observed_controls = {}
    for kind, expected in expected_controls[arch].items():
        c = data['controls']['paired'][kind]
        observed = (c['pairs'], c['all_correct'], c['output_changed_when_gold_changed'])
        assert observed == expected, (arch, kind)
        observed_controls[kind] = observed
    assert sum(v['count'] for v in data['per_task_final_reply'].values()) == data['count'] == 3734
    assert data['voice_topic_continuation']['end_to_end_correct'] == expected_scores[arch]['voice_topic_continuation'][0]
    assert data['perception']['ocr']['exact_numerator'] == data['perception']['ocr']['denominator'] == 324
    assert data['thresholds']['tool_roundtrip'] == .95
    assert data['per_task_final_reply']['tool_call']['tool_roundtrip']['rate'] < .95
    class_counts = {k: v['count'] for k, v in data['stratified_final_reply'].items() if k.startswith('vision_clothing/query_class=')}
    assert set(class_counts.values()) == {120} and sum(class_counts.values()) == 360
    relfreeze = f'docs/selftrained/results/public-raw/{arch}/freeze/frozen.json'
    fpath = BASE / 'freeze/technical-data' / relfreeze
    assert sha(fpath) == inputs[relfreeze]
    raw_fingerprints[relfreeze] = sha(fpath)
    frozen = json.loads(fpath.read_text())
    assert data['thresholds'] == frozen['thresholds']
    assert data['safe_weights_sha256'] == frozen['safe_weights_sha256']
    assert frozen['selection'] == 'validation only' and frozen['test_once'] is True
    relreceipt = f'docs/selftrained/results/public-raw/{arch}/test/evaluation-receipt.json'
    rpath = BASE / 'freeze/technical-data' / relreceipt
    assert sha(rpath) == inputs[relreceipt]
    raw_fingerprints[relreceipt] = sha(rpath)
    receipt = json.loads(rpath.read_text())
    assert receipt['completed_count'] == 3734 and receipt['status'] == 'complete'
    assert receipt['protocol_sha256'] == sha(fpath)
    metrics_check[arch] = {
        'scores_match_source': observed_scores, 'controls_match_source': observed_controls,
        'all_task_counts': data['count'], 'class_counts': class_counts,
        'ctc_ocr': data['perception']['ocr'], 'voice_topic_continuation': data['voice_topic_continuation'],
        'thresholds_equal_frozen': True, 'weights_equal_frozen': True,
        'raw_outputs_expected_sha256': receipt['outputs_sha256'],
        'raw_outputs_reread': False,
    }

text_test = [row for row in text_records() + tool_records() if row['split'] == 'test']
text_bytes = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in text_test).encode()
data_manifest = json.loads((BASE / 'freeze/technical-data/docs/selftrained/v2-manifest.json').read_text())
expected_text_sha = next(x['sha256'] for x in data_manifest['records'] if x['path'] == 'text-tools-test.jsonl')
observed_text_sha = hashlib.sha256(text_bytes).hexdigest()
assert observed_text_sha == expected_text_sha
grouped = collections.defaultdict(list)
for row in text_test:
    if row['task'] == 'text':
        s = row['supervision']
        grouped[(s['control_type'], s['pair_id'])].append(row)
group_sizes = {
    kind: {'groups': sum(k == kind for k, _ in grouped),
           'group_sizes': dict(collections.Counter(len(v) for (k, _), v in grouped.items() if k == kind))}
    for kind in ['history_topic', 'history_format', 'history_choice']
}
assert group_sizes == {'history_topic': {'groups': 40, 'group_sizes': {3: 40}},
                       'history_format': {'groups': 140, 'group_sizes': {2: 140}},
                       'history_choice': {'groups': 80, 'group_sizes': {2: 80}}}
text_reconstruction = {
    'source': '已核對SHA之prepare_text_tools.text_records()+tool_records()，只在記憶體生成test資料，依write_jsonl使用sort_keys=True序列化；沒有寫入原始資料。',
    'rows': len(text_test), 'bytes': len(text_bytes), 'sha256': observed_text_sha,
    'matches_v2_manifest_sha256': True,
    'task_counts': dict(collections.Counter(row['task'] for row in text_test)),
    'control_group_sizes': group_sizes,
}

result = {
    'command': '.venv/bin/python /workspace/work/tutorial-repair-20261008/technical-integration/checks.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'device': 'CPU', 'unicode_database': unicodedata.unidata_version},
    'implementation_fingerprints': fingerprints, 'raw_data_fingerprints': raw_fingerprints,
    'CER_cases': cer, 'NFKC': nfkc, 'CTC_greedy_cases': ctc,
    'calculator_protocol_cases': cases, 'controls_changed_vs_correct_cases': summary_groups,
    'published_metrics_crosscheck': metrics_check,
    'text_test_data_reconstruction': text_reconstruction,
    'assumed_average_exercise': {'by_question': 12/18, 'by_task': (1.+0.)/2},
    'class_distribution_baseline': {'correct': 120, 'questions': 360, 'rate': 120/360},
    'limits': ['離線小算例；沒有模型生成或重訓。', '原始彙總數值對照，不是重算3734題的原始outputs.jsonl。'],
}
out = EVIDENCE / 'execution-checks.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(out), 'sha256': sha(out), 'checks': 'all explicit assertions passed',
                  'environment': result['environment']}, ensure_ascii=False))
