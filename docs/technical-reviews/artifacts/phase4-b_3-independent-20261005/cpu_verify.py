"""Independent bounded B.3 checks. Replays saved outputs; never trains or loads weights."""

import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import applications as app
from scripts.course_experiments.common import split_records
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.retrieval import call_tool

OUT = Path(__file__).resolve().parent
REPORT = json.loads((OUT / 'frozen-input/docs/course-experiments/results/tools.json').read_text())
RESULTS = REPORT['results']
TOK = ByteTokenizer()
ROLE = {'system': 7, 'user': 3, 'assistant': 4}


def independent_prompt(messages):
    ids = [1]
    for message in messages:
        ids += [ROLE[message['role']]]
        ids += [value + 8 for value in message['content'].encode('utf-8')]
        ids += [2]
    return ids + [4]


def record_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def injected_generation(text, messages):
    ids = [value + 8 for value in text.encode()] + [2]
    return {
        'unit_test_injected_output': True,
        'messages': copy.deepcopy(messages),
        'input_ids': independent_prompt(messages),
        'generated_tokens': len(ids),
        'samples': [{'generated': text, 'generated_ids': ids, 'eos': True, 'invalid_special_tokens': []}],
    }


assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
evidence = {
    'execution_kind': 'CPU arithmetic, original-data recount, and controller replay of saved generated IDs; no model inference or training',
    'environment': {'python': sys.version, 'torch': str(torch.__version__), 'device': 'cpu', 'cuda_build': str(torch.version.cuda)},
    'results_sha256': hashlib.sha256((OUT / 'frozen-input/docs/course-experiments/results/tools.json').read_bytes()).hexdigest(),
    'checked_pointers': ['/revision', '/seed', '/device', '/torch_version', '/python_version', '/gpu', '/code_sha256', '/artifacts',
                         '/results/split', '/results/protocol', '/results/accuracy', '/results/answer_accuracy',
                         '/results/completion_rate', '/results/first_action_json_rate', '/results/correct_parameters_given_executed_call',
                         '/results/actual_tool_calls', '/results/status_counts',
                         '/results/samples/*/{family,operation,question,expected,status,answer,answer_correct,required_tool_behavior,correct,max_steps,actual_tool_calls,generated_tokens,trace}',
                         '/results/samples/*/trace/*/{step,executed,parsed,finite_result,tool_result,request_matches_question,done,error,generation}',
                         '/results/samples/*/trace/*/generation/{messages,input_ids,input_tokens,samples,generated_tokens}',
                         '/results/samples/*/trace/*/generation/samples/0/{generated,generated_ids,eos,invalid_special_tokens,generated_tokens}'],
}

# The original fence is separately executed unchanged. This mutation checks propagation.
short_checks = []
for b, expected in [(45, 5535), (46, 5658)]:
    request = {'name': 'multiply', 'arguments': {'a': 123, 'b': b}}
    result = call_tool(request)
    trace = [{'role': 'assistant', 'tool_request': request}, {'role': 'tool', 'result': result}]
    assert result == expected and result == 123 * b and trace[1]['result'] == result
    short_checks.append({'request': request, 'result': result, 'trace': trace})
evidence['short_example_and_changed_b'] = short_checks

try:
    render_chat([{'role': 'user', 'content': '123乘45'}, {'role': 'tool', 'content': '5535'},
                 {'role': 'assistant', 'content': '{"done":true,"answer":5535}'}])
    raise AssertionError('tool role should be rejected')
except ValueError as error:
    evidence['unsupported_tool_role'] = {'exception': type(error).__name__, 'message': str(error)}

# Build the finite original synthetic examples, but perform no optimization.
records = app._tool_records()
assert len(records) == 210
for row in records:
    if row['operation'] == 'copy':
        assert len(row['messages']) == 3 and row['messages'][2]['role'] == 'assistant'
    else:
        assert [m['role'] for m in row['messages']] == ['system', 'user', 'assistant', 'user', 'assistant']
        answer = row['a'] + row['b'] if row['operation'] == 'add' else row['a'] * row['b']
        assert row['answer'] == answer and row['messages'][3]['content'] == f'TOOL_RESULT:{answer}'
        assert json.loads(row['messages'][4]['content']) == {'done': True, 'answer': answer}
    assert row['messages'][0] == {'role': 'system', 'content': app._TOOLS_SYSTEM}
    x, y = render_chat(row['messages'], TOK)
    full = independent_prompt(row['messages'])[:-1]
    assert x.tolist() == full[:-1]
    expected_targets = [-100]
    for message in row['messages']:
        content = [b + 8 for b in message['content'].encode()] + [2]
        expected_targets += [-100] + (content if message['role'] == 'assistant' else [-100] * len(content))
    assert y.tolist() == expected_targets[1:]
splits = split_records(records, REPORT['seed'])
split_recount = {}
for name, rows in splits.items():
    observed = {'records': len(rows), 'families': len(set(r['family'] for r in rows)), 'sha256': record_hash(rows)}
    assert observed == RESULTS['split'][name]
    split_recount[name] = observed
assert not any(set(r['family'] for r in splits[a]) & set(r['family'] for r in splits[b])
               for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
dataset_bytes = (json.dumps(splits, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
assert dataset_sha == next(a['sha256'] for a in REPORT['artifacts'] if a['path'] == 'dataset.json')
(OUT / 'reconstructed-dataset.json').write_bytes(dataset_bytes)
evidence['split_recount'] = split_recount
evidence['reconstructed_dataset'] = {'bytes': len(dataset_bytes), 'sha256': dataset_sha, 'origin': 'Original _tool_records + split_records(seed=42); byte-identical to recorded original dataset artifact SHA; not an original-data download'}

original_sample = app._sample
replayed = []
prompts = []
original_message_aliases = []
try:
    for index, (record, saved) in enumerate(zip(splits['test'], RESULTS['samples'], strict=True)):
        assert saved['question'] == record['messages'][1]['content']
        assert saved['expected'] == record['answer'] and saved['operation'] == record['operation'] and saved['family'] == record['family']
        pending = iter(saved['trace'])
        def replay(_model, messages, _ctx, **_kwargs):
            event = next(pending)
            generation = event['generation']
            expected_input = independent_prompt(messages)
            assert generation['input_ids'] == expected_input == app._prompt_ids(messages, TOK)
            assert generation['input_tokens'] == len(expected_input)
            raw = generation['samples'][0]
            ids = raw['generated_ids']
            eos = bool(ids and ids[-1] == 2)
            body = ids[:-1] if eos else ids
            decoded = bytes(token - 8 for token in body if token >= 8).decode('utf-8', errors='replace')
            assert decoded == raw['generated'] and eos == raw['eos']
            assert [token for token in body if token < 8] == raw['invalid_special_tokens']
            assert raw['generated_tokens'] == len(ids) == generation['generated_tokens']
            prompts.append({'sample_index': index, 'step': event['step'], 'input_tokens': len(expected_input), 'messages_reconstructed_from_controller_and_matched_to_raw_input_ids': copy.deepcopy(messages)})
            if generation['messages'] != messages:
                original_message_aliases.append({'sample_index': index, 'step': event['step'], 'saved_message_count': len(generation['messages']), 'actual_prompt_message_count': len(messages)})
            output = copy.deepcopy(generation)
            output['messages'] = copy.deepcopy(messages)
            return output
        app._sample = replay
        fresh = app._tool_episode(None, record, SimpleNamespace(), max_steps=saved['max_steps'])
        for key in ['family', 'operation', 'question', 'expected', 'status', 'answer', 'answer_correct', 'required_tool_behavior', 'correct', 'max_steps', 'actual_tool_calls', 'generated_tokens']:
            assert fresh[key] == saved[key], (index, key, fresh[key], saved[key])
        assert len(fresh['trace']) == len(saved['trace'])
        for new_event, old_event in zip(fresh['trace'], saved['trace'], strict=True):
            for key in ['step', 'executed', 'parsed', 'finite_result', 'tool_result', 'request_matches_question', 'done', 'error']:
                assert new_event.get(key) == old_event.get(key), (index, key)
        replayed.append(fresh)
finally:
    app._sample = original_sample

calc = [sample for sample in replayed if sample['operation'] != 'copy']
copied = [sample for sample in replayed if sample['operation'] == 'copy']
calls = [event for sample in replayed for event in sample['trace'] if event['executed']]
assert len(calc) == 18 and len(copied) == 2 and len(calls) == 18
assert all(s['actual_tool_calls'] == 1 for s in calc)
assert all(s['actual_tool_calls'] == 0 and s['correct'] and s['status'] == 'done' for s in copied)
assert all(event['request_matches_question'] and event['finite_result'] for event in calls)
recount = {'test_questions': len(replayed), 'arithmetic_questions': len(calc), 'copy_questions': len(copied),
           'actual_tool_calls': len(calls), 'matching_name_and_parameters': sum(e['request_matches_question'] for e in calls),
           'finite_tool_results': sum(e['finite_result'] for e in calls), 'completed': sum(s['status'] == 'done' for s in replayed),
           'answer_correct': sum(s['answer_correct'] for s in replayed), 'task_correct': sum(s['correct'] for s in replayed),
           'status_counts': dict(Counter(s['status'] for s in replayed))}
assert recount['actual_tool_calls'] == RESULTS['actual_tool_calls'] and recount['status_counts'] == RESULTS['status_counts']
for key, numerator, denominator in [('accuracy', recount['task_correct'], 20), ('answer_accuracy', recount['answer_correct'], 20),
                                   ('completion_rate', recount['completed'], 20), ('first_action_json_rate', sum('parsed' in s['trace'][0] for s in replayed), 20),
                                   ('correct_parameters_given_executed_call', recount['matching_name_and_parameters'], 18)]:
    assert RESULTS[key] == {'numerator': numerator, 'denominator': denominator, 'rate': numerator / denominator}
evidence['raw_sample_recount_and_replay'] = recount
evidence['generation_messages_alias_caveat'] = {'affected_first_steps': original_message_aliases, 'method': 'Original _sample returns the mutable messages list; _tool_episode later extends it in place. Match saved input_ids to independently reconstructed step input, not generation.messages, which can contain later messages. This affects metadata, not saved generated IDs or true tool results.'}
evidence['selected_examples'] = []
for question in ['CALC:add(9,9)', 'CALC:multiply(9,9)']:
    index = next(i for i, s in enumerate(replayed) if s['question'] == question)
    saved = RESULTS['samples'][index]
    evidence['selected_examples'].append({'sample_index': index, 'question': question, 'expected': saved['expected'],
        'status': saved['status'], 'answer': saved['answer'], 'actual_tool_calls': saved['actual_tool_calls'],
        'raw_generated_strings': [e['generation']['samples'][0]['generated'] for e in saved['trace']],
        'tool_results': [e['tool_result'] for e in saved['trace'] if e['executed']],
        'input_prompts': [p for p in prompts if p['sample_index'] == index]})
assert evidence['selected_examples'][0]['tool_results'] == [18]
assert evidence['selected_examples'][1]['tool_results'] == [81]
assert evidence['selected_examples'][1]['raw_generated_strings'][-1] == '{"done":true,"answer":8}}'

# Controlled variants exercise the same controller without a model.
record = {'family': 'manual:123:45', 'operation': 'multiply', 'a': 123, 'b': 45, 'answer': 5535,
          'messages': [{'role': 'system', 'content': app._TOOLS_SYSTEM}, {'role': 'user', 'content': 'CALC:multiply(123,45)'}]}
variants = [
    ('normal', ['{"name":"multiply","arguments":{"a":123,"b":45}}', '{"done":true,"answer":5535}'], 3),
    ('changed_parameter_not_repaired_from_gold', ['{"name":"multiply","arguments":{"a":123,"b":46}}', '{"done":true,"answer":5658}'], 3),
    ('malformed_final', ['{"name":"multiply","arguments":{"a":123,"b":45}}', '{"done":true,"answer":5535}}'], 3),
    ('direct_done_without_required_tool', ['{"done":true,"answer":5535}'], 3),
    ('one_action_cap', ['{"name":"multiply","arguments":{"a":123,"b":45}}'], 1),
]
variant_results = []
try:
    for label, outputs, cap in variants:
        pending = iter(outputs)
        def controlled(_model, messages, _ctx, **_kwargs):
            return injected_generation(next(pending), messages)
        app._sample = controlled
        episode = app._tool_episode(None, record, SimpleNamespace(), max_steps=cap)
        variant_results.append({'label': label, **{k: episode[k] for k in ['status', 'answer', 'answer_correct', 'required_tool_behavior', 'correct', 'actual_tool_calls']},
                                'backfilled': [e['generation']['messages'][-1] for e in episode['trace'][1:]],
                                'tool_results': [e['tool_result'] for e in episode['trace'] if e['executed']]})
finally:
    app._sample = original_sample
assert variant_results[0]['correct'] and variant_results[0]['backfilled'] == [{'role': 'user', 'content': 'TOOL_RESULT:5535'}]
assert variant_results[1]['tool_results'] == [5658] and variant_results[1]['backfilled'] == [{'role': 'user', 'content': 'TOOL_RESULT:5658'}]
assert not variant_results[1]['correct'] and not variant_results[1]['answer_correct']
assert variant_results[2]['status'] == 'invalid_request' and variant_results[2]['actual_tool_calls'] == 1
assert variant_results[3]['status'] == 'done' and variant_results[3]['answer_correct'] and not variant_results[3]['correct'] and variant_results[3]['actual_tool_calls'] == 0
assert variant_results[4]['status'] == 'step_limit' and variant_results[4]['actual_tool_calls'] == 1 and not variant_results[4]['correct']
evidence['controlled_protocol_variants'] = variant_results
(OUT / 'reconstructed-step-prompts.json').write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + '\n')
(OUT / 'cpu-verification.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': 'all assertions passed', 'recount': recount, 'selected_examples': evidence['selected_examples'], 'controlled_variants': variant_results,
                  'environment': evidence['environment']}, ensure_ascii=False, indent=2))
