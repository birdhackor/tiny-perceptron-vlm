"""Bounded CPU checks of 18.3; read original measurements, never retrain or rescore a model."""
import ast
import copy
import hashlib
import io
import json
import os
import platform
import random
import sys
import tempfile
import time
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, CharTokenizer, IGNORE, SPECIALS, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.tokenization import generation_report

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
tok = ByteTokenizer()
print('ENV', json.dumps({'python': platform.python_version(), 'torch': str(torch.__version__),
                       'torch_git': str(torch.version.git_version), 'cuda_build': str(torch.version.cuda),
                       'device': 'cpu', 'threads': str(torch.get_num_threads())}))
namespace = dict(copy=copy, hashlib=hashlib, json=json, random=random, time=time, Path=Path,
                 torch=torch, IGNORE=IGNORE, ByteTokenizer=ByteTokenizer, render_chat=render_chat,
                 generation_report=generation_report)
tree = ast.parse((HERE / 'code/compression-recorded.py').read_bytes())
names = {'_json', '_sync', '_sha', '_chat', '_prompt', '_example', '_hard_targets'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes} == names
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'compression-recorded.py:selected-original-functions', 'exec'), namespace)
example = namespace['_example']

original = (HERE / 'fence-1.py').read_bytes()
fence_namespace = {}
print('ORIGINAL_FENCE')
exec(compile(original, 'course/chapters/18.md#18.3:original-fence', 'exec'), fence_namespace)
assert len(fence_namespace['accepted']) == 1
assert fence_namespace['accepted'][0]['messages'][1]['content'] == '4'
assert fence_namespace['accepted'][0]['teacher_revision'] is None
changed = original.replace(b'"answer": "5"', b'"answer": "4"')
assert changed != original
(HERE / 'fence-1-variation.py').write_bytes(changed)
print('FENCE_VARIATION_CHANGE_ONLY_SECOND_ANSWER')
exec(compile(changed, 'fence-1-variation.py', 'exec'), fence_namespace)
assert len(fence_namespace['accepted']) == 2
assert fence_namespace['accepted'][0] == fence_namespace['accepted'][1]

assert tok.vocab_size == 256 + len(SPECIALS) == 264
assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.user_id, tok.assistant_id,
        tok.image_id, tok.audio_id, tok.system_id) == tuple(range(8))
assert tok.encode('4') == [60]
assert CharTokenizer('45').encode('4') == [1]
assert tok.decode([tok.encode('4')[0], tok.user_id, tok.eos_id]) == '4'
report = generation_report(tok, [60, tok.user_id, tok.eos_id])
assert report['answer'] == '4<user>' and report['invalid_special_tokens'][0]['id'] == tok.user_id
print('TOKEN_CONTRACT', json.dumps({'specials': SPECIALS, 'vocab_size': tok.vocab_size,
                                  'byte_4': tok.encode('4'), 'char_4': CharTokenizer('45').encode('4'),
                                  'decode_can_hide_control': True, 'generation_report': report}, ensure_ascii=False))

measurements = json.loads((HERE / 'code/docs/course-experiments/results/distillation.json').read_text())
summary = {}
for task, expected_count, expected_correct in [('attributes', 45, 45), ('style_transfer', 185, 180)]:
    raw_task = measurements['results']['tasks'][task]
    audit = raw_task['hard_target_generation']['audit']
    assert len(audit) == expected_count
    correct = eos = zero = invalid = valid_tokens = 0
    wrong = []
    for row in audit:
        ids = row['teacher_ids']
        assert all(type(i) is int and 0 <= i < tok.vocab_size for i in ids)
        before_eos = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        hit = before_eos == tok.encode(row['gold_answer'])
        stopped = tok.eos_id in ids
        checked = generation_report(tok, ids)
        assert hit == row['teacher_correct']
        assert stopped == row['eos']
        assert len(ids) == row['valid_target_tokens']
        assert checked['answer'] == row['teacher_answer']
        for key in ['invalid_special_tokens', 'valid_answer_tokens', 'generation_status']:
            assert checked[key] == row[key]
        assert stopped and ids[-1] == tok.eos_id and ids.count(tok.eos_id) == 1
        x, y = example({'messages': [{'role': 'user', 'content': row['question']},
                                    {'role': 'assistant', 'content': row['teacher_answer']}],
                        '_hard_ids': ids, '_hard_eos': stopped}, 128)
        assert y[y != IGNORE].tolist() == ids
        correct += hit
        eos += stopped
        zero += not ids
        invalid += bool(checked['invalid_special_tokens'])
        valid_tokens += len(ids)
        if not hit:
            gold_date = date.fromisoformat(row['gold_answer'].removeprefix('已確認').removesuffix('。'))
            produced_date = date.fromisoformat(row['teacher_answer'].removeprefix('已確認').removesuffix('。'))
            assert (produced_date - gold_date).days == 10
            wrong.append({'family': row['family'], 'question': row['question'],
                          'gold_answer': row['gold_answer'], 'teacher_answer': row['teacher_answer'],
                          'day_difference': 10})
    assert correct == expected_correct and zero == invalid == 0
    summary[task] = dict(records=len(audit), correct=correct, wrong=len(wrong), eos=eos,
                         zero=zero, invalid_control_records=invalid, target_tokens=valid_tokens,
                         wrong_samples=wrong)
    assert raw_task['teacher_provenance']['config']['vocab_size'] == tok.vocab_size
assert any(r['gold_answer'] == '已確認2026-10-06。' and r['teacher_answer'] == '已確認2026-10-16。'
           for r in summary['style_transfer']['wrong_samples'])
assert sum(r['records'] for r in summary.values()) == 230
print('RAW_HARD_TARGET_AUDIT', json.dumps(summary, ensure_ascii=False))

sft = json.loads((HERE / 'sources/sft-hard-targets.json').read_text())
audit = measurements['results']['tasks']['attributes']['hard_target_generation']['audit']
assert len(sft['records']) == len(sft['audit']) == len(audit) == 45
assert sft['audit'] == audit
for record, row in zip(sft['records'], audit, strict=True):
    assert record['_hard_ids'] == row['teacher_ids']
    assert record['_hard_eos'] == row['eos']
    gold_record = copy.deepcopy(record)
    gold_record.pop('_hard_ids')
    gold_record.pop('_hard_eos')
    gold_record['messages'][-1]['content'] = row['gold_answer']
    xh, yh = example(record, 128)
    xg, yg = example(gold_record, 128)
    assert torch.equal(xh, xg) and torch.equal(yh, yg)
print('SFT_ORIGINAL_RECORDS identical_gold_and_hard_inputs_and_labels=45/45')

matched = {}
for width in (16, 32):
    runs = measurements['results']['tasks']['attributes']['runs']
    ce, hard = runs[f'w{width}_ce'], runs[f'w{width}_teacher_hard']
    keys = ['initialization_sha256', 'batch_plan_sha256', 'steps', 'training_examples',
            'effective_supervised_tokens', 'final_sha256']
    assert all(ce['training'][k] == hard['training'][k] for k in keys)
    metrics = {}
    for split in ['validation', 'test']:
        for run in [ce, hard]:
            result = run[split]
            samples = result['generated_samples']
            hit = ended = complete = 0
            for row in samples:
                ids = row['generated_ids']
                before_eos = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
                exact = before_eos == tok.encode(row['expected'])
                stop = tok.eos_id in ids
                assert exact == row['exact'] and stop == row['ended_with_eos']
                hit += exact; ended += stop; complete += exact and stop
            assert len(samples) == result['examples']
            assert result['correct'] == hit and result['completed_correct'] == complete
            assert result['eos_count'] == ended
            assert result['exact_match'] == hit / len(samples)
            assert result['completed_exact_match'] == complete / len(samples)
            assert result['eos_rate'] == ended / len(samples)
        assert ce[split]['generated_samples'] == hard[split]['generated_samples']
        for key in ['nll_sum', 'supervised_tokens', 'answer_nll', 'correct', 'exact_match',
                    'completed_correct', 'completed_exact_match', 'eos_count']:
            assert ce[split][key] == hard[split][key]
        metrics[split] = {k: ce[split][k] for k in ['correct', 'examples', 'exact_match', 'completed_correct', 'eos_count']}
    matched[str(width)] = {'training': {k: ce['training'][k] for k in keys}, 'metrics': metrics}
print('CE_AND_HARD_MATCH original_measurements', json.dumps(matched))

messages = [{'role': 'user', 'content': '2+2=?'}, {'role': 'assistant', 'content': '4'}]
prefix = namespace['_prompt']({'messages': messages}, 128)
plain = {'messages': messages}
hard = {'messages': messages, '_hard_ids': [60, 2], '_hard_eos': True}
assert all(torch.equal(a, b) for a, b in zip(example(plain, 128), example(hard, 128), strict=True))
no_eos = dict(hard, _hard_ids=[60], _hard_eos=False)
x, y = example(no_eos, 128)
assert y[y != IGNORE].tolist() == [60]
assert len(prefix) == 9 and y[len(prefix)-1].item() == 60
for value in [dict(hard, _hard_ids=[], _hard_eos=False), dict(hard, _hard_eos=False)]:
    try:
        example(value, 128)
    except ValueError:
        pass
    else:
        raise AssertionError('empty or inconsistent EOS must fail')

with tempfile.TemporaryDirectory() as temp:
    ctx = SimpleNamespace(device='cpu', output=Path(temp))
    teacher = TinyLM(ModelConfig(width=8, layers=1, max_length=128))
    calls = []
    for controlled in [[60, 2], [60], [], [60, tok.user_id]]:
        def controlled_generate(model, ids, max_new_tokens):
            calls.append({'max_new_tokens': max_new_tokens, 'prefix_shape': list(ids.shape)})
            return torch.cat((ids, torch.tensor([controlled], dtype=torch.long)), dim=1)
        namespace['generate'] = controlled_generate
        prepared, raw_report = namespace['_hard_targets'](ctx, teacher, [{'messages': messages, 'family': 'controlled'}], 'controlled', 2)
        stored = json.loads((Path(temp) / 'controlled-hard-targets.json').read_text())
        assert prepared[0]['_hard_ids'] == stored['records'][0]['_hard_ids'] == controlled
        assert stored['audit'][0]['teacher_ids'] == controlled
        assert stored['audit'][0]['eos'] == (2 in controlled)
        assert raw_report['zero_target_records'] == int(not controlled)
        assert raw_report['invalid_control_records'] == int(tok.user_id in controlled)
        if controlled:
            _, cy = example(prepared[0], 128)
            assert cy[cy != IGNORE].tolist() == controlled
        else:
            try: example(prepared[0], 128)
            except ValueError: pass
            else: raise AssertionError('zero target must not fabricate EOS')
    print('CONTROLLED_GENERATION bounded_original_hard_target_path_calls', json.dumps(calls))

x, y, valid = pad_batch([example(hard, 128), example(no_eos, 128)])
logits = torch.zeros((*x.shape, 264), requires_grad=True)
loss = masked_loss(logits, y)
assert abs(float(loss.detach()) - __import__('math').log(264)) < 1e-6
loss.backward()
assert torch.all(logits.grad[y == IGNORE] == 0)
assert (logits.grad[y != IGNORE].abs().sum(dim=-1) > 0).all()
assert y[y != IGNORE].tolist() == [60, 2, 60]
print('MASK_AND_LOSS', json.dumps({'x_shape': list(x.shape), 'logits_shape': list(logits.shape),
                                'answer_target_count': int((y != IGNORE).sum()), 'targets': y[y != IGNORE].tolist(),
                                'loss_nats': float(loss.detach()), 'expected_log264': __import__('math').log(264),
                                'ignored_gradient_exactly_zero': True, 'optimizer_updates': 0}))
print('ALL_ASSERTIONS_PASSED; no model training, new model evaluation, network access, or retained weights')
