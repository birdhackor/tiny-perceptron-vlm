"""核對視窗算例、練習、資料準備與位置容量，不執行訓練。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import split_records, text_examples
from scripts.course_experiments.text import _utf8_prefix
from tiny_perceptron.data import ByteTokenizer, shifted
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fence(code):
    stream = io.StringIO()
    error = None
    with contextlib.redirect_stdout(stream):
        try:
            exec(compile(code, '<教材程式>', 'exec'), {})
        except AssertionError:
            error = 'AssertionError'
    return {'stdout': stream.getvalue(), 'exception': error}


def main():
    directory = Path(__file__).parent
    source = ROOT / 'docs/course-review-20261019/freeze-01/sources/6.8.md'
    code = re.search(r'```python\n(.*?)```', source.read_text(), re.S).group(1)
    original = run_fence(code)
    short_tail = run_fence(code.replace('context = 3', 'context = 4'))
    adjusted = run_fence(code.replace('context = 3', 'context = 4').replace('[0, 3, 6]', '[0, 4]'))
    assert original['exception'] is None
    assert short_tail['exception'] == 'AssertionError'
    assert adjusted['exception'] is None
    ids = list(range(10))
    cases = {}
    for context, starts in [(3, [0, 3, 6]), (4, [0, 3, 6]), (4, [0, 4])]:
        rows = []
        for start in starts:
            segment = ids[start:start+context+1]
            x, y = shifted(segment)
            rows.append({'start': start, 'segment': segment, 'x': x.tolist(), 'y': y.tolist(),
                         'dtype': str(x.dtype), 'targets': len(y), 'assert_passes': len(x) == len(y) == context})
        cases[f'context={context},starts={starts}'] = rows
    assert [row['targets'] for row in cases['context=3,starts=[0, 3, 6]']] == [3, 3, 3]
    assert [row['targets'] for row in cases['context=4,starts=[0, 3, 6]']] == [4, 4, 3]
    assert [row['targets'] for row in cases['context=4,starts=[0, 4]']] == [4, 4]
    assert 9 not in sum([row['segment'] for row in cases['context=4,starts=[0, 4]']], [])
    overlap = []
    base = cases['context=3,starts=[0, 3, 6]']
    for left, right in zip(base, base[1:]):
        shared = sorted(set(left['segment']) & set(right['segment']))
        assert shared == [left['y'][-1]] == [right['x'][0]]
        assert not set(shared) & set(right['y'])
        overlap.append({'shared_ids': shared, 'left_role': '最後一個答案', 'right_role': '第一個輸入，沒有當作答案'})
    boundary_x, boundary_y = shifted([10, 11] + [20, 21])
    boundary_pairs = list(zip(boundary_x.tolist(), boundary_y.tolist()))
    assert (11, 20) in boundary_pairs
    rows = [{'text': '貓🙂' * 80 + str(i), 'family': str(i // 2)} for i in range(12)]
    parts = split_records(rows, seed=42)
    groups = {name: {row['family'] for row in values} for name, values in parts.items()}
    assert not groups['train'] & groups['validation']
    assert not groups['train'] & groups['test']
    assert not groups['validation'] & groups['test']
    prefixes = {name: [_utf8_prefix(row['text'], 256) for row in values] for name, values in parts.items()}
    assert all(len(text.encode('utf-8')) <= 256 for values in prefixes.values() for text in values)
    ascii_prefix = _utf8_prefix('a' * 300, 256)
    assert len(ascii_prefix.encode()) == 256
    byte = ByteTokenizer()
    encoded = [byte.bos_id] + byte.encode(ascii_prefix) + [byte.eos_id]
    examples = text_examples([{'text': ascii_prefix}], mode='text', max_length=384, tokenizer=byte)
    assert len(encoded) == 258 and len(examples) == 1
    assert len(examples[0][0]) == len(examples[0][1]) == 257
    torch.manual_seed(42)
    model = TinyLM(ModelConfig(width=8, layers=1, max_length=384))
    with torch.no_grad():
        output_shape = list(model(examples[0][0][None])['logits'].shape)
    assert output_shape == [1, 257, 264]
    try:
        model(torch.ones((1, 385), dtype=torch.long))
    except ValueError as error:
        capacity_error = str(error)
    else:
        raise AssertionError('超過模型容量的輸入應被拒絕')
    tree = ast.parse((ROOT / 'scripts/course_experiments/text.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_tokenizer')
    model_call = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == 'new_lm')
    configured_capacity = ast.literal_eval(next(kw.value for kw in model_call.keywords if kw.arg == 'max_length'))
    assert configured_capacity == 384
    record_path = ROOT / 'docs/course-experiments/results/tokenizer.json'
    record = json.loads(record_path.read_text())
    runs = record['results']['runs']
    test_byte = runs['byte256']['after']['test']
    test_bpe = runs['bpe512']['after']['test']
    assert test_byte['raw_utf8_bytes'] == test_bpe['raw_utf8_bytes'] == 4193
    assert test_byte['records'] == test_bpe['records'] == 20
    content_tokens = {name: run['after']['test']['effective_tokens'] - run['after']['test']['records']
                      for name, run in runs.items()}
    assert content_tokens == {'byte256': 4193, 'bpe512': 2483}
    exposure = {name: run['training_raw_utf8_bytes_exposed'] for name, run in runs.items()}
    assert len(set(exposure.values())) == 1
    command = ['curl', '--fail', '--silent', '--show-error', '--location', '--max-time', '40',
               'https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.tolist.html']
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    from fetch_sources import PlainText
    parser = PlainText()
    parser.feed(raw.decode())
    text = re.sub(r'\s+', ' ', ' '.join(parser.parts))
    needle = 'Returns the tensor as a'
    begin = text.index(needle)
    torch_receipt = {'url': command[-1], 'version': 'PyTorch 2.14 documentation; installed torch 2.14.1',
                     'accessed_at': datetime.now(timezone.utc).isoformat(), 'command_argv': command,
                     'retrieved_bytes': len(raw), 'retrieved_sha256': hashlib.sha256(raw).hexdigest(),
                     'locator': 'torch.Tensor.tolist', 'excerpt': text[begin:begin+420]}
    inspected = ['tiny_perceptron/data.py', 'tiny_perceptron/model.py',
                 'scripts/course_experiments/text.py', 'scripts/course_experiments/common.py',
                 'docs/course-experiments/results/tokenizer.json']
    result = {
        'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-8/verify.py',
        'environment': {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu', 'platform': platform.platform()},
        'source_sha256': digest(source), 'input_file_sha256': {path: digest(ROOT / path) for path in inspected},
        'range_ids': ids, 'cases': cases,
        'frozen_code_execution': original, 'short_tail_execution': short_tail, 'adjusted_starts_execution': adjusted,
        'overlap_roles': overlap, 'unmarked_document_join_pairs': boundary_pairs,
        'split_probe': {'families': {name: sorted(value) for name, value in groups.items()},
                        'prefix_bytes': {name: [len(text.encode()) for text in values] for name, values in prefixes.items()}},
        'capacity_probe': {'configured_max_length': configured_capacity, 'raw_bytes': 256,
                           'encoded_with_bos_eos': len(encoded), 'input_and_target_length': 257,
                           'examples': len(examples), 'logits_shape': output_shape, 'over_capacity_error': capacity_error},
        'experiment_record_probe': {'record_sha256': digest(record_path), 'original_device': record.get('gpu'),
                                    'raw_bytes': 4193, 'documents': 20, 'content_tokens': content_tokens,
                                    'mean_bytes_per_content_token': {name: 4193 / count for name, count in content_tokens.items()},
                                    'training_raw_bytes_exposed': exposure, 'seed': record['seed'],
                                    'steps': {name: run['training']['steps'] for name, run in runs.items()},
                                    'training_rerun': False},
        'torch_documentation': torch_receipt,
        'result': '全部算例、預期的斷言失敗、文件家族隔離、UTF-8 截取與位置容量檢查均符合預測。',
    }
    with (directory / 'verification.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
