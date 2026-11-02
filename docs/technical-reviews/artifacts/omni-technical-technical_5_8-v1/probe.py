"""5.8 的短 CPU 審閱：執行正文、重算資料與既有記錄，不訓練。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.course_experiments.common import split_records, text_examples
from scripts.course_experiments.text import _deduplicate_text
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer

CACHE = Path('/workspace/work/reference/omni-technical-technical_5_8-v1')
torch.set_num_threads(1)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def section(path, number):
    text = (ROOT / path).read_text()
    return re.search(r'^## ' + re.escape(number) + r' .*?(?=^## |\Z)', text, re.M | re.S).group()


def execute(code):
    output, scope = io.StringIO(), {}
    with contextlib.redirect_stdout(output):
        exec(compile(code, 'frozen-5.8-code', 'exec'), scope)
    return output.getvalue()


def split_check(parts, recorded):
    result = {}
    for name, rows in parts.items():
        raw = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
        assert digest(raw) == recorded[name]['sha256']
        examples = text_examples(rows, max_length=128)
        tokens = sum(int((y != -100).sum()) for _, y in examples)
        byte_count = sum(len(row['text'].encode()) for row in rows)
        assert tokens == byte_count + len(rows)
        for row in rows:
            chunks = text_examples([row], max_length=128)
            assert all(x.dtype == torch.int64 and y.dtype == torch.int64 for x, y in chunks)
            assert all(len(x) <= 128 for x, _ in chunks)
            assert [i for _, y in chunks for i in y.tolist()] == ByteTokenizer().encode(row['text']) + [2]
        result[name] = {'records': len(rows), 'jsonl_sha256': digest(raw), 'effective_targets': tokens,
                        'raw_utf8_bytes': byte_count, 'chunks': len(examples),
                        'all_labels_once_including_one_eos_per_record': True}
    return result


def record_check(record, parts, report):
    output = {}
    tok = ByteTokenizer()
    for time in ['before', 'after']:
        output[time] = {}
        for split in ['validation', 'test']:
            ev = record['results'][time][split] if 'runs' not in record['results'] else record['results']['runs']['tinystories'][time][split]
            nll = ev['nll_sum'] / ev['effective_tokens']
            assert abs(nll - ev['nll']) < 1e-12
            assert ev['effective_tokens'] == report[split]['effective_targets']
            assert ev['records'] == report[split]['records']
            assert ev['examples'] == report[split]['chunks']
            assert len(ev['samples']) == min(8, len(parts[split]))
            for row, sample in zip(parts[split][:8], ev['samples'], strict=True):
                prefix = row['text'].encode()[:24].decode(errors='ignore')
                assert sample['prompt'] == prefix
                assert tok.decode(sample['generated_ids']) == sample['generated']
                assert len(sample['generated_ids']) <= 32
            output[time][split] = {'nll': nll, 'rounded_5dp': f'{nll:.5f}',
                                  'nll_sum': ev['nll_sum'], 'effective_tokens': ev['effective_tokens'],
                                  'samples': len(ev['samples']), 'sample_prefix_and_id_decode_verified': True}
    return output


def main():
    target = Path(sys.argv[1])
    if target.exists():
        raise FileExistsError(target)
    result = {'started_at': datetime.now(UTC).isoformat(),
              'reviewer_task': '/root/omni_technical_coordinator/technical_5_8',
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'platform': platform.platform(), 'device': 'cpu', 'threads': '1'},
              'scope': 'No training and no saved-model forward replay. Recompute the literal lesson and primary recorded-run arithmetic/data/IDs.'}
    freeze = json.loads((ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json').read_bytes())
    page = next(p for p in freeze['pages'] if p['page_id'] == '5.8')
    body = (ROOT / page['snapshot']).read_text()
    assert digest(body.encode()) == page['source_sha256'] == digest(section('course/chapters/05.md', '5.8').encode())
    assert digest((ROOT / page['notebook']).read_bytes()) == page['notebook_sha256']
    assert not re.findall(r'!\[[^\]]*\]\([^)]*\.svg\)', body) and page['figures_sha256'] == {}
    code = re.findall(r'```python\n(.*?)```', body, re.S)[0]
    notebook = json.loads((ROOT / page['notebook']).read_bytes())
    assert any(''.join(c['source']).strip() == code.strip() for c in notebook['cells'] if c['cell_type'] == 'code')
    result['frozen_source'] = {'sha256': page['source_sha256'], 'notebook_sha256': page['notebook_sha256'], 'figures': {}}
    result['literal_code'] = {'sha256': digest(code.encode()), 'stdout': execute(code)}
    assert result['literal_code']['stdout'] == '正確前文下 逐token正確率 0.75 整串相同 False\n自由生成 逐token正確率 0.5 整串相同 False\n'
    changed = code.replace('teacher_forced = [1, 2, 0, 4]', 'teacher_forced = [1, 2, 3, 4]')
    result['exercise_stdout'] = execute(changed)
    assert result['exercise_stdout'] == '正確前文下 逐token正確率 1.0 整串相同 True\n自由生成 逐token正確率 0.5 整串相同 False\n'
    errors = []
    for prediction in [[1, 2, 3], [1, 2, 3, 4, 5]]:
        try:
            sum(a == b for a, b in zip([1, 2, 3, 4], prediction, strict=True))
        except ValueError as error:
            errors.append(str(error))
    assert len(errors) == 2
    result['python_contract'] = {'unequal_length_errors': errors, 'bool_sum': sum([True, False, True]),
                                 'list_equality': {'same': [1, 2] == [1, 2], 'order': [1, 2] == [2, 1], 'length': [1, 2] == [1, 2, 3]}}
    result['cross_entropy_same_argmax'] = []
    for probability in [0.6, 0.9]:
        probabilities = torch.tensor([[probability, 1 - probability]], dtype=torch.float64)
        loss = F.cross_entropy(probabilities.log(), torch.tensor([0]))
        assert abs(loss.item() + math.log(probability)) < 1e-14
        result['cross_entropy_same_argmax'].append({'true_probability': probability, 'argmax': int(probabilities.argmax()), 'nll': loss.item()})
    r = json.loads((ROOT / 'docs/course-experiments/results/real_text.json').read_bytes())
    t = json.loads((ROOT / 'docs/course-experiments/results/text_foundation.json').read_bytes())
    with tarfile.open(ROOT / 'assets/training/tinystories-v1.tar.gz') as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith('/tinystories-train-512.jsonl'))
        raw = archive.extractfile(member).read()
    assert digest(raw) == '7aa55a657de6499be64a513aa76eb21a9cd52276bf836ba725e1d7f52ea511e0'
    stories = _deduplicate_text([json.loads(line) for line in raw.decode().splitlines()])
    parts = split_records(stories, seed=42)
    denominators = split_check(parts, r['results']['runs']['tinystories']['data'])
    result['tinystories'] = {'source_rows': len(stories), 'splits': denominators,
                            'recorded_run_checks': record_check(r, parts, denominators),
                            'quoted_sample_index': 1,
                            'quoted_sample': r['results']['runs']['tinystories']['after']['validation']['samples'][1]}
    attrs = split_records(generate_records('toy-text'), seed=42)
    denominators = split_check(attrs, t['results']['data'])
    result['text_foundation'] = {'splits': denominators, 'validation_text': attrs['validation'][0]['text'],
                                 'recorded_run_checks': record_check(t, attrs, denominators),
                                 'sample': t['results']['after']['validation']['samples'][0]}
    assert attrs['validation'][0]['text'] == 'color=blue;shape=circle;side=left.'
    assert t['results']['after']['validation']['samples'][0]['generated'] == 'side=right.'
    result['same_prompt_legal_sides'] = [row['text'] for row in generate_records('toy-text') if row['text'].startswith('color=blue;shape=circle;')]
    assert len(result['same_prompt_legal_sides']) == 2
    result['primary_results'] = {}
    for name, record in [('real_text', r), ('text_foundation', t)]:
        result['primary_results'][name] = {'path': f'docs/course-experiments/results/{name}.json',
                                          'sha256': digest((ROOT / f'docs/course-experiments/results/{name}.json').read_bytes()),
                                          'revision': record['revision'], 'recorded_device': record['device'],
                                          'seed': record['seed'], 'status': record['status'], 'code_versions': {}}
        for path in ['scripts/course_experiments/common.py', 'scripts/course_experiments/text.py', 'tiny_perceptron/data.py', 'tiny_perceptron/model.py']:
            historical = subprocess.check_output(['git', 'show', record['revision'] + ':' + path], cwd=ROOT)
            assert digest(historical) == record['code_sha256'][path]
            result['primary_results'][name]['code_versions'][path] = {'recorded_sha256': digest(historical),
                'current_sha256': digest((ROOT / path).read_bytes()), 'same_bytes': historical == (ROOT / path).read_bytes()}
    result['necessary_current_reading'] = [{'source': path, 'selector': 'section:' + number,
        'sha256': digest(section(path, number).encode())} for path, number in [
            ('course/chapters/01.md', '1.8'), ('course/chapters/05.md', '5.1'),
            ('course/chapters/06.md', '6.1'), ('course/chapters/07.md', '7.12'), ('course/training.md', 'T.4')]]
    result['safe_cli_checks'] = []
    commands = [
        ['uv', 'run', '--extra', 'cpu', 'python', '-m', 'scripts.course_experiments.run', '--help'],
        ['uv', 'run', '--extra', 'cpu', 'python', '-m', 'scripts.course_experiments.run', '--list-assets', 'real_text'],
        ['uv', 'run', '--extra', 'cpu', 'python', 'scripts/fetch_training_assets.py', '--asset', 'tinystories', '--asset', 'chinese-poetry', '--output', str(CACHE / 'assets')],
    ]
    for i, command in enumerate(commands):
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        stdout = completed.stdout
        (CACHE / f'cli-{i}-stdout.txt').write_text(stdout)
        result['safe_cli_checks'].append({'command': command, 'exit_code': completed.returncode,
                                         'stdout_sha256': digest(stdout.encode()),
                                         'result': stdout.strip() if i == 1 else 'help accepted experiment/device flags' if i == 0 else 'both requested assets unpacked successfully outside Git'})
    result['finished_at'] = datetime.now(UTC).isoformat()
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'result': str(target), 'sha256': digest(target.read_bytes()), 'status': 'passed'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
