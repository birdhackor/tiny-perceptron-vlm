"""6.3 獨立技術核對：只執行正文與既有 tokenizer 資料準備，沒有 LM 訓練。"""

import contextlib
import datetime
import hashlib
import io
import json
import platform
import re
import sys
import tarfile
from pathlib import Path

import tokenizers
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import _BPE, _deduplicate_text, _utf8_prefix
from tiny_perceptron.data import SPECIALS, ByteTokenizer


CACHE = Path('/workspace/work/reference/omni-technical-technical_6_3-v1')
ARTIFACT = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def execute_block(code):
    output = io.StringIO()
    scope = {}
    exception = None
    with contextlib.redirect_stdout(output):
        try:
            exec(compile(code, 'frozen-6.3-inline.py', 'exec'), scope)
        except AssertionError:
            exception = 'AssertionError'
    return scope, output.getvalue(), exception


def encode_record(tok, text):
    encoded = tok.encode(text)
    return {
        'text': text,
        'utf8_bytes': len(text.encode('utf-8')),
        'ids': encoded.ids,
        'tokens': encoded.tokens,
        'decoded': tok.decode(encoded.ids),
        'same': tok.decode(encoded.ids) == text,
    }


def read_asset(archive, basename):
    with tarfile.open(ROOT / archive) as handle:
        members = [m for m in handle.getmembers() if m.name.endswith('/' + basename) or m.name == basename]
        assert len(members) == 1
        content = handle.extractfile(members[0]).read()
    return [json.loads(line) for line in content.decode().splitlines()], {
        'archive': archive,
        'archive_sha256': digest((ROOT / archive).read_bytes()),
        'member': members[0].name,
        'member_sha256': digest(content),
    }


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json').read_bytes())
    current = (ROOT / 'course/chapters/06.md').read_text()
    source = {}
    for identifier in ['6.1', '6.2', '6.3', '6.6']:
        page = next(p for p in freeze['pages'] if p['page_id'] == identifier)
        raw = (ROOT / page['snapshot']).read_bytes()
        now = re.search(r'^## ' + re.escape(identifier) + r' .*?(?=^## |\Z)', current, re.M | re.S).group()
        assert digest(raw) == page['source_sha256'] == digest(now.encode())
        source[identifier] = {'snapshot': page['snapshot'], 'sha256': digest(raw), 'current_same': True}
    raw = (ROOT / source['6.3']['snapshot']).read_text()
    code = re.search(r'```python\n(.*?)```', raw, re.S).group(1)
    scope, stdout, exception = execute_block(code)
    assert exception is None
    tok = scope['tok']
    definition = json.loads(tok.to_str())
    records = [encode_record(tok, text) for text in ['貓看狗。', '小鳥🙂new', '未見字🦊', ' 小鳥🙂\nnew ', 'Ａa\t\n']]
    assert tok.get_vocab_size() == 277
    assert len(definition['model']['merges']) == 21
    assert definition['normalizer'] is None
    assert all(r['same'] for r in records)
    assert [(r['utf8_bytes'], len(r['ids'])) for r in records[:3]] == [(12, 2), (13, 13), (13, 13)]
    assert [tok.decoder.decode([t]) for t in records[0]['tokens']] == ['貓看狗', '。']
    training_after_merges = [encode_record(tok, t) for t in ['貓看狗。', 'a cat sees a dog.']]
    for row in training_after_merges:
        pieces = tok.pre_tokenizer.pre_tokenize_str(row['text'])
        assert len(row['ids']) == len(pieces)
    exercise_scope, exercise_stdout, exercise_exception = execute_block(code.replace('"小鳥🙂new"', '"未見字🦊"'))
    assert exercise_exception is None
    prefix_scope, prefix_stdout, prefix_exception = execute_block(code.replace('add_prefix_space=False', 'add_prefix_space=True'))
    assert prefix_exception == 'AssertionError'
    prefix_tok = prefix_scope['tok']
    prefix_records = [encode_record(prefix_tok, t) for t in ['貓看狗。', '小鳥🙂new']]
    assert all(r['decoded'] == ' ' + r['text'] for r in prefix_records)
    pretokenized = {t: tok.pre_tokenizer.pre_tokenize_str(t) for t in ['a cat', 'a  cat', 'a\ncat', '貓看狗。']}
    assert pretokenized['a cat'] == [('a', (0, 1)), ('Ġcat', (1, 5))]
    alternate = Tokenizer(models.BPE())
    alternate.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    alternate.decoder = decoders.ByteLevel()
    alternate.train_from_iterator(['狗看貓。', 'a dog sees a cat.'], trainers.BpeTrainer(vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    differences = [{'id': i, 'first': tok.id_to_token(i), 'alternate': alternate.id_to_token(i)} for i in range(min(tok.get_vocab_size(), alternate.get_vocab_size())) if tok.id_to_token(i) != alternate.id_to_token(i)]
    assert alternate.get_vocab_size() == tok.get_vocab_size() and differences
    byte_tok = ByteTokenizer()
    byte_roundtrip = [{'text': t, 'same': byte_tok.decode(byte_tok.encode(t)) == t, 'count': len(byte_tok.encode(t))} for t in ['小鳥🙂new', '未見字🦊', ' 小鳥🙂\nnew ', '<assistant>']]
    assert all(r['same'] for r in byte_roundtrip)

    historical = json.loads((ROOT / 'docs/course-experiments/results/tokenizer.json').read_bytes())
    stories, story_asset = read_asset('assets/training/tinystories-v1.tar.gz', 'tinystories-train-512.jsonl')
    poems, poem_asset = read_asset('assets/training/chinese-poetry-v1.tar.gz', 'chinese-classical-train-365.jsonl')
    raw_rows = stories[:96] + poems[:96]
    unique_rows = _deduplicate_text(raw_rows)
    complete = split_records(unique_rows, seed=historical['seed'])
    parts = {name: [{**row, 'text': _utf8_prefix(row['text'], 256), 'complete_text_sha256': digest(row['text'].encode())} for row in rows] for name, rows in complete.items()}
    data_receipts = {}
    for name, rows in parts.items():
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
        target = CACHE / f'reconstructed-{name}.jsonl'
        target.write_bytes(content)
        expected = historical['results']['data'][name]
        assert len(rows) == expected['records']
        assert digest(content) == expected['sha256']
        assert max(len(row['text'].encode()) for row in rows) <= 256
        data_receipts[name] = {'records': len(rows), 'sha256': digest(content), 'historical_digest_matches': True, 'max_utf8_bytes': max(len(row['text'].encode()) for row in rows)}
    families = {name: {row['family'] for row in rows} for name, rows in parts.items()}
    assert all(not families[a].intersection(families[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])

    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator([row['text'] for row in parts['train']], trainers.BpeTrainer(vocab_size=512, special_tokens=list(SPECIALS), initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    rebuilt = CACHE / 'reconstructed-tokenizer-bpe512.json'
    bpe.save(str(rebuilt))
    rebuilt_definition = json.loads(bpe.to_str())
    wrapped = _BPE(bpe)
    assert wrapped.vocab_size == 512
    assert [bpe.token_to_id(t) for t in SPECIALS] == list(range(8))
    bpe_roundtrip = []
    for expected in historical['results']['roundtrip']:
        text = expected['text']
        ids = wrapped.encode(text)
        restored = wrapped.decode(ids)
        observed = {'text': text, 'ids': ids, 'restored': restored, 'same': restored == text, 'contains_control_id': any(i in wrapped.special_ids for i in ids)}
        observed['same_as_recorded_ids'] = ids == expected['ids']
        assert observed['same'] and not observed['contains_control_id']
        bpe_roundtrip.append(observed)
    expected_tokenizer = next(a for a in historical['artifacts'] if a['path'] == 'tokenizer-bpe512.json')
    validated_lengths = [len(wrapped.encode(row['text'])) for row in parts['validation']]
    results = {
        'created_at': datetime.datetime.now(datetime.UTC).isoformat(),
        'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_6_3-v1/verify_cpu.py',
        'environment': {'python': sys.version, 'torch': torch.__version__, 'tokenizers': tokenizers.__version__, 'device': 'cpu', 'platform': platform.platform()},
        'source_slices': source,
        'inline': {'code_sha256': digest(code.encode()), 'stdout': stdout, 'vocab_size': tok.get_vocab_size(), 'merge_count': len(definition['model']['merges']), 'normalizer': definition['normalizer'], 'alphabet_size': len(pre_tokenizers.ByteLevel.alphabet()), 'records': records, 'training_after_merges': training_after_merges, 'pretokenized': pretokenized},
        'exercise': {'stdout': exercise_stdout, 'exception': exercise_exception},
        'prefix_true': {'stdout_until_failure': prefix_stdout, 'exception': prefix_exception, 'records': prefix_records},
        'same_vocab_different_id_mapping': {'alternate_corpus': ['狗看貓。', 'a dog sees a cat.'], 'vocab_size': alternate.get_vocab_size(), 'differences': differences},
        'byte_roundtrip': byte_roundtrip,
        'historical': {'seed': historical['seed'], 'original_environment': {k: historical[k] for k in ['revision', 'device', 'seed', 'torch_version', 'python_version', 'gpu', 'step_scale', 'status', 'evidence_status']}, 'assets': [story_asset, poem_asset], 'raw_documents': len(raw_rows), 'unique_documents': len(unique_rows), 'parts': data_receipts, 'families_disjoint': True, 'train_only_documents_for_bpe': len(parts['train']), 'vocab_size': wrapped.vocab_size, 'special_ids': [bpe.token_to_id(t) for t in SPECIALS], 'merge_count': len(rebuilt_definition['model']['merges']), 'rebuilt_tokenizer': {'path': str(rebuilt), 'sha256': digest(rebuilt.read_bytes()), 'historical_sha256': expected_tokenizer['sha256'], 'exact_historical_digest_matches': digest(rebuilt.read_bytes()) == expected_tokenizer['sha256']}, 'roundtrip': bpe_roundtrip, 'validation_token_lengths_equal_recorded': validated_lengths == historical['results']['runs']['bpe512']['validation_text_token_lengths'], 'limits': '未執行語言模型 forward、訓練或品質評估；只重建原實驗資料切分與 153 篇短片段的 tokenizer。'},
        'status': 'all asserted checks passed',
    }
    output_path = ARTIFACT / 'cpu-result-v1.json'
    with output_path.open('x', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(stdout, end='')
    print('exercise 未見字🦊: 13 bytes / 13 tokens; roundtrip true')
    print('add_prefix_space=True:', prefix_exception, '; both tested decodes gain one leading space')
    print('split records:', {name: row['records'] for name, row in data_receipts.items()})
    print('all split hashes match historical:', all(row['historical_digest_matches'] for row in data_receipts.values()))
    print('rebuilt BPE:', json.dumps(results['historical']['rebuilt_tokenizer'], ensure_ascii=False))
    print('roundtrip recorded IDs equal:', [row['same_as_recorded_ids'] for row in bpe_roundtrip])
    print('validation token lengths equal recorded:', results['historical']['validation_token_lengths_equal_recorded'])
    print('all asserted checks passed; no language-model training')


if __name__ == '__main__':
    main()
