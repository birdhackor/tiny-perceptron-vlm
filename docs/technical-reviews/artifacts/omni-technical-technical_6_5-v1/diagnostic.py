"""6.5 獨立技術核對：原樣執行正文、重建分母，僅 CPU 推論，不訓練。"""

import contextlib
import datetime
import hashlib
import io
import json
import math
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
CACHE = Path('/workspace/work/reference/omni-technical-technical_6_5-v1')
ARTIFACT = Path(__file__).resolve().parent

import torch
import tokenizers
from tokenizers import Tokenizer

from scripts.course_experiments.common import split_records, text_examples
from scripts.course_experiments.text import _BPE, _deduplicate_text, _digest, _evaluate_tokenizer, _utf8_prefix
from tiny_perceptron.data import ByteTokenizer, IGNORE, SPECIALS, load_jsonl, shifted
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum, masked_loss
from tiny_perceptron.tokenization import ByteLevelBPE, load_tokenizer
from tiny_perceptron.training import load_checkpoint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_cache(path, data):
    if path.exists():
        assert path.read_bytes() == data
    else:
        path.write_bytes(data)


def run_inline(code):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, 'frozen-6.5-inline', 'exec'), {})
    return {'code_sha256': hashlib.sha256(code.encode()).hexdigest(), 'stdout': stream.getvalue()}


@torch.no_grad()
def score(model, rows, tok):
    """獨立從 log_softmax 取正確格；不依賴既有評估器的平均或 BPB。"""
    model.eval()
    total, targets, raw_bytes, losses = 0.0, 0, 0, []
    for row in rows:
        ids = [tok.bos_id] + tok.encode(row['text']) + [tok.eos_id]
        x, y = shifted(ids)
        logits = model(x[None])['logits']
        correct_log_probs = logits.double().log_softmax(-1).gather(-1, y[None, :, None])
        nll = float(-correct_log_probs.sum())
        expected_sum, expected_count = loss_sum(logits, y[None])
        assert abs(nll - float(expected_sum)) < 2e-6 * max(1.0, abs(nll))
        assert int(expected_count) == len(y) == len(tok.encode(row['text'])) + 1
        assert not (y == tok.bos_id).any()
        assert int(y[-1]) == tok.eos_id
        total += nll
        targets += len(y)
        raw_bytes += len(row['text'].encode('utf-8'))
        losses.append({'targets': len(y), 'raw_bytes': len(row['text'].encode('utf-8')), 'nll': nll,
                       'fp64_minus_fp32_loss_sum': nll - float(expected_sum)})
    return {'records': len(rows), 'effective_tokens': targets, 'raw_utf8_bytes': raw_bytes,
            'nll_sum': total, 'mean_token_nll': total / targets,
            'bpb_including_eos_boundary_targets': total / (raw_bytes * math.log(2))}, losses


started = datetime.datetime.now(datetime.UTC).isoformat()
torch.set_num_threads(1)
body_path = ROOT / 'docs/course-review-20261019/freeze-05/sources/6.5.md'
body = body_path.read_text()
current = re.search(r'^## 6\.5 .*?(?=^## |\Z)', (ROOT / 'course/chapters/06.md').read_text(), re.M | re.S).group()
assert current == body
assert sha(body_path) == '9bff6436d575412635d1ea4bba877f131a08131222974e9023a513cfbe9a90d5'
blocks = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(blocks) == 2
inline = [run_inline(code) for code in blocks]
inline += [run_inline(blocks[0].replace('[("示例A", 5.0), ("示例B", 4.0)]', '[("示例A", 10.0), ("示例B", 8.0)]'))]
inline += [run_inline(blocks[0].replace('raw = "貓🙂"', 'raw = "貓🙂貓🙂"'))]

record_path = ROOT / 'docs/course-experiments/results/tokenizer.json'
record = json.loads(record_path.read_text())
raw_paths = [ROOT / 'data/training/text-initial/tinystories-train-512.jsonl',
             ROOT / 'data/training/text-initial/chinese-classical-train-365.jsonl']
raw = load_jsonl(raw_paths[0])[:96] + load_jsonl(raw_paths[1])[:96]
complete = split_records(_deduplicate_text(raw), seed=42)
parts = {split: [{**row, 'text': _utf8_prefix(row['text'], 256),
                  'complete_text_sha256': hashlib.sha256(row['text'].encode()).hexdigest()} for row in rows]
         for split, rows in complete.items()}
splits = {}
for split, rows in parts.items():
    encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    assert digest == record['results']['data'][split]['sha256']
    save_cache(CACHE / f'reconstructed-{split}.jsonl', encoded)
    splits[split] = {'records': len(rows), 'sha256': digest,
                     'raw_utf8_bytes': sum(len(row['text'].encode()) for row in rows)}

tokenizer_file = CACHE / 'tokenizer-bpe512.json'
assert sha(tokenizer_file) == 'ec08c61eff41c94035b6d234fb07eb63f21503b1d471ec4fbf5033df627180e8'
tok_map = {'byte256': ByteTokenizer(), 'bpe512': ByteLevelBPE(tokenizer_file)}
training_bpe = _BPE(Tokenizer.from_file(str(tokenizer_file)))
assert all(training_bpe.encode(row['text']) == tok_map['bpe512'].encode(row['text'])
           for rows in parts.values() for row in rows)
sampler = random.Random(42)
exposure = [sampler.choices(range(len(parts['train'])), k=8) for _ in range(400)]
exposure_bytes = sum(len(parts['train'][index]['text'].encode()) for batch in exposure for index in batch)
assert exposure_bytes == 664759
assert _digest(exposure) == record['results']['raw_document_schedule_sha256']

runs, raw_checks = {}, {}
for name, tok in tok_map.items():
    cfg = ModelConfig(vocab_size=tok.vocab_size, width=32, layers=1, max_length=384)
    torch.manual_seed(42)
    before = TinyLM(cfg)
    checkpoint_file = CACHE / (name + '.pt')
    after, payload = load_checkpoint(checkpoint_file, 'cpu')
    assert payload['config'] == vars(cfg)
    assert sum(p.numel() for p in after.parameters()) == record['results']['runs'][name]['parameters']
    assert len(text_examples(parts['train'], 'text', 384, tok)) == len(parts['train'])
    assert max(len(tok.encode(row['text'])) + 2 for rows in parts.values() for row in rows) <= 258
    phases, raw_phases = {}, {}
    for phase, model in [('before', before), ('after', after)]:
        phases[phase], raw_phases[phase] = {}, {}
        for split in ['validation', 'test']:
            measured, row_losses = score(model, parts[split], tok)
            expected = record['results']['runs'][name][phase][split]
            differences = {}
            for key in ['records', 'effective_tokens', 'raw_utf8_bytes']:
                assert measured[key] == expected[key]
            for key in ['mean_token_nll', 'bpb_including_eos_boundary_targets']:
                differences[key] = measured[key] - expected[key]
                assert abs(differences[key]) < 2e-6
            phases[phase][split] = {**measured, 'difference_from_record': differences}
            raw_phases[phase][split] = row_losses
    # 真跑教材既有評估函式，確認它的分母及 32-token 生成；不更新參數。
    helper = _evaluate_tokenizer(after, parts['test'], tok)
    historical_samples = record['results']['runs'][name]['after']['test']['samples']
    same_samples = [a['generated'] == b['generated'] for a, b in zip(helper['samples'], historical_samples, strict=True)]
    assert all(same_samples)
    phases['helper_after_test'] = {key: value for key, value in helper.items() if key != 'samples'}
    phases['helper_after_test']['all_20_generated_strings_match_record'] = all(same_samples)
    phases['checkpoint'] = {'path': str(checkpoint_file), 'sha256': sha(checkpoint_file),
                            'parameters': sum(p.numel() for p in after.parameters()), 'config': payload['config'],
                            'tokenizer_binding': payload['tokenizer']}
    phases['maximum_input_length'] = max(len(tok.encode(row['text'])) + 1 for rows in parts.values() for row in rows)
    runs[name], raw_checks[name] = phases, raw_phases

# 5.2 的有效目標平均及 padding 梯度，與本節提到的 masked_loss 回傳值相同。
logits = torch.zeros(2, 4, 5, requires_grad=True)
labels = torch.tensor([[1, 2, 3, 4], [1, IGNORE, IGNORE, IGNORE]])
total, count = loss_sum(logits, labels)
mean = masked_loss(logits, labels)
mean.backward()
received = logits.grad.abs().sum(-1) > 0
assert torch.equal(received, labels != IGNORE)
padding = {'count': int(count), 'sum': float(total.detach()), 'mean': float(mean.detach()),
           'gradient_mask': received.tolist(), 'labels_dtype': str(labels.dtype), 'logits_dtype': str(logits.dtype)}

long_window = {'windows': [['A','B','C'], ['B','C','D']], 'scored_targets': [['B','C'], ['D']],
               'multiplicity': {'B': 1, 'C': 1, 'D': 1},
               'valid_nll': sum(-math.log(p) for p in [.5, .25, .125]),
               'duplicated_C_extra_nll': -math.log(.25)}
lengths = []
for split in ['validation', 'test']:
    a = record['results']['runs']['byte256']['after'][split]['samples']
    b = record['results']['runs']['bpe512']['after'][split]['samples']
    for x, y in zip(a, b, strict=True):
        lengths.append({'split': split, 'row': x['row'], 'byte_codepoints': len(x['generated']),
                        'bpe_codepoints': len(y['generated']), 'byte_decoded_utf8_bytes': len(x['generated'].encode()),
                        'bpe_decoded_utf8_bytes': len(y['generated'].encode())})
generation = {'pairs': len(lengths),
              'bpe_longer_in_codepoints': sum(row['bpe_codepoints'] > row['byte_codepoints'] for row in lengths),
              'bpe_longer_in_decoded_utf8_bytes': sum(row['bpe_decoded_utf8_bytes'] > row['byte_decoded_utf8_bytes'] for row in lengths),
              'mean_byte_codepoints': sum(row['byte_codepoints'] for row in lengths) / len(lengths),
              'mean_bpe_codepoints': sum(row['bpe_codepoints'] for row in lengths) / len(lengths),
              'scope': 'Historical 39 after-validation/test generations; CPU replay matched all 40 test strings. Decoded UTF-8 length includes replacement characters; no answer-quality inference.'}
raw_checks['generation_lengths'] = lengths
raw_file = CACHE / 'diagnostic-row-checks-final.json'
save_cache(raw_file, (json.dumps(raw_checks, ensure_ascii=False, indent=2) + '\n').encode())

same_id = 105
pairing = {'same_integer_id': same_id, 'byte_meaning': bytes([same_id - 8]).decode('ascii'),
           'bpe_vocabulary_piece': tok_map['bpe512'].tokenizer.id_to_token(same_id),
           'specials': list(SPECIALS), 'byte_example_ids': tok_map['byte256'].encode('貓🙂'),
           'bpe_example_ids': tok_map['bpe512'].encode('貓🙂')}
try:
    load_tokenizer(None, 512)
except ValueError as error:
    pairing['missing_bpe_rejected'] = str(error)
else:
    raise AssertionError('missing BPE should be rejected')

help_run = subprocess.run([sys.executable, '-m', 'scripts.course_experiments.run', '--help'], cwd=ROOT,
                          capture_output=True, text=True, check=True)
assert '--experiment ID' in help_run.stdout
recorded_code = {path: {'current_sha256': sha(ROOT / path), 'historical_sha256': record['code_sha256'][path]}
                 for path in ['tiny_perceptron/model.py','tiny_perceptron/data.py','tiny_perceptron/attention.py',
                              'scripts/course_experiments/text.py','scripts/course_experiments/common.py',
                              'tiny_perceptron/tokenization.py','scripts/course_experiments/run.py']}
for item in recorded_code.values():
    item['same'] = item['current_sha256'] == item['historical_sha256']
assert all(value['same'] for path, value in recorded_code.items() if path != 'scripts/course_experiments/run.py')
result = {'started_at': started, 'completed_at': datetime.datetime.now(datetime.UTC).isoformat(),
          'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_6_5-v1/diagnostic.py',
          'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                          'torch_git_version': torch.version.git_version, 'tokenizers': tokenizers.__version__,
                          'device': 'cpu', 'torch_threads': str(torch.get_num_threads())},
          'result': 'All assertions passed; no training, backward only on a 2x4x5 synthetic logits tensor.',
          'source_sha256': sha(body_path), 'record_sha256': sha(record_path), 'inline_runs': inline,
          'arithmetic': {'opening_totals': [10 * .6, 5 * 1.0], 'conditional_product': .5 * .25,
                         'conditional_nll': -math.log(.5) - math.log(.25), 'joint_nll': -math.log(.125),
                         'utf8_bytes': {'貓': len('貓'.encode()), '🙂': len('🙂'.encode()), '貓🙂': len('貓🙂'.encode())},
                         'rounded_five_decimal_bpb': 2.77947 * 4213 / (4193 * math.log(2))},
          'splits': splits, 'input_asset_sha256': {p.name: sha(p) for p in raw_paths},
          'exposure': {'steps': 400, 'documents_per_step': 8, 'raw_utf8_bytes': exposure_bytes,
                       'schedule_sha256': _digest(exposure)}, 'runs': runs, 'padding': padding,
          'long_window': long_window, 'generation_length': generation, 'pairing': pairing,
          'cli_help': {'exit_code': help_run.returncode, 'experiment_option_present': True},
          'implementation_same_as_recorded_experiment': recorded_code,
          'runner_drift_scope': 'run.py differs from the historical report; this diagnostic only ran its --help. All scoring, model, data and tokenizer implementations listed above are byte-identical to the recorded experiment.',
          'raw_row_evidence': {'path': str(raw_file), 'sha256': sha(raw_file)}}
(ARTIFACT / 'diagnostic-output.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'result': result['result'], 'after_test': {name: value['after']['test'] for name, value in runs.items()},
                  'generation_length': generation}, ensure_ascii=False, indent=2))
