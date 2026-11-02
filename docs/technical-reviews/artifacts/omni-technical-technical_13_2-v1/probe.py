"""13.2 的獨立 CPU 核對；只執行原文與資料前處理，不建立或訓練模型。"""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.course_experiments import behavior, text
from scripts.course_experiments.common import split_records
from tiny_perceptron.alignment import sequence_log_probability
from tiny_perceptron.data import IGNORE, SPECIALS, ByteTokenizer, render_chat
from tiny_perceptron.model import masked_loss

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-technical_13_2-v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_block(code):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, '<frozen-13.2>', 'exec'), {})
    return stdout.getvalue().splitlines()


def expected_pair(prompt, answer):
    # 獨立由原文的角色／byte 規則組出下一位置目標。
    prefix = [1, 3] + [b + 8 for b in prompt.encode('utf-8')] + [2, 4]
    content = [b + 8 for b in answer.encode('utf-8')]
    return prefix + content, [IGNORE] * (len(prefix) - 1) + content + [2]


def main():
    freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
    freeze = json.loads(freeze_path.read_text())
    page = next(p for p in freeze['pages'] if p['page_id'] == '13.2')
    snapshot = ROOT / page['snapshot']
    assert sha(snapshot) == page['source_sha256']
    body = snapshot.read_text()
    block, = re.findall(r'```python\n(.*?)```', body, re.S)
    current_chapter = (ROOT / 'course/chapters/13.md').read_text()
    current = re.search(r'^## 13\.2 .*?(?=^## |\Z)', current_chapter, re.S | re.M)[0]
    assert hashlib.sha256(current.encode()).hexdigest() == page['source_sha256']
    assert not re.search(r'!\[[^]]*\]\([^)]*\.svg\)', body)
    printed = run_block(block)
    base = {}
    for side, answer in [('chosen', '2'), ('rejected', '3')]:
        x, y = render_chat([{'role': 'user', 'content': '1+1=?'}, {'role': 'assistant', 'content': answer}])
        ex, ey = expected_pair('1+1=?', answer)
        assert x.tolist() == ex and y.tolist() == ey
        assert x.dtype == y.dtype == torch.int64
        assert y[:8].tolist() == [IGNORE] * 8
        assert x[8].item() == 4 and y[8].item() == ord(answer) + 8
        assert x[-1].item() == ord(answer) + 8 and y[-1].item() == 2
        base[side] = {'input': ex, 'labels': ey, 'effective_targets': y[y != IGNORE].tolist()}
    expected_lines = [f"{s} 輸入 {base[s]['input']} 目標 {base[s]['labels']}" for s in ['chosen', 'rejected']]
    assert printed == expected_lines
    assert list(SPECIALS)[:5] == ['<pad>', '<bos>', '<eos>', '<user>', '<assistant>']
    assert len(SPECIALS) == 8
    assert [(ord(c), ByteTokenizer().encode(c)[0]) for c in '234'] == [(50, 58), (51, 59), (52, 60)]
    exercise = run_block(block.replace('"rejected": "3"', '"rejected": "4"'))
    assert exercise[0] == printed[0]
    ex, ey = expected_pair('1+1=?', '4')
    assert exercise[1] == f'rejected 輸入 {ex} 目標 {ey}'
    unequal = {}
    for answer in ['2', '33', '二']:
        x, y = render_chat([{'role': 'user', 'content': '1+1=?'}, {'role': 'assistant', 'content': answer}])
        ex, ey = expected_pair('1+1=?', answer)
        assert x.tolist() == ex and y.tolist() == ey
        unequal[answer] = {'input_length': len(x), 'effective_targets': y[y != IGNORE].tolist()}
    labels = torch.tensor([base['chosen']['labels']])
    logits = torch.zeros(1, 10, 264, requires_grad=True)
    loss = masked_loss(logits, labels)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, :8]).item() == 0
    assert torch.count_nonzero(logits.grad[0, 8:]).item() == 528
    logp = sequence_log_probability(logits.detach(), labels).item()
    assert abs(loss.item() - math.log(264)) < 1e-6
    assert abs(logp + 2 * math.log(264)) < 2e-6

    # 使用官方固定 revision 的原始 100 筆；與 repo 來源收據逐 byte 比對。
    raw_path = CACHE / 'train-first-100-verified.jsonl'
    metadata = json.loads((ROOT / 'assets/training/sources/ultrafeedback-dpo.json').read_text())
    assert sha(raw_path) == metadata['raw_training_file']['sha256']
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(raw) == 100 and len({r['prompt_id'] for r in raw}) == 100
    viewer = json.loads((CACHE / 'ultrafeedback-first100-api.json').read_text())
    assert [r['row'] for r in viewer['rows']] == raw
    assert all(not r['truncated_cells'] for r in viewer['rows'])
    assert all(r[side][0]['content'] == r['prompt'] for r in raw for side in ['chosen', 'rejected'])

    # 只抽取實裝 _natural_dpo_pilot 到 _save_splits 為止的 AST 前綴。
    # 原始 _natural_dpo_pilot 本體、訓練 API 與模型建構均未呼叫。
    fn = ast.parse(inspect.getsource(behavior._natural_dpo_pilot)).body[0]
    boundary = next(i for i, node in enumerate(fn.body) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'chosen' for t in node.targets))
    selected = fn.body[:boundary]
    assert selected[-1].lineno == 25  # 原函式第 25 行是 _save_splits
    fn.body = selected + [ast.parse('return records, parts, manifest').body[0]]
    ast.fix_missing_locations(fn)
    namespace = {
        '_asset_rows': lambda ctx, identifier, basename: raw,
        '_utf8_prefix': text._utf8_prefix,
        '_digest': text._digest,
        'split_records': split_records,
        '_save_splits': text._save_splits,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-data-preparation-prefix>', 'exec'), namespace)
    output = CACHE / 'data-preparation-replay'
    records, parts, manifest = namespace['_natural_dpo_pilot'](SimpleNamespace(seed=42, output=output))
    historical = json.loads((ROOT / 'docs/course-experiments/results/dpo.json').read_text())
    prior = historical['results']['ultrafeedback_pilot']
    assert prior['source_records'] == 100 and manifest == prior['data']
    groups = {s: {r['family'] for r in rows} for s, rows in parts.items()}
    assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])
    max_bytes = {key: max(len(r[key].encode('utf-8')) for r in records) for key in ['prompt', 'chosen', 'rejected']}
    assert all(n <= 120 for n in max_bytes.values())
    lengths = []
    cropped = {key: 0 for key in ['prompt', 'chosen', 'rejected']}
    identical_after_crop = 0
    for record, original in zip(records, raw, strict=True):
        for key in cropped:
            original_text = original[key] if key == 'prompt' else next(m['content'] for m in reversed(original[key]) if m['role'] == 'assistant')
            assert original_text.startswith(record[key])
            cropped[key] += original_text != record[key]
        identical_after_crop += record['chosen'] == record['rejected']
        for side in ['chosen', 'rejected']:
            x, y = render_chat([{'role': 'user', 'content': record['prompt']}, {'role': 'assistant', 'content': record[side]}])
            assert y[-1].item() == 2 and len(x) <= 256
            lengths.append(len(x))
    boundary_samples = []
    for value in ['a' * 119 + '中後', 'a' * 118 + '🙂後', '中文' * 30]:
        prefix = text._utf8_prefix(value, 120)
        assert value.startswith(prefix) and len(prefix.encode()) <= 120 and '\ufffd' not in prefix
        boundary_samples.append({'original_bytes': len(value.encode()), 'prefix_bytes': len(prefix.encode())})
    result = {
        'created_at': datetime.now(UTC).isoformat(),
        'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_13_2-v1/probe.py',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'torch_git': torch.version.git_version, 'device': 'cpu'},
        'source_freeze': str(freeze_path.relative_to(ROOT)), 'source_sha256': page['source_sha256'], 'figures': {},
        'exact_textbook_stdout': printed, 'base_pair': base, 'exercise_stdout': exercise,
        'byte_values': {'2': [50, 58], '3': [51, 59], '4': [52, 60]},
        'unequal_answer_lengths': unequal,
        'alignment': {'assistant_index': 8, 'eos_index': 9, 'correct_rejected_eos_input': 59, 'mismatched_chosen_eos_input': 58, 'double_shift_answer_prediction_index': 7, 'double_shift_wrong_context_token': 2},
        'mask_probe': {'uniform_logits_shape': [1, 10, 264], 'valid_targets': 2, 'masked_loss': loss.item(), 'expected_log_264': math.log(264), 'sequence_logp': logp, 'expected_sequence_logp': -2 * math.log(264), 'ignored_positions_nonzero_gradients': 0},
        'original_dataset': {'revision': metadata['revision'], 'row_indices': [0, 99], 'records': len(raw), 'families': len(groups['train'] | groups['validation'] | groups['test']), 'raw_sha256': sha(raw_path), 'raw_bytes': raw_path.stat().st_size, 'truncated_api_cells': 0},
        'reconstructed_manifest': manifest, 'matches_historical_manifest_exactly': True,
        'data_prefix_lines': 'scripts/course_experiments/behavior.py:669-692', 'split_seed': 42, 'family_overlap': 0,
        'max_utf8_bytes': max_bytes, 'cropped_fields': cropped, 'answer_inputs_checked': len(lengths), 'max_model_input_tokens': max(lengths),
        'identical_chosen_rejected_after_crop': identical_after_crop, 'utf8_boundary_samples': boundary_samples,
        'scope': '只驗證固定版本、原文程式與資料準備；沒有模型訓練、GPU、品質或一般效益實測。截短後標籤是否保留原偏好未重標，不能由本次執行證明。',
        'result': '全部 assertions 通過',
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
