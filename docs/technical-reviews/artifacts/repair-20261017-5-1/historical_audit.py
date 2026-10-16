"""Audit existing text_foundation raw results and regenerate only tiny input data."""

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM


root = Path(__file__).resolve().parents[4]
path = root / 'docs/course-experiments/results/text_foundation.json'
raw = json.loads(path.read_text())
r = raw['results']
parts = split_records(generate_records('toy-text'), seed=raw['seed'])
split_audit = {}
for split, rows in parts.items():
    encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    observed_hash = hashlib.sha256(encoded).hexdigest()
    expected_hash = r['data'][split]['sha256']
    assert observed_hash == expected_hash
    examples = text_examples(rows)
    count = sum(int((y != IGNORE).sum()) for _, y in examples)
    split_audit[split] = {'records': rows, 'record_count': len(rows), 'effective_targets_once': count, 'sha256': observed_hash, 'matches_saved_raw_split_hash': True}
assert records_sha256(parts['train']) == r['training']['records_sha256']
assert len(parts['train']) == 9 and len(parts['validation']) == 1
assert not set(row['family'] for row in parts['train']) & set(row['family'] for row in parts['validation'])
validation = r['after']['validation']
assert validation['effective_tokens'] == split_audit['validation']['effective_targets_once']
assert abs(validation['nll_sum'] / validation['effective_tokens'] - validation['nll']) < 1e-14
rounding = {
    'initial_training_nll': {'raw': r['training']['initial_loss'], 'course': 5.79243},
    'final_training_nll': {'raw': r['training']['final_loss'], 'course': 0.06352},
    'final_validation_nll': {'raw': validation['nll'], 'course': 0.96282},
}
for comparison in rounding.values():
    comparison['absolute_error'] = abs(comparison['raw'] - comparison['course'])
    assert comparison['absolute_error'] <= 0.000005
cfg = ModelConfig(width=64, layers=2, max_length=128)
model = TinyLM(cfg)  # Construct only to count configuration parameters; no historical weights loaded.
assert sum(p.numel() for p in model.parameters()) == r['training']['parameters']
report = {
    'reviewer_task': '/root/repair_tech_5_1',
    'audit_environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu'},
    'scope': 'Executed audit of already-existing raw JSON, tiny deterministic split regeneration, arithmetic and fresh config parameter count. No checkpoint read, model inference, optimizer updates or repeat of historical GPU training.',
    'raw_result': {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'locators': ['device', 'seed', 'gpu', 'torch_version', 'step_scale', 'evidence_status', 'results.data', 'results.training', 'results.before.validation', 'results.after.validation']},
    'historical_environment': {k: raw[k] for k in ('revision', 'device', 'seed', 'torch_version', 'python_version', 'gpu', 'step_scale', 'evidence_status')},
    'historical_training': r['training'],
    'historical_validation_before': r['before']['validation'],
    'historical_validation_after': validation,
    'configuration_from_inspected_entrypoint': {'config': asdict(cfg), 'parameters': sum(p.numel() for p in model.parameters()), 'batch_size': 16, 'lr': 0.003, 'optimizer': 'AdamW default weight_decay=0.01', 'gradient_clipping_max_norm': 1.0, 'train_steps': 600, 'train_sampling': 'random.Random(42).choices(examples,k=16), with replacement', 'training_initial_and_final_scoring': 'common._nll over all original training examples once at each endpoint; summed cross entropy divided by non-IGNORE target count', 'source_locators': ['scripts/course_experiments/text.py:262 run_text_foundation', 'scripts/course_experiments/common.py:45 new_lm', 'scripts/course_experiments/common.py:82 text_examples', 'scripts/course_experiments/common.py:113 _nll', 'scripts/course_experiments/common.py:132 fit_lm']},
    'regenerated_input_only': split_audit,
    'training_records_hash_matches': True,
    'rounding_audit': rounding,
    'validation_arithmetic': {'sum': validation['nll_sum'], 'denominator': validation['effective_tokens'], 'quotient': validation['nll_sum'] / validation['effective_tokens']},
    'assertions': 'all passed',
}
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
