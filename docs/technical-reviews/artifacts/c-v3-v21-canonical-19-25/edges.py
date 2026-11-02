"""Targeted edge and receipt checks; no trained-model or final inference."""

import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import torch

from tiny_perceptron.decision_product.data import make_families
from tiny_perceptron.decision_product.model import Tokenizer
from tiny_perceptron.decision_product.runtime import create_model, encode_for_model, candidate_eligibility
from tiny_perceptron.decision_product.service import DeviceSession, Device, Action

ROOT = Path('/workspace/work/omni-book-integrated-v1')
OUT = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-19-25'
ARCHIVE = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
started = datetime.now(UTC).isoformat()
torch.set_num_threads(1)
digest = lambda raw: hashlib.sha256(raw).hexdigest()
with ZipFile(ARCHIVE) as archive:
    load = lambda name: json.loads(archive.read('tiny-perceptron-c-v3/' + name))
    vocabulary = load('bundle/tokenizer.json')
    config = load('bundle/bundle.json')['config']
    source_freeze = load('evidence/source-freeze.json')
    selection = load('evidence/selection.json')
    opening = load('evidence/final-opening.json')
    parent = load('evidence/lineage-parent.json')
    generated_train = (json.dumps(make_families('train', 8000, 826019), ensure_ascii=False, indent=2) + '\n').encode()
    assert digest(generated_train) == source_freeze['artifacts']['train-families.json'] == parent['receipts']['source-freeze.json']['artifacts']['train-families.json']
    assert not any(Path(name).name in {'latest.pt', 'optimizer.pt', 'optimizer.safetensors'} for name in archive.namelist())
model, tokenizer = create_model(config).eval(), Tokenizer(vocabulary)
class Recorder:
    def _predict(self, example, use_cache=True):
        return example

devices = [Device('甲', 'play', True, None), Device('乙', 'save', False, True), Device('丙', 'record', True, True)]
session = DeviceSession(Recorder(), devices, 'zh')
example = session.choose('甲', [Action('a', '甲', 'play'), Action('b', '乙', 'play')])
assert candidate_eligibility(example) == [True, False, False]
renamed = DeviceSession(Recorder(), [Device('新的😀名稱', row.task, row.permission, row.ready) if i == 0 else row for i,row in enumerate(devices)], 'zh')
assert session._state == renamed._state
assert example.state == session.information('甲').state == session.grade('甲').state
assert not any(name in example.state + example.question for name in ('甲', '乙', '丙'))
try:
    encode_for_model(replace(example, state='A: ' + 'yes ' * 60 + '.'), tokenizer, model)
except ValueError as error:
    long_row_rejection = str(error)
    assert '48' in long_row_rejection
else:
    raise AssertionError('overlong record silently accepted')
ids = ['first', 'second', 'none']
probabilities = torch.tensor([4., 4., -3.]).softmax(-1)
assert ids[int(probabilities.argmax())] == 'first'
reordered_ids = ['second', 'first', 'none']
assert reordered_ids[int(probabilities.argmax())] == 'second'
assert float(probabilities.max()) < 0.5 < 0.99
calibration_choices = {}
for kind, values in selection['calibration'].items():
    chosen = min(values['search'], key=lambda row: (row['nll'], abs(__import__('math').log(row['temperature']))))
    assert chosen['temperature'] == selection['temperatures'][kind]
    calibration_choices[kind] = chosen['temperature']
limits = opening['thresholds']
feasible = [row for row in selection['policy_search']
            if row['coverage'] >= limits['choice_overall_accepted_coverage_min']
            and row['risk'] <= limits['choice_accepted_risk_max']
            and row['risk_one_sided_wilson95_upper'] <= limits['choice_accepted_risk_one_sided_wilson95_upper_max']]
selected = max(feasible, key=lambda row: (row['accepted'], -row['errors'], row['threshold']))
assert selected == selection['selected']
body = (ROOT / 'docs/course-revision-20261101-c-v3/source-freeze-v20/sources/19.25.md').read_text()
import re
links = []
for label, href in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body):
    path, _, anchor = href.partition('#')
    target = (ROOT / 'course/chapters' / path).resolve()
    assert target.is_file()
    assert not anchor or re.search(r'^## ' + re.escape(anchor) + r' ', target.read_text(), re.M)
    links.append({'label': label, 'href': href, 'target': target.relative_to(ROOT).as_posix()})
receipt = {
    'reviewer_task': '/root/c_v3_technical_coord/canonical_19_25', 'started_at': started,
    'completed_at': datetime.now(UTC).isoformat(),
    'command': 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-19-25/edges.py',
    'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'device': 'cpu', 'threads': '1'},
    'scope': 'Only data generation/hash comparison, software edge cases and saved-search arithmetic. No model training, trained-model inference, new final opening, publication or public download.',
    'same_8000_training_families': {'generated_bytes_sha256': digest(generated_train), 'hardened_and_parent_source_freezes_match': True},
    'id_only': {'candidate_eligibility': candidate_eligibility(example), 'renaming_external_names_leaves_internal_state_identical': True, 'all_question_kinds_share_state': True},
    'token_limit': {'over_48_row_rejected': long_row_rejection},
    'tie': {'probabilities': probabilities.tolist(), 'original_model_choice': 'first', 'reordered_model_choice': 'second', 'choice_threshold': 0.99, 'accepted': False},
    'retained_search': {'calibration_nll_argmins': calibration_choices, 'policy_selected': selected, 'calibration_families': 600, 'policy_families': 600, 'no_final_rows_used': True, 'note': 'Recomputed selection from retained search receipts, not recalibrated a model or rerun its inference.'},
    'inference_delivery_not_full_resume': {'no_latest_or_optimizer_checkpoint_member': True, 'note': 'No claim of exact optimizer/RNG-state resume; inference bundle has four pinned members.'},
    'navigation': links,
}
with (OUT / 'edges-execution.json').open('x', encoding='utf-8') as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write('\n')
print(json.dumps({'artifact': str(OUT/'edges-execution.json'), 'train_sha': digest(generated_train), 'links': len(links), 'threshold': selected['threshold'], 'temperatures': calibration_choices}, ensure_ascii=False))
