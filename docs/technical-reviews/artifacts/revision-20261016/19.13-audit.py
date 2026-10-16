"""19.13 independent audit: inspect historical evidence/API, no model training."""
import dataclasses
import hashlib
import inspect
import json
import platform
from pathlib import Path

import torch
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig, SelftrainedLanguageModel
from scripts.reading_time import lesson_slices

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/19.13-audit.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary_path = ROOT / 'docs/selftrained/results/v2-final-public-results.json'
summary = json.loads(summary_path.read_text())
body = lesson_slices((ROOT / 'course/chapters/19.md').read_text())['19.13']
freeze = json.loads((ROOT / 'docs/course-revision-20261016/freeze-08.json').read_text())
entry = next(p for p in freeze['pages'] if p['page_id'] == '19.13')
body_sha = hashlib.sha256(body.encode()).hexdigest()
assert body_sha == entry['source_sha256']
assert body == (ROOT / entry['snapshot']).read_text()

history = {}
for architecture in ('moe', 'dense'):
    info = summary['architectures'][architecture]
    raw_path = ROOT / f'docs/selftrained/results/public-raw/{architecture}/test/metrics.json'
    raw = json.loads(raw_path.read_text())
    assert digest(raw_path) == info['final_test']['metrics_sha256']
    assert raw['count'] == info['final_test']['count'] == 3734
    assert raw['safe_weights_sha256'] == info['safe_weights']['sha256']
    assert raw['evaluation_complete'] and raw['teacher_forcing_used_for_generation'] is False
    task_scores = {}
    for task in ('text', 'vision_clothing', 'vision_relation', 'ocr', 'voice_qa', 'tool_call'):
        task_scores[task] = raw['per_task_final_reply'][task]['semantic']
        assert task_scores[task] == info['final_test']['per_task_final_reply'][task]['semantic']
    tool = task_scores['tool_call']
    assert tool['numerator'] / tool['denominator'] < raw['thresholds']['tool_roundtrip']
    assert info['capability_acceptance_passed'] is False
    history[architecture] = {
        'raw_path': raw_path.relative_to(ROOT).as_posix(), 'raw_sha256': digest(raw_path),
        'raw_version': raw['version'], 'test_run': info['final_test']['run_id'],
        'test_revision': info['final_test']['revision'], 'test_count': raw['count'],
        'evaluation_complete': raw['evaluation_complete'],
        'teacher_forcing_used_for_generation': raw['teacher_forcing_used_for_generation'],
        'selected_source': info['selected_source'], 'training_revision': info['training_revision'],
        'safe_weights_sha256': info['safe_weights']['sha256'],
        'task_semantic_scores': task_scores,
        'tool_roundtrip_threshold': raw['thresholds']['tool_roundtrip'],
        'capability_acceptance_passed': info['capability_acceptance_passed'],
    }

# The event is transparent ground truth, not an inference from trained B.
state = 2
trajectory = [state]
for assigned in (5, 1):
    state = assigned
    trajectory.append(state)
assert trajectory == [2, 5, 1] and state == 1

signatures = {
    'LimitedAssistant.generate': str(inspect.signature(LimitedAssistant.generate)),
    'SelftrainedLanguageModel.forward': str(inspect.signature(SelftrainedLanguageModel.forward)),
}
fields = [f.name for f in dataclasses.fields(SelftrainedConfig)]
for function in (LimitedAssistant.generate, SelftrainedLanguageModel.forward):
    assert not {'effort', 'rounds', 'iterations'}.intersection(inspect.signature(function).parameters)
assert not {'effort', 'rounds', 'iterations'}.intersection(fields)
forward = inspect.getsource(SelftrainedLanguageModel.forward)
assert 'for i, block in enumerate(self.blocks):' in forward

result = {
    'scope': '19.13 only. Historical metrics inspection and function introspection; no model training or regeneration of historical evaluation.',
    'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/19.13-audit.py',
    'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu (introspection and JSON audit only)'},
    'source_sha256': body_sha, 'freeze_snapshot_exact_match': True,
    'history_summary': {'path': summary_path.relative_to(ROOT).as_posix(), 'sha256': digest(summary_path), 'hf_revision': summary['hf_revision'], 'data_scope': summary['data_scope']},
    'historical_models': history,
    'event_derivation': {'semantics': 'set(v) overwrites state with v; x0=2; x1=5; x2=1', 'trajectory': trajectory, 'final_state': state, 'trained_model_inference': False},
    'existing_A_api': {'signatures': signatures, 'config_fields': fields, 'forward_fixed_block_loop_seen': True, 'native_effort_parameter': False},
    'checks_passed': True,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': OUT.relative_to(ROOT).as_posix(), 'checks_passed': True, 'source_sha256': body_sha, 'event_trajectory': trajectory, 'historical_tool_correct': {a: x['task_semantic_scores']['tool_call']['numerator'] for a,x in history.items()}, 'native_effort_parameter': False, 'environment': result['environment']}, ensure_ascii=False, indent=2))
