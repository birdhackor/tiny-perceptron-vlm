"""Original owner's two-issue source21 callback; no package/model import or Git."""

import ast
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = ROOT / 'docs/course-revision-20261101-c-v3'
PACKET = BASE / 'original-owner-callbacks-v21/omni_p5_technical_19_23.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value, pointer):
    if not pointer:
        return value
    for part in pointer.lstrip('/').split('/'):
        key = part.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


packet = json.loads(PACKET.read_text())
assert packet['intended_original_owner'] == '/root/omni_p5_technical_19_23'
assert [row['index'] for row in packet['rows']] == [429, 430, 431, 432, 478, 479]
assert {row['issue_id'] for row in packet['rows']} == {'T19.23-01', 'T19.23-02'}
freeze = ROOT / packet['current_source_freeze']['path']
assert sha(freeze) == packet['current_source_freeze']['sha256'] == 'd9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea'
stage_path = BASE / 'technical-stage-open-v21-v1.json'
stage = json.loads(stage_path.read_text())
assert stage['status'] == 'OPEN' and stage['role'] == 'technical'
assert stage['source_freeze'] == packet['current_source_freeze']
frozen_page = next(row for row in json.loads(freeze.read_text())['pages'] if row['page_id'] == '19.23')
snapshot = ROOT / frozen_page['snapshot']
assert sha(snapshot) == frozen_page['source_sha256'] == '8cd1612e8c64a6833a01cad829294da07955800139f602c08d2bb9f3ffe864a1'
assert frozen_page['figures_sha256'] == {}
current_text = snapshot.read_text()
current_blocks = re.findall(r'```python\n(.*?)\n```', current_text, re.S)
assert len(current_blocks) == 3
origin_ref = next(row['report'] for row in packet['rows'] if row['index'] == 431)
origin_path = ROOT / origin_ref['path']
assert sha(origin_path) == origin_ref['sha256']
origin = json.loads(origin_path.read_text())
assert origin['created_at'] == '2026-10-30T09:10:30.562819+00:00'
prior_refs = {}
occurrences = []
for row in packet['rows']:
    assert row['original_issue_owner'] == origin['reviewer_task']
    assert row['affected_pages'] == ['19.23'] and row['affected_files'] == []
    p = ROOT / row['report']['path']
    assert sha(p) == row['report']['sha256']
    report = json.loads(p.read_text())
    issue = resolve(report, row['json_pointer'])
    assert issue == row['original_issue']
    assert issue['necessity'] == ('necessary' if row['issue_id'] == 'T19.23-01' else 'optional')
    for ref in row['original_verification']:
        q = ROOT / ref['report']['path']
        assert sha(q) == ref['report']['sha256']
        fixed_report = json.loads(q.read_text())
        assert fixed_report['reviewer_task'] == origin['reviewer_task']
        assert fixed_report['issues'][0 if row['issue_id'] == 'T19.23-01' else 1]['status'] == 'resolved'
        prior_refs[str(q.relative_to(ROOT))] = {'path': str(q.relative_to(ROOT)), 'sha256': sha(q),
                                                'created_at': fixed_report['created_at']}
    occurrences.append({'index': row['index'], 'issue_id': row['issue_id'],
                        'occurrence_report': row['report'], 'json_pointer': row['json_pointer'],
                        'occurrence_report_created_at': report['created_at'],
                        'occurrence_report_verdict': report['verdict'],
                        'status_as_originally_recorded_at_occurrence': issue['status'],
                        'original_necessity': issue['necessity'],
                        'original_finding_created_at': origin['created_at']})

prior_fix_ref = packet['rows'][0]['original_verification'][0]['report']
prior_fix = json.loads((ROOT / prior_fix_ref['path']).read_text())
prior_snapshot_path = ROOT / next(row['path'] for row in prior_fix['sources'] if row['id'] == 's-lesson')
prior_snapshot_source = next(row for row in prior_fix['sources'] if row['id'] == 's-lesson')
assert sha(prior_snapshot_path) == prior_snapshot_source['sha256']
prior_blocks = re.findall(r'```python\n(.*?)\n```', prior_snapshot_path.read_text(), re.S)
assert prior_blocks == current_blocks
selector = next(node for node in ast.parse(current_blocks[0]).body if isinstance(node, ast.FunctionDef) and node.name == 'selected')
scope = {}
exec(compile(ast.Module(body=[selector], type_ignores=[]), 'current-19.23-selected-only', 'exec'), scope)
cases = []
for ids, scores, expected in [
    (['a', 'b'], [0.0, 1e-8], 'b'), (['b', 'a'], [1e-8, 0.0], 'b'),
    (['a', 'b'], [0.0, 0.0], 'a'), (['b', 'a'], [0.0, 0.0], 'a'),
    (['card', 'address'], [0.0, 0.0], 'address'), (['address', 'card'], [0.0, 0.0], 'address'),
]:
    actual = scope['selected']({'ids': ids, 'scores': scores, 'p': [0.5, 0.5]})
    assert actual == expected
    cases.append({'ids': ids, 'raw_scores': scores, 'supplied_equal_p': [0.5, 0.5],
                  'expected': expected, 'actual': actual})
assert '"scores": torch.tensor([0.0, 0.0])' in current_blocks[0]
precondition = '直接呼叫這裡的`ChoiceModel.predict`時，呼叫端須先確保`temperature`為正有限值；下例沿用預設1.0。這個method只檢查`temperature <= 0`，會漏過NaN與正無窮；正式服務的`CJudge`與校準入口另用`math.isfinite`拒絕非有限值，本輪固定T也通過該檢查。'
assert precondition in current_text
code_receipts = []
for identifier in ['s-choice', 's-inference', 's-reliability']:
    source = next(row for row in prior_fix['sources'] if row['id'] == identifier)
    path = ROOT / source['path']
    assert sha(path) == source['sha256']
    code_receipts.append({'path': source['path'], 'sha256': source['sha256'],
                          'byte_identical_to_prior_owner_verified_code': True})
prior_artifact = next(row for row in prior_fix['artifacts'] if row['id'] == 'a-closure-current')
assert sha(ROOT / prior_artifact['path']) == prior_artifact['sha256']
prior_execution = json.loads((ROOT / prior_artifact['path']).read_text())
assert len(prior_execution['formal_service_and_calibration_invalid_temperature_cases']) == 8
assert all(row['rejected'] for row in prior_execution['formal_service_and_calibration_invalid_temperature_cases'])
lines = current_text.splitlines()
selected_start = next(i for i, line in enumerate(lines) if line == 'def selected(result):')
precondition_line = next(i for i, line in enumerate(lines) if line == precondition)
predict_start = next(i for i, line in enumerate(lines) if line.strip().startswith('def predict(self, question, temperature='))
chapter = (ROOT / 'course/chapters/19.md').read_text()
start = re.search(r'^## 19\.23 ', chapter, re.M).start()
following = re.search(r'^## ', chapter[start + 1:], re.M)
body = chapter[start:start + 1 + following.start()] if following else chapter[start:]
assert body == current_text
proof = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'reviewer_task': origin['reviewer_task'],
    'mode': 'bounded same-original-owner two-issue still-fixed check; six occurrences retained',
    'dispatch': {'path': str(PACKET.relative_to(ROOT)), 'sha256': sha(PACKET)},
    'technical_stage_open': {'path': str(stage_path.relative_to(ROOT)), 'sha256': sha(stage_path),
                              'started_at': stage['started_at']},
    'source_freeze': packet['current_source_freeze'],
    'page': {'page_id': '19.23', 'source_sha256': frozen_page['source_sha256'],
             'figures_sha256': {}, 'snapshot': frozen_page['snapshot']},
    'original_finding_report': {**origin_ref, 'created_at': origin['created_at'], 'verdict': origin['verdict']},
    'prior_fixed_reports': list(prior_refs.values()),
    'all_six_original_occurrence_records_exactly_match_packets': True,
    'occurrences': occurrences,
    'current_all_three_code_blocks_exactly_equal_prior_owner_source12': True,
    'current_selector_source': {'start_line': selected_start + 1, 'text': '\n'.join(lines[selected_start:selected_start + 5])},
    'selector_only_cpu_cases': cases,
    'equal_softmax_rounding_reused_not_recomputed': {'artifact': {'path': prior_artifact['path'], 'sha256': prior_artifact['sha256']},
                                                  'scope': 'The prior float32 [0,1e-8] -> [.5,.5] rounding was actually executed there; current selector-only cases use supplied equal p and raw score lists.'},
    'finite_temperature_precondition': {'line': precondition_line + 1, 'text': precondition},
    'standalone_predict_current_excerpt': {'start_line': predict_start + 1, 'text': '\n'.join(lines[predict_start:predict_start + 15])},
    'unchanged_service_code_receipts': code_receipts,
    'prior_eight_finite_guard_cases_reused': {'artifact': {'path': prior_artifact['path'], 'sha256': prior_artifact['sha256']},
                                             'created_at': prior_execution['created_at'], 'case_count': 8,
                                             'basis': 'Exact unchanged choice.py/inference.py/reliability.py bytes plus currently reread guard snippets and exact current precondition.'},
    'course_section_equals_source21_snapshot_bytes': True,
    'no_models_loaded_no_tensor_forward_no_training_no_calibration_no_final_no_network_no_Git': True,
    'environment': {'python': sys.version.split()[0], 'platform': platform.platform(),
                    'execution': 'CPU, stdlib AST/hash/JSON plus extracted selected function only; torch not imported'},
}
path = OUT / 'bounded-output-v1.json'
assert not path.exists(), 'Do not overwrite reviewer evidence'
path.write_text(json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'occurrences': [row['index'] for row in occurrences], 'issues': 2,
                  'selector_cases_passed': len(cases), 'finite_precondition_present': True,
                  'prior_guard_code_bytes_unchanged': True, 'no_neural_execution': True}, ensure_ascii=False))
