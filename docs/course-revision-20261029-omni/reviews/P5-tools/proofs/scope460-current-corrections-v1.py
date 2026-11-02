"""Independent saved-row/metadata review; no model or provider imports/calls."""

from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[5]
PROOFS = Path(__file__).resolve().parent
SNAPSHOT = Path('/workspace/work/omni-P5-full-tools-review-source-v7')
OLD = Path('/workspace/work/omni-P5-full-tools-review-source-v6')
FINAL = Path('/workspace/work/omni-final-completed-41004229262/verified')
MODELS = Path('/workspace/work/omni-models-v1')
PRODUCT = Path('docs/omni-products/v1')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def no_duplicates(items):
    result = {}
    for key, value in items:
        assert key not in result, ('duplicate JSON key', key)
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_bytes(), object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)))


scope = read(ROOT / 'docs/course-revision-20261029-omni/reviews/P5-tools/scope-inventory-v7.json')
assert scope['count'] == 460 and scope['bytes'] == 40628131
strict_counts = Counter()
for binding in scope['checked_file_scope']:
    path = binding['path']
    raw = (ROOT / path).read_bytes()
    assert raw == (SNAPSHOT / path).read_bytes()
    assert len(raw) == binding['bytes'] and sha_bytes(raw) == binding['sha256']
    text = raw.decode('utf-8')
    suffix = Path(path).suffix
    if suffix == '.json':
        read(ROOT / path)
        strict_counts['JSON'] += 1
    elif suffix == '.toml':
        tomllib.loads(text)
        strict_counts['TOML'] += 1
    elif suffix == '.py':
        compile(text, path, 'exec', dont_inherit=True)
        strict_counts['Python'] += 1
    else:
        strict_counts['other_UTF8'] += 1

summary = read(ROOT / PRODUCT / 'results-summary.json')
definition = read(ROOT / PRODUCT / 'data-definition.json')
raw = {name: read(FINAL / 'measurements' / name / 'results.json')
       for name in ('av', 'sound', 'choice', 'integration')}
for part, binding in summary['final']['raw_results'].items():
    assert sha(FINAL / binding['archive_path']) == binding['sha256']

# Classify by the actual immutable payload, never by a family-name complement.
by_family = {row['family']: row for row in raw['integration']['auto']['rows']}
assert len(by_family) == 100
meta_a = [row for row in definition['Auto']['families'] if row['gold_route'] == 'A']
assert {row['family'] for row in meta_a} == {key for key, row in by_family.items() if row['gold_route'] == 'A'}
members = defaultdict(list)
for row in meta_a:
    payload = row['payload']
    assert set(payload) in ({'old_A_input'}, {'video_audio_definition'})
    task = payload.get('old_A_input', {}).get('task')
    if set(payload) == {'video_audio_definition'}:
        group = 'AV_sound_families'
    elif task == 'tool_call':
        group = 'calculator_families'
    else:
        assert task in {'vision_clothing', 'vision_relation', 'ocr', 'voice_qa'}
        group = 'legacy_media_families'
    members[group].append({'family': row['family'], 'task': task})
group_counts = {key: len(value) for key, value in members.items()}
assert group_counts == {'calculator_families': 15, 'legacy_media_families': 4, 'AV_sound_families': 1}
assert sum(group_counts.values()) == len({row['family'] for rows in members.values() for row in rows}) == 20

# Enumerate the exact, limited author transformations. Every other retained
# field must be byte-decoded-value identical to its original saved measurement.
derived = {}
route_groups = defaultdict(list)
for row in by_family.values():
    route_groups[row['gold_route']].append(row)
derived['/integration/auto/by_gold_route'] = {
    route: {'families': len(rows), **{
        field: sum(row[field] for row in rows) for field in
        ('route_correct', 'gold_route_end_to_end_correct', 'real_route_end_to_end_correct')
    }} for route, rows in route_groups.items()
}
derived['/integration/auto/A_subgroups'] = group_counts
derived['/integration/legacy_binding/old_A_manifest_sha256'] = raw['integration']['legacy_binding']['old_A']['manifest_sha256']
derived['/integration/old_A_retention/counterfactual_diagnostics/example'] = raw['integration']['old_A_retention']['counterfactual_diagnostics']['rows'][0]
candidate_groups = defaultdict(list)
for row in raw['choice']['candidate_count_secondary']['rows']:
    candidate_groups[row['total_choices_including_none']].append(row)
derived['/C/candidate_count_secondary/accuracy_by_total_choices_including_none'] = {
    str(count): {'correct': sum(row['answer'] == row['gold'] for row in rows),
                 'observations': len(rows), 'parent_families': len({row['family'] for row in rows})}
    for count, rows in candidate_groups.items()
}
timing = raw['sound']['actual_playable_timing']
timing_statistics = {}
for field in ('first_playable_seconds', 'whole_code_generation_seconds', 'whole_completion_seconds'):
    observations = [row[field] for row in timing]
    assert all(type(value) is float and value > 0 for value in observations)
    timing_statistics[field] = {'minimum': min(observations), 'median': statistics.median(observations), 'maximum': max(observations)}
derived['/sound/actual_playable_timing'] = {
    'observations': len(timing), 'seconds': timing_statistics,
    'scope': 'CPU two Torch threads, real yielded chunks after complete autoregressive code generation; excludes interpreter startup/WAV write/playback',
}
checked_leaves = []


def projection(saved, original, pointer):
    if pointer in derived:
        assert saved == derived[pointer], pointer
        checked_leaves.append(pointer)
    elif isinstance(saved, dict):
        assert isinstance(original, dict), pointer
        removed = set(original) - set(saved)
        expected_removed = {'old_A'} if pointer == '/integration/legacy_binding' else set()
        assert removed <= {'rows', 'records'} | expected_removed, (pointer, removed)
        for key, value in saved.items():
            child = pointer + '/' + key
            assert key in original or child in derived, child
            projection(value, original.get(key), child)
    else:
        assert saved == original, pointer
        checked_leaves.append(pointer)


for key, part in [('AV', 'av'), ('sound', 'sound'), ('C', 'choice'), ('integration', 'integration')]:
    projection(summary[key], raw[part], '/' + key)
assert summary['capability_verdict'] == read(FINAL / 'final-report.json')
assert summary['capability_verdict']['promote_C_service'] is False
assert summary['capability_verdict']['global_targets_all_met'] is False
selection = read(ROOT / PRODUCT / 'model-selection.json')
assert summary['selection'] == {key: selection[key] for key in ('selected', 'temperature', 'threshold', 'policy_feasible')}
assert sha(ROOT / PRODUCT / 'model-selection.json') == summary['final']['selection_sha256']

# The missing intermediate outputs cannot be reconstructed by this review.
trace_shapes = {}
for route in ('B', 'C'):
    rows = route_groups[route]
    shapes = Counter()
    for row in rows:
        for key in ('gold_route_result', 'real_route_result'):
            result = row[key]
            if route == 'B':
                assert set(result) == {'correct'}
            else:
                assert set(result) == {'correct', 'actual', 'choice_accepted', 'choice_decision', 'scope'}
                assert set(result['actual']) == {'choice', 'info', 'rubric'}
            shapes[','.join(sorted(result))] += 1
        assert set(row['route_logits']) == {'answer', 'ids', 'logits', 'probabilities', 'p_max'}
    trace_shapes[route] = {'families': len(rows), 'gold_and_real_result_shapes': dict(shapes)}
limits = summary['measurement_record_limits']['Auto']
assert 'without answers' in limits and 'without service logits or choice_info prediction' in limits
assert 'Intermediate adaptation is not persisted' in limits
assert 'not a new independent population' in summary['scope']

old_summary = ROOT / 'docs/course-revision-20261029-omni/evidence/author-results-summary-pre-technical-fix-v1.json'
assert old_summary.read_bytes() == (OLD / PRODUCT / 'results-summary.json').read_bytes()
assert read(old_summary)['integration']['auto']['A_subgroups'] == {'calculator_families': 15, 'AV_sound_families': 5}
assert read(old_summary)['capability_verdict'] == summary['capability_verdict']
rejections = []
for bad in [read(old_summary)['integration']['auto']['A_subgroups'],
            {'calculator_families': 15, 'legacy_media_families': 3, 'AV_sound_families': 2}]:
    try:
        projection(bad, None, '/integration/auto/A_subgroups')
    except AssertionError:
        rejections.append({'input': bad, 'rejected': True})
    else:
        raise AssertionError('incorrect subgroup accepted')

# The original checker/test assertions, including command order, stay intact.
test_path = ROOT / 'tests/test_windows_student_commands.py'
assert test_path.read_bytes() == (OLD / test_path.relative_to(ROOT)).read_bytes()
spec = importlib.util.spec_from_file_location('review_windows_contract', test_path)
windows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(windows)
windows_problems = {page: windows.page_command_problems(page) for page in windows.PAGES}
assert all(not problems for problems in windows_problems.values())
workflow = (ROOT / '.github/workflows/windows-student-commands.yml').read_text()
assert "-Commands 'echo $?'" in workflow and "-Commands '$LASTEXITCODE'" in workflow
assert 'exit-code observation belongs to the model check skipped above' in workflow
assert 'tool-name explanations before the English maintainer command list' in workflow
assert 'tool-name explanations of the Chinese maintainer commands already listed above' in workflow

# Resolve the final historical reference by its real immutable Git object.
historical_revision = '1659c96c06c6ccb6ae02a4aba8aea827672148cf'
historical_path = '.github/workflows/omni-execution.yml'
historical_bytes = subprocess.run(['git', 'show', historical_revision + ':' + historical_path], cwd=MODELS,
                                  check=True, capture_output=True).stdout
assert len(historical_bytes) == 2509
assert sha_bytes(historical_bytes) == 'd6be4c2ef5e868fdb3a769f23d9afe14f27919cb7d32cef5cb8c985e117c87fb'
receipt = read(ROOT / 'docs/course-revision-20261029-omni/evidence/P5-technical-source-corrections-author-v1.json')
receipt_bindings = []


def verify_links(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            assert sha(ROOT / value['path']) == value['sha256'], value['path']
            receipt_bindings.append(value['path'])
        for child in value.values():
            verify_links(child)
    elif isinstance(value, list):
        for child in value:
            verify_links(child)


verify_links(receipt)
result = {
    'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
    'scope_inventory_sha256': sha(ROOT / 'docs/course-revision-20261029-omni/reviews/P5-tools/scope-inventory-v7.json'),
    'all_460_live_and_frozen_snapshot_bytes_match': True, 'strict_parse_counts': dict(strict_counts),
    'summary_sha256': sha(ROOT / PRODUCT / 'results-summary.json'),
    'original_raw_projection_checks': len(checked_leaves), 'permitted_explicit_derived_fields': sorted(derived),
    'A_subgroups_from_frozen_payload': {'counts': group_counts, 'families': dict(members)},
    'original_trace_shapes': trace_shapes,
    'original_41162_byte_wrong_summary_preserved': {'sha256': sha(old_summary), 'bytes': old_summary.stat().st_size},
    'wrong_subgroup_controls': rejections, 'unchanged_capability_gates': True,
    'Windows_all_current_page_command_sequences': windows_problems,
    'Windows_checker_test_unchanged_from_original450': {'path': str(test_path.relative_to(ROOT)), 'sha256': sha(test_path)},
    'Windows_workflow_sha256': sha(ROOT / '.github/workflows/windows-student-commands.yml'),
    'Windows_skip_rationale_independently_inspected': 'POSIX exit probe and the immediately preceding deliberately skipped HF check have no native invocation to observe. Tool-name-only inline explanations are separately registered in actual Chinese/English order; real maintainer invocations remain in ci.yml. Invoke-Page was not used to fake the skipped process exit.',
    'author_receipt_exact_link_bindings_checked': receipt_bindings,
    'historical_reference_final_triage': {
        'original_record': 'infrastructure/execution-probe-40957487215-failure-analysis-v1.json',
        'pointer': '/read_only_inspection_assessment/reviewed_files/1', 'path': historical_path,
        'commit': historical_revision, 'bytes': len(historical_bytes), 'sha256': sha_bytes(historical_bytes),
        'meaning': 'Exact historical read-only inspection source, not a successful paid execution or current source approval.',
    },
    'limits': ['No model imports, weight deserialization, neural predictions, cloud execution, new permit, retry, publication or frozen input changes.',
               'This metadata projection does not recreate missing Auto B answers, service logits, choice_info or intermediate adaptation.',
               'Linux command synchronization checks and the full offline suite do not certify a new native Windows T11 run.',
               'Other textbook reviewers retain ownership of their individual originals and disposition; referenced reports are not blanket-resolved here.'],
}
out = PROOFS / 'scope460-current-corrections-result-v1.json'
with out.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'scope': 460, 'projection_checks': len(checked_leaves), 'A_groups': group_counts,
                  'Windows_command_problems': sum(map(len, windows_problems.values())),
                  'historical_reference_resolved': True, 'neural_calls': 0}))
