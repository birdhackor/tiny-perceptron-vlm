"""Original003 owner current-file closure, pure text/YAML and private controls."""

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parent.parent
PROOFS = Path(__file__).resolve().parent
WORKFLOW = '.github/workflows/windows-student-commands.yml'
TEST = 'tests/test_windows_student_commands.py'
PAGES = ('README.md', 'README_en.md')
ISSUE = 'P5-TOOLS-WINDOWS-COMMAND-COVERAGE-003'


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def checked_document(path, expected):
    assert ref(path)['sha256'] == expected
    return json.loads(path.read_bytes())


original_path = BASE / 'technical-bounded-original-v2.json'
original = checked_document(original_path, '1ee345294b61d037236ebb15da416ed15eab6b6dd800fcf0c5f59d252955073c')
old_current = checked_document(BASE / 'technical-current-v4.json', '416f0d7a3402ce3a31f3f94ed0328989b1f829a065b2befa15b03813db36100a')
prior_bounded = checked_document(BASE / 'bounded-source18-windows-sync-v1.json', 'bb74c811554289517c6ecb6f55cba3c02515a6b636586411e55f149e13c46b77')
issue = original['issues'][0]
assert issue['issue_id'] == ISSUE and issue['necessity'] == 'necessary'
old_resolution = next(row for row in old_current['original_issue_resolutions'] if row['issue_id'] == ISSUE)
assert old_resolution['current_status'] == 'resolved'
checked = [ref(ROOT / name) for name in (WORKFLOW, TEST, *PAGES)]
assert set(issue['affected_files']) == {row['path'] for row in checked}
current_code = {row['path']: row['sha256'] for row in checked[:2]}
assert current_code == {row['path']: row['sha256'] for row in prior_bounded['checked_files']}
assert any(current_code[row['path']] != row['sha256'] for row in old_current['checked_files'] if row['path'] in current_code)

spec = importlib.util.spec_from_file_location('owner003_actual_current_contract', ROOT / TEST)
windows = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = windows
spec.loader.exec_module(windows)
assert 'torch' not in sys.modules
current_workflow = (ROOT / WORKFLOW).read_text()
copies, problems = windows.workflow_copies()
assert problems == []
positive = {page: windows.page_command_problems(page) for page in PAGES}
assert all(not result for result in positive.values())
commands = ('echo $?', '$LASTEXITCODE', 'pytest', 'ruff check', 'ruff format --check')
ten_coverage_entries = []
for page in PAGES:
    expected, errors = windows.windows_commands(page)
    assert errors == []
    for command in commands:
        assert sum(row.text == command for row in expected) == 1
        actual = [row for row in copies if page in row.pages and row.text == command]
        assert len(actual) == 1 and actual[0].verb == 'Skip-Page'
        ten_coverage_entries.append({'page': page, 'command': command, 'verb': actual[0].verb, 'step': actual[0].step})

english = "\n".join([
    "          Skip-Page -Pages README_en.md -Reason 'tool-name explanations before the English maintainer command list' -Commands @'",
    '          pytest', '          ruff check', '          ruff format --check', "          '@", ''])
chinese = "\n".join([
    "          Skip-Page -Pages README.md -Reason 'tool-name explanations of the Chinese maintainer commands already listed above' -Commands @'",
    '          pytest', '          ruff check', '          ruff format --check', "          '@", ''])
echo = "          Skip-Page -Pages README.md,README_en.md -Reason 'POSIX exit-code example; PowerShell uses LASTEXITCODE' -Commands 'echo $?'\n"
exit_code = "          Skip-Page -Pages README.md,README_en.md -Reason 'exit-code observation belongs to the model check skipped above' -Commands '$LASTEXITCODE'\n"
for block in (english, chinese, echo, exit_code):
    assert current_workflow.count(block) == 1
assert "-Reason 'downloads a model from Hugging Face' -Commands 'python scripts/check_course_models.py --model text_foundation'" in current_workflow
assert '$global:LASTEXITCODE = 0' in current_workflow
native_read = windows.read
negative = []
variants = [
    ('remove_POSIX_echo_registration', current_workflow.replace(echo, '', 1), PAGES, ('echo $?',)),
    ('remove_skipped_HF_exit_observation_registration', current_workflow.replace(exit_code, '', 1), PAGES, ('$LASTEXITCODE',)),
    ('remove_English_three_inline_tool_names', current_workflow.replace(english, '', 1), ('README_en.md',), commands[2:]),
    ('remove_Chinese_three_inline_tool_names', current_workflow.replace(chinese, '', 1), ('README.md',), commands[2:]),
    ('move_English_inline_names_after_full_maintainer_list', current_workflow.replace(english, '', 1).replace(chinese, english + chinese, 1), ('README_en.md',), commands[2:]),
]
try:
    for name, variant, pages, missing in variants:
        windows.read = lambda path, value=variant: value if path == windows.WORKFLOW else native_read(path)
        failures = {page: windows.page_command_problems(page) for page in pages}
        assert all(result for result in failures.values()), name
        if name != 'move_English_inline_names_after_full_maintainer_list':
            assert all(all(command in '\n'.join(result) for command in missing) for result in failures.values()), name
        negative.append({'case': name, 'private_in_memory_only': True, 'rejected': True, 'problems': failures})
finally:
    windows.read = native_read
assert all(not windows.page_command_problems(page) for page in PAGES)
assert 'torch' not in sys.modules

ci = (ROOT / '.github/workflows/ci.yml').read_text()
for fragment in ('uv run --only-group dev ruff check .', 'uv run --only-group dev ruff format --check .',
                 'os: windows-latest', 'pytest --junitxml=outputs/pytest-results.xml'):
    assert fragment in ci
candidate_path = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/wave06/final-assembly-v1/manifest-author-received-v1.json'
candidate = checked_document(candidate_path, '118683e1476a54829d0cf59f157ca3cc21b11df8b5d20bab9eca4d1c47db0251')
dispositions_path = ROOT / candidate['dispositions']['path']
dispositions = checked_document(dispositions_path, candidate['dispositions']['sha256'])
decision = dispositions['items'][455]
assert decision['issue_id'] == ISSUE and decision['final_necessity'] == 'necessary'
assert decision['report']['sha256'] == ref(original_path)['sha256']
assert set(decision['affected_files']) == {row['path'] for row in checked}
assert '14 passed in 0.12s' in (PROOFS / 'current-source18-003-targeted-v1.log').read_text()
for row in checked:
    assert ref(ROOT / row['path']) == row
assert ref(original_path)['sha256'] == '1ee345294b61d037236ebb15da416ed15eab6b6dd800fcf0c5f59d252955073c'
assert ref(BASE / 'technical-current-v4.json')['sha256'] == '416f0d7a3402ce3a31f3f94ed0328989b1f829a065b2befa15b03813db36100a'

result = {
    'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
    'issue_id': ISSUE, 'original_report': ref(original_path), 'original_json_pointer': '/issues/0',
    'prior_current_resolution_report': ref(BASE / 'technical-current-v4.json'),
    'prior_current_resolution_pointer': '/original_issue_resolutions/2',
    'current_two_implementation_bytes_equal_own_source18_bounded_recheck': True,
    'prior_source18_bounded_report': ref(BASE / 'bounded-source18-windows-sync-v1.json'),
    'checked_files': checked, 'original_ten_specific_coverage_entries': ten_coverage_entries,
    'current_README_contract_problems': positive, 'workflow_call_recognition_problems': [],
    'private_negative_controls': negative, 'all_five_negative_controls_rejected': True,
    'targeted_existing_tests': {'passed': 14, 'seconds': 0.12, 'log': ref(PROOFS / 'current-source18-003-targeted-v1.log'),
                               'scope': 'Four page sequence cases, workflow call recognition, six comparator cases, three helper propagation cases; text/YAML only.'},
    'static_rationale': 'The POSIX echo registration is explicitly skipped for platform syntax. LASTEXITCODE belongs to the immediately preceding deliberately skipped HF check and is never faked by Invoke-Page resetting it to0. Inline pytest/ruff names are prose explanations, registered separately in true English-before/Chinese-after order; real maintenance invocations remain declared in CI. Manual rationale inspection is distinct from tests, which recognize both Invoke and Skip without judging why.',
    'additional_static_CI_context': ref(ROOT / '.github/workflows/ci.yml'),
    'additional_test_data_dependencies': [ref(ROOT / name) for name in ('course/first-steps.md', 'course/training.md')],
    'candidate_location_only': {'report': ref(candidate_path), 'dispositions': ref(dispositions_path), 'json_pointer': '/items/455',
                                'reviewed_scope': 'Only this original issue identity, necessity and four affected paths; no author552 candidate approval or data audit.'},
    'source_and_old_reports_unchanged': True, 'model_module_imported': False,
    'limits': ['Current original-owner003 bounded closure only. No new full460 semantic review, primary tool audit, fresh blind task, source edit, grade change or seal.',
               'No actual PowerShell/nativeWindows, command execution, network/provider, model/neural/training/kernel/GPU/Modal, Git, new science or full-book gate.',
               'The prospective kind=tool_audit per-issue verification still needs actual root receipt and candidate assembly. This proof creates neither collection nor closed-stage backfill.'],
}
output = PROOFS / 'current-source18-003-controls-result-v1.json'
with output.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'current_original003_contracts': 'pass', 'ten_specific_entries_verified': 10,
                  'negative_controls_rejected': 5, 'targeted_tests_passed': 14,
                  'checked_current_files': 4, 'model_import_or_execution': False}))
