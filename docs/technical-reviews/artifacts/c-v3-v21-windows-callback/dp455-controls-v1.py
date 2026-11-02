"""Source21 original-owner Windows text/YAML checks; never execute page commands."""

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
COURSE = ROOT / 'docs/course-revision-20261101-c-v3'
WORKFLOW = '.github/workflows/windows-student-commands.yml'
TEST = 'tests/test_windows_student_commands.py'
ORIGINAL = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-tools/technical-bounded-original-v2.json'
PRIOR = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-tools/bounded-windows-current-source18-v1.json'


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def read(path):
    return json.loads(path.read_bytes())


scope_path = ARTIFACTS / 'dp455-read-scope-v1.json'
scope = read(scope_path)
dispatch = read(ROOT / scope['dispatch']['path'])
original = read(ORIGINAL)
assert ref(ORIGINAL)['sha256'] == '1ee345294b61d037236ebb15da416ed15eab6b6dd800fcf0c5f59d252955073c'
assert original['reviewer_task'] == '/root/omni_p5_full_tools_review'
assert original['issues'][0] == dispatch['rows'][0]['original_issue']
assert dispatch['rows'][0]['index'] == 455
assert original['issues'][0]['necessity'] == 'necessary'
assert ref(PRIOR)['sha256'] == 'd8611016acb9fb2a15d050121c6b3e18f29a926ff85add9dd493f960ef953c26'
old_checked = {row['path']: row for row in read(PRIOR)['checked_files']}
for row in scope['checked_files']:
    assert ref(ROOT / row['path']) == row
assert ref(ROOT / TEST) == old_checked[TEST]

spec = importlib.util.spec_from_file_location('v21_original_owner_windows_contract', ROOT / TEST)
windows = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = windows
spec.loader.exec_module(windows)
assert 'torch' not in sys.modules
text = (ROOT / WORKFLOW).read_text()
copies, call_problems = windows.workflow_copies()
assert not call_problems
positive = {page: windows.page_command_problems(page) for page in windows.PAGES}
assert all(not value for value in positive.values())
assert windows.page_quote_problems() == []
five = ('echo $?', '$LASTEXITCODE', 'pytest', 'ruff check', 'ruff format --check')
ten = []
for page in ('README.md', 'README_en.md'):
    commands, problems = windows.windows_commands(page)
    assert not problems
    for command in five:
        assert sum(item.text == command for item in commands) == 1
        matching = [item for item in copies if page in item.pages and item.text == command]
        assert len(matching) == 1 and matching[0].verb == 'Skip-Page'
        ten.append({'page': page, 'command': command, 'verb': matching[0].verb, 'step': matching[0].step})

english = "\n".join([
    "          Skip-Page -Pages README_en.md -Reason 'tool-name explanations before the English maintainer command list' -Commands @'",
    '          pytest', '          ruff check', '          ruff format --check', "          '@", ''])
chinese = "\n".join([
    "          Skip-Page -Pages README.md -Reason 'tool-name explanations of the Chinese maintainer commands already listed above' -Commands @'",
    '          pytest', '          ruff check', '          ruff format --check', "          '@", ''])
echo = "          Skip-Page -Pages README.md,README_en.md -Reason 'POSIX exit-code example; PowerShell uses LASTEXITCODE' -Commands 'echo $?'\n"
exit_code = "          Skip-Page -Pages README.md,README_en.md -Reason 'exit-code observation belongs to the model check skipped above' -Commands '$LASTEXITCODE'\n"
for block in (english, chinese, echo, exit_code):
    assert text.count(block) == 1
training = (ROOT / 'course/training.md').read_text()
training_copies = []
for command in (
    'uv run --extra cpu python -m scripts.omni_revision.public_demo av --model-root checkpoints/omni-products-v1',
    'uv run --extra cpu python -m scripts.course_product download',
    'uv run --extra cpu python -m scripts.course_product judge',
):
    assert command in training
    matching = [item for item in copies if 'course/training.md' in item.pages and item.text == command]
    assert len(matching) == 1 and matching[0].verb == 'Invoke-Page'
    training_copies.append({'command': command, 'verb': matching[0].verb, 'step': matching[0].step,
                            'inspection': 'Textual match only; this reviewer did not execute it.'})
assert 'fixed recipes use --device cuda; nothing run here needs this download' in text
assert '命令大多寫成' in training
assert 'source21' not in training or isinstance(training, str)

before_path = Path('/workspace/work/omni-P6-source18-windows-original-owner-snapshot-v1/windows-student-commands.yml')
before = before_path.read_text()
end = '      - name: 訓練頁：下載固定新版模組並以CPU操作'
assert before.count(end) == text.count(end) == 1
assert before.split(end, 1)[0] == text.split(end, 1)[0]
# This unchanged prefix includes all helper functions and the original T9 square
# probes/strong loss, gradient and JSON assertions. No native run is inferred.
assert 'Assert-TrainReport' in text.split(end, 1)[0]
assert 'Assert-FiniteNumber' in text.split(end, 1)[0]
for marker in ('--shape square', 'grad_norm', 'loss'):
    assert marker in text.split(end, 1)[0]

native_read = windows.read
judge_line = "          Invoke-Page -Pages course/training.md -Commands 'uv run --extra cpu python -m scripts.course_product judge' | Tee-Object -Variable shown\n"
assert text.count(judge_line) == 1
variants = [
    ('remove_echo_registration', text.replace(echo, '', 1), ('README.md', 'README_en.md')),
    ('remove_LASTEXITCODE_registration', text.replace(exit_code, '', 1), ('README.md', 'README_en.md')),
    ('remove_English_tool_name_registration', text.replace(english, '', 1), ('README_en.md',)),
    ('remove_Chinese_tool_name_registration', text.replace(chinese, '', 1), ('README.md',)),
    ('move_English_tool_names_after_full_commands', text.replace(english, '', 1).replace(chinese, english + chinese, 1), ('README_en.md',)),
    ('remove_new_training_Cv3_judge_copy', text.replace(judge_line, '', 1), ('course/training.md',)),
]
negative = []
try:
    for name, variant, pages in variants:
        windows.read = lambda path, value=variant: value if path == windows.WORKFLOW else native_read(path)
        failures = {page: windows.page_command_problems(page) for page in pages}
        assert all(value for value in failures.values()), name
        negative.append({'case': name, 'rejected': True, 'in_memory_only': True, 'problems': failures})
finally:
    windows.read = native_read
assert all(not windows.page_command_problems(page) for page in windows.PAGES)
assert not windows.page_quote_problems()
assert 'torch' not in sys.modules

author_path = ROOT / 'docs/course-revision-20261029-omni/reviews/author-integration-and-visuals-findings-v1.json'
assert ref(author_path)['sha256'] == '6ad8f813282795f4c79ec6e36d8a380faec016b45304db4bfae0c270ef20de01'
author_issue = read(author_path)['issues'][2]
assert author_issue['id'] == 'AUTHOR-INTEGRATION-WINDOWS-SYNC' and author_issue['necessity'] == 'necessary'
ci = (ROOT / '.github/workflows/ci.yml').read_text()
for token in ('ruff check .', 'ruff format --check .', 'os: windows-latest', 'pytest --junitxml=outputs/pytest-results.xml'):
    assert token in ci
for row in scope['checked_files']:
    assert ref(ROOT / row['path']) == row
assert ref(ROOT / scope['training_context']['current']['path']) == scope['training_context']['current']
assert '15 passed in 0.13s' in (ARTIFACTS / 'dp455-targeted-v1.log').read_text()
result = {
    'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
    'dispatch': scope['dispatch'], 'read_scope': ref(scope_path), 'checked_files': scope['checked_files'],
    'source_freeze': scope['source_freeze'], 'training_context': scope['training_context'],
    'original_DP455': {'report': ref(ORIGINAL), 'json_pointer': '/issues/0', 'necessity': 'necessary',
                       'original_created_at_preserved': original['created_at']},
    'additional_author_DP287': {'report': ref(author_path), 'json_pointer': '/issues/2', 'necessity': 'necessary',
                                'scope': 'Independently checked file/command synchronization only. Coordinator supplies page-reader refs separately.'},
    'current_four_page_command_problems': positive, 'workflow_call_problems': [], 'current_quote_problems': [],
    'ten_original_registration_entries': ten, 'source21_training_copy_entries': training_copies,
    'test_bytes_unchanged_from_own_source18_closure': True,
    'workflow_prefix_before_public_module_steps_byte_identical_to_own_source18': True,
    'static_reason_check': 'POSIX echo has a platform reason; LASTEXITCODE follows a deliberately skipped HF model check and is not fake-tested through Invoke-Page reset0. Inline maintenance names follow actual English-before/Chinese-after page order. Existing strong helpers/T9 square probes/JSON/loss/grad checks are unchanged; only command-copy intent is reviewed, not actual model outputs.',
    'negative_controls': negative, 'all_six_rejected': True,
    'targeted_test_log': ref(ARTIFACTS / 'dp455-targeted-v1.log'), 'targeted_tests': 15, 'seconds': 0.13,
    'additional_static_CI_context': ref(ROOT / '.github/workflows/ci.yml'),
    'source_and_old_reports_changed': False,
    'limits': 'Pure Linux text/YAML and in-memory controls; no actual Windows/PowerShell, model/kernel/training, command download, provider/Modal/final, Git/stage, 17py edit, whole-book review/gate, collection or seal. Current source21 training context was read, not approved as a whole-page technical judgment.',
}
with (ARTIFACTS / 'dp455-controls-result-v1.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'current_DP455_contract': 'pass', 'independent_DP287_file_sync': 'pass',
                  'targeted_tests': 15, 'negative_controls_rejected': 6, 'native_Windows_claim': False}))
