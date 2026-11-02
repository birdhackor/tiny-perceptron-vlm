"""Original-owner bounded synchronization recheck; no workflow commands execute."""

import ast
import copy
import difflib
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parent.parent
PROOFS = Path(__file__).resolve().parent
WAVE = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/wave04'
BEFORE = WAVE / 'windows-sync-source18-before-v1'
WORKFLOW = '.github/workflows/windows-student-commands.yml'
TEST = 'tests/test_windows_student_commands.py'
SNAPSHOT = Path('/workspace/work/omni-P6-source18-windows-original-owner-snapshot-v1')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(raw), 'bytes': len(raw)}


prior_path = BASE / 'technical-current-v4.json'
assert ref(prior_path)['sha256'] == '416f0d7a3402ce3a31f3f94ed0328989b1f829a065b2befa15b03813db36100a'
prior = json.loads(prior_path.read_bytes())
unchanged, changed, total_bytes = [], [], 0
for binding in prior['checked_files']:
    raw = (ROOT / binding['path']).read_bytes()
    total_bytes += len(raw)
    actual = {'path': binding['path'], 'sha256': sha(raw), 'bytes': len(raw)}
    if actual == binding:
        unchanged.append({**actual, 'same_identity_as_original_owner_complete_basis': True})
    else:
        assert binding['path'] in {WORKFLOW, TEST}
        changed.append({'before': binding, 'current': actual})
assert len(unchanged) == 458 and len(changed) == 2
assert total_bytes == 40628635

receipt = json.loads((BEFORE / 'receipt.json').read_bytes())
original_hashes = {}
for row in receipt['files']:
    original = (ROOT / row['snapshot']).read_bytes()
    assert sha(original) == row['sha256']
    binding = next(item for item in prior['checked_files'] if item['path'] == row['current_path'])
    assert binding['sha256'] == sha(original)
    original_hashes[row['current_path']] = sha(original)

SNAPSHOT.mkdir(exist_ok=False)
rebuilt_diff = ''
for name in (WORKFLOW, TEST):
    old = (BEFORE / Path(name).name).read_bytes()
    current = (ROOT / name).read_bytes()
    target = SNAPSHOT / Path(name).name
    target.write_bytes(current)
    rebuilt_diff += ''.join(difflib.unified_diff(old.decode().splitlines(True), current.decode().splitlines(True),
                                               fromfile='before/' + name, tofile='current/' + name))
patch_path = WAVE / 'windows-sync-source18-fix-v1.patch'
assert sha(patch_path.read_bytes()) == 'd439e79e1f48c37bd3f0f850541672a25465986469e092f98a9d59cd56774ca3'
assert rebuilt_diff.encode() == patch_path.read_bytes()
old_workflow = (BEFORE / Path(WORKFLOW).name).read_text()
current_workflow = (ROOT / WORKFLOW).read_text()
old_tree = ast.parse((BEFORE / Path(TEST).name).read_text())
current_tree = ast.parse((ROOT / TEST).read_text())


def normalized(tree):
    tree = copy.deepcopy(tree)
    assert isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
    tree.body = [node for node in tree.body[1:] if not (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'PREFIXES'
                                             for target in node.targets))]
    return ast.dump(tree, include_attributes=False)


assert normalized(old_tree) == normalized(current_tree)
old_yaml = yaml.load(old_workflow, Loader=yaml.BaseLoader)
current_yaml = yaml.load(current_workflow, Loader=yaml.BaseLoader)
old_outer, current_outer = copy.deepcopy(old_yaml), copy.deepcopy(current_yaml)
step_changes = []
for job_name in old_yaml['jobs']:
    old_steps = old_outer['jobs'][job_name].pop('steps')
    current_steps = current_outer['jobs'][job_name].pop('steps')
    assert len(old_steps) == len(current_steps)
    for before, current in zip(old_steps, current_steps):
        if before == current:
            continue
        assert {key: value for key, value in before.items() if key != 'run'} == {
            key: value for key, value in current.items() if key != 'run'}
        step_changes.append(current['name'])
assert old_outer == current_outer and len(step_changes) == 2
# The first workflow change is inside the fixed-recipe Skip-Page step, after
# all PowerShell helper definitions, job/trigger/permission/matrix setup.
matcher = difflib.SequenceMatcher(a=old_workflow.splitlines(), b=current_workflow.splitlines(), autojunk=False)
first_changed_line = min(i1 + 1 for tag, i1, _, _, _ in matcher.get_opcodes() if tag != 'equal')
assert first_changed_line >= 1022

freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json'
freeze = json.loads(freeze_path.read_bytes())
training = next(row for row in freeze['pages'] if row['page_id'] == 'training')
assert sha((ROOT / training['source']).read_bytes()) == training['source_sha256']
assert (ROOT / training['source']).read_bytes() == (ROOT / training['snapshot']).read_bytes()

spec = importlib.util.spec_from_file_location('source18_owner_actual_windows_contract', ROOT / TEST)
windows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(windows)
positive_pages = {page: windows.page_command_problems(page) for page in windows.PAGES}
assert all(not result for result in positive_pages.values())
assert windows.page_quote_problems() == []
negative = []
prefix = windows.PREFIXES['course/training.md']
windows.PREFIXES['course/training.md'] = ('命令一律寫成', prefix[1])
problems = windows.page_command_problems('course/training.md')
assert problems and any('命令一律寫成' in value for value in problems)
negative.append({'case': 'restore_old_prefix_anchor', 'rejected': True, 'problems': problems})
windows.PREFIXES['course/training.md'] = prefix
current_docstring = windows.__doc__
windows.__doc__ = ast.get_docstring(old_tree, clean=False)
problems = windows.page_quote_problems()
assert problems and all(phrase in '\n'.join(problems) for phrase in (
    '效果仍以驗證題核對', '這要在圖文訓練前後，用同一批文字測試題各核對一次', '先看 `chosen`、`rejected` 是否符合自己的判準'))
negative.append({'case': 'restore_old_three_quote_items', 'rejected': True, 'problems': problems})
windows.__doc__ = current_docstring
native_read = windows.read
fixed_prefix = "Skip-Page -Pages course/training.md -Reason 'fixed recipes use --device cuda; nothing run here needs this download' -Commands @'\n"
sft_download = '          .\\.venv\\Scripts\\python.exe scripts/fetch_training_assets.py --asset ultrachat-sft\n'
sft_run = '          .\\.venv\\Scripts\\python.exe -m scripts.course_experiments.run --experiment sft --device cuda\n'
needle = fixed_prefix + sft_download + sft_run
assert current_workflow.count(needle) == 1
missing_sft = current_workflow.replace(needle, fixed_prefix, 1)
style = '          .\\.venv\\Scripts\\python.exe scripts/infer.py checkpoints/style.pt --chat --prompt "0+5=?" --tokens 32 --temperature 0\n'
dpo = '          .\\.venv\\Scripts\\python.exe scripts/train.py --task dpo --checkpoint checkpoints/style.pt --data data/generated/preference/train.jsonl --train --steps 200 --output checkpoints/preferred.pt\n'
assert current_workflow.count(style + dpo) == 1
old_dpo_order = current_workflow.replace(style + dpo, dpo + style, 1)
for label, variant in [('remove_fixed_SFT_prerequisite_copies', missing_sft), ('restore_old_DPO_order', old_dpo_order)]:
    windows.read = lambda path, value=variant: value if path == windows.WORKFLOW else native_read(path)
    problems = windows.page_command_problems('course/training.md')
    assert problems, label
    negative.append({'case': label, 'rejected': True, 'problems': problems})
windows.read = native_read
assert all(not windows.page_command_problems(page) for page in windows.PAGES)
assert windows.page_quote_problems() == []

author_logs = []
for name, result in [('omni-P6-source18-pytest-v1.log', '2 failed, 2753 passed, 5 skipped in 45.47s'),
                     ('omni-P6-source18-windows-sync-targeted-v1.log', '192 passed in 1.18s'),
                     ('omni-P6-source18-windows-sync-full-v2.log', '2755 passed, 5 skipped in 45.89s')]:
    path = ROOT / 'outputs' / name
    assert result in path.read_text()
    author_logs.append({**ref(path), 'result': result, 'kind': 'actual author execution inspected, not reviewer rerun'})
for row in receipt['files']:
    assert sha((ROOT / row['snapshot']).read_bytes()) == row['sha256']
assert ref(prior_path)['sha256'] == '416f0d7a3402ce3a31f3f94ed0328989b1f829a065b2befa15b03813db36100a'
for item in changed:
    assert sha((ROOT / item['current']['path']).read_bytes()) == item['current']['sha256']
result = {
    'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
    'kind': 'same original owner bounded source18 two-file semantic recheck plus458 individual unchanged fingerprints',
    'prior_complete_basis': ref(prior_path), 'patch': ref(patch_path), 'before_receipt': ref(BEFORE / 'receipt.json'),
    'snapshot_directory': str(SNAPSHOT), 'changed_files': changed, 'unchanged_count': len(unchanged),
    'unchanged_files': unchanged, 'combined_reference_inventory_count': 460, 'current_combined_bytes': total_bytes,
    'exact_author_diff_reconstructed_from_before_and_current': True,
    'test_AST_identical_except_module_documentation_and_PREFIXES': True,
    'workflow_outer_configuration_and_all_other_steps_identical': True,
    'two_run_steps_changed': step_changes, 'first_workflow_changed_line': first_changed_line,
    'PowerShell_helper_definitions_and_native_checks_unchanged': True,
    'training_dependency_context': {'freeze': ref(freeze_path), 'page': training, 'live_and_source18_snapshot_match': True,
                                    'scope': 'Command order, literal anchors and quote/exclusion rationale only; no textbook correctness/readability/coherence review.'},
    'current_four_page_command_sequences': positive_pages, 'current_quote_contract_problems': [],
    'private_negative_controls': negative, 'negative_count': len(negative),
    'all_six_control_groups_expected': True, 'author_execution_logs_inspected': author_logs,
    'own_targeted_test_log': ref(PROOFS / 'source18-windows-sync-owner-targeted-v1.log'),
    'semantic_assessment': 'Current source18 really states the new prefix, fixedSFT dependency and baseline-inference-before-DPO sequence. The three revised exclusion quotations remain accurate: fixedVQA requiresCUDA and is intentionally skipped; manual500step modal demonstration does not claim held-out comparison; preference labels use reader criteria. Tests keep all functions/assertions. Skip-Page adds declaration of two already-published fixed commands without executing a download/training job.',
    'limits': ['Not a new complete460 blind review or substitute for the independently assigned T04 full tool reviewer.',
               'Only192 source/contract tests and six control groups were executed here; full2755pass is author execution evidence inspected as such.',
               'No PowerShell/nativeWindows workflow commands, download, real model final/teacher/training, Modal/provider/paid operation, source edit, publication, activation, new historical appendix, collection or seal.'],
}
out = PROOFS / 'source18-windows-sync-owner-controls-result-v1.json'
with out.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'changed': 2, 'unchanged_fingerprints': 458, 'test_logic_unchanged': True,
                  'six_control_groups_expected': True, 'author_full_run': '2755pass5skip, separately attributed',
                  'new_blind_full_review': False}))
