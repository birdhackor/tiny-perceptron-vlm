"""Owner recheck of a changed optional guide; no model execution or downloads."""
import ast
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import build_course, export_course, natural_assistant as cli

BASE = ROOT / 'docs/technical-reviews/artifacts'
def sha(data): return hashlib.sha256(data).hexdigest()
def section(text, number):
    match = re.search(r'^## ' + re.escape(number) + r' .+$', text, re.M)
    nxt = re.search(r'^## ', text[match.start() + 1:], re.M)
    return text[match.start():match.start() + 1 + nxt.start() if nxt else len(text)]
old = (BASE / 'natural-r2-v4-training-source.md').read_bytes()
current = (ROOT / 'docs/natural-assistant/TRAINING.md').read_bytes()
assert sha(old) == 'b042505a2fbe6dafc223db4b532496068ca29d7156f9abdecb2b2d9b3da839ce'
old_lines, new_lines = old.decode().splitlines(), current.decode().splitlines()
changed = [(i + 1, a, b) for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
assert len(old_lines) == len(new_lines) and len(changed) == 1
assert '--max-seconds 3300' in changed[0][1] and '--max-seconds 3300' in changed[0][2]
assert re.findall(r'```bash\n(.*?)```', old.decode(), re.S) == re.findall(r'```bash\n(.*?)```', current.decode(), re.S)
assert '從固定的圖文底座建立一份新的LoRA' in current.decode()
assert '先驗證自己的模型，再固定版本做最後測試' in current.decode()

# Execute the real CLI parser only; never call main(), run_train() or a model.
options = cli.parser().parse_args(['train', '--output', 'outputs/natural-my-experiment/train', '--steps', '180', '--learning-rate', '1e-4', '--max-seconds', '3300', '--device', 'cuda', '--dtype', 'bfloat16', '--local-files-only'])
assert options.max_seconds == 3300 and options.steps == 180 and options.learning_rate == 1e-4

# Check exact current source syntax and ordering; no synthetic/model run claimed.
module_path = ROOT / 'tiny_perceptron/natural_assistant.py'
module_text = module_path.read_text()
tree = ast.parse(module_text)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_train')
manifest = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.value).startswith('load_manifest('))
start = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'started')
load = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.value).startswith('load_core('))
loop = next(n for n in fn.body if isinstance(n, ast.For))
assert manifest.lineno < start.lineno < load.lineno < loop.lineno
assert ast.unparse(start.value) == 'time.monotonic()'
check = loop.body[0]
assert isinstance(check, ast.If) and ast.unparse(check.test) == 'time.monotonic() - started >= options.max_seconds'
assert any(isinstance(n, ast.Break) for n in check.body)
assert 'time_limit_checkpoint' in ast.unparse(check)
assert sum(isinstance(n, ast.Attribute) and n.attr == 'max_seconds' for n in ast.walk(fn)) == 1
updates = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'optimizer.step']
assert len(updates) == 1 and check.lineno < updates[0].lineno
elapsed = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "record['elapsed_seconds']")
last_hash = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "record['final_adapter_tensors']")
final_save = next(n for n in fn.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'save_checkpoint')
final_result = next(n for n in fn.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'write_json' and 'result.json' in ast.unparse(n))
assert loop.end_lineno < elapsed.lineno < last_hash.lineno < final_save.lineno < final_result.lineno
manifest_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_manifest')
assert 'Asset does not match frozen manifest SHA' in ast.get_source_segment(module_text, manifest_fn)
assert 'Asset does not match frozen manifest byte count' in ast.get_source_segment(module_text, manifest_fn)

chapter_path = ROOT / 'course/chapters/20.md'
chapter = chapter_path.read_bytes()
assert chapter == (BASE / 'natural-r2-v4-chapter20-source.md').read_bytes()
body = section(chapter.decode(), '20.8')
notebook = build_course.notebook('20.8', body.splitlines()[0].removeprefix('## 20.8 '), body.split('\n', 1)[1], chapter_path)
cells = [''.join(c['source']).encode() for c in notebook['cells'] if c['cell_type'] == 'code']
v3 = json.loads((BASE / 'natural-r2-v3-execution.json').read_text())
assert [sha(c) for c in cells] == [row['sha256'] for row in v3['code_cells']]
index = json.loads((ROOT / 'course/lesson-index.json').read_text())
targets = {(ROOT / p).resolve(): n + '.md' for n, p in export_course.DOCUMENTS.items()}
lesson_targets = {}
for item in index:
    path = (ROOT / item['source']).resolve()
    targets[path] = 'chapter-' + path.stem + '.md'
    lesson_targets.setdefault(path, {})[item['id']] = item['id'] + '.md'
url = 'https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/natural-assistant/TRAINING.md'
assert '[LoRA訓練實作](' + url + ')' in export_course.reading_markdown(body, chapter_path, targets, lesson_targets, 'main')
r2path = ROOT / 'course/README.md'
r2 = section(r2path.read_text(), 'R.2')
links = re.findall(r'\[[^\]]+\]\(([^)]+)\)', export_course.reading_markdown(r2, r2path, targets, lesson_targets, 'main'))
assert links == [r['published'] for r in json.loads((BASE / 'natural-r2-v2-links.json').read_text())['links']]

source_snapshot = '\n\n'.join([
    'Actual source excerpts read by /root/natural_factual_r_2. Not other reviewers conclusions.',
    'scripts/natural_assistant.py SHA-256 ' + sha((ROOT / 'scripts/natural_assistant.py').read_bytes()),
    (ROOT / 'scripts/natural_assistant.py').read_text(),
    'tiny_perceptron/natural_assistant.py SHA-256 ' + sha(module_path.read_bytes()),
    ast.get_source_segment(module_text, manifest_fn), ast.get_source_segment(module_text, fn)
]) + '\n'
(BASE / 'natural-r2-v5-budget-source.txt').write_text(source_snapshot)
result = {
    'training_old_sha256': sha(old), 'training_current_sha256': sha(current),
    'diff': {'changed_line': changed[0][0], 'changed_lines': len(changed), 'all_bash_commands_unchanged': True, 'scope': 'Only budget prose paragraph changed.'},
    'actual_cli_parser': {'max_seconds': options.max_seconds, 'steps': options.steps, 'learning_rate': options.learning_rate, 'stage': options.stage, 'note': 'Parsed actual CLI. Main/train never called.'},
    'source_order_checks': {'manifest_verification_line': manifest.lineno, 'timer_start_line': start.lineno, 'model_load_line': load.lineno, 'loop_budget_check_line': check.lineno, 'optimizer_step_line': updates[0].lineno, 'elapsed_record_line': elapsed.lineno, 'final_fingerprint_line': last_hash.lineno, 'final_save_line': final_save.lineno, 'final_result_line': final_result.lineno, 'max_seconds_references_in_run_train': 1, 'scope': 'AST/source inspection verifies a loop-entry budget, not a hard command wall-clock cutoff. No check during a single update; final hash checks and saving occur after elapsed_seconds is captured.'},
    'whole_chapter20_sha256': sha(chapter), 'current_20.8_sha256': sha(body.encode()),
    'generated_program_bytes_match_v3_cpu_run': [sha(c) for c in cells],
    'R.2_sha256': sha(r2.encode()), 'R.2_links_count': len(links), 'R.2_links_unchanged': True,
    'training_url': url, 'training_literal_route_unchanged': True,
    'scope': 'Same owner complete TRAINING/20.8 read, actual raw diff, source inspection and real CLI-parser/exporter/code-byte checks only. No GPU, installation, Git, model download/training, quality assurance, full guide pass or extra review stage.'
}
(BASE / 'natural-r2-v5-execution.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'training_current':sha(current), 'changed_lines':len(changed), 'actual_max_seconds':options.max_seconds, 'source_budget_check_line':check.lineno, 'R.2_links':len(links), 'chapter20_unchanged':True}))
