"""Current whole-guide bounded recheck; original GPU evidence is byte-checked reuse."""
import ast, copy, difflib, hashlib, importlib.util, json, re, shlex, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
EVIDENCE = OUT.parent
ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
import torch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

report = json.loads((EVIDENCE / 'report.round1.final.original.json').read_text())
old = (EVIDENCE / 'TRAINING.initial.raw.md').read_bytes()
current = (ROOT / 'docs/natural-assistant/v4/TRAINING.md').read_bytes()
assert sha(ROOT / 'docs/natural-assistant/v4/TRAINING.md') == '657e84b95344bc751f74bdf3bf37156b354fc518b007958c97586c6743712476'
assert len(current.splitlines()) == 327
changes = [i + 1 for i, (a, b) in enumerate(zip(old.splitlines(), current.splitlines())) if a != b]
assert changes == [26] and len(old.splitlines()) == len(current.splitlines())
unchanged = {}
def same(path, expected, group):
    actual = sha(ROOT / path)
    assert actual == expected, (path, actual, expected)
    unchanged[path] = {'sha256': actual, 'group': group, 'matches_own_initial_record': True}

for s in report['sources']:
    if s['kind'] == 'repository_code': same(s['path'], s['sha256'], 'repository_source')
for a in report['artifacts']: same(a['path'], a['sha256'], 'initial_durable_evidence')
for p, h in report['figure_sha256'].items(): same(p, h, 'necessary_svg')
provenance = json.loads((EVIDENCE / 'read-view-provenance.json').read_text())
for r in provenance['read_order']:
    if r['path'] != 'docs/natural-assistant/v4/TRAINING.md': same(r['path'], r['sha256'], 'prerequisite_or_contract')
retrievals = json.loads((EVIDENCE / 'authority-retrieval-receipts.json').read_text())
for r in retrievals:
    if r.get('status') == 200:
        same('outputs/natural-v4/factual-research/training-v4/' + r['file'], r['sha256'], 'own_retrieved_original_authority')
for name in ['cpu-audit.v2.stdout.json', 'validation-audit.initial.stdout.json']:
    audit = json.loads((EVIDENCE / name).read_text())
    for p, h in audit['source_sha256'].items(): same(p, h, 'original_audit_input')
cpu = json.loads((EVIDENCE / 'cpu-audit.v2.stdout.json').read_text())
train = cpu['original_training_audit']
train_dir = 'outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review'
for name, key in [('training.json', 'training_sha256'), ('execution.json', 'execution_sha256')]:
    same(train_dir + '/' + name, train[key], 'original_gpu_training_input')
same(str(Path(train_dir).parent / 'result.json'), train['wrapper_sha256'], 'original_gpu_training_input')
for cp in train['archives']:
    same(train_dir + f"/checkpoints/step-{cp['step']:06d}/adapter_model.safetensors", cp['adapter_sha256'], 'original_gpu_adapter')

# Parse current commands again, without launching their training or heavy preparation.
doc = current.decode('utf-8')
spec = importlib.util.spec_from_file_location('round2_natural_cli', ROOT / 'scripts/natural_assistant.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
blocks = re.findall(r'```bash\n(.*?)\n```', doc, re.S)
parsed = []
for block in blocks:
    subprocess.run(['bash', '-n'], input=block, text=True, capture_output=True, check=True)
    for line in block.replace('\\\n', ' ').splitlines():
        if line.startswith('.venv-natural/bin/python scripts/natural_assistant.py '):
            args = shlex.split(line)[2:]
            args = [('1039,2077' if x == '$PENDING_CHECKPOINTS' else '/fixture/selected-adapter' if x == '$SELECTED_ADAPTER' else x) for x in args]
            ns = cli.parser().parse_args(args)
            parsed.append({'stage': ns.stage, 'model_revision': ns.model_revision, 'asr_variant': ns.asr_variant, 'max_tokens': ns.max_tokens, 'max_new_tokens': ns.max_new_tokens, 'checkpoint_steps': ns.checkpoint_steps, 'selected_only': ns.selected_only})
assert parsed == cpu['commands']['actual_cli_parser_cases'] and len(blocks) == 10

# Execute the actual timer AST slice with disclosed substitutes for process/clock/I/O.
# It retains the original statements and branch structure from started through execution assignment.
tree = ast.parse((ROOT / 'scripts/modal_natural.py').read_text())
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'execute_stage')
start = next(n for n in fn.body if isinstance(n, ast.Assign) and n.lineno == 835)
trial = next(n for n in fn.body if isinstance(n, ast.Try) and n.lineno == 836)
stop_index = next(i for i, n in enumerate(trial.body) if isinstance(n, ast.Assign) and n.lineno == 847)
slice_nodes = [copy.deepcopy(start)] + copy.deepcopy(trial.body[:stop_index + 1])
events = []
class FakePath:
    def __init__(self, name='directory'): self.name = name
    def __truediv__(self, name): return FakePath(name)
    def open(self, *args, **kwargs): events.append('log_open'); return self
    def __enter__(self): return self
    def __exit__(self, *args): events.append('log_close')
    def is_file(self): events.append('result_exists_check'); return True
    def read_text(self): events.append('result_read'); return '{"status":"completed"}'
def fake_run(*args, **kwargs): events.extend(['process_start', 'process_exit'])
def clock(): events.append('clock_start' if not events else 'clock_stop'); return 100 if len(events) == 1 else 103
def loads(value): events.append('result_json_parse'); return json.loads(value)
namespace = {'time': SimpleNamespace(monotonic=clock), 'directory': FakePath(), 'subprocess': SimpleNamespace(run=fake_run, STDOUT=-2), 'args': [], 'options': {'max_seconds': 3600, 'adapter_checkpoint': 'final', 'runtime_options_sha256': 'fixture', 'runtime_options': {}}, 'stage': 'train', 'json': SimpleNamespace(loads=loads), 'run_id': 'fixture', 'batch_id': 'fixture', 'revision': 'fixture', 'manifest_sha': 'fixture', 'commits': [], 'adapter_sha': None, 'adapters': [], 'selection_proof': None, 'external_metadata_sha': None, 'EXTERNAL_STAGES': ()}
exec(compile(ast.fix_missing_locations(ast.Module(body=slice_nodes, type_ignores=[])), '<original execute_stage timer AST slice>', 'exec'), namespace)
expected_events = ['clock_start', 'log_open', 'process_start', 'process_exit', 'log_close', 'result_exists_check', 'result_read', 'result_json_parse', 'clock_stop']
assert events == expected_events and namespace['result']['execution']['seconds'] == 3
validation = json.loads((EVIDENCE / 'validation-audit.initial.stdout.json').read_text())
seconds = validation['training_runner_subprocess_seconds']
assert round(seconds, 3) == 1728.676 and round(seconds / 60, 2) == 28.81
assert '開啟日誌、啟動Python訓練前開始' in doc.splitlines()[25] and '訓練結束並讀入結果後停止' in doc.splitlines()[25]

output = {'scope': 'Genuine full-current guide re-read plus byte-checked reuse of own prior original-authority, CPU and GPU-record audit; no new GPU/model/ASR or blind semantic grading.', 'current_document_sha256': sha(ROOT / 'docs/natural-assistant/v4/TRAINING.md'), 'line_count': 327, 'changed_lines': changes, 'diff': ''.join(difflib.unified_diff(old.decode().splitlines(True), doc.splitlines(True), fromfile='initial', tofile='current')), 'unchanged_input_and_evidence': unchanged, 'current_commands': {'bash_syntax_blocks': len(blocks), 'actual_cli_parser_cases': parsed}, 'timer_probe': {'source': 'scripts/modal_natural.py', 'source_sha256': sha(ROOT / 'scripts/modal_natural.py'), 'locator': 'execute_stage835–869; AST nodes835 and836 body837–847 inclusive', 'substitutes': 'Clock, process and filesystem operations are tiny disclosed event doubles. This checks timer ordering, not wall-clock GPU duration.', 'expected_events': expected_events, 'observed_events': events, 'observed_fixture_seconds': namespace['result']['execution']['seconds'], 'original_record_seconds': seconds, 'recomputed_minutes': seconds / 60, 'corrected_line26': doc.splitlines()[25], 'conclusion': 'Corrected outer-time start and stop descriptions now agree with actual source.'}, 'environment': {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()), 'bf16_supported': str(torch.cuda.is_bf16_supported())}}
print(json.dumps(output, ensure_ascii=False, indent=2))
