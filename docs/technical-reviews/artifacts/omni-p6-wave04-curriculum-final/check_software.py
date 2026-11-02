"""Inspect argument contracts without running a recipe or model."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
env = os.environ.copy()
env['PYTHONPATH'] = str(ROOT)
command = [sys.executable, str(ROOT / 'scripts/selftrained/train.py'), '--help']
r = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env)
spec = importlib.util.spec_from_file_location('curriculum_inspected_train', ROOT / 'scripts/selftrained/train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parser = module.parser()
args = parser.parse_args(['--records', '/unopened/example.jsonl', '--asset-dir', '/unopened/assets', '--output-dir', '/unopened/output', '--stage', 'joint', '--steps', '1', '--resume', '/unopened/own-full-checkpoint.pt', '--device', 'cpu'])
actions = {a.dest: {'options': a.option_strings, 'help': str(a.help)} for a in parser._actions if a.dest in ['resume','init_checkpoint','device','export_inference','sampling_mode']}
j = {'reviewer_task':'/root/omni_p6_technical_curriculum_final','created_at':datetime.now(timezone.utc).isoformat(),'command': ' '.join(command),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'parser_contract_without_main':actions,'parsed_own_checkpoint_without_opening_any_path': {'resume':args.resume,'device':args.device,'stage':args.stage},'failure_locator':'scripts/selftrained/train.py:679 --sampling-mode help contains unescaped 20%; argparse applies percent interpolation when formatting --help','dependency_group':tomllib.loads((ROOT / 'pyproject.toml').read_text())['dependency-groups']['notebook'],'environment':{'python':sys.version.split()[0],'device':'CPU; argparse only; no model forward or training'},'code_sha256':hashlib.sha256((ROOT / 'scripts/selftrained/train.py').read_bytes()).hexdigest()}
(Path(__file__).parent / 'software-evidence.json').write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'help_returncode':r.returncode,'parser_contract':actions,'no_training_invoked':True},ensure_ascii=False))
