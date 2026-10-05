import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
command = [str(ROOT / '.venv/bin/python'), str(BASE / 'code/verify_cpu.py')]
env = os.environ.copy()
settings = {'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'PYTHONDONTWRITEBYTECODE': '1'}
env.update(settings)
started = time.perf_counter()
with (BASE / 'runs/cpu-stdout.txt').open('wb') as stdout, (BASE / 'runs/cpu-stderr.txt').open('wb') as stderr:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                            timeout=60, check=False)
record = {'command_argv': command, 'command': shlex.join(command), 'cwd': str(ROOT),
          'timeout_seconds': 60, 'exit_code': result.returncode, 'elapsed_seconds': time.perf_counter()-started,
          'environment': {'python': platform.python_version(), 'device': 'cpu', **settings},
          'code_sha256': hashlib.sha256((BASE/'code/verify_cpu.py').read_bytes()).hexdigest()}
(BASE/'runs/cpu-execution.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
print((BASE/'runs/cpu-stdout.txt').read_text())
print((BASE/'runs/cpu-stderr.txt').read_text(), file=sys.stderr)
raise SystemExit(result.returncode)
