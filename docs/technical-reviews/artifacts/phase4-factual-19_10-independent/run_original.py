"""Run this review's unchanged section fence in the existing CPU environment."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
EXTRACT = Path('/tmp/phase4-factual-19_10-extract')
shutil.copyfile(EXTRACT / 'environment.json', OUT / 'helper-first-attempt.environment.json')
shutil.copyfile(OUT / 'original-fence.stderr.txt', OUT / 'helper-first-attempt.stderr.txt')
(EXTRACT / 'tmp').mkdir(exist_ok=True)
env = {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TZ', 'LD_LIBRARY_PATH') if key in os.environ}
env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR=str(EXTRACT / 'tmp'))
argv = [str(ROOT / '.venv/bin/python'), '-I', str(ROOT / 'docs/review-tools/section_facts.py'), '--worker', str(EXTRACT)]
with (OUT / 'original-fence.stdout.txt').open('wb') as stdout, (OUT / 'original-fence.stderr.txt').open('wb') as stderr:
    completed = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=45, check=False)
shutil.copyfile(EXTRACT / 'environment.json', OUT / 'original-fence.environment.json')
record = {'argv': argv, 'cwd': str(ROOT), 'timeout_seconds': 45, 'exit_code': completed.returncode, 'source_sha256': hashlib.sha256((OUT / 'section.md').read_bytes()).hexdigest(), 'first_attempt': 'Guard rejected temporary-directory probes before any fence. Repeated with TMPDIR inside extraction directory and bytecode writes disabled; retained original failure.'}
(OUT / 'original-fence.execution.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(record, ensure_ascii=False))
print((OUT / 'original-fence.stdout.txt').read_text())
raise SystemExit(completed.returncode)
