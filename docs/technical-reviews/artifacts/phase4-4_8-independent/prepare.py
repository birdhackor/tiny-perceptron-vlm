from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('section_facts', ROOT / 'docs/review-tools/section_facts.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
temporary = Path('/tmp/phase4-4_8-original-cpu')
if temporary.exists():
    raise RuntimeError('Refuse to overwrite an earlier run')
command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'docs/review-tools/section_facts.py'), 'course/chapters/04.md#4.8', '--output', str(temporary), '--execute', '--timeout', '40']
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=50)
for name in ('section.md', 'extraction.json', 'bootstrap.py', 'fence-1.py', 'stdout.txt', 'stderr.txt', 'environment.json', 'execution.json'):
    target = OUT / 'original' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(temporary / name, target)
paths = ['docs/review-tools/factual-reviewer-instructions.md', 'scripts/check_technical_reviews.py', 'docs/review-tools/section_facts.py', '.agents/skills/clear-tutorial/references/review-protocol.md', 'scripts/build_course.py', 'tiny_perceptron/model.py', 'tiny_perceptron/modern.py', 'tiny_perceptron/attention.py', 'tiny_perceptron/data.py', 'tiny_perceptron/training.py', 'scripts/course_experiments/text.py', 'scripts/course_experiments/common.py', 'scripts/course_experiments/run.py', 'docs/course-experiments/results/text_foundation.json', 'docs/course-experiments/public-releases/text_foundation.json']
manifest = []
for name in paths:
    raw = (ROOT / name).read_bytes()
    target = OUT / 'inputs' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    manifest.append({'original_path': name, 'saved_path': str(target.relative_to(ROOT)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
body, whole, first_line = helper.original_section(ROOT / 'course/training.md', 'T.4')
(OUT / 'inputs/T.4-reference.md').write_bytes(body)
manifest.append({'original_path': 'course/training.md#T.4', 'saved_path': str((OUT / 'inputs/T.4-reference.md').relative_to(ROOT)), 'section_first_line': first_line, 'bytes':len(body), 'sha256':hashlib.sha256(body).hexdigest(), 'full_file_sha256': hashlib.sha256(whole).hexdigest(), 'scope':'Only the directly referenced fixed text_foundation recipe; no independent review of T.4.'})
original_result = json.loads((ROOT / 'docs/course-experiments/results/text_foundation.json').read_text())
historical = {}
for name in ['tiny_perceptron/model.py', 'tiny_perceptron/modern.py', 'tiny_perceptron/attention.py', 'tiny_perceptron/data.py', 'scripts/course_experiments/text.py', 'scripts/course_experiments/common.py']:
    raw = subprocess.check_output(['git','show',f"{original_result['revision']}:{name}"], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == original_result['code_sha256'][name]
    historical[name] = {'recorded_sha256': original_result['code_sha256'][name], 'current_sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}
    target = OUT / 'historical' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
(OUT / 'input-provenance.json').write_text(json.dumps({'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'inputs':manifest,'original_fence_command':command,'original_fence_return_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'historical_result_revision':original_result['revision'],'historical_code_binding':historical,'no_weight_copy':'No checkpoint is required for this structural review. No weights or datasets downloaded; no training run.'},ensure_ascii=False,indent=2)+'\n')
print(result.stdout, end='')
print(json.dumps({'original_return_code':result.returncode,'saved_inputs':len(manifest),'historical_bindings':historical},ensure_ascii=False,indent=2))
raise SystemExit(result.returncode)
