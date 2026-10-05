import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def sha(raw):
    return hashlib.sha256(raw).hexdigest()

original = Path('/tmp/phase4-6_8-independent-original')
(OUT / 'original-run').mkdir(exist_ok=True)
for path in original.iterdir():
    if path.is_file():
        shutil.copyfile(path, OUT / 'original-run' / path.name)
current_files = [
    'course/chapters/06.md', 'course/chapters/01.md', 'course/training.md',
    'docs/review-tools/factual-reviewer-instructions.md',
    'scripts/check_technical_reviews.py', 'docs/review-tools/section_facts.py',
    '.agents/skills/clear-tutorial/references/review-protocol.md',
    'tiny_perceptron/data.py', 'tiny_perceptron/model.py',
    'scripts/course_experiments/common.py', 'scripts/course_experiments/text.py',
    'scripts/course_experiments/run.py', 'scripts/build_course.py', 'pyproject.toml',
    'docs/course-experiments/results/tokenizer.json',
]
provenance = []
for name in current_files:
    target = OUT / 'inputs/current' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / name).read_bytes())
    provenance.append({'origin':name, 'snapshot':target.relative_to(OUT).as_posix(), 'sha256':sha(target.read_bytes()), 'method':'copied original bytes'})
result = json.loads((ROOT / 'docs/course-experiments/results/tokenizer.json').read_bytes())
revision = result['revision']
for name in ['tiny_perceptron/data.py', 'tiny_perceptron/model.py', 'scripts/course_experiments/common.py', 'scripts/course_experiments/text.py']:
    raw = subprocess.run(['git','show',f'{revision}:{name}'],cwd=ROOT,capture_output=True,check=True).stdout
    assert sha(raw) == result['code_sha256'][name], (name, sha(raw), result['code_sha256'][name])
    target = OUT / 'inputs/historical' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    provenance.append({'origin':f'git:{revision}:{name}', 'snapshot':target.relative_to(OUT).as_posix(), 'sha256':sha(raw), 'method':'git show historical revision; hash matched original result'})
source = ROOT / 'outputs/text-behavior-interface-check/tokenizer'
allowed = ['tokenizer-bpe512.json','tokenizer-byte.json','data/manifest.json','data/train.jsonl','data/validation.jsonl','data/test.jsonl']
recorded_artifacts = {a['path']:a for a in result['artifacts']}
for name in allowed:
    raw = (source/name).read_bytes()
    assert sha(raw) == recorded_artifacts[name]['sha256'],name
    target = OUT / 'inputs/original-experiment' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    provenance.append({'origin':str((source/name).relative_to(ROOT)), 'snapshot':target.relative_to(OUT).as_posix(), 'sha256':sha(raw), 'method':'existing input bytes; matched original result artifact SHA-256; no dataset preparation or tokenizer retraining'})
(OUT/'input-provenance.json').write_text(json.dumps({'repo_head':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),'original_experiment_revision':revision,'files':provenance},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'copied_files':len(provenance),'historical_hash_matches':4,'original_input_hash_matches':6,'original_run_exit_code':json.loads((OUT/'original-run/execution.json').read_text())['exit_code']},indent=2))
