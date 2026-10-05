import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-16_8-independent-fresh'
digest = lambda raw: hashlib.sha256(raw).hexdigest()
records = []

def snapshot(origin, raw, name, **extra):
    target = OUT / 'inputs' / name
    target.write_bytes(raw)
    assert digest(target.read_bytes()) == digest(raw)
    records.append(dict(origin=origin, snapshot=str(target.relative_to(ROOT)), bytes=len(raw), sha256=digest(raw), **extra))

for name in ['section.md', 'fence-1.py', 'extraction.json', 'bootstrap.py']:
    p = Path('/tmp/phase4-16_8-factual-original') / name
    snapshot(str(p), p.read_bytes(), name)
for name in ['course/chapters/16.md', 'course/figures/rewrite-16-causal-mask.svg', 'docs/review-tools/factual-reviewer-instructions.md', 'scripts/check_technical_reviews.py', 'docs/review-tools/section_facts.py', '.agents/skills/clear-tutorial/SKILL.md', '.agents/skills/clear-tutorial/references/review-protocol.md']:
    snapshot(name, (ROOT/name).read_bytes(), name.replace('/', '__'))

for name in ['efficiency', 'flash_probe']:
    p = ROOT / f'docs/course-experiments/results/{name}.json'
    raw = p.read_bytes()
    snapshot(str(p.relative_to(ROOT)), raw, f'{name}.json')
    result = json.loads(raw)
    for source in ['scripts/course_experiments/architecture.py', 'tiny_perceptron/attention.py', 'tiny_perceptron/model.py', 'tiny_perceptron/training.py', 'scripts/course_experiments/common.py']:
        command = ['git', 'show', result['revision']+':'+source]
        original = subprocess.run(command, cwd=ROOT, capture_output=True, check=True).stdout
        expected = result['code_sha256'][source]
        assert digest(original) == expected, (name, source, digest(original), expected)
        snapshot(source, original, name+'__'+source.replace('/','__'), revision=result['revision'], command=command, declared_code_sha256=expected)
        tree = ast.parse(original)
        records.append(dict(kind='ast_index', result=name, source=source, nodes=[dict(name=n.name, first_line=n.lineno,last_line=n.end_lineno) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]))

(OUT/'inputs'/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'snapshots_verified':sum('snapshot' in x for x in records),'original_revision_code_files_verified':sum('revision' in x for x in records),'manifest':'inputs/manifest.json'},indent=2))
