"""Informed callback for the author's README count correction; preserves original evidence."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/tokenizer-20261027/readme'
REPORT = ROOT / 'docs/course-review-20261027-tokenizer/reviews/technical-readme/technical-original-01.json'
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name):
    return json.loads((ROOT / name).read_bytes())
f1 = load('docs/course-review-20261027-tokenizer/freeze-01.json')
f2 = load('docs/course-review-20261027-tokenizer/freeze-02.json')
p1 = next(x for x in f1['pages'] if x['page_id'] == 'readme')
p2 = next(x for x in f2['pages'] if x['page_id'] == 'readme')
old = (ROOT / p1['snapshot']).read_text()
new = (ROOT / p2['snapshot']).read_text()
assert new == (ROOT / 'README.md').read_text()
assert digest(ROOT / p2['snapshot']) == p2['source_sha256']
assert new == old.replace('共312份逐課Notebook', '共313份逐課Notebook').replace('合計338個編號小節', '合計339個編號小節')
index = load('course/lesson-index.json')
notebooks = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'notebooks').rglob('*.ipynb'))
assert notebooks == sorted(x['notebook'] for x in index)
extra = {name: re.findall(r'^## ([A-Z\d]+\.\d+) ', (ROOT / name).read_text(), re.M) for name in ['course/README.md', 'course/first-steps.md', 'course/training.md', 'course/glossary.md']}
assert len(index) == 313 and sum(map(len, extra.values())) == 26
original = json.loads(REPORT.read_bytes())
source_checks = []
for item in original['sources']:
    if item['kind'] == 'repository_code':
        observed = digest(ROOT / item['path'])
        source_checks.append({'path': item['path'], 'expected': item['sha256'], 'observed': observed, 'unchanged': observed == item['sha256']})
assert all(x['unchanged'] for x in source_checks)
artifact_checks = [{'path': x['path'], 'expected': x['sha256'], 'observed': digest(ROOT / x['path'])} for x in original['artifacts']]
assert all(x['expected'] == x['observed'] for x in artifact_checks)
english = (ROOT / 'README_en.md').read_text()
assert '[313 lessons]' in english and '313 lesson notebooks' in english and '339 numbered sections' in english
assert '[312 lessons]' not in english and '312 lesson notebooks' not in english and '338 numbered sections' not in english
result = {'basis_report': str(REPORT.relative_to(ROOT)), 'basis_sha256': digest(REPORT), 'source_sha256': p2['source_sha256'], 'freeze02_sha256': digest(ROOT / 'docs/course-review-20261027-tokenizer/freeze-02.json'), 'diff': ''.join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile='freeze-01 readme', tofile='freeze-02 readme')), 'count': {'notebooks': len(index), 'extra_ids': extra, 'extra': 26, 'total': len(index) + 26}, 'english_sha256': digest(ROOT / 'README_en.md'), 'english_number_lines': [x for x in english.splitlines() if '[313 lessons]' in x or '313 lesson notebooks' in x], 'original_repository_sources': source_checks, 'original_artifacts': artifact_checks, 'full_reread': 'Reviewer read the full current README in tool output and verified only these two count substrings differ from the independently reviewed original. All other necessary claims and source bytes remain unchanged.', 'scope': 'Narrow English number synchronization checked; not a separate full English-page technical review. No repeat installation, downloads, GPU or global environment changes.'}
(OUT / 'recheck-output-01.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'basis_sha256':result['basis_sha256'], 'source_sha256':result['source_sha256'], 'counts':result['count'], 'original_sources_unchanged':len(source_checks), 'original_artifacts_unchanged':len(artifact_checks)},ensure_ascii=False,indent=2))
