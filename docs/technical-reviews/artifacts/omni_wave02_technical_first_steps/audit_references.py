"""Check W.7's implementation index against frozen chapter slices."""
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json'
freeze = json.loads(freeze_path.read_text())
page_by_id = {p['page_id']: p for p in freeze['pages']}
source = (ROOT / 'course/first-steps.md').read_text()
start = source.index('| `tiny_perceptron/data.py`')
rows = source[start:source.index('</details>', start)]
sections = {}
for chapter in (ROOT / 'course/chapters').glob('*.md'):
    for match in re.finditer(r'^## ([A-Z\d]+\.\d+) .*?(?=^## |\Z)', chapter.read_text(), re.M | re.S):
        sections[match[1]] = (chapter.relative_to(ROOT).as_posix(), match[0])
results = []
for row in rows.splitlines():
    file_match = re.search(r'`(tiny_perceptron/[^`]+)`', row)
    if not file_match:
        continue
    file = file_match[1]
    listed = []
    for match in re.finditer(r'\[([^]]+)\]\((chapters/[^)#]+)#([^)]*)\)(?:–([\d.]+))?', row):
        first, last = match[3], match[4]
        ids = [first] if not last else [
            first.rsplit('.', 1)[0] + '.' + str(i)
            for i in range(int(first.rsplit('.', 1)[1]), int(last.rsplit('.', 1)[1]) + 1)
        ]
        for identifier in ids:
            path, section = sections[identifier]
            digest = hashlib.sha256(section.encode()).hexdigest()
            assert digest == page_by_id[identifier]['source_sha256']
            mentions = [line for line in section.splitlines() if file in line]
            assert mentions, (file, identifier)
            listed.append({'lesson_id': identifier, 'source': path, 'section_sha256': digest,
                           'explicit_file_mentions': mentions,
                           'excerpt_markers': re.findall(r'^# (?:節錄自|精簡自) ([^\n]+)', section, re.M)})
    other = []
    for identifier, (path, section) in sections.items():
        if file not in section or identifier in {r['lesson_id'] for r in listed}:
            continue
        digest = hashlib.sha256(section.encode()).hexdigest()
        assert digest == page_by_id[identifier]['source_sha256']
        other.append({'lesson_id': identifier, 'source': path, 'section_sha256': digest,
                      'explicit_file_mentions': [line for line in section.splitlines() if file in line]})
    results.append({'file': file, 'exists': (ROOT / file).exists(),
                    'listed_sections': listed, 'other_literal_mentions': other})
model_path = 'tiny_perceptron/lesson_products/model.py'
historical = subprocess.run(['git', 'show', '2b3332ca4392a946a065782815f9f60dae9a87c7:' + model_path],
                            cwd=ROOT, capture_output=True, check=True).stdout
current = (ROOT / model_path).read_bytes()
print(json.dumps({
    'created_at': datetime.now(timezone.utc).isoformat(),
    'environment': {'python': platform.python_version(), 'device': 'cpu', 'platform': platform.system()},
    'freeze_sha256': hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
    'rows': results,
    'listed_row_count': len(results),
    'listed_section_references': sum(len(row['listed_sections']) for row in results),
    'historical_16_15_model_reference': {'path': model_path,
        'commit': '2b3332ca4392a946a065782815f9f60dae9a87c7',
        'historical_sha256': hashlib.sha256(historical).hexdigest(),
        'current_sha256': hashlib.sha256(current).hexdigest(),
        'same_bytes': historical == current},
    'reviewer_scope_judgments': {
        '16.15': 'The prose explains the bidirectional CandidateScorer/Block roles and cites this exact implementation; its historical pinned bytes equal the current file. This fits the stated inclusion criterion, but is absent from the index. Optional navigation/completeness issue.',
        '19.19': 'Excluded: the explicit path is release.json deployment metadata, not Python implementation or a module component explanation.',
        '19.24': 'Excluded: the final public-grammar citation supports backend input constraints; the displayed adapter is an original control-flow example, not a presented component from inference.py.'},
    'scope': 'Verifies only W.7 reference locations and file identity, not the scientific or empirical claims of the linked chapters.'
}, ensure_ascii=False, indent=2))
