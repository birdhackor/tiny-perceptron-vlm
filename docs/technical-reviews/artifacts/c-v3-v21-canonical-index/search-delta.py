"""Bound the retained 1.1 search-probe data change; no search/runtime rerun."""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/tmp/tpb').resolve()
OLD = Path('/workspace/work/omni-book-integrated-v1/outputs/omni-P5-preview-v9/search.json')
CURRENT = ROOT/'outputs/c-v3-author-preview-current/search.json'
prior = json.loads(OLD.read_bytes())
current = json.loads(CURRENT.read_bytes())
old = next(item for item in prior['items'] if item['location'] == '1.1.html')
new = next(item for item in current['items'] if item['location'] == '1.1.html')
assert old.keys() == new.keys()
assert all(old[key] == new[key] for key in old if key != 'text')
prefix = re.match(r'^<p><small>閱讀約 [^<]+ · AI 估計</small></p> ', new['text'])
assert prefix and new['text'][prefix.end():] == old['text']
assert 'vocabulary' in old['text'] and 'vocabulary' not in old['title']
result = {
    'reviewer_task': '/root/c_v3_technical_coord/canonical_index',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'environment': {'python': sys.version.split()[0], 'device': 'CPU; read-only JSON/string comparison'},
    'old_search': {'path': str(OLD), 'sha256': hashlib.sha256(OLD.read_bytes()).hexdigest()},
    'current_search': {'path': str(CURRENT), 'sha256': hashlib.sha256(CURRENT.read_bytes()).hexdigest()},
    'probe_location': '1.1.html',
    'all_nontext_fields_identical': True,
    'current_text_equals_prefix_plus_entire_old_text': True,
    'only_added_prefix': prefix.group(),
    'old_text_sha256': hashlib.sha256(old['text'].encode()).hexdigest(),
    'current_text_after_excluding_prefix_sha256': hashlib.sha256(new['text'][prefix.end():].encode()).hexdigest(),
    'unchanged_title': old['title'],
    'unchanged_body_only_query': 'vocabulary',
    'scope': 'Only source/data identity for retaining original title and body-search observations. The added reading-time prefix and its numerical value are excluded from this review; no reading-times.json dependency or fresh query runtime claim.',
}
path = ROOT/'docs/technical-reviews/artifacts/c-v3-v21-canonical-index/search-delta-v1.json'
with path.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bounded_difference': True}))
