import re
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

C = Path('docs/course-repair-20261008/reviews/callback-02')
M = json.loads((C / 'manifest.json').read_text())
P = {p['page_id']: p for p in M['pages']}
phase = sys.argv[1]
for pid in sys.argv[2:]:
    p = P[pid]
    raw = Path(p['snapshot']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == p['source_sha256']
    source = raw.decode()
    blocks = list(re.finditer(r'<details\b[^>]*>.*?</details>', source, re.S))
    if phase == 'main':
        out = re.sub(r'<details\b[^>]*>.*?</details>', '[選讀保留原位置，主文理解保存後另讀]', source, flags=re.S)
    elif phase == 'extras':
        assert (C / 'work/continuity-all-callback/main-understanding.json').exists()
        out = '\n\n'.join('選讀 %d\n%s' % (i + 1, b.group()) for i, b in enumerate(blocks))
    else:
        raise ValueError(phase)
    destination = C / 'work/continuity-all-callback' / (pid + '-' + phase + '.txt')
    destination.write_text(out)
    print(json.dumps({'page_id': pid, 'phase': phase, 'source_sha256': p['source_sha256'],
                      'details_count': len(blocks), 'lines': len(out.splitlines()),
                      'delivery': str(destination), 'meaning': 'delivery only; reviewer must actually read'}, ensure_ascii=False))
    with (C / 'work/continuity-all-callback/source-delivery.jsonl').open('a') as f:
        f.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(), 'page_id': pid,
                            'phase': phase, 'source_sha256': p['source_sha256']}, ensure_ascii=False) + '\n')
