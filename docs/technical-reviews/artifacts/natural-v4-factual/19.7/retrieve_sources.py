from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'outputs/natural-v4/factual-research/19.7'
PROOF = Path(__file__).resolve().parent
PIN = '1df335318bda03fd771807f66976953231d5a00b'
sources = {
    'function-calling': 'https://platform.openai.com/docs/guides/function-calling',
    'utf8': 'https://www.rfc-editor.org/rfc/rfc3629.txt',
    'input-validation': 'https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/Input_Validation_Cheat_Sheet.md',
    'dpo': 'https://arxiv.org/html/2305.18290v3',
}
repo_paths = [
    'tiny_perceptron/capstone.py', 'tiny_perceptron/data.py',
    'scripts/course_experiments/capstone.py',
    'docs/course-experiments/capstone-evidence/pretrain/validation.json',
    'docs/course-experiments/capstone-evidence/sft/validation.json',
    'docs/course-experiments/capstone-evidence/joint/validation.json',
    'docs/course-experiments/capstone-evidence/dpo/validation.json',
    'docs/course-experiments/capstone-evidence/deployment/test-joint.json',
    'docs/course-experiments/capstone-evidence/deployment/data.json',
    'docs/course-experiments/results/capstone_pretrain.json',
    'docs/course-experiments/results/capstone_sft.json',
    'docs/course-experiments/results/capstone_joint.json',
    'docs/course-experiments/results/capstone_preference.json',
]
for path in repo_paths:
    sources[path] = f'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{PIN}/{path}'

def fetch(pair):
    name, url = pair
    row = {'id': name, 'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Independent factual review'}), timeout=25) as response:
            body = response.read(8_000_001)
            if len(body) > 8_000_000:
                raise ValueError('Source exceeds small-source retrieval bound')
            row.update(status=response.status, final_url=response.url, bytes=len(body), sha256=sha256(body).hexdigest())
        dest = OUT / (name.replace('/', '__') + '.original')
        dest.write_bytes(body)
        row['research_path'] = str(dest.relative_to(ROOT))
        if name in repo_paths:
            row['local_sha256'] = sha256((ROOT / name).read_bytes()).hexdigest()
            row['matches_local_bytes'] = row['local_sha256'] == row['sha256']
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    return row

OUT.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    receipts = list(pool.map(fetch, sources.items()))
(PROOF / 'retrieval-receipts.json').write_text(json.dumps(receipts, indent=2, ensure_ascii=False) + '\n')
for row in receipts:
    print(json.dumps({k: v for k, v in row.items() if k != 'research_path'}, ensure_ascii=False))
