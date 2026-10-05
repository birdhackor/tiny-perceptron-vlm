"""Fetch original official API/source snapshots for this section's review."""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json
from datetime import datetime, UTC

DEST = Path(__file__).resolve().parent
records = []
targets = [
    ('detach-2.9.html', 'https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.detach.html', 'PyTorch 2.9 API docs'),
    ('log-2.9.html', 'https://docs.pytorch.org/docs/2.9/generated/torch.log.html', 'PyTorch 2.9 API docs'),
    ('exp-2.9.html', 'https://docs.pytorch.org/docs/2.9/generated/torch.exp.html', 'PyTorch 2.9 API docs'),
]
for filename, url, version in targets:
    with urlopen(Request(url, headers={'User-Agent':'independent-factual-review/1.0'}), timeout=30) as response:
        raw = response.read()
        resolved = response.url
    (DEST / filename).write_bytes(raw)
    records.append(dict(path=filename,url=url,resolved_url=resolved,version=version,
                        accessed_on=datetime.now(UTC).date().isoformat(),bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest()))
    print(filename, len(raw), hashlib.sha256(raw).hexdigest(), resolved)

api = 'https://api.github.com/repos/openai/spinningup/commits?path=spinup/algos/pytorch/ppo/ppo.py&per_page=1'
with urlopen(Request(api,headers={'User-Agent':'independent-factual-review/1.0'}),timeout=30) as response:
    metadata=json.load(response)
commit=metadata[0]['sha']
url=f'https://raw.githubusercontent.com/openai/spinningup/{commit}/spinup/algos/pytorch/ppo/ppo.py'
with urlopen(Request(url,headers={'User-Agent':'independent-factual-review/1.0'}),timeout=30) as response:
    raw=response.read()
(DEST/'openai-spinningup-ppo.py').write_bytes(raw)
records.append(dict(path='openai-spinningup-ppo.py',url=url,version=commit,
                    accessed_on=datetime.now(UTC).date().isoformat(),bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest()))
print('openai-spinningup-ppo.py',commit,len(raw))
(DEST/'official-fetch-provenance.json').write_text(json.dumps(records,indent=2)+'\n')
