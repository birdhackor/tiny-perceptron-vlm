"""Read-only anonymous transfer check; stream without storing weights."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

root = Path('/workspace/work/omni-book-integrated-v1')
release = json.loads((root/'docs/decision-product/v3/public-release.json').read_text())
result = {'created_at':datetime.now(UTC).isoformat(), 'url':release['url'], 'authentication':'none; User-Agent only', 'expected_bytes':release['bytes'], 'expected_sha256':release['sha256']}
try:
    with urlopen(Request(release['url'], headers={'User-Agent':'R4-independent-technical-review'}), timeout=30) as response:
        sha, count = hashlib.sha256(), 0
        while block := response.read(1024 * 1024):
            sha.update(block)
            count += len(block)
            if count > release['bytes']:
                break
        result.update(http_status=response.status, observed_bytes=count, observed_sha256=sha.hexdigest())
        result['verified'] = count == release['bytes'] and sha.hexdigest() == release['sha256']
except (HTTPError, URLError, OSError) as error:
    result.update(verified=False, error_type=type(error).__name__, error=str(error), http_status=getattr(error,'code',None))
(Path(__file__).parent/'public-transport.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
