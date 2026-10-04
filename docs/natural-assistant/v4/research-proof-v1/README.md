This optional proof snapshot preserves the legally selected v4 research and review evidence. The base archive contains 1,321 raw files (297,485,318 bytes before compression); its compressed size is 225,820,572 bytes. [receipt.json](receipt.json) records the exact archive and index SHA-256, and [ATTRIBUTION.json](ATTRIBUTION.json) preserves individual source identities, artists, licenses and source URLs. The base archive was rebuilt twice with identical compressed SHA-256 and all raw/generated members were verified locally.

Ordinary Git and CI can read the small receipt, [proof-index.json](proof-index.json), [NOTICE.txt](NOTICE.txt), attribution and release notes. The raw paths in the index are archive member names. They require obtaining the external archive, verifying its recorded SHA-256 and extracting it; they are not promised to exist in an ordinary checkout. Review registry records should cite these small artifacts and an eventual verified publication receipt, rather than link absent raw paths as checkout files.

The local base archive is `outputs/natural-v4/research-proof-v1/natural-v4-research-proof-v1.tar.gz`, SHA-256 `4cb8e36795fba059b1e56883b4c95915538db37badb2bea2dd05c71f74c66f41`. Publication may use another storage filename only if a separate publication receipt binds it to these exact bytes. Research proof is optional and is outside the three training archives and the data downloader's manifest.

[Supplement 001](supplement-001/receipt.json) adds the later root-authored successful public OCR replay and independent final local manifest/archive verification. It contains 29 raw files, compresses to 465,191 bytes and was also rebuilt twice with identical SHA-256. Its archive SHA-256 is `73101e232f5db084d73fd7c94b3909a857c8d4757ff986d54ec00ccb1ab2008b`. Base v1 bytes remain unchanged. Later training, scoring or review evidence belongs in additional immutable snapshots.

The builder performs local packaging and verification. The base receipt's `draft_release_created` and `external_archive_uploaded` flags describe that builder operation. They do not describe later coordinator actions. A separate publication receipt must record any real release, transfer and public retrieval result; the local receipt is not evidence of upload success.

Historical unselected OCR research/samples, NC/SA previews, unselected corpus/model cards and full texts without confirmed redistribution are excluded. The selected photographs/text retain their declared CC BY, CC0 or Apache licenses. Course-authored code/reports retain MIT, and archived Transformers source retains its original Apache copyright/license headers.

CI can verify the small index without accessing raw evidence:

```python
import hashlib
import json
from pathlib import Path

base = Path("docs/natural-assistant/v4/research-proof-v1")
receipt = json.loads((base / "receipt.json").read_bytes())
index_bytes = (base / "proof-index.json").read_bytes()
assert hashlib.sha256(index_bytes).hexdigest() == receipt["index_sha256"]
index = json.loads(index_bytes)
assert len(index["files"]) == receipt["raw_files"]
assert sum(row["bytes"] for row in index["files"]) == receipt["raw_bytes"]
```
