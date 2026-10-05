# Exact code previously executed through .venv/bin/python - <<'PY'.
import urllib.request,hashlib,json
from pathlib import Path
A=Path('docs/technical-reviews/artifacts/phase4-5_9-independent')
u='https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/autograd.md'
with urllib.request.urlopen(u,timeout=20) as f: b=f.read()
(A/'sources/torch-autograd.md').write_bytes(b)
r={'url':u,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'accessed_on':'2026-10-05','reason':'The pinned revision moved autograd notes from .rst to .md; original .rst fetch was 404.'}
(A/'autograd-fetch-receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
