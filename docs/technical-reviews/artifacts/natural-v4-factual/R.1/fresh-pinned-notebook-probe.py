"""Retrieve only the pinned notebook actually opened through the Colab action."""
from pathlib import Path
from urllib.request import urlopen
import datetime
import hashlib
import json

url='https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/25ca4bf5f8f8dc0a9e88aa6cd70caeb4c2f419e8/notebooks/01/1.1.ipynb'
with urlopen(url,timeout=30) as response:
    data=response.read()
    receipt={'url':url,'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'status':response.status,'final_url':response.url,'bytes':len(data),
             'sha256':hashlib.sha256(data).hexdigest(),
             'matches_current':data==Path('notebooks/01/1.1.ipynb').read_bytes()}
assert receipt['status']==200 and receipt['matches_current']
Path('outputs/natural-v4/factual-research/R.1/remote-pinned-1.1.ipynb').write_bytes(data)
Path('docs/technical-reviews/artifacts/natural-v4-factual/R.1/fresh-pinned-notebook-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
