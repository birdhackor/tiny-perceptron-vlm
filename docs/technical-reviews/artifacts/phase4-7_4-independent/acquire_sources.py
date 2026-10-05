"""Fetch versioned official API pages; no model or dataset downloads."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, UTC
from urllib.request import urlopen, Request
import hashlib
import json
from html.parser import HTMLParser

OUT = Path(__file__).resolve().parent / 'sources'
URLS = {
    'nonzero': 'https://docs.pytorch.org/docs/2.9/generated/torch.nonzero.html',
    'item': 'https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.item.html',
    'cross_entropy': 'https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.cross_entropy.html',
}
class MainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag == 'article': self.active = True
    def handle_endtag(self, tag):
        if tag == 'article': self.active = False
        if self.active and tag in ('p','div','pre','li','h1','h2','h3','dt','dd'): self.parts.append('\n')
    def handle_data(self, data):
        if self.active: self.parts.append(data)

def get_source(item):
    name, url = item
    with urlopen(Request(url, headers={'User-Agent':'independent-course-factual-review/7.4'}), timeout=25) as response:
        raw = response.read()
        receipt = {'name':name, 'url':url, 'final_url':response.url, 'status':response.status,
                   'accessed_on':datetime.now(UTC).date().isoformat(),
                   'sha256':hashlib.sha256(raw).hexdigest(), 'version':'PyTorch 2.9 official documentation'}
    (OUT / ('pytorch-2.9-' + name + '.html')).write_bytes(raw)
    parser = MainText()
    parser.feed(raw.decode('utf-8'))
    text = ''.join(parser.parts)
    if not text.strip(): raise RuntimeError('No actual article text: '+name)
    (OUT / ('pytorch-2.9-' + name + '.txt')).write_text(text)
    return receipt

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(get_source, URLS.items()))
(OUT.parent / 'source-fetch-receipt.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(results, ensure_ascii=False, indent=2))
