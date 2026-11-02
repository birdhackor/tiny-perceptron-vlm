"""Fetch W.1 primary documentation; full responses remain outside ordinary Git."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-P5-W.1-primary')
DEST = Path(__file__).resolve().parent
URLS = {
    'uv-python': 'https://docs.astral.sh/uv/guides/install-python/',
    'uv-sync': 'https://docs.astral.sh/uv/reference/cli/#uv-sync',
    'git-clone': 'https://git-scm.com/docs/git-clone',
    'git-lfs-config': 'https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.1/docs/man/git-lfs-config.adoc',
    'git-lfs-install': 'https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.1/docs/man/git-lfs-install.adoc',
    'git-longpaths': 'https://raw.githubusercontent.com/wiki/git-for-windows/git/Git-cannot-create-a-file-or-directory-with-a-long-path.md',
    'python-venv': 'https://docs.python.org/3.13/library/venv.html',
    'ipykernel-install': 'https://ipython.readthedocs.io/en/stable/install/kernel_install.html',
    'jupyter-notebook': 'https://jupyterlab.readthedocs.io/en/stable/user/notebook.html',
    'powershell-policy': 'https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.5',
    'colab-faq': 'https://research.google.com/colaboratory/faq.html',
    'python-indentation': 'https://docs.python.org/3.13/tutorial/controlflow.html',
    'public-lesson': 'https://birdhackor.github.io/tiny-perceptron-vlm/1.1.html',
}

class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignore = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.ignore += 1
    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.ignore -= 1
    def handle_data(self, value):
        if not self.ignore and value.strip():
            self.parts.append(value.strip())

def fetch(item):
    name, url = item
    try:
        with urlopen(Request(url, headers={'User-Agent':'W1-independent-technical-review/1'}), timeout=40) as response:
            raw = response.read()
            headers = dict(response.headers)
            effective = response.url
        suffix = '.html' if b'<html' in raw[:2000].lower() else '.txt'
        path = CACHE / (name + suffix)
        path.write_bytes(raw)
        parser = Plain()
        parser.feed(raw.decode('utf-8', errors='replace'))
        plain = '\n'.join(parser.parts) if suffix == '.html' else raw.decode('utf-8', errors='replace')
        (CACHE / (name + '.readable.txt')).write_text(plain)
        return {'id':name,'url':url,'effective_url':effective,'accessed_at':datetime.now(timezone.utc).isoformat(),'bytes':len(raw),'sha256':sha256(raw).hexdigest(),'outside_git_path':str(path),'etag':headers.get('ETag'),'last_modified':headers.get('Last-Modified')}
    except Exception as exc:
        return {'id':name,'url':url,'error':type(exc).__name__ + ': ' + str(exc)}

if __name__ == '__main__':
    CACHE.mkdir(parents=True,exist_ok=True)
    rows = list(ThreadPoolExecutor(max_workers=6).map(fetch, URLS.items()))
    record = {'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_w_1','command':'python docs/technical-reviews/artifacts/omni-P5-W.1/acquire_primary.py','policy':'Full third-party documentation is outside Git; this receipt stores provenance only.','sources':rows}
    (DEST / 'primary-acquisition-v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(rows,ensure_ascii=False,indent=2))
