import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attributes):
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden -= 1
        if tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'li'):
            self.parts.append(' ')

    def handle_data(self, value):
        if not self.hidden:
            self.parts.append(value)


def main():
    sources = [
        {
            'url': 'https://scikit-learn.org/1.7/modules/classification_threshold.html',
            'version': 'scikit-learn 1.7.2 documentation',
            'locator': 'Tuning the decision threshold for class prediction; Options to tune the decision threshold',
            'quotes': ['the decision problem to take concrete action based on those probability predictions.',
                       'one should choose a meaningful metric for their use case.'],
        },
        {
            'url': 'https://scikit-learn.org/1.7/modules/cross_validation.html',
            'version': 'scikit-learn 1.7.2 documentation',
            'locator': 'Cross-validation: evaluating estimator performance; Cross-validation iterators for i.i.d. data',
            'quotes': ['A test set should still be held out for final evaluation',
                       'Assuming that some data is Independent and Identically Distributed (i.i.d.)'],
        },
        {
            'url': 'https://arxiv.org/html/2207.05221v4',
            'version': 'arXiv:2207.05221v4, 2022-11-21',
            'locator': 'Predicting P(IK); Discriminating What Models Know with Entropy or Loss; Loss of the Sample',
            'quotes': ['independent of the logits for language modeling', 'without achieving calibration'],
        },
        {
            'url': 'https://arxiv.org/html/2302.04761v1',
            'version': 'arXiv:2302.04761v1, 2023-02-09',
            'locator': 'Approach / Inference; Limitations',
            'quotes': ['tool-dependent, computational cost incurred from making an API call.'],
        },
    ]
    for source in sources:
        command = ['curl', '--fail', '--location', '--silent', '--show-error', source['url']]
        result = subprocess.run(command, capture_output=True, check=True)
        parser = TextParser()
        parser.feed(result.stdout.decode())
        text = re.sub(r'\s+', ' ', ''.join(parser.parts))
        for quote in source['quotes']:
            assert quote in text, source['url']
        assert sum(len(quote.split()) for quote in source['quotes']) <= 25
        source.update(command=command, exit_code=result.returncode,
                      accessed_on=datetime.now(timezone.utc).date().isoformat(),
                      downloaded_bytes=len(result.stdout),
                      downloaded_sha256=hashlib.sha256(result.stdout).hexdigest(),
                      checked_original=True, stderr=result.stderr.decode())
    print(json.dumps({
        'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-B-8/collect_sources.py',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'cpu'},
        'result': '各原始 HTTPS 頁面讀取成功，短摘錄逐字比對通過；第三方全文僅在記憶體處理。',
        'sources': sources,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
