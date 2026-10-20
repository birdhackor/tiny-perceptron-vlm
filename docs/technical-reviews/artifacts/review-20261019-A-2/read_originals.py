"""以 curl 讀取原始文件；只保存來源收據與支持主張的短摘錄。"""
import hashlib
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'section'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


specs = [
    {
        'id': 'rag', 'url': 'https://arxiv.org/html/2005.11401v4',
        'version': 'arXiv:2005.11401v4，2021-04-12',
        'locators': ['Methods', 'Generator: BART', 'Decoding', 'Index hot-swapping'],
        'excerpts': ['we simply concatenate them.', 'knowledge can be easily updated at test time.'],
        'inspection_note': '已閱讀 Methods 的查詢檢索與文件條件生成、Generator: BART 的串接輸入、Training 與 Decoding 的分別，以及 Index hot-swapping 的換索引實驗。支持推論時用外部文件的資訊，並不代表每次都能正確採用文件、也不表示本教材重現原論文架構或準確率。',
    },
    {
        'id': 'alce', 'url': 'https://arxiv.org/html/2305.14627v2',
        'version': 'arXiv:2305.14627v2，2023-10-31',
        'locators': ['Introduction', 'Citation Quality', 'Modeling'],
        'excerpts': ['citation recall, which determines if the output is entirely supported by cited passages'],
        'inspection_note': '已閱讀 Introduction 的引文用途、Citation Quality 的支持性與相關性定義，以及 Modeling 的檢索、合成和文件編號輸入。支持以來源編號回查，但不能由來源存在直接保證回答有支持；也支持附上文件仍可能產生無支持內容的限制。未將 ALCE 的測得比例套用到教材模型。',
    },
    {
        'id': 'grouped', 'url': 'https://scikit-learn.org/1.7/modules/cross_validation.html',
        'version': 'scikit-learn 1.7 版官方 User Guide',
        'locators': ['Cross-validation iterators for grouped data', 'Group K-fold'],
        'excerpts': ['GroupKFold is a variation of K-fold which ensures that the same group is not represented in both testing and training sets.'],
        'inspection_note': '已閱讀 grouped data 與 Group K-fold 正文中關於群組內樣本相依、以訓練未見群組測量泛化、以及避免個體特徵造成過度樂觀評估的解釋。教材店名家族是相同原則的自製切分，沒有宣稱使用 sklearn 或實測洩漏幅度。',
    },
]
receipts = []
for spec in specs:
    argv = ['curl', '-fLsS', '--max-time', '40', spec['url']]
    raw = subprocess.run(argv, capture_output=True, check=True).stdout
    parser = Text()
    parser.feed(raw.decode('utf-8'))
    text = re.sub(r'\s+', ' ', ''.join(parser.parts))
    for excerpt in spec['excerpts']:
        assert excerpt in text
    for locator in spec['locators']:
        assert locator in text
    assert sum(len(x.split()) for x in spec['excerpts']) <= 25
    receipts.append({**spec, 'command_argv': argv, 'http_fetch_exit_code': 0,
                     'accessed_at_utc': datetime.now(UTC).isoformat(),
                     'original_bytes': len(raw), 'original_sha256': hashlib.sha256(raw).hexdigest(),
                     'storage_scope': '原文在記憶體讀取，未保存第三方全文；此檔僅保留短摘錄與來源收據。'})
print(json.dumps({'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-2/read_originals.py',
                  'environment': {'python': platform.python_version(), 'curl': subprocess.run(['curl', '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]},
                  'status': 'all_originals_retrieved_and_excerpts_verified', 'sources': receipts}, ensure_ascii=False, indent=2))
