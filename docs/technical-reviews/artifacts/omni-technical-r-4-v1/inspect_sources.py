"""保存實讀原始來源的位置、短摘錄與指紋；第三方全文只留 Git 外。"""

import hashlib
import inspect
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-r-4-v1')
OUTPUT = Path(__file__).with_name('source-inspection.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


official = []
for entry in json.loads((CACHE / 'official-index.json').read_bytes()):
    path = CACHE / entry['filename']
    raw = path.read_bytes()
    lines = raw.decode().splitlines()
    row = dict(entry, cache_path=str(path), sha256=sha(raw), accessed_on=datetime.now(UTC).date().isoformat())
    # 每個來源只保存兩行短摘錄，其他實讀定位記 span。
    a, b = entry['spans'][0]
    row['short_excerpt'] = '\n'.join(lines[a - 1:min(a + 1, b)])
    official.append(row)

installed = []
for obj, filename in [(torch.nn.Parameter, 'torch214-parameter.py'), (torch.nn.Linear, 'torch214-linear.py')]:
    path = Path(inspect.getfile(obj))
    raw = path.read_bytes()
    same = raw == (CACHE / filename).read_bytes()
    assert same, '實裝版本與核對的官方來源不同'
    installed.append({'class': obj.__name__, 'installed_path': str(path), 'sha256': sha(raw),
                      'official_file': filename, 'byte_equal_to_official_v2_14_1': same})

historical = []
for revision, path, span in [
    ('49ce5b40b7931bc18f54174827931a5d5a623017', 'tiny_perceptron/lesson_products/train.py', [66, 129]),
    ('49ce5b40b7931bc18f54174827931a5d5a623017', 'tiny_perceptron/lesson_products/model.py', [120, 171]),
    ('07a6bc2e2dd1b0d7ee3634a0829e2fea5a86053a', 'scripts/selftrained/train.py', [393, 438]),
]:
    raw = subprocess.run(['git', 'show', revision + ':' + path], capture_output=True, check=True).stdout
    if 'lesson_products' in path:
        receipt = json.loads((ROOT / 'docs/lesson-products/results/initial-b-execution.json').read_bytes())
        assert sha(raw) == receipt['source_hashes'][path]
    historical.append({'path': path, 'version': revision, 'sha256': sha(raw), 'inspected_lines': span,
                       'url': 'https://github.com/birdhackor/tiny-perceptron-vlm/blob/' + revision + '/' + path,
                       'inspection': '本人從 Git 原物件核對初始化分支／模組；不是歷史 reviewer 摘要。'})

linked_scope = []
text = (ROOT / 'course/chapters/19.md').read_text()
patterns = {
    '19.13': '三個神經模型都用本專案訓練的權重',
    '19.21': '這是助理產品 A 的',
    '19.22': '這是公開約定的有限提示音',
    '19.23': '以下是完整的小型CPU參考實作',
    '19.24': '新版整合與保留能力的實測仍待完成',
}
for identifier, needle in patterns.items():
    match = re.search(r'^## ' + re.escape(identifier) + r'\s.*?(?=^## |\Z)', text, re.M | re.S)
    assert needle in match.group()
    paragraph = next(p for p in match.group().split('\n\n') if needle in p)
    linked_scope.append({'path': 'course/chapters/19.md', 'selector': 'section:' + identifier,
                         'section_sha256': sha(match.group().encode()), 'short_excerpt': paragraph[:450]})

data = {
    'created_at': datetime.now(UTC).isoformat(),
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-4-v1/inspect_sources.py',
    'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu',
                    'torch_git_version': torch.version.git_version},
    'official_sources': official, 'installed_official_source_comparison': installed,
    'historical_original_code': historical, 'linked_scope_inspection': linked_scope,
    'scope': '來源版本、原文定位與所宣告範圍核對。沒有執行相鄰課節的 CPU 模型、看圖、重現外部模型品質或進行新訓練。'
}
with OUTPUT.open('x', encoding='utf-8') as handle:
    json.dump(data, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps({'output': str(OUTPUT.relative_to(ROOT)), 'sha256': sha(OUTPUT.read_bytes()),
                  'torch': torch.__version__, 'official_source_count': len(official),
                  'installed_files_byte_equal': True, 'historical_code_count': len(historical)}, ensure_ascii=False))
