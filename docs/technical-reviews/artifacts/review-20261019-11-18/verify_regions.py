"""核對凍結字卡的裁切、字序、反例及本機字表。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path
from xml.etree import ElementTree

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.natural_concepts import text_error_report
from tiny_perceptron.selftrained.tokenizer import OCR_CHARACTERS, CharacterTokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance_matrix(reference, prediction):
    """以完整距離表獨立核對專案的兩列實作。"""
    table = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for row in range(len(reference) + 1):
        table[row][0] = row
    for col in range(len(prediction) + 1):
        table[0][col] = col
    for row, expected in enumerate(reference, 1):
        for col, actual in enumerate(prediction, 1):
            table[row][col] = min(
                table[row - 1][col] + 1,
                table[row][col - 1] + 1,
                table[row - 1][col - 1] + int(expected != actual),
            )
    return table


source = ROOT / 'docs/course-review-20261019/freeze-01/sources/11.18.md'
frozen = source.read_text()
code = re.search(r'```python\n(.*?)```', frozen, re.S).group(1)
namespace = {}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(code, str(source), 'exec'), namespace)
glyph = namespace['glyph']
card = namespace['card']
swapped = namespace['swapped']

predecessor = (ROOT / 'course/chapters/11.md').read_text()
predecessor = re.search(r'^## 11\.12 .*?(?=^## |\Z)', predecessor, re.M | re.S).group()
predecessor_code = re.search(r'```python\n(.*?)```', predecessor, re.S).group(1)
glyph_assignment = next(
    node for node in ast.parse(predecessor_code).body
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'glyphs' for t in node.targets)
)
previous_glyphs = ast.literal_eval(glyph_assignment.value)
glyph_agreement = {
    label: torch.equal(pixels, torch.tensor([[int(c) for c in row] for row in previous_glyphs[label]]))
    for label, pixels in glyph.items()
}

cases = [('原卡', card, '右框'), ('原卡', card, '左框'), ('互換後', swapped, '右框')]
crop_records = {}
for setting, boxes, expected_matches in (
    ('原框', {'左框': (0, 3), '右框': (4, 7)}, [['1'], ['0'], ['0']]),
    ('右框左移', {'左框': (0, 3), '右框': (3, 6)}, [[], ['0'], []]),
):
    records = []
    for (title, image, name), expected in zip(cases, expected_matches):
        start, end = boxes[name]
        crop = image[:, start:end]
        matched = [label for label, pixels in glyph.items() if torch.equal(crop, pixels)]
        assert matched == expected, '裁切結果不符預測'
        records.append({'card': title, 'box': name, 'columns': list(range(start, end)), 'pixels': crop.tolist(), 'matched': matched})
    crop_records[setting] = records

cer_records = []
for prediction in ('入口出口', '出口', '口出'):
    table = distance_matrix('出口', prediction)
    report = text_error_report('出口', prediction)
    assert table[-1][-1] == report['edits'], '編輯距離實作結果不一致'
    cer_records.append({'prediction': prediction, 'report': report, 'independent_distance_matrix': table})
assert [item['report']['edits'] for item in cer_records] == [2, 0, 2], '字串算例不符預測'

word_cases = [
    {'name': '原卡右框', 'left': '入口', 'right': '出口', 'box': 'right'},
    {'name': '原卡左框', 'left': '入口', 'right': '出口', 'box': 'left'},
    {'name': '互換後右框', 'left': '出口', 'right': '入口', 'box': 'right'},
]
policies = {
    '固定回答出口': lambda item: '出口',
    '固定讀右區': lambda item: item['right'],
    '依位置背答案': lambda item: {'left': '入口', 'right': '出口'}[item['box']],
    '按框讀內容': lambda item: item[item['box']],
}
truth_table = {
    name: [{'case': item['name'], 'prediction': policy(item), 'expected': item[item['box']], 'correct': policy(item) == item[item['box']]} for item in word_cases]
    for name, policy in policies.items()
}
assert [[row['correct'] for row in rows] for rows in truth_table.values()] == [
    [True, False, False], [True, False, True], [True, True, False], [True, True, True]
], '捷徑對照不符預測'

tokenizer = CharacterTokenizer.build([])
membership = {text: {'missing': sorted(set(text) - set(OCR_CHARACTERS)), 'ids': tokenizer.encode(text), 'unknown': tokenizer.unk_id in tokenizer.encode(text)} for text in ('入口', '出口', '開關')}
assert all(not item['missing'] and not item['unknown'] for item in membership.values()), '字卡含未知字元'

svg = ROOT / 'course/figures/new-11.18-two-regions.svg'
tree = ElementTree.fromstring(svg.read_bytes())
svg_text = [node.text for node in tree.iter() if node.tag.endswith('text')]
font_rules = [node.text for node in tree.iter() if node.tag.endswith('style')]
font_embedded = any(marker in svg.read_text().lower() for marker in ('@font-face', 'data:font', '<font ', '.otf', '.ttf', '.woff'))
assert not font_embedded, '字卡包含字型檔'

result = {
    'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-18/verify_regions.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_commit': str(torch.version.git_version), 'device': str(card.device), 'platform': platform.platform()},
    'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (source, svg, ROOT/'course/chapters/11.md', ROOT/'tiny_perceptron/natural_concepts.py', ROOT/'tiny_perceptron/selftrained/tokenizer.py')},
    'script_sha256': digest(Path(__file__)),
    'original_code_stdout': printed.getvalue(),
    'dtype': str(card.dtype),
    'card_shape': list(card.shape),
    'glyph_agreement': glyph_agreement,
    'crops': crop_records,
    'cer': cer_records,
    'shortcut_truth_table': truth_table,
    'vocabulary': OCR_CHARACTERS,
    'vocabulary_membership': membership,
    'svg': {'texts': svg_text, 'font_rules': font_rules, 'embeds_font_file': font_embedded},
    'result': '所有確定性算例與本機契約核對成功；沒有載入或訓練模型。'
}
print(json.dumps(result, ensure_ascii=False, indent=2))
