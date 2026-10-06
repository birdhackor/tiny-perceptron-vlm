import ast
import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
DEST = Path(__file__).resolve().parent
INITIAL = DEST.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
previous = json.loads((DEST / 'previous-report.json').read_bytes())
current_section = (DEST / 'current-19.2.md').read_bytes()
raw = (ROOT / 'course/chapters/19.md').read_bytes().decode()
start = re.search(r'^## 19\.2 ', raw, re.M).start()
end = re.search(r'^## 19\.3 ', raw, re.M).start()
assert raw[start:end].encode() == current_section
first = (INITIAL / '19.2.md').read_bytes().decode()
current = current_section.decode()
pattern = re.compile(r'```python\n(.*?)\n```', re.S)
first_code = pattern.search(first)[1]
current_code = pattern.search(current)[1]
assert ast.dump(ast.parse(first_code), include_attributes=False) == ast.dump(ast.parse(current_code), include_attributes=False)
assert pattern.sub('<<CODE>>', first) == pattern.sub('<<CODE>>', current)
assert current_code + '\n' == (DEST / 'current-fence.py').read_text()
assert not re.findall(r'!\[[^\]]*\]\([^)]*\)', current)
for source in previous['sources']:
    if source['kind'] == 'repository_code':
        assert sha(ROOT / source['path']) == source['sha256']
for artifact in previous['artifacts']:
    assert sha(ROOT / artifact['path']) == artifact['sha256']
for relative, snapshot in [('docs/selftrained/v2-manifest.json', 'v2-manifest.json'),
                           ('docs/selftrained/results/v2-final-public-results.json', 'v2-final-public-results.json'),
                           ('docs/selftrained/results/public-cpu-raw/source-core31-before.json', 'source-core31-before.json')]:
    assert (ROOT/relative).read_bytes() == (INITIAL/snapshot).read_bytes()
for stage in ['moe-native', 'dense-weighted', 'moe-pretrain', 'dense-pretrain']:
    for leaf in ['raw/train-receipt.json', 'raw/execution.json', 'receipt.json']:
        assert (ROOT/'docs/selftrained/results/training-raw'/stage/leaf).read_bytes() == (INITIAL/'raw-snapshots'/stage/leaf).read_bytes()
output = io.StringIO()
torch.set_num_threads(1)
with contextlib.redirect_stdout(output):
    exec(compile(current_code, 'current-fence.py', 'exec'), {})
assert output.getvalue() == (INITIAL / 'original-fence-output.txt').read_text()
(DEST/'current-fence-output.txt').write_text(output.getvalue())
per_ffn = 256*512 + 512 + 512*256 + 256
assert per_ffn == 262912
assert 933632 + 4*2*per_ffn == 3036928
assert 933632 + 4*4*per_ffn == 5140224
assert 929536 + 4*per_ffn == 1981184
assert 91907 + 126829 + 88147 == 306883
assert 539 + 11 == 550
print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'threads': torch.get_num_threads()}, ensure_ascii=False))
print('CURRENT_SOURCE_SHA256', sha(DEST/'current-19.2.md'))
print('AST_IDENTICAL True; OUTSIDE_FENCE_BYTES_IDENTICAL True; FIGURES 0')
print('UNCHANGED_REPOSITORY_CODE_SOURCES 6; UNCHANGED_PRIOR_ARTIFACTS 38; ORIGINAL_RAW_INPUT_SNAPSHOTS_MATCH True')
print('INDEPENDENT_FFN_COUNT', per_ffn)
print(output.getvalue(), end='')
print('CURRENT_FENCE_OUTPUT_EXACTLY_IDENTICAL True')
print('No training, optimizer updates, generation, heldout evaluation, download or upload.')
