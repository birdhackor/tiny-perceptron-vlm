"""Exercise declared input/stop/format contracts using the actual frozen block."""
import contextlib
import datetime
import hashlib
import io
import json
import os
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/reference/omni-technical-19-22-v1/contract-edges')
CACHE.mkdir(parents=True, exist_ok=True)
snapshot = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix04/sources/19.22.md'
raw = snapshot.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '1695cecc155f301a70e6d1d46a4ec8e5fe52af39c4836dd11553df87bc9de0d4'
code = re.search(r'```python\n(.*?)\n```', raw.decode(), re.S)[1] + '\n'
ns = {'__name__': '__contract_review__'}
old = os.getcwd()
try:
    os.chdir(CACHE)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, str(snapshot), 'exec'), ns)
finally:
    os.chdir(old)

observations = {}
def must_reject(name, function):
    try:
        function()
    except ValueError as error:
        observations[name] = {'raised': 'ValueError', 'message': str(error)}
    else:
        raise AssertionError(name + ' did not reject')

must_reject('partial_codec_block_801_samples', lambda: ns['codec'].encode(torch.zeros(801)))
for name, value in [('non_dict', None), ('missing_semantic', {}), ('non_string', {'semantic': 1})]:
    must_reject('mapping_' + name, lambda value=value: ns['map_result'](value))
    row = ns['run_case']({'gold': 'square', 'a_result': value})
    assert row['chain'] == {'status': 'parse_error'}
for name, values in [('empty', []), ('nan', [float('nan')]), ('inf', [float('inf')]), ('amplitude', [1.1])]:
    must_reject('wav_' + name, lambda values=values: ns['save_wav'](str(CACHE/'invalid.wav'), torch.tensor(values)))
for steps, expected in [(0, ([], 'max_steps')), (1, ([2], 'max_steps')), (6, ([2], 'EOS'))]:
    observed = ns['generate'](0, max_steps=steps)
    assert observed == expected
    observations['generate_max_steps_' + str(steps)] = {'codes': observed[0], 'stop': observed[1]}
observations['generated_wave_is_full_pattern'] = False
observations['formal_omni_products_directory_exists_at_review'] = (ROOT/'tiny_perceptron/omni_products').exists()
result = {'created_at': datetime.datetime.now(datetime.UTC).isoformat(),
          'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-22-v1/verify_contract_edges.py',
          'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu'},
          'source_sha256': hashlib.sha256(raw).hexdigest(), 'embedded_code_sha256': hashlib.sha256(code.encode()).hexdigest(),
          'result': 'PASS: all declared malformed-input, WAV-format guards and EOS/max_steps branches behaved as read.',
          'scope': 'One further fresh execution of the exact embedded block, one seed and one CPU update; no quality or cross-platform performance claim.',
          'observations': observations}
out = HERE/'contract-edges-v1.json'
with out.open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps({'path': str(out.relative_to(ROOT)), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'result': result['result']}))
