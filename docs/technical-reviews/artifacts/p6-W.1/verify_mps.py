"""Test the device-selection branch; this does not emulate Apple GPU computation."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tomllib

out = Path(__file__).resolve().parent
root = out.parents[3]
spec = importlib.util.spec_from_file_location('w1_check_env', root / 'scripts/check_env.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)), device=lambda name: name)
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    selected = module.pick_device(fake)
assert selected == 'mps'
print('Bounded mocked MPS availability: no Apple hardware or GPU computation measured.')
print(stream.getvalue(), end='')
print('returned_device', selected)
lock = tomllib.loads((root / 'uv.lock').read_text())
for package in lock['package']:
    if package['name'] == 'torch' and package['source'].get('registry') == 'https://download.pytorch.org/whl/cpu':
        for wheel in package.get('wheels', []):
            if 'cp313' in wheel['url'] and 'macosx' in wheel['url']:
                print('current_CPU_index_macos_wheel', wheel['url'].rsplit('/', 1)[-1].split('?')[0])
