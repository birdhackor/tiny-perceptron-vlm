"""Exercise command argument parsing without installing or registering anything."""
import contextlib
import io
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch
import ipykernel.kernelspec as ks

out = Path(__file__).resolve().parent
root = out.parents[3]
results = {}
app = ks.InstallIPythonKernelSpecApp()
app.initialize(['--sys-prefix', '--name', 'tiny-perceptron', '--display-name', 'Tiny Perceptron'])
with patch.object(ks, 'install', return_value='/EVIDENCE_ONLY/no-registration') as install:
    with contextlib.redirect_stdout(io.StringIO()):
        app.start()
args = install.call_args.kwargs
assert args['prefix'] == sys.prefix
assert args['kernel_name'] == 'tiny-perceptron'
assert args['display_name'] == 'Tiny Perceptron'
results['installer_argument_contract'] = {'parsed': args, 'write_operation_mocked': True,
                                         'actual_registration_performed': False}
nb = json.loads((root / 'notebooks/01/1.1.ipynb').read_bytes())
bootstrap = next(''.join(c['source']) for c in nb['cells'] if c['metadata'].get('course_setup'))
g = types.ModuleType('google')
g.__path__ = []
gc = types.ModuleType('google.colab')
g.colab = gc
with patch.dict(sys.modules, {'google': g, 'google.colab': gc}):
    with patch('subprocess.run') as run, patch('os.chdir') as chdir:
        exec(bootstrap, {'__name__': '__main__'})
        calls = [{'argv': c.args[0], 'check': c.kwargs.get('check'),
                  'lfs_skip': c.kwargs.get('env', {}).get('GIT_LFS_SKIP_SMUDGE')}
                 for c in run.call_args_list]
        assert len(calls) == 2
        assert calls[0]['argv'][:3] == ['git', 'clone', '--depth']
        assert calls[1]['argv'][1:5] == ['-m', 'pip', 'install', '--quiet']
        results['colab_bootstrap_contract'] = {'calls': calls, 'chdir': str(chdir.call_args.args[0]),
                                              'subprocess_calls_mocked': True,
                                              'network_or_install_performed': False}
spec = ks.get_kernel_dict()
assert spec['argv'][0] == sys.executable
results['kernel_interpreter'] = spec['argv'][0]
(out / 'command-contract-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(results, ensure_ascii=False, indent=2))
