"""Run the existing local infrastructure/quota suite against frozen modules."""
import importlib.util
import inspect
import socket
import sys
from pathlib import Path

ACTUAL = Path('/workspace/selftrained-v2')
CANDIDATE = Path('/tmp/p5-repository-card-release-candidate')

def load(name):
    spec = importlib.util.spec_from_file_location(f'scripts.selftrained.{name}', CANDIDATE / 'scripts/selftrained' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if name == 'modal_runner':
        module.ROOT = ACTUAL
    return module

transport = load('hf_transport')
runner = load('modal_runner')

def pytest_collection_modifyitems(items):
    for module in {item.module for item in items}:
        if module.__name__ not in ('test_selftrained_infra', 'test_selftrained_gross_quota'):
            continue
        module.runner = runner
        if hasattr(module, 'transport'):
            module.transport = transport
        if hasattr(module, 'remote_fixture'):
            # Use actual frozen reserve_remote AST for the pre-existing financial
            # tests, while retaining their existing assertions and base fixtures.
            source = inspect.getsource(module.remote_fixture)
            source = source.replace('(ROOT / "scripts/selftrained/modal_runner.py")', '(Path("/tmp/p5-repository-card-release-candidate/scripts/selftrained/modal_runner.py"))')
            exec(compile(source, '<existing fixture selecting frozen candidate AST>', 'exec'), module.__dict__)

def pytest_runtest_setup(item):
    def forbidden(*args, **kwargs):
        raise AssertionError('External network forbidden during independent engineering review')
    socket.socket.connect = forbidden
