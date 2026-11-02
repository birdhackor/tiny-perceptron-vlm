"""Bounded original-owner controls for actual excerpt-test function; no CLI/model calls."""

import ast
import hashlib
import importlib.util
import inspect
import io
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

BOOK = Path('/tmp/tpb')
OLD = Path('/tmp/tpc/tests/test_course_code_excerpts.py')
NEW = BOOK / 'tests/test_course_code_excerpts.py'
ART = BOOK / 'docs/technical-reviews/artifacts/c-v3-v21-release-transport'
TARGET = 'test_extracted_source_is_what_inspect_getsource_returns'
old_tree, new_tree = (ast.parse(path.read_text()) for path in (OLD, NEW))
old_other = [node for node in old_tree.body if getattr(node, 'name', None) != TARGET]
new_other = [node for node in new_tree.body if getattr(node, 'name', None) != TARGET]
assert ast.dump(ast.Module(old_other, [])) == ast.dump(ast.Module(new_other, []))
old_target = next(node for node in old_tree.body if getattr(node, 'name', None) == TARGET)
new_target = next(node for node in new_tree.body if getattr(node, 'name', None) == TARGET)
assert ast.dump(old_target.decorator_list[0]) == ast.dump(new_target.decorator_list[0])
assert ast.dump(old_target.body[1]) == ast.dump(new_target.body[3])  # unchanged try/import/ImportError handling
assert ast.dump(old_target.body[2].body[0]) == ast.dump(new_target.body[4].body[0])  # exact original assert


def actual_function(tree, root, importer, getsource=inspect.getsource):
    nodes = [node for node in tree.body if getattr(node, 'name', None) in {TARGET, 'top_level_sources', 'module_name'}]
    copied = ast.parse('\n\n'.join(ast.unparse(node) for node in nodes))
    for node in copied.body:
        node.decorator_list = []
    import pytest
    namespace = {'ast': ast, 'io': io,
                 'inspect': SimpleNamespace(getblock=inspect.getblock, getsource=getsource),
                 'DEFINITIONS': (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                 'ROOT': root, 'importlib': SimpleNamespace(import_module=importer), 'pytest': pytest}
    exec(compile(copied, 'actual-excerpt-test-AST', 'exec'), namespace)
    return namespace[TARGET]


empty_calls = []


def cli_import_sentinel(name):
    empty_calls.append(name)
    raise SystemExit(2)


main = BOOK / 'tiny_perceptron/decision_product/__main__.py'
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.parse(main.read_text()).body)
actual_function(new_tree, BOOK, cli_import_sentinel)(main)
assert empty_calls == []
try:
    actual_function(old_tree, BOOK, cli_import_sentinel)(main)
except SystemExit as error:
    assert error.code == 2
else:
    raise AssertionError('old test did not trigger empty-module import sentinel')
assert empty_calls == ['tiny_perceptron.decision_product.__main__']

source = ('def identity(value):\n    return value\n\n'
          '@identity\ndef regular(value):\n    return value + 1\n\n'
          'async def asynchronous():\n    return 2\n\n'
          'class Example:\n    def method(self):\n        return 3\n')
module_names = []
with tempfile.TemporaryDirectory(prefix='c3-excerpt-review-', dir='/tmp') as directory:
    root = Path(directory)
    path = root / 'tiny_perceptron/review_fixture.py'
    path.parent.mkdir(parents=True)
    path.write_text(source)
    alternate = root / 'alternate.py'
    alternate.write_text(source.replace('value + 1', 'value + 9'))

    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        module_names.append(name)
        spec.loader.exec_module(module)
        return module

    good = load('excerpt_private_good', path)
    bad = load('excerpt_private_mismatch', alternate)
    imports, comparisons = [], []

    def importer(name):
        imports.append(name)
        return good

    def getsource(obj):
        comparisons.append(obj.__name__)
        return inspect.getsource(obj)

    actual_function(new_tree, root, importer, getsource)(path)
    assert imports == ['tiny_perceptron.review_fixture']
    assert comparisons == ['identity', 'regular', 'asynchronous', 'Example']
    try:
        actual_function(new_tree, root, lambda name: bad)(path)
    except AssertionError as error:
        assert str(error) == 'tiny_perceptron/review_fixture.py：regular'
    else:
        raise AssertionError('a real differently-sourced function was accepted')

    def broken_import(name):
        raise RuntimeError('definition-bearing module must still be imported')

    try:
        actual_function(new_tree, root, broken_import)(path)
    except RuntimeError as error:
        assert str(error) == 'definition-bearing module must still be imported'
    else:
        raise AssertionError('definition-bearing import was skipped')
for name in module_names:
    del sys.modules[name]

result = {'created_at': datetime.now(UTC).isoformat(),
          'reviewer_task': '/root/omni_p5_full_tools_review',
          'before_sha256': hashlib.sha256(OLD.read_bytes()).hexdigest(),
          'after_sha256': hashlib.sha256(NEW.read_bytes()).hexdigest(),
          'all_other_test_module_AST_unchanged': True,
          'parametrization_import_try_and_per_definition_assert_AST_unchanged': True,
          'actual_main_has_no_top_level_definitions': True,
          'current_actual_main_case_did_not_import': True,
          'old_case_triggered_import_SystemExit_sentinel': True,
          'nonempty_fixture_imported_and_all4_sources_compared': ['identity', 'regular', 'asynchronous', 'Example'],
          'real_alternate_function_source_mismatch_still_asserts': True,
          'nonempty_module_import_failure_still_propagates': True,
          'synthetic_fixture_and_sentinel_not_actual_CLI_or_model_execution': True,
          'no_qualified_source_changes_Git_network_or_model_execution': True}
with (ART / 'ci-excerpt-current-v1-controls-result.json').open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print('PASS: empty-map import excluded; original import sentinel reproduced; all4 definitions compared; real source mismatch and nonempty import errors still fail; unrelated AST and original comparison unchanged.')
