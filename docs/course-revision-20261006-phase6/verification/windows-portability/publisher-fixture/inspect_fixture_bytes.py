"""Compare original and corrected fixture bytes under Windows defaults.

The original fixture is read from HEAD without changing the checkout.
Both cases use temporary repositories, real Git commits, and the publisher's
unchanged committed_json gate. This is local emulation, not a Windows run.
"""

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from emulate_windows_fixture import PUBLISHER, ROOT, TEST


def load_fixture(source, name):
    class FixtureLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return self.source_to_code(source, str(TEST))

    loader = FixtureLoader(name, str(TEST))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def summarize(raw):
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "crlf_count": raw.count(b"\r\n"),
        "lf_count": raw.count(b"\n"),
        "ending_hex": raw[-4:].hex(),
    }


def rejection(fixture, root, path, revision):
    try:
        fixture.publisher.committed_json(root, path, revision)
    except ValueError as error:
        assert str(error) == "File differs from selected committed Git revision"
        return str(error)
    return None


def main():
    original = subprocess.run(
        ["git", "show", "HEAD:tests/test_selftrained_data_publish.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    patched = TEST.read_bytes()
    original_write_text = Path.write_text

    def windows_write_text(path, data, encoding=None, errors=None, newline=None):
        return original_write_text(
            path, data, encoding=encoding, errors=errors, newline="\r\n" if newline is None else newline
        )

    evidence = {"actual_platform": "Linux; Windows newline defaults emulated", "cases": []}
    evidence["publisher_sha256"] = hashlib.sha256(PUBLISHER.read_bytes()).hexdigest()
    with pytest.MonkeyPatch.context() as monkeypatch:
        config_index = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
        monkeypatch.setenv(f"GIT_CONFIG_KEY_{config_index}", "core.autocrlf")
        monkeypatch.setenv(f"GIT_CONFIG_VALUE_{config_index}", "true")
        monkeypatch.setenv("GIT_CONFIG_COUNT", str(config_index + 1))
        monkeypatch.setattr(Path, "write_text", windows_write_text)
        for label, source, expected_match in (("original_HEAD", original, False), ("corrected", patched, True)):
            fixture = load_fixture(source, f"fixture_{label}")
            with tempfile.TemporaryDirectory(prefix=f"publisher-{label}-") as directory:
                root = fixture.repository.__wrapped__(Path(directory))
                path = "docs/selftrained/package-recipe.json"
                revision = fixture.commit(root, path, fixture.recipe())
                working_tree = (root / path).read_bytes()
                committed = subprocess.run(
                    ["git", "show", f"{revision}:{path}"], cwd=root, capture_output=True, check=True
                ).stdout
                matches = working_tree == committed
                assert matches is expected_match
                case = {
                    "fixture": label,
                    "source_sha256": hashlib.sha256(source).hexdigest(),
                    "git_autocrlf": fixture.git(root, "config", "--get", "core.autocrlf"),
                    "git_attributes": fixture.git(root, "check-attr", "text", "eol", "--", path).splitlines(),
                    "working_tree": summarize(working_tree),
                    "committed": summarize(committed),
                    "bytes_equal": matches,
                    "initial_gate_rejection": rejection(fixture, root, path, revision),
                }
                if expected_match:
                    value, recipe_sha = fixture.publisher.committed_json(root, path, revision)
                    assert value == fixture.recipe() and recipe_sha == hashlib.sha256(committed).hexdigest()
                    # A real CRLF-only working-tree change must still be rejected.
                    (root / path).write_bytes(committed.replace(b"\n", b"\r\n"))
                    case["crlf_working_tree_mutation_rejection"] = rejection(fixture, root, path, revision)
                    assert case["crlf_working_tree_mutation_rejection"] is not None
                else:
                    assert case["initial_gate_rejection"] is not None
                evidence["cases"].append(case)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
