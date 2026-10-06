"""Narrow independent checks using actual pinned Hub SDK and frozen code."""
import hashlib
import importlib.util
import socket
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import huggingface_hub._commit_api as commit_api
import pytest

CANDIDATE = Path('/tmp/p5-repository-card-release-candidate')
spec = importlib.util.spec_from_file_location('frozen_helpers', CANDIDATE / 'test_repository_card_release.py')
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)
transport, runner = helpers.transport, helpers.runner

@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('External network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)

@pytest.mark.parametrize('outcome', ['sent', 'sdk_race_noop', 'ambiguous_response'])
def test_actual_pinned_sdk_flag_and_cas(monkeypatch, tmp_path, outcome):
    assert huggingface_hub.__version__ == '1.33.0'
    real_api = huggingface_hub.HfApi
    sends, parent_checks, downloads, ops = [], [], [], []

    class API(real_api):
        def repo_info(self, repo_id, repo_type, revision, token=None):
            parent_checks.append((self.token, token))
            sha = helpers.NEW if token is not None else helpers.PARENT
            return SimpleNamespace(private=False, sha=sha)

        def get_paths_info(self, repo_id, paths, **kwargs):
            assert paths == ['README.md'] and kwargs['token'] is False
            return []

        def _validate_yaml(self, *args, **kwargs):
            return None

        def preupload_lfs_files(self, repo_id, additions, **kwargs):
            assert len(additions) == 1 and additions[0].path_in_repo == 'README.md'
            ops.extend(additions)
            additions[0]._upload_mode = 'regular'
            if outcome == 'sdk_race_noop':
                additions[0]._remote_oid = additions[0]._local_oid

    def send(**kwargs):
        sends.append(kwargs)
        assert kwargs['parent_commit'] == helpers.PARENT
        assert kwargs['revision'] == 'main' and kwargs['create_pr'] is False
        assert len(kwargs['operations']) == 1 and kwargs['operations'][0].path_in_repo == 'README.md'
        assert kwargs.get('retry_on_error', False) is False
        if outcome == 'ambiguous_response':
            raise RuntimeError('Synthetic lost response')
        return SimpleNamespace(oid=helpers.NEW, commit_url='https://huggingface.co/synthetic/commit/' + helpers.NEW)

    local = tmp_path / 'README.md'
    local.write_bytes(helpers.PAYLOAD)
    def download(repo_id, filename, **kwargs):
        downloads.append(kwargs)
        assert filename == 'README.md' and kwargs['token'] is False and kwargs['revision'] == helpers.NEW
        return str(local)

    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    monkeypatch.setattr(commit_api, '_send_commit', send)
    monkeypatch.setattr(huggingface_hub, 'get_hf_file_metadata', lambda url, token: SimpleNamespace(commit_hash=helpers.NEW, size=len(helpers.PAYLOAD)))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)

    if outcome == 'sent':
        result = transport.publish_repository_card(helpers.PAYLOAD, helpers.release_for(), helpers.MANIFEST_SHA, 'synthetic-placeholder')
        assert result['new_commit_created'] and ops[0]._is_committed is True
        assert len(sends) == len(downloads) == 1
    elif outcome == 'sdk_race_noop':
        with pytest.raises(ValueError, match='SDK skipped'):
            transport.publish_repository_card(helpers.PAYLOAD, helpers.release_for(), helpers.MANIFEST_SHA, 'synthetic-placeholder')
        assert not sends and not downloads and ops[0]._is_committed is False
        assert len(parent_checks) == 2
    else:
        with pytest.raises(RuntimeError, match='lost response'):
            transport.publish_repository_card(helpers.PAYLOAD, helpers.release_for(), helpers.MANIFEST_SHA, 'synthetic-placeholder')
        assert len(sends) == 1 and not downloads and ops[0]._is_committed is False

def test_exact_64k_bound_and_invalid_payload_types():
    prefix = b'---\nlicense: mit\n---\n'
    payload = prefix + b'x' * (65536 - len(prefix))
    release = helpers.release_for(payload)
    assert transport.approved_repository_card(payload, release, helpers.MANIFEST_SHA)['bytes'] == 65536
    for bad in (bytearray(payload), payload.decode(), None):
        with pytest.raises(ValueError):
            transport.approved_repository_card(bad, release, helpers.MANIFEST_SHA)
    oversized = helpers.release_for(payload + b'x')
    with pytest.raises(ValueError):
        transport.approved_repository_card(payload + b'x', oversized, helpers.MANIFEST_SHA)

def test_short_dispatch_sha_and_wrong_source_binding_fail_before_api(monkeypatch):
    calls = []
    monkeypatch.setattr(transport, 'repository_card_public_gate', lambda *args: calls.append(args))
    release = helpers.release_for()
    for revision, hashes in [('main', {transport.REPOSITORY_CARD_SOURCE: release['file']['sha256']}), ('f' * 40, {transport.REPOSITORY_CARD_SOURCE: '0' * 64})]:
        with pytest.raises(ValueError, match='source SHA'):
            runner.repository_card_source_gate(helpers.PAYLOAD, release, revision, helpers.MANIFEST_SHA, hashes)
    assert not calls

def test_unknown_mode_and_unapproved_extra_paths_fail():
    release = helpers.release_for()
    for mode in ('batch', '', None):
        job = helpers.job_for(deepcopy(release))
        job['release']['mode'] = mode
        with pytest.raises(ValueError, match='Unknown release mode'):
            runner.validate_job(job)
    changed = deepcopy(release)
    changed['exports'] = []
    with pytest.raises(ValueError, match='exact one-file'):
        transport.validate_repository_card_release(changed)

def test_existing_nonrelease_jobs_do_not_acquire_card_source():
    for stage in ('prepare', 'pretrain'):
        job = {'schema_version': 1, 'stage': stage}
        assert runner.validate_job(job) is job
        assert runner.committed_repository_card(job, 'not-used', 'not-used') is None
