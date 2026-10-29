"""Targeted offline callback against the ROOT's amended transport source.

Reuse the unchanged independent finance/inventory/fixture/ownership assertions;
retain the original defect reproduction in recheck_probes.py without rewriting it.
"""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recheck_probes as prior

spec = importlib.util.spec_from_file_location("review_transport_final", "/tmp/omni-infra-review-recheck-01-transport-final.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)
prior.gt = gt

for name, value in list(vars(prior).items()):
    if name.startswith("test_") and name != "test_reproduce_lost_create_response_falsely_claims_draft_cleanup_from_tag_404":
        globals()[name] = value


class DraftListing(prior.FakeGitHub):
    def __init__(self, *, invisible=False, foreign=False, full_pages=0):
        super().__init__(lose_create_response=True, foreign_identity=foreign)
        self.invisible, self.full_pages = invisible, full_pages
        self.list_calls = []
    def api(self, endpoint, method="GET", payload=None, allow_missing=False):
        if endpoint.startswith("releases?"):
            self.list_calls.append(endpoint)
            page = int(endpoint.rsplit("=", 1)[-1])
            if page <= self.full_pages:
                return [{"tag_name": "foreign-" + str(i), "draft": True, "body": "foreign", "id": 1000+i}
                        for i in range(100)]
            return [] if self.invisible else [deepcopy(self.release)]
        return super().api(endpoint, method, payload, allow_missing)


def test_callback_lost_response_finds_own_draft_by_listing_and_verifies_id_cleanup(tmp_path, monkeypatch):
    server = DraftListing()
    prior.setup_transport(monkeypatch, server)
    with pytest.raises(RuntimeError, match="incomplete"):
        gt.probe(tmp_path)
    receipt = json.loads((tmp_path / "github-transport.json").read_bytes())
    assert receipt["cleanup_completed"] is True and server.release is None
    assert server.list_calls and ("releases/41", "DELETE") in server.calls
    assert not receipt["success"]  # cleanup does not turn a failed roundtrip into success


def test_callback_invisible_draft_stays_unknown_and_is_never_claimed_clean(tmp_path, monkeypatch):
    server = DraftListing(invisible=True)
    prior.setup_transport(monkeypatch, server)
    with pytest.raises(RuntimeError, match="incomplete"):
        gt.probe(tmp_path)
    receipt = json.loads((tmp_path / "github-transport.json").read_bytes())
    assert receipt["cleanup_completed"] is False and receipt["creation_outcome_unknown"] is True
    assert server.release is not None and all(method != "DELETE" for _, method in server.calls)


def test_callback_foreign_nonce_body_is_preserved_after_unknown_post_outcome(tmp_path, monkeypatch):
    server = DraftListing(foreign=True)
    prior.setup_transport(monkeypatch, server)
    with pytest.raises(RuntimeError, match="incomplete"):
        gt.probe(tmp_path)
    receipt = json.loads((tmp_path / "github-transport.json").read_bytes())
    assert receipt["cleanup_completed"] is False and receipt["cleanup_error_type"] == "ValueError"
    assert server.release is not None and all(method != "DELETE" for _, method in server.calls)


@pytest.mark.parametrize("pages,found", [(2, True), (5, False)])
def test_callback_bounded_pagination_finds_or_remains_unknown(tmp_path, monkeypatch, pages, found):
    server = DraftListing(full_pages=pages)
    prior.setup_transport(monkeypatch, server)
    with pytest.raises(RuntimeError, match="incomplete"):
        gt.probe(tmp_path)
    receipt = json.loads((tmp_path / "github-transport.json").read_bytes())
    assert len(server.list_calls) == min(pages + 1, 5)
    assert receipt["cleanup_completed"] is found
    if found:
        assert server.release is None
    else:
        assert receipt["creation_outcome_unknown"] is True and server.release is not None
