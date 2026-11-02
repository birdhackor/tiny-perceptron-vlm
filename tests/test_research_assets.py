"""固定研究輸入按需取得；不把網路下載藏在離線測試內。"""

import copy
import io
import json
import subprocess
import zipfile
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts import research_assets as assets

NATURAL = "docs/natural-assistant/v4/data/fixture.json"
RECIPE = "docs/selftrained/package-recipe.json"


def write(root, name, raw):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


@pytest.fixture
def bundle(tmp_path):
    files = {NATURAL: b'{"answer": "fixed"}\n', RECIPE: b'{"recipe": "fixed"}\n'}
    catalog = {
        "schema_version": 1,
        "repository": assets.REPOSITORY,
        "source_revision": assets.SOURCE_REVISION,
        "release_tag": assets.RELEASE_TAG,
        "groups": {},
    }
    for group, name, filename in [
        ("natural-v4", NATURAL, "natural.zip"),
        ("selftrained-sources", RECIPE, "sources.zip"),
    ]:
        raw = files[name]
        catalog["groups"][group] = {"name": filename, "files": {name: {"bytes": len(raw), "sha256": assets.sha(raw)}}}
        write(tmp_path, name, raw)
    write(tmp_path, assets.CATALOG, json.dumps(catalog).encode())
    published = assets.build(tmp_path)
    write(tmp_path, assets.CATALOG, json.dumps(published).encode())
    payloads = {
        spec["url"]: (tmp_path / "outputs/research-assets" / spec["name"]).read_bytes()
        for spec in published["groups"].values()
    }
    for name in files:
        (tmp_path / name).unlink()
    for path in (tmp_path / "outputs/research-assets").glob("*.zip"):
        path.unlink()
    return tmp_path, published, files, payloads


def test_download_then_offline_reuse_preserves_every_original_byte(bundle):
    root, _, files, payloads = bundle
    calls = []

    def opener(url, timeout):
        calls.append(url)
        return io.BytesIO(payloads[url])

    assets.fetch(root, opener=opener)
    assert len(calls) == 2
    assert {name: (root / name).read_bytes() for name in files} == files
    assets.verify(root)
    assets.fetch(root, opener=lambda *a, **k: pytest.fail("已核對的本機資料不得連網"))


def test_cached_archives_materialize_without_network(bundle):
    root, catalog, _, payloads = bundle
    for spec in catalog["groups"].values():
        write(root, "outputs/research-assets/" + spec["name"], payloads[spec["url"]])
    assets.fetch(root, opener=lambda *a, **k: pytest.fail("合法快取不得連網"))
    assets.verify(root)


def test_changed_local_input_is_not_overwritten_or_downloaded(bundle):
    root, _, _, _ = bundle
    write(root, NATURAL, b"my changed research input")
    with pytest.raises(ValueError, match="SHA-256"):
        assets.fetch(root, opener=lambda *a, **k: pytest.fail("本機衝突應先報錯"))
    assert (root / NATURAL).read_bytes() == b"my changed research input"


def test_incorrect_archive_hash_is_rejected_before_extraction(bundle):
    root, _, _, _ = bundle
    with pytest.raises(ValueError, match="SHA-256"):
        assets.fetch(root, opener=lambda *a, **k: io.BytesIO(b"wrong archive"))
    assert not (root / NATURAL).exists()


@pytest.mark.parametrize("damage", ["duplicate", "extra", "missing", "symlink", "bytes"])
def test_archive_members_are_checked_before_any_file_is_written(bundle, damage):
    root, catalog, files, _ = bundle
    spec = catalog["groups"]["natural-v4"]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        if damage != "missing":
            entry = zipfile.ZipInfo(NATURAL)
            if damage == "symlink":
                entry.external_attr = 0o120777 << 16
            archive.writestr(entry, b"altered" if damage == "bytes" else files[NATURAL])
        if damage in {"duplicate", "extra"}:
            with pytest.warns(UserWarning) if damage == "duplicate" else nullcontext():
                archive.writestr(NATURAL if damage == "duplicate" else "../escape.txt", files[NATURAL])
    with pytest.raises(ValueError):
        assets.unpack(output.getvalue(), spec, root)
    assert not (root / NATURAL).exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "docs/../../escape", "docs\\escape"])
def test_path_cannot_leave_the_selected_root(tmp_path, name):
    with pytest.raises(ValueError):
        assets.safe_path(tmp_path, name)


def test_verify_reports_missing_input_without_network(bundle):
    root, _, _, _ = bundle
    with pytest.raises(ValueError, match="research_assets.py fetch"):
        assets.verify(root)


def test_source_catalog_is_read_from_the_program_commit_not_local_edits(bundle):
    root, catalog, files, _ = bundle
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", assets.CATALOG], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Catalog"],
        cwd=root,
        check=True,
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    write(root, RECIPE, files[RECIPE])
    assert assets.frozen_input(root, RECIPE, revision) == files[RECIPE]
    changed = b"changed recipe"
    catalog["groups"]["selftrained-sources"]["files"][RECIPE] = {"bytes": len(changed), "sha256": assets.sha(changed)}
    write(root, assets.CATALOG, json.dumps(catalog).encode())
    write(root, RECIPE, changed)
    with pytest.raises(ValueError, match="SHA-256"):
        assets.frozen_input(root, RECIPE, revision)
    with pytest.raises(ValueError, match="不是已列名"):
        assets.frozen_input(root, "docs/course-revision-20261029-omni/finance-policy.json", revision)


def test_catalog_cannot_reclassify_a_control_file_as_research(bundle):
    _, catalog, _, _ = bundle
    altered = copy.deepcopy(catalog)
    item = next(iter(altered["groups"]["natural-v4"]["files"].values()))
    altered["groups"]["natural-v4"]["files"] = {"docs/omni-products/v1/finance-policy.json": item}
    with pytest.raises(ValueError, match="允許"):
        assets.parse_catalog(json.dumps(altered))


def test_local_process_cannot_publish(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(assets, "gh", lambda *a: pytest.fail("本機不得呼叫發布 API"))
    with pytest.raises(ValueError, match="GitHub Actions"):
        assets.publish()


@pytest.mark.parametrize("revision", [":0", "HEAD", "a" * 39, "a" * 41])
def test_frozen_input_rejects_non_commit_references(tmp_path, monkeypatch, revision):
    monkeypatch.setattr(assets.subprocess, "check_output", lambda *a, **k: pytest.fail("不應讀取暫存區或分支"))
    with pytest.raises(ValueError, match="完整"):
        assets.frozen_input(tmp_path, RECIPE, revision)


@pytest.mark.parametrize("status", [200, 404, 403, 503])
def test_release_lookup_only_treats_actual_404_as_absent(monkeypatch, status):
    raw = f'HTTP/2.0 {status} response\r\nContent-Type: application/json\r\n\r\n{{"id":123}}'.encode()
    monkeypatch.setattr(
        assets.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=raw, returncode=0 if status == 200 else 1)
    )
    if status == 200:
        assert assets.release_by_tag(assets.RELEASE_TAG) == {"id": 123}
    elif status == 404:
        assert assets.release_by_tag(assets.RELEASE_TAG) is None
    else:
        with pytest.raises(RuntimeError):
            assets.release_by_tag(assets.RELEASE_TAG)
