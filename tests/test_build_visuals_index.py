"""實測圖不能繞過來源核對，也不能讓未知手寫圖略過metadata檢查。"""

import hashlib
import json

import pytest

from scripts import build_visuals


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    directory = tmp_path / "figures"
    directory.mkdir()
    monkeypatch.setattr(build_visuals, "ROOT", tmp_path)
    monkeypatch.setattr(build_visuals, "DIRECTORY", directory)
    monkeypatch.setattr(build_visuals, "VISUALS", {})
    return directory


def published_plot(gallery):
    name = "looped-mechanism-measured.svg"
    raw = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>'
    (gallery / name).write_bytes(raw)
    manifest = gallery.parent / build_visuals.RESULT_PLOTS[name]["manifest"]
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}}))
    return gallery / name


def test_result_plot_retains_exact_bytes_and_registered_provenance(gallery):
    plot = published_plot(gallery)
    before = plot.read_bytes()
    build_visuals.index_existing_visuals()
    item = build_visuals.VISUALS[plot.stem]
    assert item["source"] == "generated_result"
    assert item["generator"] == "scripts/lesson_products/plot_results.py"
    assert item["title"] and item["caption"]
    assert plot.read_bytes() == before


def test_changed_result_plot_is_rejected_even_after_previous_index(gallery):
    plot = published_plot(gallery)
    build_visuals.index_existing_visuals()
    plot.write_bytes(plot.read_bytes().replace(b"L1 1", b"L2 2"))
    with pytest.raises(ValueError, match="實測圖與固定發布清單不同"):
        build_visuals.index_existing_visuals()


def test_unknown_untitled_svg_is_still_rejected(gallery):
    (gallery / "other.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    with pytest.raises(ValueError, match="手寫圖解需要標題與描述"):
        build_visuals.index_existing_visuals()


def test_handwritten_svg_metadata_is_preserved_and_indexed(gallery):
    plot = gallery / "manual.svg"
    raw = '<svg xmlns="http://www.w3.org/2000/svg"><title>步驟</title><desc>沿箭頭讀。</desc></svg>'.encode()
    plot.write_bytes(raw)
    build_visuals.index_existing_visuals()
    assert build_visuals.VISUALS["manual"]["title"] == "步驟"
    assert build_visuals.VISUALS["manual"]["source"] == "handwritten"
    assert plot.read_bytes() == raw


def test_registered_handwritten_metadata_requires_the_exact_figure_bytes(gallery):
    plot = gallery / "manual.svg"
    raw = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>'
    plot.write_bytes(raw)
    (gallery / "source-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "figures": {
                    plot.name: {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                        "title": "原圖",
                        "description": "教材中原有的描述。",
                    }
                },
            },
            ensure_ascii=False,
        )
    )
    build_visuals.index_existing_visuals()
    assert plot.read_bytes() == raw
    assert build_visuals.VISUALS["manual"]["metadata_source"] == "source-metadata.json"
    plot.write_bytes(raw.replace(b"L1 1", b"L2 2"))
    with pytest.raises(ValueError, match="對應bytes的明示metadata"):
        build_visuals.index_existing_visuals()
