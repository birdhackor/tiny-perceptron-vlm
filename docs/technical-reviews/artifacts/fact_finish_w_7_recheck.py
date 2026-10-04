"""Recheck the revised W.7 index wording against official sequence semantics."""

import hashlib
import html
import json
import platform
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_finish_w_7"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    original = PREFIX.with_name(PREFIX.name + "_W_7_snapshot.md").read_bytes()
    revised = PREFIX.with_name(PREFIX.name + "_W_7_revised_snapshot.md").read_bytes()
    assert digest(original) == "01209234da532b374da316ece847188bc4553b15b804b22c2bf62338cb79eb37"
    assert digest(revised) == "c201d89e9d6bb1ad26c17a5d74b92f5ab2d17d24a28c92490c668f333e2db229"
    assert "只有索引 0、1" in original.decode("utf-8")
    assert "從前面用 0、1 編號" in revised.decode("utf-8")
    assert "只有索引 0、1" not in revised.decode("utf-8")
    url = "https://docs.python.org/3.13/library/stdtypes.html#common-sequence-operations"
    raw = urllib.request.urlopen(url, timeout=30).read()
    body = raw.decode("utf-8")
    section = body.index('<section id="common-sequence-operations"')
    start = body.index("<table", section)
    end = body.index("</table>", start) + 8
    table = html.unescape(re.sub(r"<[^>]+>", " ", body[start:end]))
    table = re.sub(r"[ \t]+", " ", table)
    start = body.index("If <em>i</em> or <em>j</em> is negative", section)
    end = body.index("</p>", start)
    negative_note = html.unescape(re.sub(r"<[^>]+>", " ", body[start:end]))
    snapshot = PREFIX.with_name(PREFIX.name + "_sequence_rechecked_official.txt")
    snapshot.write_text(
        f"URL: {url}\nPython documentation: 3.13 branch\nAccessed: 2026-10-04\n"
        f"Full fetched HTML SHA256: {digest(raw)}\n"
        "Extraction: common-sequence operation table and complete negative-index note; HTML tags stripped.\n\n"
        + table
        + "\n\n"
        + negative_note
        + "\n",
        encoding="utf-8",
    )
    animals = ["貓", "狗"]
    try:
        animals[2]
    except IndexError as error:
        third = type(error).__name__ + ": " + str(error)
    else:
        raise AssertionError("index 2 unexpectedly exists")
    print(
        json.dumps(
            {
                "environment": {"python": platform.python_version(), "device": "cpu"},
                "original_sha256": digest(original),
                "revised_sha256": digest(revised),
                "current_positive_index_results": {"0": animals[0], "1": animals[1], "2": third},
                "negative_indices_still_valid": {"-1": animals[-1], "-2": animals[-2]},
                "official_zero_origin_table": table,
                "official_negative_index_note": negative_note,
                "verdict": "verified: revised wording describes front-to-back zero-origin numbering without excluding negative indices",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
