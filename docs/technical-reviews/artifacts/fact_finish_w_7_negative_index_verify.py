"""Check the scope of W.7's claim that a two-item list has only indices 0, 1."""

import hashlib
import html
import json
import platform
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    url = "https://docs.python.org/3.13/library/stdtypes.html#common-sequence-operations"
    raw = urllib.request.urlopen(url, timeout=30).read()
    content = raw.decode("utf-8")
    marker = "If <em>i</em> or <em>j</em> is negative"
    start = content.index(marker)
    end = content.index("</p>", start)
    passage = html.unescape(re.sub(r"<[^>]+>", " ", content[start:end]))
    destination = ROOT / "docs/technical-reviews/artifacts/fact_finish_w_7_sequence_official.txt"
    destination.write_text(
        f"URL: {url}\nPython documentation: 3.13 branch\nAccessed: 2026-10-04\n"
        f"Full fetched HTML SHA256: {hashlib.sha256(raw).hexdigest()}\n"
        "Extraction: complete common-sequence-operations note on negative indices; HTML tags stripped.\n\n"
        + passage
        + "\n",
        encoding="utf-8",
    )
    animals = ["貓", "狗"]
    observed = {}
    for index in (-3, -2, -1, 0, 1, 2):
        try:
            observed[str(index)] = {"result": animals[index]}
        except IndexError as error:
            observed[str(index)] = {"exception": type(error).__name__, "message": str(error)}
    assert observed["-2"] == {"result": "貓"}
    assert observed["-1"] == {"result": "狗"}
    assert observed["2"]["exception"] == "IndexError"
    print(
        json.dumps(
            {
                "environment": {"python": platform.python_version(), "device": "cpu"},
                "input": animals,
                "observed": observed,
                "valid_integer_indices_for_this_list": [-2, -1, 0, 1],
                "official_passage": passage,
                "source_snapshot_path": destination.relative_to(ROOT).as_posix(),
                "original_statement": 'animals = ["貓", "狗"] 只有索引 0、1',
                "conclusion": "Unqualified 'only indices 0, 1' excludes valid negative indices; 'only nonnegative indices 0, 1' is correct.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
