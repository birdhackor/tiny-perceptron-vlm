"""Courier-only byte/JSON comparison. No model execution or semantic verdict."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MATRIX = Path(
    "docs/course-revision-20261029-omni/reviews/P5-actual/legacy-own-fix-technical-course/"
    "source15-to-source10-primary-identity-matrix-v1.json"
)
EXPECTED_MATRIX = "57a56fde18e96a7ba97ccd71de2cd0e80bf3a7ced16851fa6076d68df9c31ee0"
FIELDS = ("source_sha256", "figures_sha256", "notebook_sha256", "snapshot")
IDENTITY_FIELDS = FIELDS[:3]


def raw(path):
    return (ROOT / path).read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(page):
    # Preserve missing fields; absent notebook_sha256 is not a null value.
    return {field: page[field] for field in FIELDS if field in page}


matrix_raw = raw(MATRIX)
assert sha(matrix_raw) == EXPECTED_MATRIX
matrix = json.loads(matrix_raw)
freezes = {}
for side, key in (("before", "prior"), ("current", "current")):
    reference = matrix[key]
    data = raw(reference["path"])
    assert sha(data) == reference["sha256"]
    freezes[side] = json.loads(data)
    print("INPUT", side, reference["path"], len(data), sha(data))
print("MATRIX", MATRIX.as_posix(), len(matrix_raw), sha(matrix_raw))
pages = {side: {p["page_id"]: p for p in freeze["pages"]} for side, freeze in freezes.items()}
row_ids = [row["page_id"] for row in matrix["rows"]]
assert len(row_ids) == len(set(row_ids))
assert set(row_ids) == set(pages["before"]) == set(pages["current"])
figure_files = set()
notebooks_checked = 0
unchanged_notebooks_matched_both = 0
unchanged = 0
changed = 0
print("COLUMNS [matrix_index,page_id,changed_fields,F,S]")
print("F=source/figures/notebook field presence AND exact JSON values equal before/current")
print("S=both frozen source snapshots read and SHA256 equal their own freeze entries")
for index, row in enumerate(matrix["rows"]):
    page_id = row["page_id"]
    for side in ("before", "current"):
        page = pages[side][page_id]
        assert row[side] == identity(page)
        assert sha(raw(page["snapshot"])) == page["source_sha256"]
        for figure, digest in page["figures_sha256"].items():
            snapshot = freezes[side]["figure_snapshots"][figure]
            assert snapshot["sha256"] == digest
            assert sha(raw(snapshot["path"])) == digest
            figure_files.add(snapshot["path"])
    expected_changed = [
        field
        for field in IDENTITY_FIELDS
        if (field in row["before"]) != (field in row["current"])
        or row["before"].get(field) != row["current"].get(field)
    ]
    assert row["changed_fields"] == expected_changed
    fields_equal = not expected_changed
    if fields_equal:
        unchanged += 1
        for field in IDENTITY_FIELDS:
            assert (field in row["before"]) == (field in row["current"])
            if field in row["before"]:
                assert row["before"][field] == row["current"][field]
    else:
        changed += 1
    current = pages["current"][page_id]
    if "notebook" in current:
        digest = sha(raw(current["notebook"]))
        assert digest == current["notebook_sha256"]
        notebooks_checked += 1
        if fields_equal:
            assert digest == pages["before"][page_id]["notebook_sha256"]
            unchanged_notebooks_matched_both += 1
    print(json.dumps([index, page_id, row["changed_fields"], int(fields_equal), 1], separators=(",", ":")))
assert (len(row_ids), changed, unchanged) == (369, 23, 346)
print("COUNTS", json.dumps({"pages": len(row_ids), "changed": changed, "unchanged": unchanged,
                           "frozen_source_snapshot_reads": 2 * len(row_ids),
                           "unique_frozen_figure_snapshot_files": len(figure_files),
                           "current_notebook_files_hashed": notebooks_checked,
                           "unchanged_notebooks_matched_both_freezes": unchanged_notebooks_matched_both},
                          separators=(",", ":")))
print("ASSERTIONS_COMPLETE identity and byte checks only; no semantic pass, model forward, training, tuning or publication")
