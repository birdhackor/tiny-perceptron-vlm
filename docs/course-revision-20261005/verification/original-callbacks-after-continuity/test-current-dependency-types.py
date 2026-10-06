"""Exercise the actual scanner functions against independent byte mutations."""

import ast
import hashlib
from pathlib import Path
import re
import tempfile
import unittest


SCANNER = Path(__file__).with_name("scan-current-dependencies.py")
FUNCTIONS = ast.Module(body=[
    node for node in ast.parse(SCANNER.read_bytes()).body
    if isinstance(node, ast.FunctionDef)
], type_ignores=[])


class CurrentDependencyTypes(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "course/chapters/05.md"
        self.source.parent.mkdir(parents=True)
        self.raw = "# 章\n\n## 5.1 本節\n主文與程式\n\n<details>選讀</details>\n\n## 5.2 次節\n".encode()
        self.source.write_bytes(self.raw)
        self.start = self.raw.index("## 5.1".encode())
        self.end = self.raw.index(b"<details>")
        self.snapshot = self.root / "proof/current-5.1.md"
        self.snapshot.parent.mkdir()
        self.snapshot.write_bytes(self.raw[self.start:self.end])
        self.ns = {
            "ROOT": self.root, "re": re,
            "digest": lambda raw: hashlib.sha256(raw).hexdigest(),
            "checked": [], "candidates": [],
            "progress": {"records": {"5.1": {"source": "course/chapters/05.md#5.1"}}},
            "ledger": {"records": {"11.2": {"technical": {"status": "pass"}}}},
        }
        exec(compile(FUNCTIONS, str(SCANNER), "exec"), self.ns)
        self.context = {
            "source": "course/chapters/05.md#5.1",
            "raw_byte_start": self.start, "raw_byte_end_exclusive": self.end,
            "snapshot": "proof/current-5.1.md",
            "sha256": self.ns["digest"](self.snapshot.read_bytes()),
            "full_section_sha256": self.ns["section_hash"]("course/chapters/05.md#5.1"),
        }

    def scan(self):
        self.ns["walk_current_context"]("11.2", self.context, "/current_context", explicitly_current=True)
        return self.ns["candidates"]

    def test_bounded_and_full_hashes_are_distinct_and_both_current(self):
        self.assertNotEqual(self.context["sha256"], self.context["full_section_sha256"])
        self.assertEqual(self.scan(), [])
        self.assertEqual({r["kind"] for r in self.ns["checked"]}, {
            "explicit_current_context_bounded_bytes",
            "explicit_current_context_bounded_snapshot",
            "explicit_current_context_full_section",
        })

    def test_changed_excluded_tail_still_detects_stale_full_section(self):
        self.source.write_bytes(self.raw.replace("選讀".encode(), "補充".encode()))
        self.assertEqual([r["kind"] for r in self.scan()], ["explicit_current_context_full_section"])

    def test_changed_current_slice_is_not_hidden_by_unchanged_snapshot(self):
        self.source.write_bytes(self.raw.replace("主文".encode(), "正文".encode()))
        self.context["full_section_sha256"] = self.ns["section_hash"](self.context["source"])
        self.assertEqual([r["kind"] for r in self.scan()], ["explicit_current_context_bounded_bytes"])

    def test_changed_snapshot_is_not_hidden_by_current_slice(self):
        self.snapshot.write_bytes(self.snapshot.read_bytes().replace("主文".encode(), "正文".encode()))
        self.assertEqual([r["kind"] for r in self.scan()], ["explicit_current_context_bounded_snapshot"])


if __name__ == "__main__":
    unittest.main()
