from pathlib import Path
import ast
import difflib
import hashlib
import io
import json
import platform
import re
import tokenize
from datetime import datetime, UTC

root = Path("/tmp/tpb").resolve()
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def read_json(relative, expected):
    path = root / relative
    raw = path.read_bytes()
    assert sha(raw) == expected, relative
    return json.loads(raw)
old_ref = {"path": "docs/course-revision-20261101-c-v3/source-freeze-v21.json", "sha256": "d9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea"}
new_ref = {"path": "docs/course-revision-20261101-c-v3/source-freeze-v22.json", "sha256": "0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59"}
old, new = [read_json(v["path"], v["sha256"]) for v in [old_ref, new_ref]]
def frozen(page):
    raw = (root / page["snapshot"]).read_bytes()
    assert sha(raw) == page["source_sha256"], page["page_id"]
    return raw
def chapter(freeze):
    pages = [p for p in freeze["pages"] if p["source"] == "course/chapters/19.md"]
    intro = [p for p in pages if p["selector"] == "before-first-section"]
    sections = [p for p in pages if p["selector"].startswith("section:")]
    assert len(intro) == 1 and len(sections) == 25 and len(pages) == 26
    assert [p["page_id"] for p in sections] == [f"19.{i}" for i in range(1, 26)]
    return b"".join(frozen(p) for p in intro + sections)
old_chapter, new_chapter = chapter(old), chapter(new)
assert sha(old_chapter) == "6fb67227641d879216eaf5506763552a6a3173a106140c8f0ff94b700edfd61a"
live_chapter = (root / "course/chapters/19.md").read_bytes()
assert new_chapter == live_chapter
assert sha(live_chapter) == "d72c66984bac350585d79a6d21369096b372926d77069b90f1b8e81e15ebe4e5"
old_pages = {p["page_id"]: p for p in old["pages"]}
new_pages = {p["page_id"]: p for p in new["pages"]}
assert set(old_pages) == set(new_pages) and len(old_pages) == 370
changed = [k for k in old_pages if old_pages[k] != new_pages[k]]
assert changed == ["19.25"], changed
old_section, new_section = [frozen(pages["19.25"]).decode() for pages in [old_pages, new_pages]]
pattern = re.compile(r"^\x60\x60\x60(?P<lang>[^\n]*)\n(?P<code>.*?)^\x60\x60\x60[ \t]*(?=\n|\Z)", re.M | re.S)
def python_matches(text):
    return [m for m in pattern.finditer(text) if m["lang"] == "python"]
def nonpython_text(text):
    matches = python_matches(text)
    for m in reversed(matches):
        text = text[:m.start("code")] + "<PYTHON_BODY>" + text[m.end("code"):]
    return text
old_blocks, new_blocks = [python_matches(text) for text in [old_section, new_section]]
assert len(old_blocks) == len(new_blocks) == 2
nonpython_equal = nonpython_text(old_section) == nonpython_text(new_section)
assert nonpython_equal
ignored = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
def semantic_tokens(text):
    return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type not in ignored]
blocks = []
for index, (left, right) in enumerate(zip(old_blocks, new_blocks), 1):
    before, after = left["code"], right["code"]
    old_ast = ast.dump(ast.parse(before), include_attributes=False)
    new_ast = ast.dump(ast.parse(after), include_attributes=False)
    ast_equal = old_ast == new_ast
    tokens_equal = semantic_tokens(before) == semantic_tokens(after)
    token_changes = [{"operation": op, "before": semantic_tokens(before)[i:j], "after": semantic_tokens(after)[k:l]} for op, i, j, k, l in difflib.SequenceMatcher(a=semantic_tokens(before), b=semantic_tokens(after)).get_opcodes() if op != "equal"]
    assert ast_equal
    assert all(t[0] == tokenize.OP and t[1] in ("(", ")") for change in token_changes for t in change["before"] + change["after"])
    blocks.append({"index": index, "old_code_sha256": sha(before.encode()), "new_code_sha256": sha(after.encode()), "ast_sha256": sha(old_ast.encode()), "ast_equal_without_location_attributes": ast_equal, "token_stream_equal_ignoring_only_layout_tokens": tokens_equal, "nonlayout_token_changes": token_changes, "old_lines": len(before.splitlines()), "new_lines": len(after.splitlines())})
current = read_json("docs/course-revision-20261101-c-v3/technical-v21/canonical_12_2/current-v1.json", "dded8aa48e12ceeffe073433beaefca11fc2268583b7a5c6db00edc47fbed691")
repository_identity = []
for item in current["sources"]:
    if item["kind"] == "repository_code" and item["id"] != "s8":
        actual = sha((root / item["path"]).read_bytes())
        assert actual == item["sha256"], item["path"]
        repository_identity.append({"source_id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})
page = new_pages["12.2"]
assert old_pages["12.2"] == page and page["figures_sha256"] == {}
body = frozen(page)
raw12 = (root / "course/chapters/12.md").read_text()
headings = list(re.finditer(r"^## .+$", raw12, re.M))
actual_body = next(raw12[h.start():headings[i + 1].start() if i + 1 < len(headings) else len(raw12)] for i, h in enumerate(headings) if h.group().startswith("## 12.2 "))
assert actual_body.encode() == body and sha(body) == current["source_sha256"]
sentence = next(line for line in actual_body.splitlines() if "每點的差距不超過" in line)
result = {
 "schema_version": 1, "kind": "same_owner_independent_source22_format_dependency_inspection",
 "reviewer_task": "/root/c_v3_technical_coord/canonical_12_2", "created_at": datetime.now(UTC).isoformat(),
 "environment": {"python": platform.python_version(), "device": "cpu", "inspection": "stdlib ast/tokenize/hash comparison only; no model, notebook, training or PDF research execution"},
 "old_freeze": old_ref, "new_freeze": new_ref, "old_chapter19_sha256": sha(old_chapter), "new_chapter19_sha256": sha(live_chapter),
 "freeze_page_fields_changed": changed, "unchanged_pages_all_fields": 369,
 "chapter19_sections_hashed_each_version": 25, "chapter19_intro_hashed_each_version": 1,
 "changed_chapter19_sections": [p["page_id"] for p in old["pages"] if p["source"] == "course/chapters/19.md" and p["source_sha256"] != new_pages[p["page_id"]]["source_sha256"]],
 "old_19_25_sha256": sha(old_section.encode()), "new_19_25_sha256": sha(new_section.encode()),
 "python_blocks": blocks, "all_nonpython_text_and_fences_exactly_equal": nonpython_equal,
 "source12_2_metadata_all_fields_unchanged": True, "source12_2_sha256": sha(body), "source12_2_figures": {},
 "current_allclose_explanation": sentence, "unchanged_repository_dependencies": repository_identity,
 "scope_judgment": "The only changed page is19.25. Both Python ASTs match and all surrounding prose/non-Python fences match exactly. Block1 changes only layout; block2 also removes one redundant grouping-parenthesis pair around the accepted expression. This is format-only semantic equivalence, not literal whitespace-only byte equality.12.2 body, T.6 format source, tone, loader, resampling implementations and fixed raw experiment identity remain identical, so the formatting has no effect on12.2 sampling/allclose/audio-input claims. The prior owner/date-bound authority PDFs and original experiment evidence remain retained, not newly first-read or rerun.",
 "initial_check_limitation": "The first local checker also asserted identical non-layout token streams and rejected the redundant parentheses removal. This corrected checker records that exact token delta and separately requires identical ASTs; no source code or prior review was changed.",
 "does_not_establish": "No new semantic-quality result for C-v3, GPU, neural training, final test, full book or publication."
}
print(json.dumps(result, ensure_ascii=False, indent=2))
