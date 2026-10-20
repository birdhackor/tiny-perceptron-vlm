"""核對單純圍欄的邊界、實際 Notebook 程式格包含關係，以及逐筆補丁的區塊穩定性。"""
import argparse
import ast
import hashlib
import importlib.util
import io
import itertools
import json
import re
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def observe(gate, text):
    try:
        return {"accepted": True, "blocks": gate.fence_blocks(text, "fixture.md")}
    except ValueError as error:
        return {"accepted": False, "error": str(error)}


def main(args):
    repo = args.repo.resolve()
    gate = load("gate", repo / "docs/review-tools/incremental_revision.py")
    builder = load("builder", repo / "scripts/build_course.py")
    rows = []
    samples = [
        ("empty", "", True), ("plain", "new-token\n", True),
        ("lf", "```python\ninside\n```\nnew-token\n", True),
        ("crlf", "```python\r\ninside\r\n```\r\nnew-token\r\n", True),
        ("final_close_without_lf", "```python\ninside\n```", True),
        ("language_alphabet", "```AZ09_+-\ninside\n```\nnew-token\n", True),
        ("bare_pair", "```\ninside\n```\nnew-token\n", True),
        ("dot_in_language", "```python.3\ninside\n```\nnew-token\n", False),
        ("unicode_language", "```程式\ninside\n```\nnew-token\n", False),
        ("double_cr", "```python\r\r\ninside\n```\n", False),
        ("bare_cr", "```python\rinside\r```\r", False),
        ("unicode_separator", "```python\u2028inside\n```\n", False),
        ("closer_has_language", "```\n```python\nnew-token\n```\n", False),
        ("stray_opener_after_target", "new-token\n```\ninside\n", False),
        ("midline_fence", "prefix ``` suffix\nnew-token\n", False),
        ("tilde", "~~~\ninside\n~~~\nnew-token\n", False),
        ("quoted", "> ```\n> inside\n> ```\nnew-token\n", False),
        ("indented", " ```\n inside\n ```\nnew-token\n", False),
    ]
    for tail_name, tail in [("space", " "), ("tab", "\t"), ("nbsp", "\u00a0"), ("emspace", "\u2003"),
                             ("ideographic", "\u3000"), ("vertical_tab", "\v")]:
        samples.append(("tail_" + tail_name, "```python\ninside\n```" + tail + "\nnew-token\n", False))
    for name, text, expected in samples:
        result = observe(gate, text)
        rows.append({"case": name, "text": text, "expected_accept": expected, "observed": result})
        assert result["accepted"] == expected, name
    exact_text = "outside\n```python\r\ninside\r\n```\r\noutside"
    blocks = gate.fence_blocks(exact_text, "fixture.md")
    expected = [(exact_text.index("```python"), exact_text.rindex("```") + 4)]
    assert blocks == expected
    mask = [gate.fenced(exact_text, i, "fixture.md") for i in range(len(exact_text) + 1)]
    assert mask == [expected[0][0] <= i <= expected[0][1] for i in range(len(exact_text) + 1)]

    tree = ast.parse((repo / "scripts/build_course.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    heading_call = next(node for node in ast.walk(functions["build"]) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "split")
    heading_re = re.compile(ast.literal_eval(heading_call.args[0]), re.M)
    blocks_call = next(node for node in ast.walk(functions["notebook"]) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "compile")
    program_re = re.compile(ast.literal_eval(blocks_call.args[0]), re.S)
    units = ["prose\n", "```\ninside\n```\n", "```python\nx = 1\n```\n", "```bash\necho x\n```\n",
             "```python\nx = 1\n## 1.2 Split\ny = 2\n```\n"]
    corpus_digest = hashlib.sha256()
    documents = sections = cells = protected_characters = 0
    witnesses = []
    for count in range(4):
        for chunks in itertools.product(units, repeat=count):
            for ending in ("\n", "\r\n"):
                for trailing in (True, False):
                    text = ("## 1.1 Fixture\n\n" + "\n".join(chunks)).replace("\n", ending)
                    if not trailing:
                        text = text.rstrip("\r\n")
                    corpus_digest.update(text.encode()); corpus_digest.update(b"\0")
                    guard = gate.fence_blocks(text, "fixture.md")
                    normalized = io.StringIO(text, newline=None).read()
                    offsets, cursor = [], 0
                    while cursor < len(text):
                        offsets.append(cursor)
                        cursor += 2 if text.startswith("\r\n", cursor) else 1
                    assert len(offsets) == len(normalized)
                    headings = list(heading_re.finditer(normalized))
                    for index, heading in enumerate(headings):
                        stop = headings[index + 1].start() if index + 1 < len(headings) else len(normalized)
                        body_start = heading.end()
                        body = normalized[body_start:stop]
                        notebook = builder.notebook(heading[1], heading[2], body, repo / "course/chapters/fixture.md")
                        actual_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
                                        and not cell["metadata"].get("course_setup") and not cell["metadata"].get("course_figure")]
                        matches = [match for match in program_re.finditer(body) if match["python"] is not None]
                        assert len(actual_cells) == len(matches)
                        for cell, match in zip(actual_cells, matches):
                            assert "".join(cell["source"]) == match["python"].strip()
                            source_positions = [offsets[pos] for pos in range(body_start + match.start(), body_start + match.end())]
                            assert all(any(first <= position <= last for first, last in guard) for position in source_positions)
                            protected_characters += len(source_positions)
                            cells += 1
                        sections += 1
                    documents += 1
                    if len(witnesses) < 2 or count == 1 and "```python" in text:
                        witnesses.append({"text": text, "guard": guard, "section_count": len(headings)})

    patch_rows = []
    explicit = [
        ("ordinary", "new-token\n", [{"id": "a", "old": "old-token", "new": "new-token"}], True),
        ("inside", "```python\nnew-token\n```\n", [{"id": "a", "old": "old-token", "new": "new-token"}], False),
        ("create_unclosed", "``X\n", [{"id": "a", "old": "`", "new": "X"}], False),
        ("create_invalid_line", "use``X end\n", [{"id": "a", "old": "`", "new": "X"}], False),
        ("two_patch_endpoint_shift", "``X\nbody\n``Y\n", [{"id": "a", "old": "`", "new": "X"}, {"id": "b", "old": "`", "new": "Y"}], False),
        ("two_plain_edits", "X\n```python\ninside\n```\nY\n", [{"id": "a", "old": "before", "new": "X"}, {"id": "b", "old": "after", "new": "Y"}], True),
        ("unrelated_unsafe_file", "X\n~~~\ninside\n~~~\n", [{"id": "a", "old": "before", "new": "X"}], False),
    ]
    for name, text, patches, accepted in explicit:
        try:
            result = {"accepted": True, "versions": gate.undo_patches(text, list(enumerate(patches)), "fixture.md")}
        except ValueError as error:
            result = {"accepted": False, "error": str(error)}
        assert result["accepted"] == accepted
        patch_rows.append({"case": name, "text": text, "patches": patches, "result": result})
    alphabet = ["```", "```python", "body", ""]
    surroundings = [()] + list(itertools.product(alphabet, repeat=1)) + list(itertools.product(alphabet, repeat=2))
    old_values = ["`", "``", "python", "x", " ", "\u2028", "<", ">", "~", "~~"]
    tried = accepted = invalid_current = 0
    accepted_witnesses = []
    for prefix, suffix, target in itertools.product(surroundings, surroundings, ["X", "``X", "`X`", "preXpost"]):
        text = "\n".join([*prefix, target, *suffix]) + "\n"
        before = observe(gate, text)
        if not before["accepted"]:
            invalid_current += 1
            continue
        position = text.index("X")
        for old in old_values:
            assert old and all(token not in old for token in gate.PATCH_FORBIDDEN)
            patch = {"id": "edit", "old": old, "new": "X"}
            tried += 1
            try:
                versions = gate.undo_patches(text, [(0, patch)], "fixture.md")
            except ValueError:
                continue
            earlier = versions[-1][0]
            after = gate.fence_blocks(earlier, "fixture.md")
            assert not gate.fenced(text, position, "fixture.md")
            assert not any(first <= position + delta <= last for first, last in after for delta in range(len(old)))
            assert [text[first:last] for first, last in before["blocks"]] == [earlier[first:last] for first, last in after]
            accepted += 1
            if len(accepted_witnesses) < 3:
                accepted_witnesses.append({"text": text, "patch": patch, "earlier": earlier})
    print(json.dumps({"rule_cases": rows, "exact_positions": {"text": exact_text, "blocks": blocks, "expected": expected, "all_character_positions_checked": len(mask)},
                      "builder_containment": {"documents": documents, "sections": sections, "program_cells": cells,
                                              "protected_source_characters": protected_characters, "counterexamples": [], "corpus_sha256": corpus_digest.hexdigest(),
                                              "heading_pattern_from_actual_ast": heading_re.pattern, "blocks_pattern_from_actual_ast": program_re.pattern,
                                              "actual_function": "scripts/build_course.py:notebook", "witnesses": witnesses},
                      "patch_cases": patch_rows, "patch_search": {"replacements_tried": tried, "accepted": accepted, "invalid_current_documents": invalid_current,
                                                                    "old_span_inside_block": [], "changed_protected_text": [], "witnesses": accepted_witnesses},
                      "reasoning": "old/new 都不含 CR/LF，所以一筆只影響一行。現行圍欄行本身受保護，不能修改；若從非圍欄行新增合法圍欄，只有一個配對端點改變，必遇到帶語言收尾或未配對檔尾而被倒退後檢查拒絕。逐筆倒退強制這個不變量，不能只看兩個端點都單純。", 
                      "source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in ["docs/review-tools/incremental_revision.py", "scripts/build_course.py"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    main(parser.parse_args())
