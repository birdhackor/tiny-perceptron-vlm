"""用實際切片與 Markdown 處理重現註解列出的顯示差異，不把明列例外算成准入漏洞。"""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Match


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), "/Users/yclin/.cache/uv/archive-v0/_s1u9QcvnBYGXI-2", "/Users/yclin/.cache/uv/archive-v0/4qFlT7bxJOvyO4qC",
                   "/Users/yclin/.cache/uv/archive-v0/G9AZbu6IOXfcgErk", "/Users/yclin/.cache/uv/archive-v0/-TXZPbBw8D3ICYU3QHDYY",
                   "/Users/yclin/.cache/uv/archive-v0/mVzq0z1DaoiU8VaP"]
    import markdown
    import markdown_it
    import mistune
    gate = load("gate", repo / "docs/review-tools/incremental_revision.py")
    builder = load("scripts.build_course", repo / "scripts/build_course.py")
    exporter = load("scripts.export_course", repo / "scripts/export_course.py")
    html_support = load("html_support", repo / "docs/technical-reviews/artifacts/review-20261019-review-tools/fence-semantics-probe-current-04.py")
    config = tomllib.loads((repo / "zensical.toml").read_text())["project"]["markdown_extensions"]
    extensions = {name: value for name, value in config.items() if name != "pymdownx"}
    extensions.update({"pymdownx." + name: value for name, value in config["pymdownx"].items()})
    common = markdown_it.MarkdownIt("commonmark")

    def in_pre(html):
        parser = html_support.Blocks()
        parser.feed(html)
        return any("new-token" in block for block in parser.blocks)

    def rendered(text):
        site = markdown.markdown(text, extensions=list(extensions), extension_configs=extensions)
        notebook = common.render(text)
        return {"site_html": site, "site_pre_at_marker": in_pre(site), "commonmark_html": notebook, "commonmark_pre_at_marker": in_pre(notebook)}

    def fragments(text):
        parts = re.split(r"^## ([\dABC]+\.\d+) (.+)$", text, flags=re.M)
        outputs = []
        for index in range(1, len(parts), 3):
            section, title, body = parts[index:index + 3]
            book = builder.notebook(section, title, body, repo / "course/chapters/fixture.md")
            website_parts = []
            for number, cell in enumerate(book["cells"]):
                content = "".join(cell["source"])
                if cell["cell_type"] == "markdown":
                    if number == 0:
                        content = re.sub(r"^# [^\n]+\n?", "", content, count=1)
                    website_parts.append(exporter.reading_markdown(content, repo / "course/chapters/fixture.md", {}, {}, "fixture"))
                elif "course_figure" in cell.get("metadata", {}):
                    visual = cell["metadata"]["course_figure"]
                    website_parts.append(exporter.diagram_html("figures/" + visual["name"] + ".svg", visual["alt"], "fixture-diagram"))
                elif not cell["metadata"].get("course_setup"):
                    website_parts.append(exporter.code_markdown(content))
            website_text = "\n\n".join(website_parts)
            website_html = rendered(website_text)
            cells = [{"type": cell["cell_type"], "metadata": cell["metadata"], "source": "".join(cell["source"])} for cell in book["cells"]]
            markdown_html = [rendered(cell["source"]) for cell in cells if cell["type"] == "markdown"]
            outputs.append({"section": section, "cells": cells, "markdown_rendered": markdown_html,
                            "website_markdown": website_text, "website_rendered": website_html,
                            "marker_in_markdown_code": any(row["commonmark_pre_at_marker"] for row in markdown_html),
                            "marker_in_program_cell": any("new-token" in cell["source"] for cell in cells if cell["type"] == "code")})
        return outputs

    section_text = "## 1.1 First\n\n```\nfirst\n## 1.2 Second\n```\nnew-token\n```\nlast\n```\n"
    figure = sorted((repo / "course/figures").glob("*.svg"))[0]
    image_text = "## 1.1 Fixture\n\n![alt\n```\ninside\n](../figures/" + figure.name + ")\n```\nnew-token\n```\nlast\n```\n"
    fragment_rows = []
    for name, text in [("section_split", section_text), ("multiline_image", image_text)]:
        guard = gate.fence_blocks(text, "fixture.md")
        assert not gate.fenced(text, text.index("new-token"), "fixture.md")
        result = fragments(text)
        assert any(row["marker_in_markdown_code"] for row in result)
        assert any(row["website_rendered"]["site_pre_at_marker"] for row in result)
        assert not any(row["marker_in_program_cell"] for row in result)
        fragment_rows.append({"case": name, "text": text, "guard_blocks": guard, "fragments": result})

    separators = []
    for separator in ("\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"):
        text = "x" + separator + "~~~\nnew-token\nx" + separator + "~~~\n"
        guard = gate.fence_blocks(text, "fixture.md")
        assert guard == []
        rewritten = exporter.reading_markdown(text, repo / "course/chapters/fixture.md", {}, {}, "fixture")
        after = rendered(rewritten)
        assert after["site_pre_at_marker"]
        separators.append({"separator": ascii(separator), "text": text, "guard_blocks": guard,
                           "raw_rendered": rendered(text), "exported": rewritten, "exported_rendered": after})

    other_rows = []
    for name, text in [
        ("html_block_ignores_fence", "<div>\n```\n</div>\n\n```\nnew-token\n```\nx\n```\n"),
        ("indented_code", "    new-token\n"),
        ("html_pre", "<pre>new-token</pre>\n"),
    ]:
        blocks = gate.fence_blocks(text, "fixture.md")
        result = rendered(text)
        assert not gate.fenced(text, text.index("new-token"), "fixture.md")
        assert result["commonmark_pre_at_marker"]
        other_rows.append({"case": name, "text": text, "guard_blocks": blocks, "rendered": result})

    math_source = Path("/Users/yclin/.cache/uv/archive-v0/p2Je6WLnsft_um2z/nbconvert/filters/markdown_mistune.py")
    syntax = ast.parse(math_source.read_text())
    dotall = next(node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == "_dotall")
    math_if = next(node for node in syntax.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "MISTUNE_V3")
    math_class = next(node for node in math_if.body if isinstance(node, ast.ClassDef) and node.name == "MathBlockParser")
    selected = ast.Module(body=[dotall, math_class], type_ignores=[])
    namespace = {"BlockParser": mistune.BlockParser, "BlockState": mistune.BlockState, "ClassVar": ClassVar, "Iterable": Iterable,
                 "Match": Match, "MISTUNE_V3_ATX": "atx_heading" in mistune.BlockParser.SPECIFICATION}
    exec(compile(selected, str(math_source), "exec"), namespace)
    math_text = "$$\n```\n$$\n\n```\nnew-token\n```\nx\n```\n"
    math_blocks = gate.fence_blocks(math_text, "fixture.md")
    assert not gate.fenced(math_text, math_text.index("new-token"), "fixture.md")
    parser = namespace["MathBlockParser"]()
    state = parser.state_cls()
    state.process(math_text)
    parser.parse(state)
    assert any(token["type"] == "block_code" and "new-token" in token["raw"] for token in state.tokens)
    math_row = {"text": math_text, "guard_blocks": math_blocks, "notebook_math_block_tokens": state.tokens,
                "execution": "執行既有 nbconvert 原始碼中的 _dotall 與 MathBlockParser AST，原封保留類別內容；只載入需要的 Mistune／typing 名稱，沒有替換解析方法。未載入完整 nbconvert exporter 或 Notebook 前端。",
                "nbconvert_source": str(math_source), "nbconvert_source_sha256": hashlib.sha256(math_source.read_bytes()).hexdigest(),
                "mistune_version": mistune.__version__, "mistune_source_sha256": hashlib.sha256(Path(mistune.__file__).read_bytes()).hexdigest()}
    print(json.dumps({"fragment_cases": fragment_rows, "newline_cases": separators, "other_cases": other_rows, "math_case": math_row,
                      "figure_reference": {"path": str(figure.relative_to(repo)), "sha256": hashlib.sha256(figure.read_bytes()).hexdigest()},
                      "renderer": {"markdown": markdown.__version__, "markdown_it": markdown_it.__version__, "extensions": extensions},
                      "source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in ["docs/review-tools/incremental_revision.py", "scripts/build_course.py", "scripts/export_course.py", "zensical.toml"]},
                      "interpretation": "所有 marker 在守衛區塊外，而顯示可落在程式框；這些都是現行規約及註解已明列的例外，不列必要問題。實際 Python 程式格的包含承諾另由程式格探針核對。"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    main(parser.parse_args())
