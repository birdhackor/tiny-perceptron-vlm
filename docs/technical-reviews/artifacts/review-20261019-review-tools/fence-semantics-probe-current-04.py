"""將圍欄守衛與鎖定網站解析器、CommonMark 解析器作來源文字對照。"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import tomllib
from html.parser import HTMLParser
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.depth += 1
            self.blocks.append("")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.blocks[-1] += data


def main(args):
    repo = args.repo.resolve()
    sys.path[:0] = [str(args.markdown_root), str(args.pymdownx_root), str(args.commonmark_root), str(args.mdurl_root)]
    import markdown
    import pymdownx.superfences
    import pymdownx.__meta__
    import markdown_it
    import markdown_it.rules_block.fence
    import mdurl

    spec = importlib.util.spec_from_file_location("reviewed_gate", repo / "docs/review-tools/incremental_revision.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    lock = tomllib.loads((repo / "uv.lock").read_text())
    locked = {p["name"]: p["version"] for p in lock["package"] if p["name"] in ("markdown", "pymdown-extensions")}
    assert locked == {"markdown": markdown.__version__, "pymdown-extensions": pymdownx.__meta__.__version__}
    parser = markdown_it.MarkdownIt("commonmark")
    marker = "new-token"
    cases = []
    for tail_name, tail in [("none", ""), ("space", " "), ("tab", "\t"), ("spaces_tabs", " \t "),
                             ("nbsp", "\u00a0"), ("emspace", "\u2003"), ("ideographic_space", "\u3000"),
                             ("vertical_tab", "\v"), ("text", "x")]:
        for fence in ("```", "~~~"):
            cases.append(("tail_" + tail_name + ("_backtick" if fence == "```" else "_tilde"),
                          fence + 'python\n' + fence + tail + '\nprint("new-token")\n' + fence + '\n',
                          "收尾後字元的定義"))
    cases += [
        ("two_space_fake_close", '```python\n  ```\nprint("new-token")\n```\n', "縮排收尾差異"),
        ("four_space_fake_close", '```python\n    ```\nprint("new-token")\n```\n', "縮排假收尾"),
        ("quote_fake_close", '```python\n> ```\nprint("new-token")\n```\n', "引用假收尾"),
        ("longer_close", '```\na\n````\nnew-token\n```\n', "網站仍在區塊，CommonMark 已收尾"),
        ("notebook_reopens", '```\na\n````\nb\n```\nnew-token\n```\n', "CommonMark 再次開啟的區塊"),
        ("shorter_close", '````python\n```\nnew-token\n````\n', "較短記號不能收尾"),
        ("other_character", '~~~python\n```\nnew-token\n~~~\n', "不同記號不能收尾"),
        ("info_on_close", '```python\n```python\nnew-token\n```\n', "收尾不能有 info string"),
        ("invalid_backtick_info", '```bad`info\n```\nprint("new-token")\n```\n', "無效開頭造成狀態錯位"),
        ("invalid_superfences_options", '```python invalid\n```\nprint("new-token")\n```\n', "網站放棄無效選項後重新開啟"),
        ("ordinary_closed", '```python\nprint(4)\n```\nnew-token\n', "合法收尾後的普通文字"),
        ("tilde_closed", '~~~python\nprint(4)\n~~~\nnew-token\n', "合法波浪號收尾後的普通文字"),
        ("quoted_closed", '> ```\n> x\n> ```\n\nnew-token\n', "合法引用收尾"),
        ("indent_closed", '  ```\n  x\n  ```\n\nnew-token\n', "合法縮排收尾"),
        ("tab_normalization", '\t```\n\tx\n    ```\n\nnew-token\n', "網站將 tab 展開成空格"),
        ("unclosed", '```python\nprint("new-token")\n', "沒有收尾"),
        ("quote_open", '> ```python\n> print("new-token")\n> ```\n', "引用中的區塊"),
        ("container_end", '> ```\n> x\n\nnew-token\n', "CommonMark 在容器結束時關閉區塊"),
        ("indented_code_excluded", '    print("new-token")\n', "明文不涵蓋的無圍欄程式區塊"),
        ("html_pre_excluded", '<pre>print("new-token")</pre>\n', "明文不涵蓋的 HTML"),
    ]
    rows = []
    for name, text, reason in cases:
        position = text.index(marker)
        html = markdown.markdown(text, extensions=["pymdownx.superfences"])
        blocks = Blocks()
        blocks.feed(html)
        tokens = parser.parse(text)
        fences = [{"map": token.map, "content": token.content, "markup": token.markup, "info": token.info}
                  for token in tokens if token.type == "fence"]
        try:
            undo = {"accepted": True, "versions": gate.undo_patches(text, [(0, {"id": "fixture", "old": "old-token", "new": marker})], "fixture.md")}
        except ValueError as error:
            undo = {"accepted": False, "error": str(error)}
        super_guard = gate.inside_fence(text, position, gate.closes_superfences)
        common_guard = gate.inside_fence(text, position, gate.closes_commonmark)
        in_super = any(marker in block for block in blocks.blocks)
        in_common_fence = any(marker in block["content"] for block in fences)
        union = gate.fenced(text, position)
        rows.append({"case": name, "reason": reason, "text": text, "escaped_text": ascii(text),
                     "guard": {"superfences": super_guard, "commonmark": common_guard, "union": union},
                     "renderer": {"superfences_pre_at_marker": in_super, "commonmark_fence_at_marker": in_common_fence,
                                  "superfences_html": html, "commonmark_html": parser.render(text), "commonmark_fences": fences},
                     "undo": undo, "unsafe_union": (in_super or in_common_fence) and not union
                         and name not in ("indented_code_excluded", "html_pre_excluded")})
    src = [Path(markdown.__file__), Path(pymdownx.superfences.__file__), Path(markdown_it.__file__),
           args.commonmark_root / "markdown_it/rules_block/fence.py", Path(mdurl.__file__),
           repo / "uv.lock", repo / "zensical.toml", repo / "docs/review-tools/incremental_revision.py"]
    print(json.dumps({"environment": {"python": sys.version, "platform": platform.platform()},
                      "renderers": {"markdown": markdown.__version__, "pymdownx": pymdownx.__meta__.__version__,
                                    "markdown_it": markdown_it.__version__, "commonmark_preset": "commonmark",
                                    "locked": locked, "source_sha256": {str(p): digest(p) for p in src},
                                    "cache_use": "只讀既有套件快取，不安裝、不更改依賴。"},
                      "normative_reference": {"url": "https://spec.commonmark.org/0.31.2/#fenced-code-blocks",
                                              "version": "0.31.2", "closing_tail_quote": "may be followed only by spaces or tabs, which are ignored.",
                                              "opening_rule": "反引號開頭的 info string 不能含反引號；圍欄在容器結束時也會結束。"},
                      "cases": rows, "unsafe_cases": [r["case"] for r in rows if r["unsafe_union"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--markdown-root", type=Path, required=True)
    ap.add_argument("--pymdownx-root", type=Path, required=True)
    ap.add_argument("--commonmark-root", type=Path, required=True)
    ap.add_argument("--mdurl-root", type=Path, required=True)
    main(ap.parse_args())
