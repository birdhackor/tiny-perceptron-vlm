"""正文裡標記為「節錄」或「精簡」的程式，必須和 tiny_perceptron 的原始碼保持同步。

標記是 python 區塊的第一行：

- ``# 節錄自 tiny_perceptron/<路徑>.py：<名稱>[、<名稱>…]``：其後逐字等於這些頂層 def／class 的原始碼
  （``inspect.getsource``），多個名稱依序以兩個空行相隔；區塊裡不放其他程式。
- ``# 精簡自 tiny_perceptron/<路徑>.py：<名稱>（<省略了什麼>）``：作者重寫的精簡版。同一節要用
  ``from tiny_perceptron.<模組> import <名稱>`` 取得函式庫版本，並用 ``assert`` 核對兩者算出相同結果。

精簡版這裡只檢查同一節有那個 import 和一個會執行到的 assert；assert 比對得對不對，要等發布流程
執行 Notebook 才核對。冒號、頓號、括號都用全形。第一行以「# 節錄」或「# 精簡」開頭卻不符合格式，
一律是錯誤，避免寫錯的標記被當成一般程式放過。解析與檢查函式都接受根目錄參數，所以能在假的 repo 上驗證。
"""

import ast
import difflib
import importlib
import inspect
import io
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts import build_course

ROOT = Path(__file__).resolve().parents[1]

# 切節與取程式區塊的寫法，和 scripts/build_course.py 產生 Notebook 時相同；有測試核對兩者沒有分歧。
SECTION_HEADING = r"^## ([\dABC]+\.\d+) (.+)$"
PYTHON_BLOCK = r"```python\n(?P<python>.*?)```|!\[(?P<alt>[^\]]*)\]\(\.\./figures/(?P<figure>[^/)]+)\.svg\)"
BLOCKS = re.compile(PYTHON_BLOCK, re.S)

# 第一行看起來像標記的，一律要通過下面的嚴格格式。
LOOKS_LIKE_MARKER = re.compile(r"#\s*(?:節錄|精簡)")
LIBRARY_PATH = r"tiny_perceptron/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py"
NAME = r"[A-Za-z_][A-Za-z0-9_]*"
EXCERPT = re.compile(rf"# 節錄自 ({LIBRARY_PATH})：({NAME}(?:、{NAME})*)")
CONDENSED = re.compile(rf"# 精簡自 ({LIBRARY_PATH})：({NAME})（(.*)）")
FORMAT_HELP = (
    "正確寫法（冒號、頓號、括號都用全形）：\n"
    "# 節錄自 tiny_perceptron/<路徑>.py：<名稱>[、<名稱>…]\n"
    "# 精簡自 tiny_perceptron/<路徑>.py：<名稱>（<省略了什麼>）；精簡版一次只寫一個名稱"
)
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class Fence:
    page: str  # 章檔，相對 repo 根目錄、用 / 分隔
    section: str  # 節號；第一個節標題之前的導言記為「導言」
    number: int  # 本節第幾個 python 區塊，從 1 起算
    code: str  # 區塊內容；和 Notebook 一樣去掉頭尾空白


@dataclass(frozen=True)
class Marker:
    fence: Fence
    line: str  # 區塊第一行
    kind: str = ""  # 「節錄」或「精簡」；格式不符時是空字串
    path: str = ""
    names: tuple[str, ...] = ()
    omitted: str = ""
    problem: str = ""  # 格式不符的原因


def read_fences(root):
    """依 build_course.py 的方式切節，取出每個會變成 Notebook 程式格的 python 區塊。"""
    fences = []
    for chapter in sorted((root / "course/chapters").glob("*.md")):
        parts = re.split(SECTION_HEADING, chapter.read_text(encoding="utf-8"), flags=re.M)
        sections = [("導言", parts[0])] + [(parts[i], parts[i + 2]) for i in range(1, len(parts), 3)]
        for section, body in sections:
            codes = [match["python"] for match in BLOCKS.finditer(body) if match["python"] is not None]
            page = chapter.relative_to(root).as_posix()
            fences.extend(Fence(page, section, number, code.strip()) for number, code in enumerate(codes, 1))
    return fences


def parse_marker(fence):
    """區塊第一行不是標記時回傳 None；看起來像標記的一律回傳 Marker，格式不符會帶著原因。"""
    line = fence.code.partition("\n")[0]
    if not LOOKS_LIKE_MARKER.match(line):
        return None
    if match := EXCERPT.fullmatch(line):
        return Marker(fence, line, "節錄", match[1], tuple(match[2].split("、")))
    if match := CONDENSED.fullmatch(line):
        if not match[3].strip():
            return Marker(fence, line, problem="精簡版的括號裡要寫省略了什麼，不能留空。")
        return Marker(fence, line, "精簡", match[1], (match[2],), match[3].strip())
    return Marker(fence, line, problem="標記行不符合格式。")


def collect_markers(root):
    return [marker for fence in read_fences(root) if (marker := parse_marker(fence))]


def top_level_sources(text):
    """檔案頂層每個 def／class 的原始碼 {名稱: 原始碼}，內容和 inspect.getsource 相同。

    只看模組最外層的敘述：從別處 import 或 re-export 的名稱、巢狀函式與 class 的方法都不在其中。
    """
    lines = io.StringIO(text).readlines()
    sources = {}
    for node in ast.parse(text).body:
        if isinstance(node, DEFINITIONS):
            first = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
            sources[node.name] = "".join(inspect.getblock(lines[first - 1 :]))
    return sources


def module_name(path):
    parts = path.removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def executed_nodes(node):
    """程式格一執行就會跑到的節點；def 和 lambda 裡面要等到被呼叫才執行，不算。"""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield from executed_nodes(child)


def report(marker, *reasons):
    fence = marker.fence
    head = f"{fence.page} {fence.section}（第 {fence.number} 個 python 區塊）\n  標記行：{marker.line}"
    return "\n".join([head, *(textwrap.indent(reason, "  ") for reason in reasons)])


def excerpt_problems(marker, sources):
    expected = "\n\n".join(sources[name] for name in marker.names).rstrip()
    actual = marker.fence.code.partition("\n")[2]
    if actual == expected:
        return []
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile=f"{marker.path}：{'、'.join(marker.names)}（函式庫原始碼）",
        tofile=f"{marker.fence.page} {marker.fence.section} 的 python 區塊（標記行之後）",
        lineterm="",
    )
    return [
        report(
            marker,
            "標記行之後的內容必須逐字等於函式庫原始碼，多個名稱依序以兩個空行相隔：",
            "\n".join(diff) or "（逐行看不出差異，請檢查行尾空白或特殊的換行字元）",
        )
    ]


def condensed_problems(marker, siblings):
    module, name = module_name(marker.path), marker.names[0]
    imported = asserted = False
    reasons = []
    for sibling in siblings:
        try:
            tree = ast.parse(sibling.code)
        except SyntaxError as error:
            reasons.append(f"同一節第 {sibling.number} 個 python 區塊無法解析，查不出 import 與 assert：{error}")
            continue
        for node in executed_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == module:
                imported = imported or any(alias.name == name for alias in node.names)
            asserted = asserted or isinstance(node, ast.Assert)
    if not imported:
        reasons.append(
            f"同一節沒有 `from {module} import {name}`（可用 `as` 換名，避免蓋掉精簡版）；沒有函式庫版本，就無從核對。"
        )
    if not asserted:
        reasons.append(
            "同一節沒有會執行到的 `assert`（寫在 def 裡面的不算）；請用它核對精簡版與函式庫版本算出相同結果。"
        )
    return [report(marker, *reasons)] if reasons else []


def check_marker(root, marker, siblings):
    """檢查一個標記，回傳問題清單；空清單代表通過。siblings 是同一節的全部 python 區塊。"""
    if marker.problem:
        return [report(marker, marker.problem, FORMAT_HELP)]
    library = root / marker.path
    if not library.is_file():
        return [report(marker, f"找不到 {marker.path}。")]
    try:
        sources = top_level_sources(library.read_text(encoding="utf-8"))
    except SyntaxError as error:
        return [report(marker, f"{marker.path} 無法解析：{error}")]
    missing = [name for name in marker.names if name not in sources]
    if missing:
        close = [match for name in missing for match in difflib.get_close_matches(name, sources, n=3)]
        hint = f"相近的頂層名稱：{'、'.join(close)}。" if close else ""
        return [
            report(
                marker,
                f"{marker.path} 沒有頂層的 def／class 叫 {'、'.join(missing)}。"
                "只能引用這個檔案自己定義的名稱；從別處 import 或 re-export 的名稱、巢狀函式與方法都不算。" + hint,
            )
        ]
    if marker.kind == "節錄":
        return excerpt_problems(marker, sources)
    return condensed_problems(marker, siblings)


def check_course(root):
    """檢查 root 底下全部章檔的全部標記，回傳問題清單。"""
    fences = read_fences(root)
    problems = []
    for fence in fences:
        if marker := parse_marker(fence):
            siblings = [other for other in fences if (other.page, other.section) == (fence.page, fence.section)]
            problems.extend(check_marker(root, marker, siblings))
    return problems


# ---- 假的 repo -------------------------------------------------------------------------------------------

PACK = '''def pack(pair):
    """把兩個 0 到 15 的數字併成一個位元組。"""
    return pair[0] * 16 + pair[1]
'''
UNPACK = """def unpack(value):
    return divmod(value, 16)
"""
COUNTER = '''class Counter:
    """數次數。"""

    def __init__(self):
        self.count = 0

    def add(self, amount=1):
        self.count += amount
        return self.count
'''
CACHED = """@functools.lru_cache(maxsize=None)
def cached(value):
    return value * 2
"""
FETCH = """async def fetch(url):
    return url
"""
OUTER = """def outer():
    def inner():
        return 1

    return inner
"""
LIBRARY = '"""假的函式庫。"""\n\nimport functools\n\n\n' + "\n\n".join([PACK, UNPACK, COUNTER, CACHED, FETCH, OUTER])
LIBRARY_FILE = "tiny_perceptron/quantization.py"
OTHER = """def helper():
    return 1
"""


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def block(*lines):
    return "```python\n" + "\n".join(lines) + "\n```"


def marker_line(names, kind="節錄自", path=LIBRARY_FILE):
    return f"# {kind} {path}：" + "、".join(names)


def excerpt(names, *definitions):
    """作者的寫法：標記行加上貼進來的定義，定義之間保留原檔的兩個空行。"""
    return block(marker_line(names), "\n\n".join(definitions).rstrip("\n"))


def chapter_text(*sections):
    text = "# 第九章\n\n導言。\n\n"
    return text + "".join(f"## {number} 一個問題\n\n{body}\n\n" for number, body in sections)


def make_repo(root, *sections, library=LIBRARY, chapter="09"):
    write(root, LIBRARY_FILE, library)
    write(root, "tiny_perceptron/other.py", OTHER)
    write(root, f"course/chapters/{chapter}.md", chapter_text(*sections))
    return root


# ---- 標記行的解析：正例表 ----------------------------------------------------------------------------------

FORMAT_ERROR = "format error"
IGNORED = None
MARKER_LINES = [
    pytest.param(
        "# 節錄自 tiny_perceptron/quantization.py：pack_int4",
        ("節錄", "tiny_perceptron/quantization.py", ("pack_int4",), ""),
        id="excerpt-one-name",
    ),
    pytest.param(
        "# 節錄自 tiny_perceptron/quantization.py：pack_int4、unpack_int4",
        ("節錄", "tiny_perceptron/quantization.py", ("pack_int4", "unpack_int4"), ""),
        id="excerpt-two-names",
    ),
    pytest.param(
        "# 節錄自 tiny_perceptron/selftrained/tools.py：parse_tool_call",
        ("節錄", "tiny_perceptron/selftrained/tools.py", ("parse_tool_call",), ""),
        id="excerpt-from-a-subpackage",
    ),
    pytest.param(
        "# 精簡自 tiny_perceptron/model.py：TinyLM（省略 RoPE、權重共享與 cache）",
        ("精簡", "tiny_perceptron/model.py", ("TinyLM",), "省略 RoPE、權重共享與 cache"),
        id="condensed",
    ),
    pytest.param(
        "# 精簡自 tiny_perceptron/model.py：TinyLM（省略 RoPE（旋轉位置編碼）與 cache）",
        ("精簡", "tiny_perceptron/model.py", ("TinyLM",), "省略 RoPE（旋轉位置編碼）與 cache"),
        id="condensed-with-nested-parentheses",
    ),
    pytest.param("# 節錄自 tiny_perceptron/quantization.py:pack_int4", FORMAT_ERROR, id="half-width-colon"),
    pytest.param(
        "# 節錄: tiny_perceptron/quantization.py：pack_int4", FORMAT_ERROR, id="half-width-colon-after-keyword"
    ),
    pytest.param("# 節錄 tiny_perceptron/quantization.py：pack_int4", FORMAT_ERROR, id="missing-from-character"),
    pytest.param("#節錄自 tiny_perceptron/quantization.py：pack_int4", FORMAT_ERROR, id="missing-space-after-hash"),
    pytest.param("# 節錄自 pack_int4", FORMAT_ERROR, id="missing-path"),
    pytest.param("# 節錄自：pack_int4", FORMAT_ERROR, id="empty-path"),
    pytest.param("# 節錄自 tiny_perceptron/quantization.py：", FORMAT_ERROR, id="missing-name"),
    pytest.param("# 節錄自 scripts/quantize.py：quantize", FORMAT_ERROR, id="path-outside-the-library"),
    pytest.param("# 節錄自 tiny_perceptron_extra/quantization.py：pack_int4", FORMAT_ERROR, id="lookalike-directory"),
    pytest.param(
        "# 節錄自 tiny_perceptron/../scripts/quantize.py：quantize", FORMAT_ERROR, id="path-leaves-the-library"
    ),
    pytest.param("# 節錄自 tiny_perceptron/quantization.txt：pack_int4", FORMAT_ERROR, id="not-a-python-file"),
    pytest.param(
        "# 節錄自 tiny_perceptron/quantization.py：pack_int4,unpack_int4", FORMAT_ERROR, id="half-width-comma"
    ),
    pytest.param(
        "# 節錄自 tiny_perceptron/quantization.py：pack_int4，unpack_int4", FORMAT_ERROR, id="full-width-comma"
    ),
    pytest.param("# 節錄自 tiny_perceptron/quantization.py：pack_int4、", FORMAT_ERROR, id="trailing-separator"),
    pytest.param("# 節錄自 tiny_perceptron/quantization.py：Box.add", FORMAT_ERROR, id="method-name"),
    pytest.param(
        "# 節錄自 tiny_perceptron/quantization.py：pack_int4（省略檢查）", FORMAT_ERROR, id="excerpt-with-omission"
    ),
    pytest.param("# 精簡自 tiny_perceptron/model.py：TinyLM（）", FORMAT_ERROR, id="condensed-empty-parentheses"),
    pytest.param("# 精簡自 tiny_perceptron/model.py：TinyLM（ ）", FORMAT_ERROR, id="condensed-blank-parentheses"),
    pytest.param("# 精簡自 tiny_perceptron/model.py：TinyLM", FORMAT_ERROR, id="condensed-without-parentheses"),
    pytest.param(
        "# 精簡自 tiny_perceptron/model.py：TinyLM(省略 cache)", FORMAT_ERROR, id="condensed-half-width-parentheses"
    ),
    pytest.param(
        "# 精簡自 tiny_perceptron/model.py：TinyLM、Block（省略 cache）", FORMAT_ERROR, id="condensed-two-names"
    ),
    pytest.param(
        "# 精簡 tiny_perceptron/model.py：TinyLM（省略 cache）", FORMAT_ERROR, id="condensed-missing-from-character"
    ),
    pytest.param("import torch", IGNORED, id="ordinary-code"),
    pytest.param("# 先建立一批資料", IGNORED, id="ordinary-comment"),
]


@pytest.mark.parametrize(("first_line", "expected"), MARKER_LINES)
def test_marker_line_syntax(tmp_path, first_line, expected):
    write(tmp_path, "course/chapters/09.md", chapter_text(("9.1", block(first_line, "x = 1"))))
    markers = collect_markers(tmp_path)
    if expected is IGNORED:
        assert markers == []
        assert check_course(tmp_path) == []
    elif expected == FORMAT_ERROR:
        assert len(markers) == 1 and markers[0].problem
        problems = check_course(tmp_path)
        assert len(problems) == 1
        assert "9.1" in problems[0] and first_line in problems[0] and "正確寫法" in problems[0]
    else:
        assert [(m.kind, m.path, m.names, m.omitted) for m in markers] == [expected]
        assert markers[0].problem == ""


# ---- 區塊從哪裡掃：正例表 ------------------------------------------------------------------------------------

MARKED = block(marker_line(["pack"]), PACK.rstrip("\n"))
BLOCK_LOCATIONS = [
    pytest.param(chapter_text(("9.1", MARKED)), [("9.1", "pack")], id="plain-section"),
    pytest.param(
        chapter_text(("9.1", f"<details>\n<summary>補充</summary>\n\n{MARKED}\n\n</details>")),
        [("9.1", "pack")],
        id="inside-details",
    ),
    pytest.param(
        chapter_text(("9.1", "- 一個項目\n\n  " + MARKED.replace("\n", "\n  ") + "\n")),
        [("9.1", "pack")],
        id="indented-in-a-list",
    ),
    pytest.param(
        "# 第九章\n\n" + MARKED + "\n\n## 9.1 一個問題\n\n文字。\n",
        [("導言", "pack")],
        id="before-the-first-section-heading",
    ),
    pytest.param(
        chapter_text(("9.1", "文字。\n\n### 小標題\n\n" + MARKED)),
        [("9.1", "pack")],
        id="under-a-subheading",
    ),
    pytest.param(
        chapter_text(("9.1", MARKED), ("9.2", MARKED.replace("pack", "unpack"))),
        [("9.1", "pack"), ("9.2", "unpack")],
        id="two-sections",
    ),
    pytest.param(
        chapter_text(("9.1", "![圖](../figures/demo.svg)\n\n" + MARKED)),
        [("9.1", "pack")],
        id="after-a-figure",
    ),
    pytest.param(chapter_text(("9.1", MARKED.replace("```python", "```bash"))), [], id="not-a-python-fence"),
    pytest.param(chapter_text(("9.1", MARKED.replace("```python", "```py"))), [], id="not-the-notebook-fence-language"),
    pytest.param(chapter_text(("9.1", block("import torch", "print(1)"))), [], id="ordinary-fence"),
    pytest.param(chapter_text(("9.1", marker_line(["pack"]) + "\n\n" + PACK)), [], id="not-inside-a-fence"),
]


@pytest.mark.parametrize(("chapter", "expected"), BLOCK_LOCATIONS)
def test_markers_are_found_wherever_the_notebook_builder_finds_python_blocks(tmp_path, chapter, expected):
    write(tmp_path, "course/chapters/09.md", chapter)
    assert [(m.fence.section, m.names[0]) for m in collect_markers(tmp_path)] == expected


def test_new_fences_sections_and_chapters_are_collected(tmp_path):
    make_repo(tmp_path, ("9.1", excerpt(["pack"], PACK)))
    assert [(m.fence.page, m.fence.section, m.names) for m in collect_markers(tmp_path)] == [
        ("course/chapters/09.md", "9.1", ("pack",))
    ]
    chapter = tmp_path / "course/chapters/09.md"
    chapter.write_text(
        chapter_text(
            ("9.1", excerpt(["pack"], PACK) + "\n\n" + excerpt(["unpack"], UNPACK)),
            ("9.2", excerpt(["Counter"], COUNTER)),
        ),
        encoding="utf-8",
        newline="\n",
    )
    write(tmp_path, "course/chapters/10.md", chapter_text(("10.1", excerpt(["fetch"], FETCH))))
    write(tmp_path, "course/chapters/0C.md", chapter_text(("C.1", excerpt(["cached"], CACHED))))
    found = [(m.fence.page, m.fence.section, m.fence.number, m.names) for m in collect_markers(tmp_path)]
    assert found == [
        ("course/chapters/09.md", "9.1", 1, ("pack",)),
        ("course/chapters/09.md", "9.1", 2, ("unpack",)),
        ("course/chapters/09.md", "9.2", 1, ("Counter",)),
        ("course/chapters/0C.md", "C.1", 1, ("cached",)),
        ("course/chapters/10.md", "10.1", 1, ("fetch",)),
    ]
    assert check_course(tmp_path) == []


# ---- 原樣節錄 ------------------------------------------------------------------------------------------------

GOOD_EXCERPTS = [
    pytest.param(["pack"], [PACK], id="function"),
    pytest.param(["pack", "unpack"], [PACK, UNPACK], id="two-adjacent-functions"),
    pytest.param(["unpack", "pack"], [UNPACK, PACK], id="names-in-the-listed-order"),
    pytest.param(["pack", "Counter"], [PACK, COUNTER], id="function-and-class-with-the-blank-lines-between"),
    pytest.param(["Counter"], [COUNTER], id="class-with-methods"),
    pytest.param(["cached"], [CACHED], id="decorator-belongs-to-the-definition"),
    pytest.param(["fetch"], [FETCH], id="async-function"),
    pytest.param(["outer"], [OUTER], id="nested-function-inside-the-excerpt"),
]


@pytest.mark.parametrize(("names", "definitions"), GOOD_EXCERPTS)
def test_excerpt_equal_to_the_library_source_passes(tmp_path, names, definitions):
    make_repo(tmp_path, ("9.1", excerpt(names, *definitions)))
    assert check_course(tmp_path) == []


def test_blank_lines_around_the_block_are_not_part_of_the_excerpt(tmp_path):
    padded = "```python\n\n" + marker_line(["pack"]) + "\n" + PACK + "\n\n```"
    make_repo(tmp_path, ("9.1", padded))
    assert check_course(tmp_path) == []


PACK_AND_UNPACK = PACK.rstrip("\n") + "\n\n\n" + UNPACK.rstrip("\n")
BAD_EXCERPTS = [
    pytest.param(["pack"], PACK.replace("* 16", "* 17"), id="one-character-changed"),
    pytest.param(["pack"], PACK.replace("    return", "  return"), id="indentation-changed"),
    pytest.param(["pack"], PACK.replace("def pack(pair):", "def pack(pair):  "), id="trailing-spaces-added"),
    pytest.param(
        ["pack"], PACK.replace('    """把兩個 0 到 15 的數字併成一個位元組。"""\n', ""), id="docstring-removed"
    ),
    pytest.param(["pack"], "# 一行說明\n" + PACK, id="comment-before-the-definition"),
    pytest.param(["pack"], "\n" + PACK, id="blank-line-after-the-marker"),
    pytest.param(["pack"], PACK + "\nprint(pack((1, 2)))\n", id="extra-code-after-the-excerpt"),
    pytest.param(["pack", "unpack"], PACK.rstrip("\n") + "\n\n" + UNPACK, id="one-blank-line-between-names"),
    pytest.param(["pack", "unpack"], PACK.rstrip("\n") + "\n\n\n\n" + UNPACK, id="three-blank-lines-between-names"),
    pytest.param(["pack", "unpack"], UNPACK.rstrip("\n") + "\n\n\n" + PACK, id="names-in-the-wrong-order"),
    pytest.param(["pack", "unpack"], PACK, id="second-name-missing"),
    pytest.param(["pack"], PACK_AND_UNPACK + "\n", id="unlisted-definition-added"),
    pytest.param(["Counter"], COUNTER.replace("self.count += amount", "self.count += 1"), id="method-body-changed"),
    pytest.param(["cached"], CACHED.replace("@functools.lru_cache(maxsize=None)\n", ""), id="decorator-removed"),
]


@pytest.mark.parametrize(("names", "pasted"), BAD_EXCERPTS)
def test_excerpt_that_is_not_verbatim_is_rejected_with_a_diff(tmp_path, names, pasted):
    make_repo(tmp_path, ("9.1", block(marker_line(names), pasted.rstrip("\n"))))
    problems = check_course(tmp_path)
    assert len(problems) == 1
    assert "9.1" in problems[0] and marker_line(names) in problems[0]
    assert "@@" in problems[0] and "函式庫原始碼" in problems[0]


def test_changing_one_character_of_the_library_breaks_the_excerpt(tmp_path):
    make_repo(tmp_path, ("9.1", excerpt(["pack", "unpack"], PACK, UNPACK)))
    assert check_course(tmp_path) == []
    library = tmp_path / LIBRARY_FILE
    library.write_text(
        library.read_text(encoding="utf-8").replace("divmod(value, 16)", "divmod(value, 17)"), encoding="utf-8"
    )
    problems = check_course(tmp_path)
    assert len(problems) == 1
    assert "9.1" in problems[0] and marker_line(["pack", "unpack"]) in problems[0]
    assert "-    return divmod(value, 17)" in problems[0]
    assert "+    return divmod(value, 16)" in problems[0]


def test_the_report_names_the_section_the_marker_and_every_bad_excerpt(tmp_path):
    good, bad = excerpt(["pack"], PACK), excerpt(["unpack"], UNPACK.replace("16", "8"))
    make_repo(tmp_path, ("9.1", good), ("9.2", bad), ("9.3", bad.replace("unpack", "Counter")))
    heads = [problem.splitlines()[0] for problem in check_course(tmp_path)]
    assert len(heads) == 2 and "9.2" in heads[0] and "9.3" in heads[1]


# ---- 名稱必須是那個檔案自己的頂層定義 -----------------------------------------------------------------------------

REEXPORT = LIBRARY + "\n\nfrom tiny_perceptron.other import helper\nalias = pack\nvalue = 3\n"
UNRESOLVED_NAMES = [
    pytest.param("helper", REEXPORT, id="re-exported-from-another-module"),
    pytest.param("alias", REEXPORT, id="assigned-from-a-definition"),
    pytest.param("value", REEXPORT, id="plain-variable"),
    pytest.param("add", LIBRARY, id="method"),
    pytest.param("inner", LIBRARY, id="nested-function"),
    pytest.param("torch", LIBRARY + "\nimport torch\n", id="imported-module"),
    pytest.param("pack_int4", LIBRARY, id="not-defined-at-all"),
]


@pytest.mark.parametrize(("name", "library"), UNRESOLVED_NAMES)
def test_name_must_be_defined_at_the_top_level_of_that_file(tmp_path, name, library):
    make_repo(tmp_path, ("9.1", block(marker_line([name]), "def stand_in():", "    pass")), library=library)
    problems = check_course(tmp_path)
    assert len(problems) == 1
    assert "沒有頂層的 def／class" in problems[0] and name in problems[0]


def test_the_defining_module_is_accepted_and_the_reexporting_module_is_not(tmp_path):
    other = block(marker_line(["helper"], path="tiny_perceptron/other.py"), OTHER.rstrip("\n"))
    make_repo(tmp_path, ("9.1", other), library=REEXPORT)
    assert check_course(tmp_path) == []
    reexported = block(marker_line(["helper"]), OTHER.rstrip("\n"))
    make_repo(tmp_path, ("9.1", reexported), library=REEXPORT)
    assert len(check_course(tmp_path)) == 1


def test_a_close_name_is_suggested_for_a_typo(tmp_path):
    make_repo(tmp_path, ("9.1", block(marker_line(["pak"]), PACK.rstrip("\n"))))
    assert "相近的頂層名稱：pack" in check_course(tmp_path)[0]


def test_missing_and_unparsable_library_files_are_reported(tmp_path):
    make_repo(tmp_path, ("9.1", block(marker_line(["pack"], path="tiny_perceptron/absent.py"), "x = 1")))
    assert "找不到 tiny_perceptron/absent.py" in check_course(tmp_path)[0]
    make_repo(tmp_path, ("9.1", excerpt(["pack"], PACK)), library="def broken(:\n")
    assert "無法解析" in check_course(tmp_path)[0]


# ---- 精簡版 --------------------------------------------------------------------------------------------------

CONDENSED_PACK = block(
    "# 精簡自 tiny_perceptron/quantization.py：pack（省略輸入範圍檢查）",
    "def pack(pair):",
    "    return pair[0] * 16 + pair[1]",
)
CHECK_PACK = block(
    "from tiny_perceptron.quantization import pack as library_pack",
    "",
    "assert pack((3, 4)) == library_pack((3, 4))",
)
IMPORT_ONLY = block("from tiny_perceptron.quantization import pack as library_pack", "print(library_pack((3, 4)))")
ASSERT_ONLY = block("assert pack((3, 4)) == 52")
ONE_BLOCK = block(
    "# 精簡自 tiny_perceptron/quantization.py：pack（省略輸入範圍檢查）",
    "from tiny_perceptron.quantization import pack as library_pack",
    "",
    "",
    "def pack(pair):",
    "    return pair[0] * 16 + pair[1]",
    "",
    "",
    "assert pack((3, 4)) == library_pack((3, 4))",
)
CONDENSED_SECTIONS = [
    pytest.param([("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK)], True, id="import-and-assert-in-a-later-block"),
    pytest.param([("9.1", ONE_BLOCK)], True, id="everything-in-one-block"),
    pytest.param([("9.1", CONDENSED_PACK + "\n\n### 核對\n\n" + CHECK_PACK)], True, id="check-under-a-subheading"),
    pytest.param([("9.1", CHECK_PACK + "\n\n" + CONDENSED_PACK)], True, id="check-before-the-condensed-block"),
    pytest.param([("9.1", CONDENSED_PACK + "\n\n" + IMPORT_ONLY)], "assert", id="no-assert"),
    pytest.param([("9.1", CONDENSED_PACK + "\n\n" + ASSERT_ONLY)], "import", id="no-import"),
    pytest.param([("9.1", CONDENSED_PACK)], "both", id="neither-import-nor-assert"),
    pytest.param(
        [("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK.replace("import pack as", "import unpack as"))],
        "import",
        id="import-of-another-name",
    ),
    pytest.param(
        [("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK.replace("quantization", "other"))],
        "import",
        id="import-from-another-module",
    ),
    pytest.param(
        [("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK.replace("quantization", "quantization.pack"))],
        "import",
        id="import-from-a-submodule-path",
    ),
    pytest.param(
        [
            (
                "9.1",
                CONDENSED_PACK
                + "\n\n"
                + CHECK_PACK.replace("from ", "import ").replace(" import pack as library_pack", ""),
            )
        ],
        "import",
        id="module-import-instead-of-name-import",
    ),
    pytest.param(
        [
            (
                "9.1",
                CONDENSED_PACK
                + "\n\n"
                + block(
                    "# from tiny_perceptron.quantization import pack",
                    "text = 'from tiny_perceptron.quantization import pack'",
                    "assert pack((3, 4)) == 52",
                ),
            )
        ],
        "import",
        id="import-only-in-a-comment-or-string",
    ),
    pytest.param(
        [
            (
                "9.1",
                CONDENSED_PACK
                + "\n\n"
                + block(
                    "from tiny_perceptron.quantization import pack as library_pack",
                    "# assert pack((3, 4)) == library_pack((3, 4))",
                    "note = 'assert'",
                ),
            )
        ],
        "assert",
        id="assert-only-in-a-comment-or-string",
    ),
    pytest.param(
        [
            (
                "9.1",
                CONDENSED_PACK
                + "\n\n"
                + block(
                    "from tiny_perceptron.quantization import pack as library_pack",
                    "",
                    "",
                    "def check():",
                    "    assert pack((3, 4)) == library_pack((3, 4))",
                ),
            )
        ],
        "assert",
        id="assert-only-inside-a-function-body",
    ),
    pytest.param(
        [("9.1", CONDENSED_PACK), ("9.2", CHECK_PACK)], "both", id="import-and-assert-belong-to-another-section"
    ),
    pytest.param(
        [("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK + "\n\n" + block("def broken(:"))],
        "unparsable",
        id="another-block-of-the-section-does-not-parse",
    ),
]


@pytest.mark.parametrize(("sections", "expected"), CONDENSED_SECTIONS)
def test_condensed_version_needs_the_library_import_and_an_assert_in_the_same_section(tmp_path, sections, expected):
    make_repo(tmp_path, *sections)
    problems = check_course(tmp_path)
    if expected is True:
        assert problems == []
        return
    assert len(problems) == 1 and "9.1" in problems[0]
    missing_import = "from tiny_perceptron.quantization import pack`" in problems[0]
    missing_assert = "沒有會執行到的 `assert`" in problems[0]
    assert (missing_import, missing_assert) == {
        "import": (True, False),
        "assert": (False, True),
        "both": (True, True),
        "unparsable": (False, False),
    }[expected]
    assert ("無法解析" in problems[0]) is (expected == "unparsable")


def test_condensed_version_without_assert_is_rejected_and_adding_it_fixes_it(tmp_path):
    make_repo(tmp_path, ("9.1", CONDENSED_PACK + "\n\n" + IMPORT_ONLY))
    assert len(check_course(tmp_path)) == 1
    make_repo(tmp_path, ("9.1", CONDENSED_PACK + "\n\n" + IMPORT_ONLY + "\n\n" + ASSERT_ONLY))
    assert check_course(tmp_path) == []


def test_condensed_name_must_exist_in_the_library(tmp_path):
    wrong = CONDENSED_PACK.replace("：pack（", "：pak（")
    make_repo(tmp_path, ("9.1", wrong + "\n\n" + CHECK_PACK))
    problems = check_course(tmp_path)
    assert len(problems) == 1 and "沒有頂層的 def／class" in problems[0]


@pytest.mark.parametrize(
    ("path", "module"),
    [
        ("tiny_perceptron/quantization.py", "tiny_perceptron.quantization"),
        ("tiny_perceptron/selftrained/model.py", "tiny_perceptron.selftrained.model"),
        ("tiny_perceptron/__init__.py", "tiny_perceptron"),
        ("tiny_perceptron/lesson_products/__init__.py", "tiny_perceptron.lesson_products"),
    ],
)
def test_import_name_of_a_library_path(path, module):
    assert module_name(path) == module


def test_condensed_import_must_name_the_module_of_a_subpackage_file(tmp_path):
    write(tmp_path, "tiny_perceptron/selftrained/tools.py", "def parse(text):\n    return text.split()\n")
    marker = "# 精簡自 tiny_perceptron/selftrained/tools.py：parse（省略錯誤處理）"
    simplified = block(marker, "def parse(text):", "    return text.split()")

    def check(import_line):
        library = block(import_line, "assert parse('a b') == library_parse('a b')")
        write(tmp_path, "course/chapters/09.md", chapter_text(("9.1", simplified + "\n\n" + library)))
        return check_course(tmp_path)

    assert check("from tiny_perceptron.selftrained.tools import parse as library_parse") == []
    assert len(check("from tiny_perceptron.tools import parse as library_parse")) == 1
    assert len(check("from tiny_perceptron.selftrained import parse as library_parse")) == 1


# ---- 與 build_course.py 和 inspect.getsource 對照 --------------------------------------------------------------


def test_scanner_uses_the_patterns_of_the_notebook_builder():
    assert SECTION_HEADING in inspect.getsource(build_course.build)
    notebook_source = inspect.getsource(build_course.notebook)
    assert PYTHON_BLOCK in notebook_source and "re.S" in notebook_source, (
        "scripts/build_course.py 切分程式區塊的正則改了；請同步本檔的 PYTHON_BLOCK，否則會漏掃 Notebook 會執行的區塊"
    )


def library_files():
    return sorted((ROOT / "tiny_perceptron").rglob("*.py"))


@pytest.mark.parametrize("path", library_files(), ids=lambda path: path.relative_to(ROOT).as_posix())
def test_extracted_source_is_what_inspect_getsource_returns(path):
    relative = path.relative_to(ROOT).as_posix()
    try:
        module = importlib.import_module(module_name(relative))
    except ImportError as error:
        pytest.skip(f"{relative} 需要的套件沒有安裝：{error}")
    for name, source in top_level_sources(path.read_text(encoding="utf-8")).items():
        assert source == inspect.getsource(getattr(module, name)), f"{relative}：{name}"


# ---- 真實教材 ------------------------------------------------------------------------------------------------


def test_course_excerpts_exist_and_match_the_library():
    markers = collect_markers(ROOT)
    assert markers, "course/chapters 的 python 區塊裡找不到任何「# 節錄自」或「# 精簡自」標記，沒有東西可以比對。"
    problems = check_course(ROOT)
    if problems:
        pytest.fail(f"{len(problems)} 個標記有問題：\n\n" + "\n\n".join(problems), pytrace=False)
