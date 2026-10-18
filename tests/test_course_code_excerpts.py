"""正文裡標記為「節錄」或「精簡」的程式，必須和 tiny_perceptron 的原始碼保持同步。

標記是 python 區塊的第一行：

- ``# 節錄自 tiny_perceptron/<路徑>.py：<名稱>[、<名稱>…]``：其後逐字等於這些頂層 def／class 的原始碼
  （``inspect.getsource``），依標記列出的順序以兩個空行相隔，不必在原檔相鄰；區塊裡不放其他程式。
- ``# 精簡自 tiny_perceptron/<路徑>.py：<名稱>（<省略了什麼>）``：作者重寫的精簡版。同一節要用
  ``from tiny_perceptron.<模組> import <名稱> as <別名>`` 取得函式庫版本，並在精簡版之後用 ``assert``
  核對兩者；這裡用靜態規則檢查那個 assert，規則與涵蓋範圍寫在 condensed_problems。

冒號、頓號、括號都用全形。第一行以「# 節錄」或「# 精簡」開頭卻不符合格式，一律是錯誤，避免寫錯的標記被當成
一般程式放過。只有某一節裡、會變成 Notebook 程式格的區塊第一行才算標記；章檔 fence 裡看起來像標記的其他行
也一律是錯誤，見 stray_marker_problems。
解析與檢查函式都接受根目錄參數，所以能在假的 repo 上驗證。
"""

import ast
import difflib
import importlib
import inspect
import io
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts import build_course

ROOT = Path(__file__).resolve().parents[1]

# 切節與取程式區塊的兩個正則照抄 scripts/build_course.py；第一個節標題之前的導言也和它一樣不產生程式格。
# test_scanner_uses_the_patterns_of_the_notebook_builder 核對這兩個正則仍在 build_course 的原始碼裡；
# test_scanner_finds_the_code_cells_that_the_notebook_builder_makes 在假的 repo 上核對兩邊取出相同的程式格。
SECTION_HEADING = r"^## ([\dABC]+\.\d+) (.+)$"
PYTHON_BLOCK = r"```python\n(?P<python>.*?)```|!\[(?P<alt>[^\]]*)\]\(\.\./figures/(?P<figure>[^/)]+)\.svg\)"
BLOCKS = re.compile(PYTHON_BLOCK, re.S)

# 程式格第一行看起來像標記的，一律要通過下面的嚴格格式。
LOOKS_LIKE_MARKER = re.compile(r"#\s*(?:節錄|精簡)")
# fence 裡任何一行看起來像標記：「# 節錄自」「# 精簡自」，或少了「自」、多了冒號卻接著 tiny_perceptron/ 路徑。
STRAY_MARKER = re.compile(r"\s*#\s*(?:節錄|精簡)(?:自|\s*[:：]?\s*tiny_perceptron/)")
FENCE_LINE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
# build_course.py 只把「```python」之後直接換行的開頭當成程式格；作者的寫法是開頭一行（縮排除外）剛好是它。
NOTEBOOK_FENCE = "```python"
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
    section: str  # 節號
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
    """依 build_course.py 的方式切節，取出每個會變成 Notebook 程式格的 python 區塊。

    build_course.build 只為節標題之後的內容產生 Notebook，第一個節標題之前的導言不取。
    """
    fences = []
    for chapter in sorted((root / "course/chapters").glob("*.md")):
        parts = re.split(SECTION_HEADING, chapter.read_text(encoding="utf-8"), flags=re.M)
        page = chapter.relative_to(root).as_posix()
        for i in range(1, len(parts), 3):
            section, body = parts[i], parts[i + 2]
            codes = [match["python"] for match in BLOCKS.finditer(body) if match["python"] is not None]
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


def assigned_names(target):
    """指派目標綁定的名稱；``x.weight = …``、``x[0] = …`` 改的是 x。"""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return assigned_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(assigned_names(element) for element in target.elts))
    if isinstance(target, (ast.Attribute, ast.Subscript)):
        return assigned_names(target.value)
    return set()


def used_names(node):
    """節點裡出現的全部名稱；推導式裡的名稱，包含它迭代的對象，也都算進來。"""
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def assignment_sources(trees):
    """{名稱: 指派給它的值用到的名稱}，取自同一節會執行到的簡單指派。

    算進來的有 ``=``（兩邊都是等長的 tuple／list 時逐一配對，否則左邊每個名稱都依賴右邊全部名稱）、
    ``+=`` 這類增量指派，以及 ``for`` 與 ``with … as`` 的目標。推導式不必另外處理：它迭代的對象就寫在
    同一個運算式裡，used_names 已經算進來。def 與 lambda 裡面的指派要等被呼叫才執行，不算。
    """
    sources = {}

    def bind(target, names):
        for name in assigned_names(target):
            sources.setdefault(name, set()).update(names)

    def pair(target, value):
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
            and not any(isinstance(element, ast.Starred) for element in [*target.elts, *value.elts])
        ):
            for part, part_value in zip(target.elts, value.elts, strict=True):
                pair(part, part_value)
        else:
            bind(target, used_names(value))

    for tree in trees:
        for node in executed_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    pair(target, node.value)
            elif isinstance(node, ast.AugAssign):
                bind(node.target, used_names(node.value))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                bind(node.target, used_names(node.iter))
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        bind(item.optional_vars, used_names(item.context_expr))
    return sources


def traced_names(expression, sources):
    """運算式用到的名稱，加上沿 sources 一路往回追得到的名稱（遞移閉包）。"""
    seen, pending = set(), list(used_names(expression))
    while pending:
        name = pending.pop()
        if name not in seen:
            seen.add(name)
            pending.extend(sources.get(name, ()))
    return seen


def condensed_problems(marker, siblings):
    """精簡版的同一節（siblings）裡，至少要有一個會執行到、符合下列全部條件的 assert：

    - 在精簡版的定義之後：位於同一節較後面的 python 區塊，或標記區塊裡、精簡版定義結束之後的行。
    - 運算式用到的名稱，沿同一節的簡單指派往回追（assignment_sources），同時追得到精簡版的名稱
      與函式庫的別名。
    - 函式庫別名來自 ``from <模組> import <名稱> as <別名>``，而且別名和精簡版的名稱不同。同一節有會執行到、
      沒換名的 ``from <模組> import <名稱>`` 也是錯誤，不論它在精簡版的定義之前或之後。
    另外，標記區塊本身要用頂層 def／class 定義這個名稱。

    驗證：assert 是否在精簡版定義之後、它的運算式經簡單指派追得到哪些名稱、函式庫 import 有沒有用 as 換名、
    標記區塊有沒有這個名稱的頂層定義。
    不涵蓋：運算式是否真的在比較兩邊（例如 ``same = library is not None and logits.shape == (1, 3, 20)`` 之後
    ``assert same`` 會通過）、精簡版的名稱在同一節被指派或從別處 import 重新綁定；比得對不對，要等發布流程執行
    Notebook 時由 assert 本身核對。
    """
    module, name = module_name(marker.path), marker.names[0]
    trees, reasons = {}, []
    for sibling in siblings:
        try:
            trees[sibling.number] = ast.parse(sibling.code)
        except SyntaxError as error:
            reasons.append(f"同一節第 {sibling.number} 個 python 區塊無法解析，查不出 import 與 assert：{error}")
    own = trees.get(marker.fence.number)
    definitions = [node for node in (own.body if own else []) if isinstance(node, DEFINITIONS) and node.name == name]
    if own is not None and not definitions:
        reasons.append(
            f"標記區塊沒有用頂層 def／class 定義 `{name}`；精簡版要寫在標記所在的區塊裡，後面的 assert 才比得到它。"
        )
    # 精簡版定義結束的位置（第幾個 python 區塊, 行號）；找不到定義時，只算較後面區塊裡的 assert。
    defined = (marker.fence.number, definitions[-1].end_lineno if definitions else math.inf)

    library_import = f"from {module} import {name}"
    aliases, unrenamed = set(), False
    for tree in trees.values():
        for node in executed_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == module:
                for alias in node.names:
                    if alias.name == name and alias.asname not in (None, name):
                        aliases.add(alias.asname)
                    elif alias.name == name:
                        unrenamed = True
    if unrenamed:
        reasons.append(
            f"同一節的 `{library_import}` 沒有用 `as` 換成別的名稱；請改成 `{library_import} as <別名>`，"
            f"別名和 `{name}` 不同。"
        )
    elif not aliases:
        reasons.append(
            f"同一節沒有 `{library_import} as <別名>`（別名要和精簡版的 `{name}` 不同）；沒有函式庫版本，就無從核對。"
        )

    sources = assignment_sources(trees.values())
    matched, before, unmatched = False, 0, []
    for number, tree in sorted(trees.items()):
        for node in executed_nodes(tree):
            if not isinstance(node, ast.Assert):
                continue
            if (number, node.lineno) <= defined:
                before += 1
                continue
            names = traced_names(node.test, sources)
            if name in names and names & aliases:
                matched = True
            else:
                unmatched.append((number, node, names))
    if not matched and not unmatched:
        reason = (
            f"同一節在精簡版 `{name}` 的定義之後，沒有會執行到的 `assert`（寫在 def 裡面的不算）；"
            "請用它核對精簡版與函式庫版本算出相同結果。"
        )
        if before:
            reason += f"定義之前的 {before} 個 assert 不算：執行到那裡時，精簡版還沒有定義。"
        reasons.append(reason)
    elif not matched and aliases:
        labels = "、".join(f"`{alias}`" for alias in sorted(aliases))
        lines = [
            f"精簡版 `{name}` 定義之後的 assert，沒有一個同時追溯到精簡版 `{name}` 與函式庫別名（{labels}）；"
            "請讓 assert 比較兩個版本在本節例子上算出的結果。"
        ]
        for number, node, names in unmatched:
            condensed = "追得到" if name in names else "追不到"
            library = "追得到" if names & aliases else "追不到"
            lines.append(
                f"- 第 {number} 個 python 區塊的 `{ast.unparse(node)}`：{condensed}精簡版，{library}函式庫別名"
            )
        reasons.append("\n".join(lines))
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


def fenced_blocks(text):
    """[(開頭一行, [(行號, 內容)])]：章檔裡每個 fenced 區塊的開頭一行與區塊內各行；沒有關上的區塊算到檔尾。"""
    blocks, opened = [], None  # opened：(圍欄字元, 圍欄長度)
    for number, line in enumerate(text.split("\n"), 1):
        fence = FENCE_LINE.match(line)
        if opened is None:
            if fence:
                opened = (fence[1][0], len(fence[1]))
                blocks.append((line, []))
        elif fence and fence[1][0] == opened[0] and len(fence[1]) >= opened[1] and not fence[2].strip():
            opened = None
        else:
            blocks[-1][1].append((number, line))
    return blocks


def stray_marker_problems(root):
    """章檔 fence 裡看起來像標記、卻不在該在位置的字樣；回傳問題清單。

    標記只該出現在一個位置：某一節裡、開頭一行（縮排除外）剛好是 ```python 的 fence 的第一個非空白行，而且從那一行
    的行首（縮排除外）開始（parse_marker 讀的正是這一行）。節的歸屬和 build_course.py 一樣，只看前面最近的節標題。

    驗證：章檔每個 fence（任何語言、縮排或放在 <details> 裡都算）的每一行，任何位置出現 STRAY_MARKER 的字樣都算，
    包括行尾註解與字串字面值裡的；不在上述位置就報，並寫出不符合的每一條：在導言裡、fence 開頭一行不對、
    不是區塊第一個非空白行、不從行首開始。
    不涵蓋：章檔以外的頁面、fence 外的正文與 inline code，以及 STRAY_MARKER 認不出的寫法（例如沒有「自」、
    也沒接 tiny_perceptron/ 路徑的「# 節錄的重點」）。
    """
    problems = []
    for chapter in sorted((root / "course/chapters").glob("*.md")):
        text = chapter.read_text(encoding="utf-8")
        page = chapter.relative_to(root).as_posix()
        headings = [
            (number, match[1])
            for number, line in enumerate(text.split("\n"), 1)
            if (match := re.match(SECTION_HEADING, line))
        ]
        for opener, lines in fenced_blocks(text):
            first = next((number for number, line in lines if line.strip()), None)
            for number, line in lines:
                found = STRAY_MARKER.search(line)
                if not found:
                    continue
                section = next((lesson for start, lesson in reversed(headings) if start < number), None)
                reasons = []
                if section is None:
                    reasons.append(
                        "這一行在第一個 `## X.Y` 節標題之前的導言裡；build_course.py 不為導言產生 Notebook，"
                        "這個區塊不會執行，標記也就無從核對。請把它移進某一節。"
                    )
                # 開頭一行以 ```python 結尾、反引號卻不只三個（例如 ````python）時，build_course.py 的正則仍在裡面
                # 找到 ```python 開頭，照樣產生程式格。
                longer_fence = opener.lstrip() != NOTEBOOK_FENCE and opener.endswith(NOTEBOOK_FENCE)
                if opener.lstrip() != NOTEBOOK_FENCE:
                    reasons.append(
                        f"所在 fence 的開頭一行是 {opener.lstrip()!r}；標記所在的 fence，開頭一行（縮排除外）"
                        "必須剛好是 ```python。"
                        + (
                            ""
                            if longer_fence
                            else 'build_course.py 不會把 ```py、```python title="…" '
                            "或 python 後多了空白的區塊變成 Notebook 程式格，標記在那裡無從核對。"
                        )
                    )
                if number != first:
                    reasons.append("標記要寫在區塊第一行；寫在其他行，不會被當成標記核對。")
                if line[: found.start()].strip():
                    reasons.append(
                        "標記要從行首（縮排除外）的 `#` 開始、獨占一行；寫成行尾註解或接在其他文字後面，不會被當成標記核對。"
                    )
                if longer_fence and len(reasons) == 1:
                    reasons[0] += "build_course.py 仍會把這個區塊變成 Notebook 程式格，這個標記仍會被核對。"
                if reasons:
                    head = f"{page} 第 {number} 行（{section or '導言'}）\n  標記行：{line.strip()}"
                    problems.append("\n".join([head, *(textwrap.indent(reason, "  ") for reason in reasons)]))
    return problems


def check_course(root):
    """檢查 root 底下全部章檔的全部標記，以及不在核對位置的標記，回傳問題清單。"""
    fences = read_fences(root)
    problems = []
    for fence in fences:
        if marker := parse_marker(fence):
            siblings = [other for other in fences if (other.page, other.section) == (fence.page, fence.section)]
            problems.extend(check_marker(root, marker, siblings))
    return problems + stray_marker_problems(root)


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
# 假的模型檔：精簡版的規則用它寫出正文常見的「建立兩個版本、載入同一組權重、比較輸出」。只解析、不執行。
MODEL = '''"""假的模型。"""

from torch import nn


class TinyLM(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output = nn.Linear(width, 3)

    def forward(self, x, cache=None):
        return self.output(x)
'''


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
    write(root, "tiny_perceptron/model.py", MODEL)
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
    # 導言不產生 Notebook，不算程式格；裡面的標記由 stray_marker_problems 回報。
    pytest.param(
        "# 第九章\n\n" + MARKED + "\n\n## 9.1 一個問題\n\n文字。\n", [], id="before-the-first-section-heading"
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
    # build_course.py 的正則在 ````python 裡也找得到 ```python 開頭，照常產生程式格；stray_marker_problems 另外回報它。
    pytest.param(chapter_text(("9.1", MARKED.replace("```", "````"))), [("9.1", "pack")], id="four-backtick-fence"),
]


@pytest.mark.parametrize(("chapter", "expected"), BLOCK_LOCATIONS)
def test_markers_are_found_wherever_the_notebook_builder_finds_python_blocks(tmp_path, chapter, expected):
    write(tmp_path, "course/chapters/09.md", chapter)
    assert [(m.fence.section, m.names[0]) for m in collect_markers(tmp_path)] == expected


# ---- 不在核對位置的標記：反例表與正例表 ----------------------------------------------------------------------------

INTRODUCTION = "# 第九章\n\n{}\n\n## 9.1 一個問題\n\n文字。\n"
STRAY_REASONS = {
    "intro": "導言裡",
    "fence": "必須剛好是 ```python",
    "line": "標記要寫在區塊第一行",
    "trailing": "寫成行尾註解",
    "checked": "這個標記仍會被核對",
}
STRAY_MARKERS = [
    pytest.param(INTRODUCTION.format(MARKED), {"intro"}, id="before-the-first-section-heading"),
    pytest.param(
        INTRODUCTION.format(MARKED.replace("```python", "```py")),
        {"intro", "fence"},
        id="py-fence-before-the-first-section-heading",
    ),
    pytest.param(chapter_text(("9.1", MARKED.replace("```python", "```py"))), {"fence"}, id="py-fence"),
    pytest.param(
        chapter_text(("9.1", MARKED.replace("```python", '```python title="pack.py"'))),
        {"fence"},
        id="fence-with-a-title",
    ),
    pytest.param(
        chapter_text(("9.1", MARKED.replace("```python", "```python "))), {"fence"}, id="space-after-the-language"
    ),
    pytest.param(chapter_text(("9.1", MARKED.replace("```python", "```Python"))), {"fence"}, id="capitalized-language"),
    pytest.param(chapter_text(("9.1", MARKED.replace("```python", "```bash"))), {"fence"}, id="bash-fence"),
    pytest.param(chapter_text(("9.1", MARKED.replace("```", "~~~"))), {"fence"}, id="tilde-fence"),
    pytest.param(
        chapter_text(("9.1", "- 一個項目\n\n  " + MARKED.replace("```python", "```py").replace("\n", "\n  ") + "\n")),
        {"fence"},
        id="indented-py-fence-in-a-list",
    ),
    pytest.param(
        chapter_text(("9.1", block("import math", "", marker_line(["pack"]), PACK.rstrip("\n")))),
        {"line"},
        id="marker-after-the-first-line",
    ),
    pytest.param(
        chapter_text(
            ("9.1", block(marker_line(["pack"]), PACK.rstrip("\n"), "", "", marker_line(["unpack"]), UNPACK.rstrip()))
        ),
        {"line"},
        id="second-marker-in-the-same-block",
    ),
    pytest.param(
        chapter_text(("9.1", block("x = 1", "#節錄自 tiny_perceptron/quantization.py：pack"))),
        {"line"},
        id="missing-space-after-the-hash-on-a-later-line",
    ),
    pytest.param(
        chapter_text(("9.1", block("x = 1", "# 節錄: tiny_perceptron/quantization.py：pack"))),
        {"line"},
        id="colon-instead-of-from-on-a-later-line",
    ),
    pytest.param(
        chapter_text(
            ("9.1", block("def f():", "    # 精簡 tiny_perceptron/model.py：TinyLM（省略 cache）", "    pass"))
        ),
        {"line"},
        id="indented-lookalike-on-a-later-line",
    ),
    pytest.param(
        chapter_text(
            (
                "9.1",
                block(
                    "def pack(pair):  # 精簡自 tiny_perceptron/quantization.py：pack（省略輸入範圍檢查）",
                    "    return pair[0] * 16 + pair[1]",
                ),
            )
        ),
        {"trailing"},
        id="marker-as-a-trailing-comment-on-the-first-line",
    ),
    pytest.param(
        chapter_text(("9.1", block("x = 1", "y = 2  # 節錄自 tiny_perceptron/quantization.py：pack"))),
        {"line", "trailing"},
        id="marker-as-a-trailing-comment-on-a-later-line",
    ),
    pytest.param(
        chapter_text(("9.1", block("import math", 'note = "# 節錄自 tiny_perceptron/quantization.py：pack"'))),
        {"line", "trailing"},
        id="marker-inside-a-string-literal",
    ),
    pytest.param(
        chapter_text(
            (
                "9.1",
                block(
                    "## 精簡自 tiny_perceptron/quantization.py：pack（省略輸入範圍檢查）",
                    "def pack(pair):",
                    "    return pair[0] * 16 + pair[1]",
                ),
            )
        ),
        {"trailing"},
        id="one-hash-too-many-before-the-marker",
    ),
    pytest.param(
        chapter_text(
            (
                "9.1",
                block(
                    "# 先看精簡版 # 精簡自 tiny_perceptron/quantization.py：pack（省略輸入範圍檢查）",
                    "def pack(pair):",
                    "    return pair[0] * 16 + pair[1]",
                ),
            )
        ),
        {"trailing"},
        id="marker-after-another-comment-on-the-first-line",
    ),
    pytest.param(
        chapter_text(("9.1", block("import math", marker_line(["pack"])).replace("```", "````"))),
        {"fence", "line"},
        id="four-backtick-fence-with-the-marker-on-a-later-line",
    ),
    pytest.param(
        chapter_text(("9.1", MARKED.replace("```", "````").replace("````python", "````py"))),
        {"fence"},
        id="four-backtick-py-fence",
    ),
]


@pytest.mark.parametrize(("chapter", "expected"), STRAY_MARKERS)
def test_a_marker_that_no_notebook_code_cell_starts_with_is_reported(tmp_path, chapter, expected):
    write(tmp_path, "course/chapters/09.md", chapter)
    problems = stray_marker_problems(tmp_path)
    assert len(problems) == 1
    assert {kind for kind, text in STRAY_REASONS.items() if text in problems[0]} == expected
    assert problems[0] in check_course(tmp_path)


@pytest.mark.parametrize("fence", ["````", "`````"], ids=["four-backticks", "five-backticks"])
def test_a_marker_in_a_fence_with_more_than_three_backticks_is_checked_and_reported(tmp_path, fence):
    write(tmp_path, "course/chapters/09.md", chapter_text(("9.1", MARKED.replace("```", fence))))
    assert [(marker.fence.section, marker.names[0]) for marker in collect_markers(tmp_path)] == [("9.1", "pack")]
    [problem] = stray_marker_problems(tmp_path)
    assert {kind for kind, text in STRAY_REASONS.items() if text in problem} == {"fence", "checked"}
    assert "無從核對" not in problem


NOT_STRAY = [
    pytest.param(chapter_text(("9.1", MARKED)), id="marker-on-the-first-line"),
    pytest.param(chapter_text(("9.1", "```python\n\n" + MARKED.removeprefix("```python\n"))), id="blank-line-first"),
    pytest.param(
        chapter_text(("9.1", "- 一個項目\n\n  " + MARKED.replace("\n", "\n  ") + "\n")), id="indented-in-a-list"
    ),
    pytest.param(
        chapter_text(("9.1", f"<details>\n<summary>補充</summary>\n\n{MARKED}\n\n</details>")), id="inside-details"
    ),
    pytest.param(
        chapter_text(("9.1", block("x = 1", "# 精簡版和函式庫版本算出相同結果"))),
        id="comment-about-the-condensed-version",
    ),
    pytest.param(
        chapter_text(("9.1", block("x = 1", "# 節錄的重點在 tiny_perceptron/quantization.py"))),
        id="comment-that-names-a-library-file",
    ),
    pytest.param(
        chapter_text(("9.1", "```bash\n# 先讀 tiny_perceptron/quantization.py 的 pack\nls\n```")),
        id="shell-comment-that-names-a-library-file",
    ),
    pytest.param(chapter_text(("9.1", marker_line(["pack"]) + "\n\n" + PACK)), id="prose-outside-a-fence"),
    pytest.param(chapter_text(("9.1", f"標記寫成 `{marker_line(['pack'])}`。")), id="inline-code-in-prose"),
]


@pytest.mark.parametrize("chapter", NOT_STRAY)
def test_markers_in_place_and_ordinary_comments_are_not_reported_as_stray(tmp_path, chapter):
    write(tmp_path, "course/chapters/09.md", chapter)
    assert stray_marker_problems(tmp_path) == []


def test_the_report_of_a_stray_marker_names_the_page_line_and_section(tmp_path):
    chapter = chapter_text(("9.1", "文字。"), ("9.2", "文字。\n\n" + MARKED.replace("```python", "```py")))
    write(tmp_path, "course/chapters/09.md", chapter)
    line = chapter.split("\n").index(marker_line(["pack"])) + 1
    [problem] = stray_marker_problems(tmp_path)
    assert problem.splitlines()[:2] == [
        f"course/chapters/09.md 第 {line} 行（9.2）",
        f"  標記行：{marker_line(['pack'])}",
    ]
    assert "'```py'" in problem


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
    pytest.param(["Counter"], COUNTER.replace("        return self.count\n", ""), id="last-line-missing"),
    pytest.param(["Counter"], COUNTER.partition("\n\n    def add")[0], id="last-method-missing"),
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
CONDENSED_LM = block(
    "# 精簡自 tiny_perceptron/model.py：TinyLM（省略 cache）",
    "import torch",
    "from torch import nn",
    "",
    "",
    "class TinyLM(nn.Module):",
    "    def __init__(self, width):",
    "        super().__init__()",
    "        self.output = nn.Linear(width, 3)",
    "",
    "    def forward(self, x):",
    "        return self.output(x)",
)


def check_lm(*lines):
    """精簡版 TinyLM 加上核對區塊：建立兩個版本、載入同一組權重、算出精簡版的 logits，再接上 lines。"""
    return (
        CONDENSED_LM
        + "\n\n"
        + block(
            "from tiny_perceptron.model import TinyLM as LibraryTinyLM",
            "",
            "torch.manual_seed(42)",
            "model = TinyLM(8)",
            "library = LibraryTinyLM(8)",
            "library.load_state_dict(model.state_dict())",
            "x = torch.randn(1, 2, 8)",
            "logits = model(x)",
            *lines,
        )
    )


SAME = ("", "", "def same(ours, theirs):", "    return torch.allclose(ours(x), theirs(x))", "", "")
CONDENSED_REASONS = {
    "import": "沒有函式庫版本，就無從核對",
    "rebind": "沒有用 `as` 換成別的名稱",
    "definition": "沒有用頂層 def／class 定義",
    "no-assert": "沒有會執行到的 `assert`",
    "not-both": "沒有一個同時追溯到",
    "unparsable": "無法解析",
}
CONDENSED_SECTIONS = [
    pytest.param([("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK)], set(), id="import-and-assert-in-a-later-block"),
    pytest.param([("9.1", ONE_BLOCK)], set(), id="everything-in-one-block"),
    pytest.param([("9.1", CONDENSED_PACK + "\n\n### 核對\n\n" + CHECK_PACK)], set(), id="check-under-a-subheading"),
    pytest.param([("9.1", check_lm("assert torch.allclose(logits, library(x))"))], set(), id="compare-the-outputs"),
    pytest.param(
        [("9.1", check_lm("same = torch.allclose(logits, library(x))", "assert same and logits.shape == (1, 2, 3)"))],
        set(),
        id="compare-through-assigned-names",
    ),
    pytest.param(
        [
            (
                "9.1",
                check_lm("for p, q in zip(model.parameters(), library.parameters()):", "    assert torch.equal(p, q)"),
            )
        ],
        set(),
        id="loop-over-the-parameters",
    ),
    pytest.param(
        [("9.1", check_lm("assert all(torch.equal(p, q) for p, q in zip(model.parameters(), library.parameters()))"))],
        set(),
        id="comprehension-over-the-parameters",
    ),
    pytest.param([("9.1", check_lm(*SAME, "assert same(model, library)"))], set(), id="helper-function"),
    pytest.param(
        [("9.1", check_lm(*SAME, "ok = same(model, library)", "assert ok"))],
        set(),
        id="helper-function-result-in-a-name",
    ),
    pytest.param(
        [("9.1", check_lm("ours, theirs = model(x), library(x)", "assert torch.allclose(ours, theirs)"))],
        set(),
        id="tuple-unpacking",
    ),
    pytest.param(
        [
            (
                "9.1",
                check_lm(
                    "gap = 0.0",
                    "for p, q in zip(model.parameters(), library.parameters()):",
                    "    gap += (p - q).abs().sum().item()",
                    "assert gap == 0.0",
                ),
            )
        ],
        set(),
        id="sum-accumulated-in-a-loop",
    ),
    pytest.param(
        [
            (
                "9.1",
                check_lm(
                    "with torch.no_grad():", "    expected = library(x)", "assert torch.allclose(logits, expected)"
                ),
            )
        ],
        set(),
        id="assigned-inside-a-with-block",
    ),
    pytest.param(
        [
            (
                "9.1",
                check_lm(
                    "import contextlib",
                    "with contextlib.nullcontext(library) as reference:",
                    "    assert torch.allclose(logits, reference(x))",
                ),
            )
        ],
        set(),
        id="with-target",
    ),
    pytest.param([("9.1", CHECK_PACK + "\n\n" + CONDENSED_PACK)], {"no-assert"}, id="check-before-the-condensed-block"),
    pytest.param(
        [
            (
                "9.1",
                block(
                    "# 精簡自 tiny_perceptron/quantization.py：pack（省略輸入範圍檢查）",
                    "from tiny_perceptron.quantization import pack as library_pack",
                    "",
                    "assert pack((3, 4)) == library_pack((3, 4))",
                    "",
                    "",
                    "def pack(pair):",
                    "    return pair[0] * 16 + pair[1]",
                ),
            )
        ],
        {"no-assert"},
        id="check-above-the-definition-in-the-marker-block",
    ),
    pytest.param(
        [
            (
                "9.1",
                check_lm(
                    "same = torch.allclose(logits, library(x))",
                    'print("與函式庫版本相同：", same)',
                    "assert logits.shape == (1, 2, 3)",
                ),
            )
        ],
        {"not-both"},
        id="assert-does-not-use-the-library-version",
    ),
    pytest.param(
        [("9.1", check_lm("assert library(x).shape == (1, 2, 3)"))],
        {"not-both"},
        id="assert-uses-only-the-library-version",
    ),
    pytest.param(
        [
            (
                "9.1",
                check_lm(
                    "ours, theirs = model(x), library(x)", "print(ours, theirs)", "assert ours.shape == (1, 2, 3)"
                ),
            )
        ],
        {"not-both"},
        id="tuple-unpacking-keeps-the-two-versions-apart",
    ),
    pytest.param(
        [("9.1", check_lm("for p in model.parameters():", "    assert p.requires_grad"))],
        {"not-both"},
        id="loop-over-the-condensed-parameters-only",
    ),
    pytest.param(
        [
            (
                "9.1",
                CONDENSED_PACK
                + "\n\n"
                + block("from tiny_perceptron.quantization import pack", "", "assert pack((3, 4)) == 52"),
            )
        ],
        {"rebind"},
        id="import-without-as-rebinds-the-name",
    ),
    pytest.param(
        [
            (
                "9.1",
                CONDENSED_PACK
                + "\n\n"
                + block("from tiny_perceptron.quantization import pack as pack", "", "assert pack((3, 4)) == 52"),
            )
        ],
        {"rebind"},
        id="alias-equal-to-the-condensed-name",
    ),
    pytest.param(
        [
            (
                "9.1",
                CONDENSED_PACK
                + "\n\n"
                + block(
                    "from tiny_perceptron.quantization import pack as library_pack",
                    "from tiny_perceptron.quantization import pack",
                    "",
                    "assert pack((3, 4)) == library_pack((3, 4))",
                ),
            )
        ],
        {"rebind"},
        id="rebinding-import-next-to-a-renamed-one",
    ),
    pytest.param(
        [
            (
                "9.1",
                block("from tiny_perceptron.quantization import pack", "", "print(pack((3, 4)))")
                + "\n\n"
                + CONDENSED_PACK
                + "\n\n"
                + CHECK_PACK,
            )
        ],
        {"rebind"},
        id="unrenamed-import-before-the-condensed-definition",
    ),
    pytest.param(
        [
            (
                "9.1",
                block("# 精簡自 tiny_perceptron/quantization.py：pack（省略輸入範圍檢查）", "import math")
                + "\n\n"
                + block("def pack(pair):", "    return pair[0] * 16 + pair[1]")
                + "\n\n"
                + CHECK_PACK,
            )
        ],
        {"definition"},
        id="definition-in-the-next-block",
    ),
    pytest.param(
        [
            (
                "9.1",
                block(
                    "# 精簡自 tiny_perceptron/quantization.py：pack（省略輸入範圍檢查）",
                    "pack = lambda pair: pair[0] * 16 + pair[1]",
                )
                + "\n\n"
                + CHECK_PACK,
            )
        ],
        {"definition"},
        id="defined-by-assignment",
    ),
    pytest.param([("9.1", CONDENSED_PACK + "\n\n" + IMPORT_ONLY)], {"no-assert"}, id="no-assert"),
    pytest.param([("9.1", CONDENSED_PACK + "\n\n" + ASSERT_ONLY)], {"import"}, id="no-import"),
    pytest.param([("9.1", CONDENSED_PACK)], {"import", "no-assert"}, id="neither-import-nor-assert"),
    pytest.param(
        [("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK.replace("import pack as", "import unpack as"))],
        {"import"},
        id="import-of-another-name",
    ),
    pytest.param(
        [("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK.replace("quantization", "other"))],
        {"import"},
        id="import-from-another-module",
    ),
    pytest.param(
        [("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK.replace("quantization", "quantization.pack"))],
        {"import"},
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
        {"import"},
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
        {"import"},
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
        {"no-assert"},
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
        {"no-assert"},
        id="assert-only-inside-a-function-body",
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
                    "    same = pack((3, 4)) == library_pack((3, 4))",
                    "    return same",
                    "",
                    "",
                    "same = True",
                    "assert same",
                ),
            )
        ],
        {"not-both"},
        id="assignment-inside-a-function-is-not-traced",
    ),
    pytest.param(
        [("9.1", CONDENSED_PACK), ("9.2", CHECK_PACK)],
        {"import", "no-assert"},
        id="import-and-assert-belong-to-another-section",
    ),
    pytest.param(
        [("9.1", CONDENSED_PACK + "\n\n" + CHECK_PACK + "\n\n" + block("def broken(:"))],
        {"unparsable"},
        id="another-block-of-the-section-does-not-parse",
    ),
]


@pytest.mark.parametrize(("sections", "expected"), CONDENSED_SECTIONS)
def test_condensed_version_needs_an_assert_that_compares_it_with_the_library(tmp_path, sections, expected):
    make_repo(tmp_path, *sections)
    problems = check_course(tmp_path)
    if not expected:
        assert problems == []
        return
    assert len(problems) == 1 and "9.1" in problems[0]
    assert {kind for kind, text in CONDENSED_REASONS.items() if text in problems[0]} == expected


def test_the_report_shows_what_each_assert_after_the_condensed_version_reaches(tmp_path):
    make_repo(tmp_path, ("9.1", check_lm("assert logits.shape == (1, 2, 3)", "assert library(x).shape == (1, 2, 3)")))
    [problem] = check_course(tmp_path)
    assert "函式庫別名（`LibraryTinyLM`）" in problem
    assert "第 2 個 python 區塊的 `assert logits.shape == (1, 2, 3)`：追得到精簡版，追不到函式庫別名" in problem
    assert "第 2 個 python 區塊的 `assert library(x).shape == (1, 2, 3)`：追不到精簡版，追得到函式庫別名" in problem


def test_an_assert_before_the_condensed_definition_is_counted_in_the_report(tmp_path):
    make_repo(tmp_path, ("9.1", CHECK_PACK + "\n\n" + CONDENSED_PACK))
    [problem] = check_course(tmp_path)
    assert "定義之前的 1 個 assert 不算" in problem


def test_condensed_version_is_accepted_once_an_assert_compares_the_two_versions(tmp_path):
    make_repo(tmp_path, ("9.1", CONDENSED_PACK + "\n\n" + IMPORT_ONLY))
    assert len(check_course(tmp_path)) == 1
    make_repo(tmp_path, ("9.1", CONDENSED_PACK + "\n\n" + IMPORT_ONLY + "\n\n" + ASSERT_ONLY))
    assert len(check_course(tmp_path)) == 1
    comparing = block("assert pack((3, 4)) == library_pack((3, 4))")
    make_repo(tmp_path, ("9.1", CONDENSED_PACK + "\n\n" + IMPORT_ONLY + "\n\n" + comparing))
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


def test_scanner_finds_the_code_cells_that_the_notebook_builder_makes(tmp_path, monkeypatch):
    chapter = (
        "# 第九章\n\n導言。\n\n```python\nprint('導言')\n```\n\n"
        "## 9.1 一個問題\n\n```python\n\nx = 1\n\n```\n\n![圖](../figures/demo.svg)\n\n"
        '```py\ny = 2\n```\n\n```python title="z.py"\nz = 3\n```\n\n```python \nspace = 3\n```\n\n'
        "- 一個項目\n\n  ```python\n  w = 4\n  ```\n\n<details>\n<summary>補充</summary>\n\n```python\nv = 5\n```\n\n</details>\n\n"
        "## 9.2 另一個問題\n\n只有文字。\n"
    )
    write(tmp_path, "course/chapters/09.md", chapter)
    write(tmp_path, "course/chapters/0C.md", "# 必修\n\n## C.1 問題\n\n```python\nc = 1\n```\n")
    write(tmp_path, "course/figures/demo.svg", "<svg/>")
    write(
        tmp_path, "course/lesson-contract.json", json.dumps({"schema_version": 1, "lesson_ids": ["9.1", "9.2", "C.1"]})
    )
    monkeypatch.setattr(build_course, "ROOT", tmp_path)
    build_course.build()
    cells = []
    for path in sorted((tmp_path / "notebooks").rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        codes = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and not cell["metadata"].get("course_setup")
            and "course_figure" not in cell["metadata"]
        ]
        page, lesson = f"course/chapters/{path.parent.name}.md", notebook["metadata"]["lesson_id"]
        cells.extend((page, lesson, number, code) for number, code in enumerate(codes, 1))
    assert [code for *_, code in cells] == ["x = 1", "w = 4", "v = 5", "c = 1"]
    assert sorted(cells) == sorted(
        (fence.page, fence.section, fence.number, fence.code) for fence in read_fences(tmp_path)
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
