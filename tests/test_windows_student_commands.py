"""頁面教 Windows 讀者在 PowerShell 輸入的每一條命令，都要原樣寫進 Windows CI，或明列為刻意不跑。

.github/workflows/windows-student-commands.yml 在 windows-latest 照 README.md、README_en.md、course/first-steps.md、
course/training.md 的寫法跑學生命令。workflow 與頁面是兩份副本，這裡逐條把它們釘在一起：頁面上的 Windows 命令被
改寫、新增或刪除，而 workflow 沒跟著改，就在這裡失敗，訊息指出頁面與行號。

驗證：依頁面寫明的 Windows 換法，推出讀者實際輸入的命令序列，再和 workflow 裡 Invoke-Page（執行）與 Skip-Page
（不跑，-Reason 是印進記錄的簡短理由，完整理由寫在呼叫處的註解）標了該頁的命令依序逐條比對。命令取自 shell 類 fenced
區塊的每一行（MACOS_ONLY_BLOCKS 除外），以及 fence 外的 inline code——開頭是命令（COMMAND_START），或在提到
Windows／PowerShell 的句子裡帶 PowerShell 才有的寫法（POWERSHELL_ONLY）；說明裡提到的命令名稱與其他平台的寫法列在
NOT_WINDOWS。標成 text 這類純文字（PLAIN_LANGUAGES）的區塊不比對，但其中有一行符合 COMMAND_START 就報錯；fence 外出現
HTML 的 <code> 或 <pre> 也報錯，因為這裡只認得反引號與 fenced 區塊，寫在 HTML 裡的命令看不到。頁面的換法由
REPLACEMENTS（bash 行換成同頁句子裡給 Windows 的寫法）與 PREFIXES（整頁的 `.venv/bin/python` 換成 Windows 路徑）表示。
另外核對 workflow 的 Get-PageCode 找得到它要取的 Python 程式、每個 Skip-Page 都有 -Reason、push 與 pull_request 的
paths 都涵蓋 TRIGGER_PATHS，以及 Windows PowerShell 5.1 與 PowerShell 7 都跑。
不涵蓋：開頭不是 COMMAND_START、又不在 Windows 句子裡帶 PowerShell 寫法的 inline code（例如只寫腳本名的
`train.py --task vision`）；純文字區塊裡開頭不符合 COMMAND_START 的命令（例如前面帶 `PS>` 提示字元的行）；句意只認
錨點文字，換法句子改了措辭只會報「找不到」；續行的命令逐行比對；Get-PageCode 取用的 Python 程式內容不比對（CI 執行時
直接從頁面讀）；頁面寫給讀者核對的預期值（例如 W.7 印出的 `True`、training.md 的 `holdout_examples` 2、`01/1.1.ipynb`、
kernel 名稱 Tiny Perceptron）沒有和 workflow 的核對值互相釘住，頁面改了預期值而 workflow 照舊核對舊值，這裡不會失敗；
Skip-Page 的 -Reason 與註解寫的理由對不對；workflow 的命令在 Windows 上是否真的成功，要看 CI 本身。
"""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/windows-student-commands.yml"
PAGES = ("README.md", "README_en.md", "course/first-steps.md", "course/training.md")
# 這些檔改了，頁面命令的結果就可能不同，Windows CI 要重跑：頁面跑的程式、決定指標檔的 .gitattributes、決定 Python 版本的
# .python-version，以及兩條 --list 命令讀的清單。
TRIGGER_PATHS = (
    *PAGES,
    "pyproject.toml",
    "uv.lock",
    "scripts/**",
    "tiny_perceptron/**",
    "assets/training/manifest.json",
    ".gitattributes",
    ".python-version",
    "docs/course-experiments/public-models.json",
    WORKFLOW,
    "tests/test_windows_student_commands.py",
)

FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
SHELL_LANGUAGES = {"bash", "sh", "shell", "zsh", "fish", "console", "powershell", "pwsh", "ps1", "ps", "cmd", "bat"}
# 純文字區塊不比對命令，但其中有一行符合 COMMAND_START 就報錯：讀者照樣會把它打進 PowerShell。
PLAIN_LANGUAGES = {"text", "txt", "plaintext", "output"}
OTHER_LANGUAGES = {"python", "json", "jsonl", "yaml", "toml", *PLAIN_LANGUAGES}
SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
# fence 外的 HTML code 元素。這裡不解析 HTML，寫在裡面的命令看不到，所以出現就報錯。
HTML_CODE = re.compile(r"<(?:code|pre)\b", re.IGNORECASE)
# 句子在這些標點之後結束；英文的句點、分號後面要接空白或行尾。
TERMINATOR = re.compile(r"[。；！？]|[.;!?](?=\s|$)")
WINDOWS_WORDS = ("Windows", "PowerShell")
COMMAND_START = re.compile(
    r"^(?:(?:git|uv|uvx|cd|python3?|py|pip|jupyter|pytest|ruff|source|mkdir|powershell|pwsh|winget|choco|scoop)(?=\s|$)"
    r"|\.venv[/\\]|\.\.?[/\\]|\$env:|[A-Z][a-z]+-[A-Z][A-Za-z]+(?=\s|$))"
)
POWERSHELL_ONLY = re.compile(r"\\|\$|\.(?:exe|ps1)\b|^[A-Z][a-z]+-[A-Z][A-Za-z]+(?=\s|$)")

# 頁面句子裡給 Windows 的寫法，取代 bash 區塊裡的一行：(bash 行, 錨點)。取代的寫法是錨點之後、同一句裡的 inline code。
REPLACEMENTS = {
    "README.md": [
        (
            "GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git",
            "Windows PowerShell 不寫這段開頭，改成先執行",
        ),
        ("source .venv/bin/activate", "Windows PowerShell 啟用指令是"),
    ],
    "README_en.md": [
        (
            "GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git",
            "In Windows PowerShell, leave that prefix out: run",
        ),
        ("source .venv/bin/activate", "On Windows PowerShell, activate with"),
    ],
    "course/first-steps.md": [
        (
            "GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git",
            "Windows PowerShell 不寫這段開頭，改成先執行",
        ),
    ],
    "course/training.md": [
        ("mkdir -p outputs checkpoints", "Windows PowerShell 把 `mkdir -p outputs checkpoints` 換成"),
    ],
}
# 整頁命令開頭的路徑換法：(原寫法所在的錨點, Windows 寫法所在的錨點)；兩者都取錨點之後的第一段 inline code。
PREFIXES = {
    "course/training.md": ("命令一律寫成", "Windows PowerShell 把這段路徑換成"),
}
# 錨點所在那一行之後的第一個 fenced 區塊只給 macOS／Linux；Windows 讀者改用同頁另給的 PowerShell 區塊。
MACOS_ONLY_BLOCKS = {
    "course/first-steps.md": ["macOS 與 Linux："],
}
# inline code 看起來像命令，但不是 Windows 讀者要在 PowerShell 輸入的：(含那段 inline code 的原文片段, 理由)。
NOT_WINDOWS = {
    "README.md": [
        ("把 `uv sync` 那一行換成下表的寫法", "指出要換掉哪一行"),
        ("| Apple Silicon | `uv sync --frozen --group notebook` |", "Apple Silicon 的安裝選擇"),
        ("直接用 `python`", "說明啟用後直接用 python"),
        ("或每次 `uv run` 都帶相同 extra", "提到 uv run 的用法，不是完整命令"),
    ],
    "README_en.md": [
        ("replace the `uv sync` line", "指出要換掉哪一行"),
        ("use `uv sync --frozen --group notebook` on Apple Silicon", "Apple Silicon 的安裝選擇"),
        ("Installation and the `git lfs version` check are in", "提到「第一次開啟教材」那節的檢查命令"),
        ("Activate `.venv` and use `python` directly", "說明啟用後直接用 python"),
        ("pass the same extra to each `uv run`", "提到 uv run 的用法，不是完整命令"),
    ],
    "course/first-steps.md": [
        ("`git clone` 會在終端機目前所在的資料夾裡建立", "說明 git clone 建立的資料夾"),
        ("`uv sync` 根據專案的套件清單建立", "說明 uv sync 做什麼"),
        ("在 macOS 或 Linux 執行 `source .venv/bin/activate` 啟用工具箱", "macOS／Linux 的啟用寫法"),
        ("啟用後，`python` 指向剛安裝的 Python 工具箱", "說明啟用的效果；CI 在啟用後另外核對"),
        ("不允許執行 `Activate.ps1` 這類腳本", "錯誤說明裡的檔名"),
        ("| `.venv/` | `uv sync` 建立的工具箱", "路徑表的資料夾與說明"),
        ("在終端機先 `cd tiny-perceptron-vlm`，是把起點移到", "回頭說明前面已經輸入過的 cd"),
        ("macOS 與 Linux 是 `.venv/bin/python`，Windows PowerShell 是", "macOS／Linux 的寫法"),
        ("才能再用 `cd` 切換資料夾", "提到 cd 的用途，不是完整命令"),
        ("路徑要改成 `../.venv/bin/python`（Windows 是", "macOS／Linux 的寫法"),
        ("也可以在專案根目錄直接用 `.venv/bin/python` 執行", "macOS／Linux 的寫法"),
    ],
    "course/training.md": [
        ("這時工具會先執行 `git lfs pull`", "說明下載工具自己會執行的命令"),
    ],
}

CALL = re.compile(
    r"^[ \t]*(?:\$\w+[ \t]*=[ \t]*)?(?P<verb>Invoke-Page|Skip-Page)[ \t]+-Pages[ \t]+(?P<pages>\S+)"
    r"(?:[ \t]+-Reason[ \t]+'(?P<reason>(?:[^'\n]|'')*)')?"
    r"(?:[ \t]+-Stdin[ \t]+\$\w+)?[ \t]+-Commands[ \t]+(?:@'\n(?P<block>.*?)\n'@|'(?P<line>(?:[^'\n]|'')*)')",
    re.M | re.S,
)
CALL_START = re.compile(r"^[ \t]*(?:\$\w+[ \t]*=[ \t]*)?(?:Invoke|Skip)-Page\b", re.M)
PAGE_CODE = re.compile(r"Get-PageCode[ \t]+-Page[ \t]+(?P<page>\S+)[ \t]+-After[ \t]+'(?P<anchor>(?:[^'\n]|'')*)'")


@dataclass(frozen=True)
class Command:
    line: int
    text: str


@dataclass(frozen=True)
class Copy:
    step: str
    verb: str
    pages: tuple
    text: str


@dataclass
class Span:
    line: int
    start: int
    end: int
    text: str
    sentence: int
    windows: bool


def read(relative):
    # .gitattributes 讓 checkout 用 LF；沒套用時（例如 Windows 上手動複製的檔案）也照 LF 比對。
    return (ROOT / relative).read_text(encoding="utf-8").replace("\r\n", "\n")


def load_workflow():
    return yaml.load(read(WORKFLOW), Loader=yaml.BaseLoader)


def split_page(text):
    """(區塊, fence 外的行)：區塊是 [(開頭行號, 語言, [(行號, 內容)])]；fence 外的行是 [(行號, 內容)]。"""
    blocks, prose, opened = [], [], None
    for number, line in enumerate(text.split("\n"), 1):
        fence = FENCE.match(line)
        if opened is None:
            if fence:
                opened = (fence[1][0], len(fence[1]))
                blocks.append((number, (fence[2].split() or [""])[0].lower(), []))
            else:
                prose.append((number, line))
        elif fence and fence[1][0] == opened[0] and len(fence[1]) >= opened[1] and not fence[2].strip():
            opened = None
        else:
            blocks[-1][2].append((number, line))
    return blocks, prose


def sentences(line):
    """(遮掉 inline code 的行, 各句的起點與終點)：inline code 裡的標點不算句子結束。"""
    masked = list(line)
    for match in SPAN.finditer(line):
        masked[match.start() : match.end()] = "\0" * (match.end() - match.start())
    masked = "".join(masked)
    bounds = [0, *(match.end() for match in TERMINATOR.finditer(masked)), len(line) + 1]
    return masked, list(zip(bounds, bounds[1:]))


def sentence_at(line, position):
    """position 落在這一行的第幾句。"""
    _, ranges = sentences(line)
    return next(index for index, (start, end) in enumerate(ranges) if start <= position < end)


def spans_in(number, line):
    """一行 fence 外文字裡的每段 inline code，記下它在第幾句、那句有沒有提到 Windows 或 PowerShell。"""
    masked, ranges = sentences(line)
    result = []
    for match in SPAN.finditer(line):
        content = match[2]
        if content.startswith(" ") and content.endswith(" ") and content.strip():
            content = content[1:-1]
        sentence = sentence_at(line, match.start())
        start, end = ranges[sentence]
        windows = any(word in masked[start:end] for word in WINDOWS_WORDS)
        result.append(Span(number, match.start(), match.end(), content, sentence, windows))
    return result


def occurrences(prose, text):
    """text 在 fence 外每次出現的 (行號, 起點)。"""
    found = []
    for number, line in prose:
        start = line.find(text)
        while start != -1:
            found.append((number, start))
            start = line.find(text, start + 1)
    return found


def unique_anchor(page, prose, anchor, problems):
    """錨點在 fence 外剛好出現一次時回傳 (行號, 起點, 終點)；否則記下問題並回傳 None。"""
    hits = occurrences(prose, anchor)
    if len(hits) != 1:
        where = f"出現在第 {'、'.join(str(number) for number, _ in hits)} 行" if hits else "找不到"
        problems.append(f"{page}：「{anchor}」要在頁面上剛好出現一次，現在{where}；頁面的 Windows 寫法改了嗎？")
        return None
    number, start = hits[0]
    return number, start, start + len(anchor)


def windows_commands(page):
    """(命令, 問題)：照頁面寫明的 Windows 換法，依頁面順序列出 Windows 讀者在 PowerShell 輸入的命令。"""
    problems = []
    blocks, prose = split_page(read(page))
    for number, line in prose:
        masked, _ = sentences(line)
        if HTML_CODE.search(masked):
            problems.append(
                f"{page}:{number}：fence 外有 HTML 的 <code> 或 <pre>，這個測試不支援；寫在裡面的命令不會被比對，"
                "請改用反引號的 inline code 或 fenced 區塊"
            )
    spans = [span for number, line in prose for span in spans_in(number, line)]
    picked = {
        id(span)
        for span in spans
        if COMMAND_START.match(span.text) or (span.windows and POWERSHELL_ONLY.search(span.text))
    }

    excluded = set()
    for anchor in MACOS_ONLY_BLOCKS.get(page, []):
        if (hit := unique_anchor(page, prose, anchor, problems)) is None:
            continue
        following = [block for block in blocks if block[0] > hit[0]]
        if not following or not dict(prose)[hit[0]].rstrip().endswith(anchor):
            problems.append(f"{page}:{hit[0]}：「{anchor}」要在行尾，後面接著 macOS／Linux 的程式區塊")
            continue
        excluded.add(following[0][0])

    lines = []
    for start, language, body in blocks:
        if language in PLAIN_LANGUAGES:
            problems += [
                f"{page}:{number}：標成「{language}」的區塊裡，「{content.strip()}」看起來是命令；"
                "shell 命令請標 bash 或 powershell，這個測試才會比對它"
                for number, content in body
                if COMMAND_START.match(content.strip())
            ]
        if language in OTHER_LANGUAGES:
            continue
        if language not in SHELL_LANGUAGES:
            problems.append(f"{page}:{start}：程式區塊的語言「{language}」沒有分類；shell 命令請標 bash 或 powershell")
            continue
        if start not in excluded:
            lines.extend(Command(number, content.strip()) for number, content in body if content.strip())

    def after(hit, count=None):
        """錨點之後、和錨點同一句的 inline code；count 只取前幾段。"""
        number, _, end = hit
        sentence = sentence_at(dict(prose)[number], end - 1)
        chosen = [span for span in spans if span.line == number and span.start >= end and span.sentence == sentence]
        return chosen[:count] if count else chosen

    replaced = {}
    for bash, anchor in REPLACEMENTS.get(page, []):
        targets = [command for command in lines if command.text == bash]
        hit = unique_anchor(page, prose, anchor, problems)
        if len(targets) != 1:
            where = "、".join(str(command.line) for command in targets) or "無"
            problems.append(f"{page}：要換成 Windows 寫法的 bash 行「{bash}」不只一處或找不到（第 {where} 行）")
        if hit is None or len(targets) != 1:
            continue
        chosen = after(hit)
        if not chosen:
            problems.append(f"{page}:{hit[0]}：「{anchor}」之後同一句裡沒有 Windows 寫法的 inline code")
            continue
        for span in spans:
            if span.line == hit[0] and hit[1] <= span.start < chosen[-1].end:
                picked.discard(id(span))
        replaced[targets[0]] = [Command(span.line, span.text) for span in chosen]

    prefix = None
    if page in PREFIXES:
        hits = [unique_anchor(page, prose, anchor, problems) for anchor in PREFIXES[page]]
        if None not in hits:
            pair = [after(hit, 1) for hit in hits]
            if all(pair):
                for (number, start, _), [span] in zip(hits, pair):
                    for other in spans:
                        if other.line == number and start <= other.start < span.end:
                            picked.discard(id(other))
                prefix = (pair[0][0].text.rstrip(" …"), pair[1][0].text)
            else:
                problems.append(f"{page}:{hits[0][0]}：路徑換法的兩個錨點之後要各有一段 inline code")

    for snippet, _ in NOT_WINDOWS.get(page, []):
        hits = occurrences(prose, snippet)
        covered = [
            span
            for number, start in hits
            for span in spans
            if span.line == number and start <= span.start < start + len(snippet) and id(span) in picked
        ]
        if len(hits) != 1 or not covered:
            problems.append(f"{page}：NOT_WINDOWS 的「{snippet}」對不到頁面上的命令（找到 {len(hits)} 處）；請更新清單")
        for span in covered:
            picked.discard(id(span))

    items = [(command.line, 0, [command]) for command in lines if command not in replaced]
    items += [(command.line, 0, replaced[command]) for command in lines if command in replaced]
    items += [(span.line, span.start, [Command(span.line, span.text)]) for span in spans if id(span) in picked]
    commands = [command for _, _, group in sorted(items, key=lambda item: item[:2]) for command in group]
    if prefix:
        source, target = prefix
        commands = [
            Command(command.line, target + command.text[len(source) :])
            if command.text == source or command.text.startswith(source + " ")
            else command
            for command in commands
        ]
    return commands, problems


def workflow_copies():
    """(副本, 問題)：workflow 每個 run 步驟裡 Invoke-Page／Skip-Page 依序列出的命令。"""
    copies, problems = [], []
    for job in load_workflow()["jobs"].values():
        for step in job.get("steps", []):
            script = step.get("run", "")
            matches = list(CALL.finditer(script))
            if len(matches) != len(CALL_START.findall(script)):
                problems.append(f"{WORKFLOW}：步驟「{step.get('name', '')}」有 Invoke-Page／Skip-Page 的寫法認不出來")
            for match in matches:
                pages = tuple(match["pages"].split(","))
                unknown = [page for page in pages if page not in PAGES]
                if unknown:
                    problems.append(f"{WORKFLOW}：步驟「{step.get('name', '')}」的 -Pages 有不認得的頁面 {unknown}")
                if match["verb"] == "Skip-Page" and not (match["reason"] or "").strip():
                    problems.append(
                        f"{WORKFLOW}：步驟「{step.get('name', '')}」的 Skip-Page 沒有 -Reason；"
                        "記錄裡要印出不跑的理由，讀記錄的人才分得出「這裡沒跑」與「前面已經跑過」"
                    )
                if match["verb"] == "Invoke-Page" and match["reason"] is not None:
                    problems.append(f"{WORKFLOW}：步驟「{step.get('name', '')}」的 Invoke-Page 不收 -Reason")
                body = match["block"].split("\n") if match["block"] is not None else [match["line"].replace("''", "'")]
                copies.extend(
                    Copy(step.get("name", ""), match["verb"], pages, line.strip()) for line in body if line.strip()
                )
    return copies, problems


def compare(page, expected, actual):
    """逐條比對頁面推出的命令與 workflow 標了這頁的命令；回傳指出頁面行號的問題。"""
    matcher = difflib.SequenceMatcher(a=[c.text for c in expected], b=[c.text for c in actual], autojunk=False)
    problems = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        paired = min(i2 - i1, j2 - j1) if tag == "replace" else 0
        for offset in range(i2 - i1):
            command = expected[i1 + offset]
            if offset < paired:
                copy = actual[j1 + offset]
                problems.append(
                    f"{page}:{command.line}：頁面的 Windows 命令是 `{command.text}`，"
                    f"workflow 的對應位置（步驟「{copy.step}」的 {copy.verb}）是 `{copy.text}`"
                )
            else:
                problems.append(
                    f"{page}:{command.line}：頁面的 Windows 命令 `{command.text}` 在 workflow 沒有對應的 Invoke-Page 或 Skip-Page"
                )
        last = i1 + paired - 1
        near = expected[last].line if last >= 0 else (expected[0].line if expected else 1)
        for copy in actual[j1 + paired : j2]:
            problems.append(
                f"{page}:{near}：workflow 步驟「{copy.step}」的 {copy.verb} 列了 `{copy.text}`，頁面上找不到；"
                f"它在頁面上應接在第 {near} 行的命令之後"
            )
    return problems


@pytest.mark.parametrize("page", PAGES)
def test_page_windows_commands_match_workflow(page):
    commands, problems = windows_commands(page)
    copies, _ = workflow_copies()
    problems += compare(page, commands, [copy for copy in copies if page in copy.pages])
    assert not problems, (
        "頁面與 workflow 的 Windows 命令要一起改；不跑的命令寫進 Skip-Page，-Reason 寫印進記錄的簡短理由，"
        "註解寫完整理由。\n" + "\n".join(problems)
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "```text\nRemove-Item -Recurse -Force .venv\n```\n", "x.md:2：標成「text」", id="text 區塊裡的 cmdlet"
        ),
        pytest.param(
            "```output\ngit clone https://github.com/birdhackor/tiny-perceptron-vlm.git\n```\n",
            "x.md:2：標成「output」",
            id="output 區塊裡的 git",
        ),
        pytest.param(
            "```plaintext\n  .\\.venv\\Scripts\\python.exe scripts/check_env.py\n```\n",
            "x.md:2：標成「plaintext」",
            id="plaintext 區塊裡縮排的 Windows 路徑",
        ),
        pytest.param('```txt\n$env:GIT_LFS_SKIP_SMUDGE="1"\n```\n', "x.md:2：標成「txt」", id="txt 區塊裡的環境變數"),
        pytest.param(
            "```text\ncourse/chapters/     正文\nuv sync --frozen --extra cpu --group notebook\n```\n",
            "x.md:3：標成「text」",
            id="text 區塊第二行才是命令",
        ),
        pytest.param(
            "或先執行 <code>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned</code>。\n",
            "x.md:1：fence 外有 HTML",
            id="句子裡的 code 元素",
        ),
        pytest.param("<pre>\ngit lfs install --local\n</pre>\n", "x.md:1：fence 外有 HTML", id="跨行的 pre 元素"),
        pytest.param("說明 <CODE>uv sync</CODE> 的用途。\n", "x.md:1：fence 外有 HTML", id="大寫的 CODE 元素"),
        pytest.param("```text\ncourse/chapters/     教材正文與短程式\n```\n", None, id="專案結構不是命令"),
        pytest.param("HTML 的 `<code>` 標籤只是舉例。\n", None, id="反引號裡的 code 字樣"),
    ],
)
def test_commands_in_plain_blocks_or_html_code_are_reported(monkeypatch, text, expected):
    """寫在純文字區塊或 HTML code 元素裡的命令，比對時看不到，所以要報錯並指出行號；不是命令的內容不報。"""
    monkeypatch.setitem(globals(), "read", lambda relative: text)
    _, problems = windows_commands("x.md")
    if expected is None:
        assert not problems, problems
    else:
        assert any(problem.startswith(expected) for problem in problems), problems


def test_workflow_page_calls_are_recognized():
    copies, problems = workflow_copies()
    assert copies, f"{WORKFLOW} 沒有任何 Invoke-Page／Skip-Page"
    assert not problems, "\n".join(problems)


def test_page_code_anchors_point_at_python_blocks():
    found, problems = [], []
    for job in load_workflow()["jobs"].values():
        for step in job.get("steps", []):
            found += [
                (match["page"], match["anchor"].replace("''", "'")) for match in PAGE_CODE.finditer(step.get("run", ""))
            ]
    assert found, f"{WORKFLOW} 沒有 Get-PageCode；頁面上要讀者複製的 Python 程式改由誰取？"
    for page, anchor in found:
        text = read(page)
        blocks, _ = split_page(text)
        # Get-PageCode 在全頁（含程式區塊）找第一個含錨點的行，所以這裡也看全頁。
        hits = [number for number, line in enumerate(text.split("\n"), 1) if anchor in line]
        following = [block for block in blocks if hits and block[0] > hits[0]]
        if len(hits) != 1:
            problems.append(f"{page}：Get-PageCode 的錨點「{anchor}」出現在 {len(hits)} 行，要剛好一行")
        elif not following or following[0][1] != "python":
            problems.append(f"{page}:{hits[0]}：「{anchor}」之後的第一個程式區塊不是 python")
    assert not problems, "\n".join(problems)


def test_workflow_triggers_and_shells():
    document = load_workflow()
    triggers = document["on"]
    assert "workflow_dispatch" in triggers
    for event in ("push", "pull_request"):
        missing = [path for path in TRIGGER_PATHS if path not in triggers[event]["paths"]]
        assert not missing, f"{event} 的 paths 少了 {missing}；這些檔案改了，Windows CI 要重跑"
    for name, job in document["jobs"].items():
        assert job["runs-on"] == "windows-latest", f"{name} 要在 windows-latest 跑"
        shells = job["strategy"]["matrix"]["shell"]
        assert shells == ["powershell", "pwsh"], (
            f"{name} 要用 matrix 跑 Windows PowerShell 5.1 與 PowerShell 7：{shells}"
        )
        assert job["defaults"]["run"]["shell"] == "${{ matrix.shell }}", f"{name} 的 run 步驟要照 matrix 選 shell"
        overridden = [step.get("name", "") for step in job["steps"] if "run" in step and "shell" in step]
        assert not overridden, f"這些步驟改用別的 shell，沒有照 matrix 跑兩種 PowerShell：{overridden}"
