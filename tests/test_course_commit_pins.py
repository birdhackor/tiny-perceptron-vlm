"""學生會照著做的頁面裡，釘住本 repo 的 commit 必須是目前歷史裡的祖先。

改寫 Git 歷史後舊的 hash 會失效，教材卻仍叫讀者 `git checkout` 它、拿它下載資料，或連到 GitHub 上已經
不存在的版本；這裡在完整 clone 上抓出這種殘留。

頁面集合（page_set）：course/chapters 的全部章檔、scripts.build_course.READING_PAGES、README_en.md 與
docs/publishing.md，再加上從這些頁面 fence 外的相對連結 `[文字](路徑)` 連到的 repo 內 .md 與 .json 檔；
連到的 .md 繼續追它的連結，.json 不再往下追。

驗證：
- fenced 程式區塊裡、和 `git ` 在同一行的 40 位十六進位字串。
- 頁面任何位置、指向本 repo（github.com/birdhackor/tiny-perceptron-vlm）的 blob／tree／commit／raw 網址裡的
  40 位 hash。封存 repo 與其他 repo 的網址不算。
- fenced 程式區塊裡執行 scripts/fetch_natural_data.py 的命令（行尾 `\\` 續行接成同一條，不跨區塊）的
  `--revision` 值；這個腳本用它組本 repo 的 raw／media 網址。其他腳本的 `--revision`、`--model-revision`
  多半是 Hugging Face 的版本，不算。
- 頁面集合裡的 .json 檔解析後，`repo_id` 等於 birdhackor/tiny-perceptron-vlm 的物件的 40 位 `revision`。
- scripts/fetch_natural_data.py 模組層級的 `DEFAULT_REVISION`，執行時用 ast 讀出。

不涵蓋：續行的 `git` 命令（hash 不在 `git` 那一行）、shell 變數、Python 清單形式的命令、正文 inline code 與
一般敘述裡的 hash。
"""

import ast
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.build_course import READING_PAGES

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "birdhackor/tiny-perceptron-vlm"
EXTRA_PAGES = ("README_en.md", "docs/publishing.md")
FETCH_SCRIPT = "scripts/fetch_natural_data.py"
COMMIT = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{40}(?![0-9A-Fa-f])")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
# GitHub 的帳號與 repo 名稱不分大小寫；repo 名稱後面必須緊接 `/`，封存 repo（…-archive）因此不算。
REPOSITORY_URL = re.compile(
    r"github\.com/birdhackor/tiny-perceptron-vlm/(?:blob|tree|commit|raw)/([0-9A-Fa-f]{40})(?![0-9A-Fa-f])",
    re.IGNORECASE,
)
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)")
RUNS_FETCH_SCRIPT = re.compile(r"(?:^|[\s/])fetch_natural_data\.py(?!\S)")
REVISION_OPTION = re.compile(r"(?<!\S)--revision(?:=|\s+)([0-9A-Fa-f]{40})(?![0-9A-Fa-f])")


def code_blocks(text):
    """[[(行號, 內容)]]：每個 fenced 程式區塊裡的行；區塊外的文字不回傳。"""
    blocks, opened = [], None  # opened：(圍欄字元, 圍欄長度)
    for number, line in enumerate(text.split("\n"), 1):
        fence = FENCE.match(line)
        if opened is None:
            if fence:
                opened = (fence[1][0], len(fence[1]))
                blocks.append([])
        elif fence and fence[1][0] == opened[0] and len(fence[1]) >= opened[1] and not fence[2].strip():
            opened = None
        else:
            blocks[-1].append((number, line))
    return blocks


def code_lines(text):
    """fenced 程式區塊裡的 (行號, 內容)；區塊外的文字不回傳。"""
    return [item for lines in code_blocks(text) for item in lines]


def pins_in(text):
    """(行號, hash)：fenced 程式區塊裡，和 `git ` 在同一行的 40 位十六進位字串。"""
    return [(number, commit) for number, line in code_lines(text) if "git " in line for commit in COMMIT.findall(line)]


def url_pins_in(text):
    """(行號, hash)：頁面任何位置指向本 repo 的 blob／tree／commit／raw 網址裡的 40 位 hash。"""
    return [
        (number, commit) for number, line in enumerate(text.split("\n"), 1) for commit in REPOSITORY_URL.findall(line)
    ]


def commands(text):
    """(第一行的行號, 命令)：fenced 程式區塊裡的每條命令；行尾 `\\` 的續行接到下一行，不跨區塊。"""
    result = []
    for lines in code_blocks(text):
        start, parts = None, []
        for number, line in lines:
            start = number if start is None else start
            if line.rstrip().endswith("\\"):
                parts.append(line.rstrip()[:-1])
                continue
            result.append((start, " ".join([*parts, line])))
            start, parts = None, []
        if parts:
            result.append((start, " ".join(parts)))
    return result


def fetch_pins_in(text):
    """(命令第一行的行號, hash)：fenced 程式區塊裡執行 fetch_natural_data.py 的命令的 `--revision` 值。"""
    return [
        (number, commit)
        for number, command in commands(text)
        if RUNS_FETCH_SCRIPT.search(command)
        for commit in REVISION_OPTION.findall(command)
    ]


def json_pins_in(text):
    """(JSON 路徑, hash)：解析後每個 `repo_id` 等於本 repo 的物件，它的 40 位 `revision`。"""
    pins = []

    def visit(value, path):
        if isinstance(value, dict):
            revision = value.get("revision")
            if value.get("repo_id") == REPOSITORY and isinstance(revision, str) and COMMIT.fullmatch(revision):
                pins.append((".".join([*path, "revision"]), revision))
            for key, child in value.items():
                visit(child, [*path, str(key)])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, [*path, str(index)])

    visit(json.loads(text), [])
    return pins


def default_revision_pins(text):
    """(行號, 值)：模組層級 `DEFAULT_REVISION = "…"` 的字串值；用 ast 讀，行號是執行時算出的。"""
    pins = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if (
            any(isinstance(target, ast.Name) and target.id == "DEFAULT_REVISION" for target in targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            pins.append((node.lineno, node.value.value))
    return pins


def student_pages(root, reading_pages=READING_PAGES, extra_pages=EXTRA_PAGES):
    """頁面集合的起點：course/chapters 的全部章檔、reading_pages 與 extra_pages。"""
    chapters = sorted(path.relative_to(root).as_posix() for path in (root / "course/chapters").glob("*.md"))
    return chapters + list(reading_pages) + list(extra_pages)


def linked_files(root, page):
    """page 在 fence 外以相對連結連到、確實存在於 root 裡的 .md 與 .json（相對 root，用 / 分隔）。"""
    lines = (root / page).read_text(encoding="utf-8").split("\n")
    fenced = {number for number, _ in code_lines("\n".join(lines))}
    found = []
    for number, line in enumerate(lines, 1):
        if number in fenced:
            continue
        for target in LINK.findall(line):
            path_text = target.partition("#")[0]
            if not path_text or "://" in path_text or path_text.startswith("mailto:"):
                continue
            path = ((root / page).parent / path_text).resolve()
            if path.suffix in (".md", ".json") and path.is_file() and path.is_relative_to(root.resolve()):
                found.append(path.relative_to(root.resolve()).as_posix())
    return found


def page_set(root, reading_pages=READING_PAGES, extra_pages=EXTRA_PAGES):
    """起點頁面加上相對連結的遞移閉包：連到的 .md 繼續追，.json 不再往下追。回傳排序過的相對路徑。"""
    seen, pending = set(), student_pages(root, reading_pages, extra_pages)
    while pending:
        page = pending.pop()
        if page in seen:
            continue
        seen.add(page)
        if page.endswith(".md"):
            pending.extend(linked_files(root, page))
    return sorted(seen)


def course_pins(root, reading_pages=READING_PAGES, extract=pins_in, extra_pages=EXTRA_PAGES, suffix=".md"):
    """[(頁面, 位置, hash)]：頁面集合裡副檔名為 suffix 的檔案，由 extract 抽出的 commit。"""
    return [
        (page, place, commit)
        for page in page_set(root, reading_pages, extra_pages)
        if page.endswith(suffix)
        for place, commit in extract((root / page).read_text(encoding="utf-8"))
    ]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")


def history_unavailable(repo):
    """無法核對祖先關係的原因；可以核對時回傳 None。"""
    if shutil.which("git") is None:
        return "找不到 git 指令"
    shallow = git(repo, "rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0:
        return "不在 Git 工作目錄裡，沒有歷史可以核對"
    if shallow.stdout.strip() == "true":
        return "這是淺 clone，缺少歷史；請在完整 clone 執行（`git fetch --unshallow` 可補齊歷史）"
    return None


def pin_problems(repo, pins):
    """每個被釘住的 hash 都必須是 commit 物件，而且是 HEAD 的祖先；回傳問題清單。"""
    places = {}
    for page, number, commit in pins:
        places.setdefault(commit.lower(), []).append(f"{page}:{number}")
    problems = []
    for commit, where in places.items():
        label = f"{commit}（{'、'.join(where)}）"
        kind = git(repo, "cat-file", "-t", commit)
        if kind.returncode != 0 or kind.stdout.strip() != "commit":
            found = kind.stdout.strip() or kind.stderr.strip()
            problems.append(f"{label} 不是這個 repo 的 commit 物件：{found}")
            continue
        ancestor = git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
        if ancestor.returncode != 0:
            reason = "不是 HEAD 的祖先" if ancestor.returncode == 1 else ancestor.stderr.strip()
            problems.append(f"{label} 是 commit，但{reason}")
    return problems


# ---- 抽取器：正例表 ---------------------------------------------------------------------------------------

A = "1a2b3c4d5e" * 4
B = "f0e1d2c3b4" * 4
PIN_TEXTS = [
    pytest.param(f"```bash\ngit checkout --detach {A}\n```\n", [(2, A)], id="checkout-detach"),
    pytest.param(f"```bash\nGIT_LFS_SKIP_SMUDGE=1 git checkout --detach {A}\n```\n", [(2, A)], id="environment-prefix"),
    pytest.param(f"```\ngit show {A}:requirements.txt\n```\n", [(2, A)], id="show-a-path"),
    pytest.param(f"```sh\ngit diff {A} {B}\n```\n", [(2, A), (2, B)], id="two-commits-on-one-line"),
    pytest.param(f"~~~bash\ngit checkout {A}\n~~~\n", [(2, A)], id="tilde-fence"),
    pytest.param(f"```python\nrun('git checkout {A}')\n```\n", [(2, A)], id="python-fence"),
    pytest.param(
        f"<details>\n\n  ```bash\n  git checkout {A}\n  ```\n\n</details>\n", [(4, A)], id="indented-fence-in-details"
    ),
    pytest.param(f"````text\n```bash\ngit checkout {A}\n```\n````\n", [(3, A)], id="fence-inside-a-longer-fence"),
    pytest.param(f"````bash\ngit status\n```\ngit checkout {A}\n````\n", [(4, A)], id="shorter-fence-does-not-close"),
    pytest.param(f"```bash\ngit checkout {A}\n", [(2, A)], id="fence-never-closed"),
    pytest.param(f"```bash\ngit checkout {A.upper()}\n```\n", [(2, A.upper())], id="upper-case-hex"),
    pytest.param(f"先執行 git checkout {A}，再繼續。\n", [], id="prose-outside-a-fence"),
    pytest.param(f"執行 `git checkout {A}` 即可。\n", [], id="inline-code-outside-a-fence"),
    pytest.param(f"```bash\ngit status\n```\ngit checkout {A}\n", [], id="after-the-fence-closed"),
    pytest.param(f"```bash\ngit clone url\necho {A}\n```\n", [], id="commit-on-a-line-without-git"),
    pytest.param(f"```bash\ngit checkout {A[:39]}\n```\n", [], id="39-hex-digits"),
    pytest.param(f"```bash\ngit checkout {A}1\n```\n", [], id="41-hex-digits"),
    pytest.param(f"```bash\ngit log --grep {A + B[:24]}\n```\n", [], id="longer-hex-string-such-as-sha256"),
    pytest.param("```bash\ngit checkout 1a2b3c4\n```\n", [], id="abbreviated-commit"),
    pytest.param(f"```bash\ngit checkout main  # {'g' * 40}\n```\n", [], id="forty-characters-that-are-not-hex"),
]


@pytest.mark.parametrize(("text", "expected"), PIN_TEXTS)
def test_pins_are_read_from_git_lines_inside_fenced_blocks(text, expected):
    assert pins_in(text) == expected


REPO = "https://github.com/birdhackor/tiny-perceptron-vlm"
URL_TEXTS = [
    pytest.param(f"見[原始碼]({REPO}/blob/{A}/tiny_perceptron/model.py)。\n", [(1, A)], id="blob-link-in-prose"),
    pytest.param(f"\n[資料夾]({REPO}/tree/{A}/docs)\n", [(2, A)], id="tree-link"),
    pytest.param(f"<{REPO}/commit/{A}>\n", [(1, A)], id="commit-link"),
    pytest.param(f"```bash\ncurl -LO {REPO}/raw/{A}/data.json\n```\n", [(2, A)], id="raw-url-inside-a-fence"),
    pytest.param(f"{REPO}/blob/{A}/a.py 與 {REPO}/blob/{B}/b.py\n", [(1, A), (1, B)], id="two-links-on-one-line"),
    pytest.param(
        f"https://github.com/BirdHackor/Tiny-Perceptron-VLM/blob/{A}/a.py\n", [(1, A)], id="names-in-other-case"
    ),
    pytest.param(f"{REPO}-archive/blob/{A}/a.py\n", [], id="archive-repository"),
    pytest.param(f"https://github.com/pytorch/pytorch/blob/{A}/torch/nn.py\n", [], id="another-repository"),
    pytest.param(f"https://huggingface.co/birdhackor/tiny-perceptron-vlm/tree/{A}\n", [], id="hugging-face"),
    pytest.param(f"{REPO}/blob/main/tiny_perceptron/model.py\n", [], id="branch-name"),
    pytest.param(f"{REPO}/blob/1a2b3c4/a.py\n", [], id="abbreviated-commit"),
    pytest.param(f"{REPO}/blob/{A}1/a.py\n", [], id="41-hex-digits"),
]


@pytest.mark.parametrize(("text", "expected"), URL_TEXTS)
def test_pins_are_read_from_links_to_this_repository(text, expected):
    assert url_pins_in(text) == expected


FETCH = ".venv-natural/bin/python scripts/fetch_natural_data.py"
FETCH_TEXTS = [
    pytest.param(f"```bash\n{FETCH} --revision {A} --output data/natural\n```\n", [(2, A)], id="one-line"),
    pytest.param(
        f"```bash\n{FETCH} \\\n  --manifest manifest.json \\\n  --revision {A} \\\n  --list\n```\n",
        [(2, A)],
        id="continued-lines",
    ),
    pytest.param(f"```bash\n{FETCH} --revision={A}\n```\n", [(2, A)], id="equals-sign"),
    pytest.param(
        f"```bash\n{FETCH} --revision {A} --list\n\n{FETCH} \\\n  --revision {B}\n```\n",
        [(2, A), (4, B)],
        id="two-commands",
    ),
    pytest.param(f"```bash\nuv run python scripts/fetch_natural_data.py --revision {A}\n```\n", [(2, A)], id="uv-run"),
    pytest.param(f"```bash\npython fetch_natural_data.py --revision {A.upper()}\n```\n", [(2, A.upper())], id="upper"),
    pytest.param(
        f"```bash\npython scripts/selftrained/chat.py --repo someone/models --revision {A}\n```\n",
        [],
        id="hugging-face-revision-of-another-script",
    ),
    pytest.param(f"```bash\npython scripts/natural_assistant.py --model-revision {A}\n```\n", [], id="model-revision"),
    pytest.param(
        f"```bash\n{FETCH} --list\npython scripts/selftrained/chat.py \\\n  --revision {A}\n```\n",
        [],
        id="next-command-runs-another-script",
    ),
    pytest.param(
        f"```bash\n{FETCH} \\\n```\n\n```bash\n  --revision {A}\n```\n", [], id="continuation-does-not-cross-blocks"
    ),
    pytest.param(f"執行 `{FETCH} --revision {A}` 即可。\n", [], id="inline-code-in-prose"),
    pytest.param(f'```bash\n{FETCH} --revision "$REVISION"\n```\n', [], id="shell-variable"),
    pytest.param(
        f'```python\nsubprocess.run(["python", "scripts/fetch_natural_data.py", "--revision", "{A}"])\n```\n',
        [],
        id="python-list",
    ),
    pytest.param(f"```bash\n{FETCH} --revision {A[:39]}\n```\n", [], id="39-hex-digits"),
]


@pytest.mark.parametrize(("text", "expected"), FETCH_TEXTS)
def test_pins_are_read_from_the_revision_of_fetch_natural_data_commands(text, expected):
    assert fetch_pins_in(text) == expected


OTHER_REPOSITORY = "birdhackor/tiny-perceptron-course-models"
JSON_TEXTS = [
    pytest.param({"package": {"repo_id": REPOSITORY, "revision": A}}, [("package.revision", A)], id="nested-object"),
    pytest.param({"repo_id": REPOSITORY, "revision": A, "path": "a"}, [("revision", A)], id="top-level-object"),
    pytest.param(
        {"packages": [{"repo_id": REPOSITORY, "revision": A}, {"repo_id": REPOSITORY, "revision": B}]},
        [("packages.0.revision", A), ("packages.1.revision", B)],
        id="objects-in-a-list",
    ),
    pytest.param([[{"repo_id": REPOSITORY, "revision": A}]], [("0.0.revision", A)], id="list-of-lists"),
    pytest.param(
        {"release": {"data": {"package": {"repo_id": REPOSITORY, "revision": A}}}},
        [("release.data.package.revision", A)],
        id="deeply-nested-object",
    ),
    pytest.param({"revision": A, "repo_id": REPOSITORY}, [("revision", A)], id="revision-before-repo-id"),
    pytest.param({"model": {"repo_id": OTHER_REPOSITORY, "revision": A}}, [], id="another-repository"),
    pytest.param({"repo_id": REPOSITORY, "commit": A}, [], id="another-key"),
    pytest.param({"revision": A, "source": {"repo_id": REPOSITORY}}, [], id="repo-id-on-another-object"),
    pytest.param({"repo_id": REPOSITORY, "revision": "main"}, [], id="branch-name"),
    pytest.param({"note": f'"repo_id": "{REPOSITORY}", "revision": "{A}"'}, [], id="text-inside-a-string"),
]


@pytest.mark.parametrize(("value", "expected"), JSON_TEXTS)
def test_pins_are_read_from_json_objects_of_this_repository(value, expected):
    assert json_pins_in(json.dumps(value, indent=2)) == expected


DEFAULT_SOURCES = [
    pytest.param(f'DEFAULT_REVISION = "{A}"\n', [(1, A)], id="module-constant"),
    pytest.param(f'"""說明。"""\n\nimport re\n\nDEFAULT_REVISION: str = "{A}"\n', [(5, A)], id="annotated-constant"),
    pytest.param(f'# DEFAULT_REVISION = "{B}"\nDEFAULT_REVISION = "{A}"\n', [(2, A)], id="old-value-in-a-comment"),
    pytest.param(f'def main():\n    DEFAULT_REVISION = "{A}"\n', [], id="local-variable"),
    pytest.param(f'HELP = "DEFAULT_REVISION = {A}"\n', [], id="text-inside-a-string"),
    pytest.param(f'DEFAULT_MANIFEST_SHA256 = "{A}"\n', [], id="another-constant"),
]


@pytest.mark.parametrize(("source", "expected"), DEFAULT_SOURCES)
def test_the_default_revision_is_read_from_the_module_constant(source, expected):
    assert default_revision_pins(source) == expected


# ---- 頁面集合 ----------------------------------------------------------------------------------------------


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def fenced(command):
    return f"說明。\n\n```bash\n{command}\n```\n"


def test_pages_added_to_the_chapters_or_the_reading_list_are_scanned(tmp_path):
    reading_pages = {"docs/guide.md": "guide.html"}
    write(tmp_path, "course/chapters/01.md", fenced(f"git checkout {A}"))
    write(tmp_path, "docs/guide.md", fenced(f"git checkout {B}"))
    write(tmp_path, "docs/maintainer.md", fenced(f"git checkout {'c' * 40}"))
    assert {(page, commit) for page, _, commit in course_pins(tmp_path, reading_pages, extra_pages=())} == {
        ("course/chapters/01.md", A),
        ("docs/guide.md", B),
    }
    write(tmp_path, "course/chapters/0C.md", fenced(f"git checkout {'d' * 40}"))
    write(tmp_path, "docs/second.md", fenced(f"git checkout {'e' * 40}"))
    reading_pages["docs/second.md"] = "second.html"
    assert {commit for _, _, commit in course_pins(tmp_path, reading_pages, extra_pages=())} == {
        A,
        B,
        "d" * 40,
        "e" * 40,
    }
    write(tmp_path, "docs/second.md", f"[原始碼]({REPO}/blob/{'f' * 40}/a.py)\n")
    write(tmp_path, "docs/maintainer.md", f"[原始碼]({REPO}/blob/{'c' * 40}/a.py)\n")
    assert course_pins(tmp_path, reading_pages, extract=url_pins_in, extra_pages=()) == [
        ("docs/second.md", 1, "f" * 40)
    ]


def test_the_page_set_follows_relative_links_from_markdown_pages_only(tmp_path):
    root = tmp_path / "repo"
    write(tmp_path, "outside.md", "")
    write(
        root,
        "course/chapters/01.md",
        "[指引](../../docs/guide.md#安裝)、[清單](../../docs/data.json)、[外部](https://example.com/page.md)、"
        "[本頁](#1.1)、[repo 外](../../../outside.md)\n",
    )
    write(
        root,
        "docs/guide.md",
        "[更深](deeper/page.md) [圖](figure.svg) [不存在](missing.md)\n\n```bash\ncat [x](in-a-fence.md)\n```\n",
    )
    write(root, "docs/deeper/page.md", "[回去](../guide.md) [數值](values.json)\n")
    write(root, "docs/data.json", '{"see": "[x](from-json.md)"}')
    write(root, "docs/deeper/values.json", "{}")
    write(root, "README_en.md", "[English guide](docs/english.md)\n")
    for name in ("in-a-fence.md", "from-json.md", "english.md", "maintainer.md", "figure.svg"):
        write(root, f"docs/{name}", "")
    assert page_set(root, reading_pages={}, extra_pages=("README_en.md",)) == [
        "README_en.md",
        "course/chapters/01.md",
        "docs/data.json",
        "docs/deeper/page.md",
        "docs/deeper/values.json",
        "docs/english.md",
        "docs/guide.md",
    ]


def test_json_and_fetch_pins_come_from_the_linked_files(tmp_path):
    write(tmp_path, "course/chapters/01.md", "[資料](../../docs/data.md)\n")
    write(tmp_path, "docs/data.md", f"[清單](manifest.json)\n\n```bash\n{FETCH} \\\n  --revision {A}\n```\n")
    write(tmp_path, "docs/manifest.json", json.dumps({"package": {"repo_id": REPOSITORY, "revision": B}}))
    write(tmp_path, "docs/unlinked.json", json.dumps({"package": {"repo_id": REPOSITORY, "revision": "c" * 40}}))
    assert course_pins(tmp_path, {}, extract=fetch_pins_in, extra_pages=()) == [("docs/data.md", 4, A)]
    assert course_pins(tmp_path, {}, extract=json_pins_in, extra_pages=(), suffix=".json") == [
        ("docs/manifest.json", "package.revision", B)
    ]


def test_every_student_page_exists():
    pages = student_pages(ROOT)
    assert set(READING_PAGES) | set(EXTRA_PAGES) <= set(pages) and len(pages) > len(READING_PAGES)
    assert [page for page in pages if not (ROOT / page).is_file()] == []
    assert (ROOT / FETCH_SCRIPT).is_file()


# ---- 核對：暫存的 Git 歷史 ----------------------------------------------------------------------------------


def run_git(path, *args):
    """在暫存 repo 上執行 git；不讀使用者與系統的設定，避免模板 hook、簽章設定改變結果。"""
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "Pin test",
        "GIT_AUTHOR_EMAIL": "pin@example.invalid",
        "GIT_COMMITTER_NAME": "Pin test",
        "GIT_COMMITTER_EMAIL": "pin@example.invalid",
    }
    done = subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True, encoding="utf-8", env=environment
    )
    return done.stdout.strip()


@pytest.fixture
def history(tmp_path):
    """main 上有 first 與 head 兩個 commit；side 是另一條分支上、不在 HEAD 歷史裡的 commit。"""
    if shutil.which("git") is None:
        pytest.skip("找不到 git 指令")
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "--initial-branch=main")
    run_git(repo, "commit", "--allow-empty", "-m", "first")
    ids = {"first": run_git(repo, "rev-parse", "HEAD")}
    run_git(repo, "checkout", "-b", "side")
    run_git(repo, "commit", "--allow-empty", "-m", "side")
    ids["side"] = run_git(repo, "rev-parse", "HEAD")
    run_git(repo, "checkout", "main")
    run_git(repo, "commit", "--allow-empty", "-m", "second")
    ids["head"] = run_git(repo, "rev-parse", "HEAD")
    run_git(repo, "tag", "-a", "release", "-m", "annotated tag", ids["first"])
    ids["tag"] = run_git(repo, "rev-parse", "release")
    (repo / "note.txt").write_text("blob", encoding="utf-8")
    ids["blob"] = run_git(repo, "hash-object", "-w", "note.txt")
    ids["absent"] = "e" * 40
    return repo, ids


def test_ancestors_of_head_are_accepted(history):
    repo, ids = history
    pins = [("page.md", 3, ids["first"]), ("page.md", 9, ids["head"]), ("other.md", 1, ids["head"])]
    assert pin_problems(repo, pins) == []


@pytest.mark.parametrize(
    ("which", "reason"),
    [
        pytest.param("side", "不是 HEAD 的祖先", id="commit-on-another-branch"),
        pytest.param("tag", "不是這個 repo 的 commit 物件", id="annotated-tag-object"),
        pytest.param("blob", "不是這個 repo 的 commit 物件", id="blob"),
        pytest.param("absent", "不是這個 repo 的 commit 物件", id="object-that-does-not-exist"),
    ],
)
def test_commits_outside_the_history_of_head_and_other_objects_are_rejected(history, which, reason):
    repo, ids = history
    problems = pin_problems(repo, [("page.md", 7, ids[which])])
    assert len(problems) == 1
    assert ids[which] in problems[0] and "page.md:7" in problems[0] and reason in problems[0]


def test_one_problem_is_reported_for_each_hash_with_every_place_it_is_pinned(history):
    repo, ids = history
    pins = [("a.md", 1, ids["absent"]), ("b.md", 5, ids["absent"]), ("b.md", 6, ids["side"])]
    problems = pin_problems(repo, pins)
    assert len(problems) == 2
    assert "a.md:1" in problems[0] and "b.md:5" in problems[0]


def test_a_history_can_be_checked_only_when_it_is_complete(history, tmp_path, monkeypatch):
    repo, _ = history
    assert history_unavailable(repo) is None
    shallow = tmp_path / "shallow"
    run_git(tmp_path, "clone", "--quiet", "--depth", "1", "--branch", "main", repo.as_uri(), str(shallow))
    assert "淺 clone" in history_unavailable(shallow)
    outside = tmp_path / "plain"
    outside.mkdir()
    # 不讓 git 往上層目錄找到別的 repo。
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert "不在 Git 工作目錄" in history_unavailable(outside)


# ---- 真實教材 ----------------------------------------------------------------------------------------------


def check_real_pins(pins, label):
    reason = history_unavailable(ROOT)
    if reason:
        pytest.skip(reason)
    problems = pin_problems(ROOT, pins)
    if problems:
        pytest.fail(f"{label}釘住的 commit 不在目前歷史裡：\n" + "\n".join(problems), pytrace=False)


def test_student_pages_pin_a_commit_on_a_git_line():
    pins = course_pins(ROOT)
    assert pins, "學生頁面的 fenced 程式區塊裡找不到任何「git … <40 位 commit>」，沒有東西可以核對。"


def test_commits_pinned_on_git_lines_are_ancestors_of_head():
    pins = course_pins(ROOT)
    assert pins, "學生頁面的 fenced 程式區塊裡找不到任何「git … <40 位 commit>」，沒有東西可以核對。"
    check_real_pins(pins, "教材的 git 命令")


def test_commits_pinned_in_links_to_this_repository_are_ancestors_of_head():
    check_real_pins(course_pins(ROOT, extract=url_pins_in), "教材連到本 repo 的網址")


def test_revisions_given_to_fetch_natural_data_are_ancestors_of_head():
    pins = course_pins(ROOT, extract=fetch_pins_in)
    assert pins, "頁面集合裡找不到執行 fetch_natural_data.py 時帶 40 位 --revision 的命令，沒有東西可以核對。"
    check_real_pins(pins, "fetch_natural_data.py 命令的 --revision ")


def test_revisions_of_this_repository_in_linked_json_files_are_ancestors_of_head():
    pins = course_pins(ROOT, extract=json_pins_in, suffix=".json")
    assert pins, f"頁面集合的 .json 裡找不到 repo_id 為 {REPOSITORY} 的 revision，沒有東西可以核對。"
    check_real_pins(pins, "JSON 檔裡本 repo 的 revision ")


def test_the_default_revision_of_fetch_natural_data_is_an_ancestor_of_head():
    pins = default_revision_pins((ROOT / FETCH_SCRIPT).read_text(encoding="utf-8"))
    assert len(pins) == 1, f"{FETCH_SCRIPT} 應有一個模組層級的 DEFAULT_REVISION 字串常數。"
    check_real_pins([(FETCH_SCRIPT, number, value) for number, value in pins], "DEFAULT_REVISION ")
