"""學生頁面釘住的本 repo commit 必須是目前歷史裡的祖先。

改寫 Git 歷史後舊的 hash 會失效，教材卻仍叫讀者 `git checkout` 它，或連到 GitHub 上已經不存在的版本；
這裡在完整 clone 上抓出這種殘留。學生頁面 = course/chapters 的全部章檔，加上
scripts.build_course.READING_PAGES 列出的頁面。看兩種地方：

- fenced 程式區塊裡、和 `git ` 在同一行的 40 位十六進位字串。其他指令（例如腳本的 --revision 參數）上的
  hash 可能是 Hugging Face 的版本而不是這個 repo 的 commit，從文字分不出來，所以不在範圍內。
- 頁面任何位置、指向本 repo（github.com/birdhackor/tiny-perceptron-vlm）的 blob／tree／commit／raw 網址
  裡的 40 位 hash。封存 repo 與其他 repo 的網址不在範圍內。
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.build_course import READING_PAGES

ROOT = Path(__file__).resolve().parents[1]
COMMIT = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{40}(?![0-9A-Fa-f])")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
# GitHub 的帳號與 repo 名稱不分大小寫；repo 名稱後面必須緊接 `/`，封存 repo（…-archive）因此不算。
REPOSITORY_URL = re.compile(
    r"github\.com/birdhackor/tiny-perceptron-vlm/(?:blob|tree|commit|raw)/([0-9A-Fa-f]{40})(?![0-9A-Fa-f])",
    re.IGNORECASE,
)


def code_lines(text):
    """fenced 程式區塊裡的 (行號, 內容)；區塊外的文字不回傳。"""
    opened = None  # (圍欄字元, 圍欄長度)
    for number, line in enumerate(text.split("\n"), 1):
        fence = FENCE.match(line)
        if opened is None:
            if fence:
                opened = (fence[1][0], len(fence[1]))
        elif fence and fence[1][0] == opened[0] and len(fence[1]) >= opened[1] and not fence[2].strip():
            opened = None
        else:
            yield number, line


def pins_in(text):
    """(行號, hash)：fenced 程式區塊裡，和 `git ` 在同一行的 40 位十六進位字串。"""
    return [(number, commit) for number, line in code_lines(text) if "git " in line for commit in COMMIT.findall(line)]


def url_pins_in(text):
    """(行號, hash)：頁面任何位置指向本 repo 的 blob／tree／commit／raw 網址裡的 40 位 hash。"""
    return [
        (number, commit) for number, line in enumerate(text.split("\n"), 1) for commit in REPOSITORY_URL.findall(line)
    ]


def student_pages(root, reading_pages=READING_PAGES):
    chapters = sorted(path.relative_to(root).as_posix() for path in (root / "course/chapters").glob("*.md"))
    return chapters + list(reading_pages)


def course_pins(root, reading_pages=READING_PAGES, extract=pins_in):
    """[(頁面, 行號, hash)]：全部學生頁面裡，由 extract 抽出的 commit。"""
    return [
        (page, number, commit)
        for page in student_pages(root, reading_pages)
        for number, commit in extract((root / page).read_text(encoding="utf-8"))
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
        return "這是淺 clone（CI 預設如此），缺少歷史；改寫歷史前後請在完整 clone 跑全套測試"
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
    assert {(page, commit) for page, _, commit in course_pins(tmp_path, reading_pages)} == {
        ("course/chapters/01.md", A),
        ("docs/guide.md", B),
    }
    write(tmp_path, "course/chapters/0C.md", fenced(f"git checkout {'d' * 40}"))
    write(tmp_path, "docs/second.md", fenced(f"git checkout {'e' * 40}"))
    reading_pages["docs/second.md"] = "second.html"
    assert {commit for _, _, commit in course_pins(tmp_path, reading_pages)} == {A, B, "d" * 40, "e" * 40}
    write(tmp_path, "docs/second.md", f"[原始碼]({REPO}/blob/{'f' * 40}/a.py)\n")
    write(tmp_path, "docs/maintainer.md", f"[原始碼]({REPO}/blob/{'c' * 40}/a.py)\n")
    assert course_pins(tmp_path, reading_pages, extract=url_pins_in) == [("docs/second.md", 1, "f" * 40)]


def test_every_student_page_exists():
    pages = student_pages(ROOT)
    assert set(READING_PAGES) <= set(pages) and len(pages) > len(READING_PAGES)
    assert [page for page in pages if not (ROOT / page).is_file()] == []


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


def test_student_pages_pin_a_commit_on_a_git_line():
    pins = course_pins(ROOT)
    assert pins, "學生頁面的 fenced 程式區塊裡找不到任何「git … <40 位 commit>」，沒有東西可以核對。"


def test_commits_pinned_on_git_lines_are_ancestors_of_head():
    pins = course_pins(ROOT)
    assert pins, "學生頁面的 fenced 程式區塊裡找不到任何「git … <40 位 commit>」，沒有東西可以核對。"
    reason = history_unavailable(ROOT)
    if reason:
        pytest.skip(reason)
    problems = pin_problems(ROOT, pins)
    if problems:
        pytest.fail("教材釘住的 commit 不在目前歷史裡：\n" + "\n".join(problems), pytrace=False)


def test_commits_pinned_in_links_to_this_repository_are_ancestors_of_head():
    reason = history_unavailable(ROOT)
    if reason:
        pytest.skip(reason)
    problems = pin_problems(ROOT, course_pins(ROOT, extract=url_pins_in))
    if problems:
        pytest.fail("教材連到本 repo 的網址釘住的 commit 不在目前歷史裡：\n" + "\n".join(problems), pytrace=False)
