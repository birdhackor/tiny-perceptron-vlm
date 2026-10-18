"""學生照著做的頁面裡手抄的資料清單 SHA-256 與資料包 pointer，必須對得上 repo 裡的實際檔案與 Git 歷史。

清單或資料包換版後，頁面上的指紋若沒跟著改，學生照做時腳本會在核對指紋那一步停下；這裡在發布前抓出這種落差。

頁面集合（page_set）、fenced 程式區塊（code_blocks）與命令的續行規則（commands）沿用
tests/test_course_commit_pins.py。四條規則的驗證與不涵蓋範圍，寫在各自函式的說明裡：

- manifest_pins：命令同時帶的 `--manifest` 與 `--manifest-sha256`。
- heredoc_manifest_pins：heredoc 裡用 `assert` 核對檔案 SHA-256 的 Python。
- remote_manifest_problems：fetch_natural_data.py 下載時比對的、指定 Git 版本裡的清單。
- package_pointer_problems：頁面集合 .json 裡本 repo 資料包的 Git LFS pointer。

需要 Git 歷史的核對，在 blobs_unavailable 回報無法核對時（淺 clone、partial clone）跳過。

四條規則都不看的：正文（fence 外）與 inline code 裡的手抄指紋，例如 docs/selftrained/TRAINING.md 敘述裡的清單
SHA-256，以及模型卡範本 docs/selftrained/model-cards/repository-README.md 裡的清單與資料包指紋；敘述裡沒有機器讀得出的
「這是哪個檔案的指紋」。
"""

import ast
import hashlib
import json
import re
import shutil
import subprocess
import textwrap
import urllib.request
from pathlib import Path, PurePosixPath
from unittest import mock

import pytest

from scripts import fetch_natural_data
from tests.test_course_commit_pins import (
    FETCH_SCRIPT,
    REPOSITORY,
    RUNS_FETCH_SCRIPT,
    code_blocks,
    commands,
    git,
    history_unavailable,
    page_set,
    run_git,
    write,
)

ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"[0-9A-Fa-f]{64}")
SHELL_EXPANSION = re.compile(r"[$`]")
HEREDOC = re.compile(r"\bpython[0-9.]*\s+-\s*<<-?\s*(['\"]?)([A-Za-z_]\w*)\1\s*$")
RAW_MANIFEST_URL = re.compile(rf"https://raw\.githubusercontent\.com/{re.escape(REPOSITORY)}/([0-9a-f]{{40}})/(.+)")
LFS_POINTER = re.compile(
    r"version https://git-lfs\.github\.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (0|[1-9][0-9]*)\n"
)
# fetch_natural_data.py 不帶對應選項時用的清單路徑、revision 與清單 SHA-256。
SCRIPT_DEFAULTS = (
    fetch_natural_data.MANIFEST.relative_to(fetch_natural_data.ROOT).as_posix(),
    fetch_natural_data.DEFAULT_REVISION,
    fetch_natural_data.DEFAULT_MANIFEST_SHA256,
)


def page_items(root, extract, suffix=".md", **pages):
    """[(頁面, *項目)]：頁面集合裡副檔名為 suffix 的檔案，由 extract 抽出的每個項目。"""
    return [
        (page, *item)
        for page in page_set(root, **pages)
        if page.endswith(suffix)
        for item in extract((root / page).read_text(encoding="utf-8"))
    ]


def option_value(command, option):
    """命令裡最後一個 option 的值（argparse 以最後一個為準），去掉外層成對的引號；沒有這個選項時回傳 None。"""
    values = re.findall(rf"(?<!\S){re.escape(option)}(?:=|\s+)(\S+)", command)
    if not values:
        return None
    value = values[-1]
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"" else value


def local_file_problems(root, pins):
    """[(頁面, 位置, 路徑, SHA-256)] 的每一組：路徑是 repo 內的相對路徑、檔案存在，而且 SHA-256 逐字等於檔案的
    小寫 hexdigest（大寫也算不同）。同一組路徑與值只回報一次，附上所有出現的位置。"""
    places = {}
    for page, place, path, value in pins:
        places.setdefault((path, value), []).append(f"{page}:{place}")
    problems = []
    for (path, value), where in places.items():
        label = f"（{'、'.join(where)}）"
        target = (root / path).resolve()
        if Path(path).is_absolute() or PurePosixPath(path).is_absolute() or not target.is_relative_to(root.resolve()):
            problems.append(f"{path} 不是 repo 內的相對路徑{label}")
        elif not target.is_file():
            problems.append(f"找不到 {path}{label}")
        elif (actual := hashlib.sha256(target.read_bytes()).hexdigest()) != value:
            problems.append(f"{path} 的 SHA-256 是 {actual}，頁面寫的是 {value}{label}")
    return problems


# ---- 命令的 --manifest 與 --manifest-sha256 ------------------------------------------------------------------


def manifest_pins(text):
    """(命令第一行的行號, 清單路徑, SHA-256)：fenced 程式區塊裡同時帶 `--manifest <路徑>` 與
    `--manifest-sha256 <64 位十六進位>` 的命令；續行照 commands 接起來，選項重複時取最後一個。

    驗證：交給 local_file_problems，核對路徑（相對 repo 根目錄）上的檔案，SHA-256 逐字等於命令寫的值。
    不涵蓋：路徑或值是 shell 變數的命令、Python 清單形式的命令、只帶 `--manifest-sha256` 的命令（例如
    scripts/selftrained/chat.py 核對的推論清單從 Hugging Face 下載，不在本 repo）。
    """
    pins = []
    for number, command in commands(text):
        path, value = option_value(command, "--manifest"), option_value(command, "--manifest-sha256")
        if path and value and HEX64.fullmatch(value) and not SHELL_EXPANSION.search(path):
            pins.append((number, path, value))
    return pins


# ---- heredoc 裡的 assert ------------------------------------------------------------------------------------


def heredocs(text):
    """[(第一行的行號, Python 原始碼)]：fenced 程式區塊裡餵給 `python -` 的 heredoc 內容，去掉共同縮排。"""
    found = []
    for lines in code_blocks(text):
        index = 0
        while index < len(lines):
            start = HEREDOC.search(lines[index][1])
            index += 1
            if not start:
                continue
            body = []
            while index < len(lines) and lines[index][1].strip() != start[2]:
                body.append(lines[index])
                index += 1
            index += 1  # 跳過結束的分隔字
            if body:
                found.append((body[0][0], textwrap.dedent("\n".join(line for _, line in body))))
    return found


def is_text(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def literal_path(node, names):
    """node 是 Path("…")、Path.cwd()、<路徑> / "…" 或 names 裡的名稱時，回傳相對 repo 根目錄的路徑；否則 None。"""
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and is_text(node.right):
        base = literal_path(node.left, names)
        return None if base is None else base / node.right.value
    if isinstance(node, ast.Call) and not node.keywords:
        function = ast.unparse(node.func)
        if function in ("Path", "pathlib.Path") and len(node.args) == 1 and is_text(node.args[0]):
            return PurePosixPath(node.args[0].value)
        if function in ("Path.cwd", "pathlib.Path.cwd") and not node.args:
            return PurePosixPath()
    return None


def hashed_path(node, names, bare_sha256):
    """node 是 `hashlib.sha256(<路徑>.read_bytes()).hexdigest()` 時回傳 <路徑>；bare_sha256 為真時也認 `sha256(…)`。"""
    if not (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "hexdigest"
    ):
        return None
    digest = node.func.value
    if not (isinstance(digest, ast.Call) and len(digest.args) == 1 and not digest.keywords):
        return None
    function = ast.unparse(digest.func)
    if function != "hashlib.sha256" and not (function == "sha256" and bare_sha256):
        return None
    read = digest.args[0]
    if not (
        isinstance(read, ast.Call)
        and not read.args
        and not read.keywords
        and isinstance(read.func, ast.Attribute)
        and read.func.attr == "read_bytes"
    ):
        return None
    return literal_path(read.func.value, names)


def heredoc_manifest_pins(text):
    """(行號, 檔案路徑, SHA-256)：heredoc 頂層的
    `assert hashlib.sha256(<路徑>.read_bytes()).hexdigest() == "<64 位十六進位>"`；兩邊可以對調，
    有 `from hashlib import sha256` 時也認 `sha256(…)`。

    <路徑> 只認 Path("…")、Path.cwd()、用 / 接上字串的組合，以及同一段 heredoc 頂層、assert 之前最後一次被指派成
    這種組合的名稱。Path.cwd() 與相對路徑都當成 repo 根目錄，和頁面上其他命令一樣。

    驗證：交給 local_file_problems，核對路徑上的檔案，SHA-256 逐字等於 assert 寫的值。
    不涵蓋：函式、迴圈或 if 裡的 assert，路徑用其他寫法算出（例如由變數組字串）、名稱在 assert 之前被 for、with
    或 import 重新綁定，python fence 裡的程式（執行時的工作目錄不一定是 repo 根目錄），以及 `python -c`。
    """
    pins = []
    for start, source in heredocs(text):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        bare_sha256 = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "hashlib"
            and any(alias.name == "sha256" and alias.asname is None for alias in node.names)
            for node in tree.body
        )
        names = {}
        for statement in tree.body:
            if (
                isinstance(statement, ast.Assert)
                and isinstance(statement.test, ast.Compare)
                and len(statement.test.ops) == 1
                and isinstance(statement.test.ops[0], ast.Eq)
            ):
                sides = (statement.test.left, statement.test.comparators[0])
                for value, other in (sides, sides[::-1]):
                    path = hashed_path(other, names, bare_sha256) if is_text(value) else None
                    if path is not None and HEX64.fullmatch(value.value):
                        pins.append((start + statement.lineno - 1, path.as_posix(), value.value))
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                names[statement.targets[0].id] = literal_path(statement.value, names)
                continue
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.pop(node.id, None)
                elif isinstance(node, ast.alias):
                    names.pop(node.asname or node.name.split(".")[0], None)
    return pins


# ---- fetch_natural_data.py 會比對的遠端清單 --------------------------------------------------------------------


def fetch_settings(text, defaults):
    """(命令第一行的行號, 清單路徑, revision, 清單 SHA-256)：fenced 程式區塊裡執行 fetch_natural_data.py 的命令
    實際會用的三個值；沒寫的選項用 defaults（腳本的預設值）。任何一個值是 shell 變數的命令跳過。"""
    settings = []
    for number, command in commands(text):
        if not RUNS_FETCH_SCRIPT.search(command):
            continue
        values = [option_value(command, option) for option in ("--manifest", "--revision", "--manifest-sha256")]
        if any(value is not None and SHELL_EXPANSION.search(value) for value in values):
            continue
        filled = (default if value is None else value for value, default in zip(values, defaults, strict=True))
        settings.append((number, *filled))
    return settings


class Fetched(Exception):
    """假的 opener 攔下的下載請求；參數是網址。"""


def remote_manifest_location(root, manifest, revision, manifest_sha256, scratch):
    """(revision, 路徑)：fetch_natural_data.fetch_data 下載時第一個去取的 raw 清單。

    以假的 opener 攔下網址，不連網路；腳本在下載前就拒絕時，拋出它的 ValueError 或 OSError。
    """

    def opener(request, timeout=None):
        raise Fetched(request.full_url)

    def network(*args, **kwargs):
        raise AssertionError("測試不能連網路")

    with mock.patch.object(urllib.request, "urlopen", network):
        try:
            fetch_natural_data.fetch_data(
                root / manifest,
                Path(scratch).resolve() / "absent",
                revision=revision,
                manifest_sha256=manifest_sha256,
                opener=opener,
            )
        except Fetched as fetched:
            url = fetched.args[0]
        else:
            raise AssertionError("fetch_data 沒有下載任何東西就結束了")
    location = RAW_MANIFEST_URL.fullmatch(url)
    if location is None:
        raise AssertionError(f"fetch_data 第一個去取的不是本 repo 的 raw 清單：{url}")
    return location[1], location[2]


def blobs_unavailable(repo):
    """無法離線讀出舊版本檔案內容的原因；可以時回傳 None。

    除了 history_unavailable 的理由，partial clone（例如 `git clone --filter=blob:none`）缺少的檔案內容，
    git 讀取時會向 promisor remote 連網路補抓；測試不連網路，所以也跳過。partial clone 會設定
    `remote.<名稱>.promisor = true` 或 `extensions.partialClone`，兩個都當成記號。
    """
    reason = history_unavailable(repo)
    if reason:
        return reason
    promisors = git(repo, "config", "--type=bool", "--get-regexp", r"^remote\..*\.promisor$").stdout.split("\n")
    legacy = git(repo, "config", "--get", "extensions.partialClone").stdout.strip()
    if legacy or any(line.endswith(" true") for line in promisors):
        return "這是 partial clone，舊版本的檔案內容要連網路補抓；請在完整 clone 執行"
    return None


def git_blob(repo, revision, path):
    """`git cat-file blob <revision>:<path>`；stdout 是那個版本裡這個檔案的原始 bytes。"""
    return subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{revision}:{path}"], capture_output=True)


def remote_manifest_problems(repo, settings, scratch):
    """[(頁面, 位置, 清單路徑, revision, 清單 SHA-256)] 的每一組：fetch_natural_data.py 下載時，會到 revision 那個版本
    取 raw 清單，要求它的 SHA-256 等於這個值。同一組值只回報一次，附上所有出現的位置。

    驗證：用腳本自己的 fetch_data 算出它會去取的版本與路徑，再以 git_blob 取出那個版本的清單比對 SHA-256。
    路徑由腳本依清單列出的資料包決定，不一定是 `--manifest` 寫的路徑。帶 `--list`、`--verify` 的命令一樣核對：
    這兩種模式不下載，但印出的就是這組版本與指紋。
    不涵蓋：清單 archives 列出的資料包在那個版本的 LFS pointer、GitHub 上的實際內容（測試不連網路），以及
    fetch_settings 跳過的命令。
    """
    places = {}
    for page, place, manifest, revision, digest in settings:
        places.setdefault((manifest, revision, digest), []).append(f"{page}:{place}")
    problems = []
    for (manifest, revision, digest), where in places.items():
        label = f"（{'、'.join(where)}）"
        try:
            remote_revision, remote_path = remote_manifest_location(repo, manifest, revision, digest, scratch)
        except (ValueError, OSError) as error:
            problems.append(f"fetch_natural_data.py 在下載前就停下{label}：{error}")
            continue
        shown = git_blob(repo, remote_revision, remote_path)
        if shown.returncode != 0:
            reason = shown.stderr.decode("utf-8", "replace").strip()
            problems.append(f"{remote_revision} 版本取不到 {remote_path}{label}：{reason}")
        elif (actual := hashlib.sha256(shown.stdout).hexdigest()) != digest:
            problems.append(f"{remote_revision} 版本的 {remote_path} SHA-256 是 {actual}，頁面寫的是 {digest}{label}")
    return problems


# ---- JSON 裡本 repo 資料包的 LFS pointer --------------------------------------------------------------------


def package_pins(text):
    """(JSON 路徑, revision, path, sha256, bytes)：解析後 `repo_id` 等於本 repo，而且有 revision、path、sha256、
    bytes 四個鍵的物件。"""
    pins = []

    def visit(value, path):
        if isinstance(value, dict):
            if value.get("repo_id") == REPOSITORY and {"revision", "path", "sha256", "bytes"} <= value.keys():
                pins.append((".".join(path), value["revision"], value["path"], value["sha256"], value["bytes"]))
            for key, child in value.items():
                visit(child, [*path, str(key)])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, [*path, str(index)])

    visit(json.loads(text), [])
    return pins


def package_pointer_problems(repo, packages):
    """[(頁面, JSON 路徑, revision, path, sha256, bytes)] 的每一個資料包：path 在 revision 那個版本必須是
    Git LFS pointer，oid 等於 sha256、size 等於 bytes。

    驗證：git_blob 取出的內容逐字等於標準的 pointer 三行（`version https://git-lfs.github.com/spec/v1`、
    `oid sha256:<sha256>`、`size <bytes>`，各以換行結尾）；scripts/selftrained/hf_transport.py 的
    materialize_git_lfs 也是逐字比對。不符時指出是取不到、不是標準 pointer，還是 oid／size 不同。
    不涵蓋：LFS 伺服器上有沒有這個物件（測試不連網路）；revision 是否在目前歷史裡，由
    tests/test_course_commit_pins.py 核對；沒有 `repo_id` 與 `revision` 的資料包紀錄，例如
    fetch_natural_data.py 清單裡的 archives。
    """
    problems = []
    for page, place, revision, path, sha256, size in packages:
        label = f"{page}:{place}"
        shown = git_blob(repo, revision, path)
        if shown.returncode != 0:
            reason = shown.stderr.decode("utf-8", "replace").strip()
            problems.append(f"{label}：{revision} 版本取不到 {path}：{reason}")
            continue
        pointer = shown.stdout.decode("utf-8", "replace")
        if pointer == f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n":
            continue
        parsed = LFS_POINTER.fullmatch(pointer)
        if parsed is None:
            problems.append(f"{label}：{revision}:{path} 不是標準的 Git LFS pointer：{pointer[:200]!r}")
            continue
        differences = [
            f"{name} 是 {actual}，JSON 寫的是 {expected}"
            for name, actual, expected in (("oid", parsed[1], sha256), ("size", parsed[2], str(size)))
            if actual != expected
        ]
        problems.append(f"{label}：{revision}:{path} 的 pointer " + "；".join(differences))
    return problems


# ---- 抽取器：正例與反例 ---------------------------------------------------------------------------------------

S = "3e" * 32
T = "0c" * 32
M = "docs/data/manifest.json"
R1 = "1a2b3c4d5e" * 4
R2 = "f0e1d2c3b4" * 4
TRAIN = "uv run python scripts/selftrained/train_local_stage.py"
MANIFEST_TEXTS = [
    pytest.param(f"```bash\n{TRAIN} --manifest {M} --manifest-sha256 {S}\n```\n", [(2, M, S)], id="one-line"),
    pytest.param(
        f"```bash\n{TRAIN} \\\n  --manifest {M} \\\n  --manifest-sha256 {S} \\\n  --data-root data\n```\n",
        [(2, M, S)],
        id="continued-lines",
    ),
    pytest.param(f"```bash\n{TRAIN} --manifest={M} --manifest-sha256={S}\n```\n", [(2, M, S)], id="equals-signs"),
    pytest.param(f'```bash\n{TRAIN} --manifest-sha256 {S} --manifest "{M}"\n```\n', [(2, M, S)], id="quoted-path"),
    pytest.param(
        f"```bash\n{TRAIN} --manifest old.json --manifest-sha256 {T} --manifest {M} --manifest-sha256 {S}\n```\n",
        [(2, M, S)],
        id="last-option-wins",
    ),
    pytest.param(
        f"```bash\n{TRAIN} --manifest {M} --manifest-sha256 {S.upper()}\n```\n",
        [(2, M, S.upper())],
        id="upper-case-digest",
    ),
    pytest.param(
        f"```bash\n{TRAIN} --manifest {M} --manifest-sha256 {S}\n\n{TRAIN} \\\n  --manifest {M} --manifest-sha256 {T}\n```\n",
        [(2, M, S), (4, M, T)],
        id="two-commands",
    ),
    pytest.param(
        f"```bash\npython scripts/selftrained/chat.py --repo someone/models --manifest-sha256 {S}\n```\n",
        [],
        id="digest-without-a-manifest-path",
    ),
    pytest.param(f"```bash\n{TRAIN} --manifest {M}\n```\n", [], id="manifest-path-without-a-digest"),
    pytest.param(f"```bash\n{TRAIN} --manifest {M} --manifest-sha256 {S[:63]}\n```\n", [], id="63-hex-digits"),
    pytest.param(f'```bash\n{TRAIN} --manifest {M} --manifest-sha256 "$SHA"\n```\n', [], id="digest-in-a-variable"),
    pytest.param(f'```bash\n{TRAIN} --manifest "$MANIFEST" --manifest-sha256 {S}\n```\n', [], id="path-in-a-variable"),
    pytest.param(
        f"```bash\n{TRAIN} --manifest {M} \\\n```\n\n```bash\n  --manifest-sha256 {S}\n```\n",
        [],
        id="continuation-does-not-cross-blocks",
    ),
    pytest.param(f"執行 `{TRAIN} --manifest {M} --manifest-sha256 {S}` 即可。\n", [], id="inline-code-in-prose"),
]


@pytest.mark.parametrize(("text", "expected"), MANIFEST_TEXTS)
def test_manifest_pins_are_read_from_commands_with_both_options(text, expected):
    assert manifest_pins(text) == expected


def heredoc(*lines, start="uv run python - <<'PY'", end="PY"):
    return "```bash\n" + "\n".join([start, *lines, end]) + "\n```\n"


def sha_of(path_expression):
    return f"hashlib.sha256({path_expression}.read_bytes()).hexdigest()"


HEREDOC_TEXTS = [
    pytest.param(
        heredoc(
            "import hashlib",
            "from pathlib import Path",
            "",
            "repo = Path.cwd()",
            f'manifest_path = repo / "{M}"',
            f'assert {sha_of("manifest_path")} == "{S}"',
        ),
        [(8, M, S)],
        id="path-built-from-the-working-directory",
    ),
    pytest.param(
        heredoc(
            "from hashlib import sha256",
            "from pathlib import Path",
            f'assert sha256(Path("{M}").read_bytes()).hexdigest() == "{S}"',
        ),
        [(5, M, S)],
        id="sha256-imported-from-hashlib",
    ),
    pytest.param(
        heredoc("import hashlib", f'assert "{S}" == {sha_of(f"Path({M!r})")}'), [(4, M, S)], id="digest-on-the-left"
    ),
    pytest.param(
        heredoc(
            "import hashlib",
            'base = Path("docs")',
            'manifest = base / "data" / "manifest.json"',
            f'assert {sha_of("manifest")} == "{S}", "清單不同"',
        ),
        [(6, M, S)],
        id="path-joined-in-steps",
    ),
    pytest.param(
        heredoc("import hashlib", f'assert {sha_of(f"Path({M!r})")} == "{S}"', start="python3 - <<EOF", end="EOF"),
        [(4, M, S)],
        id="unquoted-delimiter",
    ),
    pytest.param(
        "<details>\n\n  ```bash\n  python - <<'PY'\n  import hashlib\n"
        f'  assert {sha_of(f"Path({M!r})")} == "{S}"\n  PY\n  ```\n\n</details>\n',
        [(6, M, S)],
        id="indented-fence",
    ),
    pytest.param(
        "```python\nimport hashlib\n" + f'assert {sha_of(f"Path({M!r})")} == "{S}"\n```\n', [], id="python-fence"
    ),
    pytest.param(
        heredoc("import hashlib", "def check():", f'    assert {sha_of(f"Path({M!r})")} == "{S}"'),
        [],
        id="assert-inside-a-function",
    ),
    pytest.param(
        heredoc(
            "import hashlib",
            f"manifest = Path({M!r})",
            "for manifest in sorted(Path('docs').glob('*.json')):",
            "    pass",
            f'assert {sha_of("manifest")} == "{S}"',
        ),
        [],
        id="name-rebound-by-a-loop",
    ),
    pytest.param(
        heredoc(f"manifest = Path({M!r})", "from somewhere import manifest", f'assert {sha_of("manifest")} == "{S}"'),
        [],
        id="name-rebound-by-an-import",
    ),
    pytest.param(
        heredoc(f'assert sha256(Path("{M}").read_bytes()).hexdigest() == "{S}"'), [], id="sha256-not-from-hashlib"
    ),
    pytest.param(heredoc("import hashlib", f'assert {sha_of(f"Path({M!r})")} != "{S}"'), [], id="not-equal"),
    pytest.param(
        heredoc("import hashlib", "import sys", f'assert {sha_of("Path(sys.argv[1])")} == "{S}"'),
        [],
        id="path-from-a-variable",
    ),
    pytest.param(heredoc("import hashlib", f'assert {sha_of(f"Path({M!r})")} == "{R1}"'), [], id="40-hex-digits"),
    pytest.param(
        "```bash\nuv run python - <<'PY'\nimport hashlib\nPY\n" + f'assert {sha_of(f"Path({M!r})")} == "{S}"\n```\n',
        [],
        id="line-after-the-delimiter",
    ),
]


@pytest.mark.parametrize(("text", "expected"), HEREDOC_TEXTS)
def test_manifest_pins_are_read_from_asserts_in_python_heredocs(text, expected):
    assert heredoc_manifest_pins(text) == expected


DEFAULTS = ("docs/default.json", R1, S)
FETCH = ".venv-natural/bin/python scripts/fetch_natural_data.py"
FETCH_SETTING_TEXTS = [
    pytest.param(
        f"```bash\n{FETCH} --manifest {M} --revision {R2} --manifest-sha256 {T} --list\n```\n",
        [(2, M, R2, T)],
        id="all-three-options",
    ),
    pytest.param(f"```bash\n{FETCH} --list\n```\n", [(2, *DEFAULTS)], id="script-defaults"),
    pytest.param(
        f"```bash\n{FETCH} \\\n  --revision={R2} \\\n  --output data\n```\n",
        [(2, DEFAULTS[0], R2, DEFAULTS[2])],
        id="some-options-on-continued-lines",
    ),
    pytest.param(f'```bash\n{FETCH} --revision "$REVISION"\n```\n', [], id="shell-variable"),
    pytest.param(f"```bash\npython scripts/other.py --manifest {M} --revision {R2}\n```\n", [], id="another-script"),
    pytest.param(f"執行 `{FETCH} --list` 即可。\n", [], id="inline-code-in-prose"),
]


@pytest.mark.parametrize(("text", "expected"), FETCH_SETTING_TEXTS)
def test_fetch_settings_fill_missing_options_with_the_script_defaults(text, expected):
    assert fetch_settings(text, DEFAULTS) == expected


OTHER_REPOSITORY = "birdhackor/tiny-perceptron-course-models"
PACKAGE = {"repo_id": REPOSITORY, "revision": R1, "path": "assets/training/a.tar.gz", "sha256": S, "bytes": 5}
PACKAGE_JSON = [
    pytest.param({"package": PACKAGE}, [("package", R1, "assets/training/a.tar.gz", S, 5)], id="nested-object"),
    pytest.param(
        {"packages": [PACKAGE, {**PACKAGE, "revision": R2}]},
        [("packages.0", R1, "assets/training/a.tar.gz", S, 5), ("packages.1", R2, "assets/training/a.tar.gz", S, 5)],
        id="objects-in-a-list",
    ),
    pytest.param({"package": {**PACKAGE, "repo_id": f"{REPOSITORY}-archive"}}, [], id="archive-repository"),
    pytest.param({"package": {**PACKAGE, "repo_id": OTHER_REPOSITORY}}, [], id="another-repository"),
    pytest.param({"package": {key: PACKAGE[key] for key in PACKAGE if key != "bytes"}}, [], id="without-bytes"),
    pytest.param({"package": {key: PACKAGE[key] for key in PACKAGE if key != "repo_id"}}, [], id="without-repo-id"),
]


@pytest.mark.parametrize(("value", "expected"), PACKAGE_JSON)
def test_packages_are_read_from_json_objects_of_this_repository(value, expected):
    assert package_pins(json.dumps(value, indent=2)) == expected


# ---- 核對：本機檔案 ----------------------------------------------------------------------------------------


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_a_pinned_digest_matches_when_it_is_the_sha256_of_the_file(tmp_path):
    write(tmp_path, M, '{"rows": []}\n')
    assert local_file_problems(tmp_path, [("a.md", 3, M, digest(tmp_path / M))]) == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param("one-byte", "的 SHA-256 是", id="one-byte-of-the-file-changed"),
        pytest.param("upper-case", "的 SHA-256 是", id="upper-case-digest"),
        pytest.param("missing", "找不到", id="file-does-not-exist"),
        pytest.param("outside", "不是 repo 內的相對路徑", id="path-outside-the-repository"),
        pytest.param("absolute", "不是 repo 內的相對路徑", id="absolute-path"),
    ],
)
def test_a_pinned_digest_that_is_not_the_sha256_of_a_file_in_the_repository_is_rejected(tmp_path, change, reason):
    root = tmp_path / "repo"
    write(root, M, '{"rows": []}\n')
    path, value = M, digest(root / M)
    if change == "one-byte":
        write(root, M, '{"rowz": []}\n')
    elif change == "upper-case":
        value = value.upper()
    elif change == "missing":
        path = "docs/data/missing.json"
    elif change == "outside":
        write(tmp_path, "outside.json", '{"rows": []}\n')
        path = "../outside.json"
    else:
        path = str(root / M)
    problems = local_file_problems(root, [("a.md", 3, path, value)])
    assert len(problems) == 1 and reason in problems[0] and "a.md:3" in problems[0]


def test_one_problem_is_reported_for_each_pinned_file_with_every_place(tmp_path):
    write(tmp_path, M, "{}\n")
    pins = [("a.md", 3, M, S), ("b.md", 8, M, S), ("b.md", 9, M, digest(tmp_path / M))]
    problems = local_file_problems(tmp_path, pins)
    assert len(problems) == 1 and "a.md:3" in problems[0] and "b.md:8" in problems[0]


# ---- 核對：暫存的 Git 歷史 ----------------------------------------------------------------------------------

V3_MANIFEST = "docs/natural-assistant/manifest.json"
V4_MANIFEST = "docs/natural-assistant/v4/manifest.json"


def natural_manifest(archives, version):
    """fetch_natural_data.validate_manifest 接受的最小清單：每個資料包一個檔案、沒有資料列。"""
    files, entries = [], []
    for path, folder in archives.items():
        item = {"path": f"{folder}/a.txt", "bytes": 1, "sha256": "a" * 64}
        files.append(item)
        entries.append({"path": path, "bytes": 1, "sha256": "b" * 64, "files": [item]})
    return {
        "schema_version": 1,
        "dataset_version": version,
        "files": files,
        "archives": entries,
        "rows": [],
        "audio_rows": [],
    }


def git_repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("找不到 git 指令")
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "--initial-branch=main")
    run_git(repo, "commit", "--allow-empty", "-m", "empty")
    return repo, {"empty": run_git(repo, "rev-parse", "HEAD")}


def commit(repo, message):
    run_git(repo, "add", "--all")
    run_git(repo, "commit", "-m", message)
    return run_git(repo, "rev-parse", "HEAD")


@pytest.fixture
def natural_history(tmp_path):
    """empty：還沒有清單；data：v3 與 v4 清單都在；changed：之後又改了一版 v4 清單。工作目錄停在 data 的內容。"""
    repo, ids = git_repository(tmp_path)
    write(repo, V3_MANIFEST, json.dumps(natural_manifest(fetch_natural_data.ARCHIVES, "v3")))
    write(repo, V4_MANIFEST, json.dumps(natural_manifest(fetch_natural_data.V4_ARCHIVES, "v4")))
    ids["data"] = commit(repo, "data")
    original = (repo / V4_MANIFEST).read_text(encoding="utf-8")
    write(repo, V4_MANIFEST, json.dumps(natural_manifest(fetch_natural_data.V4_ARCHIVES, "v4-changed")))
    ids["changed"] = commit(repo, "changed")
    write(repo, V4_MANIFEST, original)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return repo, ids, scratch


def test_a_revision_that_holds_the_same_manifests_passes(natural_history):
    repo, ids, scratch = natural_history
    settings = [
        ("a.md", 3, V3_MANIFEST, ids["data"], digest(repo / V3_MANIFEST)),
        ("a.md", 9, V4_MANIFEST, ids["data"], digest(repo / V4_MANIFEST)),
    ]
    assert remote_manifest_problems(repo, settings, scratch) == []


def test_the_manifest_is_compared_at_the_path_the_script_downloads(natural_history):
    repo, ids, scratch = natural_history
    write(repo, "copies/v4.json", (repo / V4_MANIFEST).read_text(encoding="utf-8"))
    value = digest(repo / "copies/v4.json")
    assert remote_manifest_problems(repo, [("a.md", 3, "copies/v4.json", ids["data"], value)], scratch) == []
    [problem] = remote_manifest_problems(repo, [("a.md", 3, "copies/v4.json", ids["changed"], value)], scratch)
    assert f"{ids['changed']} 版本的 {V4_MANIFEST} SHA-256 是" in problem and "a.md:3" in problem


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        pytest.param("empty", "版本取不到", id="revision-without-the-manifest"),
        pytest.param("edited", "SHA-256 是", id="manifest-edited-after-the-revision"),
        pytest.param("digest", "在下載前就停下", id="local-manifest-differs-from-the-digest"),
    ],
)
def test_a_revision_that_does_not_hold_the_pinned_manifest_is_rejected(natural_history, case, reason):
    repo, ids, scratch = natural_history
    revision, value = ids["data"], digest(repo / V3_MANIFEST)
    if case == "empty":
        revision = ids["empty"]
    elif case == "edited":
        write(repo, V3_MANIFEST, json.dumps(natural_manifest(fetch_natural_data.ARCHIVES, "v3-edited")))
        value = digest(repo / V3_MANIFEST)
    else:
        value = T
    problems = remote_manifest_problems(repo, [("a.md", 3, V3_MANIFEST, revision, value)], scratch)
    assert len(problems) == 1 and reason in problems[0] and "a.md:3" in problems[0]


def test_one_problem_is_reported_for_each_fetch_setting_with_every_place(natural_history):
    repo, ids, scratch = natural_history
    setting = (V3_MANIFEST, ids["empty"], digest(repo / V3_MANIFEST))
    problems = remote_manifest_problems(repo, [("a.md", 3, *setting), ("b.md", 7, *setting)], scratch)
    assert len(problems) == 1 and "a.md:3" in problems[0] and "b.md:7" in problems[0]


POINTER_PATH = "assets/training/data.tar.gz"


def lfs_pointer(oid, size, newline="\n"):
    return newline.join(["version https://git-lfs.github.com/spec/v1", f"oid sha256:{oid}", f"size {size}", ""])


@pytest.fixture
def pointer_history(tmp_path):
    """一個 commit，裡面有標準 pointer、CRLF 換行的 pointer，和一個不是 pointer 的壓縮包。"""
    repo, _ = git_repository(tmp_path)
    for path, content in (
        (POINTER_PATH, lfs_pointer(S, 123).encode()),
        ("assets/training/crlf.tar.gz", lfs_pointer(S, 123, "\r\n").encode()),
        ("assets/training/raw.tar.gz", b"\x1f\x8b\x08\x00 not a pointer"),
    ):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(content)
    return repo, commit(repo, "packages")


def test_a_package_matches_the_lfs_pointer_at_its_revision(pointer_history):
    repo, revision = pointer_history
    assert package_pointer_problems(repo, [("a.json", "package", revision, POINTER_PATH, S, 123)]) == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param({"sha256": T}, f"oid 是 {S}，JSON 寫的是 {T}", id="another-oid"),
        pytest.param({"bytes": 124}, "size 是 123，JSON 寫的是 124", id="another-size"),
        pytest.param({"path": "assets/training/raw.tar.gz"}, "不是標準的 Git LFS pointer", id="not-a-pointer"),
        pytest.param({"path": "assets/training/crlf.tar.gz"}, "不是標準的 Git LFS pointer", id="crlf-pointer"),
        pytest.param({"path": "assets/training/absent.tar.gz"}, "版本取不到", id="path-missing-at-the-revision"),
        pytest.param({"revision": "e" * 40}, "版本取不到", id="revision-that-does-not-exist"),
    ],
)
def test_a_package_that_differs_from_its_lfs_pointer_is_rejected(pointer_history, change, reason):
    repo, revision = pointer_history
    package = {"revision": revision, "path": POINTER_PATH, "sha256": S, "bytes": 123, **change}
    pins = [("a.json", "package", package["revision"], package["path"], package["sha256"], package["bytes"])]
    problems = package_pointer_problems(repo, pins)
    assert len(problems) == 1 and reason in problems[0] and "a.json:package" in problems[0]


def test_the_checks_read_pins_from_the_page_set(tmp_path):
    write(tmp_path, "course/chapters/01.md", "[資料](../../docs/data.md)\n")
    write(
        tmp_path,
        "docs/data.md",
        f"[清單](package.json)\n\n```bash\n{TRAIN} --manifest {M} --manifest-sha256 {S}\n```\n\n"
        + heredoc("import hashlib", f'assert {sha_of(f"Path({M!r})")} == "{T}"')
        + f"\n```bash\n{FETCH} --list\n```\n",
    )
    write(tmp_path, "docs/package.json", json.dumps({"package": PACKAGE}))
    write(tmp_path, "docs/unlinked.md", f"```bash\n{TRAIN} --manifest {M} --manifest-sha256 {T}\n```\n")
    pages = {"reading_pages": {}, "extra_pages": ()}
    assert page_items(tmp_path, manifest_pins, **pages) == [("docs/data.md", 4, M, S)]
    assert page_items(tmp_path, heredoc_manifest_pins, **pages) == [("docs/data.md", 10, M, T)]
    assert page_items(tmp_path, lambda text: fetch_settings(text, DEFAULTS), **pages) == [
        ("docs/data.md", 15, *DEFAULTS)
    ]
    assert page_items(tmp_path, package_pins, suffix=".json", **pages) == [
        ("docs/package.json", "package", R1, "assets/training/a.tar.gz", S, 5)
    ]


@pytest.mark.parametrize(
    ("key", "value", "partial"),
    [
        pytest.param("remote.origin.promisor", "true", True, id="promisor-remote"),
        pytest.param("extensions.partialClone", "origin", True, id="partial-clone-extension"),
        pytest.param("remote.origin.promisor", "false", False, id="remote-that-is-not-a-promisor"),
    ],
)
def test_old_file_contents_are_read_only_from_a_complete_clone(pointer_history, key, value, partial):
    repo, _ = pointer_history
    assert blobs_unavailable(repo) is None
    run_git(repo, "config", key, value)
    reason = blobs_unavailable(repo)
    assert ("partial clone" in reason) if partial else reason is None


def test_old_file_contents_are_not_read_when_the_history_is_unavailable(pointer_history, monkeypatch):
    repo, _ = pointer_history
    monkeypatch.setattr(shutil, "which", lambda name: None)
    reason = history_unavailable(repo)
    assert reason and blobs_unavailable(repo) == reason


# ---- 真實教材 ----------------------------------------------------------------------------------------------


def fail_with(problems, label):
    if problems:
        pytest.fail(f"{label}：\n" + "\n".join(problems), pytrace=False)


def test_manifest_digests_given_to_commands_are_the_sha256_of_the_files():
    pins = page_items(ROOT, manifest_pins)
    assert pins, "頁面集合裡找不到同時帶 --manifest 與 64 位 --manifest-sha256 的命令，沒有東西可以核對。"
    fail_with(local_file_problems(ROOT, pins), "命令的 --manifest-sha256 不是 --manifest 檔案的 SHA-256")


def test_manifest_digests_asserted_in_heredocs_are_the_sha256_of_the_files():
    pins = page_items(ROOT, heredoc_manifest_pins)
    assert pins, "頁面集合的 heredoc 裡找不到核對清單 SHA-256 的 assert，沒有東西可以核對。"
    fail_with(local_file_problems(ROOT, pins), "heredoc 的 assert 寫的不是檔案的 SHA-256")


def test_fetch_commands_find_their_pinned_manifest_at_their_revision(tmp_path):
    settings = page_items(ROOT, lambda text: fetch_settings(text, SCRIPT_DEFAULTS))
    assert settings, "頁面集合裡找不到執行 fetch_natural_data.py 的命令，沒有東西可以核對。"
    reason = blobs_unavailable(ROOT)
    if reason:
        pytest.skip(reason)
    fail_with(remote_manifest_problems(ROOT, settings, tmp_path), "照頁面上的命令執行 fetch_natural_data.py 會停下")


def test_the_defaults_of_fetch_natural_data_find_their_manifest_at_the_default_revision(tmp_path):
    reason = blobs_unavailable(ROOT)
    if reason:
        pytest.skip(reason)
    settings = [(FETCH_SCRIPT, "預設值", *SCRIPT_DEFAULTS)]
    fail_with(remote_manifest_problems(ROOT, settings, tmp_path), "不帶選項執行 fetch_natural_data.py 會停下")


def test_packages_of_this_repository_match_their_lfs_pointers():
    packages = page_items(ROOT, package_pins, suffix=".json")
    assert packages, f"頁面集合的 .json 裡找不到 repo_id 為 {REPOSITORY} 的資料包紀錄，沒有東西可以核對。"
    reason = blobs_unavailable(ROOT)
    if reason:
        pytest.skip(reason)
    fail_with(package_pointer_problems(ROOT, packages), "資料包紀錄和它釘的版本裡的 Git LFS pointer 不同")
