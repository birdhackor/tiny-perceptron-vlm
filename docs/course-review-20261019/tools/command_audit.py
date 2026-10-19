"""可觀察工具與命令稽核：列出 Codex 讀者實際下過的命令與看過的圖，標出可能越界的地方。

來源是 Codex 的 `--json` log（`command_execution` 與 `web_search` 事件）與原生 rollout 的
`tools.view_image` 呼叫。輸出只整理與標記，不產生審閱判斷；違規與否由協調者看過標記後決定。

驗證：命令字面裡出現的路徑與命令名稱、log 裡的網頁搜尋、rollout 裡 `tools.view_image` 的圖路徑。
不涵蓋：命令執行的程式在內部開了哪些檔（例如 reader.py 讀凍結檔、`python -c` 寫的程式）看不到，
這類命令標成 review，由協調者讀命令本身；用變數或萬用字元組出的路徑也看不到真正的值。
所以結果不能當成「完整隔離」的證明。
"""

import argparse
import hashlib
import json
import posixpath
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
# 讀者 session 在這個 repo 的 POSIX shell 裡執行，命令裡的絕對路徑以它為根。
SESSION_ROOT = ROOT.as_posix()
FORBIDDEN_TOP = ("course/", "notebooks/", "scripts/", "tiny_perceptron/", "tests/")
FORBIDDEN_IN_CODE = re.compile(r"(?<![\w.-])(?:course|notebooks|scripts|tiny_perceptron|tests)/")
HISTORY = {"log", "show", "diff", "grep", "blame", "cat-file", "ls-files", "ls-tree", "rev-list"}
SEARCH = {"rg", "grep", "egrep", "fgrep", "find", "fd", "ag"}
NETWORK = {"curl", "wget"}
INTERPRETERS = re.compile(r"(^|/)(python[0-9.]*|node|ruby|perl|bash|zsh|sh)$")
OPERATORS = {"|", "||", "&", "&&", ";", ";;", "(", ")", "<", ">", ">>", "<<", ">&", "<&", "|&"}
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
VIEW_IMAGE = re.compile(r"\btools\.view_image\(\s*\{\s*path\s*:\s*\"((?:[^\"\\]|\\.)*)\"")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def events(path):
    found = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            found.append(json.loads(line))
        except ValueError:
            continue
    return found


def inner_command(command):
    """Codex 以 `/bin/zsh -lc '<命令>'` 執行；拆開取得真正輸入的那一段。"""
    try:
        words = shlex.split(command)
    except ValueError:
        return command
    if len(words) == 3 and words[0].endswith(("/zsh", "/bash", "/sh")) and words[1] in ("-lc", "-c"):
        return words[2]
    return command


def without_assignments(words):
    """去掉命令前的環境變數設定（例如 PYTHONDONTWRITEBYTECODE=1），讓第一個字是程式。"""
    index = 0
    while index < len(words) and ASSIGNMENT.match(words[index]):
        index += 1
    return words[index:]


def simple_commands(text):
    """以 shell 運算子切成一段段簡單命令；切不開（引號不成對等）就整段當一個。"""
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return [text.split()]
    groups, current = [], []
    for token in tokens:
        if token in OPERATORS:
            if current:
                groups.append(current)
            current = []
        else:
            current.append(token)
    if current:
        groups.append(current)
    return groups


def path_token(token, root):
    """命令裡像路徑的字：回傳 ("repo", repo 相對路徑)、("outside", 原字) 或 None（不是路徑）。

    路徑照 POSIX 解讀，與稽核在哪個作業系統上跑無關；root 是 session 裡的 repo 根目錄。
    含空白、引號或大括號的字當成資料（例如寫進 checkpoint 的 JSON），不是路徑。
    """
    text = token.split("=", 1)[1] if token.startswith("--") and "=" in token else token
    if "/" not in text or any(ch in text for ch in " \t\n{}\"'"):
        return None
    # 不解析符號連結：.venv/bin/python 連到 repo 外的 Python，但命令寫的是 repo 內的路徑。
    candidate = posixpath.normpath(posixpath.join(root, text))
    if candidate == root:
        return "repo", "."
    if candidate.startswith(root + "/"):
        return "repo", candidate[len(root) + 1 :]
    return "outside", text


class ReaderPolicy:
    """首讀讀者的讀寫範圍；條文照讀者指示〈能看什麼〉〈不能看什麼〉。"""

    def __init__(self, task, freeze, plan, receipt):
        self.task = task
        name = task.removeprefix("/root/")
        frozen = json.loads((ROOT / freeze).read_bytes())
        self.figures = {ref["path"] for ref in frozen["figure_snapshots"].values()}
        self.pngs = {record["png"]["path"] for record in json.loads((ROOT / receipt).read_bytes())["records"]}
        self.arguments_only = {Path(freeze).as_posix(), Path(plan).as_posix()}
        self.allowed = (
            ".agents/skills/clear-tutorial/",
            "docs/course-revision-20261016/tools/reader.py",
            "docs/course-review-20261019/tools/check_reader_original.py",
            "docs/course-review-20261019/tools/executed_outputs.py",
            f"docs/course-review-20261019/reviews/{name}/",
        )

    def session_owner(self, path):
        parts = PurePosixPath(path).parts
        if len(parts) < 4:
            return None
        state = ROOT / "docs/course-revision-20261016/reader-sessions" / parts[3] / "state.json"
        if not state.exists():
            return None
        return json.loads(state.read_bytes()).get("reviewer_task")

    def path_findings(self, path, words):
        program = Path(words[0]).name if words else ""
        tool_arguments = any(word.endswith(("reader.py", "check_reader_original.py")) for word in words)
        if path.startswith(FORBIDDEN_TOP):
            return [("violation", "禁讀的教材或程式路徑")]
        if path in self.figures or path in self.pngs:
            return []
        if path in self.arguments_only:
            if tool_arguments or program in ("shasum", "sha256sum"):
                return []
            return [("violation", "凍結檔或閱讀計畫只能當工具參數或算 SHA，不能讀內容")]
        if path.startswith("docs/course-revision-20261016/reader-sessions/"):
            owner = self.session_owner(path)
            if owner == self.task:
                return []
            return [("violation" if owner else "review", f"不是自己的 session（owner={owner}）")]
        if path.startswith(self.allowed):
            return []
        if path.startswith(("docs/", "outputs/")):
            return [("violation", "允許範圍以外的 docs／outputs 路徑")]
        if path.startswith((".git/", ".github/")) or path in ("AGENTS.md", "README.md", "README_en.md"):
            return [("violation", "允許範圍以外的 repo 檔")]
        return [("review", "允許清單沒有列到的 repo 路徑")]

    def command_findings(self, words):
        findings = []
        program = Path(words[0]).name if words else ""
        if program == "git" and len(words) > 1 and words[1] in HISTORY:
            findings.append(("violation", "讀 Git 歷史或內容"))
        if program in NETWORK:
            findings.append(("violation", "連網讀取"))
        if program in SEARCH:
            findings.append(("review", "搜尋命令：看範圍是否只在自己的檔"))
        if program == "cd" and len(words) > 1:
            findings.append(("review", "切換目錄：之後的相對路徑無法逐一還原"))
        if words and INTERPRETERS.search(words[0]):
            for flag, code in zip(words[1:], words[2:]):
                if flag in ("-c", "-e"):
                    findings.append(("review", "直譯器執行字面程式：只看得到程式字面，看不到它實際開的檔"))
                    if FORBIDDEN_IN_CODE.search(code):
                        findings.append(("violation", "字面程式裡出現禁讀路徑"))
            if "-" in words[1:2]:
                findings.append(("review", "直譯器從標準輸入讀程式：程式內容要另外看"))
        return findings


def audit(policy, logs, rollouts, root=SESSION_ROOT):
    commands, searches, images = [], [], []
    for log in logs:
        for event in events(log):
            item = event.get("item") or {}
            if event.get("type") != "item.completed":
                continue
            if item.get("type") == "web_search":
                searches.append({"query": item.get("query"), "level": "violation"})
            if item.get("type") != "command_execution":
                continue
            text = inner_command(item.get("command") or "")
            findings = []
            if "<<" in text:
                findings.append({"level": "review", "reason": "here-document：內文要另外看"})
                if FORBIDDEN_IN_CODE.search(text):
                    findings.append({"level": "review", "reason": "here-document 內文提到禁讀路徑"})
            for words in simple_commands(text):
                words = without_assignments(words)
                findings += [{"level": level, "reason": reason} for level, reason in policy.command_findings(words)]
                for word in words[1:]:
                    if word == policy.task:
                        continue
                    found = path_token(word, root)
                    if found is None:
                        continue
                    kind, path = found
                    if kind == "outside":
                        findings.append({"level": "review", "reason": "repo 外的路徑", "path": path})
                        continue
                    findings += [
                        {"level": level, "reason": reason, "path": path}
                        for level, reason in policy.path_findings(path, words)
                    ]
            commands.append({"exit_code": item.get("exit_code"), "command": text, "findings": findings})
    for rollout in rollouts:
        for event in events(rollout):
            body = event.get("payload") or {}
            if body.get("type") != "custom_tool_call":
                continue
            code = body.get("input") or ""
            for raw in VIEW_IMAGE.findall(code):
                found = path_token(json.loads(f'"{raw}"'), root)
                path = found[1] if found else raw
                level = None if path in policy.pngs else "violation"
                images.append({"at": event.get("timestamp"), "path": path, "level": level})
            unparsed = code.count("tools.view_image(") - len(VIEW_IMAGE.findall(code))
            if unparsed:
                images.append({"at": event.get("timestamp"), "path": None, "level": "review", "count": unparsed})
    flagged = [entry for entry in commands if entry["findings"]]
    levels = [f["level"] for entry in flagged for f in entry["findings"]]
    levels += [s["level"] for s in searches] + [i["level"] for i in images if i["level"]]
    # 同一段字面程式常重複出現（例如每次寫 checkpoint）；按程式字面歸併，協調者每種讀一次。
    variants = {}
    for entry in flagged:
        for words in simple_commands(entry["command"]):
            words = without_assignments(words)
            for flag, code in zip(words[1:], words[2:]):
                if words and INTERPRETERS.search(words[0]) and flag in ("-c", "-e"):
                    variants[code] = variants.get(code, 0) + 1
    return {
        "commands_total": len(commands),
        "summary": {"violation": levels.count("violation"), "review": levels.count("review")},
        "inline_code_variants": [{"code": code, "count": count} for code, count in variants.items()],
        "flagged_commands": flagged,
        "web_search": searches,
        "view_image": images,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--role", required=True, choices=("reader",))
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--receipt", default="docs/course-review-20261019/figure-renders.json")
    parser.add_argument("--log", action="append", required=True, type=Path)
    parser.add_argument("--rollout", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = ReaderPolicy(args.task, args.freeze, args.plan, args.receipt)
    result = {
        "schema_version": 1,
        "kind": "可觀察工具與命令稽核",
        "task": args.task,
        "role": args.role,
        "generated_at": datetime.now(UTC).isoformat(),
        "limits": "只看得到命令字面與 rollout 的工具呼叫；程式內部開的檔、變數組出的路徑看不到。不是完整隔離的證明。",
        "sources": {
            "logs": [{"path": str(p.resolve()), "sha256": sha(p)} for p in args.log],
            "rollouts": [{"path": str(p.resolve()), "sha256": sha(p)} for p in args.rollout],
        },
        **audit(policy, args.log, args.rollout),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(json.dumps({"commands_total": result["commands_total"], **result["summary"]}, ensure_ascii=False))
    raise SystemExit(1 if result["summary"]["violation"] else 0)


if __name__ == "__main__":
    main()
