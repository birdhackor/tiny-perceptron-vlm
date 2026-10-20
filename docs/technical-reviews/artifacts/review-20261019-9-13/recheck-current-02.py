"""重跑小節並保存依賴變動、來源與既有證據完整性的收據。"""
import ast
import contextlib
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    for index, header in enumerate(headers):
        if header[0].startswith(("## " + identifier + " ").encode()):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
            return raw[header.start():end]
    raise ValueError("找不到指定小節")


class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def fetch(url, locator, excerpt, required):
    command = ["curl", "-fLsS", "--max-time", "30", url]
    result = subprocess.run(command, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    parser = Plain()
    parser.feed(result.stdout.decode())
    plain = " ".join(" ".join(parser.parts).split())
    assert excerpt in plain
    assert all(item in plain for item in required)
    print(json.dumps({"url": url, "inspection_context": plain[max(0, plain.index(excerpt)-120):plain.index(excerpt)+1000]}, ensure_ascii=False), flush=True)
    return {"url": url, "command_argv": command, "exit_code": result.returncode, "accessed_at": datetime.now().astimezone().isoformat(), "response_sha256": sha(result.stdout), "locator": locator, "excerpt": excerpt, "inspection_checks": {item: True for item in required}}


def main():
    import torch

    original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_9-13/technical-9-13-original-01.json"
    original = json.loads(original_path.read_text())
    assert sha(original_path.read_bytes()) == "d723a128fa781029ba3e6faf2f710aabf64ef26abccd6f1a9bebd91ea7145984"
    artifacts = [{"path": item["path"], "expected_sha256": item["sha256"], "current_sha256": sha((ROOT/item["path"]).read_bytes())} for item in original["artifacts"]]
    assert all(item["expected_sha256"] == item["current_sha256"] for item in artifacts)
    code_sources = [{"path": item["path"], "previous_sha256": item["sha256"], "current_sha256": sha((ROOT/item["path"]).read_bytes())} for item in original["sources"] if item["kind"] == "repository_code"]
    assert all(item["current_sha256"] == item["previous_sha256"] for item in code_sources if item["path"] != "course/chapters/09.md")
    current = (ROOT/"course/chapters/09.md").read_bytes()
    before = subprocess.check_output(["git", "show", "5ef56bc756f3:course/chapters/09.md"], cwd=ROOT)
    after = subprocess.check_output(["git", "show", "3af7f13ef959:course/chapters/09.md"], cwd=ROOT)
    assert sha(before) == "9d2039c6aba3bfe4e66c3a5c442180628321780de5cda00a0cc7fd96d85d03ee"
    assert sha(current) == sha(after) == "62274ee1570b419bf5872ebbf4d84b63115442ac4960a826464ea28fececbd8b"
    frozen = (ROOT/"docs/course-review-20261019/freeze-01/sources/9.13.md").read_bytes()
    assert section(current, "9.13") == section(before, "9.13") == frozen
    unchanged_dependencies = {identifier: {"previous_sha256": sha(section(before, identifier)), "current_sha256": sha(section(current, identifier)), "equal": section(before, identifier) == section(current, identifier)} for identifier in ("9.11", "9.14")}
    assert all(item["equal"] for item in unchanged_dependencies.values())
    assert "本章把softmax後最大的那個機率叫**信心（confidence）**" in section(current, "9.9").decode()
    diff_command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/09.md"]
    diff = subprocess.check_output(diff_command, cwd=ROOT, text=True)
    changes = [{"kind": line[:1], "text": line[1:]} for line in diff.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    assert sum(item["kind"] == "+" for item in changes) == 4

    command = [str(ROOT/".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/09.md#9.13", "--execute"]
    run = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    receipt = json.loads(run.stdout)
    assert receipt["source_sha256"] == sha(frozen)
    copies = {}
    for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json"):
        raw = (Path(receipt["artifact_directory"])/name).read_bytes()
        copies[name] = {"sha256": sha(raw), "content": raw.decode()}
    old_execution = json.loads((DEST/"audit-execution.json").read_text())
    assert copies["stdout.txt"]["content"] == old_execution["section_output_copies"]["stdout.txt"]["content"]

    tree = ast.parse(re.findall(r"```python\n(.*?)```", frozen.decode(), re.S)[0])
    target = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "materials" for t in node.targets))
    target.value.elts.append(ast.parse('{"申請編號": "K17"}', mode="eval").body)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(ast.fix_missing_locations(tree), "exercise", "exec"), {})
    assert output.getvalue() == old_execution["exercise_stdout"]

    round_receipt = fetch("https://docs.python.org/3.13/library/functions.html#round", "Built-in Functions：round", "Return number rounded to ndigits precision after the decimal point.", ["toward the even choice", "round(0.5)"])
    confidence_receipt = fetch("https://docs.typesafe.ai/confidence", "How confidence is calculated：Choice", "both have confidence 0.4.", [r"\frac{p_{\max} - \frac{1}{n}}{1 - \frac{1}{n}}", "returns no separate", "A Noul answer is a single probability"])
    scores = torch.tensor([2.0, -1.0, 0.0])
    rounded = [round(value, 3) for value in scores.softmax(-1).tolist()]
    assert rounded == [0.844, 0.042, 0.114]
    rounds = {"ties": [round(2.5), round(3.5)], "three_decimals": rounded}
    assert rounds["ties"] == [2, 4]
    concentration = (0.6 - 1/3)/(1 - 1/3)
    assert abs(concentration - 0.4) < 1e-12
    estimated, actual = 5*(1-0.9), 5*(1-0.5)
    assert abs(estimated - actual) > 1
    data = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-13/recheck-current-02.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "original_report_sha256": sha(original_path.read_bytes()),
        "previous_issues": original["issues"],
        "artifacts_integrity": artifacts,
        "repository_sources": code_sources,
        "before_revision": "5ef56bc756f3", "after_revision": "3af7f13ef959",
        "current_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diff_command_argv": diff_command, "diff_stdout": diff,
        "changed_text": changes,
        "unchanged_dependencies": unchanged_dependencies,
        "source_sha256": sha(frozen), "section_unchanged": True,
        "section_command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/09.md#9.13" --execute',
        "section_exit_code": run.returncode, "section_launcher_stdout": run.stdout, "section_launcher_stderr": run.stderr,
        "section_output_copies": copies, "exercise_stdout": output.getvalue(),
        "round_probes": rounds,
        "source_receipts": [round_receipt, confidence_receipt],
        "concentration_probe": {"probabilities": [0.6, 0.2, 0.2], "concentration": concentration},
        "cost_counterexample": {"reported_success_probability": 0.9, "actual_success_probability": 0.5, "failure_cost": 5, "estimated_cost": estimated, "actual_expected_cost": actual, "derivation": "錯誤成本為5且正確成本為0時，期望成本為5*(1-p_true)。用未經驗證的 p_reported 代入，不保證等於真實期望成本；這是透明反例，不是模型實測。"},
        "probe_adjustment": "初次探針以未拆分空白的程式字串比對 HTML 文字，因網頁 token 間插入空白而失敗。改用網頁原有的數學公式與例子精確比對；未修改教材或舊證據。",
        "result": "全部斷言通過；小節與凍結原文相同，舊證據未改，必要依賴語義成立。",
    }
    destination = DEST/"recheck-evidence-current-01.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"path": str(destination.relative_to(ROOT)), "sha256": sha(destination.read_bytes()), "result": data["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
