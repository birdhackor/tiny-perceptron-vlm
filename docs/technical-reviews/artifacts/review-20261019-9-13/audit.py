"""核對凍結例子、有限世界標籤與 Auto 決策順序，保存當次執行收據。"""

import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def captured(code):
    out = io.StringIO()
    space = {}
    with contextlib.redirect_stdout(out):
        exec(compile(code, "frozen-example", "exec"), space)
    return out.getvalue(), space


def main():
    import torch
    from tiny_perceptron.lesson_products.data import REQUIRED, c_rows
    from tiny_perceptron.lesson_products.inference import auto_solve

    command = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/09.md#9.13", "--execute"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    receipt = json.loads(run.stdout)
    run_directory = Path(receipt["artifact_directory"])
    copies = {}
    for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json"):
        raw = (run_directory / name).read_bytes()
        copies[name] = {"sha256": sha(raw), "content": raw.decode()}
    assert run.returncode == 0, run.stderr

    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/9.13.md"
    raw = frozen.read_bytes()
    assert sha(raw) == "ca34896a8386ae7cc38a398dd159ffce995f3c193324d3b300f4317f131c2b9c"
    assert receipt["source_sha256"] == sha(raw)
    fences = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
    table_stdout, namespace = captured(fences[0])
    table_expected = [
        "{'任務': '查卡片進度', '申請編號': 'K17'} → 資訊充分 yes 下一步 查卡片進度",
        "{'任務': '查卡片進度'} → 資訊充分 no 下一步 詢問申請編號",
        "{'任務': '查地址變更規則'} → 資訊充分 yes 下一步 查地址變更規則",
        "{'任務': '重設密碼'} → 資訊充分 不適用 下一步 轉給真人客服",
    ]
    assert table_stdout.splitlines() == table_expected
    probability_stdout, _ = captured(fences[1])
    assert probability_stdout.strip() == "P(yes) 0.03 P(no) 0.97 信心 0.97"

    exercise = ast.parse(fences[0])
    assignment = next(n for n in exercise.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "materials" for t in n.targets))
    assignment.value.elts.append(ast.parse('{"申請編號": "K17"}', mode="eval").body)
    exercise_source = ast.unparse(ast.fix_missing_locations(exercise))
    exercise_stdout, _ = captured(exercise_source)
    assert exercise_stdout.splitlines()[-1] == "{'申請編號': 'K17'} → 資訊充分 no 下一步 詢問任務"

    loop = next(n for n in ast.parse(fences[0]).body if isinstance(n, ast.For))
    loop_source = ast.unparse(loop)
    probes = [
        {"任務": "查卡片進度", "無關編號": "K17"},
        {"任務": "查卡片進度", "申請編號": None},
        {"任務": "查卡片進度", "申請編號": ""},
        {"任務": "查卡片進度", "申請編號": "別人的編號"},
        {"任務": "查地址變更規則", "無關編號": "K17"},
    ]
    presence_out = io.StringIO()
    with contextlib.redirect_stdout(presence_out):
        exec(loop_source, {"materials": probes, "needs": namespace["needs"]})
    assert "資訊充分 no" in presence_out.getvalue().splitlines()[0]
    assert all("資訊充分 yes" in line for line in presence_out.getvalue().splitlines()[1:])
    key_probe = {"get_missing": {}.get("任務"), "has_task_key": "任務" in {}}
    try:
        {}["任務"]
    except KeyError as error:
        key_probe["index_missing_exception"] = type(error).__name__

    worlds = [
        {"task": "card", "application": "123456", "family_id": "audit-card"},
        {"task": "address", "family_id": "audit-address"},
        {"task": "none", "family_id": "audit-none"},
    ]
    labels = [{"task": w["task"], "state": r["state"], "question": r["question"], "omitted": r["omitted"], "gold": r["gold"]}
              for w in worlds for r in c_rows(w, "test") if not r.get("missing_best")]
    assert any(r["task"] == "card" and r["question"] == "info" and r["omitted"] and r["gold"] == "no" for r in labels)
    assert all(r["question"] == "choice" for r in labels if r["task"] == "none")

    class Solver:
        auto_policy = {"version": 1, "sum_depth": 3, "low_max_events": 1, "medium_max_events": 3}

        def __init__(self):
            self.calls = []

        def synchronize(self):
            pass

        def solve(self, state, loops):
            self.calls.append({"state": state, "loops": loops})
            return {"answer": 5, "loops": loops}

    class Judge:
        def __init__(self, choice, info):
            self.choice, self.info, self.calls = choice, info, []

        def synchronize(self):
            pass

        def judge(self, state, question):
            self.calls.append(question)
            return {"selected": self.choice if question == "choice" else self.info, "seconds": 0.0}

    auto_probes = []
    for choice, info in (("none", "yes"), ("none", "no"), ("card", "yes"), ("card", "no"), ("boxes", "no"), ("boxes", "yes")):
        b, c = Solver(), Judge(choice, info)
        result = auto_solve(b, c, "兩盒追蹤;A=0;B=2;事件=A設5;查詢=A")
        auto_probes.append({"choice": choice, "info": info, "judge_calls": c.calls, "solver_calls": b.calls, "status": result["status"], "b_loops": result["b_loops"]})
        assert c.calls == ["choice", "info"]
        expected = "unsupported" if choice not in ("boxes", "sum") else "clarification_needed" if info != "yes" else "answered"
        assert result["status"] == expected
        assert bool(b.calls) == (expected == "answered")

    excerpts = {}
    requests = {
        "tiny_perceptron/lesson_products/inference.py": ("def auto_solve(", None),
        "tiny_perceptron/lesson_products/data.py": ("REQUIRED =", "EVENTS ="),
        "tiny_perceptron/lesson_products/model.py": ("def validate_candidates(", "def candidate_batch("),
    }
    for path, (start, end) in requests.items():
        data = (ROOT / path).read_bytes()
        text = data.decode()
        first = text.index(start)
        last = text.index(end, first) if end else len(text)
        excerpts[path] = {"file_sha256": sha(data), "excerpt": text[first:last]}

    report = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-13/audit.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "section_command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/09.md#9.13\" --execute",
        "section_exit_code": run.returncode,
        "section_launcher_stdout": run.stdout,
        "section_launcher_stderr": run.stderr,
        "section_output_copies": copies,
        "frozen_source_sha256": sha(raw),
        "table_stdout": table_stdout,
        "table_expected": table_expected,
        "probability_stdout": probability_stdout,
        "exercise_source": exercise_source,
        "exercise_stdout": exercise_stdout,
        "presence_probe_stdout": presence_out.getvalue(),
        "dictionary_probe": key_probe,
        "required_fields": REQUIRED,
        "generated_labels": labels,
        "auto_probes": auto_probes,
        "repository_excerpts": excerpts,
        "scope": "凍結程式與額外確定性探針；Auto 使用替身隔離決策流程，沒有訓練、載入成品權重或測量模型品質。",
        "result": "全部斷言通過；Auto 的資訊答案在不支援分支不影響狀態，但仍會事先執行資訊判斷。",
    }
    destination = DEST / "audit-execution.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"path": str(destination.relative_to(ROOT)), "sha256": sha(destination.read_bytes()), "result": report["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
