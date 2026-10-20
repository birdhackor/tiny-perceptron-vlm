"""核對依據檔的差異、策略定義，以及取位與裁切算例；只建立新證據檔。"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    match = re.search(rb"^## " + re.escape(identifier.encode()) + rb" .*?(?=^## |\Z)", raw, re.M | re.S)
    assert match is not None, "找不到指定教材標題"
    return match.group()


def run_git(arguments):
    command = ["git", *arguments]
    output = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return {"command_argv": command, "exit_code": output.returncode, "stdout": output.stdout.decode(), "stderr": output.stderr.decode()}


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_9-14/technical-9-14-original-01.json"
original = json.loads(original_path.read_bytes())
old_source = next(s for s in original["sources"] if s["id"] == "s_ch13")
old_receipt = run_git(["show", "5ef56bc756f3:course/chapters/13.md"])
new_receipt = run_git(["show", "3af7f13ef959:course/chapters/13.md"])
diff_receipt = run_git(["diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/13.md"])
current = (ROOT / old_source["path"]).read_bytes()
old = old_receipt["stdout"].encode()
assert digest(old) == old_source["sha256"], "指定舊版本與原報告不符"
assert new_receipt["stdout"].encode() == current, "指定新版本與工作樹不同"

old_policy = next(line for line in old.decode().splitlines() if "正在更新、之後實際作答的模型叫" in line)
current_policy = next(line for line in current.decode().splitlines() if "正在更新、之後實際作答的模型叫" in line)
assert old_policy == current_policy, "策略模型的定義有變動"
assert section(old, "13.4") == section(current, "13.4"), "策略模型定義所在小節有變動"
original_verification = json.loads((BASE / "verification.json").read_bytes())
assert current_policy == original_verification["policy_model_definition"], "先前保存的定義與現行內容不同"

artifact_checks = []
for artifact in original["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"], "既有證據的雜湊已改變"
    artifact_checks.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
current_section = section((ROOT / "course/chapters/09.md").read_bytes(), "9.14")
assert current_section == (ROOT / "docs/course-review-20261019/freeze-01/sources/9.14.md").read_bytes(), "小節與凍結原文不同"
assert digest(current_section) == original["source_sha256"], "小節與原報告版本不同"

round_paragraph = next(line for line in current.decode().splitlines() if line.startswith("`[round(value, 4) for value in ...]`"))
clip_paragraph = next(line for line in current.decode().splitlines() if line.startswith("ratio乘上") and "下一節" in line)
round_statement = round_paragraph.split("。", 1)[0] + "。"
clip_statement = clip_paragraph[clip_paragraph.index("下一節"):]
assert "取到小數四位" in round_statement
assert "把 ratio 裁切到一個範圍內再乘上優勢" in clip_statement

logits = torch.tensor([[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
labels = torch.tensor([[-100, 0, 2]])
valid = labels != -100
safe_labels = labels.masked_fill(~valid, 0)
selected = logits.log_softmax(-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
rounded = [round(value, 4) for value in selected[0].tolist()]
assert rounded == [-2.2395, -0.2395, -0.2395], "取位後的候選 log 機率不符"
tie_examples = {"round(0.03125, 4)": round(0.03125, 4), "round(0.09375, 4)": round(0.09375, 4), "round(2.675, 2)": round(2.675, 2)}
assert tie_examples["round(0.03125, 4)"] == 0.0312
assert tie_examples["round(0.09375, 4)"] == 0.0938

clip_section = section(current, "13.13").decode()
code = next(m.group(1) for m in re.finditer(r"```python\n(.*?)```", clip_section, re.S) if "def ppo_clipped_objective" in m.group(1))
namespace = {"torch": torch}
exec(compile(code, "course/chapters/13.md:ppo_clipped_objective", "exec"), namespace)
ratio = torch.tensor([0.7, 1.0, 1.3, 0.7, 1.0, 1.3], dtype=torch.float64, requires_grad=True)
advantage = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
old_log = torch.full((6,), 0.5, dtype=torch.float64).log()
new_log = old_log + ratio.log()
result = namespace["ppo_clipped_objective"](new_log, old_log, advantage, clip_range=0.2)
expected = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
assert torch.allclose(result["surrogate"], expected, rtol=0, atol=1e-12), "先裁切 ratio 的替代目標公式不符"
(-result["surrogate"].sum()).backward()
assert torch.allclose(ratio.grad, torch.tensor([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float64), rtol=0, atol=1e-12), "正負優勢的梯度方向不符"

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform(), "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE", "")},
    "result": "全部斷言通過；未進行模型訓練。",
    "original_report_sha256": digest(original_path.read_bytes()),
    "original_issues": original["issues"],
    "original_artifacts": artifact_checks,
    "dependency": {"path": old_source["path"], "previous_sha256": digest(old), "current_sha256": digest(current), "current_matches_specified_commit": True, "policy_section_unchanged": True, "policy_paragraph_unchanged": True, "policy_paragraph": current_policy},
    "diff_receipt": diff_receipt,
    "old_source_receipt": {"command_argv": old_receipt["command_argv"], "exit_code": old_receipt["exit_code"], "stdout_sha256": digest(old)},
    "new_source_receipt": {"command_argv": new_receipt["command_argv"], "exit_code": new_receipt["exit_code"], "stdout_sha256": digest(current)},
    "page_source_sha256": digest(current_section),
    "changed_sentences": [
        {"statement": round_statement, "status": "verified", "basis": "Python round 官方定義與當次取位實跑；沒有將 round 縮限為傳統四捨五入或保證字串顯示固定四位。"},
        {"statement": clip_statement, "status": "verified", "basis": "PPO 原論文 Clipped Surrogate Objective 的公式，以及現行教材節錄函式的正負優勢實跑；裁切 ratio 再乘優勢，最後逐項取 minimum。"},
    ],
    "round": {"selected_log_probabilities": selected.tolist(), "rounded": rounded, "tie_and_float_examples": tie_examples},
    "clipping": {"code_sha256": digest(code.encode()), "ratio": ratio.detach().tolist(), "advantage": advantage.tolist(), "clip_range": 0.2, "unclipped": result["unclipped"].tolist(), "clipped": result["clipped"].tolist(), "surrogate": result["surrogate"].tolist(), "loss_gradient_wrt_ratio": ratio.grad.tolist(), "tolerance": 1e-12, "scope": "裁切有利方向的目標增益，不保證更新後所有 ratio 都在裁切區間。"},
}
with (BASE / "dependency-check-current-01.json").open("x") as handle:
    handle.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2))
