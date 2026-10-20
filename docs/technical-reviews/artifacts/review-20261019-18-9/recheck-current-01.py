"""核對混合代價的單項方向、梯度相加及凍結教材的證據連續性。"""

import ast
import difflib
import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.alignment import distillation_kl, distillation_loss
from tiny_perceptron.model import masked_loss


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_bytes())


def command_output(arguments):
    completed = subprocess.run(arguments, cwd=ROOT, capture_output=True, check=True)
    return completed.stdout


def section(raw, lesson):
    matches = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, value in enumerate(matches) if value[0].startswith(b"## " + lesson.encode() + b" "))
    end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
    return raw[matches[index].start():end]


def fences(raw):
    return re.findall(rb"(?ms)^```python\n(.*?)^```", raw)


original_path = "docs/course-review-20261019/reviews/rv1019_technical_18-9/technical-18-9-original-01.json"
original_raw = (ROOT / original_path).read_bytes()
assert digest(original_raw) == "6df4bce6f3f232b042771fd56b0a1fee235f0e220673bb36b2eb25d91402d0e7", "原報告指紋不符"
original = json.loads(original_raw)
old_path = "docs/course-review-20261019/freeze-01/sources/18.9.md"
new_path = "docs/course-review-20261019/freeze-05/sources/18.9.md"
old = (ROOT / old_path).read_bytes()
new = (ROOT / new_path).read_bytes()
assert digest(old) == original["source_sha256"], "舊凍結原文指紋不符"
assert digest(new) == "651423a9e82c981aac5ee555b18244060960d74f1a458df89527d3b041bbebe2", "新凍結原文指紋不符"
freeze_path = "docs/course-review-20261019/freeze-05.json"
assert digest((ROOT / freeze_path).read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba", "凍結清單指紋不符"
chapter_path = "course/chapters/18.md"
chapter = (ROOT / chapter_path).read_bytes()
assert digest(chapter) == "4eaf60c696a65b9bf6ea05bf35367163a6ab23bb646aa97a97702924071e05be", "現行章檔指紋不符"
assert section(chapter, "18.9") == new, "現行小節與新凍結原文不同"
old_lines, new_lines = old.splitlines(keepends=True), new.splitlines(keepends=True)
changes = []
for tag, a, b, c, d in difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes():
    if tag != "equal":
        changes.append({"operation": tag, "old": b"".join(old_lines[a:b]).decode(), "new": b"".join(new_lines[c:d]).decode()})
assert len(changes) == 1 and changes[0]["operation"] == "replace", "差異範圍與已讀原文不同"
assert len(changes[0]["old"].splitlines()) == len(changes[0]["new"].splitlines()) == 1
assert fences(old) == fences(new), "原始程式格已有變動"
assert ast.dump(ast.parse(fences(new)[1])) == ast.dump(ast.parse(inspect.getsource(distillation_loss)))

artifact_integrity = []
for artifact in original["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"], "沿用 artifact 指紋不符"
    artifact_integrity.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
source_integrity = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = digest((ROOT / source["path"]).read_bytes())
    unchanged = actual == source["sha256"]
    assert unchanged or source["path"] == chapter_path, "未列明的 repo 證據變動"
    source_integrity.append({"path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": unchanged})

old_chapter = command_output(["git", "show", "5ef56bc756f3:course/chapters/18.md"])
assert digest(old_chapter) == next(s["sha256"] for s in original["sources"] if s.get("path") == chapter_path)
chapter_diff_command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", chapter_path]
chapter_diff = command_output(chapter_diff_command).decode()
old_evaluation = section(old_chapter, "18.10")
new_evaluation = section(chapter, "18.10")
old_definition = old_evaluation.split(b"\n\n")[:3]
new_definition = new_evaluation.split(b"\n\n")[:3]
assert old_definition == new_definition, "共同答案評估的定義已有變動"
assert "兩個指標都只看正確答案、不含教師KL" in new_evaluation.decode()

run_path = Path(sys.argv[1]).resolve()
assert run_path.parent == ROOT / "outputs/reviewer-tools/runs", "只讀本次指定 helper 紀錄"
files = {}
for name in ("execution.json", "environment.json", "extraction.json", "stdout.txt", "stderr.txt"):
    raw = (run_path / name).read_bytes()
    files[name] = {"sha256": digest(raw), "content": raw.decode()}
execution = json.loads(files["execution.json"]["content"])
extraction = json.loads(files["extraction.json"]["content"])
environment = json.loads(files["environment.json"]["content"])
assert execution["exit_code"] == 0 and execution["source_sha256"] == digest(new)
assert extraction["source_sha256"] == digest(new) and extraction["figure_sha256"] == {}
previous_stdout = json.loads(load(original["artifacts"][0]["path"])["files"]["execution.json"]["content"])
old_stdout = load(original["artifacts"][0]["path"])["files"]["stdout.txt"]["content"]
assert files["stdout.txt"]["content"] == old_stdout, "目前實跑輸出與原始算例不同"
assert files["stderr.txt"]["content"] == "", "小節實跑有 stderr"

# 檢查的是各代價項對局部下降方向的作用，不假定任意有限步長或 optimizer。
torch.set_num_threads(1)
labels = torch.tensor([[0, 1]])
teacher = torch.tensor([[[0.8, 0.2], [0.8, 0.2]]]).log()
probes = []
for probabilities in ([0.5, 0.5], [0.01, 0.99]):
    for temperature in (1.0, 2.0, 4.0):
        for alpha in (-0.25, 1.25):
            z = torch.tensor([[probabilities, probabilities]]).log().requires_grad_()
            ce = masked_loss(z, labels)
            kl = distillation_kl(z, teacher, labels, temperature=temperature)
            ce_gradient = torch.autograd.grad(ce, z, retain_graph=True)[0]
            kl_gradient = torch.autograd.grad(kl, z, retain_graph=True)[0]
            weighted_ce_gradient = torch.autograd.grad((1 - alpha) * ce, z, retain_graph=True)[0]
            weighted_kl_gradient = torch.autograd.grad(alpha * kl, z, retain_graph=True)[0]
            total_gradient = torch.autograd.grad((1 - alpha) * ce + alpha * kl, z)[0]
            assert torch.allclose(weighted_ce_gradient, (1 - alpha) * ce_gradient, atol=1e-6)
            assert torch.allclose(weighted_kl_gradient, alpha * kl_gradient, atol=1e-6)
            max_error = float((total_gradient - weighted_ce_gradient - weighted_kl_gradient).abs().max())
            assert max_error < 1e-6, "總梯度不等於各項之和"
            own_gradient = kl_gradient if alpha < 0 else ce_gradient
            component_gradient = weighted_kl_gradient if alpha < 0 else weighted_ce_gradient
            directional_derivative = float(-(own_gradient * component_gradient).sum())
            coefficient = alpha if alpha < 0 else 1 - alpha
            expected_derivative = float(-coefficient * own_gradient.square().sum())
            assert abs(directional_derivative - expected_derivative) < 1e-6
            assert directional_derivative > 0, "負係數項沒有反轉對應代價的局部下降方向"
            total_direction_derivative = float(-(own_gradient * total_gradient).sum())
            probes.append({
                "student_probabilities": probabilities, "labels": labels.tolist(),
                "temperature": temperature, "alpha": alpha,
                "negative_coefficient": coefficient,
                "monitored_objective": "教師 KL" if alpha < 0 else "正確答案 CE",
                "own_gradient": own_gradient.tolist(), "weighted_component_gradient": component_gradient.tolist(),
                "own_objective_derivative_along_component_descent": directional_derivative,
                "expected_minus_coefficient_times_squared_norm": expected_derivative,
                "own_objective_derivative_along_total_descent": total_direction_derivative,
                "gradient_sum_max_absolute_error": max_error,
            })
assert len(probes) == 12
counterexamples = [p for p in probes if p["student_probabilities"] == [0.01, 0.99] and p["temperature"] == 1]
assert all(p["own_objective_derivative_along_total_descent"] < 0 for p in counterexamples)

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-9/recheck-current-01.py " + str(run_path.relative_to(ROOT)),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "torch_git_version": str(torch.version.git_version)},
    "original_report": {"path": original_path, "sha256": digest(original_raw)},
    "old_source": {"path": old_path, "sha256": digest(old)},
    "new_source": {"path": new_path, "sha256": digest(new), "matches_current_chapter": True},
    "freeze": {"path": freeze_path, "sha256": digest((ROOT / freeze_path).read_bytes())},
    "section_changes": changes,
    "all_other_section_bytes_and_python_fences_unchanged": True,
    "displayed_implementation_ast_matches": True,
    "prior_artifact_integrity": artifact_integrity,
    "repository_source_integrity": source_integrity,
    "chapter_dependency": {"path": chapter_path, "sha256": digest(chapter), "diff_command_argv": chapter_diff_command, "diff_stdout": chapter_diff, "evaluation_definition_unchanged": True, "current_evaluation_definition": b"\n\n".join(new_definition).decode(), "scope": "只重新核對本頁原先引用的共同標準答案評估定義；其他小節修改不當作本頁的新主張"},
    "section_execution": {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/18.md#18.9\" --execute", "files": files, "stdout_matches_original": True, "exit_code": execution["exit_code"]},
    "gradient_component_probes": probes,
    "derivation": "令 C=CE、K=tau² KL，固定教師與 tau，alpha 是常數。對任何可微學生座標 theta，gC=grad C，gK=grad K，grad L=(1-alpha)gC+alpha gK。若只取 KL 項的下降分量 dK=-alpha*gK，K 的方向導數為 gK·dK=-alpha*||gK||²，alpha<0 時非負；非零梯度時為正。若只取 CE 項 dC=-(1-alpha)*gC，C 的方向導數為 -(1-alpha)*||gC||²，alpha>1 時非負；非零梯度時為正。因此各負係數項各自有反向作用。總下降方向仍為 dC+dK；交叉項可能改變總效果，不能由某一係數的符號推出總更新的結果。這是局部一階的梯度敘述；不保證任意有限步長或其他 optimizer 的離散動態。",
    "result": "新文字的兩個單項方向與總梯度相加皆成立；未改主張的證據與算例仍有效",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
