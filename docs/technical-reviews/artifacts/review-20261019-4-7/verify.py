"""獨立核對配對、代價分母、梯度與固定實驗紀錄；不執行訓練。"""

import hashlib
import ast
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import new_lm, split_records
from scripts.prepare_data import generate_records
from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import shifted
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(42)
x, y = shifted([1, 2, 3, 4])
model = TinyLM(ModelConfig(vocab_size=5, width=8))
logits = model(x[None])["logits"]
loss = masked_loss(logits, y[None])
before = {name: p.detach().clone() for name, p in model.named_parameters()}
logit_rows = logits.detach().double().reshape(-1, 5)
losses = torch.logsumexp(logit_rows, dim=-1) - logit_rows[torch.arange(3), y]
assert abs(float(loss.detach()) - float(losses.mean())) < 1e-6
loss.backward()
parameter_delta = max(float((p.detach() - before[name]).abs().max()) for name, p in model.named_parameters())
assert parameter_delta == 0
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

ignored = y[None].clone()
ignored[0, 1] = -100
ignore_loss = float(masked_loss(logits.detach(), ignored))
assert abs(ignore_loss - float(losses[[0, 2]].mean())) < 1e-6
copy_loss = float(masked_loss(logits.detach(), x[None].clone()))
short_inputs = []
for ids in ([], [1]):
    try:
        shifted(ids)
    except ValueError as error:
        short_inputs.append({"input": ids, "type": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("短序列必須拒絕")

extended_x, extended_y = shifted([1, 2, 3, 4, 0])
extended_shape = list(model(extended_x[None])["logits"].shape)
assert extended_shape == [1, 4, 5]
with torch.no_grad():
    for invalid in (-1, 5):
        try:
            model(torch.tensor([[invalid]]))
        except (IndexError, RuntimeError):
            pass
        else:
            raise AssertionError("超出字表的 ID 必須拒絕")
    try:
        model(torch.tensor([[1.0]]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("浮點 ID 必須拒絕")

probe_model = TinyLM(ModelConfig(width=64, layers=2))
probe = torch.tensor([[1, 10, 11, 12, 13]])
changed = probe.clone()
changed[0, -1] = 14
with torch.no_grad():
    original_scores = probe_model(probe)["logits"]
    changed_scores = probe_model(changed)["logits"]
    differences = (original_scores - changed_scores).abs()
    prefix_difference = float(differences[:, :-1].max())
    final_difference = float(differences[:, -1].max())
    prefix_scores = [probe_model(probe[:, :n])["logits"][:, -1] for n in range(1, 6)]
    prefix_equivalence = [float((prefix_scores[n] - original_scores[:, n]).abs().max()) for n in range(5)]
assert prefix_difference == 0.0
assert max(prefix_equivalence) < 1e-6
assert final_difference > 0

result_path = ROOT / "docs/course-experiments/results/text_foundation.json"
result = json.loads(result_path.read_bytes())
parts = split_records(generate_records("toy-text"), seed=result["seed"])
splits = {}
for name, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    splits[name] = {"records": len(rows), "sha256": hashlib.sha256(raw).hexdigest()}
    assert splits[name]["records"] == result["results"]["data"][name]["records"]
    assert splits[name]["sha256"] == result["results"]["data"][name]["sha256"]
relevant_paths = [
    "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py",
    "tiny_perceptron/modern.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
]
code_hashes = {path: digest(ROOT / path) for path in relevant_paths}
historical_comparisons = []
for path, current_hash in code_hashes.items():
    recorded_hash = result["code_sha256"][path]
    if current_hash == recorded_hash:
        continue
    search_command = ["git", "log", "--all", "--format=%H", "--", path]
    commits = subprocess.check_output(search_command, cwd=ROOT, text=True).splitlines()
    for commit in commits:
        show_command = ["git", "show", f"{commit}:{path}"]
        shown = subprocess.run(show_command, cwd=ROOT, capture_output=True)
        if shown.returncode == 0 and hashlib.sha256(shown.stdout).hexdigest() == recorded_hash:
            old = shown.stdout.decode()
            break
    else:
        raise AssertionError("找不到實驗所記的程式版本")
    targets = ["run_text_foundation", "_evaluations", "_steps"] if path.endswith("text.py") else ["new_lm", "fit_lm", "_nll", "split_records", "text_examples"]
    def functions(source):
        return {node.name: node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    current_functions, old_functions = functions((ROOT / path).read_text()), functions(old)
    assert all(ast.dump(current_functions[name]) == ast.dump(old_functions[name]) for name in targets)
    historical_comparisons.append({
        "path": path, "recorded_sha256": recorded_hash, "current_sha256": current_hash,
        "matching_commit": commit, "search_command_argv": search_command, "show_command_argv": show_command,
        "unchanged_functions_ast": targets,
        "inspected_recorded_function": ast.get_source_segment(old, old_functions[targets[0]]),
        "scope": "全檔 hash 有差異，與本節因果檢查及訓練相關的函式 AST 相同；差異位於其他切詞或取提示文字的路徑。",
    })
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
figure_receipt = next(record for record in receipt["records"] if record["page_id"] == "4.7")
assert digest(ROOT / figure_receipt["png"]["path"]) == figure_receipt["png"]["sha256"]
extraction = json.loads((Path(__file__).parent / "section-execution.json").read_bytes())["extraction"]
original = (ROOT / "docs/course-review-20261019/freeze-01/sources/4.7.md").read_bytes()
assert extraction["source_sha256"] == hashlib.sha256(original).hexdigest()
function_text = inspect.getsource(shifted)
fence = original.decode().split("```python\n")[1].split("```")[0]
assert function_text.strip() == fence.split("\n", 1)[1].strip()

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-7/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu"},
    "result": "所有斷言通過；未訓練模型、未更新參數。",
    "main": {
        "pairs": list(zip(x.tolist(), y.tolist())), "x_shape": list(x[None].shape), "dtype": str(x.dtype),
        "torch_long_is_int64": torch.long == torch.int64, "logits_shape": list(logits.shape),
        "logits": logits.detach().tolist(), "per_position_cross_entropy_float64": losses.tolist(),
        "manual_mean_float64": float(losses.mean()), "masked_loss": float(loss.detach()),
        "effective_targets": int(y.numel()), "ignored_middle_loss": ignore_loss,
        "ignored_effective_targets": 2, "same_position_copy_loss": copy_loss,
        "parameter_max_difference_after_backward": parameter_delta,
        "all_parameter_gradients_finite": True,
        "greedy_ids_before_update": logits.detach().argmax(-1).tolist(),
        "shifted_docstring": shifted.__doc__, "short_input_errors": short_inputs,
        "double_shift_pairs": list(zip(x[:-1].tolist(), y[1:].tolist())),
    },
    "exercise": {"pairs": list(zip(extended_x.tolist(), extended_y.tolist())), "logits_shape": extended_shape},
    "causal_probe_untrained_cpu": {
        "probe": probe.tolist(), "changed": changed.tolist(), "compared_prefix_shape": list(differences[:, :-1].shape),
        "prefix_max_difference": prefix_difference, "final_position_max_difference": final_difference,
        "full_sequence_vs_separate_prefix_max_differences": prefix_equivalence,
        "allowed_three_positions": attention_mask(torch.arange(3), torch.arange(3)).tolist(),
        "scope": "本回合未訓練 CPU 模型的機制檢查；不是重現既有 L4 訓練。",
    },
    "historical_experiment_inspection": {
        "path": result_path.relative_to(ROOT).as_posix(), "sha256": digest(result_path),
        "gpu": result["gpu"], "python_version": result["python_version"], "torch_version": result["torch_version"],
        "seed": result["seed"], "step_scale": result["step_scale"], "evidence_status": result["evidence_status"],
        "steps": result["results"]["training"]["steps"], "effective_training_tokens": result["results"]["training"]["effective_tokens"],
        "causal_max_difference": result["results"]["causal_max_difference"],
        "regenerated_data_splits": splits, "relevant_current_code_sha256": code_hashes,
        "historical_code_comparisons": historical_comparisons,
        "new_lm_signature": str(inspect.signature(new_lm)),
        "scope": "讀取已保存實驗結果並核對配置與產生數值的原始碼；未重新取得權重或重跑 GPU 訓練。",
    },
    "figure_receipt": figure_receipt,
    "preliminary_attempts": [
        {"command": "執行同一支 verify.py 的初稿", "result": "在未區分全檔與相關函式版本前，全檔 hash 一致斷言失敗；尚未寫出 verification.json。之後找回 hash 相符的歷史程式，逐函式核對。"},
        {"command": "git show b52935d99f58b694cd932ac158c10c4d8d92d4c2:scripts/course_experiments/text.py", "result": "該 revision 的此路徑不存在；改以實驗記錄的全檔 SHA-256 尋找歷史內容，不憑 revision 欄猜測程式版本。"},
    ],
}
with (Path(__file__).parent / "verification.json").open("x") as stream:
    stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2))
