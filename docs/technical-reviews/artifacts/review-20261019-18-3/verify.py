"""重算教師回答稽核，檢查文字目標、遮罩與生成結束條件。"""

import copy
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.compression import _example
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, generate, masked_loss
from tiny_perceptron.tokenization import generation_report


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def scores(value):
    return {key: item for key, item in value.items() if key != "seconds"}


torch.set_num_threads(1)
tok = ByteTokenizer()
result_path = ROOT / "docs/course-experiments/results/distillation.json"
raw = result_path.read_bytes()
experiment = json.loads(raw)
results = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "input": {"path": str(result_path.relative_to(ROOT)), "sha256": sha(raw)},
    "measurement_environment": {
        key: experiment[key] for key in ("revision", "seed", "device", "gpu", "python_version", "torch_version")
    },
    "tasks": {},
}
for task_name in ("attributes", "style_transfer"):
    task = experiment["results"]["tasks"][task_name]
    report = task["hard_target_generation"]
    audit = report["audit"]
    expected_count = 45 if task_name == "attributes" else 185
    require(len(audit) == expected_count == report["records"], "生成回答筆數不符")
    exact_count = eos_count = invalid_count = zero_count = 0
    same_targets = 0
    wrong = []
    for row in audit:
        ids = row["teacher_ids"]
        visible = generation_report(tok, ids)
        answer_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        exact = answer_ids == tok.encode(row["gold_answer"])
        require(exact == row["teacher_correct"], "逐筆正確旗標與原始 ID 不符")
        require(visible["answer"] == row["teacher_answer"], "文字解碼與稽核答案不符")
        require(visible["eos"] == row["eos"], "EOS 旗標與原始 ID 不符")
        require(visible["invalid_special_tokens"] == row["invalid_special_tokens"], "特殊 token 稽核不符")
        require(row["valid_target_tokens"] == len(ids), "目標長度不符")
        require(all(0 <= value < 264 for value in ids), "ID 超出字表")
        exact_count += exact
        eos_count += bool(ids and ids[-1] == tok.eos_id)
        invalid_count += bool(visible["invalid_special_tokens"])
        zero_count += not ids
        record = {"question": row["question"], "answer": row["gold_answer"]}
        hard = {**record, "_hard_ids": ids, "_hard_eos": row["eos"]}
        gold_pair, hard_pair = _example(record, 128), _example(hard, 128)
        same_targets += all(torch.equal(a, b) for a, b in zip(gold_pair, hard_pair, strict=True))
        require(hard_pair[1][hard_pair[1] != IGNORE].tolist() == ids, "原始教師 ID 沒有完整作為標籤")
        if not exact:
            wrong.append({key: row[key] for key in ("family", "question", "gold_answer", "teacher_answer")})
    require(exact_count == report["correct"], "彙總正確筆數不符")
    require(invalid_count == report["invalid_control_records"] == 0, "出現不合法控制 token")
    require(zero_count == report["zero_target_records"] == 0, "出現零長度目標")
    require(eos_count == expected_count, "並非全部以實際 EOS 結束")
    item = {"records": len(audit), "correct": exact_count, "wrong": wrong, "actual_terminal_eos": eos_count,
            "invalid_records": invalid_count, "zero_targets": zero_count, "identical_gold_and_hard_examples": same_targets,
            "split_counts": task["data"]["counts"], "student_training_records": task["data"]["student_training_records"]}
    if task_name == "attributes":
        require(exact_count == same_targets == 45, "屬性題並非全數同目標")
        item["matched_students"] = {}
        for width in (16, 32):
            ce, hard = (task["runs"][f"w{width}_{method}"] for method in ("ce", "teacher_hard"))
            fields = ("initialization_sha256", "final_sha256", "batch_plan_sha256", "steps", "batch_size", "effective_supervised_tokens")
            require(all(ce["training"][key] == hard["training"][key] for key in fields), "學生訓練紀錄不相同")
            require(all(scores(ce[split]) == scores(hard[split]) for split in ("validation", "test")), "學生分數或逐筆生成不相同")
            item["matched_students"][str(width)] = {**{key: ce["training"][key] for key in fields},
                "validation_examples": ce["validation"]["examples"], "test_examples": ce["test"]["examples"],
                "validation_exact_match": ce["validation"]["exact_match"], "test_exact_match": ce["test"]["exact_match"],
                "non_timing_evaluation_identical": True}
    else:
        require(exact_count == 180 and len(wrong) == 5, "風格題正誤筆數不符")
        for row in wrong:
            import re
            source_day = re.search(r"2026-10-(\d+)", row["question"])
            answer_day = re.search(r"2026-10-(\d+)", row["teacher_answer"])
            require(source_day is not None and answer_day is not None, "錯題不是日期題")
            require(int(answer_day[1]) - int(source_day[1]) == 10, "日期錯誤不是加十天")
    results["tasks"][task_name] = item

messages = [{"role": "user", "content": "2+2=?"}, {"role": "assistant", "content": "4"}]
x, y = render_chat(messages)
user_ids = tok.encode("2+2=?")
require(x[:len(user_ids) + 3].tolist() == [tok.bos_id, tok.user_id] + user_ids + [tok.eos_id], "題目不是可見前文")
require(y[y != IGNORE].tolist() == tok.encode("4") + [tok.eos_id], "監督位置包含了題目或漏了 EOS")
torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=64))
other = copy.deepcopy(model)
loss = masked_loss(model(x[None])["logits"], y[None])
other_loss = masked_loss(other(x[None])["logits"], y[None])
loss.backward()
other_loss.backward()
require(torch.equal(loss, other_loss), "同目標的 loss 不同")
require(all(torch.equal(a.grad, b.grad) for a, b in zip(model.parameters(), other.parameters(), strict=True)), "同目標的梯度不同")
translation_targets = {text: tok.encode(text) + [tok.eos_id] for text in ("早安", "早上好")}
require(translation_targets["早安"] != translation_targets["早上好"], "不同文字被當成同 token 序列")
results["target_checks"] = {"visible_input_ids": x.tolist(), "shifted_labels": y.tolist(),
    "effective_target_ids": y[y != IGNORE].tolist(), "loss": float(loss.detach()),
    "same_loss_and_all_parameter_gradients": True, "translation_target_ids": translation_targets,
    "explanatory_answer_tokens_with_eos": len(tok.encode("2個加2個共有4個，所以答案是4")) + 1,
    "short_answer_tokens_with_eos": len(tok.encode("4")) + 1}

no_eos = {"question": "2+2=?", "answer": "4", "_hard_ids": tok.encode("4"), "_hard_eos": False}
_, no_eos_labels = _example(no_eos, 64)
require(no_eos_labels[no_eos_labels != IGNORE].tolist() == tok.encode("4"), "補上了不存在的 EOS")
try:
    _example({**no_eos, "_hard_ids": []}, 64)
except ValueError as error:
    zero_error = str(error)
else:
    raise AssertionError("零目標未被拒絕")
with torch.no_grad():
    model.output.weight.zero_()
    model.output.weight[100] = 1
generated = generate(model, x[None], max_new_tokens=0)[0].tolist()
require(generated == x.tolist(), "零生成預算產生了 token")
visible = tok.decode(tok.encode("4") + [tok.eos_id])
require(visible == tok.decode(tok.encode("4")) == "4", "普通解碼不會隱藏 EOS 的假設不符")
results["termination_checks"] = {"no_eos_supervised_ids": no_eos_labels[no_eos_labels != IGNORE].tolist(),
    "empty_target_error": zero_error, "zero_budget_generated_suffix": [], "decode_with_and_without_eos": visible}

results["implementation_binding"] = {}
for name in ("scripts/course_experiments/compression.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/tokenization.py"):
    current = sha((ROOT / name).read_bytes())
    results["implementation_binding"][name] = {"current_sha256": current, "measurement_sha256": experiment["code_sha256"][name],
                                               "same": current == experiment["code_sha256"][name]}
probe = subprocess.run(["git", "show", experiment["revision"] + ":scripts/course_experiments/compression.py"], cwd=ROOT, capture_output=True)
results["historical_implementation_lookup"] = {"command": "git show " + experiment["revision"] + ":scripts/course_experiments/compression.py",
    "exit_code": probe.returncode, "stderr": probe.stderr.decode()}
results["scope"] = "本次在 CPU 重算既有實驗紀錄與逐筆原始 ID，並實跑目標及遮罩；未重新訓練或下載歷史 GPU 權重。權重相等限於原始實驗所記參數雜湊。"
print(json.dumps(results, ensure_ascii=False, indent=2))
