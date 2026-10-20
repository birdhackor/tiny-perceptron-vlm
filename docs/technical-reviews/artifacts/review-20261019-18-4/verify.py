"""核對軟目標算例、autograd 行為與教師訊號的保存範圍。"""

import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.compression import _cache_text, _examples
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(q, targets):
    rows = []
    for p in targets:
        analytic_ce = -sum(a * math.log(b) for a, b in zip(p, q, strict=True))
        analytic_gradient = [b - a for a, b in zip(p, q, strict=True)]
        logits = torch.tensor(q).log().detach().clone().requires_grad_()
        target = torch.tensor(p)
        loss = -(target * logits.log_softmax(0)).sum()
        loss.backward()
        assert abs(loss.item() - analytic_ce) < 2e-7, "交叉熵與獨立計算不符"
        assert torch.allclose(logits.grad, torch.tensor(analytic_gradient), atol=1e-7), "梯度與解析式不符"
        rows.append({"target": p, "math_ce": analytic_ce, "torch_ce": loss.item(),
                     "rounded_ce": round(loss.item(), 4), "math_gradient": analytic_gradient,
                     "torch_gradient": logits.grad.tolist()})
    return rows


cases = {
    "arithmetic": calculate([0.5, 0.4, 0.1], [[1, 0, 0], [1, 0, 0], [0.7, 0.2, 0.1]]),
    "classification": calculate([0.6, 0.2, 0.2], [[1, 0, 0], [1, 0, 0], [0.8, 0.15, 0.05]]),
    "wrong_teacher": calculate([0.6, 0.2, 0.2], [[1, 0, 0], [0, 1, 0], [0.15, 0.8, 0.05]]),
}
assert [r["rounded_ce"] for r in cases["arithmetic"]] == [0.6931, 0.6931, 0.8987], "主例代價不符"
assert [r["rounded_ce"] for r in cases["classification"]] == [0.5108, 0.5108, 0.7305], "分類代價不符"
assert [r["rounded_ce"] for r in cases["wrong_teacher"]] == [0.5108, 1.6094, 1.4446], "偏錯教師代價不符"
q = torch.tensor([0.5, 0.4, 0.1])
intermediate = q.log()
detached = intermediate.detach()
leaf = detached.clone().requires_grad_()
assert not q.requires_grad and detached.grad_fn is None, "detach 狀態不符"
assert detached.data_ptr() == intermediate.data_ptr(), "detach 的儲存關係不符"
assert leaf.data_ptr() != detached.data_ptr() and leaf.is_leaf, "clone 必須獨立儲存並形成 leaf"
assert torch.allclose(leaf.softmax(0), q), "log 後 softmax 未還原比例"
(-(torch.tensor([0.7, 0.2, 0.1]) * leaf.log_softmax(0)).sum()).backward()
first_gradient = leaf.grad.clone()
(-(torch.tensor([0.7, 0.2, 0.1]) * leaf.log_softmax(0)).sum()).backward()
assert torch.allclose(leaf.grad, first_gradient * 2), "同一 leaf 的梯度應累加"
fresh_leaf = detached.clone().requires_grad_()
assert fresh_leaf.grad is None, "新 leaf 不應帶著先前梯度"
z = torch.tensor([0.5, 0.4, 0.1], dtype=torch.float64).log()
gradient = torch.tensor([-0.2, 0.2, 0.0], dtype=torch.float64)
updated = (z - 0.1 * gradient).softmax(0)
assert updated[2] != 0.1 and abs(updated.sum().item() - 1) < 1e-14, "分母連動的預期不符"


class RecordingTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_length=128)
        self.calls = []

    def forward(self, x, valid=None):
        self.calls.append({"input_ids": x.tolist(), "valid": valid.tolist()})
        logits = torch.arange(ByteTokenizer.vocab_size, dtype=torch.float32).expand(*x.shape, -1)
        return {"logits": logits}


records = [{"question": "color?", "answer": "green"}, {"question": "shape?", "answer": "square"}]
teacher = RecordingTeacher()
cache, _ = _cache_text(teacher, records, "cpu")
examples = _examples(records, 128)
x, labels, valid = pad_batch(examples)
assert teacher.calls[0]["input_ids"] == x.tolist(), "教師輸入不是標準序列"
assert [list(t.shape) for t in cache] == [[int((y != IGNORE).sum()), 264] for _, y in examples], "快取維度或遮罩不符"
assert all(torch.equal(t, torch.arange(264).expand_as(t)) for t in cache), "快取未保留全部原始 logits"
assert all(not t.requires_grad and t.device.type == "cpu" for t in cache), "快取的梯度或裝置不符"
assert all((t.argmax(-1) != y[y != IGNORE]).all() for t, (_, y) in zip(cache, examples, strict=True)), "反例應讓最高分候選全部錯誤"

record_path = ROOT / "docs/course-experiments/results/distillation.json"
experiment = json.loads(record_path.read_text())
attributes = experiment["results"]["tasks"]["attributes"]
manifest = next(a for a in experiment["artifacts"] if a["path"] == "sft-teacher-logits.pt")
assert attributes["teacher_cache"]["file_bytes"] == manifest["bytes"] == 345101, "既有快取大小紀錄互不相符"
audit = attributes["hard_target_generation"]["audit"]
assert len(audit) == 45 and all(isinstance(row["teacher_ids"], list) and row["teacher_ids"] for row in audit), "逐筆 ID 記錄不完整"
assert sum(len(row["teacher_ids"]) for row in audit) == 315, "有效目標總數不符"

compression_path = ROOT / "scripts/course_experiments/compression.py"
tree = ast.parse(compression_path.read_text())
code = compression_path.read_text().splitlines()
excerpts = {n.name: "\n".join(code[n.lineno - 1:n.end_lineno]) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in {"_example", "_examples", "_cache_text", "_hard_targets"}}
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-4/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git": torch.version.git_version,
                    "device": "cpu", "platform": platform.platform()},
    "result": "所有算例、autograd、快取遮罩與既有紀錄核對均通過；未訓練模型。",
    "cases": cases,
    "autograd": {"q_requires_grad": q.requires_grad, "detached_requires_grad": detached.requires_grad,
                 "clone_independent_storage": True, "clone_is_leaf": leaf.is_leaf,
                 "repeated_backward_accumulates": True, "fresh_leaf_grad_is_none": True},
    "softmax_update": {"before": q.tolist(), "step": 0.1, "direct_score_gradient": gradient.tolist(),
                       "after": updated.tolist(), "sum": updated.sum().item()},
    "cache_probe": {"records": records, "teacher_calls": teacher.calls, "labels": labels.tolist(),
                    "cache_shapes": [list(t.shape) for t in cache], "raw_logits_preserved": True,
                    "all_argmax_wrong_but_full_rows_retained": True},
    "experiment_receipt": {"path": str(record_path.relative_to(ROOT)), "sha256": sha(record_path),
        "revision": experiment["revision"], "torch_version": experiment["torch_version"], "device": experiment["device"],
        "teacher_config": attributes["teacher_provenance"]["config"], "cache": attributes["teacher_cache"],
        "manifest": manifest, "hard_target_file": {k: v for k,v in attributes["hard_target_generation"].items() if k != "audit"},
        "audit_records": len(audit), "effective_targets": sum(len(row["teacher_ids"]) for row in audit),
        "first_audit_row": audit[0], "scope": "原始實驗 JSON 的保存紀錄；本回合沒有讀取快取二進位檔或重新量測其大小。"},
    "repository_code": {"path": str(compression_path.relative_to(ROOT)), "sha256": sha(compression_path),
        "recorded_run_sha256": experiment["code_sha256"]["scripts/course_experiments/compression.py"],
        "scope": "現行程式全檔 hash 與既有實驗所記版本不同；本回合探針只支持所引用函式的現行行為。",
        "excerpts": excerpts},
    "derivation": "q_i=exp(z_i)/S，sum(p)=1；L=-sum_i p_i log(q_i)=-sum_i p_i z_i+log(S)，故 dL/dz_k=q_k-p_k。當 p 是 one-hot，L=-log(q_correct)。固定 z_k 不代表 q_k 固定，因為 S 含所有候選。L(p,q)=H(p)+KL(p||q)，因此不同 p 的 CE 底值與偏好不同，不能用本頁兩個 CE 替方法排名。"
}
print(json.dumps(output, ensure_ascii=False, indent=2))
