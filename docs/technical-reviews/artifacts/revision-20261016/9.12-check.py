"""9.12 獨立技術核對；只跑原節短程式與透明機制算例，不訓練。"""

import contextlib
import hashlib
import io
import itertools
import json
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices


body = lesson_slices((ROOT / "course/chapters/09.md").read_text())["9.12"]
code = re.search(r"```python\n(.*?)```", body, re.S)[1]
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(code, "course/chapters/09.md#9.12", "exec"), {})

scores = {"查卡片進度": 2.0, "更改地址": -1.0, "以上皆不適合": 0.0}
base = torch.tensor(list(scores.values()), dtype=torch.float64).softmax(0)
permutations = []
for order in itertools.permutations(scores):
    p = torch.tensor([scores[name] for name in order], dtype=torch.float64).softmax(0)
    restored = torch.tensor([p[order.index(name)] for name in scores])
    error = float((restored - base).abs().max())
    assert error < 1e-14
    assert order[int(p.argmax())] == "查卡片進度"
    permutations.append({"order": order, "max_abs_error": error})

tie_choices = []
for order in [("id-A", "id-B"), ("id-B", "id-A")]:
    p = torch.tensor([1.0, 1.0]).softmax(0)
    tie_choices.append({
        "order": order,
        "plain_argmax": order[int(p.argmax())],
        "fixed_id_tiebreak": min(order),
    })
assert tie_choices[0]["plain_argmax"] != tie_choices[1]["plain_argmax"]
assert {x["fixed_id_tiebreak"] for x in tie_choices} == {"id-A"}

# 共享評分器完全忽略 question 仍可通過排列檢查：這是反例，不是模型成果。
question_counterexample = []
for question in ["下一步服務？", "已知哪種資訊？"]:
    order = list(scores)
    p = torch.tensor([scores[name] for name in order]).softmax(0)
    question_counterexample.append({"question": question, "choice": order[int(p.argmax())]})
assert question_counterexample[0]["choice"] == question_counterexample[1]["choice"]

# 有限封閉集合仍正規化；沒有 none 的輸出欄位。
closed_order = ["查卡片進度", "更改地址"]
closed_p = torch.tensor([2.0, -1.0], dtype=torch.float64).softmax(0)
assert abs(float(closed_p.sum()) - 1.0) < 1e-14
assert len(closed_p) == len(closed_order)

added = torch.tensor([2.0, -1.0, 0.0, 2.0], dtype=torch.float64).softmax(0)
removed = torch.tensor([2.0], dtype=torch.float64).softmax(0)
assert float(added[0]) < float(base[0]) < float(removed[0])

original_logits = torch.tensor([[2.0, -1.0, 0.0]], dtype=torch.float64)
swapped_logits = torch.tensor([[-1.0, 2.0, 0.0]], dtype=torch.float64)
original_loss = float(torch.nn.functional.cross_entropy(original_logits, torch.tensor([0])))
remapped_loss = float(torch.nn.functional.cross_entropy(swapped_logits, torch.tensor([1])))
stale_loss = float(torch.nn.functional.cross_entropy(swapped_logits, torch.tensor([0])))
assert abs(original_loss - remapped_loss) < 1e-14
assert abs(stale_loss - original_loss - 3.0) < 1e-14

result = {
    "lesson_id": "9.12",
    "reviewer_task": "/root/tech_20261016_9_12",
    "command": "uv run python docs/technical-reviews/artifacts/revision-20261016/9.12-check.py",
    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "platform": platform.platform(),
        "cuda_available": str(torch.cuda.is_available()),
    },
    "original_section_code_stdout": capture.getvalue(),
    "all_six_permutations": permutations,
    "tie_counterexample": tie_choices,
    "question_ignoring_scorer_counterexample": question_counterexample,
    "closed_set": {"order": closed_order, "p": closed_p.tolist(), "sum": float(closed_p.sum())},
    "candidate_set_changes": {
        "base_order": list(scores),
        "base_p": base.tolist(),
        "add_one_equal_score_synonym_p": added.tolist(),
        "keep_only_first_p": removed.tolist(),
        "note": "同義詞分數只是人工相同 logits 假設；原候選個別機率下降，語義合計機率不一定下降。",
    },
    "label_remapping": {
        "original_semantic_id": "查卡片進度",
        "original_target": 0,
        "target_after_swap": 1,
        "loss_original": original_loss,
        "loss_after_correct_remap": remapped_loss,
        "loss_after_stale_index": stale_loss,
    },
    "result": "全部 assertions 通過；原文兩種排序都選查卡片進度。只驗證 CPU 人工算例，未訓練、未量測模型品質。",
}
output = ROOT / "docs/technical-reviews/artifacts/revision-20261016/9.12-execution.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
