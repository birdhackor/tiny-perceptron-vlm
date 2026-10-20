"""複核平均 log 特徵與先後答案作為監督目標的意義。"""

import contextlib
import io
import json
import math
import runpy
from pathlib import Path

import torch
from torch.nn import functional as F

base = Path(__file__).resolve().parent
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    previous_program = runpy.run_path(str(base / "verify_audio_order.py"))
previous_result = json.loads(captured.getvalue())
frames = previous_program["frames"]

# 兩筆資料只改變順序，先後標籤也跟著改變。
low = torch.tensor([1.0, 0.0], dtype=torch.float64)
high = torch.tensor([0.0, 1.0], dtype=torch.float64)
inputs = torch.stack((torch.stack((low, high)), torch.stack((high, low))))
targets = torch.tensor([0, 1], dtype=torch.long)
pooled = inputs.mean(dim=1)
shared_logits = torch.zeros((2, 2), dtype=torch.float64)
first_frame_logits = 8.0 * (2.0 * inputs[:, 0] - 1.0)
shared_loss = F.cross_entropy(shared_logits, targets)
ordered_loss = F.cross_entropy(first_frame_logits, targets)

result = {
    "replayed_program": "docs/technical-reviews/artifacts/review-20261019-12-13/verify_audio_order.py",
    "replayed_result": previous_result,
    "mean_log_feature_check": {
        "frames": list(frames.shape),
        "mean_values_count": frames.mean(0).numel(),
        "mean_matches_sum_divided_by_frame_count": bool(
            torch.allclose(frames.mean(0), frames.sum(0) / frames.shape[0])
        ),
        "meaning": "沿時間平均每個已取 log 的 mel 頻帶值；不是沿時間加總原功率。",
    },
    "supervised_order_objective": {
        "inputs": inputs.tolist(),
        "labels": targets.tolist(),
        "label_meaning": {"0": "低頻框先來", "1": "高頻框先來"},
        "same_mean_input": bool(torch.equal(pooled[0], pooled[1])),
        "shared_logits": shared_logits.tolist(),
        "shared_cross_entropy": float(shared_loss),
        "best_shared_distribution_loss": math.log(2),
        "first_frame_logits": first_frame_logits.tolist(),
        "first_frame_predictions": first_frame_logits.argmax(-1).tolist(),
        "first_frame_cross_entropy": float(ordered_loss),
        "lower_loss_with_order_information": bool(ordered_loss < shared_loss),
        "derivation": "兩筆相反標籤若只能共用機率 p、1-p，平均 loss=-0.5*ln(p)-0.5*ln(1-p)，由 p(1-p)<=1/4 得最小值 ln(2)。保留順序則可對兩筆輸出不同預測，降低此監督 loss。",
        "scope": "這是兩筆均衡標籤的確定性 loss 計算，沒有訓練模型。標註讓目標要求區分順序，不保證最佳化成功、資料無捷徑或泛化能力。",
    },
}
print(json.dumps(result, ensure_ascii=False, indent=2))
