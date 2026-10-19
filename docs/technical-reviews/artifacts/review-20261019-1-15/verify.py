"""核對固定分數下的溫度機率、排序、抽樣與數值邊界。"""

import hashlib
import json
import math
import platform
from pathlib import Path

import torch


def independent_softmax(scores, temperature):
    scaled = [value / temperature for value in scores]
    weights = [math.exp(value - max(scaled)) for value in scaled]
    return [value / sum(weights) for value in weights]


def calculate(scores, expected):
    tensor = torch.tensor(scores, dtype=torch.float32, device="cpu")
    original = tensor.clone()
    results = []
    for temperature, highest in zip([0.5, 1.0, 2.0], expected, strict=True):
        probability = (tensor / temperature).softmax(dim=0)
        reference = independent_softmax(scores, temperature)
        error = max(abs(a - b) for a, b in zip(probability.tolist(), reference, strict=True))
        assert error < 1e-7, "獨立指數計算與 PyTorch 超出容忍差"
        assert abs(probability.sum().item() - 1) < 1e-6, "候選機率總和不等於一"
        assert bool((probability > 0).all()), "有限算例應有正機率"
        assert abs(reference[0] - highest) <= 0.0005, "正文的三位小數超出四捨五入容忍差"
        assert probability.argmax().item() == 0, "最高候選的 ID 不符"
        assert torch.equal(tensor.argsort(), (tensor / temperature).argsort()), "分數排序不符"
        results.append({
            "temperature": temperature,
            "scaled_scores": (tensor / temperature).tolist(),
            "torch_probability": probability.tolist(),
            "independent_probability": reference,
            "highest_rounded": round(reference[0], 3),
            "absolute_error": error,
            "entropy_nats": -sum(p * math.log(p) for p in reference),
        })
    assert torch.equal(tensor, original), "選字計算不應修改原分數"
    assert all(a["entropy_nats"] < b["entropy_nats"] for a, b in zip(results, results[1:])), "算例的熵應隨溫度增加"
    return results


original = calculate([3.0, 1.0, 0.0], [0.980, 0.844, 0.629])
exercise = calculate([3.0, 1.0, 2.0], [0.867, 0.665, 0.506])
for before, after in zip(original, exercise, strict=True):
    assert after["torch_probability"][2] > before["torch_probability"][2], "提高第三項分數後的機率應增加"

scores = torch.tensor([3.0, 1.0, 0.0], dtype=torch.float32)
probability = scores.softmax(dim=0)
sequences = {}
for seed in [42, 7]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sequences[str(seed)] = [torch.multinomial(probability, 1, generator=generator).item() for _ in range(24)]
assert sequences["42"] != sequences["7"], "這組種子應產生不同的抽樣序列"

tie = torch.tensor([3.0, 3.0, 0.0])
assert tie.argmax().item() == 0, "並列時應取第一個最高值"
zero_result = (scores / 0.0).softmax(dim=0)
assert bool(torch.isnan(zero_result).all()), "零溫度不應形成有效機率"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-15/verify.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": "torch.float32；獨立參照使用 Python binary64",
    },
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "original": original,
    "exercise": exercise,
    "seed_sequences": sequences,
    "tie_argmax": tie.argmax().item(),
    "zero_temperature": "softmax 結果全為 NaN，不能作為機率",
    "derivation": {
        "probability": "p_i(tau)=exp(z_i/tau)/sum_j exp(z_j/tau)，tau>0。",
        "ordering": "p_i/p_j=exp((z_i-z_j)/tau)；正溫度保留大小與並列。唯一最高分時 tau 趨零的極限為該候選機率一；並列最大值時極限在最高分之間均分。",
        "flattening": "令 beta=1/tau，H=log(sum exp(beta*z))-beta*E[z]；dH/dtau=Var(z)/tau^3>=0。有限候選且分數有限時，tau 趨無限使分配趨均勻。",
        "score_change": "固定 tau 下，dp_k/dz_k=p_k*(1-p_k)/tau>0（其他候選也有正機率時）。",
        "accuracy_limit": "最高分若對應錯誤答案，tau 趨零仍會把機率集中在錯誤答案；選字規則沒有把語意正確與否加入公式。",
        "controlled_comparison": "同時改模型與選字規則，觀察到的文字差異可能來自任何一者。固定溫度與選字規則是隔離訓練影響的必要控制之一，仍須固定題目、資料與其他條件；隨機抽樣也需控制種子或重複評估。",
    },
    "result": "所有斷言通過；無模型訓練、語言品質或跨裝置效能實驗。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
