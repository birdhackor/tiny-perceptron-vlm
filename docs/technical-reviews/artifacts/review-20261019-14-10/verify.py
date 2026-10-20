"""獨立重算轉角與 softmax，核對本次圖文版本。"""

import hashlib
import json
import math
import platform
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[4]
length, extended = 8, 16
scale = extended / length
groups = []
for beta in (2.0, 32.0):
    for theta in (90.0, 60.0, 15.0):
        turns = length * theta / 360
        gamma = min(max((turns - 1) / (beta - 1), 0), 1)
        new_theta = (1 - gamma) * theta / scale + gamma * theta
        groups.append({"beta": beta, "theta": theta, "turns": turns, "gamma": gamma, "new_theta": new_theta})
expected_angles = (90.0, 40.0, 7.5, 46.45161290322581, 30.32258064516129, 7.5)
for row, expected in zip(groups, expected_angles, strict=True):
    if not math.isclose(row["new_theta"], expected, abs_tol=1e-12):
        raise AssertionError("分組轉角與獨立預期不符")

weights = []
for coefficient in (1.0, 2.0, (1 + 0.1 * math.log(scale)) ** 2):
    for scores in ((1.0, 0.0), (0.0, 0.0)):
        tensor = torch.tensor(scores, dtype=torch.float64)
        actual = (tensor * coefficient).softmax(-1)
        denominator = sum(math.exp(coefficient * x) for x in scores)
        expected = [math.exp(coefficient * x) / denominator for x in scores]
        if not torch.allclose(actual, torch.tensor(expected, dtype=torch.float64), atol=1e-15, rtol=0):
            raise AssertionError("softmax 與指數除總和的計算不符")
        weights.append({"coefficient": coefficient, "scores": scores, "expected_from_exp": expected, "observed": actual.tolist()})

slow = {"original_half_open_angle_range": [0, length * 15], "last_angle_full_interpolation": 15 * 7.5, "mixed_gamma": 0.5, "mixed_speed": 0.5 * 7.5 + 0.5 * 15, "mixed_last_angle": 15 * (0.5 * 7.5 + 0.5 * 15)}
if not (slow["last_angle_full_interpolation"] < 120 < slow["mixed_last_angle"]):
    raise AssertionError("完整插值與部分混合的範圍比較不符")

chapter = (root / "course/chapters/14.md").read_bytes()
current = chapter[chapter.index("## 14.10 ".encode()):chapter.index("## 14.11 ".encode())]
snapshot = root / "docs/course-review-20261019/freeze-01/sources/14.10.md"
if current != snapshot.read_bytes():
    raise AssertionError("現行小節與凍結原文不同")
figure_path = "course/figures/rewrite-14-10-yarn-components.svg"
png_path = "outputs/course-review-20261019/figure-renders-04/77bcf6a5e4b261b0f51c58843c12a1766cadf4777c731c522416ef49f37acad9.png"
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-10/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "CPU", "dtype": "torch.float64"},
    "source_sha256": sha(snapshot),
    "freeze_sha256": sha(root / "docs/course-review-20261019/freeze-04.json"),
    "current_equals_freeze": True,
    "figure": {"path": figure_path, "sha256": sha(root / figure_path), "inspected_png": png_path, "png_sha256": sha(root / png_path)},
    "groups": groups,
    "softmax": weights,
    "slow_group_range": slow,
    "result": "轉角、範圍、兩種分數的 softmax 與正文及練習一致；不含模型能力實驗。",
}, ensure_ascii=False, indent=2))
