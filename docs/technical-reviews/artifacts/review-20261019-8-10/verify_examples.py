"""核對給定標籤的 tensor 計算、修改練習與連結的算術驗證器。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[4]
source = root / "docs/course-review-20261019/freeze-01/sources/8.10.md"
chapter = root / "course/chapters/0C.md"
human = torch.tensor([1, 0, 1, 0])
recorded = torch.tensor([1, 1, 1, 0])
agree = human == recorded
modified = recorded.clone()
modified[1] = 0
modified_agree = human == modified
alternative = torch.tensor([0, 0, 1, 0])

section = chapter.read_text().split("## C.2 ", 1)[1].split("\n## ", 1)[0]
verifier_code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
verifier_stdout = io.StringIO()
verifier_namespace = {}
with contextlib.redirect_stdout(verifier_stdout):
    exec(compile(verifier_code, str(chapter), "exec"), verifier_namespace)

assert agree.dtype == torch.bool
assert agree.tolist() == [True, False, True, True]
assert agree.float().tolist() == [1.0, 0.0, 1.0, 1.0]
assert agree.float().mean().item() == 3 / 4
assert (~agree).tolist() == [False, True, False, False]
assert (~agree).nonzero().tolist() == [[1]]
assert (~agree).nonzero().flatten().tolist() == [1]
assert modified_agree.float().mean().item() == 1.0
assert (~modified_agree).nonzero().flatten().tolist() == []
assert (human == alternative).float().mean().item() == 0.75
assert verifier_namespace["valid"] == [False, True]
assert verifier_namespace["linked"] == [True]
assert verifier_namespace["used"] == [2, 2, -1]

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-10/verify_examples.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": str(human.device),
        "platform": platform.platform(),
    },
    "inputs": {
        "human": human.tolist(),
        "recorded_judge": recorded.tolist(),
        "shape": list(human.shape),
        "dtype": str(human.dtype),
        "source_sha256": digest(source),
    },
    "original": {
        "agree": agree.tolist(),
        "agree_dtype": str(agree.dtype),
        "float_values": agree.float().tolist(),
        "float_dtype": str(agree.float().dtype),
        "agreement_numerator": int(agree.sum().item()),
        "agreement_denominator": agree.numel(),
        "agreement": agree.float().mean().item(),
        "inverted": (~agree).tolist(),
        "nonzero_shape": list((~agree).nonzero().shape),
        "nonzero": (~agree).nonzero().tolist(),
        "disagreement_indices": (~agree).nonzero().flatten().tolist(),
        "human_positive_count": int(human.sum().item()),
        "recorded_positive_count": int(recorded.sum().item()),
    },
    "modified": {
        "recorded_judge": modified.tolist(),
        "agreement": modified_agree.float().mean().item(),
        "nonzero_shape": list((~modified_agree).nonzero().shape),
        "disagreement_indices": (~modified_agree).nonzero().flatten().tolist(),
    },
    "same_agreement_different_error": {
        "recorded_judge": alternative.tolist(),
        "agreement": (human == alternative).float().mean().item(),
        "disagreement_indices": (human != alternative).nonzero().flatten().tolist(),
        "interpretation": "同為 0.75，一份把人工 0 判成 1，另一份把人工 1 判成 0；總比例無法識別錯誤類型，更不能證明動機。",
    },
    "linked_verifier": {
        "source": str(chapter.relative_to(root)),
        "source_sha256": digest(chapter),
        "code": verifier_code,
        "code_sha256": hashlib.sha256(verifier_code.encode()).hexdigest(),
        "stdout": verifier_stdout.getvalue(),
    },
    "result": "全部斷言通過；沒有呼叫真實裁判，也沒有訓練模型。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
