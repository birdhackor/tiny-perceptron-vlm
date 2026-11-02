"""只核 W.3 的 B/T/D 名稱與三軸形狀例子，不宣稱是文字模型實測。"""

import json
import platform
from datetime import UTC, datetime

import torch

torch.set_num_threads(1)
B, T, D = 2, 4, 3
sample = torch.arange(B * T * D).reshape(B, T, D)
assert [B, T, D] == [2, 4, 3]
assert list(sample.shape) == [2, 4, 3]
assert list(sample[0].shape) == [4, 3]
print(json.dumps({
    "created_at": datetime.now(UTC).isoformat(),
    "command": "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-w-3-v1/notation_probe.py > docs/technical-reviews/artifacts/omni-technical-w-3-v1/notation-execution.json",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "assignment": "B, T, D = 2, 4, 3",
    "ordinary_names_list": [B, T, D],
    "tensor_shape": list(sample.shape),
    "after_first_axis_index_0_shape": list(sample[0].shape),
    "assertions": "all passed",
    "scope": "B/T/D 是本節自行指定的軸標記；範例只有 24 個合成整數，未執行句子編碼或模型。",
}, ensure_ascii=False, indent=2))
