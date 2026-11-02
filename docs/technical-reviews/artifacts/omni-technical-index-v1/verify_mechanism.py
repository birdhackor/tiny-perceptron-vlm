"""首頁接字例子的必要前提：原版1.1程式與BigramLM的CPU核對。"""

import contextlib
import hashlib
import io
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.simple import BigramLM


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    chapter = ROOT / "course/chapters/01.md"
    source = chapter.read_text()
    section = source.split("## 1.1 ", 1)[1].split("\n## ", 1)[0]
    code = re.search(r"```python\n(.*?)\n```", section, re.S)[1]
    stream = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "course/chapters/01.md#1.1:first-python-fence", "exec"), namespace)
    assert namespace["restored"] == "貓看狗，狗看貓。"
    assert namespace["ids"] == [3, 2, 1, 4, 1, 2, 3, 0]
    model = BigramLM(5)
    with torch.no_grad():
        model.table.weight.copy_(torch.arange(25, dtype=torch.float32).reshape(5, 5))
    contexts = torch.tensor([[3, 2], [1, 2]], dtype=torch.int64)
    logits = model(contexts)
    assert torch.equal(logits[0], logits[1])
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-index-v1/verify_mechanism.py",
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu"},
        "inline_prerequisite": {"source": "course/chapters/01.md#1.1:first-python-fence",
                                "full_source_sha256": sha(chapter.read_bytes()), "code_sha256": sha(code.encode()),
                                "input": "text = '貓看狗，狗看貓。'", "stdout": stream.getvalue(),
                                "restored": namespace["restored"], "ids": namespace["ids"]},
        "bigram": {"source": "tiny_perceptron/simple.py:7-13", "source_sha256": sha((ROOT / "tiny_perceptron/simple.py").read_bytes()),
                   "character_ids": {"貓": 3, "狗": 1, "看": 2}, "contexts": contexts.tolist(),
                   "dtype": str(contexts.dtype), "table_shape": [5, 5], "table_fill": "arange(25).reshape(5,5)",
                   "logits": logits.tolist(), "prefix_ignored_same_logits": True},
        "scope": "1.1當版內嵌程式的記錄可逆；固定上一字的接字表會選相同列。沒有訓練、沒有品質或泛化推論；首頁本身無程式圍欄。",
    }
    Path(__file__).with_name("mechanism-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
