"""只核必要前置19.16的逐候選完整編碼；不測歷史模型品質。"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
source = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix04/sources/19.16.md"
raw = source.read_bytes()
assert sha256(raw).hexdigest() == "0f36f932b20323884dc778ac8c51724d79191953a515e7582b46a7b0469e7ca0"
code = re.findall(r"^```python\n(.*?)^```", raw.decode(), re.M | re.S)[0]
stream, ns = StringIO(), {}
with redirect_stdout(stream):
    exec(compile(code, str(source) + ":python-0", "exec"), ns)
torch = ns["torch"]
assert len(ns["sequences"]) == 4
assert len(ns["tokens"]) == 4
record = {
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-23-v1/context-execution.py",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source": str(source.relative_to(ROOT)), "source_sha256": sha256(raw).hexdigest(),
    "code_sha256": sha256(code.encode()).hexdigest(),
    "repository_model_sha256": sha256((ROOT / "tiny_perceptron/lesson_products/model.py").read_bytes()).hexdigest(),
    "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "dtype": "float32", "seed": "0", "threads": "1"},
    "stdout": stream.getvalue(),
    "result": "PASS: 舊19.16原樣第一段為每個候選各建state/question/meaning完整序列，兩種順序的四組assert均通過；只核舊新版分工，不核舊checkpoint品質。",
}
out = Path(__file__).with_name("context-execution.json")
assert not out.exists()
out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(record["result"])
