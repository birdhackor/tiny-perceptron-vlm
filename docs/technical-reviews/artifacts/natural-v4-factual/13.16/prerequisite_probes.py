"""Only the explicit prerequisites' small examples/interfaces used by 13.16."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch

OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
records = {}
for lesson in ["13.10", "13.15", "13.5", "7.13", "13.13", "13.14"]:
    path = OUT / f"prerequisite-{lesson}-original.md"
    code = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)[0]
    buf = io.StringIO()
    ns = {}
    with contextlib.redirect_stdout(buf):
        exec(compile(code, f"{lesson}-prerequisite-example", "exec"), ns)
    if lesson == "13.5":
        assert abs(ns["loss"].item() - math.log(2)) < 1e-6
        assert abs(ns["chosen"].grad.item() + 0.05) < 1e-7
        assert abs(ns["rejected"].grad.item() - 0.05) < 1e-7
    if lesson == "13.13":
        assert torch.allclose(ns["result"]["surrogate"], torch.tensor([0.7,1.0,1.2,-0.8,-1.0,-1.3]), atol=1e-6)
        assert torch.allclose(ns["ratio"].grad, torch.tensor([-1.,-1.,0.,0.,1.,1.]), atol=1e-6)
    if lesson == "13.14":
        assert abs(ns["value_loss"].item() - 0.36) < 1e-6
        assert abs(ns["value"].grad.item() + 1.2) < 1e-6
    if lesson == "13.15":
        assert ns["action"].item() == 0
        assert ns["terms"]["ratio"].item() == 1
        assert not torch.equal(ns["before"], next(ns["policy"].parameters()))
        assert not any(p.grad is not None for p in ns["reward_model"].parameters())
        assert not any(p.grad is not None for p in ns["reference"].parameters())
    records[lesson] = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "stdout": buf.getvalue()}
result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": "2"},
          "records": records,
          "scope": "Read all six prerequisite sections. Executed their first small Python example to inspect interfaces/figure numbers; did not independently review unrelated LM training/free-generation results in those prerequisites."}
(OUT / "prerequisite-probes.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
