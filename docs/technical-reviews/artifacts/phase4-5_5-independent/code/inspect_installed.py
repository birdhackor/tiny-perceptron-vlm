"""Record the exact installed CPU implementation, not an inferred current API."""
from pathlib import Path
import hashlib
import inspect
import json
import platform
import sys
import torch
import torch.optim.adam as adam_module
import torch.optim.adamw as adamw_module
import torch.optim.optimizer as optimizer_module
import torch.nn.parameter as parameter_module

BASE = Path(__file__).resolve().parents[1]
records = []
for name, module in [("adam", adam_module), ("adamw", adamw_module),
                     ("optimizer", optimizer_module), ("parameter", parameter_module)]:
    original = Path(inspect.getfile(module))
    data = original.read_bytes()
    path = BASE / "sources" / ("installed-" + name + ".py")
    path.write_bytes(data)
    records.append({"module": module.__name__, "input_path": str(original),
                    "artifact_path": str(path.relative_to(BASE)),
                    "sha256": hashlib.sha256(data).hexdigest()})
doc = torch.zeros_like.__doc__
(BASE / "sources/installed-zeros-like-docstring.txt").write_text(doc)
record = {"python": sys.version, "python_executable": sys.executable,
          "torch": torch.__version__, "torch_git_commit": torch.version.git_version,
          "cuda_build": str(torch.version.cuda), "cuda_available": torch.cuda.is_available(),
          "device": "cpu", "platform": platform.platform(), "threads": torch.get_num_threads(),
          "sources": records}
(BASE / "results/installed-environment.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
