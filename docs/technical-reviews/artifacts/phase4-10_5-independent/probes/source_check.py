"""Source-version receipt; copies are raw upstream bytes, not other reviews."""
import hashlib
import importlib
import json
from pathlib import Path
import torch

ART = Path(__file__).resolve().parents[1]
MODULES = {
    "torch--_tensor.py": "torch._tensor",
    "torch--_torch_docs.py": "torch._torch_docs",
    "torch--nn--functional.py": "torch.nn.functional",
    "torch--nn--modules--linear.py": "torch.nn.modules.linear",
    "torch--nn--parameter.py": "torch.nn.parameter",
    "torch--nn--modules--loss.py": "torch.nn.modules.loss",
}
result = {"torch_version": str(torch.__version__), "torch_git_version": torch.version.git_version,
          "source_check": [], "scope": "compare installed modules to the immutable upstream source snapshots; manual reading is recorded separately"}
for name, module in MODULES.items():
    installed = Path(importlib.import_module(module).__file__)
    saved = ART / "sources" / name
    a = hashlib.sha256(installed.read_bytes()).hexdigest()
    b = hashlib.sha256(saved.read_bytes()).hexdigest()
    assert a == b
    result["source_check"].append({"module": module, "installed_file": str(installed),
                                   "snapshot_file": str(saved), "snapshot_sha256": b,
                                   "matches_installed_bytes": True})
(ART / "probes/source-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
