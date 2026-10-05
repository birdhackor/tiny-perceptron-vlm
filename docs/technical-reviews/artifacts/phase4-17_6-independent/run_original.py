"""Execute the unmodified lesson fence in the repository's CPU environment."""
import hashlib
import json
import platform
from pathlib import Path
import sys

import torch

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
artifact = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.manual_seed(42)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
code = (artifact / "fence-1.py").read_bytes()
environment = {
    "python": sys.version,
    "torch": torch.__version__,
    "torch_git": torch.version.git_version,
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "platform": platform.platform(),
    "fence_sha256": hashlib.sha256(code).hexdigest(),
    "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-17_6-independent/run_original.py",
}
(artifact / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
exec(compile(code, str(artifact / "fence-1.py"), "exec"), {"__name__": "__main__"})
