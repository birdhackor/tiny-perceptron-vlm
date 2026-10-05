import sys
from pathlib import Path
import torch
artifact = Path(__file__).resolve().parent
root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
torch.set_num_threads(1)
torch.manual_seed(42)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
code = (artifact / "exercise-fence.py").read_bytes()
exec(compile(code, str(artifact / "exercise-fence.py"), "exec"), {"__name__": "__main__"})
