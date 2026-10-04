from pathlib import Path
import runpy
import sys
import torch

sys.path.insert(0, str(Path.cwd()))
torch.set_num_threads(1)
torch.manual_seed(42)
runpy.run_path(sys.argv[1], run_name="__main__")
