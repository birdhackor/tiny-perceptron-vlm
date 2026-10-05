"""Current API namespace receipt; CPU inspection only, no CUDA calls."""
import json
import platform
import torch

assert torch.__version__ == '2.14.1+cpu'
assert torch.cuda.max_memory_allocated is torch.cuda.memory.max_memory_allocated
print(json.dumps({'python':platform.python_version(), 'torch':torch.__version__,
                  'api_module':torch.cuda.memory.max_memory_allocated.__module__,
                  'legacy_export_same_object':True, 'cuda_memory_api_called':False},indent=2))
