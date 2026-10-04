"""Bounded matched CPU timing; sample interval excludes profiler start/stop."""
import json
import platform
import statistics
import sys
import time
from pathlib import Path

OUT = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch
from torch.profiler import profile, ProfilerActivity
from tiny_perceptron.model import TinyLM, ModelConfig

torch.manual_seed(42)
torch.set_num_threads(1)
model = TinyLM(ModelConfig(width=8)).eval()
ids = torch.tensor([[1, 2, 3]])
plain, instrumented = [], []
with torch.no_grad():
    for _ in range(3):
        model(ids)
    with profile(activities=[ProfilerActivity.CPU]):
        for _ in range(3):
            model(ids)
    for i in range(9):
        for profiled in ((False, True) if i % 2 == 0 else (True, False)):
            if profiled:
                with profile(activities=[ProfilerActivity.CPU]):
                    begin = time.perf_counter()
                    for _ in range(30):
                        model(ids)
                    instrumented.append((time.perf_counter()-begin)/30)
            else:
                begin = time.perf_counter()
                for _ in range(30):
                    model(ids)
                plain.append((time.perf_counter()-begin)/30)

result = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
          "device": "cpu", "threads": "1", "dtype": "torch.float32",
          "model_mode": "eval/no_grad", "seed": "42"},
          "input_ids": [[1,2,3]], "warmup_plain": 3, "warmup_instrumented": 3,
          "measured_samples_per_mode": 9, "forwards_per_sample": 30,
          "plain_seconds_per_forward": plain,
          "profiled_seconds_per_forward": instrumented,
          "plain_median_seconds": statistics.median(plain),
          "profiled_median_seconds": statistics.median(instrumented),
          "scope": "perf_counter around the same 30 forwards; profile start/stop/export excluded; interleaved order; single process with shared machine load"}
(OUT/'timing-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
