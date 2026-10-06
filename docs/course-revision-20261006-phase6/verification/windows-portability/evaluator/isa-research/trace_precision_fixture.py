"""Observe an existing disposable unit fixture without changing its assertions."""

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path.cwd()))
from scripts.course_experiments import architecture


class ObserveAddmm(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) == "aten.addmm.default":
            print(
                json.dumps(
                    {
                        "event": "fixture_addmm",
                        "operator": str(func),
                        "inputs": [
                            {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "stride": list(value.stride()),
                            }
                            for value in args[:3]
                        ],
                    }
                ),
                flush=True,
            )
        return func(*args, **(kwargs or {}))


original_nll = architecture._nll


def observe_nll(model, examples, ctx, dtype=None, forward=None):
    print(
        json.dumps(
            {
                "event": "fixture_nll",
                "autocast_dtype": str(dtype),
                "device": str(ctx.device),
            }
        ),
        flush=True,
    )
    with ObserveAddmm():
        return original_nll(model, examples, ctx, dtype, forward)


architecture._nll = observe_nll
print(
    json.dumps(
        {
            "torch": str(torch.__version__),
            "aten_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        }
    )
)
print(torch.__config__.show())
raise SystemExit(
    pytest.main(
        [
            "-s",
            "-q",
            "tests/test_architecture_completion.py",
            "-k",
            "public_entries and precision",
        ]
    )
)
