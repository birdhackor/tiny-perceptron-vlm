"""Tiny same-source continuation and missing-alarm rejection probes."""
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BASE / "original-source"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "original"
sys.path.insert(0, str(SOURCE))
import torch
from scripts.omni_revision import formal_local
from tiny_perceptron.omni_products import training_runtime as trainer
from tiny_perceptron.omni_products.inference import load_trained
from tiny_perceptron.omni_products.sound import SoundCodec, CodecConfig

root = BASE / "probes" / (LABEL + "-runtime-limits")
root.mkdir(parents=True, exist_ok=False)
config = {"seed": 31, "updates": 4, "training_seconds": 20, "batch_size": 8,
          "learning_rate": 0.001, "validate_every": 1, "training_dead_code_reset_every": 2,
          "model": asdict(CodecConfig(hop=80, width=16, codes=32)), "scope": "tiny review train/dev synthetic blocks"}

def run(name, cutoff=None, resume=None):
    trainer.deterministic(31 if resume is None else 999)
    model = SoundCodec(CodecConfig(hop=80, width=16, codes=32))
    counter = [0]
    def batch(generator):
        counter[0] += 1
        return torch.randn(8, 80, generator=generator) * random.random()
    clock = SimpleNamespace(monotonic=lambda: 100.0 if cutoff is not None and counter[0] >= cutoff else 0.0)
    try:
        with patch.object(trainer, "time", clock):
            result = trainer.run_updates(model, batch, lambda m, x: m(x), lambda _: 0.0,
                                         config, root / name, resume=resume)
        return model, result, None
    except Exception as error:
        return model, None, {"type": type(error).__name__, "reason": str(error)}

direct, direct_result, _ = run("direct")
_, partial, _ = run("partial", cutoff=2)
checkpoint = root / "partial/latest-continuation.pt"
resumed, resumed_result, resume_error = run("resumed-newdir", resume=(checkpoint, trainer.file_sha(checkpoint)))
continuation = {"partial_updates": partial["updates"], "direct_complete": direct_result["complete"],
                "resumed_complete": None if resumed_result is None else resumed_result["complete"],
                "resume_error": resume_error, "new_directory_has_best_weights": (root / "resumed-newdir/weights.safetensors").is_file(),
                "parameter_equal": trainer.parameter_sha(direct) == trainer.parameter_sha(resumed)}
try:
    load_trained(root / "resumed-newdir", "codec")
    continuation["inference_load_accepted"] = True
except Exception as error:
    continuation.update(inference_load_accepted=False, inference_error_type=type(error).__name__, inference_error=str(error))

frozen = root / "frozen"
frozen.mkdir()
(frozen / "formal-recipes.json").write_text("{}")
output = root / "missing-alarm-output"
selected = {"kind": "av", "training": {"training_seconds": 2}}
document = {"data_definition_sha256": "test-data", "metrics_sha256": "test-metrics"}
train_calls = []
with patch.object(formal_local, "checked_job", return_value=(selected, document)), \
     patch.object(formal_local, "signal", SimpleNamespace(signal=lambda *_: None)), \
     patch.object(formal_local, "train", side_effect=lambda *args: train_calls.append(args)):
    try:
        formal_local.execute("review-job", frozen, output)
    except Exception as error:
        alarm_error = {"type": type(error).__name__, "reason": str(error)}
    else:
        alarm_error = None
alarm = {"scope": "Simulated platform without SIGALRM/alarm; checked_job only was bypassed, no training invoked.",
         "exception": alarm_error, "train_calls": len(train_calls),
         "lock_left_behind": (output / ".formal-running").exists(),
         "job_origin_written": (output / "review-job/job-origin.json").exists(),
         "job_result_written": (output / "review-job/job-result.json").exists()}

value = {"scope": "Independent review directory, tiny unit training only; no formal success/final predictions.",
         "same_source_continuation": continuation, "missing_alarm": alarm, "formal_success_claimed": False}
trainer.json_file(BASE / (LABEL + "-runtime-limits-result.json"), value)
print("REVIEW_RESULT=" + json.dumps(value,ensure_ascii=False))
