"""Check the trainer's real deadline granularity on CPU."""

import json
import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.capstone import train_stage  # noqa: E402


def main():
    with TemporaryDirectory(prefix="fact_finish_19_11-budget") as temporary:
        report = train_stage("pretrain", temporary, steps=4, seconds=0.001, batch_size=2, dense=True, validation=False)
        saved = torch.load(Path(temporary) / "model-training.pt", weights_only=True)
        result = {
            "requested_loop_budget_seconds": 0.001,
            "observed_training_loop_seconds": report["seconds"],
            "over_budget": report["seconds"] > 0.001,
            "steps": report["steps"],
            "requested_steps": 4,
            "budget_exhausted": report["budget_exhausted"],
            "schedule_completed": report["schedule_completed"],
            "saved_step": saved["step"],
            "model_training_exists": (Path(temporary) / "model-training.pt").is_file(),
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU"},
            "interpretation": "Deadline is checked only before the next complete optimizer update, not a hard wall-time limit.",
        }
    destination = ROOT / "docs/technical-reviews/artifacts/fact_finish_19_11-budget.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
