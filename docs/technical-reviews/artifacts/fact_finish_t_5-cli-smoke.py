"""Run the T.5 generator/train/evaluate interface with one CPU update per task."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "docs/technical-reviews/artifacts/fact_finish_t_5-cli-smoke-output.json"
WORK = Path("/tmp/fact_finish_t_5-cli")


def main():
    commands = []
    checks = {}
    for kind in ("style", "safety"):
        checkpoint = WORK / f"{kind}.pt"
        report_path = WORK / f"{kind}-validation.json"
        for arguments in (
            ["scripts/prepare_data.py", "--kind", kind, "--output", str(WORK / "data")],
            [
                "scripts/train.py",
                "--task",
                "sft",
                "--data",
                str(WORK / "data" / kind / "train.jsonl"),
                "--train",
                "--steps",
                "1",
                "--max-length",
                "256",
                "--width",
                "16",
                "--layers",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--output",
                str(checkpoint),
            ],
            [
                "scripts/evaluate.py",
                str(checkpoint),
                "--data",
                str(WORK / "data" / kind / "validation.jsonl"),
                "--mode",
                "sft",
                "--tokens",
                "2",
                "--limit",
                "all",
                "--device",
                "cpu",
                "--output",
                str(report_path),
            ],
        ):
            command = [sys.executable, *arguments]
            record = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            commands.append(
                {"command": command, "returncode": record.returncode, "stdout": record.stdout, "stderr": record.stderr}
            )
        manifest = json.loads((WORK / "data" / kind / "manifest.json").read_text())
        report = json.loads(report_path.read_text())
        data = [json.loads(line) for line in (WORK / "data" / kind / "validation.jsonl").read_text().splitlines()]
        assert report["skipped"] == []
        assert (
            report["records_read"] == report["records_selected"] == report["generation_evaluated_records"] == len(data)
        )
        assert report["effective_tokens"] == sum(len(row["messages"][-1]["content"].encode()) + 1 for row in data)
        assert report["effective_tokens"] == manifest["splits"]["validation"]["effective_answer_tokens"]
        assert [row["row"] for row in report["samples"]] == list(range(len(data)))
        assert all(row["target"] == data[row["row"]]["messages"][-1]["content"] for row in report["samples"])
        assert all(len(row["generated_ids"]) <= 2 for row in report["samples"])
        checks[kind] = {"manifest": manifest, "report": report}
    OUTPUT.write_text(
        json.dumps(
            {
                "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_finish_t_5-cli-smoke.py",
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
                "result": "All six generator/training/evaluation commands and row/denominator assertions passed",
                "commands": commands,
                "checks": checks,
                "scope": "CLI smoke with one update, width16/layer1/batch2 and two generated units. No 500-step quality claim; complete original recipes checked against source.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("Six commands passed; style/safety row indices, targets and loss denominators match source manifests.")


if __name__ == "__main__":
    main()
