"""Short CPU checks for the actual T.4 commands and every displayed table number."""

import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
records = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "commands": [],
}


def invoke(arguments, cwd=ROOT, success=True):
    proc = subprocess.run([str(ROOT / ".venv/bin/python"), *arguments], cwd=cwd, text=True, capture_output=True)
    records["commands"].append(
        {
            "command": proc.args,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    )
    assert (proc.returncode == 0) == success
    return proc


with tempfile.TemporaryDirectory(prefix="fact_finish_t_4-") as temporary:
    task_dir = Path(temporary)
    for kind in ("toy-text", "attributes-sft"):
        invoke(["scripts/prepare_data.py", "--kind", kind, "--seed", "42", "--output", str(task_dir / "generated")])
        manifest = json.loads((task_dir / "generated" / kind / "manifest.json").read_text())
        records[kind] = manifest
        assert [manifest["splits"][split]["records"] for split in ("train", "validation", "test")] == (
            [9, 1, 2] if kind == "toy-text" else [45, 5, 10]
        )
        assert [manifest["splits"][split]["families"] for split in ("train", "validation", "test")] == [9, 1, 2]
    data = str(task_dir / "generated" / "toy-text" / "train.jsonl")
    full, middle, resumed = [str(task_dir / file) for file in ("full.pt", "middle.pt", "resumed.pt")]
    args = [
        "scripts/train.py",
        "--task",
        "text",
        "--data",
        data,
        "--train",
        "--steps",
        "4",
        "--width",
        "8",
        "--device",
        "cpu",
        "--seed",
        "42",
    ]
    invoke([*args, "--output", full])
    invoke([*args, "--stop-after", "2", "--output", middle])
    invoke([*args, "--checkpoint", middle, "--resume", "--output", resumed])
    direct = torch.load(full, weights_only=True)
    continued = torch.load(resumed, weights_only=True)
    assert direct["step"] == continued["step"] == 4
    differences = {
        key: float((tensor - continued["model"][key]).abs().max()) for key, tensor in direct["model"].items()
    }
    assert max(differences.values()) == 0
    records["resume_probe"] = {
        "total_steps": 4,
        "saved_step": 2,
        "per_tensor_max_difference": differences,
        "same_lr_plan": True,
    }
    invoke([*args, "--checkpoint", resumed, "--resume", "--output", str(task_dir / "reject.pt")], success=False)
    invoke(
        [*args, "--checkpoint", middle, "--resume", "--steps", "5", "--output", str(task_dir / "reject.pt")],
        success=False,
    )
    invoke([*args, "--checkpoint", full, "--steps", "1", "--output", str(task_dir / "new-stage.pt")])
    stage = torch.load(task_dir / "new-stage.pt", weights_only=True)
    assert stage["step"] == 1
    assert {int(value["step"]) for value in stage["optimizer"]["state"].values()} == {1}
    records["new_stage"] = {"saved_step": stage["step"], "optimizer_steps": [1], "origin_step": 4}

foundation = json.loads((OUT / "text_foundation-original.json").read_text())["results"]
expected_short = {"train": ("5.79243", "0.06352"), "validation": ("5.73454", "0.96282"), "test": ("5.74439", "0.61395")}
records["tables"] = {"short_text": {}}
for split, expected in expected_short.items():
    values = (
        (foundation["training"]["initial_loss"], foundation["training"]["final_loss"])
        if split == "train"
        else (foundation["before"][split]["nll"], foundation["after"][split]["nll"])
    )
    actual = tuple(f"{v:.5f}" for v in values)
    assert actual == expected
    records["tables"]["short_text"][split] = actual
real = json.loads((OUT / "real_text-original.json").read_text())["results"]["runs"]
for identifier, expected in {
    "tinystories": [("5.76175", "1.72200"), ("5.76091", "1.80083"), ("5.75491", "1.78880")],
    "chinese-poetry": [("5.70992", "1.85955"), ("5.71337", "2.58325"), ("5.71744", "2.56432")],
}.items():
    run = real[identifier]
    observed = [(f"{run['training']['initial_loss']:.5f}", f"{run['training']['final_loss']:.5f}")]
    observed += [
        (f"{run['before'][side]['nll']:.5f}", f"{run['after'][side]['nll']:.5f}") for side in ("validation", "test")
    ]
    assert observed == expected
    records["tables"][identifier] = observed
assert f"{real['tinystories']['after']['validation']['bpb_including_eos_boundary_targets']:.5f}" == "2.60117"
assert f"{real['chinese-poetry']['after']['validation']['bpb_including_eos_boundary_targets']:.5f}" == "3.74695"
body = next(body for lid, body in sections(ROOT / "course/training.md") if lid == "T.4")
assert hashlib.sha256(body.encode()).hexdigest() == hashlib.sha256((OUT / "section.md").read_bytes()).hexdigest()
records["source_sha256"] = hashlib.sha256(body.encode()).hexdigest()
records["scope"] = (
    "Only four updates on a width-8 CPU toy model test CLI state restoration; no L4 or full schedule replay."
)
(OUT / "contracts.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            "status": "pass",
            "tables": records["tables"],
            "resume_max_difference": max(records["resume_probe"]["per_tensor_max_difference"].values()),
            "environment": records["environment"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
