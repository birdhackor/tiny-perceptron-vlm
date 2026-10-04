"""Verify original evidence is preserved and release snapshots are byte equivalent."""

import hashlib
import json
import platform
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_17_"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt_path = OUT / (PREFIX + "closure_copy_receipt.json")
    receipt = json.loads(receipt_path.read_text())
    previous = json.loads((OUT / (PREFIX + "before_closure_report.json")).read_text())
    assert sha(OUT / (PREFIX + "before_closure_report.json")) == receipt["original_report_sha256"]
    for snapshot in receipt["previous_versions"].values():
        assert sha(ROOT / snapshot["saved_path"]) == snapshot["sha256"]
    for original, record in receipt["copied_original_bytes"].items():
        source, copy = ROOT / original, ROOT / record["copied_path"]
        assert source.read_bytes() == copy.read_bytes()
        assert sha(source) == sha(copy) == record["original_sha256"] == record["copied_sha256"]
    raw = json.loads((OUT / (PREFIX + "published_raw_experiment.json")).read_text())
    published = json.loads((ROOT / "docs/course-experiments/results/posttraining.json").read_text())
    assert raw == published["results"]
    sft = json.loads((OUT / (PREFIX + "fresh_sft_state.json")).read_text())
    source_checkpoint = ROOT / sft["source_checkpoint_path"]
    assert sha(source_checkpoint) == sft["source_checkpoint_sha256"]
    original = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    reconstructed = {}
    for name, item in sft["state"].items():
        assert item["dtype"] == "torch.float32"
        value = torch.tensor(item["values"], dtype=torch.float32).reshape(item["shape"])
        assert torch.equal(value, original["model"][name])
        reconstructed[name] = value
    assert sum(value.numel() for value in reconstructed.values()) == 148
    state_digest = hashlib.sha256()
    for name, value in sorted(reconstructed.items()):
        state_digest.update(name.encode())
        state_digest.update(value.numpy().tobytes())
    assert state_digest.hexdigest() == sft["state_sha256"] == published["results"]["reference_state_sha256_before"]
    for source in previous["sources"]:
        if source["kind"] == "repository_code":
            assert sha(ROOT / source["path"]) == source["sha256"]
    current_body = dict(sections(ROOT / "course/chapters/13.md"))["13.17"]
    assert hashlib.sha256(current_body.encode()).hexdigest() == previous["source_sha256"]
    for location, expected in previous["review_scope"]["read_prerequisite_sha256"].items():
        path, lesson = location.split("#")
        body = dict(sections(ROOT / path))[lesson]
        assert hashlib.sha256(body.encode()).hexdigest() == expected
    receipt["command"] = ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_17_closure_verify.py"
    receipt["environment"] = {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}
    receipt["result"] = "All assertions passed: original files retained; raw/records snapshots byte-identical; 148 SFT tensor values losslessly preserved; reviewed lesson, prerequisites and repository sources unchanged. No training."
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(receipt["result"])


if __name__ == "__main__":
    main()
