"""Keep public experiment evidence separate from billing and private data samples."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TEXT_KEYS = {"samples", "messages", "prompt", "generated", "text", "target", "chosen", "rejected"}


def clarify_backup_scope(result):
    """Preserve the original observation and annotate an inaccurate older scope sentence."""
    hf = result.get("hf", {})
    if "encoder and simple inference snapshots do not contain those states" in hf.get("checkpoint_resume_scope", ""):
        hf["checkpoint_resume_scope_correction"] = (
            "The original scope sentence is inaccurate for the private encoder files: run_encoders copies "
            "the custom _fit payload, including optimizer, CPU torch RNG, Python RNG and step, into its final "
            "vision.pt/audio.pt. These custom files are not native CLI resume checkpoints, and they omit CUDA "
            "RNG state. Simple-v1 files are weights/vocabulary only. Public inference exports remove optimizer "
            "and RNG state. Full-directory backup alone does not establish exact resumed training."
        )
    return result


def redact(value, private=False):
    if isinstance(value, list):
        return [redact(item, private) for item in value]
    if isinstance(value, dict):
        private = private or value.get("private_only") is True
        result = {
            key: redact(item, private) for key, item in value.items() if not (private and key in PRIVATE_TEXT_KEYS)
        }
        if value.get("private_only") is True:
            result["public_evidence_note"] = (
                "Only aggregate metrics and provenance; all sample text and derived weights remain private."
            )
        return result
    return value


def ingest(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("evidence_status") != "complete_run":
        raise ValueError("Only a complete formal run may replace a public result")
    billing = result.pop("billing", {})
    result.pop("preflight", None)
    result = clarify_backup_scope(redact(result))
    destination = ROOT / "docs/course-experiments/results" / f"{result['experiment_id']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"evidence": str(destination.relative_to(ROOT)), "reserved_total_usd": billing.get("reserved_total_usd")}
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    ingest(args.result)


if __name__ == "__main__":
    main()
