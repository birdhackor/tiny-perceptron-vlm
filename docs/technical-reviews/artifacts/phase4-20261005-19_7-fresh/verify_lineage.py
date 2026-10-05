import json
from pathlib import Path

ART = Path(__file__).resolve().parent
p = json.loads((ART / "raw-provenance-inspection.json").read_bytes())
assert p["joint"]["raw_provenance"]["parent_checkpoint_sha256"] == p["sft"]["raw_provenance"]["inference_export"]["sha256"]
assert p["dpo"]["raw_provenance"]["parent_checkpoint_sha256"] == p["joint"]["raw_provenance"]["inference_export"]["sha256"]
test = json.loads((ART / "originals/docs/course-experiments/capstone-evidence/deployment/test-joint.json").read_bytes())
assert test["checkpoint_sha256"] == p["joint"]["raw_provenance"]["inference_export"]["sha256"]
for stage in ("sft", "joint", "dpo"):
    assert p[stage]["raw_provenance"]["code_sha256"]["tiny_perceptron/capstone.py"] == "d02d1cd86cb4f6bbfbc1718c140107a5d6d3a128a9e20e183d4a9cd38e45edbe"
    assert p[stage]["raw_provenance"]["code_sha256"]["scripts/course_experiments/capstone.py"] == "63fcc9d1d13503fa3b2ce28d6ac7fa6040d659f51fd8865ed18fabbe3d0fa5cf"
print("SFT -> joint -> DPO parent checkpoint lineage, joint test checkpoint, and code SHA provenance agree.")
