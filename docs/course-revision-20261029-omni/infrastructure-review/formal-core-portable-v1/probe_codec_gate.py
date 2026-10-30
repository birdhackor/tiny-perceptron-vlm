"""Review-only tiny development codec and formal prerequisite failure injection."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "original-source"
sys.path.insert(0, str(SOURCE))
from scripts.omni_revision import formal_local
from tiny_perceptron.omni_products import training_runtime as trainer
from tiny_perceptron.omni_products.inference import load_trained

probe = BASE / "probes/codec-gate"
probe.mkdir(parents=True, exist_ok=False)
output_root = probe / "review-output"
codec_path = output_root / "codec/training"
config = {"seed": 20261029, "updates": 2, "training_seconds": 20, "batch_size": 8,
          "learning_rate": 0.002, "validate_every": 1, "hop": 80, "codes": 256,
          "width": 32, "variants": 1, "training_dead_code_reset_every": 1,
          "review_scope": "tiny development, not formal initialization"}
result = trainer.train_codec(config, codec_path, "cpu")
loaded = load_trained(codec_path, "codec")
assert result["complete"] and result["updates"] == 2 and result["source_unchanged_during_training"]

frozen = probe / "review-frozen"
frozen.mkdir()
recipe_path = frozen / "formal-recipes.json"
recipe_path.write_bytes((SOURCE / "docs/omni-products/v1/formal-recipes.json").read_bytes())
document = json.loads(recipe_path.read_bytes())
trainer.json_file(frozen / "codec-selection.json", {
    "formal_recipe_sha256": trainer.file_sha(recipe_path),
    "weights_sha256": trainer.file_sha(codec_path / "weights.safetensors"),
    "training_result_sha256": trainer.file_sha(codec_path / "training-result.json"),
    "final_opened": False,
    "review_fixture": "Deliberately points at a two-update development codec, no formal codec job receipt exists."
})
entered = []

def refuse_actual_talker(config, output, device, codec, resume=None):
    entered.append({"codec_parameter_sha256": trainer.parameter_sha(codec),
                    "codes": codec.config.codes, "width": codec.config.width,
                    "codec_all_frozen": all(not p.requires_grad for p in codec.parameters()),
                    "talker_requested_updates": config["updates"]})
    raise RuntimeError("review sentinel: no Talker training performed")

with patch.object(formal_local, "checked_job", return_value=(document["jobs"]["talker"], document)), \
     patch.object(trainer, "train_talker", side_effect=refuse_actual_talker):
    try:
        formal_local.execute("talker", frozen, output_root)
    except RuntimeError as error:
        exception = str(error)
    else:
        exception = None

value = {"scope": "Review prerequisite gate only; checked_job audit fixture was bypassed; actual codec loader and execute/train dispatch were used. No formal output-root, final data or Talker optimizer was used.",
         "development_codec_complete": result["complete"], "development_codec_updates": result["updates"],
         "required_formal_codec_updates": document["jobs"]["codec"]["training"]["updates"],
         "development_codec_metadata_sha256": trainer.file_sha(codec_path / "training-result.json"),
         "formal_codec_job_receipt_exists": (output_root / "codec/job-result.json").exists(),
         "formal_codec_origin_exists": (output_root / "codec/job-origin.json").exists(),
         "talker_training_entry_reached": bool(entered), "entered": entered,
         "result_is_formal_success": False, "sentinel_caught": exception is not None}
trainer.json_file(BASE / "original-codec-gate-result.json", value)
print("REVIEW_RESULT=" + json.dumps(value, ensure_ascii=False))
