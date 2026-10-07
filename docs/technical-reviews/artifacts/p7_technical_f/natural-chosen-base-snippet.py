import json
import shutil
from hashlib import sha256
from pathlib import Path

manifest = Path("docs/natural-assistant/v4/manifest.json")
validation = Path("outputs/natural-my-v4/validation/result.json")
protocol = Path("docs/natural-assistant/v4/validation-protocol-lower-lr.json")
record = json.loads(validation.read_text())
assert record["status"] == "completed" and record["split"] == "validation"
assert record["manifest_sha256"] == sha256(manifest.read_bytes()).hexdigest()
configuration = {
    key: record[key]
    for key in (
        "model", "model_revision", "asr_model", "asr_revision",
        "device", "dtype", "min_pixels", "max_pixels", "max_tokens", "seed",
        "manifest_sha256",
    )
}
configuration.update(
    selected_variant="base", adapter=None, max_new_tokens=384, do_sample=False,
)
snapshot = Path("outputs/natural-my-v4/chosen-base-before-test")
snapshot.mkdir()
for source in (manifest, validation, protocol):
    shutil.copyfile(source, snapshot / source.name)
(snapshot / "configuration.json").write_text(
    json.dumps(configuration, ensure_ascii=False, indent=2) + "\n"
)
(snapshot / "files.sha256").write_text(
    "".join(
        f"{sha256(file.read_bytes()).hexdigest()}  {file.name}\n"
        for file in sorted(snapshot.iterdir())
    )
)
print(snapshot)