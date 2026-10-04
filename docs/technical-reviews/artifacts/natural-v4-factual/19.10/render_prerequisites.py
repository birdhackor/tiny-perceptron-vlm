from pathlib import Path
import hashlib
import json
import subprocess

out = Path("docs/technical-reviews/artifacts/natural-v4-factual/19.10")
receipts = []
for name in ["architecture_quantize_grid", "architecture_int4_packing", "capstone_resources", "capstone_pipeline", "capstone_tool_loop"]:
    source = Path("course/figures") / (name + ".svg")
    destination = out / (name + ".png")
    command = ["inkscape", str(source), "--export-type=png", "--export-filename=" + str(destination)]
    run = subprocess.run(command, capture_output=True, text=True)
    receipt = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "command": command, "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    if run.returncode == 0:
        receipt.update(render=str(destination), render_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    receipts.append(receipt)
    print(json.dumps(receipt))
    run.check_returncode()
out.joinpath("render-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
