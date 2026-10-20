import hashlib
import json
import subprocess
from datetime import UTC, datetime


TORCH = "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/"
PYTHON = "https://raw.githubusercontent.com/python/cpython/v3.13.16/"
SPECS = [
    ("adam", TORCH + "torch/optim/adam.py", [("Exponential moving average of gradient values", 8), ("# update step", 8), ("# Decay the first and second moment running average coefficient", 26), ("step = _get_value(step_t)", 29)]),
    ("adamw", TORCH + "torch/optim/adamw.py", [("class AdamW(Adam):", 48)]),
    ("serialization", TORCH + "docs/source/notes/serialization.md", [("## `torch.load` with `weights_only=True`", 26), ("## Saving and loading `torch.nn.Modules`", 35)]),
    ("reproducibility", TORCH + "docs/source/notes/randomness.md", [("# Reproducibility", 15), ("### Python", 9)]),
    ("module", TORCH + "torch/nn/modules/module.py", [("    def load_state_dict(", 27)]),
    ("os_replace", PYTHON + "Doc/library/os.rst", [(".. function:: replace(src, dst", 8)]),
    ("temporary_directory", PYTHON + "Doc/library/tempfile.rst", [(".. class:: TemporaryDirectory(", 14)]),
    ("path_operators", PYTHON + "Doc/library/pathlib.rst", [("The slash operator helps create child paths", 17)]),
    ("checkpoint_tutorial", "https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/saving_loading_models.py", [("Saving & Loading a General Checkpoint", 16), ("When saving a general checkpoint", 23)]),
]


def main():
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-7/inspect_original_sources.py", "accessed_on": datetime.now(UTC).date().isoformat(), "retention": "全文僅在記憶體中處理；保存 URL、原文 SHA-256 與支持主張的小段摘錄", "sources": []}
    for identifier, url, selections in SPECS:
        command = ["curl", "-fsSL", "--max-time", "30", url]
        fetched = subprocess.run(command, capture_output=True, check=False)
        item = {"id": identifier, "url": url, "curl_argv": command, "curl_exit_code": fetched.returncode, "stderr": fetched.stderr.decode(), "full_source_sha256": hashlib.sha256(fetched.stdout).hexdigest(), "excerpts": []}
        if fetched.returncode == 0:
            lines = fetched.stdout.decode().splitlines()
            for marker, count in selections:
                matches = [i for i, line in enumerate(lines) if marker in line]
                if matches:
                    position = matches[0]
                    item["excerpts"].append({"locator": marker, "first_line_computed": position + 1, "text": "\n".join(lines[position:position + count])})
                else:
                    item["excerpts"].append({"locator": marker, "found": False})
        result["sources"].append(item)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
