"""讀取原始來源，只保存定位、雜湊與支持查核的小段摘錄。"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
TORCH_COMMIT = "5c4886908584029761b579af026dcfb627c84070"


def fetch(url):
    command = ["curl", "-fsSL", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, command


def record(identifier, url, version, selectors):
    raw, command = fetch(url)
    text = raw.decode("utf-8")
    excerpts = []
    for selector, start, end in selectors:
        first = text.index(start)
        last = text.index(end, first) + len(end)
        excerpts.append({"locator": selector, "excerpt": text[first:last]})
    return {
        "id": identifier,
        "url": url,
        "version": version,
        "command_argv": command,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "full_response_sha256": hashlib.sha256(raw).hexdigest(),
        "full_response_bytes": len(raw),
        "excerpts": excerpts,
    }


records = [
    record(
        "lora-readme",
        "https://raw.githubusercontent.com/microsoft/LoRA/main/README.md",
        "main; exact response identified by full_response_sha256",
        [("Quickstart: loading a checkpoint", "# Load the pretrained checkpoint first", "# Then load the LoRA checkpoint")],
    ),
    record(
        "lora-linear",
        "https://raw.githubusercontent.com/microsoft/LoRA/main/loralib/layers.py",
        "main; exact response identified by full_response_sha256",
        [
            ("Linear.__init__: scaling", "self.scaling = self.lora_alpha / self.r", "self.scaling = self.lora_alpha / self.r"),
            ("Linear.train: merge", "self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling", "self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling"),
        ],
    ),
    record(
        "torch-grad-mode",
        f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_COMMIT}/torch/autograd/grad_mode.py",
        "PyTorch 2.14.1; " + TORCH_COMMIT,
        [("no_grad", "Context-manager that disables gradient calculation.", "Context-manager that disables gradient calculation.")],
    ),
    record(
        "torch-tensor-docs",
        f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_COMMIT}/torch/_tensor_docs.py",
        "PyTorch 2.14.1; " + TORCH_COMMIT,
        [
            ("Tensor.copy_", "Copies the elements from :attr:`src` into :attr:`self` tensor and returns\n:attr:`self`.", "Copies the elements from :attr:`src` into :attr:`self` tensor and returns\n:attr:`self`."),
            ("Tensor.fill_", "Fills :attr:`self` tensor with the specified value.", "Fills :attr:`self` tensor with the specified value."),
        ],
    ),
    record(
        "torch-function-docs",
        f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_COMMIT}/torch/_torch_docs.py",
        "PyTorch 2.14.1; " + TORCH_COMMIT,
        [
            ("torch.clone", "Returns a copy of :attr:`input`.", "Returns a copy of :attr:`input`."),
            ("torch.equal", "``True`` if two tensors have the same size and elements, ``False`` otherwise.", "``True`` if two tensors have the same size and elements, ``False`` otherwise."),
        ],
    ),
    record(
        "torch-numerical-accuracy",
        f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_COMMIT}/docs/source/notes/numerical_accuracy.md",
        "PyTorch 2.14.1; " + TORCH_COMMIT,
        [("Floating point operations", "floating point addition and multiplication are not\nassociative", "floating point addition and multiplication are not\nassociative")],
    ),
]
output = DIRECTORY / "original-sources.json"
with output.open("x", encoding="utf-8") as handle:
    json.dump({"sources": records}, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"path": str(output.relative_to(Path.cwd())), "sources": len(records)}, ensure_ascii=False))
