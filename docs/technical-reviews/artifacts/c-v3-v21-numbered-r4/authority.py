"""Save bounded original-source excerpts for the independent R.4 review."""
import hashlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import torch

DEST = Path(__file__).parent
records = []
for title, url, phrases in [
    (
        "PyTorch Learn the Basics: Optimization Loop",
        "https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/basics/optimization_tutorial.py",
        ["weights and biases", "training loop", "test loop", "optimizer.step", "test_loop"],
    ),
    (
        "PyTorch Saving and Loading Models",
        "https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/saving_loading_models.py",
        ["learnable parameters", "weights and biases", "state_dict"],
    ),
    (
        "OpenAI Whisper original README",
        "https://raw.githubusercontent.com/openai/whisper/main/README.md",
        ["trained", "load_model", "Models and languages"],
    ),
    (
        "Qwen3-VL-2B-Instruct original model card",
        "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/README.md",
        ["from_pretrained", "Qwen3-VL", "Pre-trained"],
    ),
]:
    raw = urlopen(Request(url, headers={"User-Agent": "R4-technical-review"}), timeout=25).read()
    lines = raw.decode().splitlines()
    selected = set()
    for i, line in enumerate(lines):
        if any(phrase.lower() in line.lower() for phrase in phrases):
            selected.update(range(max(0, i - 2), min(len(lines), i + 5)))
    # Bound quotations, while retaining exact line locators and the full-file identity.
    excerpts = [{"line": i + 1, "text": lines[i]} for i in sorted(selected)[:85]]
    assert excerpts
    records.append({
        "title": title, "url": url, "version": "main at retrieval; full bytes pinned by SHA-256",
        "accessed_at": datetime.now(UTC).isoformat(), "original_bytes": len(raw),
        "original_sha256": hashlib.sha256(raw).hexdigest(), "excerpts": excerpts,
    })
for cls, module_path in [(torch.nn.Parameter, 'parameter.py'), (torch.nn.Embedding, 'modules/sparse.py'), (torch.nn.Linear, 'modules/linear.py')]:
    original_file = Path(inspect.getfile(cls))
    records.append({
        "title": f"PyTorch original {cls.__name__} implementation and docstring",
        "url": "https://github.com/pytorch/pytorch/blob/main/torch/nn/" + module_path,
        "version": "installed official torch " + torch.__version__,
        "accessed_at": datetime.now(UTC).isoformat(),
        "original_sha256": hashlib.sha256(original_file.read_bytes()).hexdigest(),
        "original_path": str(original_file),
        "locator": f"class {cls.__name__} docstring",
        "excerpt": inspect.getdoc(cls),
    })
out = {"reviewer_task": "/root/c_v3_technical_coord/numbered_r4", "sources": records}
(DEST / "authority.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"sources": [{"title": r['title'], "sha256": r['original_sha256']} for r in records]}, ensure_ascii=False))
