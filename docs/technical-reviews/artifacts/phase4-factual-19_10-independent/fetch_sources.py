"""Save official primary sources; no model or training-data download."""
import hashlib
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'sources'
SOURCES = {
    'pytorch-linear.py': 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/linear.py',
    'pytorch-module.py': 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/module.py',
    'pytorch-functional.py': 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py',
    'pytorch-tensor-docs.py': 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_tensor_docs.py',
    'pytorch-torch-docs.py': 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py',
    'pytorch-quantization.rst': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/quantization.rst',
    'linked-capstone-deployment.json': 'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_deployment.json',
    'linked-capstone-student.json': 'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_student.json',
}
receipts = []
for name, url in SOURCES.items():
    with urllib.request.urlopen(url, timeout=25) as response:
        raw = response.read()
        final_url = response.url
    (OUT / name).write_bytes(raw)
    item = {'file': name, 'url': url, 'final_url': final_url, 'accessed_on': '2026-10-05', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    receipts.append(item)
    print(json.dumps(item))
(OUT / 'receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
