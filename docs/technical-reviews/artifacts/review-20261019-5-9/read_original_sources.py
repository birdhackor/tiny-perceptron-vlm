"""以 curl 讀取官方原文，只保存支持主張的短摘錄與來源收據。"""
import concurrent.futures
import hashlib
import html
import json
import re
import subprocess
from datetime import date
from pathlib import Path

DESTINATION = Path(__file__).resolve().parent
SPECS = [
    ("dtype", "https://docs.pytorch.org/docs/2.14/tensor_attributes.html", "PyTorch 2.14", ["32-bit floating point", "A torch.dtype is an object"]),
    ("no_grad", "https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html", "PyTorch 2.14", ["Disabling gradient calculation", "the result of every computation"]),
    ("eval", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html", "PyTorch 2.14", ["Set the module in evaluation mode", "This is equivalent with self.train(False)"]),
    ("autograd", "https://docs.pytorch.org/docs/2.14/notes/autograd.html", "PyTorch 2.14", ["Some operations need intermediary results", "requires_grad"]),
    ("adamw", "https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html", "PyTorch 2.14", ["first moment", "second moment"]),
    ("cuda", "https://docs.pytorch.org/docs/2.14/notes/cuda.html", "PyTorch 2.14", ["By default, GPU operations are asynchronous", "time measurements without synchronizations"]),
    ("benchmark", "https://docs.pytorch.org/docs/2.14/benchmark_utils.html", "PyTorch 2.14", ["Timer will perform warmups", "Focus on replicates"]),
    ("clock", "https://docs.python.org/3.13/library/time.html", "Python 3.13.16 documentation", ["a clock with the highest available resolution"]),
    ("byte", "https://csrc.nist.gov/glossary/term/byte", "NIST CSRC glossary, accessed edition", ["eight bits", "8 bits"]),
]


def read(spec):
    key, url, version, needles = spec
    command = ["curl", "-fsSL", "--max-time", "30", url]
    result = subprocess.run(command, capture_output=True, check=False)
    raw = result.stdout
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", text)))
    excerpts = []
    for needle in needles:
        match = re.search(re.escape(needle), text, re.I)
        excerpts.append({"locator": needle, "found": bool(match), "excerpt": text[max(0, match.start()-70):match.end()+340] if match else ""})
    return {"id": key, "url": url, "version": version, "accessed_on": date.today().isoformat(), "command_argv": command, "exit_code": result.returncode, "stderr": result.stderr.decode(), "response_sha256": hashlib.sha256(raw).hexdigest(), "excerpts": excerpts}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    receipts = list(executor.map(read, SPECS))
(DESTINATION / "official-source-excerpts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(receipts, ensure_ascii=False, indent=2))
