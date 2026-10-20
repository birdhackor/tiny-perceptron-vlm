import ast
import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def download(url):
    command = ["curl", "-L", "--fail", "--silent", "--show-error", url]
    result = subprocess.run(command, check=True, capture_output=True)
    return result.stdout, command


def text(raw):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split())


commit = "5c4886908584029761b579af026dcfb627c84070"
base = "https://raw.githubusercontent.com/pytorch/pytorch/" + commit
records = []
url = "https://docs.python.org/3.13/reference/expressions.html"
raw, command = download(url)
page = raw.decode()
start = page.index('<section id="shifting-operations"')
end = page.index('<section id="comparisons"', start)
section = text(page[start:end])
shift = section[section.index("A right shift by"):section.index("Binary bitwise operations")]
and_start = section.index("The & operator")
and_end = section.index("The ^ operator", and_start)
or_start = section.index("The | operator")
records.append({
    "url": url,
    "command": command,
    "version": text(re.search(r"<title>(.*?)</title>", page, re.S).group(1)),
    "download_sha256": hashlib.sha256(raw).hexdigest(),
    "excerpts": {"Shifting operations": shift.strip(), "Binary bitwise operations, AND": section[and_start:and_end].strip(), "Binary bitwise operations, OR": section[or_start:].strip()},
})

url = base + "/torch/_torch_docs.py"
raw, command = download(url)
tree = ast.parse(raw.decode())
names = {"bitwise_left_shift", "bitwise_right_shift", "bitwise_and", "bitwise_or", "equal", "flatten", "stack"}
excerpts = {}
for node in tree.body:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        continue
    call = node.value
    if not isinstance(call.func, ast.Name) or call.func.id != "add_docstr" or len(call.args) < 2:
        continue
    target = call.args[0]
    if not isinstance(target, ast.Attribute) or target.attr not in names:
        continue
    literals = [value.value for value in ast.walk(call.args[1]) if isinstance(value, ast.Constant) and isinstance(value.value, str)]
    doc = max(literals, key=len)
    paragraphs = [part.strip() for part in doc.split("\n\n") if part.strip()]
    description = paragraphs[1]
    if target.attr == "equal":
        description = paragraphs[1]
    excerpts[target.attr] = " ".join(description.split())
if set(excerpts) != names:
    raise ValueError("缺少預期的 PyTorch API 原文")
records.append({"url": url, "command": command, "version": "PyTorch 2.14.1 git " + commit, "download_sha256": hashlib.sha256(raw).hexdigest(), "excerpts": excerpts})

url = base + "/torch/ao/quantization/fake_quantize.py"
raw, command = download(url)
tree = ast.parse(raw.decode())
node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FakeQuantize")
doc = ast.get_docstring(node)
excerpt = doc[:doc.index("* :attr:`is_dynamic`")].strip()
records.append({"url": url, "command": command, "version": "PyTorch 2.14.1 git " + commit, "download_sha256": hashlib.sha256(raw).hexdigest(), "excerpts": {"FakeQuantize output formula": excerpt}})

result = {"accessed_at_utc": datetime.now(UTC).isoformat(), "scope": "只保存支持主張的小段原文；下載全文僅在記憶體處理。", "sources": records}
output = Path(__file__).with_name("source-excerpts.json")
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
