"""Save inspected original paragraphs; read only immutable PDF bytes, no reviews."""
import hashlib
import json
import subprocess
from pathlib import Path

base = Path(__file__).resolve().parent
repo = base.parents[4]
ranges = {
    "instructgpt-v1": [(1,42), (85,104), (294,320), (411,518)],
    "dpo-v3": [(1,40), (150,280)],
    "ppo-v2": [(1,49), (51,254)],
    "deepseek-r1-v1": [(1,25), (219,311), (320,360), (427,565)],
}
rows = json.loads((base / "input-provenance.json").read_text())
for row in rows:
    name = row["name"]
    original = repo / row["original_lookup_path"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == row["pdf_sha256"]
    command = ["pdftotext", "-layout", str(original), "-"]
    result = subprocess.run(command, capture_output=True, timeout=10, check=True)
    lines = result.stdout.decode("utf-8").split("\n")
    excerpt = (f"Original PDF: {row['original_lookup_path']}\nOriginal HTTPS URL: {row['url']}\n"
               f"Original PDF SHA-256: {row['pdf_sha256']}\n"
               "Extraction: pdftotext -layout original.pdf - (Poppler 25.03.0)\n"
               "L numbers refer to newline-delimited original extraction; form feeds are retained.\n")
    for start, end in ranges[name]:
        excerpt += f"\n--- Independently inspected original lines {start}-{end} ---\n"
        excerpt += "\n".join(f"L{i}: {lines[i-1]}" for i in range(start, end+1)) + "\n"
    path = base / (name + "-inspected-excerpts.txt")
    path.write_text(excerpt)
    row.update(excerpt_ranges=ranges[name], actual_extraction_argv=command, extraction_exit_code=result.returncode,
               permanent_excerpt=str(path.relative_to(repo)), excerpt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(base / "input-provenance.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
print("Four original PDF hashes verified; independently read sections retained with original newline line locators.")
