"""Check only the two explicit forward references made in frozen W.2."""

from datetime import UTC, datetime
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[4]
files = [root / f"course/chapters/{name}.md" for name in ("01", "02", "03")]
found = []
for path in files:
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("class "):
            found.append({"path": str(path.relative_to(root)), "line": number, "text": line})
assert len(found) == 1 and found[0]["text"] == "class CausalAttention(nn.Module):"


def section(text, name):
    start = text.index(f"## {name} ")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


assert found[0]["text"] in section((root / "course/chapters/03.md").read_text(), "3.7")
part = section((root / "course/chapters/17.md").read_text(), "17.8")
expressions = ["return values[0::2] | (values[1::2] << 4)", "packed & 15, packed >> 4"]
assert all(expression in part for expression in expressions)
print(json.dumps({"created_at": datetime.now(UTC).isoformat(), "python": sys.version, "device": "cpu",
                  "scope": "Scan class declarations in chapters 01-03; check linked 17.8 expressions only. No independent review of those chapters or figures.",
                  "first_class_declaration": found, "packing_expressions": expressions,
                  "source_files": [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                   for path in files + [root / "course/chapters/17.md"]], "all_passed": True},
                 ensure_ascii=False, indent=2))
