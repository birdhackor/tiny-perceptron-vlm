"""Run a small, explicitly selected set of unchanged chapter code fences."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
chosen = ["1.8", "2.4", "3.4", "3.6", "4.6", "4.8"]
records = []
for section_id in chosen:
    chapter = ROOT / f"course/chapters/{int(section_id.split('.')[0]):02d}.md"
    text = chapter.read_text()
    headings = list(re.finditer(r"^## ([0-9]+\.[0-9]+) [^\n]+\n", text, re.M))
    index = next(i for i, m in enumerate(headings) if m[1] == section_id)
    end = headings[index+1].start() if index+1<len(headings) else len(text)
    section = text[headings[index].start():end]
    fences = re.findall(r"```python\n([\s\S]*?)\n```", section)
    code = "\n\n".join(fences) + "\n"
    path = OUT / "numeric" / (section_id + ".py")
    path.parent.mkdir(exist_ok=True)
    path.write_text(code)
    started = datetime.now(timezone.utc).isoformat()
    argv = [str(ROOT / ".venv/bin/python"), str(path)]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    stdout = path.with_suffix(".stdout.txt")
    stderr = path.with_suffix(".stderr.txt")
    stdout.write_text(result.stdout)
    stderr.write_text(result.stderr)
    records.append({"section_id": section_id, "argv": argv, "cwd": str(ROOT),
                    "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(),
                    "exit_code": result.returncode, "source": str(chapter.relative_to(ROOT)),
                    "code_fences_executed": len(fences), "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                    "extracted_code": str(path.relative_to(ROOT)),
                    "stdout": str(stdout.relative_to(ROOT)), "stderr": str(stderr.relative_to(ROOT)),
                    "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest()})
    print(section_id, "exit",result.returncode)
    print(result.stdout, end="")
    if result.stderr: print(result.stderr, file=sys.stderr, end="")
(OUT / "numeric-execution.json").write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n")
raise SystemExit(0 if all(r["exit_code"] == 0 for r in records) else 1)
