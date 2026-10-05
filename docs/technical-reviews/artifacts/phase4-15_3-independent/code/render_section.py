"""Render only the frozen target section for factual visual inspection.

This is a standalone inspection layout, not a production course-page screenshot.
"""
from pathlib import Path
import html
import json
import re
import subprocess
import tempfile

BASE = Path(__file__).resolve().parents[1]
source = (BASE / "inputs/15.3.md").read_text()

def inline(text):
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

elements = []
code = None
for block in source.split("\n\n"):
    if block.startswith("```python"):
        # The fence contains internal blank lines; handled by the state machine below.
        break
paragraph = []
fence = []
inside_fence = False
def flush():
    if paragraph:
        elements.append("<p>" + inline("\n".join(paragraph)) + "</p>")
        paragraph.clear()
for line in source.splitlines():
    if line.startswith("```"):
        flush()
        if inside_fence:
            elements.append('<pre id="code"><code>' + html.escape("\n".join(fence)) + '</code></pre>')
            fence.clear()
        inside_fence = not inside_fence
    elif inside_fence:
        fence.append(line)
    elif not line:
        flush()
    elif line.startswith("## "):
        flush()
        elements.append("<h2>" + html.escape(line[3:]) + "</h2>")
    elif line.startswith("<details>"):
        flush()
        elements.append('<details open id="report">')
    elif line.startswith("<summary>") or line.startswith("</details>"):
        flush()
        elements.append(line)
    else:
        paragraph.append(line)
flush()
assert not inside_fence
document = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>15.3 Frozen factual inspection</title><style>
body{font-family:sans-serif;line-height:1.65;color:#151515;background:#fff;margin:0;padding:24px}
main{max-width:880px;margin:auto}h2{font-size:26px}p{font-size:17px;margin:16px 0}
code{font-family:monospace;font-size:.91em;background:#f2f4f7;padding:1px 3px}
pre{overflow:auto;background:#f2f4f7;padding:16px;line-height:1.5;font-size:14px}
pre code{padding:0}a{color:#1555ad}details{padding:12px;border:1px solid #ccc}
summary{font-weight:bold}@media(max-width:600px){body{padding:14px}p{font-size:16px}h2{font-size:23px}}
</style><main>''' + "\n".join(elements) + "</main></html>"
target = BASE / "renders/15.3-inspection.html"
target.write_text(document)
records = []
for label, size, anchor in [("desktop-start", "1280,800", ""), ("mobile-start", "390,844", ""), ("desktop-code", "1280,800", "#code"), ("mobile-report", "390,844", "#report")]:
    profile = tempfile.mkdtemp(prefix="factual-15-3-chromium-")
    command = ["/usr/bin/chromium", "--headless", "--no-sandbox", "--single-process", "--no-zygote", "--disable-crash-reporter", "--disable-gpu", "--disable-dev-shm-usage", "--user-data-dir=" + profile, "--virtual-time-budget=1000", "--hide-scrollbars", "--no-first-run", "--disable-background-networking", "--disable-extensions", "--font-render-hinting=none", "--force-device-scale-factor=1", "--window-size=" + size, "--screenshot=" + str(BASE / "renders" / (label + ".png")), target.as_uri() + anchor]
    (BASE / "renders" / (label + ".command.json")).write_text(json.dumps(command, indent=2) + "\n")
    try:
        result = subprocess.run(command, capture_output=True, timeout=35, check=False)
    except subprocess.TimeoutExpired as error:
        (BASE / "renders" / (label + ".stdout.txt")).write_bytes(error.stdout or b"")
        (BASE / "renders" / (label + ".stderr.txt")).write_bytes(error.stderr or b"")
        records.append({"label": label, "command_argv": command, "exit_code": 124, "timed_out": True, "screenshot_created": (BASE / "renders" / (label + ".png")).is_file()})
        (BASE / "renders/render-execution.json").write_text(json.dumps(records, indent=2) + "\n")
        raise
    (BASE / "renders" / (label + ".stdout.txt")).write_bytes(result.stdout)
    (BASE / "renders" / (label + ".stderr.txt")).write_bytes(result.stderr)
    record = {"label": label, "command_argv": command, "exit_code": result.returncode, "screenshot_created": (BASE / "renders" / (label + ".png")).is_file()}
    records.append(record)
    assert result.returncode == 0 and record["screenshot_created"], record
(BASE / "renders/render-execution.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
