"""Section-only visual proof, with the optional subsection expanded for inspection."""
from pathlib import Path
import markdown

HERE = Path(__file__).resolve().parent
raw = (HERE / "section.md").read_text()
render_input = raw.replace("<details>", '<details open markdown="1">')
body = markdown.markdown(render_input, extensions=["fenced_code", "tables", "md_in_html"])
html = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>15.8 frozen section review</title><style>body{font:18px/1.7 sans-serif;margin:24px auto;padding:0 20px;max-width:980px;color:#202124;background:white}pre{font:14px/1.5 monospace;padding:12px;background:#f2f4f7;overflow:auto}code{background:#f2f4f7}h2{font-size:28px}details{border:1px solid #ccc;padding:12px}a{color:#145caa}@media(max-width:500px){body{font-size:16px;margin:12px auto;padding:0 12px}h2{font-size:23px}pre{font-size:12px}}</style><body>' + body + '</body></html>'
(HERE / "section-render.html").write_text(html)
print("Created section-only render; details expanded; markdown", markdown.__version__)
