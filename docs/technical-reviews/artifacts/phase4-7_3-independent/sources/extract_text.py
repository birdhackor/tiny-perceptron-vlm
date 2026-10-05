"""Plain readable text from downloaded original HTML; preserve raw snapshots separately."""
from html.parser import HTMLParser
from pathlib import Path
import re

class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.skip=0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}: self.skip += 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "h5", "pre", "dt", "dd", "br", "div"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script", "style"}: self.skip -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "h5", "pre", "dt", "dd", "div"}: self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)

for source in sorted(Path(__file__).parent.glob("*.html")):
    parser=PlainText(); parser.feed(source.read_text())
    lines=[re.sub(r"[ \t]+", " ", line).strip() for line in "".join(parser.parts).splitlines()]
    text="\n".join(line for line in lines if line) + "\n"
    source.with_suffix(".txt").write_text(text)
    print(source.name, '->', source.with_suffix('.txt').name, len(text))
