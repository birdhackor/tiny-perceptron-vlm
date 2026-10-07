"""Fetch an original document without interpreting its claims."""
import hashlib
import json
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

class Text(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.hidden=0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.hidden+=1
        if tag in ('p','div','li','h1','h2','h3','h4','pre','tr'): self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script','style'): self.hidden-=1
        if tag in ('p','div','li','h1','h2','h3','h4','pre','tr'): self.parts.append('\n')
    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)

url, name = sys.argv[1:]
folder=Path('docs/technical-reviews/artifacts/p7_technical_c/originals'); folder.mkdir(parents=True, exist_ok=True)
response=urllib.request.urlopen(url, timeout=45)
data=response.read(); path=folder/f'{name}.html'; path.write_bytes(data)
parser=Text(); parser.feed(data.decode('utf-8'))
text='\n'.join(line.strip() for line in ''.join(parser.parts).splitlines() if line.strip())
txt=folder/f'{name}.txt';txt.write_text(text+'\n')
print(json.dumps({'url':response.url,'html_path':str(path),'html_sha256':hashlib.sha256(data).hexdigest(),'text_path':str(txt),'text_sha256':hashlib.sha256(txt.read_bytes()).hexdigest()}))
