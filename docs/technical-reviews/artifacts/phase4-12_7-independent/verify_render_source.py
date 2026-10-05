from pathlib import Path
import json,hashlib
import markdown
from bs4 import BeautifulSoup
base=Path(__file__).resolve().parent
original=BeautifulSoup(markdown.markdown((base/'section.md').read_text(),extensions=['fenced_code']),'html.parser')
rendered=BeautifulSoup((base/'rendered-page.html').read_text(),'html.parser').find('article')
normalize=lambda s:' '.join(s.split())
expected=[normalize(x.get_text()) for x in original.find_all('p')]
actual=[normalize(x.get_text()) for x in rendered.find_all('p')]
positions=[]
for p in expected:
 assert p in actual,p
 positions.append(actual.index(p))
assert positions==sorted(positions)
code=rendered.find('pre').get_text()
assert code.strip()==(base/'fence-1.py').read_text().strip()
record={'section_sha256':hashlib.sha256((base/'section.md').read_bytes()).hexdigest(),'rendered_html_sha256':hashlib.sha256((base/'rendered-page.html').read_bytes()).hexdigest(),'all_original_paragraphs_exact_after_whitespace_normalization':True,'original_paragraph_count':len(expected),'paragraph_positions':positions,'fence_exact_ignoring_outer_blank_lines':True,'note':'Generated page adds navigation and Colab links; checked original content and all code against current frozen section.'}
(base/'render-source-match.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False))
