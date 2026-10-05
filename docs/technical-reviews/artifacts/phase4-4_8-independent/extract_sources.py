from pathlib import Path
import json
import subprocess
from bs4 import BeautifulSoup

OUT = Path(__file__).resolve().parent
command = ['pdftotext', '-layout', str(OUT/'sources/vaswani-1706.03762v7.pdf'), str(OUT/'sources/vaswani-1706.03762v7.txt')]
completed = subprocess.run(command,capture_output=True,text=True,check=True)
for chapter,name in [(5,'ml'),(6,'mlp')]:
    html=OUT / f'sources/deeplearning-chapter{chapter}-{name}.html'
    soup=BeautifulSoup(html.read_bytes(),'html.parser')
    text='\n\n'.join(f"HTML page {page.get('data-page-no')}\n"+page.get_text(' ',strip=True) for page in soup.find_all('div',class_='pf'))
    (html.with_suffix('.txt')).write_text(text+'\n')
receipt={'paper_conversion_command':command,'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,'bs4':'4.15.0','text_scope':'Author HTML text layer divided by original data-page-no; unmodified HTML snapshot remains available.'}
(OUT/'source-conversion.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
