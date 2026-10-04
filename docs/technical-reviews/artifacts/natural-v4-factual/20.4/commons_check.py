"""Fresh bounded official Commons metadata audit; no images or models fetched."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import platform
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[5]
RAW = ROOT / 'outputs/natural-v4/factual-research/20.4'
ART = ROOT / 'docs/technical-reviews/artifacts/natural-v4-factual/20.4'
sources = json.loads((ROOT/'docs/natural-assistant/v4/data/ocr-sources.json').read_text())
rows = [r for r in sources['sources'] if r.get('source_page')]

def fetch(group):
    response = requests.get('https://commons.wikimedia.org/w/api.php',params={
        'action':'query','format':'json',
        'pageids':'|'.join(str(r['source_page_id']) for r in group),
        'prop':'imageinfo','iiprop':'extmetadata|url|sha1','iiextmetadatalanguage':'en'},
        headers={'User-Agent':'IndependentEducationalSourceAudit/1.0'},timeout=35)
    response.raise_for_status()
    document = response.json()
    (RAW/('commons-final-'+str(group[0]['source_page_id'])+'.json')).write_bytes(response.content)
    receipt = {'url':response.url,'status':response.status_code,'bytes':len(response.content),
               'sha256':hashlib.sha256(response.content).hexdigest()}
    notes = []
    for row in group:
        info = document['query']['pages'][str(row['source_page_id'])]['imageinfo'][0]
        meta = info['extmetadata']
        license = meta.get('LicenseShortName',{}).get('value','')
        artist = BeautifulSoup(meta.get('Artist',{}).get('value',''),'html.parser').get_text(' ',strip=True)
        original_artist = BeautifulSoup(row['artist_html'],'html.parser').get_text(' ',strip=True)
        note = {'source_id':row['source_id'],'page':row['source_page'],
            'original_page_id':row['source_page_id'],'license':license,'author':artist,
            'matches_retained_license':license==row['license'],
            'matches_retained_author':artist==original_artist,
            'matches_retained_original_sha1':info['sha1']==row['original_sha1']}
        assert all(note[k] for k in ['matches_retained_license','matches_retained_author','matches_retained_original_sha1']),note
        notes.append(note)
    return receipt,notes

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(fetch,[rows[:50],rows[50:]]))
out = {'retrievals':[r[0] for r in results],
       'checks':[n for r in results for n in r[1]],
       'scope':'Fresh official Commons imageinfo checks of all70 selected original pages; confirms retained license, credited artist and original-file SHA1. Does not re-transcribe70 photographs.',
       'environment':{'python':platform.python_version(),'requests':requests.__version__,'device':'cpu','network':'HTTPS official Commons API; two bounded queries'}}
assert len(out['checks']) == 70
(ART/'commons-license-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'original_pages':70,'license_author_sha1_matches':70,'official_queries':2,'result':'all assertions passed'},ensure_ascii=False))
