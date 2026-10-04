"""Independent CPU-only raw OCR review. Does not read score/result/review counts."""
import base64, csv, hashlib, io, json, platform, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/technical-reviews/artifacts/natural-final-fact-20.6-recheck-audit.json'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads((ROOT/path).read_text())
def distance(a,b):
    # Wagner-Fischer recurrence written here; unit-cost Unicode codepoints.
    prior = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):
            row.append(min(row[j-1]+1,prior[j]+1,prior[j-1]+int(x!=y)))
        prior=row
    return prior[-1]
def norm(s,remove):
    t=unicodedata.normalize('NFKC',s)
    return ''.join(c for c in t if not c.isspace()) if remove else t

body=(ROOT/'course/chapters/20.md').read_text()
match=re.search(r'^## 20\.6 .*?(?=^## |\Z)',body,re.M|re.S)
lesson=match[0]
output={'environment':{'python':sys.version,'platform':platform.platform(),'device':'CPU','unicode_database':unicodedata.unidata_version},'lesson_sha256':hashlib.sha256(lesson.encode()).hexdigest(),'inputs':{},'examples':[],'raw_records':{},'summaries':{},'source_checks':{}}
(ROOT/'docs/technical-reviews/artifacts/natural-final-fact-20.6-recheck-section.md').write_text(lesson)
fence=re.search(r'```python\n(.*?)```',lesson,re.S)[1]
for predicted in ['麵包\n牛奶','牛奶\n麵包','牛奶麵包']:
    print('EXECUTE EXACT LESSON FENCE predicted=',repr(predicted))
    code=fence.replace('predicted = "麵包\\n牛奶"','predicted = '+repr(predicted))
    capture=io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(capture):exec(compile(code,'20.6-independent-fence','exec'),{})
    print(capture.getvalue(),end='')
    output['examples'].append({'prediction':predicted,'stdout':capture.getvalue(),'distance':distance('牛奶\n麵包',predicted)})
output['nfkc_examples']={s:norm(s,False) for s in ['Ａ','A','臺','台']}
print('NFKC',output['nfkc_examples'])
manifest=read('docs/natural-assistant/manifest.json')
rows={r['id']:r for r in manifest['rows']}
for group in ['final','validation','external-ocr']:
    path=Path('docs/natural-assistant/evidence')/group/'generations.json'
    data=read(path)
    output['inputs'][str(path)]=sha(ROOT/path)
    data=[r for r in data if r['task'] in ['ocr','text_presence']]
    cleaned=[]
    for r in data:
        r={k:v for k,v in r.items() if k!='score'}
        ids=r['generated_token_ids']
        eos=bool(ids and ids[-1] in r['eos_token_ids'])
        reached=len(ids)>=384
        assert len(ids)==r['generated_tokens']
        assert eos==r['ended_with_eos']
        assert (reached and not eos)==r['truncated']
        assert (not reached and not eos)==r['completion_unknown']
        assert ('eos' if eos else 'max_new_tokens' if reached else 'unknown')==r['stop_reason']
        assert reached==r['reached_max_new_tokens']
        ref,pred=r['reference_answer'],r['prediction']
        if group!='external-ocr':
            assert ref==rows[r['id']]['answer']
            remove=rows[r['id']]['references'].get('strip_whitespace',True)
        else: remove=True
        nr,np=norm(ref,remove),norm(pred,remove)
        ar,ap=norm(ref,True),norm(pred,True)
        r['independent']={'raw_reference_characters':len(ref),'raw_prediction_characters':len(pred),'raw_exact':ref==pred,'raw_distance':distance(ref,pred),'declared_reference_characters':len(nr),'declared_exact':nr==np,'declared_distance':distance(nr,np),'all_whitespace_removed_reference_characters':len(ar),'all_whitespace_removed_exact':ar==ap,'all_whitespace_removed_distance':distance(ar,ap),'category':'presence' if r['task']=='text_presence' else 'order' if '\n' in ref else 'single','complete':eos and not r['truncated'] and not r['completion_unknown']}
        cleaned.append(r)
    output['raw_records'][group]=cleaned
    totals={}
    for variant in sorted({r['variant'] for r in cleaned}):
        categories={}
        for category in ['single','order','ocr','presence','all']:
            selected=[r for r in cleaned if r['variant']==variant and (category=='all' or (category=='ocr' and r['task']=='ocr') or r['independent']['category']==category)]
            categories[category]={'samples':len(selected),'eos':sum(r['independent']['complete'] for r in selected),'truncated':sum(r['truncated'] for r in selected),'reference_newlines':sum(r['reference_answer'].count('\n') for r in selected)}
            for mode in ['raw','declared','all_whitespace_removed']:
                categories[category][mode]={'exact':sum(r['independent'][mode+'_exact'] and r['independent']['complete'] for r in selected),'distance':sum(r['independent'][mode+'_distance'] for r in selected),'characters':sum(r['independent'][mode+'_reference_characters'] for r in selected)}
        totals[variant]=categories
    output['summaries'][group]=totals
    print(group,json.dumps(totals,ensure_ascii=False))

# Raw source manifest, every render font, grouped holdout and complete file hashes.
ocr_path=ROOT/'outputs/natural-extension/data/ocr/manifest.json'
ocr=json.loads(ocr_path.read_text())
output['inputs'][str(ocr_path.relative_to(ROOT))]=sha(ocr_path)
assignment={};counts=Counter();fonts=defaultdict(Counter)
for r in ocr['rows']:
    for key in ['family','image']:
        k=(key,r[key]);assert assignment.setdefault(k,r['split'])==r['split']
    counts[(r['split'],r['task'],r['references']['kind'])]+=1
for im in ocr['images']:
    assert sha(ocr_path.parent/im['image'])==im['sha256']
    fonts[im['image'].split('-')[0]][str(im['font_source'])]+=1
for name,expected in [('NotoSansCJKtc-Regular.otf','dce08bd4fd91aa8aa76ed8fea4b694c2dfb8550f67871e326843212ddbeb88b4'),('NotoSerifCJKtc-Regular.otf','234301038e76e7c35c43113785024700c4e4fe7bdce1d1fbbc42fca7e6683798')]:
    p=ROOT/'outputs/natural-extension/font-cache'/name;assert sha(p)==expected
    output['inputs'][str(p.relative_to(ROOT))]=sha(p)
output['source_checks']['synthetic']={'counts':{str(k):v for k,v in counts.items()},'fonts_by_split':{k:dict(v) for k,v in fonts.items()},'all_image_files_verified':len(ocr['images']),'group_split_consistent':True}

# Full pinned TSV hashes and independently selected source rows decoded to originals.
ext=read('docs/natural-assistant/external-ocr.json')
csv.field_size_limit(2**31-1)
source_rows=[]
for source in ext['sources']:
    p=ROOT/'outputs/natural-extension/external-ocr-cache'/source['cache_file']
    assert p.stat().st_size==source['bytes'] and sha(p)==source['sha256']
    output['inputs'][str(p.relative_to(ROOT))]=sha(p)
    with p.open(newline='') as f:
        upstream=list(csv.DictReader(f,delimiter='\t'))
    for r in ext['rows']:
        note=r['source']
        if note['id']!=source['id']:continue
        u=upstream[note['row_index']]
        raw=base64.b64decode(u['image']);gt=u['answer']
        img=ROOT/'outputs/natural-extension/external-ocr'/r['image']
        assert raw==img.read_bytes()
        assert gt==r['answer']
        assert u['image_name']==note['image_name']
        for pred in output['raw_records']['external-ocr']:
            if pred['id']==r['id']:assert pred['reference_answer']==gt
        source_rows.append({'id':r['id'],'source_url':source['url'],'source_index':note['row_index'],'original_gt':gt,'raw_characters':len(gt),'nfkc_no_whitespace_characters':len(norm(gt,True)),'image_name':u['image_name'],'image_sha256':sha(img),'original_gt_sha256':hashlib.sha256(gt.encode()).hexdigest()})
output['source_checks']['external_fixed_tsv']=source_rows
OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print('WROTE',str(OUT.relative_to(ROOT)),'sha256',sha(OUT))
