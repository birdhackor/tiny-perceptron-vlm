import pathlib,hashlib
for label,rows in glyphs.items():print('glyph pixel sum',label,sum(int(c) for row in rows for c in row))
variant=glyphs['1'].copy();variant[0]='110';print('variant1 same label, pixels',sum(int(c) for row in variant for c in row))
p=pathlib.Path('docs/course-experiments/results/ocr.json');raw=p.read_bytes();pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/ocr-raw.json').write_bytes(raw);print('rawsha',hashlib.sha256(raw).hexdigest());r=json.loads(raw)['results'];splits=r['data']['splits'];families={n:{x['family'] for x in v['records']} for n,v in splits.items()}
print('families disjoint',all(not(families[a]&families[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]))
print('train characters',sorted({c for x in splits['train']['records'] for c in x['answer']}))
print('split_counts',[(n,v['count'],len(v['records']),len(families[n])) for n,v in splits.items()])
s=r['test']['samples'];print('exact recount',sum(x['generated']==x['target'] for x in s),len(s),'scope',r['scope'])
