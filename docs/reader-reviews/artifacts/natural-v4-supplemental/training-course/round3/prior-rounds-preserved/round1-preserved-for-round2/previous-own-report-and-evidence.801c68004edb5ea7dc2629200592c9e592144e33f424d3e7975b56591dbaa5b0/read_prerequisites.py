from pathlib import Path
import hashlib,json,re,sys
ROOT=Path('/workspace/tiny-perceptron-vlm'); OUT=ROOT/'docs/reader-reviews/artifacts/natural-v4-supplemental/training-course'
sections=[('first-steps.md','W.1'),('chapters/01.md','1.2'),('chapters/05.md','5.1'),('chapters/01.md','1.5'),('chapters/02.md','2.2'),('chapters/01.md','1.8'),('chapters/02.md','2.3'),('chapters/04.md','4.7'),('chapters/07.md','7.1'),('chapters/07.md','7.3'),('chapters/07.md','7.11'),('chapters/07.md','7.17'),('chapters/07.md','7.18'),('chapters/08.md','8.1'),('chapters/09.md','9.1'),('chapters/10.md','10.5'),('chapters/11.md','11.1'),('chapters/12.md','12.8'),('chapters/13.md','13.1'),('chapters/13.md','13.4'),('chapters/04.md','4.5'),('chapters/15.md','15.13'),('chapters/17.md','17.2'),('chapters/17.md','17.8'),('chapters/18.md','18.1'),('chapters/18.md','18.6'),('chapters/05.md','5.15')]
receipts=[]
start,end=map(int,sys.argv[1:])
for order,(f,section) in enumerate(sections,1):
 p=ROOT/'course'/f;raw=p.read_bytes();full=raw.decode();lines=full.splitlines(keepends=True)
 beginning=next(i for i,line in enumerate(lines) if re.match(r'## '+re.escape(section)+r' ',line))
 ending=next((i for i in range(beginning+1,len(lines)) if lines[i].startswith('## ')),len(lines))
 snippet=''.join(lines[beginning:ending]);b=snippet.encode();h=hashlib.sha256(b).hexdigest()
 name='prerequisite-'+section+'.'+h+'.md'
 if not (OUT/name).exists(): (OUT/name).write_bytes(b)
 refs=re.findall(r'\(([^)]*\.svg)\)',snippet)
 receipts.append({'order':order,'source':'course/'+f,'section':section,'line_start':beginning+1,'line_end':ending,'full_source_sha256':hashlib.sha256(raw).hexdigest(),'section_sha256':h,'snapshot':name,'svg_refs':refs})
 if start<=order<=end:
  print('\nREAD ORDER '+str(order)+' '+f+' '+section+' LINES '+str(beginning+1)+'–'+str(ending))
  print(''.join(f'{i+1}: {lines[i]}' for i in range(beginning,ending)),end='')
(OUT/'prerequisite-read-receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
