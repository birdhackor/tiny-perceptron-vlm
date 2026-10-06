from pathlib import Path
import ast,collections,hashlib,json,sys,tarfile
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.selftrained.evaluate import normalize,score_reply
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=ROOT/'scripts/selftrained/evaluate.py';tree=ast.parse(p.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='normalize')
strip_chars=next(x.args[0].value for x in ast.walk(fn) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='strip')
card=ROOT/'docs/selftrained/model-cards/repository-README.md'
rule_line=next(line for line in card.read_text().splitlines() if line.startswith('文字的字串比對與OCR的整段、CER評分'))
documented_chars=rule_line.split('`')[1]
assert documented_chars==strip_chars
normal=[]
for value in ['大 小！',' \t\n「大小」?!。 ',chr(92)+'大小'+chr(92),'大。小','大,小']:
 record={'task':'ocr','messages':[{'role':'user','content':'讀字'},{'role':'assistant','content':'大小'}],'supervision':{'ocr_text':'大小'}}
 normal.append({'input':value,'normalized':normalize(value),'score':score_reply(record,value,{'tool_call':None,'status':'no_tool','executed':False,'tool_result':None})})
manifest=ROOT/'docs/selftrained/v2-manifest.json';d=json.loads(manifest.read_text())
record_names={x['path'] for x in d['records']};asset_names={x['path'] for x in d['assets']}
archive=ROOT/d['package']['path'];members={};suffixes=collections.Counter();asset_suffixes=collections.Counter()
with tarfile.open(archive,'r:gz') as t:
 for m in t:
  if not m.isfile():continue
  members[m.name]=m.size;suffixes[Path(m.name).suffix]+=1
  if m.name in asset_names:asset_suffixes[Path(m.name).suffix]+=1
assert set(members)==record_names|asset_names
assert len(record_names)==12 and len(asset_names)==8950 and len(members)==8962
assert sum(members.values())==d['package']['unpacked_bytes']==127161811
assert sha(archive)==d['package']['sha256'] and archive.stat().st_size==d['package']['bytes']
result={'command':'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-public-b/recheck/recheck-probe.py','environment':{'python':sys.version,'device':'cpu'},'normalize':{'source_sha256':sha(p),'locator':f'normalize lines{fn.lineno}-{fn.end_lineno}; score_reply lines239-306','actual_strip_character_set':strip_chars,'strip_character_codepoints':[ord(x) for x in strip_chars],'backslash_in_strip_set':'\\' in strip_chars,'documented_strip_character_set':documented_chars,'documented_strip_codepoints':[ord(x) for x in documented_chars],'documented_equals_actual':documented_chars==strip_chars,'current_card_sha256':sha(card),'cases':normal},'package':{'manifest_sha256':sha(manifest),'manifest_topkeys':{k:type(v).__name__ for k,v in d.items()},'checked_pointers':['/package','/records/*/path','/assets/*/path'],'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'unpacked_bytes':sum(members.values()),'record_declarations':len(record_names),'other_declarations':len(asset_names),'actual_files':len(members),'record_jsonl_files':sorted(record_names),'other_jsonl_files':sorted(x for x in asset_names if x.endswith('.jsonl')),'actual_suffix_counts':dict(suffixes),'other_file_suffix_counts':dict(asset_suffixes)}}
(OUT/'probe-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'strip_chars':strip_chars,'backslash_in_strip_set':'\\' in strip_chars,'all_jsonl':suffixes['.jsonl'],'other_jsonl':result['package']['other_jsonl_files'],'other_suffixes':dict(asset_suffixes)},ensure_ascii=False))
