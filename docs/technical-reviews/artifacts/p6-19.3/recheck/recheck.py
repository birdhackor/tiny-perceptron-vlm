"""Original reviewer's bounded reinspection after the wording-only 19.3 repair."""
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import hashlib,json,re,platform
ROOT=Path(__file__).resolve().parents[5]
ART=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def emit(tag,value):print(tag,json.dumps(value,ensure_ascii=False,sort_keys=True))
old=load(OUT/'initial-19.3-report.json')
assert old['reviewer_task']=='/root/p6_fact_19_3' and old['verdict']=='revise'
previous=(ART/'section-19.3.md').read_bytes()
raw=(ROOT/'course/chapters/19.md').read_bytes();text=raw.decode('utf-8');headings=list(re.finditer(r'^## .+$',text,re.M))
index,head=next((i,h) for i,h in enumerate(headings) if h.group().startswith('## 19.3 '))
end=headings[index+1].start() if index+1<len(headings) else len(text)
current=text[head.start():end].encode('utf-8')
before='| 字卡 | 2–4字組合按字串隔離；同一畫布的區域問題及訓練增強版本不跨份 |'.encode()
after='| 字卡 | 2–4字組合按字串隔離；同一來源家族的區域問題及訓練增強版本留在同一份 |'.encode()
assert previous.count(before)==1 and current==previous.replace(before,after)
assert current==(OUT/'current-section-19.3.md').read_bytes()
emit('section_delta',{'previous_sha256':hashlib.sha256(previous).hexdigest(),'current_sha256':hashlib.sha256(current).hexdigest(),'only_change':'字卡表的同一画布→同一来源家族，保留2–4字字串隔离及既知字/两字体/ROI范围','full_current_section_read':True})
checked=[]
for source in old['sources']:
 if source['kind']=='repository_code':
  assert sha(ROOT/source['path'])==source['sha256'];checked.append({'kind':'source','path':source['path'],'sha256':source['sha256']})
for artifact in old['artifacts']:
 assert sha(ROOT/artifact['path'])==artifact['sha256'];checked.append({'kind':'artifact','path':artifact['path'],'sha256':artifact['sha256']})
for path,digest in old['figure_sha256'].items():
 assert sha(ROOT/path)==digest;checked.append({'kind':'figure','path':path,'sha256':digest})
emit('unchanged_previous_evidence',checked)
manifest_path=ROOT/'docs/selftrained/v2-manifest.json';manifest=load(manifest_path)
original_manifest=json.loads(next(line[len('manifest '):] for line in (ART/'verification-output.txt').read_text().splitlines() if line.startswith('manifest ')))
assert sha(manifest_path)==original_manifest['sha256']
archive=ROOT/manifest['package']['path']
assert sha(archive)==manifest['package']['sha256'] and archive.stat().st_size==manifest['package']['bytes']
emit('fixed_package_unchanged',{'manifest_sha256':sha(manifest_path),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'prior_all_8962_member_checks_reused_after_identical_package_hash':True})
data=ROOT/'outputs/selftrained-v2/data';all_rows=[];record_hashes=[]
for item in manifest['records']:
 path=data/item['path'];assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes']
 all_rows.extend(json.loads(line) for line in path.read_bytes().splitlines())
 record_hashes.append({'path':item['path'],'sha256':item['sha256'],'bytes':item['bytes']})
emit('current_record_files_unchanged',record_hashes)
base=load(ROOT/'docs/selftrained/manifest.json')
entry=next(e for e in base['records'] if e['path']=='ocr-train.jsonl')
raw_train=(data/'ocr-train.jsonl').read_bytes();original_prefix=raw_train[:entry['bytes']]
assert hashlib.sha256(original_prefix).hexdigest()==entry['sha256']
parents=[json.loads(line) for line in original_prefix.splitlines()]
children=[json.loads(line) for line in raw_train[entry['bytes']:].splitlines()]
family_records=defaultdict(set)
for p in parents:
 assert p['split']=='train'
 family_records[p['group_id']].add((p['supervision']['ocr_text'],tuple(p['roi'])))
for c in children:
 assert c['split']=='train' and c['group_id'] in family_records
 assert (c['supervision']['ocr_text'],tuple(c['roi'])) in family_records[c['group_id']]
groups=defaultdict(set)
for r in all_rows:groups[r['split']].add(r['group_id'])
intersections={f'{a}/{b}':len(groups[a]&groups[b]) for a,b in combinations(['train','validation','test'],2)}
assert not any(intersections.values())
emit('family_contract_recheck',{'base_ocr_train_records':len(parents),'augmented_ocr_train_records':len(children),'every_augmented_child_has_train_parent_family_and_same_target_roi':True,'group_id_cross_split':intersections,'claim_verified':'同一来源家族的区域问题及训练增强版本留在同一份','not_inferred':'all rendered PNG/pixels across all independent single-character families are disjoint'})
collision=load(ART/'ocr-pixel-collision.json')[0]
assert sha(ART/'ocr-pixel-collision-0.png')==sha(ART/'ocr-pixel-collision-1.png')==collision['records'][0]['image_file_sha256']==collision['records'][1]['image_file_sha256']
assert collision['records'][0]['group_id']!=collision['records'][1]['group_id']
emit('collision_preserved',{'full_png_sha256':collision['records'][0]['image_file_sha256'],'target':'左','distinct_source_families':True,'train_validation_collision_exists':True,'collision_not_erased_or_relabelled':'The current manuscript restricts its guarantee to source-family lineage rather than every single-character PNG pixel; original known-character and font limitations remain.'})
emit('environment',{'python':platform.python_version(),'device':'cpu','model_training_or_generation_or_heldout_eval':'none'})
emit('recheck_conclusion',{'verdict':'pass','issue':'c8','resolution':'完整重读当前19.3后，新表述明确限定同一来源家族，初始同PNG反例不再与此承诺矛盾；原像素碰撞继续保留为资料范围限制。'})
