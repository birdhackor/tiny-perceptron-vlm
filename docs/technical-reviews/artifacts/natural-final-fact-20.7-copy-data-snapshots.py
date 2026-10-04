from pathlib import Path
from datetime import datetime, UTC
import hashlib,json,tarfile,platform
P='docs/technical-reviews/artifacts/natural-final-fact-20.7-'
root=Path.cwd();dest=Path(P+'data-source-snapshots');dest.mkdir(parents=True,exist_ok=True)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def file_sha(path):return sha(Path(path).read_bytes())
report_path=Path('docs/technical-reviews/20.7.json');old_bytes=report_path.read_bytes();old=json.loads(old_bytes)
assert old['verdict']=='pass' and old['source_sha256']=='d8c8a088c7498248d0d133e661898006876d1f7241ec1813e461b56a5dadebd2'
prior_path=Path(P+'pass-before-durable-speech-metadata.json')
if prior_path.exists():assert prior_path.read_bytes()==old_bytes
else:prior_path.write_bytes(old_bytes)
original=Path('data/natural/speech/manifest.json').read_bytes();m=json.loads(original);source=m['sources'][0]
assert source['dataset']=='google/fleurs' and source['config']=='cmn_hans_cn' and source['revision']=='d7c758a6dceecd54a98cac43404d3d576e721f07'
assert source['license']=='CC-BY-4.0' and source['license_url']=='https://creativecommons.org/licenses/by/4.0/'
combined=json.loads(Path('docs/natural-assistant/manifest.json').read_bytes());packages={x['path']:{'sha256':file_sha(x['path']),'bytes':Path(x['path']).stat().st_size,'declared_sha256':x['sha256']} for x in combined['archives']}
assert len(packages)==3 and all(v['sha256']==v['declared_sha256'] for v in packages.values())
archive=next(x for x in combined['archives'] if x['path']=='assets/training/natural-speech-v3.tar.gz');entries={x['path']:x for x in archive['files']}
mapping=[('a_speech_manifest','manifest.json'),('a_tsv_test','official-sources/test.tsv'),('a_tsv_train','official-sources/train.tsv'),('a_tsv_dev','official-sources/dev.tsv')];records=[]
with tarfile.open(archive['path'],'r:gz') as tar:
 for aid,relative in mapping:
  before_path=Path('data/natural/speech')/relative;raw=before_path.read_bytes();registered=next(x for x in old['artifacts'] if x['id']==aid)
  assert registered['path']==str(before_path) and sha(raw)==registered['sha256']
  member='speech/'+relative;assert entries[member]['sha256']==sha(raw) and entries[member]['bytes']==len(raw)
  assert tar.extractfile(member).read()==raw  # Metadata only; no WAV or weight extraction.
  dst=dest/relative;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert dst.read_bytes()==raw
  else:dst.write_bytes(raw)
  assert dst.read_bytes()==raw and before_path.read_bytes()==raw and file_sha(dst)==registered['sha256']
  if relative.endswith('.tsv'):
   split=Path(relative).stem;pin=source['files'][split]['transcripts'];assert pin['sha256']==sha(raw) and pin['bytes']==len(raw)
   assert source['revision'] in pin['url']
  else:pin={'source':'Original local selected-data manifest prepared from the pinned public FLEURS files','sha256':sha(raw),'bytes':len(raw)}
  records.append({'artifact_id':aid,'previous_live_path':str(before_path),'durable_snapshot_path':str(dst),'bytes':len(raw),'before_sha256':sha(raw),'after_sha256':file_sha(dst),'byte_exact':True,'original_file_unchanged':True,'official_source':pin,'existing_training_archive':archive['path'],'archive_member':member,'archive_member_sha256':entries[member]['sha256'],'archive_member_bytes_equal':True})
# Re-read every old and new snapshot: same original references and official TSV semantics, not other similarly-named data.
meta={}
for split in ['train','dev','test']:
 meta[split]={x[1]:x for x in [line.split('\t') for line in (dest/'official-sources'/f'{split}.tsv').read_text().splitlines()]}
rows=m['audio_rows'];assert len(rows)==42
for row in rows:
 t=meta[row['source_split']][Path(row['audio']).name];assert row['user']==row['raw_transcription']==t[2] and row['normalized_transcription']==t[3] and row['num_samples']==int(t[5]) and row['answer'] is None
actual_manifest=json.loads((dest/'manifest.json').read_bytes());assert actual_manifest==m
assert all(file_sha(path)==v['sha256'] and Path(path).stat().st_size==v['bytes'] for path,v in packages.items())
card=Path(P+'fleurs-card.md');assert file_sha(card)==source['card_sha256'];assert 'cc-by-4.0' in card.read_text()
receipt={'reviewer_task':'/root/natural_factual_final_20_7','recorded_at':datetime.now(UTC).isoformat(),'command':'.venv/bin/python '+P+'copy-data-snapshots.py','environment':{'python':platform.python_version(),'device':'CPU byte copying, SHA256 and metadata inspection; no model or audio playback'},'reason':'Remove four current technical-review evidence dependencies on ignored data/ files so the report can be validated from published repository evidence.','preserved_prior_pass_report':{'path':str(prior_path),'sha256':sha(old_bytes),'verdict':old['verdict'],'lesson_source_sha256':old['source_sha256'],'scope_preserved':True,'scope_note':'Historical full pass report retained byte-exact; its old live-data paths describe the previous review delivery, not current delivery dependencies.'},'mapping':records,'copied_metadata_files':4,'copied_metadata_bytes':sum(x['bytes'] for x in records),'reference_rows_rechecked':42,'license':{'name':source['license'],'url':source['license_url'],'attribution':source['attribution'],'official_card_url':source['card_url'],'official_card_snapshot_path':str(card),'official_card_sha256':file_sha(card),'notice':'The three official TSV files and local selection manifest are byte-exact public-source metadata snapshots for technical review. No source reference was rewritten. The local manifest records a small selection from the official corpus. This is not an additional training-data bundle.'},'provenance':{'dataset':source['dataset'],'config':source['config'],'revision':source['revision'],'selection':m['selection'],'combined_manifest_path':'docs/natural-assistant/manifest.json','combined_manifest_sha256':file_sha('docs/natural-assistant/manifest.json'),'existing_three_training_packages_unchanged':packages},'scope':{'lesson_body_changed':False,'original_reference_or_log_changed':False,'WAV_copied':False,'weights_copied':False,'model_or_ASR_rerun':False,'audio_played':False,'training_or_GPU':False,'new_training_bundle':False,'previous_claim_verdicts_unchanged':True,'prior_context_limit_unchanged':'No full serialized original inference prompts; shared context remains supported by frozen manifest, recorded runtime source and raw input fields.'}}
Path(P+'copy-data-snapshots.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'copied_metadata_files':4,'copied_metadata_bytes':receipt['copied_metadata_bytes'],'source_revision':source['revision'],'license':source['license'],'old_report_sha256':sha(old_bytes),'mapping':records,'three_packages_unchanged':packages,'reference_rows_rechecked':42},ensure_ascii=False,indent=2))
