from pathlib import Path
import hashlib,json,datetime,platform,sys,re,shutil
root=Path(__file__).resolve().parents[4];dir=Path(__file__).resolve().parent;path=root/'course/training.md';raw=path.read_bytes();sha=lambda b:hashlib.sha256(b).hexdigest()
heads=list(re.finditer(rb'(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+',raw));i=next(i for i,h in enumerate(heads) if h[1]==b'T.9');start,end=heads[i].start(),heads[i+1].start();before=raw[start:end]
assert sha(before)=='51a362b51eed67671a188f247031838fd3651e38ca1d58a4ae6ee0fc3d82e0d9'
report=root/'docs/technical-reviews/T.9.json';report_raw=report.read_bytes();assert sha(report_raw)=='e91a5dbdfa9fb660ead13d11427553b8f1abe756552a26df75238b61fb0c03a0' and json.loads(report_raw)['verdict']=='revise'
archive=root/'docs/technical-reviews/history/phase4-T_9-initial-revise-20261005.opaque.json';assert not archive.exists();shutil.copyfile(report,archive)
old='設定選完才把相同三條命令的資料路徑改為`test.jsonl`、另取輸出檔名，最後題不拿來重新選設定。'
new='設定選完才把相同三條命令的資料路徑改為`test.jsonl`，加上`--split-label test`並另取輸出檔名，最後題不拿來重新選設定。'
assert before.count(old.encode())==1;after=before.replace(old.encode(),new.encode());assert re.findall(rb'```[^\n]*\n.*?```',before,re.S)==re.findall(rb'```[^\n]*\n.*?```',after,re.S)
(dir/'before-T.9.md').write_bytes(before);(dir/'after-T.9.md').write_bytes(after);updated=raw[:start]+after+raw[end:];path.write_bytes(updated);assert path.read_bytes()==updated
print(json.dumps({'executed_at':datetime.datetime.now(datetime.UTC).isoformat(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'source_path':'course/training.md#T.9','before_source_sha256':sha(before),'after_source_sha256':sha(after),'before_whole_input_sha256':sha(raw),'after_whole_input_sha256':sha(updated),'intro_unchanged':raw[:heads[0].start()]==updated[:heads[0].start()],'fences_unchanged':True,'figures_changed':[],'inline_command_change':'Add --split-label test to test-side evaluation switch instruction; original reviewer already executed both label variants and must personally recheck current instruction.','initial_report_history_path':archive.relative_to(root).as_posix(),'initial_report_sha256':sha(report_raw),'scope':'One test-side split provenance instruction correction after original reviewer formal initial revise; no fenced example, core Notebook code, implementation, data, model or measured quality changed.'},ensure_ascii=False,indent=2))
