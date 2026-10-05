from pathlib import Path
import hashlib,json,datetime,platform,sys,re,shutil
root=Path(__file__).resolve().parents[4]
dir=Path(__file__).resolve().parent
path=root/'course/training.md'
raw=path.read_bytes()
heads=list(re.finditer(rb'(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+',raw))
i=next(i for i,h in enumerate(heads) if h[1]==b'T.6')
start,end=heads[i].start(),heads[i+1].start()
before=raw[start:end]
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(before)=='bde2b3d818c7ecbb1e9a9a8333d79d51d04ebfa1df2021a17e4243e6a2f2ca75'
report=root/'docs/technical-reviews/T.6.json'
report_raw=report.read_bytes()
assert sha(report_raw)=='c4be2ca669182297f8ad56f4f2772f8796c50d0f8b66a26ee06dc05761254d64'
assert json.loads(report_raw)['verdict']=='revise'
archive=root/'docs/technical-reviews/history/phase4-T_6-initial-revise-20261005.opaque.json'
assert not archive.exists()
shutil.copyfile(report,archive)
old='視覺留出兩張新位置圖，方形類別 0、圓形 1；'
new='視覺留出兩張向右偏移的藍色圖，方形、圓形各一張；同位置的紅、綠圖仍用於訓練，留出的是未見過的顏色與位置組合。方形類別 0、圓形 1；'
assert before.count(old.encode())==1
after=before.replace(old.encode(),new.encode())
assert re.findall(rb'```[^\n]*\n.*?```',before,re.S)==re.findall(rb'```[^\n]*\n.*?```',after,re.S)
(dir/'before-T.6.md').write_bytes(before)
(dir/'after-T.6.md').write_bytes(after)
updated=raw[:start]+after+raw[end:]
path.write_bytes(updated)
assert path.read_bytes()==updated
print(json.dumps({'executed_at':datetime.datetime.now(datetime.UTC).isoformat(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'source_path':'course/training.md#T.6','before_source_sha256':sha(before),'after_source_sha256':sha(after),'before_whole_input_sha256':sha(raw),'after_whole_input_sha256':sha(updated),'intro_unchanged':raw[:heads[0].start()]==updated[:heads[0].start()],'fences_unchanged':True,'figures_changed':[],'initial_report_history_path':archive.relative_to(root).as_posix(),'initial_report_sha256':sha(report_raw),'scope':'One holdout-split wording correction after original independent reviewer formal initial revise; no code, recipe, figure, implementation, data or claimed measured quality changed.'},ensure_ascii=False,indent=2))
