from pathlib import Path
import hashlib,json,datetime,sys,platform,re,shutil
root=Path(__file__).resolve().parents[4];dir=Path(__file__).resolve().parent;path=root/'course/glossary.md';raw=path.read_bytes();sha=lambda b:hashlib.sha256(b).hexdigest()
heads=list(re.finditer(rb'(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+',raw));i=next(i for i,h in enumerate(heads) if h[1]==b'G.4');start,end=heads[i].start(),len(raw);before=raw[start:end]
assert sha(before)=='428fb6f215a88db3503f4cf583da030c09c5867450c8b2d2af122b878193a5c7'
report=root/'docs/technical-reviews/G.4.json';report_raw=report.read_bytes();assert sha(report_raw)=='fbd88dfb7fd8fc8ff543d9e9637df4def74fcffe74d9d9b97d5f1ab3f78bc060' and json.loads(report_raw)['verdict']=='revise'
archive=root/'docs/technical-reviews/history/phase4-G_4-initial-revise-20261005.opaque.json';assert not archive.exists();shutil.copyfile(report,archive)
old='保留後訓練起點的模型副本，用來比較回答傾向偏移多少'
new='保留這次偏好或強化學習訓練起點的模型副本，用來比較回答傾向偏移多少'
assert before.count(old.encode())==1;after=before.replace(old.encode(),new.encode());assert re.findall(rb'```[^\n]*\n.*?```',before,re.S)==re.findall(rb'```[^\n]*\n.*?```',after,re.S)
(dir/'before-G.4.md').write_bytes(before);(dir/'after-G.4.md').write_bytes(after);updated=raw[:start]+after;path.write_bytes(updated);assert path.read_bytes()==updated
print(json.dumps({'executed_at':datetime.datetime.now(datetime.UTC).isoformat(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'source_path':'course/glossary.md#G.4','before_source_sha256':sha(before),'after_source_sha256':sha(after),'before_whole_input_sha256':sha(raw),'after_whole_input_sha256':sha(updated),'intro_unchanged':raw[:heads[0].start()]==updated[:heads[0].start()],'fences_unchanged':True,'figures_changed':[],'initial_report_history_path':archive.relative_to(root).as_posix(),'initial_report_sha256':sha(report_raw),'scope':'One fixed-reference starting-point definition narrowed to this preference/RL training stage after original independent formal revise; no other row, code, diagram, implementation, data, new measurement or other section changed.'},ensure_ascii=False,indent=2))
