from pathlib import Path
import hashlib,json,datetime,platform,sys,re,shutil
root=Path(__file__).resolve().parents[4]
dir=Path(__file__).resolve().parent
path=root/'course/training.md'
raw=path.read_bytes();updated=raw
sha=lambda b:hashlib.sha256(b).hexdigest()
heads=list(re.finditer(rb'(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+',raw))
changes=[('T.5','e6047ebd89ea07c182211fd25cbd28d7b4938e8476a55857c24b6b9675812a03','de71c4923b1c43f0a5e86f2e7b5467a448246988b2d82abe34bad730d81bce27','使用示例與原始欄位見[實驗說明](../docs/course-experiments/README.md)。','固定實驗入口與基底核對提醒見[實驗說明](../docs/course-experiments/README.md)。'),('T.8','e9d9c6cb3fbe51565dc5535614a16dd098adff791a8fc0c73eaa3cdfc43dd722','003e715fac0a3b39a43d2d1672e37b0c2877867ba5901c2c053f315a9ff157ec','報告的`results.verification_passed`若為`true`，表示這次探針的核對條件通過；未通過或未執行時，先看原因，不據此聲稱使用成功。仍要核對實際後端、數值容差及記憶體範圍，不能由SDPA這個API名稱推定已用Flash；','報告的`results.verification_passed=true`只表示`supported_routes`至少有一條已核驗Flash後端的路線，而且這個集合內的路線都已完成；其他路線仍可能失敗。先查看各路線的`status`、後端核驗與數值容差，再以`results.schedule_completed`和`results.status`判讀整次探針是否完成，以及是否有不支援的路線。未執行或未完成時，先看原因，不據此聲稱使用成功。還要核對記憶體範圍，不能由SDPA這個API名稱推定已用Flash；')]
receipts=[]
for lesson,source_sha,report_sha,old,new in changes:
 i=next(i for i,h in enumerate(heads) if h[1].decode()==lesson)
 start,end=heads[i].start(),heads[i+1].start()
 before=raw[start:end]
 assert sha(before)==source_sha
 report=root/f'docs/technical-reviews/{lesson}.json';report_raw=report.read_bytes();assert sha(report_raw)==report_sha and json.loads(report_raw)['verdict']=='revise'
 archive=root/f'docs/technical-reviews/history/phase4-{lesson.replace(".","_")}-initial-revise-20261005.opaque.json';assert not archive.exists();shutil.copyfile(report,archive)
 assert before.count(old.encode())==1
 after=before.replace(old.encode(),new.encode())
 assert re.findall(rb'```[^\n]*\n.*?```',before,re.S)==re.findall(rb'```[^\n]*\n.*?```',after,re.S)
 (dir/f'before-{lesson}.md').write_bytes(before);(dir/f'after-{lesson}.md').write_bytes(after)
 assert updated.count(before)==1;updated=updated.replace(before,after)
 receipts.append({'lesson_id':lesson,'before_source_sha256':sha(before),'after_source_sha256':sha(after),'initial_report_history_path':archive.relative_to(root).as_posix(),'initial_report_sha256':sha(report_raw),'fences_unchanged':True,'figures_changed':[]})
assert re.findall(rb'```[^\n]*\n.*?```',raw,re.S)==re.findall(rb'```[^\n]*\n.*?```',updated,re.S)
path.write_bytes(updated);assert path.read_bytes()==updated
print(json.dumps({'executed_at':datetime.datetime.now(datetime.UTC).isoformat(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'source_path':'course/training.md','before_whole_input_sha256':sha(raw),'after_whole_input_sha256':sha(updated),'intro_unchanged':raw[:heads[0].start()]==updated[:heads[0].start()],'whole_fences_unchanged':True,'changes':receipts,'scope':'Two localized pure-prose corrections after each original owner formal initial revise. LoRA reference promise now matches actual linked method page; Flash flag scope distinguished from whole schedule status. No code, experiment recipe, figures, implementation, data, measured outcomes or other section changed.'},ensure_ascii=False,indent=2))
