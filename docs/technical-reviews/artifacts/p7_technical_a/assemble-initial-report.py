"""Preserve this owner's completed assessments/checkpoints; only fix schema metadata in new report."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path('.').resolve();BASE=Path('docs/technical-reviews/artifacts/p7_technical_a')
manifest=Path('docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json')
trace=Path('docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/a/86ddcaf7b50f4afeafb3f5b041a7162c.jsonl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads(manifest.read_text());g=next(g for g in m['groups'] if g['group']=='a');rows=[json.loads(x) for x in trace.read_text().splitlines()]
assert len(g['primary_page_ids'])==60 and rows[-1]['event']=='complete'
pages=[];changes=[]
for pid in g['primary_page_ids']:
 p=BASE/'pages'/f'{pid}.json';r=json.loads(p.read_text())
 r['original_saved_assessment']={'path':str(p),'sha256':sha(p)}
 for a in r['artifacts']:
  if a['kind']=='execution' and isinstance(a.get('environment'),str):
   a['environment']={'recorded_versions_and_device':a['environment']}
   changes.append({'page_id':pid,'field':f'artifacts.{a["id"]}.environment','change':'保留实际版本/装置原字串，仅包装成schema字段映射；原assessment不改'})
  replacements={'docs/course-experiments/results/simple_models.json':str(BASE/'sources/simple-models-original-snapshot.json'),'course/figures/window_training.svg':str(BASE/'sources/window-training-original-snapshot.svg')}
  if a['path'] in replacements:
   original=a['path'];a['path']=replacements[original];assert sha(a['path'])==a['sha256']
   changes.append({'page_id':pid,'field':f'artifacts.{a["id"]}.path','change':'正式路径指向原bytes本人副本，SHA未变','original_path':original,'copied_path':a['path']})
 for c in r['claims']:
  if c['kind']=='numeric' and not isinstance(c['verification'].get('tolerance'),str):
   old=c['verification']['tolerance'];c['verification']['tolerance']='exact (absolute tolerance 0)' if old==0 else f'absolute tolerance {old} for calculated unrounded values; displayed approximations follow the stated rounding'
   changes.append({'page_id':pid,'field':f'claims.{c["id"]}.verification.tolerance','change':'原数值容差转为明确字串，无判断/数字更改','original_value':old})
 # 5.12 explicitly depends on our previous actual dedup CPU execution, not only its grams execution.
 if pid=='5.12':
  ref=json.loads((BASE/'pages/5.11.json').read_text());a=next(a for a in ref['artifacts'] if a['id']=='variation');a={**a,'id':'prior-dedup-cpu','description':'本人5.11实际已执行的原资产/dedup核算，5.12仅引用实际完整去重范围'}
  if isinstance(a.get('environment'),str):a['environment']={'recorded_versions_and_device':a['environment']}
  r['artifacts'].append(a);r['sources'].append({'id':'prior-dedup-run','kind':'execution','title':'本人前页原去重CPU','verified':True,'artifact_id':'prior-dedup-cpu'})
  c=next(c for c in r['claims'] if c['id']=='scope');c['artifact_ids'].append('prior-dedup-cpu');c['evidence'].append({'source_id':'prior-dedup-run','locator':'datasets original_text_preserved/family_full_content_hash','supports':'真正运行原函数核完整hash/family，非5.12 grams程序自动证明'})
  changes.append({'page_id':pid,'field':'scope execution reference','change':'补列本人真实已执行5.11的原dedup artifact，判断未改，原assessment保留'})
 pages.append(r)
# Local pure-schema inspection only, not truth approval or group stage gate.
sys.path.insert(0,str(ROOT/'scripts'));import check_technical_reviews as technical
for p in pages:
 errors=[];aa=technical._artifacts(ROOT,p['artifacts'],errors);ss=technical._sources(ROOT,p['sources'],aa,errors);technical._claims(p['claims'],ss,aa,errors)
 if errors:raise RuntimeError((p['page_id'],errors))
reading=[]
for r in rows[1:-1]:
 reading.append({k:r[k] for k in ['event_index','recorded_at','page_id','unit_index','unit_sha256','source_sha256','understanding','materials_and_labels','expected_change','confusion_and_quote','missing_visuals','issues','rechecks']})
report={
 'schema_version':1,'review_policy':'phase7_grouped','batch_id':'phase7','stage':'technical','group':'a','reviewer_task':'/root/p7_technical_a','reviewer_context':'fresh','manifest_sha256':sha(manifest),'verdict':'pass',
 'trace_files':[{'path':str(trace),'sha256':sha(trace)}],
 'scope':{'primary_pages':60,'range':'入口/暖身/第1–5章（manifest group a 精确primary顺序）','mandatory_boundary_context_pages':g['context_page_ids'],'session':'86ddcaf7b50f4afeafb3f5b041a7162c','actual_checkpoints':len(reading),'started_at':rows[0]['recorded_at'],'completed_at':rows[-1]['recorded_at'],'reader_four_questions':'technical阶段question_refs=[]，保留五栏与当时未知'},
 'judgment':'本人在实际读到的60页中未留下未解决的必要教材技术问题；各页概念/数字/软件/历史实测支持范围分别列明。此判断只覆盖本组与此freeze，不代表其他组、reader/continuity或全书stage gate。',
 'procedural_notes':[
  '本人初始为fork-none全新独立技术审阅者，非作者/研究者/旧reader，reviewer_context fresh指首次起点；之后多次中断恢复仍同owner，不能当新盲读者或连续逐字记忆。',
  '首次start第一shell返回丢失后重复启动；发现第一PID153923停止且尚无正文/checkpoint，保留第二实际shell30466持续poll完成start。没有将无正文的进程算阅读。',
  '02:06恢复摘要仅至1.2，current-only实际1.3unit2；仅本人past-unit-only事件44/45恢复必要旧单元，重看必要SVG/位置并做新的recoveryCPU，不覆写原checkpoint。',
  '02:45恢复摘要仅至2.3，current-only实际2.5unit3；本人past-unit-only101/102/103恢复0–2；必要Bengio原§2重读、图重新亲看与新recoveryCPU。原2.3/2.4证据属本人真实旧执行，但恢复时无其逐字上下文，未假装连续记忆。',
  '后续恢复摘要至chapter03，current-only实际3.1unit0，已有chapter03完成；继续同session、旧记录不改。其后一次恢复仅保至5.6摘要，03:39 parent通知已至5.10；current-only实际5.11unit0且已有53页。仅本人5.6–5.10 summary/source/variation恢复必要证据，不重造其正文/逐字记忆。',
  '未读raw state、未解锁教材、作者摘要/研究REPORT/他人review/旧pass。必要前文限自己已读单元；原代码及本页引用raw实验结果可以含其它函数/measurements，但不当未来教材阅读完成。',
  '2.5首次capture因previewconnectionrefused失败，没有写观看receipt；parent恢复同freeze预览后唯一capture真完成、真view才记。3.4首次--text查询SVGalt而非可见段落失败无receipt，改已读原段落后成功真view。服务/locator失败不是教材问题。',
  '5.6一次3图view工具输出因context截断，未据此声称已看；后在中断前实际重看并留本人5-6-placement-view-receipt。只对收到并看见的图像范围写receipt。',
  '必要SVG本人真正640/360view；必要桌面1280×800/手机390×844的位置/表格完成capture后本人view。无必要位置明确requiredfalse/unverified，不造全页视觉观看；PNG可在ignoredoutputs，永久小receipt绑定精确SHA及范围。',
  '1.12 broad raw search曾显示text_foundation原测量片段，非后文教材且未用于该页。文件搜索显示禁止review文件名但未读内容。4.7原raw一次广输出截断，仅visible必要字段计已读，之后按需定向读取；5.13scaling初次广输出截断，随后全部所需12值/metadata定向真读；5.11官方Python广rg截断，只有后续定向范围计阅读。',
  '原attention.py/model.py/prepare_data.py等必要source读取中可显示同文件其它函数（cache/RoPE/generate/任务分支）；这是真读原码，报告仅支持当页所用函数和scope，不声称未来教材已读或未核的功能已过。',
  '保留本人原预测错误与纠正：1.5猫索引初预测4实际3；2.1初遗漏batch axis；4.8初预测三位置实际两位置；5.9初假定layers2但原默认1。原五栏及unknown未改成事后正确预测。',
  '1.2最初systemPython缺torch/import失败及证据时间更正保留原样。1.15 T0 artifact含Python NaN示例；不把非标准JSON NaN当有效概率。',
  '5.11NIST FIPS180-4 PDF GET403未计核，改用实际IETF RFC6234§1/§9及官方Python；原PDF/整篇抽取文字只ignoredcache，正式source保URL/version/locator/PDFSHA与短必要摘录。',
  '没有重跑历史长训练或使用GPU；历史loss是读取原JSON并核sum/count、数据/原代码SHA/可重建sampler，不声称重训。5.1只实际跑40个tiny CPU toyupdates作为当页必要验证。原historicalcheckpoint没有逐个权重重新核时明确记scope。',
  '本文完整保留230条本人checkpoint五栏/当时疑问及0条正式issue；疑问后续原句解释和本人补查过程不删除。当前没有必要revise不表示其它owner范围通关。',
  'report中metadata修正仅环境字段包装、数值容差字串和正式原bytes副本/本人既有execution引用。既有pageassessments/checkpoints/trace/artifacts不覆写，每页链接原assessmentSHA。',
  '部分早期/变体CPU从stdin实际执行，保存文件中command为stdin用途标签而非完整literalargv/stdin脚本；原页逐块CPU的实际helper命令/原代码/SHA/输出均保留，已有独立check脚本的命令可复跑。未声称所有早期stdin变体都已保存完整输入或可逐字重播；中断不重造丢失输入。',
  'preflight只核本组metadata，不能证明理解/事实真或全部阶段gate，也不会收件或创建FINALreceipt；实际运行结果另存本人proof后再交FINAL。'
 ],
 'metadata_normalizations':changes,'reading_checkpoints':reading,'pages':pages
}
path=BASE/'group-a-technical-initial-20261007.json'
with path.open('x',encoding='utf-8') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(path),'sha256':sha(path),'pages':len(pages),'checkpoints':len(reading),'trace_sha256':sha(trace),'normalizations':len(changes)},ensure_ascii=False))
