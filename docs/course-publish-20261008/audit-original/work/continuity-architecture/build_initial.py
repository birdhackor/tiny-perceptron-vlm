from pathlib import Path
import json,hashlib,datetime,copy,re
B=Path('/workspace/work/tutorial-audit-20261008');M=json.loads((B/'manifest.json').read_text());P={p['page_id']:p for p in M['inventory']['pages']};G=M['groups']['architecture']['pages']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
main=B/'reports/continuity-architecture-main-notes.json'; mainseal=json.loads((B/'work/continuity-architecture/main-notes-seal.json').read_text());assert sha(main)==mainseal['sha256']
R=json.loads(main.read_text());E=json.loads((B/'work/continuity-architecture/actual-extras-notes.json').read_text());V=json.loads((B/'work/continuity-architecture/page-visual-observations.json').read_text())
V.append({'paths':[s+'-360.png' for s in ['flash_allocated_memory','rewrite-17-qat-flow','rewrite-15-combine','p7-16.14-draft-verify']],'kind':'renders','judgment':'先view後紀錄：360Flash灰65基線與彩增量/總峰值四條清楚，Flash極小增量細線有數值不靠顏色大小讀；QAT每階段及未step文字完整；expert合併A/B比例對應与歸一相乘、草稿首錯停止後棄嗎都无需放大可辨。'})
V.append({'paths':['flash_allocated_memory-640.png'],'kind':'renders','judgment':'extras16.9新增圖先實看640再判：四柱同尺度，灰色65基線與彩新增清楚，97.75 vs67.03總峰值另列，圖底限定核心/非整卡或LM/no loss-step與正文相同。不以新增倍數說整卡省。'})
for v in V:
 v['viewed_before_judgment']=True
 v['assets']=[]
 for name in v['paths']:
  path=B/('renders' if v.get('kind')=='renders' else 'page-captures')/name
  assert path.exists(),path
  v['assets'].append({'path':str(path),'sha256':sha(path)})
quotes={
'3.2':'點積也受數字大小影響。',
'3.3':'query 表示提出需求，key 表示提供匹配線索',
'4.1':'因果許可決定誰能讀誰，位置特徵給每位置額外數字，兩者用途不同。',
'4.3':'每位置先減自己的平均，再除自己的標準差',
'4.5':'主路沒有被 LN 永久替換；LN 在分支中。',
'4.4':'共享配方不等於位置互相讀取',
'2.1':'ID 只選列，數字 1 或 2 本身不代表字義',
'4.6':'續寫只先用最後那排選新 ID，再把它接回輸入重算',
'first-steps':'當調整量越來越小，「代價改變數 ÷ 旋鈕改變數」會靠近某個數',
'4.2':'殘差讓已有資訊沿直接路往後算，也讓梯度沿這條路往前層傳回。',
'3.4':'整條運算可寫 `softmax(QKᵀ)V`',
'3.7':'兩頭都讀整串允許的位置',
'5.17':'評估模式控制前向行為，不負責關閉求導。',
'7.3':'Y中的-100表示忽略直接答案代價，不是輸入token。',
'7.11':'仍是下一token和交叉熵，不需另換一種模型才能回答。',
'7.4':'原序列索引5的A，配到Y索引4',
'10.7':'先展開、再對齊，讓這個責任只發生一次。'
}
for p in R['actual_prerequisites']:
 raw=Path(P[p['page_id']]['snapshot']).read_text(); q=quotes[p['page_id']];assert q in raw
 p['quoted_basis']={'quote':q,'path':P[p['page_id']]['snapshot'],'line':raw[:raw.index(q)].count('\n')+1}
 p['reader_background_classification']='實需補讀的已存在主文先備；本組局部review開始未預設讀過；不是教材缺前文。'
byE={p['page_id']:p for p in E['pages']}
for p in R['pages']:
 raw=Path(P[p['page_id']]['snapshot']).read_text()
 for q in p['quoted_basis']:q['line']=raw[:raw.index(q['quote'])].count('\n')+1
 p['supplement_after_main_seal']=byE[p['page_id']]
 p['figures_sha256']=P[p['page_id']]['figures_sha256']
for issue in R['issues']:
 pid=issue['page_id'];raw=Path(P[pid]['snapshot']).read_text(); issue.update({'source_sha256':P[pid]['source_sha256'],'quote_line':raw[:raw.index(issue['quote'])].count('\n')+1,'reviewer':'/root/continuity_architecture','type':'方法理由（第二改動need）','mechanism_status':'已足夠','need_status':'需補說明','supplement_followup':'main封存後讀14.10折疊，補γ/c公式及來源，仍未指擴長何種分布需要尺度管理；不回寫main筆記','repair_scope_reason':'只需補第二改動與延長需求的因果一句／一段，不否定分頻近遠說明，無須改章順或重寫整頁。','severity_reason':'可操作並看懂scale性質，但原文「為什麼還有下面那一步」承諾的採用理由須自補；不阻整章機制。'})
figs={f:h for pid in G for f,h in P[pid]['figures_sha256'].items()}
static=[]
for f,h in sorted(figs.items()):
 name=Path(f).stem;path=B/'renders'/f'{name}-640.png';assert path.exists()
 static.append({'source':f,'source_sha256':h,'viewed_render':str(path),'render_sha256':sha(path),'scope':'实际view_image640，看后判对应；25张全组含选读flash图。原位页面只另列采样，不宣称全站布局通过。'})
for vo in R['visual_observations']:
 vo['source_figures']=[{'source':f,'sha256':h} for f,h in {f:h for p in P.values() for f,h in p['figures_sha256'].items()}.items() if Path(f).stem in vo['figures']]
R.update({'schema_version':1,'criteria_sha256':{k:v for k,v in M['criteria_sha256'].items() if k in ['SKILL.md','references/review-protocol.md','references/calibration.md']},'freeze_baseline_commit':M['baseline_commit'],'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'independence':{'source_reading':'完整主文銜接審，非第一輪盲增量；第三輪可以讀全組來源與實需先備。','unseen_before_seal':['reports除自身稿件','synthesis','notes','traces','checks的人類結論','舊/同行review','repo歷史reader/technical/continuity/validation-review'],'peer_hints':'root僅提供stage/group通用規準與純PNG來源路徑，無問題答案；此報封存前未見其他判斷。','ai_review':'AI輔助，不是真人學生測試。'},'coverage':{'assigned':len(G),'main_actually_read':len(R['pages']),'main_ids_in_reading_order':G,'supplements_actually_read':len(E['pages']),'pages_with_details':sum(p['details_count']>0 for p in E['pages']),'source_delivery_is_not_comprehension_evidence':True,'actual_page_judgments':str(main),'actual_supplement_judgments':str(B/'work/continuity-architecture/actual-extras-notes.json'),'main_notes_immutable_sha256':mainseal['sha256'],'main_notes_sealed_at':mainseal['at'],'necessary':1,'optional':0,'blocker':0,'burden':1,'rewrite_scales':{'paragraph':1,'page':0,'chapter':0,'order':0}},'visual_scope':{'all_group_static_figures_640':static,'post_seal_page_and_mobile_observations':V,'actual_pages_sampled':['14.1','15.6','16.4','17.8','18.6','18.1圖裁片','16.7選讀手機開表'],'new_preview_limitation':'新站page-captures省略runtime outputs與reading annotations；未讀這些介面狀態，不宣稱全頁互動驗收。','not_verified':['其它頁原位手機/桌面布局','真實點擊折疊/放大/切換及滾動連續行為','其它11张360單體（但25图640均实看，若同图有新390capture另记）']},'execution_scope':{'executed':'只有來源delivery、筆記/一致性/哈希封存腳本；沒有執行教材算例、模型訓練或重做GPU量測。','code_arithmetic':'隨閱讀逐數核toy算式/形狀/依賴；不以此冒runtime執行證據。','original_raw_result_jsons':'未開直接連結的原始實報，所有實測數字與原文版本屬未驗證，不自動判錯。','external_papers':'僅讀正文/折疊引用位置與主張，未取得原論文全文，無外部查證宣称。'},'strong_passes':[
{'pages':['14.1','14.7','14.8','14.9'],'relationship':'相對詞距→共同旋轉保點積→訓長/執行上限/任務能力分離→保字壓尺→近處差角取捨，各有正文及實看圖依據。'},
{'pages':['15.4','15.5','15.6','15.7'],'relationship':'chosen索引只選谁/weights決定份量；先選再执行省算；同寬合併vs殘差；按token來源列累加；top1 p/p消梯度而保原p傳任務訊號，延期理由確已回收。'},
{'pages':['16.3','16.4','16.5','16.6'],'relationship':'cache因果不變可重用与offset/失效；四Q共少KV减少未扩張保存；packing同文與因果交集/跨边界目标；有效token總分母多backward末一次step。'},
{'pages':['16.8','16.9','16.12','16.13','16.14'],'relationship':'SDPA軸/True允許/一次縮放/dropout契約；online softmax新基準分子分母同縮；分块存储安排vs滑窗/局部連結；greedy草稿首錯修正丟後綴保目標序列。'},
{'pages':['17.2','17.3','17.5','17.6','17.7','17.8','17.14','17.15'],'relationship':'碼與浮點對應、數值範圍vs容器bytes/pack、組scale細節與metadata代价、fakeQ/STE/浮主權重/step/最後打包；小MAE不保EOS與生成選擇。'},
{'pages':['18.1','18.3','18.4','18.5','18.6','18.8','18.9','18.10','18.13','18.14'],'relationship':'原標籤/同硬答/教師比例的資訊與梯度不同；温度次候选权重需要；同候選身份/前文/答位；有效labels仅maskKL/T²一次；架構/信號/量化四组隔離成本與任務，模態展開按同答案前一格配列。'}
],'unknowns':[
{'scope':'原始實測數字/參數設定/效果比較','classification':'未驗證','reason':'本輪從主文与折疊核教學界線，未開原始JSON/實作/論文；不能計為效果验证通過，亦不据未知報錯。'},
{'scope':'選讀與後續詳法','classification':'可選/合理待教','reason':'完整位置頻段係數、Flash GPU核心、隨機speculative算法、GQA多後端、外詞表映射等不需當頁詳算法；正文基本角色、規則與限定已教。'},
{'scope':'整站閱讀路徑/互動','classification':'未驗證','reason':'僅實看列出的desktop/mobile截屏及單體圖；不能聲稱全部頁runtime/互動已驗。'}
]})
assert len(R['pages'])==71 and set(p['page_id'] for p in R['pages'])==set(G)
out=B/'reports/continuity-architecture-initial.json';out.write_text(json.dumps(R,ensure_ascii=False,indent=2)+'\n')
seal={'reviewer':'/root/continuity_architecture','role':'continuity','group':'architecture','sealed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'initial_path':str(out),'initial_sha256':sha(out),'main_notes_path':str(main),'main_notes_sha256':sha(main),'supplement_notes_sha256':sha(B/'work/continuity-architecture/actual-extras-notes.json'),'meaning':'独立来源初判已完整；封存前未读旧或同行人类结论；後续交叉应另存，不修改此檔。'}
sp=B/'reports/continuity-architecture-seal.json';sp.write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'initial':str(out),'seal':str(sp),'sha256':seal['initial_sha256'],'pages':71,'necessary':1,'optional':0,'unknown':'實測原始數字、未抽查原位頁互動','rewrite_scale':'paragraph'},ensure_ascii=False))
