from pathlib import Path
import json,hashlib,datetime,urllib.request,re
from html.parser import HTMLParser
root=Path('docs/course-revision-20261006-phase6/continuity');base=root/'artifacts/a/recheck-01';freeze=root/'revised-01';order=['19.1','19.2','19.3','19.4','19.5'];old_path=root/'reports/a.json';old=json.loads(old_path.read_text());old_sha=hashlib.sha256(old_path.read_bytes()).hexdigest();oldtrace=root/'traces/a.jsonl';oldtrace_sha=hashlib.sha256(oldtrace.read_bytes()).hexdigest();inventory_path=freeze/'inventory.json';inv=json.loads(inventory_path.read_text());ids=set(p['page_id'] for p in old['canonical_pages']);entries={p['page_id']:p for p in inv['pages'] if p['page_id'] in ids};traces=[json.loads(line) for line in (root/'traces/a-recheck-01.jsonl').read_text().splitlines()];bytrace={t['page_id']:t for t in traces}
assert [t['page_id'] for t in traces]==order
assert [t['sequence'] for t in traces]==list(range(1,6))
assert all(len(t['five_point_understanding'])==5 for t in traces)
class Codes(HTMLParser):
 def __init__(self):super().__init__();self.depth=0;self.buf=[];self.codes=[]
 def handle_starttag(self,tag,attrs):
  if tag=='code':self.depth+=1;self.buf=[]
 def handle_endtag(self,tag):
  if tag=='code' and self.depth:self.codes.append(''.join(self.buf));self.depth-=1
 def handle_data(self,data):
  if self.depth:self.buf.append(data)
version=[];retained=[];canonical=[]
for original in old['canonical_pages']:
 page=original['page_id'];entry=entries[page];p=Path(entry['snapshot']);raw=p.read_bytes();text=raw.decode();sha=hashlib.sha256(raw).hexdigest();assert sha==entry['source_sha256'];source_contains=text.strip() in Path(entry['source']).read_text();assert source_contains
 fresh=page in order;old_source_same=original['source_sha256']==entry['source_sha256'];old_fig_same=original['figures_sha256']==entry['figures_sha256'];v={'page_id':page,'source_sha256':sha,'current_canonical_section_contains_snapshot':source_contains,'same_source_as_original':old_source_same,'same_figures_as_original':old_fig_same,'new_actual_read_this_recheck':fresh,'figures':[]}
 for f,expected in entry['figures_sha256'].items():
  q=Path(f);assert hashlib.sha256(q.read_bytes()).hexdigest()==expected
  url='http://127.0.0.1:8793/figures/'+q.name
  with urllib.request.urlopen(url,timeout=20) as resp:served=resp.read();status=resp.status
  served_sha=hashlib.sha256(served).hexdigest();assert served_sha==expected and status==200
  v['figures'].append({'path':f,'source_sha256':expected,'preview_url':url,'preview_sha256':served_sha,'preview_status':status})
 if fresh:
  saved=base/'source-pages'/p.name;assert saved.read_bytes()==raw
  url=f'http://127.0.0.1:8793/{page}.html'
  with urllib.request.urlopen(url,timeout=20) as resp:html=resp.read();status=resp.status
  assert status==200;(base/'browser'/f'{page}.html').write_bytes(html);parser=Codes();parser.feed(html.decode());blocks=re.findall(r'```python\n(.*?)\n```',text,re.S);code_match=all(any(a.strip()==b.strip() for a in parser.codes) for b in blocks);assert code_match
  v.update({'own_saved_source_sha256':hashlib.sha256(saved.read_bytes()).hexdigest(),'preview_url':url,'preview_status':status,'preview_html_sha256':hashlib.sha256(html).hexdigest(),'all_source_python_blocks_in_actual_html':code_match})
  summary=bytrace[page]['own_paragraph_summary'];status_name='actual_new_reread';evidence={'trace_sequence':bytrace[page]['sequence'],'source_snapshot':str(saved)}
 else:
  assert old_source_same and old_fig_same
  assert hashlib.sha256(Path(original['raw_utf8_snapshot']).read_bytes()).hexdigest()==sha
  summary=original['own_summary'];status_name='retained_original_actual_evidence_not_newly_read';evidence={'original_report':str(old_path),'original_trace':str(oldtrace),'original_trace_sequence':original['trace_sequence'],'original_source_snapshot':original['raw_utf8_snapshot']}
  retained.append({'page_id':page,'source_sha256':sha,'figures_sha256':entry['figures_sha256'],'same_source_and_figures_verified':True,'newly_read_this_recheck':False,'retained_original_verdict':original['verdict'],**evidence})
 canonical.append({'page_id':page,'source':entry['source'],'selector':entry['selector'],'source_sha256':sha,'figures_sha256':entry['figures_sha256'],'verdict':'pass','evidence_status':status_name,'own_summary':summary,'summary_origin':'本次當頁摘要' if fresh else '原真正實讀報告摘要，非本次新讀',**evidence});version.append(v)
# Supplementary pages were genuinely read in the original pass; verify bytes only now.
retained_supp=[]
for original in old['supplementary_reads']:
 page=original['page_id'];p=freeze/'source-pages'/f'{page}.md';sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==original['source_sha256'];figs=original['figures_sha256'];assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==s for f,s in figs.items())
 retained_supp.append({'page_id':page,'source_sha256':sha,'figures_sha256':figs,'newly_read_this_recheck':False,'same_source_and_figures_verified':True,'original_report':str(old_path),'original_trace':str(oldtrace),'original_source_snapshot':original['source_utf8_snapshot'],'reason':'保留18.12明確連結的必要背景實讀證據；本次只作版次指紋核對，沒有標為新的逐頁閱讀。'})
visual=[]
notes={
 '19.1':'同一核心前有文字歷史、商品、指定字卡、實際錄音四路；工具實算後交回核心。圖在入口說明後、能力表前，原始手機尺寸可讀。',
 '19.3':'訓練來源A框住單物件、配對、換位，驗證B用選點、最後C定版核對，正接正文家族規則和後續用途表。',
 '19.4':'八階段箭頭與每支箭頭載自己的best、新段重建更新/抽樣狀態可讀。圖位於權重鏈引入後、更新模組表之前；桌機兩次滾動看完，手機圖完整顯示。'
}
for page in notes:
 for r in json.loads((base/'browser'/f'{page}-capture.json').read_text()):
  r.update({'actually_viewed_by_this_reviewer_now':True,'verdict':'pass','position_caption_and_readability':notes[page],'capture_note':'實際viewport截圖；桌機第二張上部捲出屬正常閱讀滾動，沒有依element自動截圖遮擋判定layout。','viewed_file_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in r['viewed_files']}});visual.append(r)
bridges={
 ('19.1','19.2'):'入口與共用核心→核心內的四expert選二；以App/商品向量開場接同一任務，再把公開推論與隨機結構計數分開。',
 ('19.2','19.3'):'計數不能當能力→如何確保成品考題不用已見素材；同包衍生題與材料家族是前頁留出的生成評估條件。',
 ('19.3','19.4'):'固定資料及驗證選點→固定本路線父權重鏈；best/完成與選定步數都有對象，init與resume不混淆。',
 ('19.4','19.5'):'結尾先看SFT完整示範→條件變答案變的作者示範；短程式的labels/EOS目的不被當成新生成或訓練成果。'
}
transitions=[{'from_page':a,'to_page':b,'verdict':'pass','own_bridge_judgment':bridges[(a,b)],'five_point_record':bytrace[b]['five_point_understanding'],'trace_sequence':bytrace[b]['sequence'],'necessary_questions':[]} for a,b in zip(order,order[1:])]
check={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inventory':str(inventory_path),'inventory_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),'pages':version,'retained_supplementary':retained_supp,'all_pass':True};(base/'version-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
tracepath=root/'traces/a-recheck-01.jsonl';report={'schema_version':1,'reviewer_task':'a-recheck-01','original_reviewer_task':'a','reviewer_identity':{'task_name':'/root/p6_continuity_a','original_third_round_owner_actual_reread':True,'reader_background':'數學不錯高中生／基本數學大學生與入門Python；不以專家背景補本文缺橋。','read_author_revision_notes_or_other_reports':False,'delegated_agents':False},'verdict':'pass','finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_order':order,'current_newly_read_scope':order,'canonical_pages':canonical,'transitions':transitions,'issues':[],'necessary_questions':[],'necessary_revisions':[],'trace_file':str(tracepath),'trace_sha256':hashlib.sha256(tracepath.read_bytes()).hexdigest(),'source_versions':{'inventory':str(inventory_path),'inventory_sha256':check['inventory_sha256'],'hash_scheme':'source_sha256對應section snapshot，非整章檔SHA。','version_check':str(base/'version-check.json'),'current_section_source_and_figures_passed':True,'current_preview_python_code_and_served_figures_match_source':True,'raw_utf8_new_read_sources':str(base/'source-pages')},'visual_checks':visual,'notebook_checks':[{'page_id':t['page_id'],**t['notebook_read']} for t in traces],'retained_original_scope':retained,'retained_original_supplementary_scope':retained_supp,'original_evidence_preserved':{'report':str(old_path),'report_sha256':old_sha,'trace':str(oldtrace),'trace_sha256':oldtrace_sha,'modified':False},'original_optional_issue_status':{'id':'a-optional-18.14-PTQ','status':'保留原optional，18.14本次未新讀且來源/圖一致；非必要修訂。'},'recheck_judgment':'新版19.2配置分行、19.4repo搜尋行排版都保留原程式對象與作用；今天實讀五頁、當頁trace、三圖實際桌機/手機查看與當前kernel輸出後，四個交接點仍成立。','unverified':['本次沒有重新閱讀18.12、18.13、18.14、chapter-19或原補讀18.6/18.8；它們只保留真實舊實讀證據且核對section/圖相同。','沒有新跑GPU、蒸餾、訓練或模型生成；19.1/19.5的公開CPU回答是本文既有例，不宣稱此次重做。','沒有重新運行資料去重準備、逐檔父checkpoint驗證或wrapper重做配方；本次讀了已執行kernel的相應離線紀錄。','19.4短程式已讀新版source與actual HTML並核對kernel輸出；手機程式長行的全部水平捲動未作額外互動測試。','沒有讀19.6以後正文，也沒有讀其他作者、讀者或技術報告。'],'scope_complete':True}
path=root/'reports/a-recheck-01.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');report_sha=hashlib.sha256(path.read_bytes()).hexdigest();assert hashlib.sha256(old_path.read_bytes()).hexdigest()==old_sha and hashlib.sha256(oldtrace.read_bytes()).hexdigest()==oldtrace_sha
receipt={'report':str(path),'report_sha256':report_sha,'trace':str(tracepath),'trace_sha256':report['trace_sha256'],'new_actual_read_scope':order,'retained_original_scope':[r['page_id'] for r in retained],'actual_viewport_screenshots':sum(len(v['viewed_files']) for v in visual),'current_versions_verified':True,'old_report_trace_unchanged':True,'verdict':'pass'};(base/'own-evidence-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2))
