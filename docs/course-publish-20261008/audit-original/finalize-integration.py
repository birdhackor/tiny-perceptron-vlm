import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
b=Path(__file__).resolve().parent
main_path=b/'reports/reader-integration-main.json';main_raw=main_path.read_bytes();main_hash=hashlib.sha256(main_raw).hexdigest()
trace_path=b/'traces/reader-integration.jsonl';records=[json.loads(x) for x in trace_path.read_text().splitlines()]
seal=[r for r in records if r['kind']=='sealed_main_report'][-1];assert seal['sha256']==main_hash
report=json.loads(main_raw);blocks=json.loads((b/'work-integration-extras.json').read_text())
notes=[json.loads((b/f'notes/integration-extras-{n:02}.json').read_text()) for n in [1,2]]
read_blocks=[block for note in notes for block in note['blocks']]
assert len(read_blocks)==len(blocks)==13
assert {(x['page_id'],x['block_index']) for x in read_blocks}=={(x['page_id'],x['block_index']) for x in blocks}
now=datetime.now(timezone.utc).isoformat()
for note in read_blocks:
 block=next(x for x in blocks if x['page_id']==note['page_id'] and x['block_index']==note['block_index'])
 with trace_path.open('a') as f:
  f.write(json.dumps({'kind':'optional_block_direct_read','at':now,'reviewer':'/root/read_integration','phase':'extras','page_id':block['page_id'],'block_index':block['block_index'],'label':block['label'],'block_sha256':block['sha256'],'main_report_sha256':main_hash,'note':note},ensure_ascii=False)+'\n')
report['kind']='independent reader audit with post-seal optional supplement';report['created_at']=now
report['main_report']={'path':str(main_path),'sha256':main_hash,'sealed_at':seal['at'],'original_bytes_preserved':True}
for page in report['pages']:
 page['optional_reading']={'folded_blocks_read':[n for n in read_blocks if n['page_id']==page['page_id']],'no_folded_blocks':not any(n['page_id']==page['page_id'] for n in read_blocks),'main_judgment_preserved':True}
report['post_seal_supplement']={'route':'父協調者授權封存後直接讀凍結來源的折疊區；未使用next重讀已完成正文，也未冒稱新一輪逐段正文首讀。','blocks':[{'page_id':x['page_id'],'block_index':x['block_index'],'label':x['label'],'sha256':x['sha256'],'start':x['start'],'end':x['end']} for x in blocks],'notes':notes,'new_necessary_issues':[],'impact':'13折疊區補操作、来源定位、詞表限定、CTC訓練/簡單解碼與工具細項範圍；沒有補19.3/19.6承諾的材料使用控制觀察摘要，因此INT-03保留。','main_source_rechecks':['19.12全文','19.3全文','19.6/19.7關鍵控制原句'],'main_source_recheck_effect':'只為精確引用/範圍追加核對，不改原始理解與主文封存內容。'}
for issue in report['issues']:
 if issue['id']=='INT-03':
  issue['final_disposition']={'necessary':True,'severity':'burden','status':'待局部補写','scope':'19.3/19.6所承諾的感知材料、位置、文字格式/續聊條件的答案分布基準與成對控制觀察；19.7提出工具返回值重放作法，但不替這些感知控制給結果。','exact_promises':[{'page_id':'19.3','quote':'資料隔離完成後，還要看答案分佈和成對題：例如只回答「包」能猜中多少商品題，同圖換位置是否真的改變答案。零家族交集排除了指定來源的跨份重複，不能單獨證明模型使用了材料。這些檢查會在感知和最終驗收時與模型回答一起看。'},{'page_id':'19.6','quote':'是否能依新素材改答、能否保留文字格式與續聊主題，統一用19.12的留出題及成對控制核對。'}],'strongest_support':'19.12保留各用途最後考卷成功數、分母、必要判尺與能力限定，也給入口CTC324/324對最终OCR252/324、281/324的中間/最終對照，足以支持有限考卷報告。','remaining_gap':'沒有呈現答同一常見類別的基準，或條件保持/交換時配對答案的預期與實際觀察；讀者仍需外跳選取原始證據，不能在主文回收已承諾材料依賴核對。','minimal_repair':'在19.12追加短摘要，至少明列原承諾的答案分布基準與必要成對控制條件、期望改變、實際結果/分母。從既有紀錄選必要關係即可，不要求全面新評測或重跑訓練。','rewrite_scale':'局部小節補充；不需19/20章重排或重寫。','post_optional_status':'19.12折疊區只有工具邊界細項，不能消除此必要缺口。'}
 else:
  issue['final_disposition']={'necessary':False,'severity':'optional','status':'可保留原文或極短名稱補充','benefit':'便於把清楚的既有角色對到常用名稱/查閱規則；沒有新的機制理解收益，也不補必要概念。','minimal_repair':'僅在全書實際首次名稱处視需要補短括註；若已讀先備已有則保留。','added_burden':'不宜展開通用術語小課或額外演算法，避免打斷操作/判尺主線。','later_clarification':'19.5已說監督式微調，20.1已說語音辨識，20.4已說圖片文字辨識；20.13明說NFKC字元形式正規化。角色含義不依赖英文全名；早期名稱未知紀錄保留。'}
report['coverage'].update({'completed_optional_blocks':13,'pages_screened_for_folded_sections':30,'optional_blocks_pages':len(set(x['page_id'] for x in blocks)),'optional_reading_completed':True,'main_checkpoints_unchanged':148,'external_answer_or_implementation_read':False,'not_read':['project-context','作者歷史','舊審閱','同行報告','實作與外部答案','折疊區外鏈原始證據檔、論文、未選遠程分支指南'],'report_aggregation':'主文只機械彙整148個已手寫checkpoint；選讀只機械附入13塊的已手寫理解，無自動生成回答。'})
report['visual_scope']['optional_new_figures']=[]
report['summary']['final_rewrite_scale']='必要：19.12局部補承諾的材料使用控制摘要。其餘閱讀證據不支持大改/重寫；可選名称收益有限，宜短补或保留。'
out=b/'reports/reader-integration.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
assert hashlib.sha256(main_path.read_bytes()).hexdigest()==main_hash
print(json.dumps({'path':str(out),'main_sha256':main_hash,'main_pages':len(report['pages']),'checkpoints':148,'optional_blocks':13,'necessary_issues':1,'new_optional_necessary_issues':0},ensure_ascii=False))
