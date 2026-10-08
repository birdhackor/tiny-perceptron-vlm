from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,re
B=Path('/workspace/work/tutorial-audit-20261008'); M=json.loads((B/'manifest.json').read_text()); P={p['page_id']:p for p in M['inventory']['pages']}
main_path=B/'reports/continuity-integration-main-notes.json'; main=json.loads(main_path.read_text()); main_seal=json.loads((B/'work/continuity-integration/main-notes-seal.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(main_path)==main_seal['sha256']
def q(pid,quote):
 raw=Path(P[pid]['snapshot']).read_text(); assert quote in raw,(pid,quote)
 return {'page_id':pid,'line':raw[:raw.index(quote)].count('\n')+1,'quote':quote}
reason={
'chapter-19':'回查已教概念定位；本轮已实读所需前文，不靠此链接自动放行。',
'19.1':'uv环境、匿名固定权重推论及answer/EOS/selected_step字段；主文已明示该表为真生成与历史来源，安装可选。',
'19.2':'550普通字/控制ID来源与未知字元范围更细；主文参数表无需知道每一ID的建立算法。',
'19.3':'manifest/家族标记与另外像素检查；主文早已给真正隔离规则，补充是复做查证。',
'19.4':'wrapper具体argv及父来源验指纹规则；主文已教新段/恢复状态、完工/选定边界。',
'19.5':'训练示范/编码器/CPU原始记录位置；不把外链命中当已验证数字。',
'19.6':'CTC合并相邻重覆、去blank及兼容路径总概率首次出现；这是主文32欄对几字的基本机制关系。来源/素材/argv第二块为可选查证。',
'19.7':'两次模型generations与tool_trace查证；主文已经说清解析/真算/回填边界。',
'19.11':'四安全檔下载复用及已知验证例操作；主文包裹角色已足够。',
'19.12':'另72边界题与主表构成3734，主表明确为主要能力；这块并未偷换第二考卷或给高分补真往返。',
'20.2':'Linux CPU固定commit、venv、版本、延迟加载操作。安装在折叠有清楚入口，主文input/state规则不依赖安装命令。',
'20.8':'候选实际训练/评分源位置；主文拒绝语音聊天及EOS已有直接证据，完整公式在已读training。'
}
extras=[]
for pid in M['groups']['integration']['pages']:
 raw=Path(P[pid]['snapshot']).read_text(); d=list(re.finditer(r'<details\b[^>]*>.*?</details>',raw,re.S));quotes=[]
 for x in d:
  t=re.search(r'<summary[^>]*>(.*?)</summary>',x[0],re.S)
  quotes.append(q(pid,t[1]))
 extras.append({'page_id':pid,'details_count':len(d),'actually_read_all_details':True,'quoted_labels':quotes,'supplement_delta':reason.get(pid,'本页没有折叠区；主文判定无需由补充替换。')})
extras_path=B/'work/continuity-integration/extras-notes.json';extras_path.write_text(json.dumps({'reviewer':'/root/continuity_integration','at':datetime.now(timezone.utc).isoformat(),'main_notes_sha256':main_seal['sha256'],'pages':extras},ensure_ascii=False,indent=2)+'\n')
issues=[{
'id':'continuity-integration-CTC-main','page_id':'19.6','type':'正文与选读 / 方法机制','severity':'burden','rewrite_scale':'paragraph',
'quote':q('19.6','讀字入口的預備訓練使用**CTC**（Connectionist Temporal Classification，連結式時序分類），處理「影像有32個欄位，答案卻只有幾個字」的對齊問題。'),
'already_taught':'主文已完整交代入口先裁框、按左至右32特征位置、最后由核心生成；CTC的目标是不用逐欄人工標註而教整串标准文字。用途明确，不否定整合数据流。',
'missing_minimal_relation':'32個欄位的預測如何變成短字串，以及训练怎样利用只给整串的标签；基本合并重复/blank及兼容路径的关系只在details首次出现。',
'reader_impact':'读者知道CTC要解决什么，却只能把方法当作自动对齐名称，无法解释这里新采用的整串训练关系；19.12比较CTC独立串与核心串时更需回开折叠才能理解两种串从何来。整合主要数据流可懂，因此不是blocker。',
'mechanism':{'judgment':'needs_main_explanation','strongest_main_support':'主文说明32欄/短标签及不用逐欄标注，未给转成短串的规则。','supplement_support':q('19.6','合併相鄰重複後是 `大、〈空白〉、小`，再移除空白符號，得到 `大小`。')},
'need':{'judgment':'sufficient','basis':q('19.6','訓練只給整串標準文字，不要求人工逐欄標出哪裡屬於哪個字。'),'inference':'入口保留多欄影像但标签为整串短字，须处理长度/位置对应；无需作者重教全面时序算法。'},
'minimal_repair':'在主文CTC段加入一小句/例：欄位可重复字或留blank，先合并相邻重复再去blank得到短串；训练提高所有能合成标准串的路径总概率。完整路径求和公式、重复字例及解码最优性仍留折叠。',
'cost_and_scope':'约2句及一个大小短例，段落级，无须移动章节、改目標或重训。',
'discovery_timing':'after_main_notes_seal_during_assigned_details_read','discovery_source':'independent assigned source only; no peer/old issue prompt',
'original_main_judgment':'主文封存原判认为CTC完整对齐推导可待补充，未列该基本机制缺口；原判保留未修改。','unverified':'未外查原CTC论文/运行CTC；此项只判必要解释位置，不指控CTC技术错。'
},{
'id':'continuity-integration-NFKC-name','page_id':'20.10','type':'名称补充','severity':'optional','rewrite_scale':'paragraph',
'quote':q('20.10','NFKC可把全形Ａ換成A，並不把臺改成台。'),
'already_taught':'已给全形/半形转换例、不做臺台/繁簡转换、先约定空白标点及留原样，读者能理解此页CER与exact用途。',
'missing_minimal_relation':'缺的是名称全称及中文名；没有缺当前需要的归一化对象/作用关系。',
'reader_impact':'首次见到缩写者不易把术语与后续能力卡/操作规约名对应；属于小幅定位收益，不影响判断臺/台是否逐字相同。',
'mechanism':{'judgment':'sufficient','basis':q('20.10','NFKC可把全形Ａ換成A，並不把臺改成台。')},
'need':{'judgment':'sufficient','basis':q('20.10','核對前先約定哪些整理允許。')},
'minimal_repair':'第一次出现加Unicode兼容正規化（Normalization Form KC，NFKC）名称即可，不扩写全部Unicode转换表。',
'cost_and_scope':'短括号，段落级；20.13沿用不需重复。','discovery_timing':'main_read_before_main_notes_seal','discovery_source':'independent assigned source only; no peer/old issue prompt','unverified':'全书较早是否已展开该名称未穷尽检索；此为本页首次读路线上可选名称帮助，不据此判全书从未介绍。'
}]
report=json.loads(json.dumps(main));report['at']=datetime.now(timezone.utc).isoformat();report['kind']='source-independent continuity initial; AI-assisted, not human students';report['main_notes_seal']=main_seal;report['main_notes_preserved']=True;report['issues']=issues;report['extras_notes']={'path':str(extras_path),'sha256':sha(extras_path),'pages':extras}
for page in report['pages']:
 page['main_sealed_original_source_judgment']=page['source_judgment'];page['figures_sha256']=P[page['page_id']]['figures_sha256'];page['unverified']=[page['limits_and_unverified']]
 page['supplement_delta']=next(e for e in extras if e['page_id']==page['page_id'])
 page['issues']=[i['id'] for i in issues if i['page_id']==page['page_id']]
 if page['page_id']=='19.6':
  page['source_judgment']='sufficient_integration_flow_with_required_CTC_main_mechanism_repair'
  page['checks'].append({'focus':'CTC基本对齐关系补读后位置复核','mechanism':issues[0]['mechanism'],'need':issues[0]['need'],'preserves_main_judgment':True,'discovery_timing':issues[0]['discovery_timing']})
report['strong_pass_rechecks']=[
 {'page_id':'19.2','claim':'MoE选择/参数容量与公平比较','strongest_support':'四选二仍存全部且与Dense每位置两FFN/一FFN不同，实际训练目标也不同，因此真实成品对照不作架构归因。','basis':q('19.2','兩條路線的最終訓練目標與選定歷程也不同，不能把得分差直接歸因於 MoE。'),'mechanism':'sufficient','need':'sufficient'},
 {'page_id':'19.4','claim':'新段与resume不是仅两个文件名','strongest_support':'具体说明权重外optimizer/RNG/sampler/计数恢复及总steps；目的和最低操作各自有正文。','basis':q('19.4','新段步數、token計數歸零'),'mechanism':'sufficient','need':'sufficient'},
 {'page_id':'19.6','claim':'soft bridge与最终生成分工','strongest_support':'完整候选程度和内容交核心，18.1只存第一名丢候选差异可正常推得保留资讯作用；head错/最终对例说明评分不同，不证明soft优于hard。','basis':q('19.6','這個公開語音例的回答正確，獨立分類頭卻沒有選對類別，兩件事不能混稱通過。'),'mechanism':'sufficient','need':'sufficient','limit':'CTC另一独立机制仍为问题，不能被此有效理由包办。'},
 {'page_id':'19.7','claim':'可执行与正确参数与读回三层','strongest_support':'错抄26为25仍可真算，神经两次/手工raw分别明说；主文给实际边界与用途。','basis':q('19.7','若模型把26抄成25，請求仍可能合法，工具也會正確計算25×16，整個任務卻已偏離原題。'),'mechanism':'sufficient','need':'sufficient'},
 {'page_id':'20.6','claim':'更新机制不代省参数或用途改善','strongest_support':'14>12主动限制局部例，rank1=7才省；独立副本前后与step教更新边界。','basis':q('20.6','14反而比原W的3×4=12多，不能用它示範省參數。'),'mechanism':'sufficient','need':'sufficient'},
 {'page_id':'20.8','claim':'无需完整综合公式才能理解候选拒绝','strongest_support':'各语音路不得退步，正确稿4→2已失败，加EOS截断即可拒；四列非完整选版得分声明清楚。','basis':q('20.8','正確逐字稿與真正ASR這兩條語音聊天路線，各自都不能比底座少答對。'),'mechanism':'sufficient','need':'sufficient'},
 {'page_id':'natural-v4-data','claim':'本机代码/data revision/manifest配对','strongest_support':'表和完整指纹明确分别固定什么，list/verify/download作用分开，允许不同commit但同manifest核对；非任意约定靠猜。','basis':q('natural-v4-data','本頁的資料版本指定從哪一次Git提交取得清單與資料包，並不切換本機程式。'),'mechanism':'sufficient','need':'sufficient'},
 {'page_id':'natural-v4-training','claim':'最近进度/固定候选/选定配置状态','strongest_support':'pending来源completed_steps，过1039不重复要求，train/resumed候选各路径明说；验证三配置非叠加。','basis':q('natural-v4-training','例如1,039步在原目錄、2,077步在續訓目錄，就只替換第二份路徑。'),'mechanism':'sufficient','need':'sufficient'}
]
report['main_route_conclusion']='主文整合/方法需要/比较及状态规则基本成立；补读后另核出CTC基本字串对齐机制只在折叠，需段落级补充。NFKC名称为可选。原主文判断已封存。'
report['coverage']={'assigned_pages':30,'actually_read_main_pages':M['groups']['integration']['pages'],'actually_read_extras_pages':M['groups']['integration']['pages'],'all_main_read':True,'all_assigned_details_read':True,'detail_blocks_read':sum(e['details_count'] for e in extras),'full_page_incremental_blind_test':False,'prerequisites_read':29,'main_standalone_strong_pass_pages':29,'required_issues':1,'optional_issues':1,'unverified_issue_candidates':0}
report['unknown_scope']=['未独立追查/执行正文runtime与原始评分事实；数字仅核解释及边界。','未重新安装、下载、生成、重建、训练或测资源。','除19.6三图及20.7对应context外，未验其余页面完整桌面/手机排版与交互。','29先备文字真正读完，先备图没有作为证据实看；全书较早术语展开未穷尽检查。']
report['rewrite_assessment']={'paragraph':1,'optional_paragraph':1,'page':0,'chapter':0,'order':0,'cumulative_burden':'在此组未见章级依赖倒置或跨页连续猜测；必要修补限CTC一段。'}
report['visual_scope']['original_render_fingerprints']={}
for f in sorted((B/'renders').glob('*.png')):
 if f.stem.rsplit('-',1)[0] in report['visual_scope']['observations']:
  report['visual_scope']['original_render_fingerprints'][str(f)]=sha(f)
report['visual_scope']['actual_page_capture_fingerprints']={str(B/'page-captures'/f'{pid}-{w}-context-{n}.png'):sha(B/'page-captures'/f'{pid}-{w}-context-{n}.png') for pid,ns in [('19.6',range(3)),('20.7',range(1))] for n in ns for w in [1280,390]}
for page in report['pages']:
 assert sha(Path(P[page['page_id']]['snapshot']))==page['source_sha256']
assert sha(main_path)==main_seal['sha256']
p=B/'reports/continuity-integration-initial.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
seal={'reviewer':'/root/continuity_integration','role':'continuity','group':'integration','sealed_at':datetime.now(timezone.utc).isoformat(),'path':str(p),'sha256':sha(p),'main_notes_path':str(main_path),'main_notes_sha256':sha(main_path),'extras_notes_sha256':sha(extras_path),'criteria_sha256':M['criteria_sha256'],'coverage':report['coverage'],'independence':'No reports by others, synthesis, notes, traces, old reviews or human checks read before this seal. Own main preserved.','status':'complete initial; awaiting cross-review task'}
s=B/'reports/continuity-integration-seal.json';s.write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(p),'seal':str(s),'sha256':seal['sha256'],'main_sha256':seal['main_notes_sha256'],'counts':report['coverage'],'render_views':len(report['visual_scope']['original_render_fingerprints']),'page_context_views':len(report['visual_scope']['actual_page_capture_fingerprints'])},ensure_ascii=False))
