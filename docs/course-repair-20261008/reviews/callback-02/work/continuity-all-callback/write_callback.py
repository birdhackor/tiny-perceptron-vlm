import json, hashlib, re, sys
from pathlib import Path
from datetime import datetime, timezone

C=Path('docs/course-repair-20261008/reviews/callback-02')
B=Path('docs/course-repair-20261008/reviews/freeze-01')
W=C/'work/continuity-all-callback'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
M=json.loads((C/'manifest.json').read_text())
BM=json.loads((B/'manifest.json').read_text());BP={p['page_id']:p for p in BM['inventory']['pages']}
main=json.loads((W/'main-understanding.json').read_text())
main_seal=json.loads((W/'main-understanding.seal.json').read_text())
assert sha(W/'main-understanding.json')==main_seal['sha256']
gate=json.loads((C/'checks/reader-callbacks-all-sealed.json').read_text())
assert gate['manifest_sha256']==sha(C/'manifest.json') and len(gate['records'])==7
assert sorted(gate['full_read_estimate_page_ids'])==sorted(p['page_id'] for p in M['pages'])
preserved={
 B/'reports/continuity-extensions-main-notes.json':'dad0c2790feafb5c9014f5f3232af1b07b946f3fb9306a0b37337d014d77e879',
 B/'reports/continuity-extensions-initial.json':'9927c7bee0eb70de327bad4537e2c5fed2e1f6693029a34eb2a92aa0abd47fe8',
 B/'reports/continuity-extensions-initial.seal.json':'813a75fdf9d165dc792228ed1afce495c1e938d082465d27bf4448ffd6f6d270',
 B/'reports/cross-extensions.json':'b9f03cbf77be711365bf69dab58c2d41512fddd3dab07fd4207a692abb0aad68',
 B/'reports/cross-extensions.seal.json':'e4b5894147c9e69acd30b6975e3917c493c082388e8832c23b3c9c692f2b907d',
}
for p,h in preserved.items(): assert sha(p)==h
checks=['site-config-followup.json','site-config-followup-completed.json','root-ui-visual-final.json','curriculum-scroll-path-final.json',
        'repair-ui-captures-final.json','page-layout-summary.json']
ui={n:json.loads((C/'checks'/n).read_text()) for n in checks}
assert ui['root-ui-visual-final.json']['source_and_config_identity']['callback_manifest_sha256']==sha(C/'manifest.json')
assert sha(Path('zensical.toml'))=='81ae2eb91cae75202afe975356c3a098ffe24b0f659d1216328ffb73d745f94a'
old_cfg=(B/'freeze/original/zensical.toml').read_bytes()
assert old_cfg.replace(b'  "navigation.top",\n',b'')==Path('zensical.toml').read_bytes()

report={
 'reviewer':'/root/repair_continuity_extensions',
 'role':'原extensions銜接／累積負擔審閱者，擴至指定十頁的知情局部continuity-all callback',
 'original_group':'extensions','callback_scope':'all ten pages in C manifest','at':datetime.now(timezone.utc).isoformat(),
 'new_blind_source_initial':False,
 'gate':{'path':'checks/reader-callbacks-all-sealed.json','sha256':sha(C/'checks/reader-callbacks-all-sealed.json'),'at':gate['at']},
 'manifest_sha256':sha(C/'manifest.json'),
 'instruction':{'path':'/workspace/work/tutorial-repair-20261008/callback-evidence-instructions.md','sha256':sha(Path('/workspace/work/tutorial-repair-20261008/callback-evidence-instructions.md'))},
 'criteria_used_unchanged':{k:M['criteria_unchanged'][k] for k in ['SKILL.md','references/review-protocol.md','references/calibration.md']},
 'known_B_reports_preserved':{str(p.relative_to(B)):h for p,h in preserved.items()},
 'reading_sequence':{
  'main':'完整十頁去details正文逐段實讀；training 1–170、171–285、286–397，其他九頁全部主文，無截斷。先保存主文理解及seal。',
  'main_understanding':{'path':'work/continuity-all-callback/main-understanding.json','sha256':main_seal['sha256'],'seal_sha256':sha(W/'main-understanding.seal.json')},
  'after_main':'全部15個details完整讀取：2.5、5.4、9.6、13.3、17.14、19.12各1；training9；其餘3頁0。必要前文與圖另讀。',
  'new_prerequisites':json.loads((W/'prerequisite-delivery.json').read_text())+[{'page_id':'13.6','source_sha256':BP['13.6']['source_sha256'],'scope':'main全文及唯一details，核13.3兩支版本定位'}],
  'necessary_details':'19.6讀字入口的第一details實讀：逐欄最高分→先合併相鄰重複→移除空白；未開19.6其他素材區。',
  'reused_prerequisites':json.loads((W/'reused-prerequisites.json').read_text()),
  'diff_limits':'完整C正文／选读理解已保存後，才用B同頁原source與C比較定位已知修改；比較不是通過依據，不聲稱本人曾盲讀另外七組目標頁。',
  'unread_peer_limits':'本輪未讀七位reader callback判斷或估時正文，僅確認其封存gate。未讀作者紀錄、root教材裁定、舊audit／問題清單；只依派工允許的root UI evidence。'
 },
 'pages':[], 'figure_evidence':[], 'ui_followup':{},
 'new_necessary_findings':[], 'remaining_optional_findings':[],
 'unverified_scopes':[],
}

changes={
'2.5':('原來表中9篇／1篇與平均代價的計分單位未在這一句分開；本人不是B該頁的獨立發現者。',
 '表中的平均代價將全部下一字與結束目標的代價合在一起平均；每篇短文提供的計分位置數可能不同。',
 '9／1篇提供切分和材料範圍，代價加總／平均則以有效下一字與EOS位置為單位；不是每篇等權平均。多窗口同時改參數，完整正文與選讀已保留1篇驗證／非普遍最優限制。',
 '一個計量句，釐清原表而無新增算法。'),
'5.4':('原有Adam尺度計算已有機制，這次要追能否說明原需要／選用理由而非從兩梯度猜哪步合適。',
 '單靠這兩個梯度，還不能判哪個步子合適。…若縮小共同學習率才讓某些格不跨太大，另一些格卻幾乎不動…',
 '共同LR面對各格歷史尺度有取捨，依各格尺度調步幅是目標；是否較合適仍需結果比較。第一步同大小只是本例，不推Adam必勝。1.12的減更新量與負梯度、後續mhat決定方向已連。',
 '兩句目的與限定，降低由數值相同猜優勢的負擔。選讀只把長訓練另列，並非閱讀先備。'),
'9.6':('原17題3/3與13/14數字需各自對應觀察事件，不把未拒絕等同完成。',
 '混合版在拒絕題出現拒絕句3/3，正常題完整答案ID匹配13/14。',
 '正文mask先分兩個2題组；實測則3拒絕題只搜固定短句、14正常含澄清者核完整內容ID。選讀交代17題各责任與EOS另驗，9.4給可信情境。加法別卷，不能混成拒絕分母或一般安全。',
 '在原句把兩個分子命名；未新增分類器、規則或訓練前置。'),
'13.3':('选讀原「兩支」需定位實際模型版本；來源計分约定與不同答案身份也須保留。',
 '13.6的兩個加法偏好訓練版本各250次更新。',
 '13.2已明chosen／rejected各自输入、答案＋EOS且不再shift；13.3給mask後sum、總log與mean別量。13.6主文／選讀真正定位beta0.1與1兩版、各250步；9448是两侧有效目标的累计曝光、非9448不同答案。該回讀只核既有兩支身份，不要求先完成長訓練。',
 '新增精確回鏈和名稱，无新數值或效果保證。'),
'17.14':('求導比較的hard建立步驟，以及6/10、4/10究竟驗何事，須能沿主文解釋。',
 'hard先用detach()斷開舊求導關係、clone()複製相同數值，再用requires_grad_()開啟這份新張量的求導記錄。…10道留出的顏色、形狀與音高屬性問答，按完整回答內容ID匹配判答對。',
 'hard的新求導記錄與x/STE分開；detach項不追使前向還原／反向外x，短程式仍沒有step。17.7支援每列scale、存碼反量化浮點；選讀區分正式逐列QAT與手算共用scale、350更新／39348曝光、10題69目標、5題36目標。6/10與4/10才是逐題完整內容結果；梯度、NLL、EOS與部署logit0差均不是能力保證。',
 '三個原本在代碼中的方法解釋、一句任務判準和NLL括注；減少查實作／猜測，不加入長配方必讀。'),
'19.12':('兩格位置的能力列須對應實際要求，不能擴成所有商品關係問答；其他原分母／控制限定也同時覆核。',
 '回答左／右／上／下指定格位的商品類別。',
 '由19.3家族／记录单位、19.6公開格位／接頭到本页要求，可以讀成依指定格位答商品類別，非任意关系理解。完整表3662题＋選讀4類72題=3734；控制组是同卷重組，回答改變不等改對。CTC逐欄解碼全串324/324與核心252/324、281/324別量；先合併再去blank的19.6必要選讀可說清该入口檢查。30錄音非人數、固定猜120/360非新跑模型、兩版未全驗收及20章不同起點都保留。',
 '只收窄一句能力列；CTC等已教前文的定向回讀有明確當前用途，没有要求讀完整實作／全原始檔才懂此表。'),
'20.10':('CER按預測變回原文的方向敘述，替換例應與漏字插入例一致。',
 '台換成臺需要一次替換，參考5字。',
 '依既定方向預測台→參考臺；漏北要向預測插入北，分母始終原文5個Unicode字元。20.9依任務判可見事實／語意；本節原樣轉寫另須exact/CER，NFKC不繁簡換字、空參考和無字圖另判。例子同方向无新增規約或材料宣称。',
 '交換兩字的方向，減少讀者自行修正例句，不新增概念。'),
'training':('追本人B cross的joint選用目的、temperature0、CE-OPT-1；全文11路線與父權重也重新讀完。',
 '這是一組示範設定，效果仍以留出題核對。…temperature 0在本工具中表示每步直接選最高分候選…T.10教師需先完成其中的…配方。',
 '三個freeze範圍已有定義，新句說明joint讓接頭／首末文字塊共同適應兩輸入的示範用途，沒稱projector不可用或partial必優。1.15已教argmax/greedy但原只講正T，新句把CLI零值明接該規則，固定比較用途可解。三summary明選T10的条件必做，九details实读仍能定位sft/style/moe六檔；CUDA完整與mini各有來源，runner不補依賴。其餘T1家族、T2通路非更新、T4有效目标、T5独立目标、T8同宽非同预算、T9纯张量bytes、T10架构vs教师信号、T11版本/分母/原记录均維持边界。',
 '一個目的句、一個接口句、三个条件式summary；不強迫未選T10者先跑完整實驗，也不要求重复名稱算法。'),
'readme':('追本人B cross鏡像錯承諾，並核中文同類取包措辭及所有入口／版本。',
 '各版本的顯卡、驅動與套件配套選擇見環境說明。Colab操作另見W.1。…先用list查看名稱，再用asset NAME下載、核對並解包。',
 '已读且同SHA的environment确实给设备/驱动/配套，W1确实给CPU登录、先工具/全部执行。镜像承诺删掉而没假造指南。list与asset两动作及NAME替换可接T1，不把清单当取包。仍分19自训有限v2、旧局部档、20上游、推论vs完整续训，Colab/MPS公开限制原样保留。',
 '导航分两句，取包加下一条与替换说明；增加的操作是原承诺必需动作，不是新训练前置。'),
'readme-en':('追本人B cross英语list被称choose/unpack的问题；不是另一名独立英语首读。',
 'List available snapshots with --list, then download, verify and unpack the chosen archive with --asset NAME, replacing NAME with a listed asset name.',
 '两个命令明确分查看与取包，NAME从列表选，与中文/T1例可接；不先要求读未验外链才知道下一步。其余完整正文保持版本、平台、训练/推论/能力限定；没由新命令宣称本机已下载或一般能力通過。',
 '一个两步骤句，补原操作缺口；不另写英文下载长教程。'),
}
extra={
'2.5':'唯一區完整讀；T3固定比较与同表旧曲线横轴是窗口、非时间，参数同時改且最後2篇別用途。',
'5.4':'唯一區完整讀；只提供長配方另入口，明說短例不需先長訓。',
'9.6':'唯一區完整讀；3拒绝/14正常的细分、固定句匹配/完整ID/EOS分开；普通加法不混17题。',
'13.3':'唯一區完整讀；EOS计入、角色/问题/PAD不计，beta兩版各250步和重复目标曝光量已定位。',
'17.14':'唯一區完整讀；正式逐列规则/部署/FP32主权重/两种packed同350步骤/有效目标与逐题分母可对照；原论文及长执行未做。',
'19.12':'唯一區完整讀；4類工具边界总72题仍来自完整3734，不是另卷。',
'20.10':'無details，全文就是所读主文。',
'training':'九區197行全部讀，1–110和111–197分段；三个T10summary清楚条件前置，其他独立实验/格式/许可证/模态输入/效率实测界限没有变成额外主线必做。',
'readme':'無details。',
'readme-en':'無details。',
}
for i,p in enumerate(M['pages']):
 pid=p['page_id'];source=Path(p['snapshot']);assert sha(source)==p['source_sha256']
 assert source.read_text().strip() in Path(p['source']).read_text()
 for fig,h in p['figures_sha256'].items():assert sha(Path(fig))==h
 n=main['pages'][i];assert n['page_id']==pid
 old_problem,quote,reason,burden=changes[pid]
 report['pages'].append({'page_id':pid,'source_sha256':p['source_sha256'],'previous_source_sha256':p['previous_source_sha256'],
 'figures_sha256':p['figures_sha256'],'main_note_reference':{'path':'work/continuity-all-callback/main-understanding.json','sha256':main_seal['sha256'],'json_pointer':'/pages/'+str(i)},
 'original_problem_tracking':old_problem,'current_quote':quote,
 'Q2_mechanism_property':n['mechanism'],'Q2_need_use':n['need'],
 'operation_example_boundaries':n['operation_boundary'],
 'after_extras_and_prerequisites':reason,'actual_extras_read':extra[pid],
 'added_burden_review':burden,'judgment':'原關係在目前完整來源／必要前文可接；沒有辨認到新必要語義缺口。',
 'limits':'这是知情局部來源理解；未在此執行模型／安裝／資料下載／長訓練／原資料重算，不把來源說法等同本人技術驗真。'})

stems=['rewrite-02-visible-window','rewrite-17-qat-flow','rewrite-01-character-ids','window_training']
for stem in stems:
 svg=Path('course/figures')/(stem+'.svg')
 pngs={str((B/'renders'/f'{stem}-{w}.png').relative_to(B)):sha(B/'renders'/f'{stem}-{w}.png') for w in [640,360]}
 report['figure_evidence'].append({'figure':str(svg),'sha256':sha(svg),'actual_original_svg_text_read':True,'actual_isolated_pngs_seen':pngs,
  'interpretation':{'rewrite-02-visible-window':'两答案圆/方，灰字未输入，1/3窗口同输入，5才含红蓝；可见非学好。',
   'rewrite-17-qat-flow':'浮点主权重→临时量化/还原→STE近似backward→step更新→最终打包；小例未step。',
   'rewrite-01-character-ids':'貓看狗，/狗看貓。上下ID3/2/1/4、1/2/3/0按原句顺序查回，不按数值大小释义。',
   'window_training':'横轴1/3/5窗口与833/1345/1857参数，不是更新进度；蓝9篇/橙1篇平均下一项代价，当前主文解释有效目标加权。'}[stem],
  'limits':'原SVG与孤立640/360視驗不等原位整頁通過；必要前文的其余图未宣稱看過。'})

viewed=[]
for r in ui['root-ui-visual-final.json']['actual_capture_views']:
 path=Path(r['capture']);assert sha(path)==r['sha256']
 viewed.append({'path':str(path.relative_to(C)),'sha256':r['sha256'],'actually_seen_by_this_reviewer':True,
                'interaction_owner':'root，本人只以view_image看静态文件，未操作浏览器'})
report['ui_followup']={
 'evidence_files_read':{f'checks/{n}':sha(C/'checks'/n) for n in checks},
 'config_identity':{'previous_sha256':sha(B/'freeze/original/zensical.toml'),'current_sha256':sha(Path('zensical.toml')),
  'actual_byte_comparison':'只移除一行navigation.top；navigation/path/footer、toc.follow、search/code-copy保留。'},
 'navigation_semantics':'中央回頂是可選定位控制，不是理解當前圖／進下一課必需前提；現有章節導航、目錄、一般捲動仍可定位。移除避免原位上捲遮圖，無新增教材或先備概念。',
 'root_actual_interaction_evidence':'390x844与1280x800均先下捲到图，再上捲90px，记录topButtons=[]；T10 SFT点链接后必要时手动开，style/MoE手动开；手机MoE CODE实际scrollLeft252、client311/scroll563。',
 'own_actual_static_views':viewed,
 'own_static_interpretation':'curriculum两尺寸所示图框没有中央回顶覆盖，手机X可读前文mask与Y计分位置可追；右下TOC仍近边字。手机MoE目标summary与第二命令experiment moe/device cuda可读，第一命令末端近复制按钮，不泛称所有命令已全字可见。19.12手机组分母/内容正确/回答改变三栏可对照，但一个视窗不是全部五列。5.4主文用途/保留验证及17.14hard/十题ID限定在未开details的长捕捉图可见。',
 'root_mechanical_summary':'相关10页两尺寸零全页溢出／缺图，只是机械获取；root只记6张实看，非全10页真人现场首读。build-info executed_cpu_outputs=false，reading_time_pages325为AI估时覆盖，不是新模型执行或真人耗时。',
 'limits':['本人没有浏览器捲动、点击、横捲、复制或真实Colab动作。','不能证明锚点会自动开closed details；root实际路线允许手动开。','只支持已测中央回頂遮挡解决；右下TOC仍可能盖局部边字，不称旧W1或全部浮动UI都解决。','site-config-followup原pending保留当时bytes，新增completed记录和实际root UI证据只补已测局部；本人未重新build/check_site或验证所有页。'],
}
report['own_added_source_checks']=[
 {'page_id':'2.5','origin':'知情回读后本人纸上材料，不是原模型实测／独立首发现',
  'material':'短文A有2有效目标、总NLL2；B有6有效目标、总NLL18（均包括各自EOS）。',
  'prediction_reason':'按位置平均(2+18)/(2+6)=2.5；每篇平均后等权会是(1+3)/2=2。当前新增句能区分两个量，不用把9/1篇当计分位置数。','executed':False},
 {'page_id':'20.10','origin':'知情回读后本人纸上材料，不是原来源既有例或模型证据',
  'material':'参考今天去臺北，预测今天去台北呀。',
  'prediction_reason':'预测台替换臺、删除尾呀共2编辑，分母参考5，CER0.4；意思相近也不满足原样转写。统一预测→原文方向与主文漏字插入例相容。','executed':False},
]
report['tracked_B_extensions_items']=[
 {'id':'extensions-training-joint-freeze-purpose','status':'来源语义缺口已补：实际选定示范目的／留出检验，未声称必须或优于projector。'},
 {'id':'extensions-training-temperature-zero','status':'来源操作约定已明示0逐步最高分及1.15greedy；本人未核CLI实现新事实，技术吻合范围另由技术角色验证。'},
 {'id':'extensions-readme-environment-mirror-route','status':'错镜像承诺已删；environment配套与W1 Colab明确分路，现有必要前文同SHA且可接。'},
 {'id':'extensions-readme-en-list-unpack','status':'两个操作及NAME替换已明示；中文同步澄清，未由本人下载/解包。'},
 {'id':'CE-OPT-1','status':'可选改善已落實：三个summary对选T10者标明条件必做；运行及自动开details未验。'},
 {'id':'extensions-training-names','status':'原低优先可选留存；training roles已教，2.5已给MLP全名、7.1已给SFT，不要求每页重讲算法。'},
]
report['remaining_optional_findings']=[{'id':'extensions-training-names','severity':'選讀改善','scope':'仅直接跳training者名字联系的低优先收益；不影响当前机制／指标／操作关系，不升为必修。'}]
report['unverified_scopes']=[
 '未进行任何训练／推論／环境安装／数据下载／原记录重计／全部kernel，不认证新增文字中的实际接口、数字或历史真实性；本角色只核来源关系。',
 '模型效果、QAT原正式数据、DPO目标曝光／beta版本、3734评评分/控制CTC324数字真实性由技术查证，本人未借root总结冒充核原数据。',
 '所有10页源语义完整实读；原位网页只实看6张root提供静态捕捉图，不是全10页或全287页完整互动验收。',
 '必要新前文只读列明main、19.6 CTC必要details、13.6唯一details；未称前文其他图/选读/外部论文全部验过。',
 'B原initial/cross及main notes保留原bytes；此次受提示已知范围扩大不冒称其他组原角色或新独立首读，未改教学来源／配置／估时。',
]
report['overall']='十頁原需要、用途、計分／轉寫／操作前提，在本次完整實讀範圍均連得上；未辨認到未解的必要銜接語義項或新必要負擔。原四項extensions必要語義缺口已補，CE-OPT-1可選標示已落實，名稱補充仍僅可選。中央回頂原位問題依root實際兩尺寸捲動與本人靜態視讀有限解除；不泛稱全部浮動UI／平台／模型效果通過。'
report['counts']={'pages':10,'assigned_details_read':15,'new_prerequisite_mains_read':8,'necessary_prerequisite_details_read':2,'isolated_figures_seen_640_360':4,'root_capture_files_actually_viewed':6,'new_necessary_findings':0}
assert [p['page_id'] for p in report['pages']]==[p['page_id'] for p in M['pages']]
for p,h in preserved.items():assert sha(p)==h
assert sha(W/'main-understanding.json')==main_seal['sha256']
sys.path.insert(0,str(B/'work/continuity-extensions/pydeps'))
from opencc import OpenCC
out=C/'reports/continuity-all-callback.json';seal=C/'reports/continuity-all-callback.seal.json'
assert not out.exists() and not seal.exists()
out.write_text(OpenCC('s2t').convert(json.dumps(report,ensure_ascii=False,indent=2))+'\n')
receipt={'reviewer':report['reviewer'],'role':report['role'],'at':datetime.now(timezone.utc).isoformat(),'path':str(out.relative_to(C)),'sha256':sha(out),'manifest_sha256':sha(C/'manifest.json'),'pages':10,'kind':'informed related-page continuity callback, never a new blind initial'}
seal.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
for p,h in preserved.items():assert sha(p)==h
print(json.dumps({'report':str(out),'sha256':sha(out),'seal':str(seal),'seal_sha256':sha(seal),'counts':report['counts'],'B_original_report_bytes_preserved':True},ensure_ascii=False,indent=2))
