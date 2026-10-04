from pathlib import Path
import hashlib
import json

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/reader-reviews/artifacts/natural-v4-supplemental/student'
REL=str(OUT.relative_to(ROOT))
def rel(name): return REL+'/'+name
def read(name): return json.loads((OUT/name).read_text())
source=read('source-receipt.json')
prereqs=read('prerequisite-source-receipt.json')
details=read('browser/details-receipt.json')
raw_pages=read('browser/http-raw-receipt.json')
routes=read('browser/routes-receipt.json')
for item in [source,*prereqs,*details['figures']]:
    src=item.get('path',item.get('source'))
    digest=item.get('sha256',item.get('source_sha256'))
    assert hashlib.sha256((ROOT/src).read_bytes()).hexdigest()==digest, src+' changed after reading'

figure_notes=[
    '親自查看頁內圖，並以900px寬Chromium獨立渲染再次完整查看。打字箭頭直接入歷史＋問題；錄音先經聽寫才會合；實際圖片另入同一聊天底座。框內可選微調不表示本版一定啟用；最下方明示沒有語音合成。',
    '親自查看完整圖：上方人踩在車上、下方人站在車旁，兩幅都有同樣人與腳踏車。因此名詞吻合和活動、關係吻合是不同判尺。圖明示人工概念圖，不能當自然照片實測。',
    '親自查看完整圖：購物單第一行牛奶、第二行麵包，向下箭頭指定先上後下；綠框與橘框的字元相同，但行序相反。圖下方另說多欄與直排要另外規定讀法。',
    '親自查看完整图：同一來源問題分成正確逐字稿與真錄音兩路，右路多經ASR與原始辨識稿；兩路進同一聊天模型，再比較各自回答，且保持權重、歷史與照片相同。這清楚分開聽錯與回錯。'
]
figures=[]
for f,note in zip(details['figures'],figure_notes):
    f=dict(f)
    f['personally_rendered']=True
    f['personally_viewed']=True
    f['view_tool']='functions tools.view_image; first full unobscured PNG view after browser_details.py first render; renewed screenshot view where relevant'
    f['own_view_notes']=note
    figures.append(f)

issues=[
    {
      'id':'student-entry-guide-destination',
      'severity':'blocking for the assigned current preview navigation',
      'location':{'source':'course/chapters/20.md','lines':[43,90],'page':'20.2.html','visible_label':'學生操作指引','actual_destination':'https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/natural-assistant/v4/STUDENT.md'},
      'finding':'從首頁實際點第20章，再點20.2，最後點正文的學生操作指引，瀏覽器離開本機current export，落到GitHub main。實際畫面是404 page not found，頁名是File not found · GitHub。兩處正文href都指main；同頁側欄操作指引卻有正確本機natural-v4-student.html。學生要查容量、GPU與離線時，正文路線不能帶到所指的現行指引。',
      'proposed_correction':'讓20.2正文兩處學生操作指引在學生export內落到同版本natural-v4-student.html；重新從首頁走完整路線，確認指南標題與七個步驟確實可讀。',
      'evidence':[rel('browser/details-receipt.json'),rel('browser/20.2.html.http-raw.html'),rel('browser/20.2-student-link-external-destination.png'),rel('browser/20.2-student-link-external-destination.html')],
      'status':'unresolved; no source or exporter edit by this reader',
      'limits':'這是實際preview的閱讀路線finding；不把外部main視為現行canonical，不宣稱發布後結果。'
    },
    {
      'id':'entry-execution-output-expectation',
      'severity':'blocking for an unqualified pass of the assigned current preview entry scope',
      'location':{'source':'course/README.md','lines':[7,78],'page':'course.html','sections':['先讀一節，不必先安裝','小實驗、訓練與譬喻怎麼配合']},
      'finding':'閱讀指南說程式下方已有這個版本實際執行的結果，並說網站短程式下的結果是在CPU上驗證；首頁又明說本建置尚未附執行結果。本人讀到的20.1、20.3、20.9–20.13是程式後接正文的預期輸出解釋，沒有實際執行輸出區塊。預期值本身可理解，但入口對當前讀者看得到什麼給出不同承諾。',
      'proposed_correction':'在current preview入口明示尚未附獨立CPU輸出，或待真正執行完成並把真結果放入新export後，再由本讀者重看入口與必要概念頁。未完成的計畫或執行不能當作已關閉證據。',
      'evidence':[rel('browser/index.html.http-raw.html'),rel('browser/course.html.http-raw.html'),rel('browser/index.html.png'),rel('browser/course-section-0.png'),rel('browser/course-section-3.png'),rel('browser/20.1-code-with-prose.png')],
      'status':'unresolved in this inspected unpublished preview',
      'limits':'不質疑來源裡操作實測數字；這一finding只處理本次preview的可見結果與閱讀期待。'
    }
]

command_explanations=[
 {'location':'STUDENT.md:11–20','commands':'git --version; GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout ... tiny-perceptron-natural; cd tiny-perceptron-natural; GIT_LFS_SKIP_SMUDGE=1 git checkout --detach 59a1eda4ed7b6e8609892ec2b9013c821ac93e69; python3.12 --version','inputs':'指定GitHub專案、新資料夾名稱、固定提交、Python3.12命令。','expected_outputs_and_use':'先確認Git有安裝，取得程式與教材但跳過大型訓練資料；切到新根目錄並固定實測所配的程式／清單；最後看到Python3.12.x。不是下載完整模型。','executed':False},
 {'location':'STUDENT.md:28–40','commands':'python3.12 -m venv .venv-natural; .venv-natural/bin/python -m pip install --upgrade pip; CPU torch/torchvision pip command; pip install -r requirements-natural.txt','inputs':'新環境路徑、torch2.8.0、torchvision0.23.0、CPU官方wheel來源、固定需求檔。','expected_outputs_and_use':'產生獨立環境，再裝CPU套件與其餘配套；每次明寫此環境Python，避免和教材小實驗的.venv混裝。安裝訊息不代表助理能回答。','executed':False},
 {'location':'STUDENT.md:44–51','commands':'nvidia-smi; CUDA cu128 torch/torchvision pip command; pip install -r requirements-natural.txt; python -c import torch and print CUDA/bfloat16 support','inputs':'本人電腦顯示卡與驅動、CUDA12.8配套、相同需求檔。','expected_outputs_and_use':'顯示卡與驅動列表，之後印GPU可用與bfloat16支援兩個True/False；兩項True時採預設GPU啟動，GPU可用但bf16不支援則用float16。CPU/GPU安装擇一，語音聽寫仍走CPU。','executed':False},
 {'location':'STUDENT.md:75–90','commands':'fetch_natural_release.py --manifest docs/natural-assistant/v4/public-release.json --list; same script --output checkpoints/natural-assistant/release-v4; same --verify','inputs':'固定公開零件清單與下載目錄。','expected_outputs_and_use':'--list列出配對；下載先得到兩份小型配置文件並逐檔核對大小與SHA-256；--verify只重查配置目錄。此時不是已下載官方完整模型，也不是品質驗收。失敗可重跑；內容不同時保留原目錄並另用新路徑，之後--output一起改。','executed':False},
 {'location':'STUDENT.md:96–124','commands':'same manifest/output with --serve --device cpu --dtype float32; --serve --device cuda; --serve --device cuda --dtype float16','inputs':'已核對的配置目錄、CPU或CUDA裝置、數值形式。','expected_outputs_and_use':'第一次服務啟動取得圖文底座並載入，等終端機服務開始；bf16不支援的GPU先關原服務再改float16。第一次按語音辨識才另下載載入ASR，不由一次文字成功推成語音已齊備。','executed':False},
 {'location':'STUDENT.md:108–146','commands':'browser open http://127.0.0.1:8766/; terminal Ctrl+C; append --local-files-only to original serve command','inputs':'同一臺電腦的本機服務、已完整試過兩模型的快取。','expected_outputs_and_use':'8766是本機助理網址，須讓終端機持續開著；文字回應後才加照片／語音。Ctrl+C結束並刪本次上傳，下載快取保留；離線旗標只讀本機快取，缺檔不再補抓。','executed':False},
]

report={
 'report_kind':'independent fresh whole-document readability only, plus explicitly assigned actual unpublished browser preview/navigation',
 'reviewer_identity':'/root/v4_review_coordinator/reader_guide_student',
 'coordinator_identity':'/root/v4_review_coordinator',
 'actual_task':(OUT/'task.txt').read_text(),
 'contract':{'path':'outputs/natural-v4/review-plan/supplemental-reader-contract.md','read_before_canonical':True},
 'fresh_context':{'role':'數學能力足夠的高中或大學生，第一次接觸此專案；以文字說明與基本算术理解流程，不預設大型模型實作知識。','prior_reviews_read':False,'drafts_read':False,'author_notes_or_expected_verdict_read':False,'subagents_spawned':False,'evidence_directory_existed_before_this_assignment':False},
 'assigned_scope':{'canonical_complete':'docs/natural-assistant/v4/STUDENT.md','frozen_expected_sha256':'6732af69f9505ef1842c42d10eac86e60b1a54fb0c62a6c26a67798b8d1ddb7c','preview_page':'natural-v4-student.html','preview_base':'http://127.0.0.1:8769/','additional_scope':'student entry navigation and actually necessary linked concept/limit pages','publication_state':'unpublished local preview','reading_time_stage':'not begun; no reading-time estimates inserted'},
 'verdict':'revise',
 'verdict_by_scope':{'canonical_STUDENT_complete_readability':'pass','current_browser_STUDENT_content_readability':'pass','assigned_current_student_entry_and_necessary_navigation':'revise'},
 'verdict_reason':'七步學生操作正文可完整理解；目前實際入口仍有正文指南落到GitHub404，以及可見CPU結果承諾不一致。這兩項actual preview問題尚未關閉，因此不給整個派送scope無條件pass。',
 'complete_documents':[
   {**source,'all_lines_read':True,'read_order':1,'own_summary':'先取得固定程式並確認Python3.12；另建.venv-natural；CPU或NVIDIA路線擇一；看清單，再下載與核對兩份配置；啟動時才取圖文底座，第一次語音再取辨識器；同機8766依序試文字、照片、語音原稿更正與送出；保留歷史追問或清除新對話；Ctrl+C終止，完整缓存后可離線。正文同時限制格式、容量、平台與實測數字用途。','understanding':'能在不執行安裝與模型的前提下，說清每一步的輸入、產物、成功觀察及停止／改路條件。'},
   {**prereqs[1],'all_lines_read':True,'read_order':2,'own_summary':'這是固定配置零件表：selected_variant為base、adapter_parameters為0；指定Qwen與Whisper官方revision、需求與程式配對、runtime設定；公開下載清單只有README.md與release-provenance.json。reviewed等欄位是清單資料，沒有被當作本人review通過理由。','understanding':'兩份配置文件不是兩個官方模型權重，兩種數量與用途能分清。'},
   {**prereqs[0],'all_lines_read':True,'read_order':3,'own_summary':'閱讀指南把由接字到多模態的路線拆成可跳讀的問題，允許先讀概念再動手；第20章沿用成熟底座，資料／候選微調／驗收與第19章從零小模型分開。操作、訓練、上游能力及示範程式各有不同證據範圍。','understanding':'第20章入口與前置表可理解；R.1/R.4對目前網站已有CPU輸出的描述與這次preview狀態不一致，保留issue。'}
 ],
 'actual_read_order':['契約；STUDENT.md全150行','Chromium student頁全正文與七個步驟畫面','public-release.json全66行','chapter20來源20.1/20.3/20.9–20.13必要段落','Chromium首頁、閱讀指南、第20章導讀正文','course/README.md全96行；student頁rendered全文再對照','Chromium必要概念頁全正文及actual clicks；4張SVG頁內實際view','chapter20來源20.2全節；Chromium實際入口鏈index→chapter20→20.2→學生操作指引','4 SVG独立完整Chromium render/view；raw HTTP頁面body保存；長命令实际水平捲動','optional DATA/TRAINING只讀可見標題與引言，確認是未來改資料／GPU微調的目的地，沒有把它們當本次完整文檔review'],
 'necessary_prerequisite_scope':{
   'source_receipt':rel('prerequisite-source-receipt.json'),
   'chapter20_backing_file':prereqs[2],
   'sections_read_completely':['20.1','20.2','20.3','20.9','20.10','20.11','20.12','20.13'],
   'why_necessary':'20.1解釋輸入會合；20.2是實際學生入口鏈；20.3解釋權重與運行空間不同；20.9–20.11給照片／逐字／行序判尺；20.12定位聽寫與回答；20.13給STUDENT反覆指向的能力限制。',
   'additional_recursive_prerequisites_read':[],
   'why_no_recursive_expansion':'已讀段落自身解釋ASR、底座、參數、GiB、字元錯誤、行序與完整回答，足以理解這份使用指引；更早的7/11/12/19章連結屬概念延伸，未構成本次操作理解缺口。20.4貓例的可見描述也已在20.9寫清楚，不需開啟訓練示範圖才能理解判尺。',
   'optional_link_scope':'DATA與TRAINING只作轉入未來資料／GPU新實驗的目的地語意檢視；未完整讀或review其canonical文件。操作核對及其他技術provenance連結的href已記錄，未讀作者證據結論。Git／PyTorch安裝外鏈在本次existing環境且不執行安裝的scope裡不需另作完整讀。'
 },
 'own_workflow_model_and_data_counts':{
   'models':'兩個官方模型：圖文Qwen聊天核心，Whisper聽寫站；本版不開LoRA，不重新訓練ASR，回答為文字。',
   'configuration_vs_models':'兩份公開配置文件共9262bytes＝說明與來源記錄；兩個官方快照是23個檔、5889111977bytes約5.4847GiB。這不是本review者下載結果，亦不是最低磁碟需求。',
   'memory':'13496104KiB約12.8709GiB是文件報告一次Linux服務最大常駐記憶體，不含瀏覽器；float32每個數32bits=4bytes；20.3的2127532032個底座參數乘4僅算權重，不能當總RAM。',
   'duration_numbers':'24.24秒起動、四次聊天3.42/42.87/36.58/37.11秒、18.03秒首次ASR、170.33秒完整工作，各量測範圍正文有分開；本人不重新量測、不估readingtime。',
   'input_limits':'照片PNG/JPEG/WebP、單張静態、8MiB约8.39MB且1600萬像素；中文錄音WAV/FLAC/MP3/OGG、30秒且8MiB。',
   'ability_card':'42張照片的42描述與84問答共用圖；有無字18題、單區抄寫10題、讀序3題、文字13題是不同用途，不能加成彼此獨立情境總分。4個語音問句分正確文字與實ASR兩路；22段真人聽寫CER用字元編輯次數作分子，字元數作分母，不是聊天通過率。'
 },
 'five_checks':[
   {'check':'background_and_links','assessment':'正文所需背景可由具體20.x連結補足，無需先訓練前章；actual入口有指南404與preview輸出期待矛盾，故連結／入口此項需修。','status':'revise','issue_ids':[x['id'] for x in issues]},
   {'check':'terminology','assessment':'底座、聽寫、逐字稿、SHA-256、float32、CPU/GPU、cu128、快取、互操作執行緒都有用途或本地解釋；20.3補足2B與GiB，20.10/20.12補足CER。LoRA不被當成本版啟用配置。','status':'pass'},
   {'check':'examples_symbols_and_numbers','assessment':'我能分清bytes與Unicode字元、32bits與4bytes、MiB/MB、權重量與RAM；照片名詞／活動、臺/台與意思、牛奶/麵包行序、漏不字條件反轉各示例都能自述。表中實測不能推出最低設備或固定等待。','status':'pass'},
   {'check':'programs_and_commands','assessment':'全10個browser程式區塊已完整讀；長命令的CODE元素可水平捲動，實際捲到右端看見--verify、--serve、--device與--dtype。每條命令目的與輸出可解釋；依限制一條安裝、下載、模型serve或GPU命令都未執行。','status':'pass for readability; runtime unverified'},
   {'check':'exercises_and_self_checks','assessment':'學生頁沒有另設抽象數學習題；它的任務就是依次文字回應、照片問題與追問、逐字核對、聽写原稿更正再送出、文字與語音同條件比較、新對話與停止。每一步觀察到的成功／未成功都有可說明的對象；必要概念小程式練習可預測，未實跑。','status':'pass','not_applicable':'未要求重做最後考卷、GPU訓練或benchmark；不把能輸出回答當品質通過。'}
 ],
 'commands_read_not_executed':command_explanations,
 'concept_programs_read_not_executed':[
   {'section':'20.1','own_prediction':'相同手寫問題接相同history，True與招牌問題；更改recognized_question就False。不是ASR或讀圖執行。'},
   {'section':'20.3','own_prediction':'參數乘4或2，約7.93/3.96GiB，只有權重數值；328128參數小數兩位GiB顯0.0是捨入，不是零權重。'},
   {'section':'20.9','own_prediction':'兩張手寫卡物件都True，活動只有卡0True、卡1False。不是自然照片能力分數。'},
   {'section':'20.10','own_prediction':'臺換台：False、1、5、0.2；少北仍以原5字作分母。'},
   {'section':'20.11','own_prediction':'Counter相同True，整串False，splitlines顯牛奶/麵包交換；去換行失去指定分行結構。'},
   {'section':'20.12','own_prediction':'漏不字：1次編輯、9參考字元、11.1%、False；補回為0，但聊天品質仍要另判。'},
   {'section':'20.13','own_prediction':'不同bytes的完整hash比較False；received與expected同bytes後True，但指紋配對仍不能代替能力卡。'}
 ],
 'actual_execution':{
   'browser':'Existing .venv Playwright with /usr/bin/chromium, Chromium151.0.7922.173, headless 1440x1000; own scripts execute actual rendering, click navigation, screenshots and rendered body retrieval.',
   'successful_commands':['.venv/bin/python '+rel('browser_inspect.py'),'.venv/bin/python '+rel('browser_routes.py'),'.venv/bin/python '+rel('browser_raw.py'),'.venv/bin/python '+rel('browser_details.py'),'.venv/bin/python '+rel('browser_code.py')],
   'first_attempt_failure':'First browser_details.py rendered figures and clicked index→chapter20→20.2→external main, then timed out waiting for article h1 at the external destination. Saved original script as browser_details_first_attempt.py; corrected only own evidence script to record external title/URL without assuming an article, then reran successfully. Product files were untouched.',
   'bounded_offline_concept_CPU_examples_executed':False,
   'why_no_CPU_examples':'文字例子足以由基本算术與比較運算理解；本人不需要實跑它們才作readability判斷。預期值與真正執行證據分列。',
   'installs_model_or_data_downloads_GPU_work_training_model_service':False,
   'preview_server':'8769 is the preexisting unpublished documentation export; this reader did not build, publish or start it. 8766 model UI was not started or visited as a running assistant.',
   'source_figures_code_checkers_navigation_time_Git_environment_modified':False
 },
 'actual_browser_evidence':{
   'raw_HTTP_receipts':rel('browser/http-raw-receipt.json'),
   'raw_HTTP_pages':raw_pages,
   'DOM_and_visible_text_note':'*.http-raw.html stores original response.body bytes. Earlier *.html and student-retrieved.html store page.content DOM serialization; *.txt stores real rendered inner_text. These are deliberately distinguished.',
   'main_student':{'URL':'http://127.0.0.1:8769/natural-v4-student.html','status':200,'heading':'在自己的電腦開啟照片與語音助理','whole_body_read':True,'screens':[rel('browser/student-section-%02d.png'%i) for i in range(8)],'table_screen':rel('browser/student-measurement-table.png'),'inventory':rel('browser/student-inventory.json'),'rendered_text':rel('browser/student-article.txt')},
   'inspected_pages_and_actual_clicks':routes,
   'entry_chain':details['entry_clicks'],
   'semantic_destinations':{'index.html':'由接字學習開始，可選閱讀指南或第20章','course.html':'跳章背景與閱讀／實驗證據說明','chapter-20.html':'自然照片、讀字、語音的原理導讀與小節入口','20.1.html':'打字／聽写接同一歷史和圖片','20.2.html':'成品CPU試用步驟；正文指南實際錯到外部main','20.3.html':'小份修正不代表底座載入量變小','20.9.html':'按物件、動作、關係與畫面證據判照片','20.10.html':'逐字、異體字、CER與有無字各別判','20.11.html':'完整行序與版面規約','20.12.html':'同問題正確文字與實ASR兩站對照','20.13.html':'配置配對與用途能力卡／未測範圍','natural-v4-data.html':'下一階段自己準備資料與来源授權；只檢視引言及標題','natural-v4-training.html':'下一階段自己用GPU建立候選、驗證與最後測試；只檢視引言及標題'},
   'long_code_scroll_receipt':rel('browser/code-horizontal-scroll-receipt.json'),
   'personally_viewed_images':'All eight student section screenshots; measurement table; index page; course R.1/R.4 screenshots; 20.1 code-with-prose; external GitHub404; all four full unobscured SVG renders; representative long-code suffix screenshots were genuinely opened using view_image.',
   'scope_limits':'未檢視全書或其他分章的所有連結，未讀作者舊review；optional DATA/TRAINING完整body雖保存，但僅引言、標題與linkidentity實際閱讀，不宣稱完整readability review。'
 },
 'original_figure_map':figures,
 'STUDENT_embedded_SVGs':{'count':0,'not_applicable':'STUDENT正文與其actual頁沒有SVG；本次4張圖均来自必要20.x概念頁。'},
 'issues':issues,
 'remaining_issues':[x['id'] for x in issues],
 'nonblocking_issues':[],
 'limitations':['Readability only: no factual audit, benchmark replication or runtime correctness claim.','No install, model/data download, GPU training or model-serving command executed.','Current unpublished export had no independent CPU execution output blocks when inspected.','No new reading-time record or estimate.','External main destination identity was inspected; no external main guide assumed current or read as current.'],
 'final_current_source_check':{'all_recorded_source_and_original_figure_hashes_still_match':True,'canonical_STUDENT_sha256':source['sha256']},
 'own_notes':rel('notes.md')
}

(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(OUT/'notes.md').write_text('''# 本次自己的閱讀筆記

本人是 /root/v4_review_coordinator/reader_guide_student，以第一次接觸專案的數學基礎學生角色閱讀。先讀唯一契約，再讀現行STUDENT.md全150行，沒有讀draft、舊review或作者notes。此處沒有虛構lesson ID、其他讀者或讀時估計。

七步路線能自述：固定程式→獨立Python環境→CPU/GPU擇一→列清單→下載核對兩份配置→啟動取得圖文模型→同機文字／照片／語音依序試→保留或清除對話→停止與完整快取後離線。兩份小文件、兩個官方模型、23個快照檔與運行RAM是不同計量。ASR原稿、更正送出文本、聊天回答是不同紀錄；照片讀字要按題目逐字與行序核對，不由店名或名詞吻合宣稱完成。

Actual Chromium browser頁面、click與900px完整SVG render已保存。最早page img screenshot有sticky header遮住頂部的兩張圖，因此另作完整無遮擋圖render並真正view；hash確認拿到的是原SVG相同bytes。長Bash行在CODE元素水平捲動，右端--verify/--serve/--device/--dtype能看見。最初捲PRE沒有移動，後來檢查實際overflow容器才完成真正水平捲動；前面的未移動receipt與screens仍保存，不把它們假称成功。

本次combined scope裁決revise。STUDENT正文與其direct actual頁可讀通過；入口scope有兩项未閉合：首頁→第20章→20.2→正文學生操作指引实际去GitHub main，画面404；course入口承諾有實際CPU結果，但index明說沒有，目前概念頁只有正文預期結果。下一步可由作者修真export後再由原讀者重看，計畫或開始執行不是已閉合證據。

本人所有產品命令未執行，沒有安裝、下載模型資料、GPU／長訓練、model serve、source／Git／checker／time修改。existing browser evidence automation是真執行，20.x概念輸出是自己的理解預測，兩者沒有混在一起。source raw bytes、fullfile SHA、部分章scope、4 SVG map、actual URL與raw response body等詳細位置見report.json。
''',encoding='utf-8')
manifest=[]
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name!='evidence-manifest.json':
        data=path.read_bytes()
        manifest.append({'path':str(path.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
(OUT/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'verdict':report['verdict'],'canonical_sha256':source['sha256'],'issues':report['remaining_issues'],'report':rel('report.json'),'files_preserved':len(manifest)},ensure_ascii=False))
