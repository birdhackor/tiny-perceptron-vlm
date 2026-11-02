# Kev／Omni 主線修訂的三輪審閱與增量發布

本文件是協調與工具安排，不是審閱結果。使用者已批准主線改寫與第19章新產品，並再次要求：易讀易理解、正確性、連貫性三輪獨立審閱；理解所需程式碼須直接在教材內，不依賴開啟Colab。依據為repo的[編輯規約](../editorial-guide.md)、[技術規約](../technical-review-guide.md)及[clear-tutorial流程](../../.agents/skills/clear-tutorial/references/review-protocol.md)。

## 凍結與範圍

本輪開始的active是`docs/course-review-20261027-tokenizer/assembly-all-01/manifest.json`，policy為`tutorial_incremental_v2`，共351個canonical閱讀頁。這是2026-10-29 P1的數量，新增後必須重算，不能把313份逐課Notebook、編號小節數及canonical頁數混用。

本輪guard固定認證該manifest的SHA `22280b0c680f4011f96f9d1281878de559ebaecae5561953f023f823724d5e43`。只有這個實際path＋SHA、且讀入內容一致的既有manifest可用legacy豁免；新candidate缺requirements即拒絕。comparison必須等於此accepted manifest綁定的source freeze，不能自選current，也不能刪page_ids縮範圍。固定manifest是本輪已知批准起點；activation後仍讀此不可變檔，不依賴active當時恰好指向哪一輪。

P1作者工具`/root/omni_p1_review_setup`與其他作者均列入新manifest的`author_tasks`；作者、reader、technical、continuity不得混任。工具準備者不能自任自己的獨立工具技術審閱。新增檔案或新模型實作被教材依賴時，也列入implementation freeze及獨立工具／證據審查，不能只保留舊工具pass。

先完成全部新增材料，再改完既有教材與入口，之後開始reader。正式GPU工程在三輪後；已批准的有界P1基礎設施例外另依預算規約，不能拿來提早訓練。

以下命令從repo根目錄執行。輸出每次用新路徑；工具拒絕覆寫，不能以新hash替代重讀。

```bash
uv run --extra cpu python scripts/omni_revision/review_setup.py inventory --output data/omni-review-01.json
uv run --extra cpu python scripts/build_course.py
uv run --extra cpu python scripts/omni_revision/review_setup.py prepare --output docs/course-revision-20261029-omni/reviews/wave-01 --wave omni-mainline-01 --author /root/omni_p1_review_setup --author /root/omni_p1_infrastructure
```

第一個命令比較目前頁面與active source freeze，並列出技術報告的候選依賴漂移。第二個命令先同步由正文產生的Notebook。第三個命令重用`freeze_sources.py`建立快照，保留舊assignment與舊證據；新增／有改動頁的三個角色寫成`mode: evidence_required`。它只產生不可發布的草稿，不產生review/pass，不變更active pointer。P1 smoke曾在`data/omni-review-setup-smoke`執行，正式輪不要沿用該測試wave。

正文／圖改動需新的reader、technical及continuity。即使本節文字未變，舊technical report所引用整份章檔／實作變動，仍可能要fresh technical；逐筆核對來源與既有author-patch／historical sidecar後決定。掃描只能找候選，不能自動判新增重審數。P1基線掃描已有9筆合法歷史漂移候選，但三道既有發布閘門全過，不能誤報成9個新回歸。

`section-composite`的舊gate固定比較**20261008原始baseline_freeze**，不是目前active的source_freeze；它要求introduction與該原基線相同，並覆蓋所有相對該原基線變更的節。現行first-steps、training、glossary的introduction已不相同，因此即使本輪不再改introduction，仍不能使用此模式。course目前intro雖相同，所需集合仍含R.1–R.4，不能只交本輪delta。

**本輪R／W／T／G整檔頁統一不用section-composite：**只要正文／圖改動，就對該整頁內所有編號節各派真fresh technical，再另派不同fresh task獨立核整頁及前言，以既有`mode: review`對接canonical gate。逐節證據保存在另有SHA的sidecar，由helper額外核對，不把整頁一份pass冒稱每節fresh；下方有可執行的凍結、自查與收件流程。

新章導讀、首頁、README、course、curriculum、glossary、training、validation等按實際canonical inventory處理，不只審新增正文。`notebook`路徑改變即使bytes相同，也會列在scope-diff，不能歸為fully_unchanged；相同內容的路徑搬移由協調者另作下載／路線裁定，不自動聲稱內容需重新首讀。未變頁可保留舊證據，仍需銜接角色裁定依賴是否受影響；不能稱這輪重新首讀全書。

## 第一輪：易讀、易理解

協調者先真正spawn reader，使用`fork_turns="none"`。只給讀者背景、通用判準、當次plan與cursor指令；不給作者plan、預期答案、概念清單、舊問題、模型結果或技術解答。允許一位reader依序讀多頁，每頁都須保留逐段記錄。必要前文由實際路線指定，不把名稱曾出現當成已懂。

由協調者用實際task名稱與真實main/context頁ID產生閱讀計畫：

```bash
uv run --extra cpu python scripts/omni_revision/review_setup.py reading-plan --freeze docs/course-revision-20261029-omni/reviews/wave-01/source-freeze.json --output docs/course-revision-20261029-omni/reviews/reader-group-a-plan.json --reviewer /root/REAL_READER_TASK --page ACTUAL_PAGE_ID --context-page ACTUAL_PREREQUISITE_ID
```

`REAL_READER_TASK`及頁ID是命令格式位置，必須換成實際派工，不是可提交的假身分。`reading-plan`不證明task真的被spawn，也不能取代完整作者排除名單的核對。reader只執行下列舊cursor工具，不自行cat尚未開放的全文：

```bash
uv run --extra cpu python docs/course-revision-20261016/tools/reader.py supplement ACTUAL_PREREQUISITE_ID --reviewer /root/REAL_READER_TASK --plan ACTUAL_PLAN.json --freeze ACTUAL_FREEZE.json
uv run --extra cpu python docs/course-revision-20261016/tools/reader.py start ACTUAL_PAGE_ID --reviewer /root/REAL_READER_TASK --plan ACTUAL_PLAN.json --freeze ACTUAL_FREEZE.json
uv run --extra cpu python docs/course-revision-20261016/tools/reader.py next ACTUAL_SESSION --checkpoint ACTUAL_NOTE.json
```

先保存當前段的理解才`next`。每段note有`unit_index`與五欄觀察：`understanding`、`materials_and_labels`、`expected_change`、`confusion_and_quote`、`missing_visuals`；`concept_checks`逐必要概念記`concept / identity / mechanism / need / usage / example_scope / source_quote / verdict`。note由reader寫，cursor補來源與時間，不能作者批次代填。必要圖仍在原位置，明記實際看過的圖與未驗證範圍；正文讀完前不開選讀補洞。

每頁original report記`page_id`、`source_sha256`、`figures_sha256`、`reader_summary`、`missing_visuals`、`checks`、`issues`、`verdict`、AI閱讀時間範圍，以及`session`、`session_evidence.opening/notes`。報告根記`reviewer_task`、`source_freeze`、`reading_plan`。AI估時不是人類學生實測。

**本輪增加的實質檢查：**讀者能否僅由正文追蹤所有理解所需程式碼、輸入、關鍵輸出與理由？遇到未展開的專案helper、只指向Colab的計算或缺失的關鍵行，按實際理解影響列問題。一般`torch`基本API可引用已教前文；完整下載、長訓練與發布操作可另連操作指引，不能把機制本身藏在操作入口。

reader本人先跑格式自查，不代表自動證明理解：

```bash
uv run --extra cpu python docs/course-review-20261019/tools/check_reader_original.py ACTUAL_READER_ORIGINAL.json
```

保留original，不以後文修掉早期卡點。必要問題由作者修正後，原reader讀完整受影響單位與圖／過渡，另寫current及basis；新freeze與新session並存。任何note的概念判定不是`已足夠`，最終還要reader自己的`checkpoint_closure`，按note SHA及concept index逐項處理；作者不能把疑問清空。原reader不可用時，保存實際不可用工具回應，另派fresh reader完整重審。

## 第二輪：正確性、數字與證據

第一輪全部必要問題收束、實際收件及reader seal完成，才開始technical。**每節一個fresh task，`fork_turns="none"`；不能一個task改名或包多節。**提供當節凍結來源、明確前置、可查原始來源及實驗證據位置，不給作者推理或預期pass。

依`docs/technical-review-guide.md`的schema保存：根欄位`schema_version=1 / review_stage=technical / lesson_id / source / reviewer_task / reviewer_context=fresh / source_sha256 / figure_sha256 / verdict`；`claims`區分concept、numeric、software、empirical，逐項附location、status、evidence、scope。五項checks為`factual_accuracy / numeric_verification / figure_consistency / source_verification / limitations`，各有理由及claim IDs。沒有圖或數字可具理由NA，含命令／程式的頁不得整頁假NA。

原論文／官方程式需實際閱讀版本與定位，不能拿作者摘要當權威。numeric的手算與executed分開，真正執行的artifact記command、result、environment和digest。新模組或GPU成果更要分清forward、梯度、更新、訓練結果及最終泛化。本文示意值不能包裝實測；成功下載不能代表模型能力。必要程式須核對教材內嵌版本，不能只執行外部模組便宣稱正文正確。

```bash
uv run --extra cpu python docs/course-review-20261019/tools/check_technical_report.py ACTUAL_PAGE_ID ACTUAL_TECHNICAL_REPORT.json --freeze ACTUAL_FREEZE.json --reader-task /root/REAL_READER_TASK
```

此指令讀凍結原文並用既有`_validate`驗schema／來源／artifact／身分，不替代真正技術判斷。verdict為revise的原始報告會因未解主張失敗，應保存；修正後由同一位reviewer親自重核，另存不可變current報告。

### whole-file頁內R／W／T／G的實際路徑

canonical freeze只有course、first-steps、training、glossary整頁ID，不能把R.1等直接交上面的舊自查命令。先從**同一canonical凍結稿**切出輔助section freeze；按prepare輸出的`whole_files_needing_numbered_reviews`選實際頁，以下四頁是命令格式示範：

```bash
uv run --extra cpu python scripts/omni_revision/review_setup.py freeze-numbered --freeze ACTUAL_CANONICAL_FREEZE.json --output ACTUAL_NUMBERED_FREEZE.json --page course --page first-steps --page training --page glossary
uv run --extra cpu python docs/course-review-20261019/tools/check_technical_report.py R.1 ACTUAL_R1_REPORT.json --freeze ACTUAL_NUMBERED_FREEZE.json --reader-task /root/REAL_READER_TASK
```

輔助freeze的pages含R.1、W.1、T.1、G.1等精確切片，source仍是原教材路徑，圖只含該節引用；它明標`auxiliary_numbered_technical_freeze_not_canonical`，**不能替換正式canonical source_freeze或增加公開頁數**。每個編號節仍有不同真task、完整技術schema、原始報告及必要修後current。舊自查API現在能實際找到該節並跑原`_validate`。

協調者將真實交件引用組成`numbered_section_reviews` sidecar：根有`canonical_source_freeze`及`section_freeze`兩個path＋SHA reference，`sections`每項有`lesson_id / report / original_report / original_section_freeze`。original_section_freeze是原判真正讀過的auxiliary freeze；它又綁原canonical稿，helper核精確切片、原圖快照及原判schema／claims／checks，不接受只有owner／ID的空殼。原判可以revise及有未決claim，不能因此改成pass。所有新original/current技術報告另記真實`created_at`（含時區）。

每個entry另有`original_collection`與`current_collection`，各記真實`wave / collection_role / received_at`；對照seal的`collection_index`必須有唯一path＋SHA＋同received_at，建立時間與派工窗均在該實際stage內，original→current不倒置。original對應首次technical派工；current若在後續wave或continuity收件，另記真正`current_dispatch_window`，不能回填早期seal。sidecar只是收件索引，不含作者生成的pass。若原始逐節報告有問題，同reviewer修後current另有`original_issue_resolutions`，逐項用原報告的`json_pointer`及具體`resolution`保存個人收束；無問題時清單為空，原問題永不刪除。

在候選manifest將sidecar以path＋SHA放入`numbered_section_reviews`；prepare已記`numbered_section_requirements.comparison_source_freeze`與page_ids，helper依本輪固定認證的accepted baseline重新算範圍，要求變動整頁的**所有**編號節，不只新節。先自查逐節schema、原判保留及派工；intake前也先按下段將逐節問題索引接入主manifest：

```bash
uv run --extra cpu python scripts/omni_revision/review_setup.py check-numbered --manifest ACTUAL_MANIFEST_DRAFT.json --intake --output ACTUAL_NUMBERED_INTAKE.json
```

intake只回`schema_checked_collection_pending`，不表示階段收件／整頁審閱已完成。另派整頁fresh technical reviewer，先獨立核完整正文及前言、保存初判，再核sidecar和逐節結果；不能先讀他人答案而聲稱獨立發現。其canonical報告lesson_id是course等整頁ID，以正常claim／source／artifact schema做自己的判斷；修後current的artifacts用`source_snapshot`類型綁定sidecar path＋SHA，旁附它實際執行的檢查紀錄。collector proof除舊欄位外加`original_source_freeze`（原canonical freeze）、`original_collection / current_collection`及必要callback派工窗，同樣驗原判來源與實際收件；sidecar不反向引用整頁報告，避免循環hash。

原判核完整schema、sources、claims、checks及其實際閱讀的正文／圖freeze，容許誠實的`revise`及舊版本，不能用僅有header的空殼替代。正文舊全文只由原freeze逐byte重建。若原判真的引用後來修改的程式，whole proof或section entry可加`original_source_snapshots`清單；每項僅有`source`與`snapshot`（path＋SHA）。保存bytes的digest必須等於不可變原報告`sources`內同一個`repository_code`路徑已登記的SHA；拒絕未登記路徑、不同bytes、重複版本及任意`earlier_files`。這只認證原判讀過的舊版本，當前報告仍須通過現行來源的完整核對。

`assignments`同時保留各編號節和整頁collector的獨立technical task、lesson_id、fresh context及真實wave／時間窗；所有逐節original/current、整頁original/current都實際收入該technical seal。技術階段關閉後再跑：

```bash
uv run --extra cpu python scripts/omni_revision/review_setup.py check-numbered --manifest ACTUAL_MANIFEST_DRAFT.json --output ACTUAL_NUMBERED_SEALED.json
```

sealed檢查將各節交給原`_validate`，以全候選canonical與逐節報告的共同owner清單拒絕task跨節重用、reader／作者混任，並核精確切片、圖、完整原判、原始問題收束、唯一collection_index receipts及建立／派工／收件先後。collector本身亦用原`_validate`核完整整頁，再核它對sidecar的SHA綁定。不能因section ID非canonical而略過問題或偽造canonical身分。

### 逐節問題接入同一主dispositions

舊gate的fixed verification只接受canonical頁，不能把T.1硬當training整頁。此處使用舊gate已有的`additional_issue_inventories`，不修改checker或把section偽裝canonical。協調者從真報告產生索引：

```bash
uv run --extra cpu python scripts/omni_revision/review_setup.py index-numbered-issues --bundle ACTUAL_NUMBERED_BUNDLE.json --output ACTUAL_SECTION_ISSUES.json --author /root
```

索引明標作者處理，`reports`完整列每份逐節original/current path＋SHA；`issues`每項只定位真報告SHA＋原issue pointer、lesson_id及canonical parent，不捏造問題或判定。helper重新計算索引，要求所有原判、修後判定、必要及可選問題一對一完整。將索引reference放入sidecar的`issue_inventory`，也放主manifest的`issue_reports`；在`additional_issue_inventories`加入`{"report": 索引reference, "json_pointer": "/issues"}`。索引不反向綁sidecar，無循環hash。

主代理在原有**同一份dispositions**逐項審理索引`/issues/0`等，照舊填source_support、learning_benefit、change_burden、reason、final_necessity、action及affected_pages（包含真正parent）。必要問題只准fixed；可選仍須記keep／defer／reject／fixed的理解收益及負擔，不能省略。

每項另有`section_verification`指向真正同section reviewer的current report：原問題指向其`/original_issue_resolutions/N`，current已解問題指向其原issues節點。新guard核精確SHA／pointer及同reviewer的當前來源／圖；空泛resolution不能取代先前的整份current `_validate`。fixed項的舊`verification`則引用真正fresh整頁collector的current canonical report，由它驗parent內此修正與上下文。這兩份證據分工清楚，不冒充原section reviewer已做整頁獨立審閱。

sealed guard驗索引／主處置一對一、主代理四項理由、必要問題已修、同reviewer實核及canonical collector，並重用舊`check_dispositions`核完整主表及既有historical fix鏈。歷史原報告／原處置／不可用收據／recheck sidecar全保留，不以新索引刪掉舊問題。intake只核索引接入，尚未封存的主處置與收件明記pending，不冒稱完成。

最終`review_setup.py gate`先執行既有all gate，再額外執行此sealed檢查；任何一項失敗都不能activation。既有gate仍以整頁`mode: review`核canonical來源、真正整頁報告及三輪；新增sidecar額外保住逐編號節fresh契約。新工具及此額外檢查也需獨立工具稽核。

### 多代歷史修正與原連貫性角色不可用

實際整合發現兩項既有格式無法表示的情況：已接受的上一輪本身引用更早的歷史修正，以及原連貫性審閱者已不可呼叫。不能改舊報告、舊處置、SHA或降級必要性來消除漂移。2026-10-29新增`historical_fix_rechecks_v2.py`及其測試，並對`incremental_revision.py`作必要的格式分派與工具範圍增補；原入口精確副本保存在本輪`frozen-implementation/`。這是新的工具修改，不能再稱入口檔完全未改，須經獨立工具原判、修後複核及真baseline replay，才可用於發布。

schema=1仍走原v1路徑，保留原有拒絕規則。明示schema=2的sidecar使用`chains`列出各自有path＋SHA的schema=1子鏈；每條都由真active pointer歷史、其已完成gate、原freeze及原處置認證。子鏈不能再巢狀schema=2、不能重複承接同一個問題／verification，也不能出現未使用的承接項。各代原處置檔與hash保留，主表沿用項須與原處置內容相等，並非把舊verified版本改成最近一版。

v2子鏈可另含`kind: continuity`。其`full_review / original_review / original_source_freeze / appendix / collection / unavailability`必須來自同一位新fresh連貫性審閱者，實際派工及原判、當前全文、後續appendix均在正確continuity階段收件；先保存原判，才給舊問題。原判本身須有完整schema、原freeze的實際正文／圖SHA、非空checks及真實原句，不接受任意source SHA；報告建立時間及實際收件時間各自符合original→current→appendix。受改頁仍必須有同wave完整fresh reader與完整fresh technical，重用現行trace／closure與`_validate`；technical原判另以其`original_source_freeze`及必要的已認證程式快照核完整結構，三階段不能重疊。跨兩頁的舊依賴問題必須核完整`affected_pages`，不因其中一頁未改而省掉它的當前連貫性實讀。

appendix的`historical_issue_rechecks`逐項保留`disposition_ref / original_report / json_pointer / historical_fix_verification / issue_id / necessity`，另記`status: still_fixed / verdict: pass / reason / current_evidence`及`pages`。`pages`以所有affected page ID為key，每項有當前`source_sha256 / figures_sha256 / source_quote / reason`；引文須實際出現在當前凍結正文。原角色不可用仍需真followup回應與空listing，並保留原來的fixed處置。新工具只核證據契約，不能替代本人閱讀或語意判斷；其完整檔案與測試由工具審閱凍結，未獨立收束前不啟用。

## 第三輪：連貫性與頁面

technical必要問題收束、實際收件及technical seal後，另派fresh continuity reviewer；不得是作者或reader／technical。沿實際前文→本節→下一步建立依賴，檢查新增概念首次需要位置、程式表示轉換、段落動機、主文／選讀與累積負擔，不只重述舊報告。

continuity report每頁留來源／圖SHA、verdict、原句依據、四題關係、問題與修後核對；全書dependency audit列當時完整`inventory_page_ids`，另明記真正讀過的路線、只掃描的頁及未驗範圍，不能把351-ID清單當作全書實讀。獨立保存初判後，才交叉核對其他角色與舊問題。

桌面及手機實際看圖文、公式、折疊、下載／Colab入口；截圖存Git外，manifest只留SHA與取得方法。主代理另抽看未報問題的通過圖，記實看範圍。建置成功不代表圖看過，也不以「Colab裡看得到」替代正文可理解。

## 真實派工、收件與組裝

沿用`tutorial_incremental_v2`，不改checker標準。`assignments.tasks`每筆記真實`task / role / context=fresh / wave / dispatch_window`，reader／continuity有`pages`，technical有單一`lesson_id`。派工時間窗落在實際階段內，不回填虛構時間。

每wave的三份seal依序建立，均有`role / status=completed / started_at / completed_at / report_sha256s`，另保留`collection_index`的實際收件時間。階段不能重疊；未收報告不能先封存再塞SHA。若後來有修稿／結果回填，新增callback wave與真實收件，不改過去的seal。

每頁proof的`review`引用current、original、json pointer及wave；reader另有basis、sessions及必要closure。R／W／T／G整頁採上述canonical review＋逐節sidecar，其餘不任意更換舊模式。未改頁保留先前proof，不把現行review任意改成只允許20261008原基線的`inherit`。manifest還需完整criteria／implementation freeze、baseline_freeze/supplements、tool_audit、issue_reports、dispositions/dispositions_collection及最後dependency_audit。

`dispositions`由主代理逐項裁定所有必要與可選發現，保留source_support、learning_benefit、change_burden與reason；必要問題只能fixed，且有原reviewer的實際核對。保留舊處置與historical_fix_rechecks鏈；新來源使舊verification過期時，依現有historical sidecar補真實fresh完整審閱／appendix及實際原task不可用收據，不能換舊hash或刪歷史問題。9筆P1掃描候選不自動成新問題。

開放收件時可跑局部preflight；只核draft列出的頁／角色，pending seal不算通過發布：

```bash
uv run --extra cpu python docs/course-review-20261019/tools/preflight.py ACTUAL_MANIFEST_DRAFT.json --mode intake --stage reader
uv run --extra cpu python docs/course-review-20261019/tools/issue_inventory.py ACTUAL_MANIFEST_DRAFT.json --mode intake --stage all
uv run --extra cpu python scripts/omni_revision/review_setup.py gate --manifest ACTUAL_MANIFEST_DRAFT.json --stage all --output ACTUAL_GATE_RESULT.json
```

最後一個命令呼叫既有`incremental_revision.audit_manifest`，包括baseline replay、canonical覆蓋、Notebook／source／figure凍結、角色身分／時序、工具、問題處置及依賴審閱；technical／all另加逐節sidecar檢查。新wrapper不放寬任何既有條件。草稿沒新審閱必須失敗；不可把`evidence_required`改成pass。

完整all gate真實成功後才另存當時active pointer，建立含previous_pointer及actual_completed_gate的新pointer，再跑三道正式檢查。新helper不執行activation：

```bash
uv run --extra cpu python scripts/check_course_reviews.py
uv run --extra cpu python scripts/check_technical_reviews.py
uv run --extra cpu python docs/review-tools/check_review_round.py
```

## 訓練結果回填及發布

正式工程後，新增數字、圖表、練習、能力界線或下載行為都算新來源。凍結新的完整閱讀單位，新增reader→technical→continuity callback；保存舊首讀、真結果的原始證據與每次修後覆核。不能只改原報告SHA，或把看過研究答案的顧問當首次讀者。

重建Notebook與閱讀時間，執行適當離線／kernel／網站檢查，再核對公開build-info revision、實際頁面及模型／資料下載digest。三輪審閱、程式執行、模型能力和發布成功分別報告；AI審閱不稱真人學生測試。本文件及P1工具自查不構成任何一輪獨立審閱。

未發布的本機預覽提供實際同節Notebook下載及Colab首頁的手動上傳說明，不把舊commit的Notebook當作新版直連。正式發布使用`--require-reading-times`時，exporter必須先以Git object逐一核對全部當前Notebook與40碼commit的bytes完全相等，才輸出GitHub Colab直連；`check_site.py`同時核建置模式及每頁入口。這是連結／版本查核，不代表已登入或實際執行Colab runtime。新的`baseline-supplements-v2.json`保留原四項還原，另用舊雜湊表認證的原封`check_site.py`、`build_visuals.py`保存bytes重放舊基準，共六項，不修改任何歷史判定。v1五項清單與它的實跑紀錄保留。

圖解重建另核對兩類既有來源：實測圖必須符合原發布清單的SHA-256與長度；缺內嵌標題／描述的既有手寫圖，必須有`course/figures/source-metadata.json`中逐圖明列且符合當前bytes的標題與描述。這是正式圖解來源metadata，並非豁免審閱或以檔名猜圖意。未知且缺metadata的圖、修改後未重新綁定的圖仍須失敗；重建必須保留已審圖解的bytes，不能覆回舊標註。工具新scope包含生成器、metadata、索引、固定實測圖清單及相應測試。
