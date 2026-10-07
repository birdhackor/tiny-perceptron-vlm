# Phase7 審閱基礎設施盤點

這是唯讀準備報告，不是教材審閱。基準為 `3c0f3254d4762f5ed87e77bd55e5eaf9be2ec248`；沒有首次閱讀、技術判定、銜接判定、圖像檢視、教材執行或發布。唯一寫入內容是本目錄的報告與清單。協調者已選擇真正的 grouped 批次；新增 `7.20–7.22`、`16.14` 定稿後才凍結來源並開始三輪閱讀。

## 1. 完整來源清單與範圍

[canonical-pages-baseline-321.json](canonical-pages-baseline-321.json) 列出全部 321 個 canonical 公開頁的原稿、selector、行號、來源 SHA、Git blob、必要圖 SHA，並列出 283 個 Notebook 的來源路徑與 SHA。這是原稿切片清單，沒有複製正文或大型實驗資料。首頁以 `home_introduction(index, executed_outputs=True)` 為 selector，須與正式匯出的 `--executed` 模式一致。四個完整入口頁另列其 26 個編號子節，方便對照舊 gates。

| 來源範圍 | 基準數量 | 新增後預期 |
| --- | ---: | ---: |
| lesson URL／Notebook | 283 | 287 |
| 章導讀 | 23 | 23 |
| 課程入口完整頁 | 7 | 7 |
| 技術與操作參考完整頁 | 7 | 7 |
| 首頁 | 1 | 1 |
| canonical 公開頁總數 | 321 | 325 |
| 舊編號小節 gates 的節數 | 309 | 313 |

`canonical_course` 是 313 個基準頁；`complete_site` 才是全部 321 頁。404、重複章導航及 Notebook 下載不是額外的教材正文。321 頁共對應 38 個來源檔、107 個本機圖檔；145 個來源／圖檔的 worktree bytes 已核對與基準 Git 版本一致。原 `units()` 切分約有 1378 個候選閱讀單元；這只是工具切分數，不代表教學斷點或已讀紀錄。

新增後必須重新產生完整 inventory；不能直接將四個未定稿頁塞進此基準 manifest、預填 SHA 或預填通過。新增會使首頁中的小節總數改變，因此首頁亦是新版本；07／16 的章導讀或其他導覽若改動也須納入重新閱讀。

## 2. 課文 → Notebook → 網站

| 檔案／函式 | 現有契約與用途 |
| --- | --- |
| `scripts/build_course.py:175 build()` | 按章原稿 `## <ID> <標題>` 建立 Notebook、`lesson-index.json`、`course/lessons.md`；標題只取原稿。先檢查 `lesson-contract.json` 的明列 ID 集合，缺節或多節都失敗。 |
| `scripts/build_course.py:122 notebook()` | 同一份正文拆成 Markdown／程式 cells，加入本節 bootstrap；閱讀連結由 `notebook_reading_links():86` 接到網站。 |
| `scripts/check_notebooks.py:34 run_kernel()` | 使用新 Jupyter kernel 執行 Notebook，保存執行副本；`main():53` 支援 `--lesson`、`--output`、`--workers`。 |
| `scripts/export_course.py:96 checked_notebook()` | 匯出前逐 cell 核對來源與執行副本；拒絕來源不同、未執行或 error output。這不證明執行結果的科學結論。 |
| `scripts/export_course.py:212 build()` | 整理原稿與核對後的 Notebook outputs，執行 Zensical。`DOCUMENTS:41` 明列 14 個完整入口／參考來源；`:314–326` 核對 nav 與全部公開頁集合。 |
| `scripts/reading_time.py:82 build_inventory()` | canonical 小節、独立章前言、完整文件與首頁各算一次；`lesson_slices():55` 保存原始標題及空白；`figure_hashes():67` 支援 Markdown 圖與 HTML `img`。 |
| `scripts/check_site.py:32 check()` | 核對 HTML 連結、完整下載 Notebook、Colab 入口與來源副本。 |

新內容需要修改的檔案是 `course/chapters/07.md`、`course/chapters/16.md`、`course/lesson-contract.json`、`zensical.toml`，以及必要圖與圖的生成器。再用 builder 生成 `notebooks/07/7.20.ipynb` 至 `7.22.ipynb`、`notebooks/16/16.14.ipynb`、`course/lesson-index.json`、`course/lessons.md`。保留現有 ID，不能重編既有 URL。

`docs/course-experiments/review-round-additions.json` 已保留原 248 節基準之外的 61 節新增範圍；若保留舊模式的 inventory 檢查，新四節要以原 `baseline_revision` 登記來源和新增理由。原 `review-round-baseline.json` 不改寫。新內容涉及的閱讀路線須檢查 `course/reading-time-routes.json`；路線只能列確實存在的 canonical IDs，不需要把每個新節強制加到每條路線。

CPU 小算例或梯度示範可直接放教材／獨立 helper，無須把新增知識綁成新的付費訓練工作。現有 KL helper 是 `tiny_perceptron/alignment.py:45 distillation_kl()`／`:58 distillation_loss()`；既有 teacher cache 與蒸餾 runner 在 `scripts/course_experiments/compression.py:847 run_distillation()`、`:991 run_multimodal_distillation()`。MTP 若需共享 hidden states，相關契約是 `tiny_perceptron/model.py:53 TinyLM` 的 forward 與 `loss_sum():92`。只有實際新增正式 runner／成品實測時，才需要改 `docs/course-experiments/plan.json` 及 `scripts/course_experiments/run.py:17 experiment_spec()` 的對應；本次沒有判定應採哪種模型實作。

## 3. 舊審閱 schema／gates 的邊界

| 工具 | 現有檢查 | 對新批次的限制 |
| --- | --- | --- |
| `incremental_reader.py:19 inventory()`、`:46 units()`、`:114 main()` | 309 編號節；第一節附導言；程式獨立成單元；目前單元記錄先追加到 durable JSONL，才顯示下一單元。 | 不接受任意 canonical 頁；只强制五個非空 string；不檢查最新四題、方法介紹原句對照、確實已讀前文或實際 view_image。 |
| `scripts/check_course_reviews.py:41 main()` | 本文／SVG／報告版本、五項 checks、摘要與 issues，以及每節獨立 task。 | 不涵蓋其他完整頁；同一 grouped task 被第二節使用即失敗；`checked():20` 還接受寬鬆 bool／string，不是最新理解證據 gate。 |
| `scripts/check_technical_reviews.py:212 _validate()` | source、fresh 身分、作者／讀者分離、claims、權威原始來源、永久 artifacts、分母及 checks。 | 每節 task 唯一，且只由編號節 `check():304` 建立 inventory；grouped 與完整頁必須明確另接 scope。 |
| `check_review_round.py:21 expected_inventory()`／`:37 main()` | 固定 248 節加明列新增；導言 SHA／摘要；兩輪身分分離。 | 只排除最早 baseline 的舊 reader tasks，不能據此宣稱 phase7 是最新全新批次。 |
| `coordinator_progress.py:10 PATH` | 保存派工與收件、trace 順序、正文／導言／圖版本、複查新 trace。 | 硬編 `20261005/review-progress.json`；dispatch 只寫宣告，不建立代理；task 名稱也依舊規約自組。 |
| `audit_readability_round.py:24 trace_errors()`／`:44 audit()` | 真實 unit SHA 順序、完整 checkpoints、五欄、source／intro／figure、派工與報告。 | 硬編舊 progress；full pass 範圍仍是309節，沒有最新四題 schema；不能證明看圖或看懂。 |
| `audit_factual_round.py:19 dispatch_errors()`／`:39 audit()` | 本輪 fresh 派工、收件 digest、目前證據、舊報告 backup bytes；保留 void 派工。 | `PROGRESS:12` 硬編舊路徑；309節；不能代替真實科學核實。 |
| `continuity-reviewer-instructions.md` | 要求真正順序閱讀、當節即追加理解、相關前後文與桌面／手機檢查、原讀者複查。 | 是方法文件，沒有通用全頁 coverage checker；phase6 的 closure 是本輪專用產物。 |

最新 `.agents/skills/clear-tutorial/SKILL.md` 與 `references/review-protocol.md` 還要求：首次使用／小節末四題逐項依據、每個重要新方法的身分／目的／本節步驟關係原句對照、來源狀態及推論關係、早期卡點保留、必要圖片實看、後文釐清只能追加。舊工具成功不等於這些要求已滿足。

六組不能用虛構的每節 task 去滿足舊檢查。新發布應明確選擇 `phase7_grouped` active batch，再核對真實 group owners、全部 325 頁及三輪證據。舊模式維持原有規則，用於既有歷史檢查；不得靜默把它降成任意 task 重用。

## 4. Phase6 的真實記錄方式可以沿用哪些部分

- 編號小節實際使用 `docs/reader-reviews/traces/19.3/p6_reader_19_3-39dad21d.jsonl` 等：第一行是凍結 session／身分／來源／圖，後續每次 `next` 才追加 unit_index、unit_sha256 及五項理解。原 unit 的全文留在 ignored session state，永久 trace 保存 hashes 與讀者原話，沒有把整包實驗資料複製到每個 checkpoint。
- `docs/reader-reviews/traces/public-pages-p6-a/round4-20261006T130420684929Z/` 的 manifest 只指向該輪實讀來源、trace、報告與 backup；trace 依實際小段行號記 immediate understanding。`checks.json` 明列哪些頁這輪新讀、哪些僅保留先前同 bytes 的實读，沒有把保留項目改稱新讀。
- 銜接用的 `docs/course-revision-20261006-phase6/continuity/artifacts/c/review_step.py` 將來源按標題拆分；`read` 輸出當前段，`append` 保存讀者輸入、時間、來源與段 SHA。它可借用切分／追加概念，但 `read` 本身沒有阻止跳段，且 `append` 又複製一次 page snapshot，不適合直接複製成新框架。
- 估時的 `docs/course-revision-20261006-phase6/reading-times/artifacts/a/reader.py` 逐頁展示來源與 Notebook outputs、記分鐘區間及本人理解，record 強制 assignment 中的頁序。它仍不是首次閱讀工具：`read(pid)` 可直接讀指定頁全文；部分渲染用單一960寬，不能代替最新兩寬視覺要求。
- `phase6/continuity/closure.json` 的321頁是先前真正全書讀者、phase6改動範圍及原owner callbacks 的組合；它不是「phase6六人重新首讀全部321頁」。phase7 若宣稱最新完整三輪覆蓋，所有頁應在本批次有實際記錄，不能把舊 closure 的保留頁改標成新閱讀。

上述皆是對保存機制與 metadata 的檢視，本次沒有重判舊 readers 的理解或內容結論。

## 5. 六組順序建議

[six-sequential-groups-proposal.json](six-sequential-groups-proposal.json) 保存完整且不重疊的 baseline primary_page_ids、章邊界候選前文及新增後投影。內容仍為 `not_dispatched`，沒有任何 reviewer 身分或通過判定。

| 組 | 主要範圍 | 基準頁數 | 新增後 |
| --- | --- | ---: | ---: |
| a | 首頁、閱讀指南、基礎暖身、第1–5章與導讀 | 60 | 60 |
| b | 第6–9章與導讀，含新增7.20–7.22 | 58 | 61 |
| c | 第10–12章與導讀 | 48 | 48 |
| d | 第13–16章與導讀，含新增16.14 | 57 | 58 |
| e | 第17–19章與導讀 | 44 | 44 |
| f | 第20章、A／B／C附錄與導讀、三個v4指南、其餘操作／參考入口 | 54 | 54 |

第一輪、技術輪、銜接輪分別使用新的六組 owners；同一階段保留同組 owner 複查，三階段與作者身分分離。這是分工建議，不是「a讀過的前文便自動成為b讀過」：每組開始前提供實際必要的前文原稿／圖，讓該讀者真的讀，保存原句、單元與 SHA。JSON 所列三個邊界前節只是候選，不能當全部先備知識。若某組沒有實讀所有早期前文，報告須寫明提供範圍與限制。

第20章使用成熟模型延伸入口，不能假定是第19章同一權重。完整名詞頁安排在後半組，前半讀者只依正文的精確前置連結讀當下必要條目，避免未教的名詞被預先拿來補洞。跨組邊界需有前後相鄰節實讀；銜接輪不能只讀各组中間、漏掉交接。

## 6. 最小 wrapper + active batch gate 建議

不再複製一套 review framework。只需要一個 canonical source 的分段 wrapper，以及一個 active batch 核對入口，沿用現有函式與證據 schema。

**Wrapper：**沿用 `reading_time.build_inventory/lesson_slices/chapter_introduction/figure_hashes` 建立真實 source selector，沿用 `incremental_reader.units()` 切分正文及 code fences。可提供 `start <group>`／`next --checkpoint`／`summary`；下一單元只在當前紀錄寫入原始 JSONL 後開放。每頁一個開始／結束事件，每段一個 checkpoint；session state 保留全部內容但不得讓讀者讀 state。不同於舊 numbered 模式，章前言本身是一個 canonical 頁，不需在第一節重複計數。

保留原五欄，僅加最新 skill 必要的結構：`visible_before` 的真實已讀 unit refs；首次使用／小節末的四题各題回答及每個核心主張的來源狀態、原句／圖定位、推論關係；重要新方法原句對照；當下問題 IDs、圖需求及本人 visual refs。讀者自行辨識重要新方法；工具不能預寫其概念介紹答案，不能依關鍵字替他回答。後文釐清另追加 resolution 事件，不改早期 checkpoint。工具可檢查欄位完整，判讀哪些介绍還缺仍由實讀者負責。

**Gate：**active pointer 指向新 batch manifest／來源 freeze／三輪 group reports 與 raw traces；每輪 primary scopes 的聯集須等於完整 inventory、無漏頁重複，boundary／prerequisite context 另列。核對每組真實派工紀錄、owner 與 fresh context、來源／圖／Notebook版本、每段 SHA 和記錄順序、四題與方法對照時點、未解決必要问题、原owner複查、三階段先後與身分分離。保存發派／工具 receipt，不能以一個 metadata 欄位冒充實際 `spawn_agent`。

可直接復用 `audit_readability_round.trace_errors()` 的單元與版本核對，但以 canonical page 與新欄位包一層；不要直接呼叫硬編舊 progress 的 `audit()`。技術 claim 的 `artifacts`／`sources`／`claims`／`checks` 保持 `check_technical_reviews.py` 現有 schema，复用 `_artifacts():77`、`_sources():103`、`_claims():155`。正式執行證據仍放 `docs/technical-reviews/artifacts/<新批次>/`，免改現有永久證據路徑限制；同一原件只保存一次，各page claim引用其id／path／SHA及真正讀到的行號或JSON pointer。

`_validate():212` 的 per-section task 唯一性與編號節來源契約，不能直接挪成 grouped policy。新 gate 應驗證每一真實 group owner 唯一、頁面確在其派工範圍，並保留 fresh／作者／讀者分離及逐頁 source checks；若抽共用 validation helper，legacy 默认规则仍不變。技術輪頁面記錄與 group報告可合併，但每個頁面的主要主張與支持範圍不可只剩一個整組「通過」。首頁與導航無實質主張時按原 applicability 規約給具體理由，不能含命令卻整頁NA。

只在三輪全部實際完成、目前版本一致後，發布 workflow 選擇 active grouped gate。新 `active` 指向未完成 batch 時，發布須明確失敗；不得回退舊報告假裝通過。`passed_partial` 僅供進度保存。`scripts/check_course_reviews.py`／`check_technical_reviews.py`／`check_review_round.py` 可保留 legacy CLI，網站發布和相關 CI 須使用新的明確入口；實作位置由下一步決定。

## 7. 凍結與 immutable 保存

教材定稿後只做一次 source freeze：保存 commit＋來源檔／圖／Notebook SHA＋完整 inventory；未提交 worktree 部分使用一次 content-addressed snapshot，或先提交可重建版本。每個 reader／複查只記自己實際讀到的 slice refs，不再拷貝整個 source tree、Notebook bootstrap、巨大訓練 archive 或他人報告。相同 bytes 的圖可共用渲染檔，讀者仍需各自實看並留下自己的 tool receipt；placement／圖說改變時仍要看對應頁面。

旧 reports、traces、closures 與 evidence 原 bytes 保持原位。新批次每輪、每次複查用新 session／trace／report 路徑，active collection 保存原報告 SHA與真實收到時間。先前卡點、初判revise、void派工及失败命令均保留，不能以新的摘要取代。若仍須更新 legacy 正式 report 路徑，先 opaque copy 舊檔至唯一 history 路徑、記原 SHA；新審閱者不讀舊內容，協調者不代改其結論。

後續保存新 raw evidence 時，應在 `.gitattributes` 追加 phase7 raw evidence `-text` 規則，延續 phase6 的 bytes保護；formatter不得改寫已綁SHA的證據程式。這兩項是待實作建議，本次沒有修改設定。原始大型資料可留既有LFS／公開SHA來源，claim只保存親自核對的必要小份原始片段與定位；不可把「已有SHA」說成讀过證據。

## 8. 圖、瀏覽器與證據驗證

初讀每次只顯示已解鎖單元及其圖／圖說。現有 SVG 可用 Inkscape 在640及360寬渲染後 `view_image` 實看；PNG／JPEG／HTML img 也須從原稿圖 inventory列入。把width、來源SHA、渲染SHA、本人觀察與真正工具receipt列入紀錄，不用SVG原碼或存在性證明可讀。

完整小節解鎖後，在正式匯出頁實際檢查桌面1280×800與手機390×844：圖的必要文字／箭頭、公式、輸出、前後阅读位置、details、横向捲動及動畫停用。若需要在段落階段看頁面，只给當前已解鎖範圍；禁止瀏覽器先載入未讀全文再補作初讀紀錄。Source圖渲染與實際頁面placement是不同證據。Notebook有圖片輸出時，須提取實際 output圖並檢視；共用bootstrap可依實際已讀版本引用，不必每頁複製原文。

環境有 `/usr/bin/inkscape`、`/usr/bin/chromium`、system Python 的Playwright package，但本次沒有啟動瀏覽器，沒有做教材圖片／页面檢視。沒有MCP瀏覽器工具；shell Playwright是可行候選，launch／layout能力仍待實際驗證。

技術閱讀指定原始結果前先看頂層keys／型別，再取必要JSON pointers；原件完整SHA照存，不讀作者额外勝敗評語或舊審閱答案，也不删除原件中的注記制造乾淨來源。`section_facts.py` 只提取原碼／SVG和原始fences，`--execute`才有bounded CPU執行；它不給主張判定。只有真实新執行可寫成重跑，核對既有結果需标為既有证據核對。

## 9. 必要檢查與估時

新增內容及wrapper完成後，先跑針對性的既有tests：`test_review_inventory_extension.py`、`test_readability_trace_audit.py`、`test_factual_dispatch_audit.py`、`test_technical_reviews.py`、`test_reading_time.py`、`test_course_export.py`；若共享KL／MTP模型逻辑有改動，另跑 `test_models.py`、`test_course_compression.py` 及新機制必要的遮罩／對齊／梯度／teacher-frozen checks。不要為文字修改重跑GPU訓練。

新 grouped gate 应有真正能失敗的fixture：active batch錯版／未完成不可回退、漏canonical頁／錯scope／重複owner、跨階段作者與reader身分重用、未保存當段就解鎖、同數checkpoint但unit／intro／figure改版、四題／原句來源缺失、未解決必要問題、假複查沿用原trace、永久artifact被改。只验证記錄完整性，不用合成fixture寫成正式reader-pass報告。

定稿後 `build_course.py --check` 檢查287份同步。新增及程式受影响節先在新CPU kernel執行；發布沿 `.github/workflows/pages.yml` 要求對全287份獨立kernel核對，`--workers 1`避開既有Jupyter port競態。正式匯出應 `--executed <當批新output> --require-reading-times --revision <實際code commit>`，再 `check_site.py` 與 active grouped gate。source／figure／output改動後，不能把舊生成物直接當新驗證。

`reading_time.load_estimates():164` 強制正整數區間、本人reason／task、每頁來源與圖SHA；`load_routes():205` 核對真實canonical IDs；`validate --executed` 需325頁完整。原321筆保留opaque歷史；僅同source及figure bytes的估時可原樣續用，不能替改版頁修改hash冒充新估時。新頁和改版頁由實際讀過目前版本的讀者寫区間與理由。最后銜接讀者也可在自己的實讀後給估時，無需再加一套全文盲讀框架；估時是AI判斷，不是實測平均。

## 10. 本次實際驗證與未做項目

已用標準庫與repo的inventory函式核對321页唯一集合、六組完整不重複覆盖、38來源与107圖的基準Git bytes、283 Notebook來源SHA，以及新四页加入後的数量投影。未執行pytest、Notebook或Zensical：repo目前沒有 `.venv/bin/python`，system Python没有torch／pytest／nbclient／nbformat／zensical；現有 `outputs/notebooks`、`outputs/site` 只是舊產物存在。環境安裝／runtime設定由協調者另做，本次没有安装或修改設定。

這份資料可供下一步最小wrapper与gate實作使用。正文未定稿前不要dispatch任何閱讀，也不要建立passed report。
