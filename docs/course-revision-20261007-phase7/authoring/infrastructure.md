# Phase7 grouped 審閱：CLI 與記錄格式

本文件只提供流程與空白格式，不提供任何教材答案。工具是 `docs/review-tools/phase7_review.py`；只核對可追查記錄，不能证明看懂、主張正確或真实獨立派工。首次讀者不要讀工具的 `outputs/grouped-review/*/state.json`、未解鎖原稿、他人／作者報告或未讀全文。每次取得一個單元，在獨立回合真的理解與看必要圖後，自己寫 checkpoint，再執行 `next`。禁止讀完整 state 後回填。

## 協調者：定稿後凍結一次，尚未定稿不可執行

```bash
.venv/bin/python docs/review-tools/phase7_review.py freeze \
  --output docs/course-revision-20261007-phase7/reviews/freeze-01 \
  --groups docs/course-revision-20261007-phase7/research/infrastructure/six-sequential-groups-proposal.json \
  --expected-pages 325 --batch-id phase7 \
  --author /root --author /root/p7_distillation_sources \
  --exclude-reviewer /root/p7_mtp_sources_and_capstone \
  --exclude-reviewer /root/p7_review_infrastructure_inventory
```

輸出共享 `manifest.json`、canonical 原稿切片、每個原圖的單份快照及 skill／protocol 快照；原稿／圖SHA、原始Notebook SHA、首頁的executed模式與審查規約SHA綁定版本。這不是閱讀、派工或通過。六組新增頁由方案的pending IDs插回對應章序；每頁只分配一次。凍結範圍應是287小節＋23章導讀＋14完整文件＋1首頁；reader主審的完整入口中R／W／G／T編號小節末也需要四題。規約內容若改，舊criteria的結果不能沿用；需依新規約安排真實新審閱，不能靠換manifest SHA解除失效。

真實 `spawn_agent(fork_turns="none")` 返回後，協調者將實際receipt存入這個freeze旁的 `dispatch.json`。先寫receipt再讓讀者start，不捏造每節身分。下列是格式範例，尖括號必須替換成真實工具回覆：

```json
{
  "schema_version": 1,
  "batch_id": "phase7",
  "dispatches": [{
    "stage": "reader",
    "group": "a",
    "reviewer_task": "<actual canonical task>",
    "fork_turns": "none",
    "dispatched_at": "<actual ISO timestamp with timezone>",
    "spawn_receipt": {
      "tool": "collaboration.spawn_agent",
      "task_name": "<same actual canonical task>",
      "agent_id": "<actual returned agent id>"
    }
  }],
  "followups": []
}
```

各組owner是真實task；第一輪、技術輪、銜接輪互不重用，也不重用作者／研究者／工具作者與舊reader身分。起始派工中的`fork_turns=none`記錄不能替代實際工具呼叫。共用filesystem並非技術隔離。

## 讀者：只取得當前單元

```bash
.venv/bin/python docs/review-tools/phase7_review.py start a \
  --manifest docs/course-revision-20261007-phase7/reviews/freeze-01/manifest.json \
  --stage reader --reviewer '<actual canonical task>'
```

工具只輸出當前正文、當前可見圖的共享快照路徑與已讀圖清單；`primary`明列主審或前文，`stage`明列輪次。必要前文以完整原稿頁先讀並保存，同樣逐段五欄；派工的context必讀。可用重複`--context <canonical-page-id>`加自己需要的前文。新發現前置需求可另開同owner `start a --context-only --context <id>` session，真讀後引用其中checkpoint；不能以作者摘要、尚未讀的導覽或課外知識當作前文。暖身context是本人實讀前文，不冒充另一份正式pass，不強制逐节四題與方法對照；必要未知仍須issues，拿來支持主審核心主張的引用仍須指向本人真讀原句。

普通單元最低記錄如下；內容由讀者當場寫，`page_id`／`unit_index`取本次工具輸出。其餘array可省略，工具只保存空集合，不生成答案。

```json
{
  "page_id": "<current page>",
  "unit_index": 0,
  "understanding": "<目前在做什麼以及理由，以自己的話寫>",
  "materials_and_labels": "<材料、答案、編號與已見图元素的意思>",
  "expected_change": "<下一步預期改什麼及原句依據>",
  "confusion_and_quote": "<此時需要猜測／回讀之處及當時原句；若無則實際說明>",
  "missing_visuals": "<是否需要自行想像素材／對應；具體缺圖或文字已足夠的理由>"
}
```

```bash
.venv/bin/python docs/review-tools/phase7_review.py next '<returned session>' \
  --checkpoint outputs/reader-checkpoints/<actual-task>/unit-000.json
```

`next`先驗證和追加當前原始記錄才解鎖下一段。Trace保存在freeze旁的`traces/<stage>/<group>/<session>.jsonl`，每行帶真實時間、段SHA和前行SHA；不得改早期記錄。末段輸出complete後才做整頁／整組復述。狀態在ignored outputs；永久trace和共享source快照足以重建當時原句。

## 已解鎖材料的恢復與實際程式輸出

若工具回覆在上下文整理時截斷，可用協調者提供的 `authoring/current-only.py <session>`；它只重顯已解鎖目前單元，不advance，也不印raw state或後文。讀者仍不得自行讀state。

要重新引用本人已讀前文，可用 `authoring/past-unit-only.py <own-trace> <actual-event-index> --reviewer <own-task>`。它核對 trace chain、owner 與凍結指紋，只重顯那一個真正 checkpoint 過的單元；不印整頁、其他人的原句或尚未記錄段落。這是回讀，不能用來覆寫當時首次理解。

`authoring/unlocked-outputs.py <session> [--page-id <已讀頁>]` 只回傳本人已解鎖code對應的既有CPU輸出與單cell artifact。來源Notebook必須與freeze匹配，執行副本也須通過exporter來源檢查；沒有已解鎖code或沒有guide的執行副本時明列不可用。它不新執行、不提供作者解釋、不顯示其他code。先前的實際輸出觀察只能追加，不能回填舊trace。

核心主張仍依本人實際已讀原句／code引用；輸出觀察可寫在五欄並引用單cell artifact，不能將stdout冒稱原稿explicit句子。分清閱讀包當時未提供stdout，與完整網站是否真的缺素材。這些程序helper在本批次authoring保存，PNG rendering工具仍不產生閱讀判斷。

整頁已讀完後，從repo根目錄執行 `python docs/course-revision-20261007-phase7/authoring/page-capture.py <page-id> --owner <actual-task-basename>`，可擷取預覽網站的桌面與手機版；owner標籤用不含斜線的本人task末段，receipt仍用完整task。目前ignored的等同副本是`outputs/phase7-render/page.py`。每次使用唯一新目錄，不覆寫先前的PNG。若圖放在折疊補充，可加 `--open-details`，另存真正展開後的圖與周邊文字；舊的關閉畫面仍保留。需要完整表格時可加 `--tables`，按垂直段落和必要的表格容器橫向捲動位置另存畫面。依實際回傳的 artifact 路徑觀看，不假定PNG名稱或用已隱藏的圖當作位置證據。截圖程式只產生材料，必須等 tool 回傳、本人實際查看後才記觀看與理解；同一 exec 中等待截圖再自動寫「已看」不算本人已觀看。

2026-10-06 的 helper 統計訂正：舊版未加 `--tables` 時沒有查表格，卻輸出 `visible_table_count:0`，該值不能支持「頁上沒有表格」。現版會始終查可見表格，另以 `table_captures_requested` 表示是否要求擷取表格。折疊內表格仍使用 `--open-details --tables`。舊輸出與原讀者記錄保留，不回改當時的觀察。

2026-10-07 增加已讀文字定位：完整頁已讀完後，可以加 `--text '已讀段落中的一段文字'`，重複指定可擷取數個段落。它只定位目前可見的段落；沒有符合段落會停止，折疊內文字須先使用 `--open-details`。這補足無圖頁的預設截圖只涵蓋頁首、沒有拍到中段說明的範圍限制。輸出記實際匹配數、位置與新PNG；仍須本人真正觀看，不能把找到文字或生成PNG寫成已看懂。讀者也可用自己的Playwright定位已完整讀頁的文字，保存相同viewport、來源與實際觀看紀錄。

## 首次使用及小節末：四題與命名方法原句對照

以下是第一輪reader主審頁的要求，technical／continuity不重填這套四題及方法對照。reader遇重要新概念首次使用時自己辨識並填；工具的`four_questions_required=true`只標出主審末段強制檢查，不能替人判斷首次使用。在普通checkpoint加以下欄位。`at`可為`first_use`、`section_end`，或`["first_use","section_end"]`，同時發生不需重複四份答案。末段若工具列出多個`required_section_ends`，`scope`可列那些IDs；同一份真正適用的答案可同時引用。工具不预告作者希望讀者答什麼。

```json
{
  "introduced_methods": ["<讀者在當段識別的重要新方法>"],
  "four_questions": [{
    "at": ["first_use", "section_end"],
    "scope": "<current page or returned numbered section scope>",
    "subject": "<主要概念或整個方法>",
    "questions": [
      {"number": 1, "answer": "<是什麼，與已讀前文的關係>", "claims": [{"claim": "<核心主張>", "status": "explicit", "quote": "<當前unit的原句>"}]},
      {"number": 2, "answer": "<為什麼在這裡學，回應什麼問題>", "claims": [{"claim": "<核心主張>", "status": "explicit", "quote": "<當前unit的原句>"}]},
      {"number": 3, "answer": "<如何使用，需要什麼、得到什麼及用途>", "claims": [{"claim": "<核心主張>", "status": "explicit", "quote": "<當前unit的原句>"}]},
      {"number": 4, "answer": "<例子／圖支持什麼，哪些還不能推出>", "claims": [{"claim": "<核心主張>", "status": "explicit", "quote": "<當前unit的原句>"}]}
    ]
  }],
  "method_introductions": [{
    "name": "<same introduced method>",
    "identity": {"assessment": "sufficient", "reason": "<身分與整体角色當下夠用的理由>", "evidence": [{"claim": "<該主張>", "status": "explicit", "quote": "<原句>"}]},
    "purpose": {"assessment": "sufficient", "reason": "<目的當下夠用的理由>", "evidence": [{"claim": "<該主張>", "status": "explicit", "quote": "<原句>"}]},
    "step_relation": {"assessment": "sufficient", "reason": "<本節步驟與整個方法關係的理由>", "evidence": [{"claim": "<該主張>", "status": "explicit", "quote": "<原句>"}]}
  }]
}
```

每個核心主張一份來源。`explicit`是當前正文明說；`prior`必須指向本人已讀的`trace_file`＋`event_index`＋真實`quote`，不能引用未讀前文；`inferred`另加`reason`交代推論關係；`unknown`用`needed_now` bool及`reason`說缺什麼，必要未知另加`issue_id`并列入當段issues。各輪／context普通checkpoint若要保存來源，可用同格式的頂層`claims:[]`，不必為此補四題。 `needs_explanation` 的對照項物件本身也需 `issue_id`，指向當段真實的 blocker／burden；只有 evidence 內的 ID 不足以表示整項关联。首次使用的 `subject` 與对应 `method_introductions.name` 使用同一名稱，答案與依據仍由本人寫，不為格式改變實際判斷。方法assessment可為`sufficient`、`needs_explanation`、`not_yet_needed`；後兩者須說理由，必要介紹缺漏不能只標推得便判完整。

圖中依據以`figure`原稿路徑、`figure_sha256`、`locator`、本人觀察`quote`及`visual_receipt:{path,sha256}`記錄，圖須在當前／實際已讀unit顯示過。不是靠搜尋SVG標籤推定箭頭關係。

## 問題及視覺證據

當場issues保留`id,type,severity,location,quote,understanding,missing`。severity是`blocker`／`burden`／`optional`；`quote`必須是當前原句。四題未知`needed_now=true`或方法`needs_explanation`必須有相應blocker／burden，不能降為optional。後文釐清追加位置，早期記錄不删。

真正render／view的PNG可留`outputs/`。持久小receipt寫在本批次artifacts，包含實際tool、owner、source/render SHA、觀看時間及範圍；每名reader仍須真的view，共享render不等於共享觀看。範例如下，不是已完成的檢查：

```json
{
  "tool": "view_image",
  "reviewer_task": "<actual task>",
  "received_at": "<actual timezone-aware timestamp>",
  "source_sha256": "<viewed source figure SHA>",
  "artifacts": [
    {"path": "outputs/reader-figures/<figure>/640.png", "sha256": "<actual PNG SHA>", "width": 640},
    {"path": "outputs/reader-figures/<figure>/360.png", "sha256": "<actual PNG SHA>", "width": 360}
  ]
}
```

Checkpoint／page的`visual_checks`項目格式：`figure,source_sha256,status,details,observation,receipt:{path,sha256},artifacts:[上述render refs]`。`status=verified`只用於本人真view；未做就是`unverified`並說範圍。必要SVG要求640與360兩寬。永久receipt保留render SHA，發布乾淨checkout沒有ignored PNG時仍可驗receipt，PNG仍在本機時會驗其實際SHA。

整頁解鎖後檢查實際網站桌面1280×800與手機390×844，`page_visual_check`使用同樣status／details／observation／receipt／artifacts，但render項目寫`viewport:[1280,800]`或`[390,844]`；check與receipt的`source_sha256`都綁當頁canonical正文SHA。必要圖的頁面placement必做；沒有圖且確實不需要layout驗證時可明列`status:unverified,required:false`，gate會保留未驗證清單，不宣稱圖像已讀。有瀏覽器條件時依skill檢查實際公式和閱讀流程。必要圖片未驗證不能通過。

## 真正group報告與收件

每個新report用唯一新路徑，不覆寫初判報告。以下只是schema片段；所有答案、判定、SHA與refs由本人閱讀後填入：

```json
{
  "schema_version": 1, "review_policy": "phase7_grouped", "batch_id": "phase7",
  "stage": "reader", "group": "a", "reviewer_task": "<actual task>", "reviewer_context": "fresh",
  "manifest_sha256": "<freeze actually used>", "verdict": "<pass or revise>",
  "trace_files": [{"path": "<actual durable trace>", "sha256": "<actual trace SHA>"}],
  "pages": [{
    "page_id": "<first primary page, then every page in assigned order>",
    "source_sha256": "<actual current source>", "figures_sha256": {},
    "verdict": "<own verdict>", "summary": "<本人理解及限制>",
    "trace_file": "<current-version full-page actual trace>",
    "question_refs": ["<numeric event_index of every first-use/end question checkpoint>"],
    "issues": [], "missing_visuals": "<實際判斷>", "visual_checks": [],
    "page_visual_check": {"status": "unverified", "required": false, "details": "<哪些未做及為何當下不必要>"},
    "variation": {"change": "<有意義的小變化>", "prediction": "<真正預測>", "reason": "<已教概念支持的理由>"}
  }]
}
```

reader的`question_refs`實際必須是整數，完全對應當頁chosen trace；technical／continuity未填四題時為`[]`，仍需自己的逐段五欄、問題、看圖與每頁輪次證據。示例中的尖括號是說明，不能直接提交。沒有主要例子的入口可用`variation:{not_applicable:true,reason:"本人具體理由"}`。

owner完成自己的完整新報告後，可使用 `authoring/group-preflight.py --manifest <actual manifest> --report <own actual report>` 執行同一個 `group_report` 核對，抓出欄位型別、必要圖收據、原問題保留及頁面證據等缺漏。這是單組 metadata preflight，不是全輪gate，不會收件、寫FINAL receipt或替人填理解／判定；仍由本人交實際FINAL後，協調者另保存receipt並collect。工具沒有新增審查標準。

技術輪每頁另有現有`check_technical_reviews.py`的`artifacts,sources,claims,checks` schema；claim各自有原始權威来源、定位、支持範圍及必要執行／分母，來源庫和舊review verdict不能當證據。執行artifact仍用`docs/technical-reviews/artifacts/<batch>/`。無實質主張的导航頁以`applicability:{substantive_claims:false,reason:...}`列理由；含program／command fences不能整頁NA。銜接輪每頁另有`continuity_checks:{prior_relation,transition,new_prerequisites,terminology,example_change}`，各寫當節實際銜接。stage初始fresh context是原owner的真實起點；同ownercallbacks不是新盲讀者。

協調者收到真實final後，先保存小receipt：`{final_received:true,reviewer_task,received_at,receipt:"真實工具／mailbox final定位"}`，再收件：

```bash
.venv/bin/python docs/review-tools/phase7_review.py collect \
  --manifest <current-freeze>/manifest.json --report <actual-new-report> --receipt <actual-final-receipt>
.venv/bin/python docs/review-tools/phase7_review.py gate --manifest <current-freeze>/manifest.json --stage reader
.venv/bin/python docs/review-tools/phase7_review.py seal-stage --manifest <current-freeze>/manifest.json --stage reader
```

`collect`追加不可回改的collections行，保存report SHA與final receipt。`seal-stage`只在完整gate成功後保存下一階段的初始readiness小receipt。技術spawn的dispatch需`ready_receipt:{path,sha256}`指reader seal；銜接spawn指technical seal；真正派工時間必須在此receipt之後。下一階段start前仍檢查當前必要前輪完整性。

## 修正與callbacks：同batch新freeze，不重讀無關頁

1. 初讀或後續輪發現問題，owner交revise，保存原始report／trace與具體issue。協調者修正文／圖，其他owner不代改理解或判定。
2. 用同batch、同scope再凍結新版本，`--previous <old-freeze>/manifest.json`。共享新snapshot只存一次；旧freeze原文／圖／dispatch／readiness／報告不改。舊dispatch與collection records傳承至新版本；不要更換真owner或造新fork-none身份。
3. 協調者真正`followup_task`原owner後，在**新freeze**的dispatch追加`followups`項目：`stage,group,reviewer_task,called_at,followup_receipt:{tool:"collaboration.followup_task",task_name:<same task>,receipt:<actual call receipt>}`。callbacks用真followup，不假造新spawn。
4. 原owner開新session，只讀受影響完整頁與必要前文；例如：

```bash
.venv/bin/python docs/review-tools/phase7_review.py freeze \
  --output <new-freeze> --groups <same-groups-plan> --batch-id phase7 --expected-pages 325 \
  --previous <old-freeze>/manifest.json
.venv/bin/python docs/review-tools/phase7_review.py start a --manifest <new-freeze>/manifest.json \
  --stage reader --reviewer '<same actual owner>' --page '<affected canonical page>' \
  --recheck-of '<old original durable trace>' --issue '<original issue id>'
```

`--issue`只列真正原trace已存在的issue；純版本callback而原reader沒有自己的issue時可省略。新checkpoint用`rechecks:[原issue id]`記實際複查，仍保存五欄與必要四題／原句。最後group報告保留全部原trace refs、原问题及新trace，按原primary順序列每頁；改頁指新current trace，未改頁可指本人已實讀且SHA完全相同的原trace。原owner更新自己整組report的manifestSHA，用新report路徑再collect，不能由root修改其verdict／摘要。

必要issue的final條目保留id／severity，`status:resolved,resolution:<實際如何修正>,recheck:{trace_file:<new session>,event_index:<真正重查當段>}`。不能在同一早期trace填resolved就冒充修正後讀。後文只釐清但未修必要介紹時仍保留revise；選讀改善可用optional／clarified_later並寫處置理由。

技術／銜接修正後，受影響頁必須由原reader回讀、需要時原技術owner回查，原continuity owner也回讀。最終gate要求**每階段每頁都綁當前來源／圖／Notebook**。但原technical／continuity的初始spawn仍依當時真实stage-ready receipt判順序，不拿最新callback收件時間倒推舊spawn違規；必要後續callback由同batch來源覆蓋與原owner新session證明。无关同bytes頁保持原真讀记录，不逼325頁重讀。

## Active發布入口與估時

協調者另保存固定 `docs/course-reviews-active.json`：`{schema_version:1,review_policy:"phase7_grouped",manifest:<current manifest repo path>,manifest_sha256:<actual SHA>}`。工具不自動建立或更新這個pointer。有pointer時三個既有CLIs都路由active：reader checker檢查reader；technical checker檢查reader＋technical；review-round檢查全部三輪。active錯版／未完成必失敗，不能回退舊report；無active才保持legacy原規則。`phase7_review.py gate --stage reader|technical|all`可單獨核對。

最後continuity owners可在本人實讀後給現有reading_time格式的`page_id,minutes_min,minutes_max,reviewer_task,reason,source_sha256,figures_sha256`；不用新估時框架。協調者只彙整原row，不替改版頁換SHA冒充新估時。輸出是AI估計，不是實測平均。

本文件範例和自動化test fixtures從未是正式教材審閱報告；只有真實讀者自己的記錄可以進入active batch。
