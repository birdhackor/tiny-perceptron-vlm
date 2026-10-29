# Live legacy reconciliation 獨立技術判斷

審閱者：`/root/omni_p1_infra_review`。日期：2026-10-29 UTC。這是 ROOT 委派的獨立證據審查；我不是 `legacy-stop-evidence.json` 的作者，沒有修改該索引、作者證據、舊 ledger／quota、P1 程式、原初判或複查。沒有 initialize、workflow dispatch、遠端 mutation 或付費工作。

**判斷：14 筆舊 retained attempts 的身分與操作性停止證據足夠。由我建立真正的 [legacy-reconciliation.json](../legacy-reconciliation.json)，精確保留全部 US$4.73 bounds。這允許 ROOT 在提交全部必要證據後，獨立執行純 client initialize；initialize 仍必須重讀 fresh inventory／official snapshot，通過指紋、無 active／unknown、source gate 與非重設 CAS。沒有放行 GPU。**

## 真實輸入與獨立核對

起始 HEAD 為 `80c5db17e8b077b2c0a5e4e9baa48e760ccb0532`。P1 inventory workflow run為 `40835550319`，transport為 `40835555383`，兩者source revision均為此main。四個實作檔的commit bytes仍與複查01核准的SHA相同。作者索引SHA-256為 `2eb097d2d96418fc52cbce383059c71d35829636cd5653298f84a5a6e685689f`；索引原本的「awaiting-independent-review」和逐筆pending欄位沒有被我改成自通過。

我實際讀取 `/workspace/work/omni-p1-live-inventory/` 原始檔，而不是拿作者摘要代替。原始指紋如下：

| 原件 | SHA-256 |
| --- | --- |
| inventory.json | `d94edf3c66aadb1a9b9eda404b1a9e623c5b3d463088cf680d8304747a46a963` |
| legacy-budget.json | `19660f4f7b8fd5d85e167fb3b10fb530a997797db7add5356447f603bae28bfb` |
| legacy-gross-quota.json | `6bf09f23841293f476a132a3ca713189eec0d12a442e0b84ff0f09f9844878a7` |
| legacy-policy.json | `557b2bd1135c41beff0fdc38bc69ed8a4c36ddc06d14b917d16bf5b732867372` |

inventory的觀測時間是 `2026-10-29T07:28:03.185797+00:00`，workspace=`birdhackor`、environment=`main`，active／unknown／errors均空。這是歷史觀測，不是假裝成現在的新baseline；實際initialize不可沿用它繞過120秒freshness。

原始quota有13筆未釋放attempts，原政策carry-in另1筆；我以Decimal重算14筆合計 **4.73**，與重新執行 `legacy.verify` 的結果完全一致。ledger全部reservation合計及reported audit total均為 **52.59**。52.59是歷史audit，沒有拿它當carry，也沒有釋放已完成、失敗或已部分metered的4.73。

我另外實際執行以下核對，結果存於 [verification.json](legacy-reconciliation-artifacts/verification.json)：

- 14份原始completion JSON（必要時取`/execution`）的raw SHA、run ID、source revision與canonical SHA逐一對live ledger；14/14吻合且remote status皆為completed。
- 12份歷史controller source重新讀11份本機Git object，publisher那份另向官方GitHub blob API讀取；全部SHA對ledger來源binding，並重算作者摘錄的原始行與excerpt SHA。publisher官方blob與Git外cache逐byte相同，見 [source收據](legacy-reconciliation-artifacts/publish-source-recheck.json)。
- 14份Git外完整decoded log逐份重算SHA／bytes，核對每個引用行；14份原始run／jobs metadata檔重算SHA。沒有採用作者記錄中exit0但stdout為空的`gh run view --log`結果。
- 我獨立重新呼叫官方GitHub connector讀取14組jobs，確認真controller終態；兩筆B/C為failure，其餘12筆為success。另重新讀取那5筆缺App completed訊息的完整官方log，均與Git外原件逐byte相同，見 [官方重讀收據](legacy-reconciliation-artifacts/official-rechecks.json)。
- 我重新取得2個live artifact與3個恢復用artifact的官方metadata，核對本機ZIP digest與被引用JSON的exact member bytes，共5/5通過。live inventory ZIP為`75e73e4f4f48ce4e7857ad893100e6f873fc479c3136d6fff7f96d9d502e1464`；live transport ZIP為`c360e34a5d116fddf6a0db97a510beb1c39c13827c3a4ea35e7daebe5e82a162`。恢復用metadata在 [recovered-artifact-rechecks.json](legacy-reconciliation-artifacts/recovered-artifact-rechecks.json)。

完整log、ZIP與上游SDK全文只留在Git外；本報告引用既有證據路徑／digest，沒有重複複製已有原件。

## 逐筆停止判斷

下表「remote completed」只表示精確收據對上live ledger；「controller」保持真實GitHub結論，兩者不能混寫。9筆有官方log明確的`Stopping app`／`App completed`；其餘5筆使用後述完整lifecycle鏈。

| attempt | bound US$ | controller | 獨立停止依據 |
| --- | ---: | --- | --- |
| gha-37441599016-1 | 0.54 | success | remote completed＋明確App completed |
| gha-37445564925-1 | 0.07 | success | remote completed＋明確App completed |
| gha-37446402227-1 | 0.54 | success | remote completed＋明確App completed |
| gha-37448172517-1 | 0.54 | success | remote completed＋明確App completed |
| gha-37450520298-1 | 0.54 | success | remote completed＋明確App completed |
| gha-37451905991-1 | 0.54 | success | remote completed＋明確App completed |
| gha-37479108854-1 | 0.07 | success | remote completed＋明確App completed |
| gha-37603079766-1 | 0.07 | success | remote completed＋明確App completed |
| curriculum-38864076350-1 | 0.54 | **failure** | remote completed＋DeserializationError原件＋非detach context正常退出 |
| curriculum-38864352441-1 | 0.54 | **failure** | remote completed＋DeserializationError原件＋非detach context正常退出 |
| curriculum-38867090399-1 | 0.30 | success | remote completed＋同步get回傳＋非detach context正常退出 |
| curriculum-38867093745-1 | 0.30 | success | remote completed＋同步get回傳＋非detach context正常退出 |
| lesson-release-38871109465-1 | 0.07 | success | remote completed＋同步get回傳＋非detach context正常退出 |

我逐筆核對ledger finished_at先於最後controller訊息，最後訊息又先於官方job completed_at；精確時間與SHA在verification結果中，沒有把印log的時間冒充provider帳單finality。

### 缺少App completed訊息的5筆為何足夠

原始 `49ce5b40...:scripts/lesson_products/modal_runner.py:546-602` 與 `0b57069e...:scripts/lesson_products/modal_runner.py:548-604` 均為無參數`with app.run():`，沒有detach。它們先spawn，再同步`call.get()`；exception在with內被記錄，finally重讀ledger／寫controller結果；最後status的`print`在with之外。官方完整log的該status行證明程式確實越過context退出，並非僅憑「原始碼理論上可能跑到」。

兩筆B/C的controller原始JSON明確保持`status=failed`、`error_type=DeserializationError`；function_call_id分別為 `fc-01M51R1NC200V50GYZ0E12NGER`、`fc-01M51R6EYFNC949G97N17NBDZ6`，都與live ledger相同。remote source在訓練／傳輸操作結束後才將completed receipt寫ledger並commit（該版`:415-477`）；其canonical receipt SHA與live ledger相符。這支持「remote完成、controller解碼失敗」，沒有把controller failure改成success，也沒有把它當成本輪模型能力驗收。

publisher的官方固定blob（SHA-256=`6efa33d83da7249c386acbb51724b023aab0d0bae7e73253d9f2bea62216df9b`）`:339-371`同樣是無detach的app.run、同步call.get、context外最後print。

先前已核對的真Modal 1.6.0 SDK，`App.run`預設detach=false；`modal/runner.py:484-496`在正常context退出時await disconnect，`:326-340`的AppClientDisconnect終止ephemeral app的running tasks。SDK來源與版本digest沿用 [初判primary-source收據](initial-artifacts/reference-receipts.json)，沒有用當前私有RPC自造停止語意。

所以，這5筆的依據是**精確remote完成收據＋實際同步controller路徑＋正常非detach退出的log＋官方job終態**，再用現在沒有active apps的metadata作補充。沒有以空app list單獨推出歷史停止，也不保證某一個精確毫秒已完成所有provider後端清算。

## 既有Volume與財務界線

保留的3個既有named Volumes為：`learn-to-yolo-checkpoints`（`vo-9CE0qbHL0U9baf24hrlSnT`）、`tiny-perceptron-checkpoints`（`vo-XZLJ77e3Jgk4IssX0VVBtL`）、`tiny-perceptron-course`（`vo-WrpfGCTOfZBhKuFKdTMgyX`）。其排序後name／object_id／created_at完整清單的canonical SHA綁入reconciliation。沒有讀取別專案資料檔、刪除、搬動或新增Volume。

sizes_measured=false。rates原件顯示當時volume單價為US$0.09／GiB-month，但不知道三個Volume容量，不能推導持續storage金額，也不聲稱全workspace未來費用有絕對上界。這次承認的是已知存在與保留，不是財務settlement。14筆bounds保持原額，未退款、未釋放，歷史audit52.59維持排除。

live transport receipt顯示4145-byte fixture、authenticated roundtrip、optimizer/RNG同環境exact continuation及draft cleanup均成功，且Modal compute_started=false；其source revision與官方artifact digest已核對。這不延伸成正式GPU checkpoint的跨環境resume保證。

## Reconciliation與下一步

reconciliation的reviewer_task是我實際的 `/root/omni_p1_infra_review`，逐項pin raw ledger／quota／old policy、重新驗出的legacy_verification、3個既有Volume清單，以及14筆各自的stop evidence。既有completion JSON只引用原路徑，兩筆B/C另必須引用原始failed controller結果；共享metadata／log／source摘錄只存一份。我的這份獨立判斷也被SHA固定，不由作者代簽。

實際命令 `uv run --no-sync python docs/course-revision-20261029-omni/infrastructure-review/legacy-reconciliation-artifacts/verify_live_evidence.py` 已成功；初次完成後補入嚴格時間先後斷言再跑一次，兩次輸出均保留於 [命令結果](legacy-reconciliation-artifacts/verification-command-result.txt)。它只寫本審閱結果JSON，不會初始化或修改帳本。reconciliation寫入後再用真正`preflight.load_reconciliation`及`finance.legacy_carry`離線讀回驗SHA／schema／精確4.73；讀回結果另存收據。

ROOT需把reconciliation與所有新引用的docs證據一起提交，才符合source_revision逐檔git-show gate。ROOT執行initialize時，P1會重新取得fresh inventory與gross；若raw／policy／retained set／Volume identity變動、有active／unknown或freshness不符就停止，不能沿用本報告放寬檢查。正式GPU仍須教材三輪及新runner的費用／runtime／image／preemption／checkpoint審閱。本報告只對這14筆舊工作能否帶著全部bounds進入新round作判斷。
