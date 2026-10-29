# P1 新基礎設施獨立初判

日期：2026-10-29 UTC。審閱工作：`/root/omni_p1_infra_review`，委派者：`/root`。審閱者不是這批程式的作者，沒有修改被審程式、commit、push、觸發 workflow 或派送付費運算。

**Verdict：BLOCK `initialize`、`hf-probe` 及正式 GPU。可繼續唯讀 inventory、修正與離線核對。** 這份結果只涵蓋 P1 infrastructure；不算教材逐節審閱，更不算全書三輪。

## 真實任務與標的

原始委派全文、角色、授權限制、實際分支／HEAD 及摘要保存在 [review-task-and-source.json](initial-artifacts/review-task-and-source.json)。原始委派允許的遠端寫入僅為經審閱、有完整上界的 CPU HF／Volume probe，正式 GPU 要等三輪教材審閱。Modal 授權是 `birdhackor/main` 新 round gross 上限 US$60，不可超額。

母代理在本次審閱中確認採保守守門：`retained official increment + 未釋放新 bounds + 精確舊未對帳 carry + proposed <= 60`，並另鎖不可遞減的 `sum(新 bounds) <= 60`。這是現授權內較嚴格的實作選擇，沒有請使用者放寬。所有歷史 terminal 預約不能直接視為未入帳 carry。

審閱時 HEAD 與線上 `main` 均為 `2b3332ca4392a946a065782815f9f60dae9a87c7`，工作分支為 `revision/kev-omni-mainline-20261029`。五個新檔均尚未進 HEAD。為避免後續作者修正抹去初判標的，原始五檔僅保存一份完整版本於 [initial-source-snapshot.json](initial-artifacts/initial-source-snapshot.json)，以下來源定位的行號均指這個版本；本機執行副本位於 `/tmp/omni-infra-review-initial-20261029/`。

| 被審檔案 | 初判 SHA-256 |
| --- | --- |
| `scripts/omni_revision/__init__.py` | `f3948056a97c27498ccca8116f3ee98ddc8e26771f5e6a182e4d9965a04e5aa3` |
| `scripts/omni_revision/finance.py` | `3a2817344d8ad873ec4e165665364866109ce8ad1763ecad37ed3e7ad552cae2` |
| `scripts/omni_revision/preflight.py` | `532016e9279a64aef01213874acfdeb5531b241b8411951c5e3e3258bf9fd4ed` |
| `tests/test_omni_revision_finance.py` | `e6c07e98b6fb24e6f3ece3ed7660e0e32fbf5628b6e72588f091b6fbfc21880f` |
| `.github/workflows/omni-revision.yml` | `ed6375b19e89cd4bc90b09336c1bba32ca931c1202d4f0a1418e8bfef405a0a9` |

另實讀 repo `AGENTS.md`、舊 `scripts/selftrained/finance.py`、`scripts/selftrained/modal_runner.py`；其 SHA-256 在 task JSON，內容可由上述 HEAD 取得。既有 workflow 僅查共享 concurrency 宣告。沒有把五檔以外的 helper、舊 policy 資料檔或其他 runner 當成本次已審實作。

## 實際命令與結果

所有測試離線、無 GPU；fake HF token 為明示的測試字串，沒有讀取任何真 token 值。

| 命令／動作 | 實際結果與收據 |
| --- | --- |
| `uv run --no-sync pytest -q tests/test_omni_revision_finance.py` | **16 passed**，見 [執行輸出](initial-artifacts/author-test-independent-run.txt)。這是我親跑作者測試，不是獨立正確性的充分證據。 |
| `OMNI_REVIEW_SOURCE_ROOT=/tmp/omni-infra-review-initial-20261029 uv run --no-sync pytest -q docs/course-revision-20261029-omni/infrastructure-review/initial-artifacts/independent_probes.py` | **16 passed**，見 [獨立 probe 原始碼](initial-artifacts/independent_probes.py) 與 [輸出](initial-artifacts/independent-probes-result.txt)。其中 `test_reproduce_*` 是斷言缺陷確實可重現，通過不是安全認證。 |
| `uv pip install --target /tmp/omni-review-modal-1.6.0 modal==1.6.0` | 成功，僅新增本機隔離審閱套件，不改 project dependency／lock。官方 wheel digest 為 `2f5981ba656570ad96379d29f0ddff876880708c84966024ce130e9e1794fa68`。 |
| `PYTHONPATH=/tmp/omni-review-modal-1.6.0 uv run --no-sync python docs/course-revision-20261029-omni/infrastructure-review/initial-artifacts/sdk_local_validation.py` | 使用真 Modal 1.6 驗證本機 decorator／資源參數登錄，明確在 `App.run` 前拋 `ReviewRemoteBlocked`，見 [腳本](initial-artifacts/sdk_local_validation.py) 與 [結果](initial-artifacts/sdk-local-validation-result.json)。沒有建立遠端 App 或 Volume。 |
| SDK 驗證 harness 第一次嘗試 | `Volume.objects` 每次取值回傳不同 manager，patch instance 未攔住 lookup；SDK 在 `Client.from_env` 發現沒有 Modal token 即拋 AuthError，尚未執行 RPC。完整 [失敗輸出](initial-artifacts/sdk-local-validation-first-attempt.txt) 保留。隨後改成明確 fake facade 才取得上述成功結果。這是審閱 harness 的修正，不是被審程式的修正。 |
| `uv run --no-sync python -c 'from scripts.omni_revision.preflight import source_revision; print(source_revision())'` | 如預期失敗：HEAD 沒有新 finance 檔；見 [原始輸出](initial-artifacts/uncommitted-source-gate-result.txt)。不能以髒工作樹當已釘選 revision 執行付費操作。 |
| `gh api repos/birdhackor/tiny-perceptron-vlm --jq '{default_branch: .default_branch, pushed_at: .pushed_at}'`；`gh api repos/birdhackor/tiny-perceptron-vlm/commits/main --jq '{sha: .sha}'` | 唯讀，回傳 default branch=`main`、pushed_at=`2026-10-29T05:52:07Z`、main SHA=`2b3332ca4392a946a065782815f9f60dae9a87c7`。 |
| `git show origin/main:.github/workflows/omni-revision.yml` | exit 128：檔案尚不在 `origin/main`。官方 raw `main` URL 另實際 GET **404**。 |
| `uv run --no-sync python docs/course-revision-20261029-omni/infrastructure-review/initial-artifacts/reference_receipts.py` | 10 次公開 primary-source GET，9 次 200、workflow raw URL 404；共保存 20 筆來源收據。見 [腳本](initial-artifacts/reference_receipts.py)、[結果](initial-artifacts/reference-receipts-result.txt)、[URL／版本／bytes／SHA-256](initial-artifacts/reference-receipts.json)。 |

雲端環境技能讀取後，`environment_status` 顯示 network policy=`unrestricted/enforced`、current=true；此 review executor 只有 Git LFS binding ready，沒有 Modal／HF binding ready。沒有 dump environment、讀 secret file 或印出 credential。外部文件全文只在 `/tmp/omni-infra-review-external-20261029/`；repo 只存收據，不放第三方全文。

## Findings

### F1 — 阻擋：官方 gross 沒有納入 US$60 額度式

定位：`finance.py:121`、`:159-171`。現在只限制新預約加總；官方 increment 的條件是 `increment <= previous + legacy_carry`。這是「可以解釋」條件，不是 gross 額度條件。它也把所有歷史 terminal 預約加成 carry，擴大可以接受的 gross。

獨立反例一：baseline gross=10、歷史 terminal bounds=100、尚無新預約，gross=90（本輪增量80）時仍能預約1。反例二：已保留59且完成、official increment=59，仍可再預約1；`official 59 + retained 59 + proposed 1 = 119`，程式仍接受，因新 bounds 只有60。前兩個 independent probes 直接重現。

舊 `finance.py:216-247` 明確保留 per-month high-water，並採 official retained increment、pending、carry、proposed 相加守門。新舊額度要隔離，不代表能省略新 round 的 gross 守門。需同時保留兩道上限與未解釋增量停止機制，重新做獨立反例核對；不要藉降低 carry 或修改歷史基準讓這個反例消失。

### F2 — 阻擋：legacy 分類沒有對帳證據，quota 內容完全未進守門

定位：`preflight.py:69-82,93-111`、`finance.py:90-105,121,136-141`。`gross-quota.json` 只被 parse、hash、另存，沒有驗證 schema、與 ledger 的關係或其中 attempts。每個 ledger terminal reservation 無論是否已入帳、是否 quota scope，全部列為 `terminated_pending`。`app_id` 可以缺失；terminal 字串加「目前 main 的 app list 沒活躍項」就能初始化。

獨立 fake inventory 提供 `gross-quota.json={}`、只有一筆 completed=48.4 且無 app_id 的 ledger、空 app list；初始化仍成功、carry=48.4。這證明 inventory 結論並未由 quota 或停止證據支持，不是聲稱線上帳本現在就是這些 bytes。

需以實際 raw legacy ledger＋quota＋被它們釘選的政策／receipt 定義，精確識別真正未對帳 bounds；一般歷史 audit total 不得冒充 carry。unknown／open 狀態目前確實會阻擋，值得保留；但此五檔沒有 reconcile operation、沒有獨立停止證據驗證，也不能把狀態改成 terminal 就視為已停止。`reserved`／`unresolved` 亦沒有已審閱的恢復路徑。

### F3 — 阻擋寫入 probe：Volume cleanup 的 ownership 不是原子的

定位：`preflight.py:156-162,415-420`。先 list 不存在，再 `from_name(...create_if_missing=True)`，最後依 name delete。Modal 1.6 的 `volume.py:708-760` 證實 handle 是 lazy hydration，實際 RPC 使用 `CREATE_IF_MISSING`，會接上已存在的物件；`objects.delete` 又會重新按 name lookup。

如果兩次操作間出現同名 Volume，probe 可寫入並刪除後來者。獨立 fake 直接核對這條事件路徑。共享 workflow mutex 排除受控 workflow 重疊，無法讓「先查、後建立」變成 exclusive create。

Modal 1.6 已有公開 `Volume.objects.create(...allow_existing=False)`（`volume.py:277-342`）。需先取得排他的建立成功證據，記錄精確物件 identity／ownership；collision 或不確定建立結果要停，不能刪除別人的 Volume。所有 cleanup 只能處理此 attempt 能證明擁有的物件。

### F4 — 阻擋下一次付費：HF 殘留與不確定結果可被標成已結束

定位：`preflight.py:255-258,365-379`、`finance.py:194-205`。HF branch POST 在伺服器成功、回應遺失時，`hf_branch_created` 仍是 false，finally 不嘗試核對；離線協定故障注入實際重現。另 `finish` 只以 compute_stopped＋volume_deleted 決定 terminal，不要求 HF cleanup。receipt 明寫 branch 未刪除時，仍是 `failed-stopped`，可預約下一次付費，另一個獨立 probe 已重現。

需區分「未建立」「已建立」「建立結果不明」，並在任何必要 cleanup 未完成時留下 unresolved、阻擋新付費，直到有只處理自有 branch 的獨立 reconciliation。不能在回應不明時為了清理而刪除同名既有 branch。現有 409 collision 不設 created flag，因此不會自動刪既有 branch，這點正確。

### F5 — 阻擋 envelope 認證：完整費用上界仍缺證據

定位：`finance.py:208-224`、`preflight.py:164-178,221-226,365-379,385-420,471-479`。CPU `(1,1)`、memory `(2048,2048)`、單 container、180s timeout、retries=0、scaledown=2 都能用真 Modal 1.6 登錄。memory tuple 對應硬上限；CPU limit 是 soft throttle，應按官方規則估價並有餘裕，不能宣稱 CPU 實際用量絕不越1。

但 `retries=0` 只關閉設定的 failure retries；官方 preemption 頁明確說中斷的 input 會自動重啟，現程式沒有 idempotent branch transaction 或重放範圍證明。controller 的360s SIGALRM 是單次 signal；可被 `except Exception` 捕獲後繼續 finally，Volume delete／list／commit 沒有獨立總期限。workflow12分鐘不是已證實的 named Volume TTL；清理失敗後仍可能有持續 storage。`storage_cleanup=0.10`、`build=0.10`、`egress=0.02` 也沒有從實際單價、bytes、保留期間與恢復時限推導的證據。

`Image.debian_slim` 沒加自訂層，但「沒有 image RUN」的註解不精確：Modal 1.6 `_image.py:2590-2639` 的基底定義含 apt／pip RUN，屬 global namespace。這不證明使用者會被收取新的建置費；同樣也不能直接當成已核定的 build gross 上界。需釐清實際 global base 使用／快取行為或使用能證明上界的已釘選 image。

結論是目前不能把不到US$1的 components 加總當成完整 envelope 已被驗證。修正需包含 preemption／重放、startup＋execution＋shutdown、清理 deadline、有限 storage retention 及定價來源的完整說明；不得為此改租 GPU 或加購更大形狀。本次沒有量測任何付費實際費用。

### F6 — 中度：儲存的 high-water 可以丟失歷史而不被發現

定位：`finance.py:132-141,164-168`。一般官方 gross 低於完整已存 high-water 時會拒絕，正常路徑正確；但沒有驗證 stored high-water coverage／baseline／歷史 attempts snapshots 的一致性。把已初始化 state 的 high-water 改成 `{}`，gross 從10降到9仍可再預約，獨立 probe 已重現。

這需要帳本被其他寫者改壞，並非聲稱正常單一 writer 會自動產生該狀態。CAS 防止並行覆寫，不驗證帳本內容。舊 finance 已從固定 baseline＋所有實際 attempt snapshots 重建高水位（舊`:216-233`）。新版本也應重建／驗證 history、偵測 baseline 或 reservation 總額被降低，而不只相信可刪除的 map。

## 已確認與仍未確認的邊界

| 項目 | 本次證據與限制 |
| --- | --- |
| 新舊額度隔離 | 新 round 固定 policy、獨立 `budget/omni-20261029:omni-budget.json`，不寫舊 Volume；舊 raw bytes 可留存。F1／F2 仍阻擋財務正確性。 |
| GH durable reservation | `finance.py:235-272` 符合 contents PUT 的 branch＋base64 content＋observed blob SHA；官方 Contents API 要求更新使用被替換 blob SHA。branch 存在不重設，branch 建成但初次內容寫入失敗時也停，沒有自動 reset。沒有實際建立 budget branch、做 Contents PUT 或測試線上權限／branch protection；CAS 的遠端成功尚未驗證。 |
| 單 writer | 新 workflow 與既有主要 GPU workflow 使用相同固定 `tiny-perceptron-course-gpu`、cancel-in-progress=false。官方 mutex 範圍是同 repo 中採相同 group 的 workflows；不是全 workspace 或任意 CLI 的分散式鎖。不能讓舊付費 workflow 在新 round 旁另外消耗額度。 |
| reserve 順序 | `preflight.py:467-477`：先完成 CAS、再建立 App／Image／Volume；獨立 main fault injection 在 CAS409前攔住，沒有進入 paid boundary。Secret 選擇 helper 在 workflow 預約之前執行，但其實作不在本次五檔範圍，標籤的「metadata only」不能替代該 helper 的審閱。 |
| 官方 gross／跨月／NaN | Modal 1.6 `_workspace.py:429-491`／`types.py:458-488` 確認 summary、YYYY-MM、UTC start/end、Decimal，以及 metered_cost 是調整前成本，不能拿 billed_cost／credits 抵扣。作者測試覆蓋跨月與通常回退；獨立 probe 另驗真／字串 NaN、sNaN、Infinity、負值、stale/future、missing／duplicate month 均拒絕。後端 billing delay 仍不是120s freshness 可以保證的 finality。 |
| HF REST | 與已安裝官方 `huggingface_hub==1.33.0` 的 `hf_api.py:7160-7172,7227-7237`、`_commit_api.py:748-779,928-963`、`lfs.py:181-220,269-309` 逐欄比較：branch startingPoint、preupload、basic LFS batch／PUT／verify、NDJSON commit parentCommit／lfsFile 協定大致一致。preupload `shouldIgnore` 未處理、LFS 僅 basic 的限制需明示；能拒絕不支援 transfer，不代表線上一定選 basic。沒有對 HF 做真正 POST、PUT、DELETE，也沒有證明帳號具有 binary upload 權限。 |
| HF secret | 這五檔只在 Modal Secret 綁定後的 remote body 讀 HF_TOKEN，沒有回傳它。HTTP 失敗只保存 exception type／status，跨 host redirect 去掉 Authorization，獨立 redirect probe 通過。沒有把真 token 帶出 Modal；workflow 外部 helper 未在範圍內，不能宣稱整條 secret selection 路徑已完成審閱。 |
| 小型 checkpoint transport | fake HF 協定完整走過兩個 float weights、SGD momentum＋step、Python RNG state 的 tar，HF 讀取使用 commitOid 固定 revision，digest一致；我另解包並可恢復 RNG state。外層有 Volume 讀回 digest gate。這只是離線協定／bytes核對，沒有真 HF／Volume roundtrip，更不證明正式模型 optimizer／RNG exact resume。 |
| 可發布／可 dispatch | 官方 GitHub events 文件 `workflow_dispatch` 章要求 workflow 先存在 default branch。這個新 workflow 此刻 raw main URL404且 HEAD未含新程式，不能聲稱 `gh workflow run --ref <feature>` 已可用。需先通過修正與複查，再用正常版本流程讓 workflow 在 main 生效；不能為方便借既有付費 workflow繞過。 |

## 允許的具體下一步

1. 維持只讀，取得實際舊 ledger／quota raw bytes、停止／未知 app 明細、Volume inventory、官方 summary／rates；此審閱 executor 沒有 Modal 憑證，故尚未取得任何 live inventory。可由已授權且具憑證的唯讀路徑執行，不能把新的未登錄 workflow 當成已可 dispatch。
2. 由作者修正 F1–F6，補齊精確 legacy reconciliation 與有限的 CPU／storage／cleanup envelope；本審閱者不動作者程式。保留這份初判、snapshot、probe 與收據，複查另建文件。
3. 修正後先做獨立離線複查；真 CPU permission／transport probe 仍須先建立有效的不可重設 round、完成 durable CAS、保留完整 bound，再碰付費邊界。任何 unknown 或 cleanup 不明都停。
4. 此刻不可 initialize 錯誤 legacy carry，不可派 HF／Volume 寫入 probe，不可派任何正式 GPU。正式 GPU 還另外受教材三輪完成條件約束；本報告不能替代該條件。

所有遠端 mutating 呼叫：**0**。所有 paid probes：**0**。沒有用本次測試通過去推論發布、授權、帳單 finality 或全書教材審閱已完成。
