# P1 infrastructure 複查 01

獨立審閱：`/root/omni_p1_infra_review`，2026-10-29 UTC。我沒有修改作者程式、commit、push、dispatch workflow、執行遠端 mutation 或啟動 Modal 運算。原始初判與其中15件封存檔的 digest 全部維持不變，核對結果在 [initial-preservation-check.json](recheck-01-artifacts/initial-preservation-check.json)。

**Verdict：本次所列 P1 可 commit、merge 到 main；main 生效後，可先執行 `inventory` 與 `transport-probe`。這兩個操作不必等 live reconciliation。實際 `initialize` 仍須先取得真實 inventory、逐筆停止證據與獨立審過且 committed 的 reconciliation。沒有放行正式 GPU 或任何 Modal 付費 probe。**

## 範圍與版本

依 ROOT 明確指示，P1 改成純 client inventory／GitHub CAS initialize，另以 GitHub Actions 標準 `contents: write` token 驗證小型 draft Release 資產傳輸。原本 Modal CPU／HF／新 Volume 路徑已刪除。這是授權內的資產通道調整；不是聲稱舊付費路徑已修妥，也不是取消後續 A/C／聲音訓練或完整模型發布。

實讀並審閱以下八檔，另讀未修改的舊 finance 與其 quota policy 作 legacy 比較。ROOT 的 `review_setup.py` 不在這次範圍，也未被本次准用操作匯入或呼叫。

| 檔案 | 最終被審 SHA-256 |
| --- | --- |
| `scripts/omni_revision/__init__.py` | `f3948056a97c27498ccca8116f3ee98ddc8e26771f5e6a182e4d9965a04e5aa3` |
| `scripts/omni_revision/finance.py` | `e7ae36850db4cf379f19057f4219624904f5890a0988ec6e00dd3d0fb5d4130b` |
| `scripts/omni_revision/legacy.py` | `ebbcf89f2b6c8aa7f47c1a9f9cec5df0db288a0c6769958d425bac7510342453` |
| `scripts/omni_revision/preflight.py` | `232999c62306f8ec4215f1e62b820dfa3731da42595ceec23a5210b5c6947cbd` |
| `scripts/omni_revision/github_transport_probe.py` | `3618abbd85b47bb27ee42e134df8c7ba8633e96b214241bbd02ca677b4054e31` |
| `tests/test_omni_revision_finance.py` | `9cefaa7a47c6d1199ee352ef2fccb1a30aac41d73ee65e1a9a32786cc8366da8` |
| `tests/test_omni_revision_legacy.py` | `a0feb2737e18bc004be40b431dcde87c9d2f00232547d8da5632ce4332d2909a` |
| `.github/workflows/omni-revision.yml` | `6c716958c794c46d48711c0365c900b1b6bd6fe593941f5cd4a58d0e200685d2` |

複查開始時 HEAD 仍是 `2b3332ca4392a946a065782815f9f60dae9a87c7`，新程式尚未進該 HEAD。完整候選版本及舊依賴存於 [source-snapshot.json](recheck-01-artifacts/source-snapshot.json)。其中 transport 第一版的 digest 為 `0db5af9cb911a2c6fde4c0f4ccbf90cddcce3eba76de844c89d35c0e0fef3441`；我找到 R1 後，由 ROOT 修正，最終版本另存 [transport-callback-source.json](recheck-01-artifacts/transport-callback-source.json)。沒有回改任何先前版本或原反例。

## 實際驗證

實際工作、命令與結果另存 [task-and-commands.json](recheck-01-artifacts/task-and-commands.json)。作者交接筆記只作 schema 導覽，沒有拿其自報測試通過當作正確證據。

| 實際命令 | 結果 |
| --- | --- |
| `uv run --no-sync pytest -q tests/test_omni_revision_finance.py tests/test_omni_revision_legacy.py` | 親跑 **36 passed in 0.04s**。 |
| `uv run --no-sync pytest -q docs/course-revision-20261029-omni/infrastructure-review/recheck-01-artifacts/recheck_probes.py` | **13 passed in 0.10s**，[原始碼](recheck-01-artifacts/recheck_probes.py)、[輸出](recheck-01-artifacts/independent-probes-result.txt)。其中 R1 的 `test_reproduce_*` 通過表示缺陷成功重現。 |
| `uv run --no-sync pytest -q docs/course-revision-20261029-omni/infrastructure-review/recheck-01-artifacts/transport_callback_probes.py` | 最終 transport 版本 **17 passed in 0.05s**，[callback 原始碼](recheck-01-artifacts/transport_callback_probes.py)、[輸出](recheck-01-artifacts/transport-callback-result.txt)。重用12項適用的原斷言，另驗5個 draft 回應不明／分頁／ownership 情境。 |
| SHA-256 重核原初判 manifest 所列15檔 | 全部相同，沒有改寫初判或證據。 |

所有 GH／Modal mutation 在 probes 中均為記憶體 fake；真實網路僅 GET 官方來源。真 GH Actions binary upload／download、真 Modal inventory、真 CAS initialize 都還沒由本審閱者執行。

## 原始 findings 的複查結果

| 初判 | 結果、來源定位與界線 |
| --- | --- |
| F1：gross 額度式 | **修正通過。** `finance.py:242-282` 同時檢查新 bounds 總和與 `retained increment + previous + exact carry + proposed <= 60`，另拒絕未解釋 increment。我的獨立 fixture 將歷史 audit 設100、真 carry 設0.50：gross 增80不再可預約；先預約59、gross增59、再預約1也被拒絕。完成／失敗都不自動退回 bounds。 |
| F2：legacy carry／停止證據 | **實作核對通過，真資料對帳尚待進行。** `legacy.py:17-95` 核對 marker／policy raw與canonical SHA、scope IDs、carry immutable SHA、逐筆 snapshot／bound／chronology及完整高水位，carry只等於 sidecar未釋放bounds＋政策carry-in，不再使用一般歷史 audit total。`preflight.py:57-108` 保留 raw；`finance.py:95-142` 要求精確 reconciliation、stop evidence 與既有 Volume inventory pin。空quota＋completed字串的原反例現在只能取得帶errors的inventory，不能初始化。真停止證據的語意必須由真人／實際獨立 reviewer 核對，SHA不會自行證明「已停止」是真的。 |
| F3：新 Volume ownership race | **危險路徑已移除。** 新 preflight 唯一 `Volume.from_name` 指向既有 `tiny-perceptron-course`，明確 `create_if_missing=False`，只讀檔與列metadata。沒有新的 named Volume、App、Image 或 Secret 建立／刪除。不是認可舊 check-then-create／delete-by-name。 |
| F4：HF cleanup 未保持 unresolved | **HF路徑已移除。** 沒有 HF token、HF寫入或branch cleanup。未來用的純函式 `resources_verified`／`finish`（`finance.py:312-335`）要求明確compute、resource驗證與非空且全True的cleanup checks，否則unresolved；我的cleanup=False反例被阻擋。P4仍須針對真資源重新審receipt內容，不能用任意True字串代替。 |
| F5：完整 provider 成本上界 | **P1不再有此付費邊界；尚未認證P4。** 沒有 Modal dispatch、CPU container、GPU、image build、新持續儲存或HF mirror。未來GPU的runtime、image、preemption、absolute deadline、storage／cleanup與費用上界另審，不藉這輪純client測試推論通過。 |
| F6：高水位歷史可丟失 | **修正通過。** `finance.py:145-239` 以origin digest綁初始化baseline／inventory／policy／revision，從baseline及每筆reservation identity綁定的實際snapshot重建month high-water、時間、總額與guard arithmetic；`legacy.py:64-79` 另重建舊epoch。我的刪high-water map反例不再可預約。這是偵測不一致／意外改壞，不是能抵抗repo owner全面重造資料的簽章。 |

## 新 GitHub draft 工具的發現與回修

**R1：第一版在 draft 建立成功但回應遺失時，可能虛報 cleanup 完成；ROOT已修，callback通過。**

第一版 `github_transport_probe.py:179-181` 在沒有記下 release_id 時，呼叫 `GET releases/tags/{tag}`，404就當作不存在。官方 GitHub Releases 文件的該節定位為「Get a published release with the specified tag」；官方 gh CLI v2.81.0 `pkg/cmd/release/shared/fetch.go:163-222` 明確另走draft lookup，因REST tag lookup不能找pending draft。我以fake server保留成功建立的draft並丟失POST回應，實際重現「draft還在但cleanup_completed=True」。

ROOT修後，最終 `github_transport_probe.py:106-121,130-131,150-164,197-208` 在POST前固定create_started；無ID時以authenticated release listing查找，最多5頁、每頁100筆，精確比對nonce tag、draft及body；找不到且POST可能已送出就留下 `creation_outcome_unknown=True`、cleanup=false。找到自己的draft才按實際ID刪除並GET該ID確認404；不符合ownership就保留。

獨立callback實際核對：失落回應找到自有draft後清理；列表暫時看不到則unknown；foreign body不刪；第3頁找到可清；5頁到頂仍找不到則unknown。清理成功也不會把失敗的roundtrip改成success。

本次新增的兩份primary來源收據在 [reference-receipts.json](recheck-01-artifacts/reference-receipts.json)：GitHub Releases官方文件 SHA-256=`72c9f3fb02cf4fea53ce01bbba12c7613093561ca5562778f9bcdbeff016498c`；官方CLI固定版source SHA-256=`35baab14f6263e1ca576a64fb23e9a0385c74db6ea058b82e1e2f24ba8411f7e`。第三方全文僅留`/tmp/omni-infra-review-recheck-01-external/`，repo只存收據。

## 傳輸、憑證與發布條件

最終工具的fixture含16 bytes binary weights、完整兩參數SGD momentum state、learning rate、Python RNG state、next_step與config，內部逐檔hash及外層archive SHA均驗證。gzip實際fixture小於32KiB；重新載入後再更新一次，與同環境未中斷更新的weights逐byte相同。我的獨立測試故意改optimizer或RNG，即使重算內部manifest，續算仍會失敗。

這只證明tiny fixture的傳輸程式與同環境接續核對機制。完整mock roundtrip通過不代表真實upload權限已證、正式A/C checkpoint exact resume已證，或公開anonymous download已證。後續必須保存真正Actions `github-transport.json`、run／revision與cleanup結果。

`command`使用capture_output與90s單次timeout，失敗只回傳通用訊息；probe只保存error type，stdout只列stage／success／cleanup／bytes。我以帶假credential與redirect URL的stderr驗證不會被印出。程式不讀HF／Modal秘密，GH token由gh正常使用。

workflow `omni-revision.yml:28-65` 的job環境沒有Modal token；只有inventory／initialize兩個條件step各自綁現有Modal credentials。`transport-probe`會跳過那兩步，不安裝Modal SDK，只執行標準庫＋gh的draft工具。固定同repo concurrency group繼續保留；它不是workspace全域鎖。

原初判已確認新workflow尚不在main，且GitHub `workflow_dispatch`要先在default branch生效。先merge main再從該main執行這兩個准用operation；不假稱未合併feature的 `--ref` 可以解決default-branch登錄。GitHub官方Create release文件亦提示：target commit若相對default branch改了workflow，token可能需要workflow修改權限；從已合併的main執行可避免把這個差異混入probe。

## 可以立即進行與必須等待的動作

- **可以：** commit／merge本報告列出的P1程式、測試與workflow；main生效後dispatch `inventory`，取得真raw ledger／quota、app／Volume metadata與official billing／rates。我的main fault injection明確驗證inventory不呼叫load_reconciliation、不走initialize source gate、不寫GitHub帳本或Modal資源；沒有live reconciliation不阻擋此步。
- **可以：** main生效後dispatch `transport-probe`，以標準Actions token測小draft資產。它不涉及Modal算力或額度。任何success=false、cleanup=false或outcome_unknown都要保留receipt並處理確定自有draft，不能宣稱通道已驗收或自行刪他人資產。
- **尚不可實際initialize：** 先根據真raw取得獨立核對的停止證據與完整reconciliation，再commit相關證據；`source_revision`會逐檔對照git show，包括舊finance／policy及reconciliation所有證據，才允許新budget branch CAS。不得用synthetic fixture初始化，不重設既有round，不寫舊額度。
- **尚不可GPU或任何Modal付費工作：** 三輪教材審閱與具體P4 runner／費用／checkpoint獨立審核仍是另外的門檻。本複查只放行上述純client／非Modal通道。

審閱程式屬原始bytes證據，沒有拿全repo formatter改寫。ROOT新增本輪父層`.ruff.toml`保留evidence，我另在infrastructure-review放更窄的局部規約；只排除此目錄的封存Python，不改原pyproject，也不放寬產品scripts/tests。局部`ruff check`通過；初判15檔的SHA仍全部匹配。
