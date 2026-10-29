# P1 作者修訂交接 v2

作者：/root/omni_p1_infrastructure。依ROOT新指示和已封存獨立初判F1–F6修正。未修改初判、原始snapshot、independent probes或舊policy／ledger。未commit／push／dispatch，未派任何付費工作。

## 審閱標的

- scripts/omni_revision/__init__.py
- scripts/omni_revision/finance.py
- scripts/omni_revision/legacy.py（新增）
- scripts/omni_revision/preflight.py
- .github/workflows/omni-revision.yml
- tests/test_omni_revision_finance.py
- tests/test_omni_revision_legacy.py（新增）
- ROOT自行負責 scripts/omni_revision/github_transport_probe.py；作者沒有修改此檔。workflow將呼叫其 --output 介面，source_revision將它列入必要committed來源。

實測：36 tests passed；ruff check及format通過。沒有Modal套件的系統Python可執行preflight --help，列出的操作只有inventory／initialize。

## 核心修正

F1：同時鎖兩道上限：retained official new increment＋全部新未釋放bounds＋精確舊未對帳carry＋proposed≤60；另sum全部新bounds≤60。未解釋gross increment不得被下一筆proposed掩蓋。加入「先保留59，gross增加59，再保留1」反例，現在拒絕。

F2：legacy.py以未變動的舊validate_policy核對舊政策身份，但不呼叫舊check_quota，也不拿舊USD30驗新工作。核對ledger marker、quota schema／round／policy raw及canonical SHA、sidecar IDs與ledger scoped IDs完全相同、carry-in immutable SHA、逐筆actual billing snapshot SHA、數額、時間先後；從舊baseline及全部quota snapshots重建monthly high-water。未釋放carry只等於舊sidecar全部bounds＋政策carry-in；歷史audit48.40不進carry。只有terminal字串仍不足：initialize另要求獨立審過、committed且SHA釘選的reconciliation，逐筆提供停止證據。

F6：新origin以SHA固定baseline／初始化legacy inventory／政策／revision；每次reserve從baseline和每筆SHA固定預約的實際snapshot重建高水位、時間、總額及當次guard算式，再比對stored map／total。刪map、改baseline、刪月份、改bound／snapshot都拒絕。新baseline包含舊policy最早月起的完整official summaries，不能低於重建舊monthly high-water。

F3/F4/F5：依ROOT決定，P1整個hf-probe mode和付費實作已移除，沒有App／Image／Secret／新Volume／Modal CPU或GPU boundary。舊findings完整保留，這是取消本次該路徑，**不是聲稱原paid probe已修到能安全執行**。正式P4 GPU image／preemption／重放／startup／shutdown／storage定價仍須另審。本次沒有任何一般完整費用envelope保證。

future finish：要求compute_stopped、resources_verified及非空cleanup_checks全部明確True。缺證據或任何check false留下unresolved，阻擋後續reserve；完成、失敗都不返還bound。P4 runner仍須針對真正自有資源定義並獨立核對這份receipt，不能用空「已結束」字串替代。

## 唯讀及初始化操作

inventory從client讀舊Volume兩個固定JSON、既有Volume清單、公開app state、official monthly summary／rates。raw bytes先保存，即使quota relational validation失敗也不丟掉已取的原件。沒有reconciliation也可做inventory，用來讓ROOT及independent reviewer先取得事實；錯誤／未知照實保存，不填猜測carry。

initialize讀committed independent reconciliation、其逐筆停止證據bytes並驗SHA，然後source_revision逐檔對git show確認新runtime sources／workflow、ROOT transport script、舊finance／政策、reconciliation與所有證據都是該commit bytes；才建立GitHub獨立budget branch。此步不會建立任何Modal運算或儲存資源。

新workflow保留同repo tiny-perceptron-course-gpu mutex；它不是全workspace lock。transport-probe step不接收Modal token、不裝Modal SDK，只跑ROOT的GitHub腳本；inventory／initialize步驟才綁現有Modal secrets並安裝client SDK。沒有HF token相關步驟。

## 待真inventory後由獨立核對者產生的reconciliation

預設路徑 docs/course-revision-20261029-omni/legacy-reconciliation.json；workflow可指定另一個docs相對路徑。

必需結構（下例全是示意，不能直接使用）：

    {
      "schema": "omni-legacy-reconciliation-v1",
      "reviewer_task": "/root/actual_independent_reviewer",
      "ledger_sha256": "實際raw ledger digest",
      "quota_sha256": "實際raw quota digest",
      "legacy_policy_sha256": "既有committed政策raw digest",
      "legacy_verification_sha256": "canonical SHA of inventory.legacy_verification",
      "existing_volumes_sha256": "canonical SHA of inventory.existing_volumes",
      "retained_bounds": [
        {
          "run_id": "每個實際quota/carry ID",
          "reserved_usd": "exact bound",
          "stop_evidence": [{"path": "docs/actual-evidence.json", "sha256": "actual raw evidence digest"}]
        }
      ]
    }

retained_bounds必須完整等於驗證出的quota/carry集合，不能刪減已完成者或拿歷史audit替代；既有Volume清單含name/object_id/created_at且已排序，需獨立承認它們是保留的既有持續儲存，後續清單有變就停止新預約。raw ledger/quota/policy任一指紋改變也停。

這份JSON不是本作者代替獨立審閱的工具。停止證據必須由ROOT／實際independent reviewer對真資料核對；目前尚無live raw inventory，作者沒有產生任何可initialize的reconciliation。

## 未驗證／保留界線

- GitHub CAS真正寫入、workflow default branch生效、Modal client新inventory、ROOT Actions transport probe仍由ROOT在獨立複查後執行。
- 官方gross是workspace級且可延遲；精確carry表示「依既有政策仍未釋放的bounds」，不是斷言所有這些費用尚未入帳。新guard故意保守重複保留。
- 停止證據bytes與SHA能固定被審材料，不能讓程式自行證明文字主張是真的；真實獨立核對仍必須完成。
- origin/reservation SHA驗意外改壞及不一致；CAS防競爭，不把它稱為能抵抗repository owner蓄意偽造全部歷史的簽章。
- P4 future reserve/finish的純函式測試不代替正式runner的成本、GPU、preemption、儲存和checkpoint exact-resume驗收。
- new named Volume、HF mirror均不在本P1範圍；完整weights＋optimizer/RNG改走ROOT已授權的GitHub外部資產路線，實際transport驗證未由本作者做。
