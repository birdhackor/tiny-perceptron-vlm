# 逐節事實與技術審閱

本階段核對主張是否成立、數字是否算對、圖是否表達同一件事，以及引述來源和實驗能支持多大的結論。它與[讀者理解審閱](editorial-guide.md)分開：一段容易理解的解釋仍可能不正確，一段數學正確的解釋也可能讓初學者讀不懂。兩種審閱各有自己的報告與獨立審閱者，不能拿其中一種的 pass 代替另一種。

先完成使用者要求的正式實驗、分析與教材修改，再啟動這個階段。本文件與checker只是準備規約，不代表已開始或完成審閱。範圍是 `course/chapters/*.md` 與 `course/first-steps.md`、`README.md`、`training.md`、`glossary.md` 的所有編號 `##` 小節；目前共248節，其中26節來自這四份前置文件。checker依目前原文建立清單，不把248當作永遠不變的常數。

## 派工與獨立性

協調者為每節派一個全新的technical reviewer，`fork_turns="none"`，不能把多節合併給同一個task，也不能重用reader審閱task。只提供本節檔案與編號、它明確連結的必要前置，以及可查的候選來源庫與實驗證據位置。不要提供作者的論證歷史、預期pass或先前審閱結論。協調者核對實際作者派工紀錄，避免讓作者審閱自己寫的實質內容。

審閱者不必知道歷史作者task，不能為了填欄位猜一個。若協調者有可靠的作者task紀錄，可以提供 `author_tasks`；checker會拒絕作者task與reviewer相同。這個欄位是可選的，作者獨立性仍由實際派工紀錄負責。`reviewer_task`必須是審閱者自己的完整task名稱，`reviewer_context`填 `fresh`。checker能核對報告中的身分與唯一性，不能證明某個task真的執行過、真的讀過來源，或作者沒有偽造署名。

報告由該審閱者寫到 `docs/technical-reviews/<節號>.json`。作者修訂教材後，讓同一位審閱者重新核對；原task已無法呼叫時，再派一位新審閱者完整重審。作者與協調者不能只更新舊報告的hash或自行改成pass。舊版本可從Git歷史追查。

## 每節實際要做的核對

先列出本節的實質主張，而不是照段落複製全文。把定義與方法歸為 `concept`，具體算例與預期數值歸為 `numeric`，API、命令與本專案程式行為歸為 `software`，測得的品質、速度、記憶體與比較結果歸為 `empirical`。一個主張只放一件能核對的事；每個主張記錄原文位置、證據與支持範圍。

方法或研究概念核對原始論文、權威文件或原始碼。引用論文時提供HTTPS URL、實際讀取版本、公式或小節定位，說清楚它支持的是哪一步，不只放論文標題。PyTorch介面與後端行為核對目前使用版本的官方文件／原始碼，記錄版本、API或行號；`stable`網址可能改變，要記錄讀取日期與版本，必要時保存小份來源快照。不要把網站出處或論文名自動當成正確，仍要讀原文、條件與限制。

`docs/technical-sources/` 是找證據的候選庫。它的摘要、連結清單、其他代理的筆記不是權威證據，也不代表審閱者已核對原文。對外部來源要明記 `checked_original: true`、讀取版本與 `inspection_note`；找不到原文、無法定位或無法核實時，主張保留 `unresolved`，全節維持 `revise`，不要補一個看似可信的引用湊通過。

暖身數學的透明算例不必硬找研究論文。可以逐步手算，記公式、代入數字、預期值、實際算得的值與容忍差；把推導當作 `derivation` 證據。軟體設定與確定性的程式事實也不必硬加論文，但要核對相關命令／程式並實際執行，把輸出與版本保存為證據。一次跑通只支持該環境的行為，不能由此證明所有版本的平台契約或研究方法的一般性質。

數值主張的 `verification` 分 `hand_calculation` 與 `executed`。手算不能寫成實際執行過；執行要引用有command、result、environment的 `execution` artifact。驗證程式時核對輸入、遮罩、dtype、維度、分母、公式與生成結果，而不只看exit code。實測主張另列 `denominators`，依主張填有效token、樣本、更新步數、暖機／量測次數、seed或資料切分等；artifact應能找到完整配置與當次結果。CPU示範不能當GPU加速證據，單seed短訓不能當普遍品質保證。

圖解必須核對數字、方向、位置、尺度與正文支持的結論。必要時實際渲染SVG查看，並引用保存的render或核對記錄；checker的hash只證明檔案相同，不能證明圖看過或畫對。數字對、程式能跑、paper有引用，也不能取代限制條件的檢查：分清恆等式與浮點近似、玩具機制與訓練品質、儲存量與實際工作量、固定配置與外推結論。

## 報告格式

以下是尚未核實的格式示意，不是通過紀錄。例中的節號、路徑與定位都要換成當節實際內容；占位欄位不能作證據。

```json
{
  "schema_version": 1,
  "review_stage": "technical",
  "lesson_id": "14.1",
  "source": "course/chapters/14.md#14.1",
  "reviewer_task": "/root/technical_14_1",
  "reviewer_context": "fresh",
  "source_sha256": "填入目前小節原文的SHA-256",
  "figure_sha256": {
    "course/figures/rope.svg": "填入目前圖檔的SHA-256"
  },
  "verdict": "revise",
  "claims": [
    {
      "id": "c1",
      "kind": "concept",
      "statement": "填入待核對的具體方法主張",
      "location": "第幾段／哪段程式／圖中哪個標籤",
      "status": "unresolved",
      "evidence": [
        {
          "source_id": "s1",
          "locator": "填入實際核對的公式或小節",
          "supports": "說明原始來源支持哪部分主張"
        }
      ],
      "artifact_ids": [],
      "scope": "填入成立條件與不能外推的結論"
    },
    {
      "id": "c2",
      "kind": "numeric",
      "statement": "填入某個具體預期數值",
      "location": "程式輸出或正文算例",
      "status": "unresolved",
      "evidence": [],
      "artifact_ids": ["a1"],
      "verification": {
        "method": "executed",
        "expected": "填入待核對的預期值",
        "observed": "填入當次實際輸出",
        "tolerance": "精確相等或數值容忍差及理由",
        "details": "輸入、算法與比較方式"
      },
      "scope": "只支持這個配置與算例"
    }
  ],
  "sources": [
    {
      "id": "s1",
      "kind": "paper",
      "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
      "url": "https://arxiv.org/abs/2104.09864",
      "version": "填入實際讀取修訂版",
      "verified": false,
      "checked_original": false,
      "accessed_on": "2026-10-02",
      "authority_reason": "說明作者／出版方及與主張的關係",
      "inspection_note": "記錄實際閱讀定位、條件與發現"
    }
  ],
  "artifacts": [
    {
      "id": "a1",
      "kind": "execution",
      "path": "docs/technical-reviews/artifacts/14.1-output.txt",
      "sha256": "填入已保存輸出的SHA-256",
      "description": "此輸出核對哪個主張",
      "command": "當次實際執行命令",
      "result": "實際執行結果與預期是否相符",
      "environment": {
        "python": "實際版本",
        "torch": "實際版本",
        "device": "實際裝置"
      }
    }
  ],
  "issues": [
    {
      "claim_id": "c1",
      "status": "unresolved",
      "details": "目前尚不能核實的原因"
    }
  ],
  "checks": {
    "factual_accuracy": {"status": "revise", "details": "待逐主張核對", "claim_ids": ["c1"]},
    "numeric_verification": {"status": "revise", "details": "待算例核對", "claim_ids": ["c2"]},
    "figure_consistency": {"status": "revise", "details": "待圖文核對", "claim_ids": []},
    "source_verification": {"status": "revise", "details": "待原始來源核對", "claim_ids": ["c1"]},
    "limitations": {"status": "revise", "details": "待條件與外推範圍核對", "claim_ids": ["c1", "c2"]}
  }
}
```

所有通過的主張用 `status: verified`；無法核實用 `unresolved`，發現矛盾用 `contradicted`。後兩者都需要全節 `verdict: revise`。修訂後保留已解決問題時，issue填 `status: resolved` 與具體 `resolution`；仍有未解決問題不能pass。`checks`五個固定名稱都必須存在，各有狀態與具體details，不能用布林、空字串或一個「看起來正確」代替。

`sources`可用六種kind：`paper`、`official_docs`、`official_source`需要上述原始URL、版本、讀取與權威理由；`repository_code`改用相對repo的 `path`、全檔 `sha256`、`version`、`inspection_note`；`derivation`用完整 `details` 展開可追蹤推導；`execution`用 `artifact_id`引用執行紀錄。每個來源都要 `id`、`title`與真正核對後的 `verified: true`。概念主張不能只用本專案程式或一次執行來證明，透明數字與本專案行為則可使用相應直接證據。

artifact可用 `execution`、`derivation`、`figure_render`、`source_snapshot`、`code`，每項需唯一id、相對repo的path、全檔SHA-256及description。執行artifact另需實際command、result與非空environment，版本／裝置值用字串。保存小份輸出、算例、來源快照或既有實驗報告；大量原始資料與模型權重沿用既有artifact儲存規約。不要把僅存在 `/tmp`、無法再取得的輸出當持久證據。核對時checker只讀本地證據與hash，不自動下載來源、不自動執行artifact裡的命令。

## 不適用與版本核對

沒有圖時 `figure_sha256`用 `{}`，`figure_consistency`可填 `not_applicable`並說明本節沒有圖。沒有數字或實測主張時，`numeric_verification`可同樣記NA與理由。仍有方法、數學或軟體主張的節，其factual/source/limits檢查必須完成；暖身與軟體設定不是整節免審的理由。

純導覽或行政說明若確實沒有實質主張，可用 `claims: []`、`sources: []`、`artifacts: []`，另記 `applicability: {"substantive_claims": false, "reason": "具體說明本節只提供哪些導覽資訊"}`；factual/source/limits可記有理由的NA。含程式或命令區塊時不能整節NA。這是審閱者讀完之後的適用性判斷，不能由作者先替未讀小節批次填NA。

`source_sha256`與reader審閱相同：從本節 `## 編號 標題`起，到下一個 `## `之前，使用未正規化的UTF-8原文；包含空白與換行。`figure_sha256`必須覆蓋本節全部SVG引用，也可加入實際核對的必要前置圖；每張圖記相對repo路徑與原檔SHA-256。正文、圖、程式來源或引用artifact改變後，舊hash不再通過，需重新核對，不能只更新hash。

```bash
# 協調者在一節真正完成後可先核對該節；不會替它產生報告
.venv/bin/python scripts/check_technical_reviews.py --lesson 14.1

# 正式發布前核對目前全部編號小節
.venv/bin/python scripts/check_technical_reviews.py
```

最終報告把「實驗／程式驗證」、「事實與技術審閱」、「初學讀者理解審閱」分開列出。checker通過表示範圍、來源定位、證據格式、獨立身分聲明與目前版本一致，不能把它說成自動證明全部論述正確，也不能把代理審閱說成真人學生測試。
