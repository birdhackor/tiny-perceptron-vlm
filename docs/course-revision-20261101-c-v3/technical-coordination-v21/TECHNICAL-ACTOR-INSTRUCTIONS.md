# Source21 fresh technical actor 契約

這是派工格式與證據規則，不提供教材答案或預期判定。真 R 已結束，真 T OPEN 為 `technical-stage-open-v21-v1.json`；只做協調者分配的單一 lesson_id，不另派 child。

先讀 repo `AGENTS.md` 及 `docs/technical-review-guide.md`。逐主張核對目前分配的真 source21 切片、圖與必要原始證據；不作可選改寫。原判保存之前，不讀 reader 答案、peer 判定、舊技術報告或歷史 issue 清單。原碼、配置、原始測量、固定資料包與權威原文可在實際必要範圍內讀取；作者說通過不算證據。

每個 actor 只擁有一個真正 lesson_id。原判必須完整且有自己的來源初判；不能先寫格式空殼或先產生 current。先保存 `<協調者指定report目錄>/original-v1.json`，回報 path/SHA、verdict 與必要阻礙，等協調者真 parse/hash/native 收件後，才按同 owner callback 寫 `current-v1.json`。original 永不覆寫。current 必須對 original 的所有真 issues 留 `original_issue_resolutions`，每列有精確 `json_pointer` 與具體 `resolution`；沒有 original issues 則 `[]`。

使用 native schema1：`review_stage: technical`、本人完整 `reviewer_task`、`reviewer_context: fresh`、指定 lesson/source/source_sha256/figure_sha256、真 UTC `created_at`、完整 claims/sources/artifacts/issues 及五個 checks。必要主張未核實或有矛盾就 `revise`。不要用 reviewer-context、NA、空 claims 或作者已通過等欄位掩過真技術內容。

核對以本節實際內容為準，特別是新介面、神經模型與 ID-only 分工、輸入／樣本／token 分母、固定配置的採用規則、真正 CPU 示例與外推限制。沒有這些主張的導覽頁只核其真實導覽事實。變動以外的 source-identical 數學／研究與 qualified 成品可在核對 identity 和既有 immutable 證據後明示 retention；不要重做舊訓練或把保留證據稱為本輪新實驗。舊結論或 history 在自己原判真收後才釋放。

正式新 artifact 僅使用協調者指定的全新 `docs/technical-reviews/artifacts/c-v3-v21-*/`。native `_artifacts` 不接受其他根目錄。repository_code sources 可直接指 repo 內原碼／配置／JSON，附真全檔 SHA、version 與實讀 inspection_note；`docs/decision-product/v3/metadata/` 的八份固定 ZIP 原 bytes 與 `acquisition.json` 是原始來源，不是 execution。真正執行／推導／圖核對要保存小型原始收據，列 command/result/environment 或透明推導。不得把 ignored outputs、repo 外 ZIP、權重或原始訓練輸出直接列為 Git 內 native artifact；權重與大型輸出留固定包／Release。

必要新 C 證據候選位置：`scripts/course_product.py`、`tests/test_course_product_entry.py`、`examples/decision_product/`、`tiny_perceptron/decision_product/`、`docs/decision-product/v3/{public-release.json,qualification-summary.json,metadata/*}`。只在課文主張需要時查。固定本機 ZIP 可由 `/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip` 實核；需自行核指紋與 ZIP 原成員，不以檔名或作者結論代替驗證。公開附件 transport 尚未完成，不得宣稱本輪匿名下載成功。

用 shared 環境執行：`UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python ...`。不訓練、不開 final、不用 Modal、不發佈、不 stage/commit。不可修改課文／Notebook／17個產品 Python／舊證據／DP／active pointer／checker。

磁碟約僅 400 MiB 可用：不 uv sync／安裝／另建 venv／複製整 repo、ZIP 或權重／重建整站。既有332個 kernel 的原始執行收據在 `docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source21-v1.json`；它是候選原始執行證據，需實讀其 source identity 與限制，不是全局技術正確性結論。共享已有小型原始 probe，不為每頁複製大型輸入或整套重跑。

`check_original` 在 `scripts/omni_revision/review_setup.py`；介面是 `check_original(report, freeze_ref, lesson_id, parent_id, task, validator, api, original_source_snapshots=None)`，其中 validator=`scripts/check_technical_reviews.py`、api=`docs/review-tools/incremental_revision.py`。canonical freeze=`docs/course-revision-20261101-c-v3/source-freeze-v21.json` SHA `d9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea`；numbered freeze=`docs/course-revision-20261101-c-v3/numbered-source-freeze-v21.json` SHA `0db79be8a75904d464b0dd8245eefe93b0dd617b85583e79cd631d116fb598fd`。新 numbered parent 另由派工提供；canonical 的 parent_id 就是 lesson_id。current 用原生 `_validate` 核來源及 schema，不能改 checker 放行。
