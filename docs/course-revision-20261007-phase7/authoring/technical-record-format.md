# 技術輪記錄欄位提示

這是未填的格式說明，不是教材答案、來源核實或通過紀錄。內容由真實技術owner讀原稿、看圖和核對原始證據後自己填。完整要求仍以當批凍結的skill／protocol及infrastructure為準。

每頁除了group報告的共通欄位，另有四組欄位：

- `artifacts`：`id,kind,path,sha256,description`。正式檔案放在`docs/technical-reviews/artifacts/<自己的task>/`。`execution`再填真實`command,result,environment`；版本與裝置均用字串。
- `sources`：`id,kind,title,verified`。原論文／官方文件再填`url,version,authority_reason,checked_original,inspection_note,accessed_on`。`repository_code`填`path,sha256,version,inspection_note`；`derivation`填自己的`details`；`execution`填指向自己實際執行artifact的`artifact_id`。
- `claims`：`id,kind,statement,location,scope,status,evidence,artifact_ids`。`evidence`每項填`source_id,locator,supports`，不是只放URL。概念須有實際查過的原始權威來源；程式實作與數值也按實際支持範圍分開核對。
- `checks`：`factual_accuracy,numeric_verification,figure_consistency,source_verification,limitations`；每項填`status,details,claim_ids`。自己說明核對結果和範圍，不能用相同的通用句代替各頁判斷。

數字、軟體與實測claim另填`verification`：`method,expected,observed,details`。數字需`tolerance`；實測需列實際`denominators`。`method`僅接受`hand_calculation`或`executed`。軟體和實測用`executed`且引用實際execution artifact；讀取並核算既有原始結果可以支持歷史成績，不能聲稱重新訓練。

不確定或矛盾時保留`revise`與具體issue，不為滿足格式填`verified`。導航頁若確實沒有實質主張，可以填`applicability:{substantive_claims:false,reason:本人理由}`；有程式或命令的頁面不能整頁這樣略過。

技術輪的`question_refs`為空清單，不重填第一輪四題。每段仍保存自己的五欄、當時未知與真實觀看；每頁的必要圖、手機／桌面位置與表格必須本人看，產生截圖或讀SVG文字不能代替觀看。

這份提示沒有生成任何正式artifact、來源、claim、閱讀checkpoint或verdict。
