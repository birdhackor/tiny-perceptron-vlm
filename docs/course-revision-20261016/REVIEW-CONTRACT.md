# 本輪審閱證據契約

本輪沿用clear-tutorial同一組教學判準。改用增量catalog是為容納新增頁與實際局部複查，不放寬必要缺口、獨立性或原始證據要求。

## 舊證據

`baseline_replay.py`在暫存副本復原舊來源、Notebook、判準及必要metadata，原樣執行`composite_review.py`。覆蓋前先解除硬連結，避免動到目前檔案；新sidecar只保存找不到相同既有副本的舊bytes。舊manifest、archive index及判定都不改。

新catalog保留全部舊canonical ID。只對來源／圖未變的頁沿用舊角色證據；Notebook也須相同。來源相同不能自動證明先備無影響，另由本輪銜接角色核對變動路線並明列未重讀全書的範圍。

## 首次閱讀

使用fresh context與作者分離的讀者；先給真正必要且實際讀過的前文，再逐段開正文。`tools/reader.py`先保存原始checkpoint，才開下一單位；四題按必要概念分項，第二題分機制與需要。整組正文結束後才補選讀。共享檔案系統是程序隔離，不能宣稱物理隔離。

`reader-sessions/<session>/opening.json`及`note-*.json`保存原bytes，state只是游標。catalog引用opening、每筆note與其凍結正文，不用重新生成的摘要代替原判斷。原始總報告中的`已足夠`與`pass`是等價的接受值；`需補說明`、`合理待教`、`未驗證`及有附帶未解事項的其他字串都不能由工具自動升成pass。

來源改動後，由原讀者另存知情複查，保留舊判斷及新來源hash；不稱新的盲讀。閱讀時間由實際讀完該頁的讀者本人給區間與理由，來源變更時由本人更新；作者不按字數公式或只換hash。

## 技術與銜接

首讀缺口修正與複查完成後，才開始技術階段。每個受影響的編號小節派全新的technical task，不合併多節，不沿用reader或作者。逐節報告沿用`docs/technical-review-guide.md`格式及`check_technical_reviews._validate`；直接查原文、推導與實際執行證據。新report與回查另存，原始判斷不覆寫。

閱讀指南／操作／名詞等整頁中，若只改部分編號小節，逐節比較舊凍結稿：未變部分沿用舊證據，變動小節必須恰好由新的逐節報告覆蓋；導言變動不可混過。沒有編號的章導言與首頁等，以canonical ID作獨立技術報告，不編造本不存在的程式證據。

技術處置完成後，由另一組未參與作者／reader／technical的fresh task做銜接初判；先保存自己的來源判斷，再交叉核對其他報告與舊問題。核對實際路線、必要前文、表示轉換、圖與手機／桌面頁面，原始與後補發現分開。作者對全部必要及可選建議逐項處置，必要缺口修後複查才結案。

三階段各保留實際開始與完成時點、角色派工、原始報告、處置及複查。gate檢查順序、身分與來源完整性，不替代協調者核對實際派工；不能靠改時間或填task名製造獨立性。

## 工程後回填

正式工程／訓練在上述三階段之後。新實測、命令、固定權重與成本回填會再改來源：另凍結版本，重新核對受影響正文與實際證據，不由作者只更新hash。最終gate同時鎖定目前來源、圖、Notebook、判準與實作；全部角色證據及未解處置不足時維持fail，不切換發布pointer。

AI首讀與估時不等於真人學生測試；截圖hash不等於實看；build成功不等於kernel執行；權重上傳不等於模型通過能力門檻。各項依實際紀錄分別驗收。

## 工具獨立核對後的補強

原始工具審核與反例保存在`reviews/tool-audit-original.json`，不改成pass。原始checkpoint當時的「需補說明／合理待教／未驗證」保留；本人另存逐項收束，連到原note的SHA與concept索引、後補依據及目前頁面。較晚的格式補證記真實建立與收件時間，列入技術階段收件，不倒填初始reader seal。源文修改的較晚知情複查同樣另記實際collection phase。

新helper操作須有作者事先凍結的reading plan：`reviewer_task`、`author_task`、`created_at`、`source_freeze`（path＋SHA）、`main_pages`及不重疊的`context_pages`。start與supplement皆傳`--plan`；必要前文可先讀，正文選讀必須等整組main checkpoints完成，另存補讀存取收據。已完成的舊session保留其真實程序與未具備此機械保護的限制，不事後偽造plan。圖依凍結副本與SHA顯示；共享檔案系統仍只能程序隔離。

gate另核對原始報告owner／page／來源、本人新版判斷及估時依據、派工範圍、checkpoint所屬階段、逐項必要問題複查與新增實作清單。Notebook必須仍由目前正文逐位元組生成；這個同步核對不等於獨立kernel執行。工具與證據的檢查只能保障一致性，不能自動證明真實閱讀或判斷正確。
