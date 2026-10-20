# 複查判斷與證據範圍

原始報告沒有 issue，故 resolved_issue_ids 為空；這不表示漏核原問題。first-steps 的現行原文與指定凍結原文 SHA-256 相同。原報告的 12 項 artifact 全部逐檔核對為相同 bytes，原本 114 項頁面主張全部保留；另完整列出三項參照章檔的新敘述作為本次變動核對。

## 變動的句子

- 「第二項的重要程度變成四倍，其梯度也變成四倍」成立。改的是係數 0.5→2，比值 2/0.5=4。固定 w=[1,2]，第二項從 2 變成 8，其偏導從 2 變成 8；第一項代價 4、偏導 −4 不變。現行原碼與改寫練習均實跑，總代價 12、梯度 [-4,8]。這是常數倍求導，不涉及一次梯度步能改善多少。
- 「感知器常用來指各輸入乘權重、加總，再經過一個轉折規則的小單元；把單元接成幾層就是多層感知器」成立。以 curl 直接讀 Cornell Virtual Workshop 的 The Basic Building Block: The Perceptron，原文寫出 z=Σw_i x_i+b、ŷ=g(z)，並使用 perceptron／artificial neuron 的現代簡化說明；D2L 1.0.3 的 Multilayer Perceptrons 原文給出 affine transformation 後接非線性，並定名為 MLP。這些證據只支持目前通用的簡化用法，不支持原句已刪除的歷史起源判斷，也不表示串接純線性層即可形成非線性能力。
- 「有限樣本通常有抽樣誤差，所以不必恰好等於理論值」成立。原程式的 seed=42、每個 d 各 1000 組、float32、torch.std 預設 correction=1 均保留實跑。d=4：未縮放 1.9595416784、縮放 0.9797708392；d=64：未縮放 7.7894287109、縮放 0.9736785889，接近 2、8 與 1。理論每格乘積的變異數為 1，D 個獨立項加總變異數 D，標準差 √D；樣本 std 是隨機統計量，不要求恰等於母體標準差。也不能說有限樣本必定不等：九個 Rademacher 樣本六正三負時，sum=3、sum of squares=9，correction=1 的樣本變異數 (9−3²/9)/(9−1)=1，實跑 std=1。這只檢查有限樣本限定，未做訓練效果比較。

## 引用仍成立的原因

逐一確認原始 report 的 repository_code 來源：只有派工明列的三個檔案改變；其餘直接引用檔案相同。三個 git diff 的基準檔 SHA 等於原報告所記值，目標 commit 的檔案內容與現行檔案相同。重新讀取目前的 nn、Embedding、copy_、Linear、矩陣乘法與 CausalAttention 的教學位置，位置與命名未改。故 python_class 與 warmup_curriculum_order 成立，報告改記現行檔案 SHA 並指向本次證據。

舊導覽 artifact 內提及的其餘章檔也有變動，所以不只更新三個直接來源的 hash：所有 16 個共用程式對照列及連結範圍均重新展開，逐個檢查現行說明位置與片段，將本次章檔 SHA 保存於 recheck-current-01.json。兩處敘述片段已不同，但仍在同一位置明確點名並說明相同函式：thin_stroke_report 的欄索引改成明說從零起算，AudioEncoder 的平均改成明說 adaptive_avg_pool1d 的區間取整與可能重疊。此處只核對 first-steps 導覽表所宣稱的實作說明位置；沒有擴大審查連結章節的其他技術結論。

## 重跑與沿用

verify-current-01.py 由原 verify.py 複製成新檔，只將凍結 manifest 路徑改成 freeze-06.json。原始工具、輸出、外部來源短摘錄、Notebook 獨立 kernel 執行與所有原始 artifact 都沒有改寫。重新執行全部 8 段 Python、表格、練習與不更新權重的一步 dry run；16 組確定性輸出欄位和原執行相等。原環境的 check_env 仍成功；核心版本與裝置見本次 execution。未重裝 Notebook 工具、未重新啟動 Jupyter；因其原碼與原證據相同，沿用原 kernel、伺服器 API 與安裝收據，不能將沿用寫成再次實測。

## 圖

本次再次使用 view_image 親看下列兩個 PNG，圖中數值、方向、位置、尺度仍與正文相同：

- outputs/course-review-20261019/figure-renders-04/74c93bfd89b0b164573f8dea1506179982c79250b77bfc862102895b02387138.png：兩列成績與每人／每考平均的分組、dim、結果均對應。
- outputs/course-review-20261019/figure-renders-04/e81b9f6de2af7b7d9d8bc9cbe32528794d207d1817014c46624d1619820ca942.png：w=1 到 1.4 的下降連線、L=4 到 2.56，以及 w=3 的最低點均正確；箭頭不是切線。

SVG 與前次相同，既有座標算式與誤差核對可沿用。PNG 沒有複製或列成 artifact。

## 獨立性與限制

只讀自己的原始報告及其明確引用的 artifact、指定凍結檔、目前教材與程式、指定 git diff 和新查的原出版來源。沒有讀其他角色的報告、處置、seal、閱讀計畫、提示詞或歷史審閱；沒有再執行全樹 git status。本次沒有新增被排除內容的暴露；原先 status 僅暴露路徑與修改狀態的紀錄如實沿用，不把它說成從未發生。

未驗證範圍仍為 Windows／Linux 端到端安裝、Colab runtime、瀏覽器 UI 與版面、全書重跑、全新空機的完整依賴重裝，以及先前 Notebook 關閉的程序查詢限制或 Node/yarn 啟動警告是否在其他環境重現。本次沒有新增模型品質或速度主張，沒有長時間訓練，也沒有觸發外部寫入、commit、push 或 workflow。
