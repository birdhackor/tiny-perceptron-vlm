# 獨立技術核對筆記

審閱對象為凍結原文與指定 SVG。沒有讀取其他角色報告、舊技術審閱、歷史問題或其他 outputs。
本次工具自行產生的執行紀錄已逐檔讀取，並將原始文字與雜湊保存在 `section-execution.json`。

## 目標與程式

本例明訂 token，不推論真實 tokenizer 會如何切中文。原序列有五項，輸入位置取 0、1、2、3；
目標存在的條件是 t+d<5，存在時為 tokens[t+d]。完整答案為：

| 已讀最後位置 | 距離 1 | 距離 2 |
| --- | --- | --- |
| 〈B〉 | 雨 | 停 |
| 雨 | 停 | 了 |
| 停 | 了 | 〈E〉 |
| 了 | 〈E〉 | 沒有目標 |

EOS 本身不再作預測位置，因為本例序列已終止。練習序列加入「又」後，位置 1 的兩份目標變成又、停。
兩份序列的全部列都以人工答案比對原始程式，精確相同；不涉及模型訓練或品質量測。

因果許可 j≤t 在 t=1 時只允許 j=0、1，即〈B〉與雨。embedding、逐位置 FFN 與逐位置正規化不會引入未來位置；
注意力使用這個許可後，各層在 t 的表示都只依賴不晚於 t 的位置。整段以矩陣運算一起算不改變這個依賴關係。
多位置計算是在不同 t 產生表示；多 token 目標是在固定 t 下改變 d，兩者獨立。

## 平行分支與生成

原論文的每個分支計算為 softmax(f_u(f_hi(f_s(prefix))))。分支 i 不把分支 i−1 猜出的 token 接回輸入；
核心與 f_u 共用，f_hi 各自獨立。這裡的同一位置圖只表達依賴與監督關係，省略分支內的 Transformer 計算與輸出投影。
這不保證各分支的最高分組合相容：取聯合分布 AX=2/5、BY=7/20、CY=1/4，
第一個位置的邊際最高分為 A，第二個為 Y，但 AY 的聯合機率為零。這是精確分數的手工反例，不是本課模型實測。

只用距離 1 分支得到下一項，再將選出的 token 接回前文，仍是普通自回歸。
使用多個分支提出草稿時，若要沿用主模型的 greedy 生成規則，還需驗證接受前綴的程序；
多個訓練目標本身不等於已實作這個解碼流程，也不直接支持加速倍率。

## 圖的實際檢視

已以 `view_image` 看過 PNG：
`outputs/course-review-20261019/figure-renders-04/3e567d3daba30b8ac31248ead82592f9e7ea4e0827e9bec34abda863be21d58b.png`。
PNG SHA-256 為 `93e4e4f1f0b7a07ae875f66d1086d19ae1329f988859f0d7483e14664c68c09e`，
並核對 `docs/course-review-20261019/figure-renders-04.json` 中此圖的收據。
PNG 只在原有位置檢視，沒有複製或登記為 artifact。

箭頭由〈B〉／雨進入共用核心，再分到兩條分支；左邊距離 1 對停，右邊距離 2 對了。
沒有把右邊分支接在左邊猜測之後，也沒有把綠色正確答案送回核心；底下注明答案只供核對。
圖是分類目標的示意，沒有用框大小表達參數量、成本或機率，不能讀成量測圖或保證答對。
已讀 SVG 原文並比對算繪標籤、左右位置、箭頭方向與正文；沒有驗證桌面或手機的完整網站頁面。

## 核對範圍

外部證據實讀固定版本的原文 HTML：MTP 的 Method、Inference、Experiments on real data、
Benefits scale with model size、Faster inference；Transformer 的 Model Architecture、
Encoder and Decoder Stacks、Scaled Dot-Product Attention、Multi-Head Attention、
Position-wise Feed-Forward Networks、Embeddings and Softmax；Blockwise Parallel Decoding 的同名算法段落。
來源 URL、版本、讀取時間、全文雜湊、最小摘錄與 curl 指令見 `source-receipts.json`；未保存第三方全文。
正文的必要前置與生成連結已讀，引用的無效目標處理段落也確有排除超出序列的距離目標。

未重現論文訓練、品質或速度實驗，也未測 GPU/MPS；本頁沒有需重現的實測數字或保證性能力主張。
沒有已駁回／不要重提清單可核對；本次不修正文、程式或圖，未見裁決標記或以歷史紀錄為理由的註解。
