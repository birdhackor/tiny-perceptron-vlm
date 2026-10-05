# 14.7 本人親讀與支持範圍

Reviewer: `/root/phase4_factual_coordinator/factual_14_7`，本輪 fresh 單節 reviewer。讀取日期 2026-10-05。

## 凍結輸入

`frozen/course/chapters/14.md` 是本次凍結的整章 raw UTF-8 input，SHA-256 `292b95b0fdd08459dde76bc6c237da58129852b35f7aabd8bd5f6ee7f88b37a3`；這不是後續修訂章版本的宣稱。
`frozen/14.7.md` 對應本節原 bytes，SHA-256 `a81c83ec38f12bc3cabb61c4cd0da670f06121c95e3d57bd9537452ec1fc6bb8`。
只審 14.7，非第一節，不代審導言。必要回讀為 14.1 整小節原教材（含原教材 details 中課程短訓比較與限制），其 raw bytes 另存 `frozen/14.1.md`。這些前文成績不作 14.7 證據，不做 14.1 的獨立通過判定。
所有本節引用圖只有一張，其 raw SVG 已凍結，已親讀 XML，且用 Inkscape 實际渲染并透過 view_image 查看。

## 來源查證

直接從教材所連的 HTTPS arXiv PDF URLs 下載三份原論文；以其原 PDF 首頁作者、標題及 arXiv 版本戳確認版本。沒有用候選庫的摘要取代原文。

- RoFormer，arXiv:2104.09864v5（首頁 8 Nov 2023）：親讀 PDF pp.2–5，§2.1 式1–2、§3.1 式11、§3.2 式12–16。Q/K 用於比對，softmax 注意力權重加權 V；成對旋轉及 R(m)^T R(n)=R(n-m) 支持相對點積。任意 m 可由三角式計算，公式不需要可學習的逐位置查表。這些公式不是長文章任務品質保證。
- Position Interpolation，arXiv:2306.15595v2（首頁 28 Jun 2023）：親讀首頁、導言後部、PDF pp.3–5 的 §2.1–2.3、式1–4及 Fine-tuning 段。§2.2 明說直接伸展訓練未見窗口可能發生注意力/語言建模惡化，且相近線索亦可能失效；不把某個模型結果普遍化成所有外推都失敗。§2.3 式4把位置從 [0,L') 壓到 [0,L)，支持本節的下一節方法預告；L=8,L'=16 時最後位置15縮成7.5，而不是已見整數7。這是連續座標區間，不宣稱所有新旋轉是已受訓練的整數座標。
- RULER，arXiv:2404.06654v3（首頁 6 Aug 2024）：親讀首頁/摘要、PDF pp.5–6 §3.3 後部、§3.4 與 §4 Models & Inference setup / Task configurations / Effective Context Size / Model Ranking Criteria / Main Results 起頭。§4 的有效長度是指定 13 任務及門檻下的測得概念，與 claimed context size 分開；支持軟體允許長度不能代替指定任務驗收。沒有重跑 RULER，也未在本節引用其特定成績或模型排名。

原論文文字的初次定位搜尋包含三份 downloaded PDF 的 extracted text keyword hits；真正作為主張支持的是以上親讀段落，精簡 snapshot 保留對應 pdftotext 行號。所有 snapshot 與 PDF 的 SHA-256 由 manifest 保存。

## 自己的核對

原稿沒有 Python fence、其他 fence 或課程實測結果。CPU `check_numeric.py` 實際執行成功：16 個字、位置0..15；8卡與12卡訓練假設的最遠距離分別7與11，延長後15；距離單位是 token index 間隔，分母不是字數16。SVG 的字/編號順序精確相等。
依 RoFormer 的成對旋轉與頻率公式，四維固定 Q/K 的有限位置及共同平移 dot identity 最大誤差 2.220446049250313e-16，容忍差1e-12。這只核數學，沒有證明模型會利用遠線索。另按 PI 式4核16個座標不删卡，最大7.5在[0,8)內。
Inkscape 1.4 的640x592渲染可讀；藍色0..7、橘色8..15、同句折兩行、兩個距離說明均吻合。Chromium 首次在未產生圖時中止；隔離 profile 重試 timeout 124。未完成 course 頁面桌面/手機 responsive 視覺檢查，沒有把它寫成已驗證。

## 真實讀取界線

已讀審查方法、schema、section_facts helper、clear-tutorial skill及 protocol；cloud setup 技能與其 onboarding reference 已讀，環境未改配置。
沒有讀舊 technical/reader report 正文、作者審閱、course-revision 來源筆記、額外作者結果檔或修正摘要。初次 filename-only rg 搜尋列到了舊證據目錄名，沒有讀其內容。original-paper/source-locators JSON 僅讀必要上層 keys/types及 locator records，沒有找到三個對應 id，因此改直接下載；沒有採用舊審閱結論。
不做訓練、完整模型評測、GPU、資料或權重下載；正式永久證據均在本資料夾，不複製 runtime workspace、不跟 symlink、不留 weights.pt。

## 判定

此版本沒有未解決的實質問題，14.7 可 pass。支持限於原稿的紙上位置例子、成熟方法機制與驗收區分。文中的16張卡沒有被寫成已訓練或已測模型。未驗證事項為上述頁面 responsive 畫面及任何實際模型任務能力，不是本節已提出的能力宣稱。
