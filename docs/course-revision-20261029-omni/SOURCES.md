# 原始來源與資產授權準備

作者來源準備：/root/omni_p1_sources，2026-10-29T06:42:02.769907+00:00。這些實讀紀錄不代替完成後的易讀易理解／正確性／連貫性三輪獨立審閱。完整第三方全文在Git外的/workspace/work/reference；精確URL、版本、SHA、實讀定位及逐主張範圍見[source-claims.json](source-claims.json)，逐資產授權見[licenses.json](licenses.json)。未下載tensor、訓練、使用GPU／Modal、HF寫入或執行成熟模型。

來源位元組勘誤：上述初始紀錄的 `kev-model`、`kev-data`、`kev-readme` 三筆 SHA 與大小，對應原檔末尾多加一個 LF 的本機副本。核對固定上游原件時，請使用[勘誤中的原件 SHA 與大小](source-digest-corrections-v1.json)；原始紀錄保留供追查，程式版本與引用內容沒有改變。

## 已讀來源

| 來源 | 精確版本與實讀位置 | 教學用途 |
|---|---|---|
| Qwen2.5-Omni | [2503.20215v1](https://arxiv.org/html/2503.20215v1)，§2.1–2.4 | Thinker/Talker、40ms時間、2秒交錯、文字雙條件、blockencoders與DiT視窗 |
| 3B實際config | Hub f75b40e3da2003cdd6e1829b1f420ca70797c34e，thinker/audio/talker/token2wav | 3B實際零件；不把原報告7B分數搬來 |
| Qwen3-Omni | [2509.17765v1](https://arxiv.org/html/2509.17765v1)，§2.1–2.5 | AuT、80ms時間、角分配、文字表示解耦、同框MTP、Code2Wav |
| Qwen3config/card | Hub 26291f793822fb6be9555850f06dfe95f2d7e695 | 實際modeltypes、Thinker128experts/top8、帶條件BF16/FA2理論記憶體表 |
| Kev碼版 | Git 2074cc44256200bb37540103fa4572901faa10f3，PointerHead／DecisionModel／metrics／data | 計分、問題隔離、hybridcache、校準／資料增廣 |
| Kev-4B卡片／訓練設定 | Hub 6cfce5c2fa4b4bd64026336ab649c5ca78857d52 | English、T2.41、accepted65536/validated8192、baseSHA |
| VQ-VAE | [1711.00937v1](https://arxiv.org/pdf/1711.00937v1)，§3.1–3.2、eqs1–3 | 最近碼本、stopgradient、STE與三路loss |
| EnCodec | [2210.13438v1](https://arxiv.org/pdf/2210.13438v1)，§3.1–3.2、§4.6 | RVQ、EMA另配方、chunkstate／padding／normalization、RTF差異 |

## 作者需要保留的技術分界

離散碼的梯度：argmin硬選碼對encoder無可用重建梯度；STE另選近似反向規則。若採 z_st=z_e+(z_q-z_e).detach()，前向是z_q，反向對z_e為恆等；重建不由此更新碼本。VQloss ||stopgrad(z_e)-e||² 更新碼本；commitment β||z_e-stopgrad(e)||² 更新encoder。最小化重建用NLL或MSE，不照抄原eq3的likelihood符號成要最小化的正loglikelihood。EnCodec以EMA更新碼本是另一配方，正文／程式要一致。12.18固定碼本直覺與17.16學習式量化的分工合理。

兩代Talker：2.5接Thinker高階文字表示＋抽樣文字tokenembedding；3解耦相同高階文字表示，但§2.4仍明列歷史文字、當前streamedtext、多模態條件，不是完全不吃文字。3的MTP補同音框殘差碼本，應分畫時間軸與碼本軸，不是未來文字token。2.5的DiT有2blockslookback＋1lookahead，不能一概叫零lookahead；3才明列left-context-onlycausalCode2Wav。

輸入與輸出因果性：blockencoder先收足窗口；AuT1–8秒attentionwindow不等於逐sample因果。EnCodec除左padding，還保存轉置卷積未完成尾部，並避免跨時間統計normalization偷看未來。這些影響理解的操作要直接內建教材程式，不能讓學生只開Colab才看到。真chunk推論應驗未來碼擾動不改已輸出prefix與離線／逐chunk一致，整段輸出切片不構成此證據。

量測：API真回可播放chunk才報該API首段等待；整段generate完成時間不能替代。EnCodec的RTF是音訊長度/計算時間，Qwen3是計算時間/音訊長度，引用先統一定義。首段等待、全段完成、穩態速度、音訊內容與真人自然度分別需要證據。

Kev：4B實際config為24linear_attention＋8full_attention。KV以外還有conv/recurrentstate；普通attentionmask不能隔離DeltaNet。每題row可隔離問題，題內因果選項仍互動；pointer形式不保證重排等變。本課獨立候選／對稱none的保證依自己的結構與輸入約定。

溫度：evalhead可能已套T，另fit前還原raw，不改trainingmode繞過。正T保同題argmax，不保一般跨題信心排序。確定計算反例：A=[1,0,0]、B=[.5,0,-10]；T=1時p_max約.5761/.6224，T=.1時約.99991/.99331，順序反轉；這是標準庫softmax算例，不是學習成果。KevREADME450的cannotreorderconfidences過強，勿照引。coverage_at_error看當批正誤選覆蓋，是post-hoc曲線；固定部署門檻要在policy選後套final。source內的groupedfold不是完整sourceholdout。

版本：Kevadapter_config revision=null；training_config另外有base_revision=1001bb4d826a52d1f399e183466143f4da7b741b。runner必須強制basepin，head/tokenizer/adapter/code各自固定。卡片歷史139fdd94與現v1.0tag 6cfce5c2fa4b4bd64026336ab649c5ca78857d52 分開。此處config/card檢查不是tensorSHA／runtime驗證。

## 授權與使用

- Kev程式、Qwen2.5/3GitHub程式、Qwen3.5-4B-Base實際LICENSE已讀Apache-2.0；复制程式保留通知。adapter/head、資料與輸出另外記。
- 3BpinnedLICENSE為Qwen RESEARCH LICENSE AGREEMENT，§1(i)/§2限研究／評估，商用另需授權。本輪依已批准研究推論，不附其權重／衍生模型。§3再散布條件與§4(b)利用Materials／outputs改進distributedAI的Built with Qwen或Improved using Qwen署名各自保留；不能說輸出一律禁訓，也不能推論輸出一律繼承研究授權。本輪不以Qwen/Kev權重或輸出訓練A/C。
- Qwen3卡片實讀frontmatter為license:other、license_name:apache-2.0；本輪只公開資料分析，不下載／執行30B；不能把metadata閱讀冒稱tensor再散布審查。
- 論文公開不表示全文／圖自動適用本repoMIT；用原創解說、數值及圖，引用精確版本，完整PDF／HTML留Git外。
- 現有課程軟體MIT。新自製聲音／影格、生成器、權重與optimizer/RNG續訓狀態，在發布前各有provenance／license／版本卡；尚未產生的資產不標已發布。
- 研究輸出文字、聲音、metrics逐資產核對可公開性；不要把模型授權當成私人或第三方輸入也可公開。證據在cache清理前匯出合法公開包或保存合法重建方式。

目前沒有阻止原創主線＋自製資料＋從零訓練的已發現授權缺口。條件性停止點是3B用途越出研究／評估，或要公開未清理第三方內容；前者採授權相容方案，後者不進公開包。一般來源讀取不需使用者補新權限。
