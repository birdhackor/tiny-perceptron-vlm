# 中文 OCR 合法公開資料：v4 精選與重建記錄

這一輪採用 **NVIDIA 明確授權的合成文字資料＋Wikimedia Commons 逐張核對授權的真實照片**。兩者用途不同：合成行提供較多字形與連續轉寫練習；自然照片提供場景、直排、傾斜、磨損、夜景及尋找指定招牌的練習。合成背景圖片不等同真正拍到的中文招牌。

經獨立審查並凍結、可重建的資料：

- `docs/natural-assistant/v4/data/ocr-sources.json`：每張作者、個別授權、固定下載 URL、SHA、原始頁 ID／HDF5 index、crop 範圍與 split。
- `docs/natural-assistant/v4/data/ocr-labels.jsonl`：逐區域轉寫、閱讀順序、整幅／裁片任務及來源 family。
- `outputs/natural-v4/data/ocr/`：訓練與觀察用影像。
- `docs/natural-assistant/evidence/v4-research/ocr/`：原始授權頁／API bytes、採集和重建證據，受 `.gitattributes -text` 保護。

## 實際取得的內容

|來源|獨立來源影像|任務記錄|切分與限制|
|---|---:|---:|---|
|NVIDIA|153 個實際引用的原始合成頁（共取得 185 頁）|710 單行＋117 兩／三行裁片|全部 train；同頁不跨 split。|
|Commons|70 張實拍照片|62 個完整 train 區域裁片、85 個整幅定位轉寫、50 個有中文字 presence|50 張 train；validation／test 各 10 張整幅、各 1 presence＋1 指定區域轉寫。|
|共享 DOCCI|19 張已看過的自然照片|3 train／8 validation／8 test 無中文字 presence|原樣沿用 vision cluster family、split、作者及檔案 SHA，不重新分割。|

基礎資料合計 1,083 筆任務；不同任務與 crop/whole 多尺度監督不能當成新增的獨立照片。Root 另加的 presence 記錄由最終 assembler 計算，沒有灌入本表。Commons 正向區域由本 agent 看圖轉寫，再由另一位 agent 獨立看過全部 20 張 held-out、16 個 held-out negative、85 個 train 裁片及對應整幅題目。NVIDIA 轉寫採官方生成器標註，另一位 agent 抽查 20 單行／10 多行並看完 38 個高窄影像；依實圖記錄直排與旋轉方向，單行題目不強迫錯誤方向。這是 **AI agent 的獨立目視複核**，不稱人工 crowdsourced gold。

另有 `ocr-order-labels.jsonl` 補上 validation／test 各 3 筆自然多行依序轉寫：同一批 held-out 整幅照片，明確指定 2／3 行中文或文字區域，保留換行。作者重新實際看過 20 張照片後選定 6 題，第二位 agent 再看全部 6 張整幅及字形細節，逐字、定位與閱讀順序均接受。這 6 筆沒有增加來源照片、runtime 影像或訓練例子；與基礎資料一起是 1,089 筆 OCR／presence 任務。照片家族與原 split、bytes／SHA 保持相同；獨立複核原件與紀錄見 `evidence/v4-research/ocr-order-peer-review/`。這是小量多行**依序轉寫**觀察，字形錯誤也會造成 exact 失敗，不能直接當成純閱讀順序能力測量。

## NVIDIA：圖片與標註一起適用 CC BY 4.0

官方資料卡 pin：[`69696a1cc543ef3a0f8e9892a89c17293e915263`](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1/blob/69696a1cc543ef3a0f8e9892a89c17293e915263/README.md)。原始 README SHA-256：`4e39811b7105c1101b472917fde0a1c5e0a23d244e100954f4eb17128315d962`。

資料卡的 “License/Terms of Use” 原文：

> Dataset Governing Terms: Use of the dataset is governed by the Creative Commons Attribution 4.0 International License (CC BY 4.0).

另明寫：

> This dataset is ready for commercial/non-commercial use.

作者是 NVIDIA Corporation。`zh_hant` 公開資料共 2,214,304 頁，官方 train 有 77 個 shard；我們讀第一個 train shard 中的 185 個固定頁面，最終使用 153 頁上的合格裁片。該 shard 10,208,139,294 bytes，**沒有整檔下載**。官方整檔 LFS SHA 僅為 upstream metadata，不能聲稱我們驗過整檔。

實際取得採 256 KiB exact HTTP 206 ranges，首次 starter 25,942,046 bytes 加本次擴充 54,515,742 bytes，共 80,457,788 bytes；另外先前 3 頁探針 2,349,086 bytes。程式拒絕 HTTP 200 整檔 fallback。重建另有自己的 128 MiB 硬上限及 range receipts。

裁片採官方 quad 完整在圖內、最短邊至少 20 px、含 CJK、單行 2–64 字。多行只取同段落、連續、由上到下排列的水平行，每片 2／3 行，保留原換行；所有單／多行裁片還逐頁依官方四邊形檢查：裁片矩形若與未選取行有超過 1 平方像素的多邊形交集便跳過，確保 gold 只對應指定文字；不是拿旋轉字框的矩形包絡代替交集。這仍不能保證背景照片裡沒有未標註的文字，因此也保留目視抽查。**官方 `zh_hant` 桶實際含簡體及中英混合文字**；原文字面保持不轉換，不能把全部稱為純繁中。

## Commons：授權按照片核對

每張資料保留 Commons 檔案頁 ID、原始照片 SHA-1／可得的 timestamp、實際下載 bytes/SHA-256、artist、credit、license URL 及 metadata API 原文。採用的照片只有 **CC BY 3.0、CC BY 4.0 或 CC0**；來源圖片仍保留各自原授權，新增轉寫標註用 CC BY 4.0。

CC BY 允許訓練、裁切及再散布，條件包含保留作者／指定 credit、授權連結、來源及有修改的說明；裁片明列 crop 座標，縮圖明列 Wikimedia scaling。公開檔案不直接等同任意使用：本輪沒有以 Commons 全站或程式碼授權代替個別影像授權。[Commons reuse 官方說明](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia)與 [CC BY 4.0 legal code §3(a)](https://creativecommons.org/licenses/by/4.0/legalcode.en)原文皆留 byte proof。

兩個外站轉入的來源另查了 Commons 檔案描述 revisions：CC0 的社交距離告示附 `FlickreviewR ... status=passed ... reviewlicense=cc-zero`；懷舊餐廳照片附 `cc-by-3.0|lienyuan lee` 與 Panoramio review。兩份完整原文見 canonical `commons/file-description-*.json`。作者與原站／archive 來源也留在 metadata。

選取的精確抄寫不補被樹葉、畫面邊缘或鏽蝕遮掉的字，也不根據檔名猜牌坊名。若只指定完整可見的部分，題目明確指出該部分。新北市 welcome 及兩塊菸酒牌的相似家族留在 train；held-out 每題讀取整幅 JPEG，target 座標只是 gold 的證據，沒有把裁片當自然全圖測試。

## 未採用的候選與原因

|候選|查到的官方授權／可得性|本輪決定|
|---|---|---|
|[CTW](https://ctwdataset.github.io/)|官方頁明寫 annotations/models 屬 CSCG、images 屬 Tencent，兩者 CC BY-NC-SA 4.0；baseline code 多數 MIT。32,285 高解析圖、約百萬字；下載分多個 GB 級包，需填表。|NC/SA 不在本輪採用範圍；不能拿 code MIT 宣稱照片 MIT。|
|[XFUND](https://github.com/doc-analysis/XFUND)|官方 README “License”：CC BY-NC-SA 4.0。中文 train 149／val 50 個表單。|排除 NC/SA。不是已核實的 CC BY 4.0。|
|[RCTW](https://rctw.vlrlab.net/dataset.html)|官方說明超過 12,000 張自然照片／截圖，train 圖＋標註 7.6G、test 3.8G；取得的頁面沒有找到明確影像再使用授權。|授權未知，沒有只因可公開下載就採用。|
|ReCTS／LSVT|官方 RRC／Baidu 端點本次回 503，未成功取得官方條款。|無法驗證，沒有憑印象判定可訓練或可再散布。|
|ChineseOCRBench、ESTVQA 衍生集合|第三方 HF header 可寫 Apache-2.0，但上游照片／手寫素材來源並不因此取得相同授權；取得的 ESTVQA README 只給下載連結，未見照片權利明確授予。|不以衍生 uploader 的 metadata 代替原影像授權。|
|CC-OCR 過往外部觀察圖|程式 MIT 與 benchmark 照片 copyright 是不同的事。|不拿這些圖當 MIT 訓練影像打包。|

CTW 的確涵蓋自然場景字、低照明、遠距、平面／立體文字，也有專家標註；因此適合日後另有合適授權與預算時研究。本輪優先用權利清楚、實际已下載及查看的小集合。

## 可重建性與實際驗證

產品重建程式：`scripts/prepare_natural_v4_ocr.py`。固定 `pillow==12.3.0 h5py==3.16.0 numpy==2.5.3`；本地 research Python 3.13.5，Pillow zlib 1.3.1／libjpeg-turbo 3.1.4.1。圖片 bytes 與 PNG 都逐件驗 frozen SHA，碰到上游改版或 codec 差異就拒絕；不讀 tokens、不重新抽樣、不重建新的未審查標註。

```bash
python scripts/prepare_natural_v4_ocr.py \
  --sources docs/natural-assistant/v4/data/ocr-sources.json \
  --output /tmp/natural-v4-ocr-final-fresh \
  --local-sources /workspace/tiny-perceptron-vlm \
  --receipt docs/natural-assistant/evidence/v4-research/ocr/fresh-rebuild-receipt.json
```

全新目錄重建的完成狀態與逐件核對數字，以 canonical `fresh-rebuild-receipt.json` 為準；本地重建從已下載、逐件原 SHA 驗過的原始照片／頁面重新生成 PNG，不把現有裁片当 input，receipt 清楚標示 0 HTTP。此時 cloud egress 的三個網域連線遇 503，沒有宣稱 public fresh download 成功；預設 GitHub Actions 重建仍走固定公開 URL／bounded ranges 與同樣 SHA 驗證。Peer review 另保存自己的 snapshot／SHA。移除 `--local-sources` 便使用公開網路重建路徑，來源校驗與 PNG 校驗標準相同。這些是來源、標註與可重建性的工程證據，並不是模型 OCR 能力的測量結果。正式能力結果須由 root 的凍結測試後產生。
