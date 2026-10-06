# 教材資產存放：Git、LFS與Hugging Face

一份程式、一批練習題與一個訓練好的模型，是三種不同的東西。程式說明怎麼做；資料提供要學的例子；權重保存學習後的數字。把它們分清楚，才能知道自己現在需要下載什麼，也能把某次實驗完整交給另一位讀者。

可以把它們想成食譜、食材和做好的菜。食譜適合逐行比較；食材常是一整包圖片或聲音；做好的菜則要附上用了哪份食譜與食材，才能知道如何重做。

## 1. 哪些檔案放在哪裡？

| 內容 | 本課的保存方式 | 讀者怎麼使用？ |
| --- | --- | --- |
| Python、正文、Notebook來源、設定與檔案清單 | 普通Git | 與專案一起取得，可逐行查看修改。 |
| 可重建的小世界資料 | Git保存生成器、seed與設定 | 在本機生成，不必下載每一筆合成樣本。 |
| 固定版本的照片、文字與音訊資料包 | Git LFS | 按實驗下載指定包，再核對內容。 |
| 已選定的教材推論權重 | Hugging Face模型庫 | 依清單取得需要的模型與配套版本。 |
| 自己的下載快取、訓練中間檔與輸出 | 本機忽略目錄 | 保存自己的工作；需要分享時再挑選完整實驗。 |

第20章成品使用指定版本的官方圖文底座與語音辨識器。本專案的公開模型庫提供配對說明與來源記錄，下載程式再取得官方模型；沒有把兩份官方權重另外複製成自製權重。用到LoRA時，修正權重也必須與正確底座配對。

## 2. LFS保存的是檔案，不是縮小檔案

普通Git保存資料包的一張小卡片，叫pointer；真正的壓縮包由LFS另行保存。卡片包含實體內容的指紋與大小。收到卡片只表示知道要找哪個包，還沒有收到包裡的圖片與錄音。

本專案以`/assets/training/*.tar.gz`追蹤固定資料包；`data/`仍是本機解包與快取位置。可先跳過資料下載，取得程式：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
```

接著依[訓練資料入口](../assets/training/README.md)或[第20章資料說明](natural-assistant/v4/DATA.md)，只取得該項實驗的包。清單會指定版本、大小、SHA-256及解包後的逐檔資訊；SHA-256是完整內容的指紋，能核對檔案是否與指定快照一致。

LFS不會讓檔案本身變小。保存不同內容的版本需要更多儲存空間，讀者下載也會使用擁有者的流量。因此不適合把每一步訓練結果或完整模型快取都放進LFS。方案與額度可能改變，規劃大量下載時查[GitHub官方計費說明](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)。

## 3. 本課資料包有多大？

| 資料範圍 | 壓縮後大小 | 用途 |
| --- | ---: | --- |
| 基礎自然資料，8個分來源包 | 30,854,837 bytes，約30.85 MB | [30組正式小型實驗](course-experiments/README.md)。 |
| 第19章自行訓練v2，1個包 | 67,862,431 bytes，約67.86 MB | [有限多模態主線的訓練與驗收](selftrained/TRAINING.md)。 |
| 第20章照片、中文讀字與語音，3個包 | 146,543,309 bytes，約146.54 MB | [自然助理的微調與驗收](natural-assistant/v4/DATA.md)。 |

三列是不同資料範圍，`fetch_training_assets.py --asset all`只取基礎8包。v2的 `assets/training/selftrained-v2.tar.gz` 由[固定manifest](selftrained/v2-manifest.json)指定Git版本、包指紋與全部8,962個檔案，含12個JSONL與8,950個其他檔案（包含圖片、錄音與來源說明）；訓練／驗證／最後測試共有28,876／2,435／3,734題，題數不等於獨立素材數。下載與核對步驟見[本機訓練指引](selftrained/TRAINING.md)。第20章另有自己的清單與下載入口；其語音錄音用來驗證聽寫與聊天，沒有進入本課的LoRA更新。

這些數字是壓縮包的實際大小；解包後另有檔案與容器成本。模型權重、Python套件及運算記憶體不包含在表中。完整外部資料集可能更大，本課只發布有明確用途、來源與授權的固定子集。

## 4. 權重和續訓檔為什麼要分開？

推論像拿做好的模型回答問題，只需要配對的模型設定、權重與輸入處理工具。續訓則像從上次下課的地方繼續學習，還要保存更新器、步數、隨機狀態與資料進度；只拿到推論權重，不足以保證接著更新的每一步相同。

[公開課程模型庫](https://huggingface.co/birdhackor/tiny-perceptron-course-models)保留[各章歷史推論清單](course-experiments/public-models.json)與[舊合成整合清單](course-experiments/capstone-public.json)。第19章v2另固定於HF revision `979cdfacc588ad0536f1c64fff96f264571cf054` 下的 `selftrained/v2/`：MoE pretrain、MoE SFT、MoE joint、Dense joint四組輸出，共16個檔案，採MIT授權。每組包括 `model.safetensors`、`model-config.json`、`tokenizer.json` 與 `inference-manifest.json`，由本課自訂PyTorch類別和[chat入口](selftrained/v2-public-cpu-commands.md)讀取。

v2公開檔案沒有optimizer、RNG或sampler狀態，目前trainer也不把它們當成新訓練段的起點。自己重做用[本機stage wrapper](selftrained/TRAINING.md)，從隨機初始化開始，後續接自己的 `best.pt`；同段精確續訓則接自己的 `latest.pt`。不需要作者的私人Volume。[5.7](../course/chapters/05.md#5.7)介紹接續小模型，第20章另用自己的[LoRA訓練指引](natural-assistant/v4/TRAINING.md)。

完整訓練備份可以保存到自己的私有模型庫；公開時選定需要的推論里程碑，附上能力與限制。Hugging Face的儲存政策與帳戶方案見[官方說明](https://huggingface.co/docs/hub/storage-limits)，不要把公開模型庫當成無限制的私人備份。

## 5. 一張清單要能回答哪些問題？

分享資料或模型時，至少讓接收者知道：

- 這份檔案做什麼，對應哪個單元。
- 來源、作者、授權與做過的修改。
- 固定版本、檔名、大小與內容指紋。
- 配對的程式、模型設定、tokenizer或處理器。
- 用了哪些訓練題，以及哪些題只留給驗證與最後測試。
- 能力結果支持什麼，尚未測過什麼。

下載核對成功只證明取得相同檔案。答案是否正確、資料是否合適、模型是否真的使用圖片或前文，要另做相應的檢查，方法見[教材實作驗證](validation.md)。

想發布自己的資料包時，先保存來源與授權、建立清單，再上傳LFS實體物件和Git中的pointer，最後從新的快取重新下載核對。想發布權重時，固定Hugging Face revision，再把配套清單放回Git。這樣讀者拿到的是可核對的一組材料，後面的[網站發布](publishing.md)就只負責把教材與入口整理出來。
