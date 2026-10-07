# 首批訓練資料：Git LFS 快照

此處發布8個分來源資料包，共29.4 MiB、1,447筆整理好的 training 紀錄；內容是已收集的教學 pilot，沒有模型訓練成果。程式與教材用普通 Git，`.tar.gz` 實體資料用 Git LFS，解包後放回被忽略的 `data/training/`。

第19章v2與第20章目前使用的資料各有專用清單，見本頁末段。`--asset all`只指首批8包，不包含這兩份成品資料。

| ID | Training 紀錄 | 資料授權 |
| --- | ---: | --- |
| `tinystories` | 512個完整英文故事 | CDLA-Sharing-1.0 |
| `chinese-poetry` | 365首古典中文詩 | MIT |
| `ultrachat-sft` | 100筆多輪對話 | MIT |
| `ultrafeedback-dpo` | 100組偏好對 | MIT |
| `pku-safe-rlhf` | 100組安全／有用性比較 | CC-BY-NC-4.0，非商用 |
| `fashion-mnist` | 50張圖與分類模板 | MIT |
| `gsm8k` | 200題人寫解答 | MIT |
| `fsdd` | 20段training語音 | CC-BY-SA-4.0 |

FSDD另保留各20段validation／test，原錄音是8 kHz，使用16 kHz入口前須明確重採樣。Fashion-MNIST包內另保留完整60,000筆來源training parquet；50筆是已配對的pilot，不是自然圖片caption／VQA。GSM8K解答不是教師模型生成的蒸餾資料。

各包保留來源、固定revision、原始授權證據與修改／選樣說明。`manifest.json`記錄壓縮包與每個檔案的大小、SHA-256；`sources/`保存來源metadata，這兩者用普通Git。資料依各自授權，不能統一當成此repo的MIT內容；非商用資料獨立成包，按需求選取。

## 取得資料

要在clone時先只取程式與LFS pointer：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
```

Windows PowerShell先設定 `$env:GIT_LFS_SKIP_SMUDGE="1"` 再clone。需安裝 [Git LFS](https://git-lfs.com/)，clone後在repo根目錄執行 `git lfs install --local`，並依主README安裝Python環境。啟用`.venv`後：

```bash
python scripts/fetch_training_assets.py --list
python scripts/fetch_training_assets.py --asset tinystories --asset chinese-poetry
python scripts/fetch_training_assets.py --asset all
```

只指定需要的ID即可。入口先取得對應LFS物件，核對包與逐檔雜湊，再解到`data/training/`。已有相同內容會跳過；已有不同內容會停止，可用`--output`另選目錄。它不會更新權重或執行訓練。

## 使用方式與範圍

不同資料包保留自己的上游格式：TinyStories／唐詩是 `text`，UltraChat 是 `messages`，UltraFeedback 是 `chosen/rejected`，GSM8K 是 `question/answer`。下載後不能把所有包直接交給同一個訓練入口；問題、答案、圖片與錄音要先轉成該實驗使用的表示。

本課既有實驗已各自處理這一步。下表給出配方名稱與學習入口；想重做，先讀[重跑一組實驗](../../docs/course-experiments/README.md#重跑一組實驗)，再以表中的 ID 選擇 `--experiment`。設備及前置權重依[實驗清單](../../docs/course-experiments/plan.json)準備；有前置實驗的項目，先完成它列出的前段。

| 首批資料包 | 既有實驗 ID | 對應的教材與操作 |
| --- | --- | --- |
| TinyStories／唐詩 | `real_text` | [5.10：完整文章的切分](../../course/chapters/05.md#5.10)、[T.4：文字與問答格式](../../course/training.md#T.4) |
| UltraChat | `sft` | [7.1：對話表示](../../course/chapters/07.md#7.1)、[T.4：對話訓練](../../course/training.md#T.4) |
| UltraFeedback | `dpo` | [13.1–13.2：同題兩份回答](../../course/chapters/13.md#13.1)、[T.7：偏好訓練](../../course/training.md#T.7) |
| PKU安全比較 | `safety` | [9.4：安全回應](../../course/chapters/09.md#9.4)、[T.5：拆開訓練目標](../../course/training.md#T.5) |
| Fashion-MNIST／FSDD | `real_modal` | [10.5：圖片入口](../../course/chapters/10.md#10.5)、[12.2：取樣率](../../course/chapters/12.md#12.2)、[T.6：圖片與聲音訓練](../../course/training.md#T.6) |
| GSM8K | `distillation`、`reasoning` | [T.10：教師與學生](../../course/training.md#T.10)、[C.1：可檢查的中間步驟](../../course/chapters/0C.md#C.1) |

各來源的實際轉換實作由實驗清單中的 `module` 與 `function` 指到 `scripts/course_experiments/`。這些配方會依任務整理輸入與標註；圖片不能只當類別字串，錄音也不能略過取樣率轉換。

自己開新實驗時，仍需按完整故事、詩、題目、圖片及近重複家族建立不洩漏的holdout。課程30組正式實驗已由配對入口建立各自的切分、轉換與指紋，詳見[實驗紀錄](../../docs/course-experiments/README.md)；包中的上游training身份與課程自訂留出側是兩件事。較偏好／較安全的回答不自動等於安全SFT正例。教材主線的離線小實驗仍用規則生成器，下載本資料不是執行各節Notebook的必要條件。

若要重建同一快照，先解包全部資料，再執行`python scripts/build_training_assets.py --source data/training --output outputs/rebuilt-training-assets`；壓縮包不包含時間戳差異，可比對manifest SHA-256。每次實質改動發布新版本，不覆蓋v1。

## 第19章v2資料：自行訓練的有限多模態主線

`selftrained-v2.tar.gz` 是固定Git LFS快照，大小67,862,431 bytes，解包內容127,161,811 bytes；包含12個JSONL與8,950個其他檔案（包含圖片、錄音與來源說明），共8,962個檔案。訓練／驗證／最後測試題數為28,876／2,435／3,734，題數不等於獨立圖片或錄音數。[v2-manifest.json](../../docs/selftrained/v2-manifest.json)指定Git commit `08761dac87a6ef360db95883d9bcd338c44fe76d`、包SHA-256與逐檔指紋。

文字與工具題由本課編寫；圖片取Fashion-MNIST三類服飾，另組成公開兩格位置圖。OCR使用12個已知繁體字與一個使用者指定的連續1–4字區域；v2訓練已包含Sans與Serif兩種字型。MInDS-14中文錄音限地址、App與卡片三種銀行客服主題；訓練／驗證／測試各62／15／30段錄音，沒有已知speaker ID，不能宣稱說話者隔離。文字與服飾來源採MIT，錄音採CC BY 4.0，字型採SIL OFL 1.1；各來源與原授權保留在包內，不能把全部素材改稱MIT。

只想試成品，按[公開CPU操作](../../docs/selftrained/v2-public-cpu-commands.md)取得示範素材與匿名推論權重；要從隨機初始化重做，用[本機訓練指引](../../docs/selftrained/TRAINING.md)核對整包資料，逐段接自己的checkpoint，不需要作者私人Volume。兩版固定最後測試與未達標項目見[19.12](../../course/chapters/19.md#19.12)和[完整結果](../../docs/selftrained/results/v2-final-public-results.json)。

## 第20章資料：照片、中文文件與真人語音

目前成品的來源、授權、固定下載與逐檔核對見[新版資料說明](../../docs/natural-assistant/v4/DATA.md)，重新訓練見[訓練指引](../../docs/natural-assistant/v4/TRAINING.md)。只想開啟模型時，直接用[學生操作指引](../../docs/natural-assistant/v4/STUDENT.md)，不用先取得全部訓練資料。

需要回查舊版v3資料時，可查看[舊整合清單](../../docs/natural-assistant/manifest.json)；目前第20章使用上方資料說明中的固定清單與下載步驟。
