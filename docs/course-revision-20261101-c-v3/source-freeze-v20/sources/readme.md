# 小小感知機 Tiny Perceptron

看到「今天天氣」，電腦怎麼猜下一個字？這套教材從幾個字的編號與接字表開始，用完整例子、圖解和短程式，逐步解釋文字、圖片與聲音模型的運作。

繁體中文 | [English](README_en.md)

**[線上教材](https://birdhackor.github.io/tiny-perceptron-vlm/) · [閱讀路線](course/README.md) · [操作與數學暖身](course/first-steps.md) · [全部小節](course/lessons.md) · [名詞頁](course/glossary.md) · [訓練操作](course/training.md)**

教材預覽版：理解需要的程式碼、關鍵輸出與解說直接放在正文，網站另呈現逐節CPU（電腦處理器）執行輸出，可以當書讀，不必先安裝；第一次從[1.1](course/chapters/01.md#1.1)開始。想自己改數字再跑，照下方步驟開同一節的Notebook，或使用網站頁尾的Colab入口。Colab是重跑實驗的方式，閱讀不依賴它；使用Colab需要Google帳號，其雲端執行尚未實機驗證。小程式不必先租GPU（圖形運算處理器）。

![文字與識別編號的對照](course/figures/rewrite-01-character-ids.svg)

## 教材提供什麼

20章與A／B延伸單元及C推理計算必修單元，共332個逐課小節，Notebook由同一份來源產生；閱讀指南、暖身、訓練操作與名詞頁另有26個編號小節，合計358個編號小節。每節聚焦一個問題，可以回跳，也不必一路訓練同一個模型。

只記「上一字後面常接什麼」的表，分不開結尾相同、前文不同的情境；所以先改讓可從例子調整的模型讀前文，再用問答示範練習回應要求與停止。若回答要用來選服務或做下一步，還得確認任務是否支援、資訊是否齊備；線索來自圖片與聲音時，則要保留內容與發生時間。材料變長、同一材料要問多題，或聲音要及早開始播放，又會碰到重算、等待與記憶體限制，於是再研究哪些計算可共用、輸出怎樣分塊，以及怎樣用較少資源保存與執行模型，最後整合成可逐項驗收的成品。

用幾個字看懂運作後，[7.17](course/chapters/07.md#7.17)解釋為什麼常先預訓練，再做後訓練；[第13章](course/chapters/13.md)逐步介紹偏好回饋、獎勵模型、PPO與DPO。指令遵循分開檢查內容、指定範圍、格式與停止；長上下文則分開討論容納長度、位置表示、計算成本與資訊使用。

[第19章整合專題](course/chapters/19.md#19.1)把自行訓練的文字、圖片、聲音與工具使用組成一位小助理，也就是 MoE 助理（產品A）。已發布的 v2 基線有 MoE 與較簡單的 Dense 兩種架構，文字核心、視覺／語音入口與接頭都由本專案從零訓練，已完成訓練、公開推論權重，並在兩版各3,734題的固定測試資料上評估。MoE共5,447,107參數，Dense共2,288,067參數；兩版仍未通過全部原定能力判準。任務限於三類服飾與兩格位置、12個已知中文字的指定連續1–4字區域、三種銀行客服語音主題，以及有限文字對話和計算器。完整成績與限制見[19.12](course/chapters/19.md#19.12)，先試用可走[公開CPU操作](docs/selftrained/v2-public-cpu-commands.md)。另一組合成小世界整合實驗的權重、配方與逐題報告在[固定實驗](docs/course-experiments/README.md)，那些是另一組模型的成績。

本輪新增19節，把候選判斷、影音時間與聲音生成融入主線：[7.26](course/chapters/07.md#7.26)與[9.17–9.18](course/chapters/09.md#9.17)教資料變化與可靠決策；[10.12](course/chapters/10.md#10.12)、[12.17–12.18](course/chapters/12.md#12.17)教影格時間、圖音線索與可還原聲音碼；[14.19](course/chapters/14.md#14.19)、[16.15–16.16](course/chapters/16.md#16.15)、[17.16–17.17](course/chapters/17.md#17.16)再教時間位置、共享計算、分塊輸出及碼本學習。這些方法在[19.21–19.24](course/chapters/19.md#19.21)接到MoE助理（產品A）的影音擴充、有限音調、機率判斷器的C-v2與自動選路（Auto）整合。這裡的A／C是第19章的成品代號，前面的A／B延伸與C必修則是教材單元編號。十個新模型工作已在標準Linux CPU完成真訓練與固定最後測試量測。影音任務問「聲音響起時畫面是圓還是方」：每個情境只移動聲音形成兩題，兩題皆答對才算一個配對成功，事前選定模型為200／200。有限音型生成則按指定頻率、次序、時值與停頓判整段音型，200／200通過；換波形內容的影音診斷仍有失敗。該批C-v2從候選動作中選答案（Choice），60題答對39題；固定規則接受其中53題，裡面19題答錯，未達升版門檻，當時保留研究候選及舊版預設。結果、限制與權重入口見[新版模型操作](docs/omni-products/README.md)，舊成績仍留在原版本。

目前裝置決策推薦使用後續完成的[19.25：C-v3](course/chapters/19.md#19.25)。它接受3–6個裝置的明確JSON欄位，從零訓練後，在1,500個未見4–6裝置事實家族上，Choice、資訊與固定規準評分各答對1,500／1,500；1,025筆可答動作全部接受、零錯誤，其餘475筆正確拒答。三型共用家族，不能當成4,500個獨立世界；三裝置另有手工檢查與舊題回歸。請用[合格C-v3下載與操作](docs/decision-product/README.md)，CPU即可；這個介面處理明確欄位的裝置請求；Looped解題器B仍用原本自動選輪數的策略（Auto effort），C-v2的神經路由則保留原版紀錄。理解所需的程式直接放在正文中。

[第20章](course/chapters/20.md#20.1)保留Qwen3-VL與Whisper的成熟模型應用，另在[20.14–20.16](course/chapters/20.md#20.14)對照Kev（針對事先列出的候選答案回傳機率的開源系統）及兩代Omni的公開實作、架構與驗證方法。既有Qwen／Whisper路線有[操作](docs/natural-assistant/v4/STUDENT.md)、[資料](docs/natural-assistant/v4/DATA.md)與[訓練](docs/natural-assistant/v4/TRAINING.md)指引；上游能力與自行訓練主線分開驗收。

Python、tensor、機率、矩陣與梯度有[按需暖身](course/first-steps.md)。各章的訓練命令集中在[操作頁](course/training.md)，來源與實測證據見[驗證範圍](docs/validation.md)。

Thinking effort（思考投入）、Looped Transformer（循環深度）及Jev式機率判斷已納入必修，分別見[7.23](course/chapters/07.md#7.23)、[14.11](course/chapters/14.md#14.11)與[9.11](course/chapters/09.md#9.11)。C.1–C.5接在第7章後。第19章的[三個成品](course/chapters/19.md#19.13)是產品A（MoE助理）、產品B（有限文字的Looped解題器）與產品C（獨立的機率判斷器），使用者依任務選用；機率判斷器還能接到 Looped 解題器，先判斷題目是否支援、資訊是否齊全，再依規則選 Looped 解題器要跑的輪數，這叫 Auto effort（見[19.17](course/chapters/19.md#19.17)）。既有Looped解題器與第一版機率判斷器已在NVIDIA L4完成訓練，設定凍結後才在測試資料上評估：Looped 解題器的兩盒追蹤答對 2928/2956、三數加法 2/42；機率判斷器在固定測試題的候選判斷上全對，但換一種說法就會出錯。固定下載與命令見[產品操作](docs/lesson-products/README.md)，完整分母、變化測試與限制見[19.20](course/chapters/19.md#19.20)。

## 安裝與開啟 Notebook

先安裝 [uv](https://docs.astral.sh/uv/getting-started/installation/) 與用來取得專案的[Git](https://git-scm.com/downloads)。訓練資料包用 Git LFS（Git 處理大型檔案的擴充工具）保存，下面第一行開頭的 `GIT_LFS_SKIP_SMUDGE=1` 讓 clone 不下載這些資料包；沒有安裝 Git LFS 也可以照打。Windows PowerShell 不寫這段開頭，改成先執行 `git config --global core.longpaths true`，再依序執行 `$env:GIT_LFS_SKIP_SMUDGE="1"` 與 `git clone https://github.com/birdhackor/tiny-perceptron-vlm.git`。專案裡有些檔的路徑很深，Windows 上的 Git 預設建不出超過 260 字元的路徑；設定 `core.longpaths` 的那一行讓這台電腦上的 Git 都允許長路徑，之後的 clone 與切換版本也適用。CPU閱讀與練習按下列順序進行。到了 `check_env.py` 那一行，先確認下方說明的結果，再繼續註冊kernel與啟動Jupyter。這個腳本列出Python與套件版本，選擇可用裝置，並在該裝置做一次小矩陣的前向與反向計算：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync --frozen --extra cpu --group notebook
source .venv/bin/activate
python scripts/check_env.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
jupyter lab notebooks
```

環境檢查應正常結束，套件各列出版本、沒有「未安裝」，且 `compute` 顯示所選裝置的 `OK`。以下節錄一次CPU實跑；版本與執行緒數依環境而異：

```text
torch           2.14.1+cpu
device          CPU（5 執行緒）
compute         OK（cpu）
```

這次檢查只確認該CPU上的基本矩陣計算；整套教材、其他後端與模型能力仍需各自核驗。

版本清單中的`torch`是PyTorch套件，用來做數字計算與訓練模型。

Windows PowerShell 啟用指令是 `.venv\Scripts\Activate.ps1`；不啟用的寫法見[W.1](course/first-steps.md#W.1)。JupyterLab 從 `notebooks` 資料夾列出檔案；選擇 **Tiny Perceptron** kernel（負責執行程式格的核心），先開啟其中的 `01/1.1.ipynb`。詳細操作見[暖身指南](course/first-steps.md)。只讀教材與跑各節短程式不需要 Git LFS；要下載訓練資料包時才需要，安裝與確認方式見[W.1](course/first-steps.md#W.1)。

上面的命令裝的是 CPU 用的 PyTorch，在 Apple Silicon 的 Mac 上也能用。要改用其他安裝選擇時，把 `uv sync` 那一行換成下表的寫法：

| 環境 | 安裝選擇 |
| --- | --- |
| CPU | `uv sync --frozen --extra cpu --group notebook` |
| Apple Silicon | `uv sync --frozen --group notebook` |
| NVIDIA CUDA 13.0 相容硬體／驅動 | `uv sync --frozen --extra cu130 --group notebook` |
| NVIDIA CUDA 12.6 相容硬體／驅動 | `uv sync --frozen --extra cu126 --group notebook` |

各版本的顯卡、驅動與套件配套選擇見[環境說明](docs/environment.md)。Colab操作另見[W.1](course/first-steps.md#W.1)。使用 extra 後，啟用 `.venv` 直接用 `python`，或每次 `uv run` 都帶相同 extra，避免更換 PyTorch。本機小節核驗使用CPU；前19章既有固定配方的GPU訓練曾在Modal的NVIDIA L4、PyTorch 2.14.1+cu126上實測，版本與結果保存於[實驗紀錄](docs/course-experiments/README.md)。第20章既有Qwen／Whisper路線另用Python 3.12與PyTorch 2.8.0的獨立環境；GPU訓練用CUDA 12.8配套，安裝步驟見[學生操作指引](docs/natural-assistant/v4/STUDENT.md)。Apple MPS已在macOS CI通過基本矩陣前向／反向與CPU比對；整套教材與正式模型訓練尚未在MPS驗證，範圍見[CI紀錄](docs/validation-artifacts/release-compatibility-ci.json)。

只想閱讀，可直接開啟線上教材。全站使用 [Zensical](https://zensical.org/) 建置，提供全文搜尋、章節目錄、頁內目錄、深淺色模式與程式碼複製；每節頁尾可下載 Notebook，或在 Colab 開啟同一份內容。SVG 可放大並保留靜態標註，支援系統減少動態效果。網站建置另需 `--group site`；本機閱讀與 GitHub Pages 操作見[教材發布](docs/publishing.md)。公式由網站隨附的 MathJax 在瀏覽器排版。

## 最小流程

先做一次通路檢查：第一行準備小段文字，第二行算出模型對接下來內容的機率預測，再與資料中的已知答案比較。`loss`（損失）把這個比較合成一個懲罰數字：給正確答案的機率越低，懲罰越大；在同一批資料上，loss越低表示吻合得越好。反向傳播接著算出梯度，表示模型中各個可調數字（參數）的小變化會怎樣影響loss；`grad_norm`是這些梯度的整體大小，用來看反向計算有沒有產生有限、非零的數值訊號。這裡只算訊號，不更新參數，也不保存權重檔。

```bash
python scripts/prepare_data.py --kind toy-text
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl
```

以下是一次CPU實跑報告的關鍵欄位，其餘欄位省略：

```json
{
  "mode": "dry-run-no-weight-update",
  "device": "cpu",
  "history": [
    {
      "step": 1,
      "loss": 5.770333290100098,
      "grad_norm": 0.9595053195953369
    }
  ]
}
```

確認 `mode` 為 `dry-run-no-weight-update`；逐步紀錄 `history` 第一筆的 `loss` 與 `grad_norm` 都是有限數字（不是 `NaN` 或無限大），而且 `grad_norm` 大於0。例中的5.77左右來自預測與答案的比較，0.96左右表示反向算出了非零梯度；梯度大小不是答對率。軟體版本不同時末位數值可能不同，重點是這些欄位的意義與條件。這次只核對CPU通路，沒有更新參數，也不證明學會回答、能答對新題或新增模型已有能力。

正式訓練再明確加 `--train`；字元模型、多模態、風格與安全、偏好和蒸餾配方在[訓練操作](course/training.md)。[首批固定訓練資料](assets/training/README.md)以 Git LFS 發布，下載前先依[W.1](course/first-steps.md#W.1)安裝 Git LFS；先用 `python scripts/fetch_training_assets.py --list` 查看名稱（這一步不需要 Git LFS），再用 `python scripts/fetch_training_assets.py --asset NAME` 下載、核對並解包所選包；將`NAME`換成清單中的名稱。原始快取與權重仍放在被 Git 忽略的 `data/`、`checkpoints/`；分工見[資產管理](docs/asset-storage.md)。

已設定 Modal 與 Hugging Face 帳號時，可手動執行 GitHub Actions 的 **GPU training smoke test**，驗證 GPU 權重更新、HF checkpoint 上傳與下載續訓。帳號設定、時間限制與結果判讀見 [GPU 操作說明](docs/gpu-training.md)。

想先觀察第19章的v2成品，可按[公開CPU操作](docs/selftrained/v2-public-cpu-commands.md)安裝 `cpu` 和 `selftrained` 配套，匿名取得[固定HF版本](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2)。該版有MoE pretrain、MoE SFT、MoE joint與Dense joint四組推論輸出，共16個配對檔案，採MIT授權。MoE joint選用最後一段第1,000步，該段實際完成4,000步；Dense joint選用第1,000步，該段完成10,000步。這些是依驗證選出的權重，成功示範不代表測試題全部通過。

要自己從隨機初始化重做，使用[本機訓練指引](docs/selftrained/TRAINING.md)和[固定v2資料清單](docs/selftrained/v2-manifest.json)：Git LFS資料包67,862,431 bytes，含8,962個檔案，訓練／驗證／測試題數為28,876／2,435／3,734。各段接自己的完整檢查點（checkpoint），不需要作者的私人Volume；公開safetensors推論權重不含最佳化器等訓練狀態，不能精確續訓。

各章的局部比較仍可下載[公開學生權重](https://huggingface.co/birdhackor/tiny-perceptron-course-models)：30組實驗共120份權重檔，包含不同尺寸與教師／學生等比較版本。[下載清單](docs/course-experiments/public-models.json)保留固定版本與檔案指紋；[合成小世界整合清單](docs/course-experiments/capstone-public.json)另保留11份階段、量化及Dense學生檔。例如在上面已安裝的 CPU 環境中：

```bash
python scripts/fetch_course_models.py --list
python scripts/check_course_models.py --model text_foundation
```

第一條只列清單，不下載。第二條匿名下載文字模型`text_foundation`、核對檔案並在CPU上推論，不必另給題目：它先用固定起始token核對各權重的前向計算，再按公開清單用文字提示`color=blue;shape=circle;`生成。本次該提示得到`side=right.`；這筆輸入與回答保存於檢查報告，不把它當成整體正確率。

終端機最後一行的重要欄位如下；這裡省略會依工作目錄與時間改變的`evidence`報告路徑：

```json
{"model": "text_foundation", "status": "passed"}
```

`status`為`passed`且命令正常結束，表示檔案核對、載入與CPU執行都通過；若是`failed`，先查看錯誤訊息與`evidence`報告。正常結束的exit code（結束碼）為0：macOS／Linux在該命令後立即輸入`echo $?`查看，PowerShell用`$LASTEXITCODE`。這是檔案與執行通路的檢查；回答是否正確要看對應實驗在驗證與測試題上的評估。推論權重已去除最佳化器與隨機狀態，不能拿來精確接續同一次訓練；教材的完整訓練備份另行保存。

## 專案結構與維護

```text
course/chapters/     教材正文與短程式，單一來源
course/figures/      自製 SVG
notebooks/          每小節一份，由正文產生
tiny_perceptron/    模型、資料、對齊、模態與壓縮零件
scripts/            資料、訓練、推論、評估與教材建置
tests/              離線契約測試
docs/               大綱、研究、環境與驗證紀錄
```

```bash
python scripts/build_visuals.py
python scripts/format_course_code.py course/chapters/01.md
python scripts/build_course.py
python scripts/build_course.py --check
python scripts/check_course_reviews.py
python scripts/check_technical_reviews.py
python docs/review-tools/check_review_round.py
python scripts/check_notebooks.py --mode python
python scripts/check_notebooks.py --mode kernel --lesson 3.6
pytest -ra
ruff check .
ruff format --check .
```

`pytest`執行測試；`ruff check`檢查程式碼規則，`ruff format --check`確認排版。

修改教材請改 `course/chapters/`，先格式化該檔的 Python 範例再產生 Notebook。每節依序由獨立讀者核對易讀易理解程度、另一位審閱者查核正確性與證據，再由銜接審閱者檢查前後連貫性，分別見[編輯說明](docs/editorial-guide.md)與[正確性審閱](docs/technical-review-guide.md)；版本不符會阻止發布。執行輸出與網頁放在 `outputs/`。`python`模式替每節使用新的Python命名空間，適合快速檢查；完整Notebook檢查用`--mode kernel`，每節另外啟動獨立Jupyter核心並保存執行結果。GitHub Actions 保留 Linux、macOS、Windows 的核心測試；Notebook整套驗證另行執行。

## 參考與授權

參考 [nanoGPT](https://github.com/karpathy/nanoGPT)、[nanochat](https://github.com/karpathy/nanochat)、[MiniMind](https://github.com/jingyaogong/minimind)、[MiniMind-V](https://github.com/jingyaogong/minimind-v) 與 [nanoVLM](https://github.com/huggingface/nanoVLM)。公開課與學習者回饋的來源、取捨與查證限制保留在[大綱](docs/curriculum.md)與研究筆記。

程式、教材與自製圖解使用 [MIT](LICENSE)。外部資料依各自授權。
