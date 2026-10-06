---
license: mit
language:
  - zh
library_name: pytorch
tags:
  - educational
  - custom-torch
  - from-scratch
  - multimodal
  - mixture-of-experts
  - safetensors
---

# 小小感知機：自行訓練模型與課程權重

這個模型庫配合繁體中文教材[小小感知機 Tiny Perceptron](https://birdhackor.github.io/tiny-perceptron-vlm/)。教材從接字表、小矩陣與注意力開始，逐步理解文字、圖片、聲音、工具與模型壓縮；[程式庫](https://github.com/birdhackor/tiny-perceptron-vlm)提供實作、固定資料清單、訓練配方和逐題結果。

第19章的主線v2由本課自行訓練：文字核心、視覺／OCR／語音入口和接頭都由本課從零訓練，沒有載入預訓練神經網路。它已完成訓練、公開推論輸出與MoE、Dense兩版各3,734題的固定最後測試；**兩版都沒有通過全部原定能力判準**。本卡保留實際成績與限制，適合學習有限任務的訓練、模態接入與交付。

第20章另有沿用成熟Qwen3-VL／Whisper的應用延伸；它的上游能力不是本課從零訓練的成果。原各章局部實驗及舊合成整合的權重也保留，各用自己的配方與結果。

## 自行訓練v2：四組固定推論輸出

公開revision固定為 **`979cdfacc588ad0536f1c64fff96f264571cf054`**，四組位於 `selftrained/v2/`，共16個檔案、81,501,640 bytes，採MIT授權。每組包含 `model.safetensors`、`model-config.json`、`tokenizer.json` 和 `inference-manifest.json`。請將四個檔案作為一組，由本課自訂PyTorch模型和 `scripts/selftrained/chat.py` 載入。

| 公開路徑與固定下載 | 用途 | 該段完成步數 | 選定步數 | inference manifest SHA-256 |
| --- | --- | ---: | ---: | --- |
| [selftrained/v2/moe-pretrain](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-pretrain) | 文字預訓練里程碑 | 1,000 | 250 | `6c5fbeb80491d32743ed6739b5e0dc0622d3d31b168b396af5677c2df35a6d15` |
| [selftrained/v2/moe-sft](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-sft) | 對話示範訓練里程碑 | 8,000 | 8,000 | `b909f7347b6f49e6ca746a32ec1e834afe6b95ec9fa38715f5bf54e194109a79` |
| [selftrained/v2/moe-joint](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-joint) | 最後選定MoE：native joint段 | 4,000 | 1,000 | `f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e` |
| [selftrained/v2/dense-joint](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/dense-joint) | 最後選定Dense：weighted joint段 | 10,000 | 1,000 | `dcb8ff538a958cd1a69b765056522e421df95b991566209aeb85046817596269` |

選定權重依validation比較保存，步數是該段中的位置，不是整條訓練歷史的累計步數。pretrain和SFT是中間里程碑，下面的最後測試表只評最後兩組joint權重，不能套到前兩組。

MoE最後段的tool／numeric／native-voice loss weight為4／1／4，Dense最後段為4／4／1；兩版的選定歷史也不同。完整12段baseline以及最後目標分支的來源見[訓練索引](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/v2-training-stage-index.json)和[最後權重選擇](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/results/v2-final-training-source-selection.json)。

## 模型與有限任務

兩版共用550個token的字元tokenizer、width256、4層、4個query heads、2個KV heads、FFN hidden512與512位置的context上限。tokenizer只由本課train資料建立；三類模態共享文字核心與對話紀錄。接頭把入口特徵轉成文字核心使用的向量。

| 參數範圍 | MoE | Dense |
| --- | ---: | ---: |
| 全部神經參數，含三個感知入口 | 5,447,107 | 2,288,067 |
| 文字核心全部參數 | 5,140,224 | 1,981,184 |
| 每文字token的active language參數 | 3,036,928 | 1,981,184 |
| 圖片／OCR／音訊入口 | 91,907／126,829／88,147 | 91,907／126,829／88,147 |

MoE每層保存4個專家FFN、每token選2個；Dense每層使用1個同hidden width的FFN。active計數包含全部共享文字權重及被選FFN，包括完整embedding／output矩陣；感知入口按實際輸入使用，不能全部當作每文字token的工作。這不是FLOPs、速度或記憶體測量。兩版並非同總參數、同active FFN數或同最後目標的受控架構比較。

任務限定如下：

- 文字：本課教過的有限中文請求、既有對話條件與指定格式，不是開放式中文知識助理。
- 圖片：Fashion-MNIST三類服飾及由這些素材組成的公開兩格位置圖，不是任意照片辨識。
- OCR：12個已知繁體字，使用者提供**一個連續1–4字的區域ROI**；不自行找文字、不讀整頁文件。Sans和Serif兩種字型都出現在v2訓練中，歷史欄位 `unseen_font` 不代表本版測到未見字型。
- 語音：MInDS-14中文錄音中的地址、App、卡片三種銀行客服主題，聲音特徵直接接文字核心產生文字回答；沒有執行ASR或用轉寫稿代替音訊。不是一般逐字聽寫、自由語音聊天或語音合成。
- 工具：模型產生有限計算器請求，由真正executor計算，再把結果放回對話讓模型回答。合法JSON、正確參數、工具實際執行與完整往返各自計分。

生成遮住只供輸入排版的BOS／PAD／role／modality ID，仍可產生EOS及可見UNK；沒有答案查表或替模型從問題擷取正確工具參數。操作所需的ROI與圖片位置是公開輸入，不是隱藏標籤。

## 固定資料與授權

v2資料清單固定SHA-256為 `3443e3d32ff1e63f8126c2327011be541238765824623af9c8fdcff6b056cb1b`。Git LFS包為程式庫commit `08761dac87a6ef360db95883d9bcd338c44fe76d` 的 `assets/training/selftrained-v2.tar.gz`，壓縮大小67,862,431 bytes，解包內容127,161,811 bytes，包SHA-256為 `0976073a3bc7c331a65cedb4c31d54f5c6e9a5014ff2443698a8e2b8cad7e78a`。

包內12個JSONL與8,950個素材共8,962個固定檔案，逐檔指紋在[manifest](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/v2-manifest.json)。訓練／validation／test各28,876／2,435／3,734題；錄音各62／15／30段。問法、格式與衍生材料可能共享來源，題數不是獨立素材數；錄音沒有已知speaker ID，不宣稱不同側由不同說話者錄製。

本課原創文字與工具題採MIT；Fashion-MNIST由Han Xiao、Kashif Rasul、Roland Vollgraf與Zalando Research提供，採MIT；[PolyAI MInDS-14](https://huggingface.co/datasets/PolyAI/minds14/tree/40ce77cb32a384e4d50a568e1ec39ac804019d33)中文錄音採CC BY 4.0；Noto CJK Sans／Serif字型採SIL OFL 1.1。包內保留來源、固定版本、授權及修改說明。公開模型權重的MIT授權不會取代原始資料或字型授權。

## 最後測試：完成執行仍有未達標能力

配置與公開safe權重先在2,435題validation側凍結，MoE和Dense再各做**一次完整3,734題test**；沒有用test繼續選版或訓練。完整原分母、判準、配對控制與評估完整性見[19.12能力卡](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/course/chapters/19.md#19.12)、[固定結果JSON](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/results/v2-final-public-results.json)及[完整性審閱](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/results/v2-final-public-results-integrity-review.json)。

| test用途與判準 | MoE | Dense |
| --- | ---: | ---: |
| 文字：內容符合語意規則 | 464/560 | 429/560 |
| 同批文字：完全符合目標字串 | 454/560 | 419/560 |
| 同批文字：指定格式正確 | 553/560 | 553/560 |
| 三類服飾：完整回答正確 | 357/360 | 356/360 |
| 兩格位置關係：完整回答正確 | 1,427/1,440 | 1,426/1,440 |
| OCR：指定區域整段完全正確 | 252/324 | 281/324 |
| 工具：正確請求、實際執行、正確最終回答的完整往返 | 0/276 | 7/276 |
| 已提供工具結果：最終回答正確 | 551/552 | 468/552 |
| 語音：有限主題回答正確 | 42/90 | 37/90 |
| 語音後續對話：真正第一輪回答接第二輪皆正確 | 26/60 | 24/60 |

表內有同批題的不同判尺，不能把列相加。完整JSON另列工具概念、缺資訊、不支援與工具不可用等題型；test共3,734題沒有丟掉失敗題。語音後續對話保留第一輪**真正模型生成**的回答，再送第二輪文字要求，沒有貼入標準示範回答。

OCR字元錯誤率CER是錯誤編輯數除以目標字元數，MoE為7.32%，Dense為4.37%；它與整段答對比例不同。原判準包含每文字意圖和每語音意圖至少80%、格式90%、各服飾／關係類80%、OCR整段80%且CER至多10%、工具往返95%及不支援情境澄清80%。總平均不能取代每個分層條件；兩版原判準仍未全數達標。

圖片換位、改ROI、改工具結果與改對話條件另有配對控制；輸出會改變也不保證兩題都答對。此測試只支持固定有限材料的表現，沒有一般中文、任意街景OCR、未見字型、說話者泛化、真實金融客服或通用安全能力的驗收。

## 在自己的CPU試用

先安裝Git LFS與[uv](https://docs.astral.sh/uv/getting-started/installation/)，取得含v2程式與操作頁的版本；在repo根目錄執行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch selftrained-v2 https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
git lfs install --local
uv sync --frozen --extra cpu --extra selftrained
git lfs pull --include="assets/training/selftrained-v2.tar.gz" --exclude=""
```

依[公開CPU命令](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/v2-public-cpu-commands.md)先用checksum-bound helper解開示範資料，再執行文字示範：

```bash
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/text.messages.json \
  --task text --device cpu --max-new-tokens 128 --threads 2
```

入口匿名取得固定revision的四個safe files，核對配對manifest與檔案指紋；不需要HF token、作者私人 `.pt`、optimizer或Modal。完整操作頁另有OCR、服飾、位置、真正工具執行、語音回答與保存history的兩輪對話。

這些是挑選的成功validation情境，已用公開權重完成CPU smoke：8次chat呼叫與1次history append，回答和保存的GPU validation示範相同。這證明當時環境中的下載、載入及介面可運作，不是未知問題的成功率，也沒有改變上表的工具／語音失敗。

## 從隨機權重重做自己的訓練

[本機訓練指引](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/TRAINING.md)使用 `scripts/selftrained/train_local_stage.py`，核對整個固定資料包後，依序執行自己的pretrain、SFT、vision、OCR、audio、joint與最後weighted／native分支。後段載入自己的selected `best.pt`；中斷後同段exact resume載入自己的 `latest.pt`，恢復optimizer、RNG、sampler與步數。每次attempt保存實際指令、檔案指紋、日誌和真實execution／training receipts，不需要作者私人Volume或原GHA run ID。

公開safetensors只供推論，沒有optimizer、RNG或sampler；目前trainer也沒有公開safe權重的training-init載入器。從它們推論，不能叫作精確續訓。全量配方與一個batch的教學smoke不同；本機wrapper工程驗證用width16合成CPU資料檢查成功、失敗、中斷及續訓契約，沒有重跑全量production training或另證模型能力。正文或圖解改版不要求重訓全書；實作、資料或能力宣稱改變時才驗收相應任務。

## 第20章：成熟模型應用延伸

[學生操作](https://birdhackor.github.io/tiny-perceptron-vlm/natural-v4-student.html)使用另建的Python3.12／PyTorch2.8.0環境，載入固定[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)與[Whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9)。照片與文字交給圖文底座；Whisper先把語音轉成文字，再放入同一聊天紀錄，回覆為文字。

本版依validation保留原底座，不加本課LoRA。圖文模型2,127,532,032參數、語音辨識器808,878,080參數；兩站能力來自上游訓練。其[固定公開配套](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/d3954d6900b3cf81e593d99d9b8b1a91e6f9741d/natural-v3/assistant-2b-v4)保留版本與說明，沒有重新散佈上游權重。Qwen採Apache-2.0，Whisper採MIT，來源資料另遵守各自授權。

第20章有限最後測試中，自然照片完整短描述25/42、同組可見事實58/84、自然中文字完整轉寫8/10、多區塊文字順序1/3、文字聊天2/13，4段真人問句真正ASR後聊天2/4。完整條件與不足見[20.13](https://birdhackor.github.io/tiny-perceptron-vlm/20.13.html)、[資料](https://birdhackor.github.io/tiny-perceptron-vlm/natural-v4-data.html)與[LoRA訓練指引](https://birdhackor.github.io/tiny-perceptron-vlm/natural-v4-training.html)。這些題目、規模與起點不同，不能拿來代替第19章從零模型的驗收。

## 原各章權重與舊合成整合索引

原30組正式局部實驗共120份模型存檔仍在本庫。下載由[public-models.json](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/course-experiments/public-models.json)固定各組revision、配對檔案、大小與SHA-256；使用 `scripts/fetch_course_models.py` 和對應實驗入口。完整recipe、原始評估及proof保留在[歷史實驗紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/course-experiments/README.md)。局部機制與歷史結果各有用途，不是v2成品的能力證明。

舊合成整合另保留[course-integration-v2權重](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed/course/course-integration-v2)：8份階段／量化檔及3份Dense學生檔，配對[capstone-public.json](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/course-experiments/capstone-public.json)。舊328k文字底座是合成小世界的局部機制與歷史輔助材料，不是上面的5,447,107參數v2 MoE。舊joint與DPO各78/90、舊Dense示範／蒸餾各62/90與61/90的結果仍保留，不換成新版最後測試分數。

下列入口各連到原固定版本的模型卡與檔案：

| 原實驗ID | 固定模型卡與檔案 |
| --- | --- |
| `simple_models` | [course/course-v1/simple_models](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1/simple_models) |
| `text_foundation` | [course/course-v1/text_foundation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation) |
| `real_text` | [course/course-v1/real_text](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/c8416bcf4d54cf40fbd270d3f636655a522310a0/course/course-v1/real_text) |
| `tokenizer` | [course/course-v1/tokenizer](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/58eb946c0eb9a37b8abb8d1930900685b42790bf/course/course-v1/tokenizer) |
| `sft` | [course/course-v1/sft](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/14293af762e79c5a65da5bdc5afed1bea68175a6/course/course-v1/sft) |
| `sft_ablation` | [course/course-v1/sft_ablation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/e30b15712b798cecacf47b0787a92590d2a7d876/course/course-v1/sft_ablation) |
| `style` | [course/course-v1/style](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/23b58c077a2da2ec6b94f4902e88510c0364e3fc/course/course-v1/style) |
| `lora` | [course/course-v1/lora](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora) |
| `safety` | [course/course-v1/safety](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/308aa207d6917d5dc8f9c819f2d7257d5cad7f8e/course/course-v1/safety) |
| `dpo` | [course/course-v1/dpo](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo) |
| `encoders` | [course/course-v1/encoders](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/93e45c77a7f6f21047bcd9f6e7ac3b5a7ad6c294/course/course-v1/encoders) |
| `contrastive` | [course/course-v1/contrastive](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/03cddf7bf0a2cdd69c1df876d3cff4b5e88cd621/course/course-v1/contrastive) |
| `projector` | [course/course-v1/projector](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/6d115860fc0af4669fccaa44bcdf2fad7c97e91e/course/course-v1/projector) |
| `vqa` | [course/course-v1/vqa](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa) |
| `vision_ablation` | [course/course-v1/vision_ablation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/71b6a22c6934bb6aec8ef92f9a0ce75b25f2aacf/course/course-v1/vision_ablation) |
| `ocr` | [course/course-v1/ocr](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/ce803a4c1eb75c428397fc4a49352354d9c82e8e/course/course-v1/ocr) |
| `audio` | [course/course-v1/audio](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/1c946fcb7f1d75948711962f22559b18d879e8b9/course/course-v1/audio) |
| `joint` | [course/course-v1/joint](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/b7908e15691515c15ab93caa9e71327b7ffda87b/course/course-v1/joint) |
| `real_modal` | [course/course-v1/real_modal](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/0c284d926ee729291fa023976fc03d9260db4068/course/course-v1/real_modal) |
| `modern` | [course/course-v1/modern](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern) |
| `moe` | [course/course-v1/moe](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe) |
| `efficiency` | [course/course-v1/efficiency](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency) |
| `precision` | [course/course-v1/precision](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/fb1d740aa140204efd41329a2c28e2a87daae6a4/course/course-v1/precision) |
| `quantization` | [course/course-v1/quantization](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/b41d097e4c10b29f6e6218728cbe1bafcc9db93c/course/course-v1/quantization) |
| `qat` | [course/course-v1/qat](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat) |
| `distillation` | [course/course-v1/distillation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation) |
| `multimodal_distillation` | [course/course-v1/multimodal_distillation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation) |
| `rag` | [course/course-v1/rag](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/660fa1d1fa67770b651c8397b1c4f44b11d200e5/course/course-v1/rag) |
| `tools` | [course/course-v1/tools](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/973d02736f4ebbfd7188e7f032fff1dfeddbc66c/course/course-v1/tools) |
| `reasoning` | [course/course-v1/reasoning](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning) |

## 授權

本課程式、教材、自製圖解及四組v2推論輸出採[MIT](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/LICENSE)。原各章模型卡保留自己的資料許可、再散佈限制與實際訓練範圍；不同資料包不可統一改稱MIT。第20章配套說明採MIT，上游Qwen／Whisper和來源照片、文件、錄音保留各自授權。

Copyright (c) 2026 birdhackor.
