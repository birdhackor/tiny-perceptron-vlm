## T.6 讓圖片與聲音先有可用的基礎特徵

先讀[10.5的視覺特徵](chapters/10.md#10.5)、[11.1的尺寸與理解](chapters/11.md#11.1)，音訊則先讀[12.8](chapters/12.md#12.8)。轉接頭只是把一排數字轉成文字模型所需尺寸；如果圖片編碼器還只產生隨機特徵，接上它不會自然得到看圖能力。

先讓兩個編碼器各練一個有標準答案的小任務：

```bash
.venv/bin/python scripts/pretrain_encoders.py --modality vision --train --steps 300 --output checkpoints/vision-encoder.pt
.venv/bin/python scripts/pretrain_encoders.py --modality audio --train --steps 300 --output checkpoints/audio-encoder.pt
```

視覺任務辨別合成形狀，聲音任務辨別高低音，均保留新組合做檢查。視覺保留兩張藍色、位移1的圖：方形標準類別0、圓形1；音訊保留180Hz低音類別0、1000Hz高音1。終端最後的JSON中，`mode`應為`train`，`holdout_examples`應為2，`holdout_accuracy`是這兩題答對比例；本流程要求兩題全對，即1.0，才繼續下一階段。0.5只表示答對一題，不能因已有保存檔就算通過。若未達標，先停在編碼器階段，回看10.5的逐題分類核對與12.8的形狀檢查，檢查題目、答案及更新流程後再測，不讓接頭訓練掩蓋前段問題。這只是兩道合成保留題的極小檢查，不是自然照片理解或語音辨識，也未保證300步就會達標。保存檔與檢查都符合後，搭配本頁 [T.4](#T.4) 的屬性文字模型，再練轉接頭：

```bash
.venv/bin/python scripts/train.py --task vision --checkpoint checkpoints/attributes.pt --vision-encoder checkpoints/vision-encoder.pt --freeze projector --train --steps 500 --output checkpoints/vision.pt
.venv/bin/python scripts/train.py --task audio --checkpoint checkpoints/attributes.pt --audio-encoder checkpoints/audio-encoder.pt --freeze projector --train --steps 500 --output checkpoints/audio.pt
.venv/bin/python scripts/train.py --task joint --checkpoint checkpoints/attributes.pt --vision-encoder checkpoints/vision-encoder.pt --audio-encoder checkpoints/audio-encoder.pt --freeze partial --train --steps 500 --output checkpoints/joint.pt
.venv/bin/python scripts/infer_modal.py checkpoints/vision.pt --color blue --shape circle --prompt "shape?" --tokens 16
.venv/bin/python scripts/infer_modal.py checkpoints/audio.pt --frequency 880 --prompt "pitch?" --tokens 16
.venv/bin/python scripts/infer_modal.py checkpoints/joint.pt --color red --shape square --frequency 220 --prompt "joint?" --tokens 16
```

三個推論問題已在命令明定：`shape?`問形狀，藍色圓形的理想回答是`circle`；`pitch?`問高低音，880Hz應為`high`；`joint?`同時問形狀與音高，紅色方形加220Hz應為`square,low`。終端輸出JSON，讀`answer`才是模型實際回答，`task`是所載入任務。例如`{"task":"audio","answer":"high"}`只是理想欄位示意，並非已完成500步的成績。逐題記下問題、理想回答與實際`answer`，若回答不同就記錯，不能只看命令沒有報錯。

這一組命令有順序：先完成 [T.4](#T.4) 取得 `attributes.pt`，再完成上面的編碼器，才有這些載入檔案。`--freeze projector` 只允許轉接頭更新，`partial` 還允許文字模型首末層更新，`none` 允許全部零件更新。實際更新仍取決於這次輸入：例如只用圖片與文字時，聲音編碼器與聲音轉接頭沒有收到梯度，就不參與這次更新。凍結與資料安排各自影響結果，要一次比較一個條件。

本節使用`--output`指定的原生`multimodal-v1`完整檔，例如`vision.pt`。它保存文字模型、圖片與聲音編碼器、接頭、配置、步數、優化器、隨機狀態與可訓練名單，可以載入推論，也供相同排程恢復進度。僅有權重的推論快照沒有完整恢復資料；選續訓檔時，要按內容與工具支援判斷，不能只看檔名。

若原排程還沒跑完，用相同任務、資料、總步數、批次大小與學習率，讓`--checkpoint checkpoints/vision.pt --resume`恢復已存步數後的部分；原先只訓接頭，就要保持同一凍結範圍。續訓時移除首訓命令的`--vision-encoder`與`--audio-encoder`：完整檔已保存它們，入口會拒絕一面恢復、一面重新載入編碼器。例如上面的圖片排程尚未完成500步時：

```bash
.venv/bin/python scripts/train.py --task vision --checkpoint checkpoints/vision.pt --resume --freeze projector --train --steps 500 --output checkpoints/vision.pt
```

這裡500是原排程的總步數，不是另外再更新500次；已經跑完的檔案不需要這條命令。載入工具會核對配置與可訓練名單。這不表示任意權重檔都能精確續訓：只含編碼器權重的檔案與其他僅供推論的快照，都沒有這項保證。換成另一訓練階段則載入原生檔、不要加`--resume`，再明定新的凍結範圍。

純文字能力檢查要對包裝內的`model.language`使用同一批留出題，原理見[11.7](chapters/11.md#11.7)；原生多模態檔不能直接交給只接受文字模型的`infer.py`。在專案根目錄執行以下Python，沿用T.4準備的同一份屬性validation，並將文字底座與圖片模型裡的文字部分並排評估：

```python
import json
from tiny_perceptron.data import load_jsonl
from tiny_perceptron.training import load_checkpoint
from scripts.evaluate import evaluate

records = load_jsonl("data/generated/attributes-sft/validation.jsonl")
before, _ = load_checkpoint("checkpoints/attributes.pt", "cpu")
after, _ = load_checkpoint("checkpoints/vision.pt", "cpu")
for label, language in [("before", before), ("after", after.language)]:
    report = evaluate(language, records, mode="sft", max_new_tokens=24)
    print(label, json.dumps(report, ensure_ascii=False))
```

`load_checkpoint`返回模型與保存資訊，`_`表示這裡不使用後者。`after.language`只取多模態包裝裡的文字模型，沒有餵圖片；`evaluate`以同一份題目與24個新token上限評估，並返回可逐題查閱的JSON。保存兩行輸出，按`samples`的`row`並排理想`target`與實際`generated`，再比較`exact_match`及它在`metric_denominators`中的分母；正常結束另看`eos_rate`。這段不更新權重，也沒有保證兩行會答對幾題。要檢查聲音或聯合訓練後的文字部分，只換第二個載入路徑為`audio.pt`或`joint.pt`，保留同一文字題目與生成設定。

### 重跑第11章的固定實驗

上面的500步CLI是自己練習圖片、聲音與聯合任務的路線。第11章的正式比較用另一套固定資料與工具，沒有把這三條命令的成績套進表格。兩處都叫`partial`，實際開放範圍如下：

| 使用入口 | `partial`開放什麼？ | 兩層文字模型會開幾個區塊？ |
| --- | --- | ---: |
| `scripts/train.py --freeze partial` | 圖片與聲音接頭，以及首末文字區塊。 | 兩個；字表與輸出表仍凍結。 |
| 固定`vqa`實驗 | 圖片接頭，以及最後文字區塊。 | 一個。 |

名稱只是選項的代號。第一條適合自行安排聯合練習，第二條才對應[11.5的「接頭＋最後文字區塊」](chapters/11.md#11.5)；它們的任務、資料、更新量與起點也不同，不能把表中的12/12當成一般CLI的預期輸出。

要重跑固定比較，先完成[T.4](#T.4)的正式`sft`實驗，在預設`outputs/course-experiments/course-v1/sft/`保留`model.pt`與`dataset.json`。再按依賴順序執行：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment encoders --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment projector --device cuda
.venv/bin/python -m scripts.course_experiments.run --experiment vqa --device cuda
```

`encoders`準備固定合成特徵，`projector`只練圖片接頭描述，`vqa`再從同一份接頭起點比較圖片問答。輸出依序放在預設目錄的`encoders/`、`projector/`與`vqa/`；不同輸出根目錄要用`--output`與`--dependencies`一起指定，讓工具找到真正的前置檔案。

正式編碼器各訓練250步，視覺六類新位置測試6/6，音訊14段新頻率測試11/14，見[10.5](chapters/10.md#10.5)與[12.8](chapters/12.md#12.8)。這是另一份考卷，不是前面CLI兩題的`holdout_accuracy`。描述對齊的完整起點、300步配方與生成反例集中在[11.3](chapters/11.md#11.3)。

`vqa`把相同圖片分別問顏色與形狀，各版本再更新160次。可訓練範圍與新圖片成績見[11.5](chapters/11.md#11.5)，原文字能力與回放比較見[11.7](chapters/11.md#11.7)。回放0.5指每題抽到文字的機率，不代表一半有效目標；比較結果時，把圖片與原文字用途一起檢查。

另用18,238個有效回答目標的兩階段預算，與直接問答18,256個比較，最後圖片題9/12對10/12；直接路線沒有另測文字保留，不能由此選出兼顧所有能力的勝者。這些計數讓你能沿著[11.6](chapters/11.md#11.6)重算預算，也提醒步數、參數多寡與單一新任務高分，不能代替完整用途的選擇。

真實資料實跑另保存`fashion-mnist.pt`與`fsdd.pt`，各從直接SFT底座接新編碼器，全開250次；前者30張教學圖、後者jackson的20段教學錄音，每次四筆。它們不是上面預訓練合成編碼器、凍結接頭500步的CLI配方。服飾只在驗證4/10、測試3/10，真人數字驗證5/20、測試3/20；有效回答目標各7,528與2,000，總參數145,664與148,736，後者多了較長錄音所需的位置表。參數較多與訓練探針代價很低，都沒有保證新題答對。

這條真實資料路徑把服飾28×28灰階轉RGB並縮至16×16、除255；音訊則先將原8kHz PCM16（每個樣本用16位元整數表示）真正重採樣成16kHz FLOAT（浮點數樣本），再按16kHz算log-mel。沒有另做每張圖的平均／標準差像素歸一化，也沒有每段錄音的峰值／平均能量歸一化；特徵內的[LayerNorm層正規化](chapters/04.md#4.3)是模型中的另一層處理，不能拿它代替輸入說明。原圖或原聲音變得更複雜時，這套小尺度入口可能已經失去所需線索，[10.5](chapters/10.md#10.5)與[12.8](chapters/12.md#12.8)保留了實際錯例。

你也可以先下載[T.3](#T.3)的公開`ocr`與`real_modal`權重，依模型卡的`--image`、`--audio`範例檢查自己的CPU入口。我們實際用數字圖片得到`42`、用第一張服飾測試圖得到`Ankle boot`；第一段數字0的測試錄音卻答成`8`。後者直接讀原8kHz檔並加`--resample-audio`，有正常結束，但答案仍然錯。這三個單次輸出只確認檔案能走過圖片／聲音讀取與回答流程，不能取代整份測試集的答對率；輸入指紋、命令與原始回答見[操作紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/student-checks/media-and-raw-adapter-cli.json)。

固定實驗的編碼器檔與上面的原生模態檔格式不同，本節CLI沒有對前者提供完整精確續訓入口。能載入其中的權重，不等於能恢復原來的更新器與隨機進度；要精確續訓，使用自己命令指定的完整原生檔及相配的配方。

如果改用自己的資料，一行可以寫成：

```json
{"image":"images/12.png","question":"read digits","answer":"12"}
```

圖片路徑相對於 JSONL 所在資料夾，入口會轉成 RGB 並縮到 16×16。音訊需非空、16 kHz 單聲道可解碼檔案，不會自動重採樣。這些小尺寸是為玩具任務設計的；字體或自然照片的細節可能已在縮圖時丟失，增加步數無法找回。

數字圖片的原理見[11.12](chapters/11.md#11.12)。要做正常圖、空白圖與錯配圖練習，先另準備OCR資料，再訓練能回答`read digits`的圖片模型；前面的形狀模型沒有學過這個問題，不能直接拿它當OCR模型。

```bash
.venv/bin/python scripts/prepare_ocr.py --output data/generated/ocr --seed 42
.venv/bin/python scripts/train.py --task vision --data data/generated/ocr/train.jsonl --checkpoint checkpoints/attributes.pt --vision-encoder checkpoints/vision-encoder.pt --freeze none --train --steps 500 --output checkpoints/ocr.pt
.venv/bin/python scripts/evaluate_modal.py checkpoints/ocr.pt --data data/generated/ocr/validation.jsonl --ablation none --limit 30 --tokens 16 --seed 42 --output outputs/ocr-normal.json
.venv/bin/python scripts/evaluate_modal.py checkpoints/ocr.pt --data data/generated/ocr/validation.jsonl --ablation blank --limit 30 --tokens 16 --seed 42 --output outputs/ocr-blank.json
.venv/bin/python scripts/evaluate_modal.py checkpoints/ocr.pt --data data/generated/ocr/validation.jsonl --ablation shuffle --limit 30 --tokens 16 --seed 42 --output outputs/ocr-shuffle.json
```

產生器保存`train.jsonl`、`validation.jsonl`、`test.jsonl`與相對圖片路徑，三種位移的同一數字留在同側；固定seed42的驗證檔有30筆。訓練只讀train，三次檢查都讀同一份validation、同一模型和生成長度；`none`保留原圖，`blank`用全零圖，`shuffle`換入另一筆的圖但保留原問題與原答案。終端只有總覽；開啟三份輸出JSON的`samples`，按`row`並排`target`、`generated`及`exact_match`，從同份validation的第`row+1`行找到原圖與`question`。

已完成的[正式OCR配方](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/ocr.json)使用同樣固定字型與按字串分組的資料，但由實驗工具載入直接SFT底座，批次八張；它不是上面CLI命令的實跑成績。500次更新後驗證9/30、測試2/30，測試CER為40/57，EOS卻是30/30。這份失敗結果在[11.13](chapters/11.md#11.13)有逐字解釋，提醒我們保存成功、會停止生成與能讀對數字都要分別核對。

錯配報告另外保存`donor_row`，是換入圖片來源的0起算行號。到validation第`donor_row+1`行查看它的圖片與答案，另記「生成文字是否符合換入圖的數字」。報告的`exact_match`仍對原答案計分，因此模型正確讀出換入圖時，這欄反而可能為false；若換入圖剛好數字相同，也不能用該題辨認影響。空白圖沒有可讀數字，本資料並未教特定空白理想回答，這裡只記它生成什麼，不預設它必須拒絕。

先預測正常圖應回答原數字，再觀察實際結果。只有正常圖能答對、且不同數字的換入圖使回答對應換入內容，才有理由進一步檢查模型是否使用圖片；三組都很差或只有任意文字變化，都不足以證明這一點。500步與保存檔不能替代逐題證據。最後用一小表保存原題行號、原數字、換入行號與數字、三份生成回答，說明哪一題支持或不支持你的判斷。

