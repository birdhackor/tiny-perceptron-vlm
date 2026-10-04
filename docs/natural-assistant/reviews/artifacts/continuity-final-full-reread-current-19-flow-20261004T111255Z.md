## 19.9 接上較快的做法，先檢查什麼？

已經能騎的腳踏車換一條輕量鏈條，第一件事是確認踩一下仍帶動相同方向，而不是馬上比較繞公園用了幾秒。模型換attention後端或加入快取，也要先確認沒有改掉可讀前文與輸出規則。更快但回答不同，需要先弄清楚差異來自數值尾數、快取錯誤，還是實際換了計算問題。

前置是[KV快取](16.md#16.3)、[GQA](16.md#16.4)、[SDPA](16.md#16.8)、[本成品的MoE與expert分派](19.md#19.2)與[推薦joint版本](19.md#19.4)。joint是文字、圖音共同訓練後的版本；它的MoE仍用可讀Python程式把每個位置交給選中的experts，沒有專用分派加速核心。KV cache儲存前文每層已算出的Key、Value，讓之後不必每次從頭計算；GQA讓多個Query頭共享較少組KV。本成品保留兩個Query頭與一組KV，快取較少，但這仍不代表整張GPU的所有記憶體跟著減半。

```python
import torch

from tiny_perceptron.capstone import CapstoneModel

torch.manual_seed(42)
core = CapstoneModel().language
core.eval()
ids = torch.tensor([[1, 21, 22, 23, 24]])
with torch.no_grad():
    full = core(ids)["logits"][:, -1]
    cache = core(ids[:, :3])["cache"]
    cached = core(ids[:, 3:], cache=cache)["logits"][:, -1]
print("最大分數差", (full - cached).abs().max().item(), "接近", torch.allclose(full, cached, atol=1e-4, rtol=1e-4))
```

五個ID是固定的純文字機制輸入，模型仍是隨機權重，沒有加入圖片或音訊。第一次整段計算；第二次先存前三個位置，再只送後二個位置，最後比較同一最後位置的264項候選分數。結果應接近True，最大差通常是浮點尾數；不能把這個單次例子當成多模態成品快取速度成績。

正式多模態生成還必須把素材編碼一次、與前文一起prefill，再送新token；prefill就是首次處理完整前文並建快取。若每步又處理圖片、重算全部前文，就沒有完成這條效率路線。測試需比較完整原始生成ID與停止原因，單看解碼文字相同不夠。

SDPA是attention運算介面，會依裝置、格式與形狀選擇後端；它不保證採用FlashAttention。本小模型可以用CPU FP32重跑正確性，GPU混合精度與Flash收益則需單獨核驗。本成品不會將不同experts的Python分派叫作已經有專用MoE加速核心。

正式部署另載入推薦joint，從固定驗證資料每項任務取第一題，共12題，選取時沒有先看生成結果。每題最多生成16個新編號，完整保留這次生成的ID序列；其中可能因上限截斷，所以這是機制對照，不是又一次完整能力評分。full與cache的原始ID全部一致，同一歷史上的每步logits也全部在`atol=1e-4, rtol=1e-4`範圍內接近。[完整快取對照](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/cache-consistency.json)保留兩條生成路徑與每步分數差，包含有圖片和聲音的任務。

確認一致後，才量一個事先選好的短問句：「照抄數字15，只要答案。」同一個joint、同一張NVIDIA L4、FP32，兩路各暖機3次，再量10次；兩路都生成`DIRECT:15`與EOS，共10個新編號，每次原始ID都相同。GPU具體型號記在[部署總實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_deployment.json)的`gpu`欄；下方逐次計時檔的`cuda:0`只表示裝置編號，不能單從它推斷型號。

| 同一句短生成 | 10次實測的中位時間 |
| --- | --- |
| full，每步重算前文 | 68.054毫秒 |
| cache，沿用前文KV | 59.978毫秒 |

這個cache在該短句較快，但不是任意長度、任意硬體的速度保證。計時包含前文準備、裝置傳輸、貪婪生成與解碼，GPU在每次呼叫前後同步；不含載入權重、啟動、上傳下載或檔案核驗。這次生成對照沒有量記憶體峰值，不能把19.2前向／反向的峰值貼過來稱成快取記憶體。[逐次生成計時實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/generation-benchmark.json)保留所有暖機與測量輸出、選題規則與範圍。

練習只把最後ID24改成25，保留前三個位置不變，再比較兩路；應仍接近。接著若修改第一位置，就必須重建cache，因為這份快取儲存的是那一段前文，不是一份通用記憶。

## 19.10 如何把成品帶到較小的電腦？

把一套工具運到小教室，可以換緊湊的包裝，也可以換較小的工具。量化比較像前者：用較少bits近似記錄每個數字。蒸餾比較像後者：先選一個較小學生模型，再讓它向教師學習。本成品的主MoE與便攜學生是不同交付選項，不能給同尺寸權重改名student就宣佈變小。

前置是[量化與反量化](17.md#17.2)、[真實4-bit打包](17.md#17.8)、[學生架構先變小](18.md#18.2)、[蒸餾後再量化](18.md#18.14)與[Dense／MoE結構差別](19.md#19.2)。PTQ是訓練完成後再量化；這一步會改變數字，應重新載入儲存後的版本做任務評估，而不是只量記憶體中的一個副本。

```python
import torch
from torch import nn

from tiny_perceptron.quantization import QuantizedLinear

torch.manual_seed(42)
layer = nn.Linear(8, 4)
compressed = QuantizedLinear(layer, bits=4)
x = torch.ones(1, 8)
with torch.no_grad():
    error = (layer(x) - compressed(x)).abs().max().item()
print("FP32權重加bias bytes", sum(p.numel() * p.element_size() for p in layer.parameters()))
print("打包權重、刻度與bias bytes", compressed.storage_bytes(), "最大輸出差", error)
```

`torch.manual_seed(42)`固定隨機起點，讓重跑時取得同一張權重表。`nn.Linear(8, 4)`有四套各讀八項特徵的配方，共32個權重與4個bias；量化版取自這一層，沒有另外造一份隨機權重。`torch.ones(1, 8)`是一筆八項都是1的輸入，原版與量化版都得到形狀`(1, 4)`的四項輸出。`torch.no_grad()`只算分數、不累積梯度，這裡沒有訓練。

兩版輸出逐格相減，再用`abs()`取絕對值、`max()`選四項裡最大的差、`item()`取出單一Python數字，所以`error`是本輸入最大的輸出差，不能當成整份模型的任務誤差。第一個bytes用各參數的元素數乘每元素大小，FP32權重和bias合計144。`storage_bytes()`只數物件裡儲存的張量：4-bit打包權重16 bytes、每行刻度16 bytes、bias 16 bytes，合計48；它沒有算檔案標頭、Python物件開銷或推論過程的全部記憶體。後面的完整成品表會另外列實際檔案bytes。

輸出通常有非零差，說明小包裝丟失了些細節。這個參考實現每次反量化成浮點計算，沒有低位元專用運算核心，因此不能從較少儲存量推出CPU變快。

完整成品的PTQ格式把非router的Linear權重量化，包含expert、注意力、素材接頭與輸出表；字嵌入、正規化、bias與router保留FP32。每一列輸出權重各存一個刻度，稱per-output-channel。router保留原精度有助於把壓縮影響分開檢查，但其他權重改變後，送進router的特徵也可能改變，所以仍可能選到不同expert。

部署時先讀取打包檔、還原成FP32權重，再以普通浮點運算推論。它減少下載與儲存量，沒有讓載入後整個模型只用4-bit記憶體。工具參數、拒絕、EOS與多模態成分都應重新檢查。若同問題的FP32版已經答錯，壓縮後仍錯不能寫成「量化沒有影響能力」。

便攜Dense學生每個位置都使用同一組前饋規則，不再由router選expert。它另有真正較小的結構，這次也已實跑：兩層、寬度48、79,920個參數，同一個264編號詞表與同一份資料家族切分。它從隨機權重開始，沒有把MoE直接刪掉幾位expert當成學生。教師事先選定為DPO比較分支，完整指紋以`b2428be8`開頭，不能稱為推薦joint的蒸餾版。

推薦joint的兩種PTQ已真的存檔、重新載入，再重跑同一份90題最後檢查。下面的檔案bytes取自部署測試當時的檔案，包含容器與metadata；後續公開版若移除內部路徑等資訊，檔案大小與指紋可能改變，公開下載要核對19.11的正式清單。

| 推薦joint版本 | 測試當時檔案bytes | 儲存張量bytes | 最後整題正確 |
| --- | --- | --- | --- |
| FP32來源檔 | 1,345,023 | 1,312,512 | 78／90 |
| PTQ 4-bit | 275,381 | 248,864 | 78／90 |
| PTQ 8-bit | 429,365 | 402,720 | 78／90 |

三列各任務的正確數也相同，但4-bit並沒有逐token相同：12個完整計算迴圈裡，原本「請算1加0」讀回後答`111`，4-bit改成`11`，兩個都錯，因此總分不變。整份90題共有1題的生成ID序列改變；8-bit在這90題的生成ID則沒有觀察到差異。這與19.9同一權重的快取一致對照是不同實驗，不能互相替代。

DPO比較分支的FP32／4-bit／8-bit同樣都是78／90；其4-bit有2題生成ID改變，8-bit沒有觀察到差異。因此「同分」只說明這份題庫的評分結果，不等於整個模型數字、所有回答或其他輸入都相同。本輪保留原本就錯的形狀與工具讀回案例，沒有把同分說成量化修好了它們。

儲存規格、來源指紋與量化欄位見[部署實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_deployment.json)；逐題可並排看[joint FP32](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint.json)、[joint 4-bit](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint-ptq4.json)與[joint 8-bit](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint-ptq8.json)。這些檔案載入後仍還原FP32，不代表執行記憶體也縮為八分之一。

學生比較分兩支：CE只學標準答案的交叉熵；KD則同時學標準答案與教師的候選分布，配方為`0.5 CE + 0.5 KL(teacher||student)`、溫度2並保留溫度平方縮放。KL量兩份分布的差，詳細直覺見[18.8](18.md#18.8)，混合示範與教師訊號見[18.9](18.md#18.9)。教師保持固定，只有學生更新。

兩支複製同一份隨機Dense起點，初始張量指紋以`3c1018fd`開頭；程式也固定相同的抽樣種子、批次規則與350次更新，各累計145,163個有效回答位置。這是配方控制條件：報告有初始指紋與總數，沒有逐批保存GPU取樣清單，不能稱做過逐批實錄核對。NVIDIA L4上，CE訓練迴圈為10.931秒，KD為12.063秒；KD時間包含額外的教師前向計算，不是同樣工作變慢。兩支訓練、評估與本地存檔合計32.018秒，排除啟動、映像建置與Hugging Face傳輸。

驗證各為59／84，最後整題正確卻是CE 62／90、KD 61／90。這次蒸餾沒有勝過同尺寸、同起點的CE學生。真正差一題的例子是「1+8等於多少？」：CE請求`TOOL:calculator:1+8`，工具回9，最後答9；KD卻請求`TOOL:calculator:1+9`，工具因而回10，模型最後又答11。請求參數和讀回兩層都錯了，不能只看最後總分來說「大致都保留下來」。兩支完整計算迴圈分別只有1／12與0／12，獨立回填都1／6、照抄都0／3、形狀都0／9。

| 小Dense交付版本 | 測試當時檔案bytes | 儲存張量bytes | 最後整題正確 |
| --- | --- | --- | --- |
| CE學生FP32 | 342,451 | 319,680 | 62／90 |
| KD學生FP32 | 342,451 | 319,680 | 61／90 |
| KD學生4-bit | 106,229 | 91,680 | 61／90 |

KD的4-bit也真的存檔、重載；它有6／90題的動作或最後生成ID改變，均發生在原本已錯的題目，總分才仍是61／90。小模型與打包確實減少儲存，這個固定350步支線卻沒有完整保留教師能力。本輪也沒有量學生的推論速度或生成記憶體峰值，所以不能從參數較少推出一個速度數字；這不是對所有Dense、所有蒸餾配方的結論。

所有條件與分數見[學生實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_student.json)，同題可並排看[CE逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/student/test-ce.json)、[KD逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/student/test-kd.json)與[KD 4-bit逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/student/test-kd-ptq4.json)。若你想重做壓縮研究，先保留這組比較與失敗，再為新配方另切驗證／最後考卷；不要看這次最後題目後挑比較好的一次重跑冒充原實驗。

練習只把bits改成8，先預測打包數值增加、刻度和bias仍佔空間，再執行。權重數字變細通常減少近似誤差，但這次單輸入偶然差更小，仍不構成所有任務的保證。

## 19.11 下載一份權重，就能從中斷處繼續訓練嗎？

把助理交給另一位學生時，有兩種需求：一位想直接試用，另一位想接著完成訓練。第一位需要模型權重與正確的結構、tokenizer；第二位還需要更新器狀態、隨機狀態、抽樣位置與訓練階段。只帶走期末作業，和帶走能繼續作業的完整工作桌，資料量與用途不同。

前置是[checkpoint與resume](05.md#5.7)、[tokenizer與模型的配套](06.md#6.5)、[本章父階段](19.md#19.4)與[量化、Dense學生兩條壓縮路線](19.md#19.10)。FP32成品使用`capstone-v1`格式，量化儲存版使用`capstone-ptq-v1`；檔案另外標明資料版本與階段。公開推論檔可以去掉更新器等狀態，方便下載；去掉之後就不能稱能完全恢復原訓練軌跡。

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_perceptron.capstone import CapstoneModel, load_capstone, save_capstone

model = CapstoneModel()
with TemporaryDirectory() as directory:
    path = Path(directory) / "example.pt"
    save_capstone(path, model, stage="sft", step=0, inference_only=True)
    loaded, metadata = load_capstone(path)
    print("格式", metadata["format_version"], "訓練步數", metadata["step"])
    print("只供推論", metadata["inference_only"], "參數相同", model.description() == loaded.description())
```

這段把隨機模型存到暫存資料夾，再真的讀回，結束後暫存夾自動清除。step=0明確表示沒有完成SFT；`stage="sft"`是格式標籤，不能拿來冒充已訓成果。輸出應為capstone-v1、0、True、True；最後一個True只確認架構與參數數量描述一致，完整數值與逐題輸出相等還要另檢。

正式交付把小型合成資料與來源規格保留在本專案，權重放[Hugging Face公開模型庫](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed/course/course-integration-v2)。本輪已逐檔匿名下載、核對指紋並讀回全部11份模型，正式[公開清單](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-public.json)固定在revision `33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed`。revision像這一版包裹的封條編號；SHA-256則是每個檔案的內容指紋，不只檢查檔名相同。下載不需要登入或提供token。

先照[19.1](19.md#19.1)準備Python環境，在專案資料夾執行：

```bash
python scripts/fetch_capstone.py --list
python scripts/fetch_capstone.py --stage joint
```

`--list`只印出11份清單，不下載權重，也不訓練。每一行的`stage`是下一行`--stage`可用的名字；`joint`是驗證後選出的推薦成品。其他名稱可這樣讀：

| 可以下載的名字 | 用途 |
| --- | --- |
| `pretrain`、`sft`、`joint`、`dpo` | 四站FP32推論檔，用來比較加入一項訓練後的變化 |
| `joint-int4`、`joint-int8` | 推薦joint的4／8-bit儲存版 |
| `dpo-int4`、`dpo-int8` | 偏好分支的4／8-bit儲存版 |
| `student-ce`、`student-kd`、`student-kd-int4` | [19.10](19.md#19.10)較小Dense學生的答案訓練、蒸餾與蒸餾後量化比較 |

FP32指權重以32-bit浮點數儲存；`int4`與`int8`指這裡的壓縮儲存格式，載入後仍還原成FP32計算，並不保證較少執行記憶體或更快。Dense學生的能力也不能直接當成joint能力。每個下載資料夾都包含`model.pt`、模型卡`README.md`、`LICENSE`、`THIRD_PARTY_NOTICES.md`與這次合成資料`data.json`；設定與tokenizer另存在模型檔內。各份用途、父檔來源與失敗案例要連同模型卡閱讀。

第二行會安裝到`checkpoints/capstone/joint/`。其中公開的`model.pt`是1,327,950 bytes，SHA-256為`d85cca83cdb4952f4653ef6b58d94562b41403db3a9db2ea31ff57e31246ca16`。這是移除訓練內部狀態與整理交付資訊後的正式檔案，不能直接拿[19.10](19.md#19.10)實驗當時的檔案bytes當它的大小；權重數值保留相同，封裝資訊有變。下載程式會先核對全部五個檔案的大小、SHA與模型身分，成功後才把完整資料夾放到目的地。若joint已在，就直接使用它；想另留一份，可以改用`python scripts/fetch_capstone.py --stage joint --output checkpoints/capstone-second`，後續模型路徑也改成`checkpoints/capstone-second/joint/model.pt`。

量化版可以另外下載，並用相同的CPU命令讀回：

```bash
python scripts/fetch_capstone.py --stage joint-int4
python scripts/capstone.py infer --checkpoint checkpoints/capstone/joint-int4/model.pt --prompt "1+2等於多少？" --device cpu
```

這份公開joint-int4檔是270,855 bytes；下載清單保留它自己的SHA。命令會選正確載入方式，再跑同一工具迴圈。若換成`student-kd-int4`，要把下載名稱與路徑一起換；成功下載、成功載入，仍不代表答案正確。例如本輪這位學生把1+2請求錯寫成2+9，工具回11後自己答12。我們保留這類失敗，不把「程式能跑」算成「模型會答」。本機網頁的`serve`目前只接受FP32格式，所以仍使用19.1的`joint/model.pt`；不要把`joint-int4/model.pt`交給它。

中斷續訓使用另外的`model-training.pt`與原有資料版本。DPO還需儲存reference狀態；只重新建立一份會隨當前policy改變的reference，並不等於接著原實驗跑。是否能逐步重現，也受裝置與運算實現影響，因此報告應區分同環境恢復測試與跨環境重新訓練。

如果要自己重做四站，下面才是會更新權重的命令；它們不是上方試用的必要步驟，也不會在本節短程式中自動執行。`--batch-size 24`表示每次更新取24筆訓練資料一起計算誤差；`--seed 42`用同一個種子數字設定新實驗的隨機初值與取樣起點，方便在相同條件下比較實驗。先用自己的資料版本與程式產生pretrain，再把每站真正完成的原始`model.pt`交給下一站。請一行一行執行，先讀該站的`train-report.json`，確認`schedule_completed`為`true`，才往下走：

```bash
python scripts/capstone.py train --stage pretrain --output checkpoints/my-capstone/pretrain --steps 300 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage sft --input-checkpoint checkpoints/my-capstone/pretrain/model.pt --output checkpoints/my-capstone/sft --steps 1400 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage joint --input-checkpoint checkpoints/my-capstone/sft/model.pt --output checkpoints/my-capstone/joint --steps 600 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage dpo --input-checkpoint checkpoints/my-capstone/joint/model.pt --output checkpoints/my-capstone/dpo --steps 100 --seed 42 --batch-size 24 --device cpu
```

這四行示範本課原配方的階段關係，並沒有另跑一次CPU訓練來宣稱重現正式GPU分數；各站實報在[19.4](19.md#19.4)。DPO是要觀察的比較分支，不是成品必須接受的最後更新：本輪推薦的是joint。命令每次預設以540秒作為訓練循環的時間預算，程式在每批更新前檢查時間；最後一批更新與存檔可能超過預算，之後的匯出、驗證與整理報告也另外花時間。電腦較慢時可能只完成部分步數，訓練循環停止時仍會留下工作檔；例如自己那站joint原本排定600步、中途停下，可以用：

```bash
python scripts/capstone.py train --stage joint --resume checkpoints/my-capstone/joint/model-training.pt --output checkpoints/my-capstone/joint --steps 600 --seed 42 --batch-size 24 --device cpu
```

這裡的600是原本總步數，不是追加600步。`--resume`要求同一站、原定步數、batch大小、資料指紋與程式指紋一致，並讀回更新器、隨機狀態和進度；取樣接著工作檔的位置走，不會因為再次寫seed 42就從第一筆重來。DPO還要原reference。若尚未完成，先繼續同一站，不要以不完整父檔開始下一站。跨站的`--input-checkpoint`則以已完成父模型開始一份新的更新器與抽樣流程，兩者用途不同。實際檢查條件可見[訓練器原始碼](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/scripts/course_experiments/capstone.py)。

公開推論檔沒有上述完整工作狀態，也沒有本課串階段命令所需的原始`data_manifest`與完成排程證明，因此不能把公開`checkpoints/capstone/joint/model.pt`代入這裡的`--resume`或正式`--input-checkpoint`。推論權重本身仍可作為另寫訓練程式的初始數值；那是建立一個新實驗，並不是恢復本課已驗證的訓練。公開交付的是可試用、可比較的11份推論版本，不包含作者的完整續訓工作檔。

練習把step改成1，觀察metadata確實顯示1。但模型仍沒有更新過，這說明格式欄位可以寫入，卻不能自行證明訓練發生；正式報告還需有效目標、權重變更與執行記錄。

## 19.12 完成整合後，怎樣回答「這個模型會什麼」？

一張總分單可能寫著80分，卻沒告訴你計算、聽音、看圖各會多少。多能力模型也是如此：如果照抄題很多，照抄表現很好，就可能蓋住工具流程和圖片的失敗。最終成品要交一張每項能力都有固定分母的矩陣，讓你知道成功在哪裡、限制在哪裡。

前置是[獨立評估](05.md#5.10)、[工具選擇不同錯誤](0B.md#B.7)與[本章成品試用](19.md#19.1)。矩陣是一張同考卷、不同階段並排的表。pretrain是先練習接下一個文字片段；SFT是照示範回答練習；joint在本章指文字、圖片、聲音與工具題的聯合訓練，階段接續見[19.4](19.md#19.4)。DPO則用同題較偏好的回答作比較訓練，見[19.8](19.md#19.8)。這些版本和部署版本都做同一檢查。加入新能力後舊能力下降，就留在表裡；最後選擇哪份權重，也要基於預先說明的validation規則，而不是看test挑最漂亮的一欄。

先確認每項考題真的存在：

```python
from collections import Counter

from tiny_perceptron.capstone import build_dataset

splits, _ = build_dataset()
counts = Counter(row["task"] for row in splits["test"])
for task, count in sorted(counts.items()):
    print("最後檢查", task, "分母", count)
print("合計", sum(counts.values()), "這只是考題數量，不是答對數量")
```

程式只數固定test資料，完全沒有生成回答。每個任務的count是那項分母；沒有對應任務就不能報它的成功率。正式評分另外讀取每題原始生成ID、EOS與解析結果，再按該任務規則數正確答案。平均方式也要明示：所有題直接平均會讓題多的能力佔較大比重，各任務等權平均則回答另一個問題。

最後90題已依凍結配方跑完，所有原始回答都保留。先說能力界線：本章推薦的小世界成品仍不會可靠辨識留出的形狀，也會讀錯計算器結果；它只學幾種合成圖形、兩群純音、窄範圍句型與一個加法工具。本章沒有訓練或驗收一般照片理解、中文讀字與真人語音辨識，也沒有網路搜尋或通用安全與信心校準保證。下面每個分母都對應實際題目，而不是把教過的技術名字當成能力。

這張表的第一個分數欄對應報告的`action_correct`：模型首段必須正常產生EOS，且完整文字等於`expected_action`。TOOL包含工具名與有順序的參數，DIRECT／ASK也包含回答或求助內容，不是只選對標記。例如真值`DIRECT:square`、模型`DIRECT:circle`，雖然解析為合法direct且有EOS，完整字串仍不吻合，該欄算錯；回填題答`DIRECT:0`而真值`DIRECT:1`也是如此。

第二個分數欄`end_to_end_correct`再要求最後答案符合真值。直接回答沒有額外一次生成；工具題則要先執行，再讀回並由模型回答，所以首段請求全對仍可能整題錯。這解釋了下面80／90個首段吻合，但只78／90題完成：差兩題就是計算器得到1、模型讀回後答錯的例子。這個精確字串指標與B.7的選卡分類不是同一個分數。

| 推薦joint的最後任務 | 首段完整輸出吻合 | 整題正確 | 結果的範圍 |
| --- | --- | --- | --- |
| 計算器完整迴圈 | 12／12 | 10／12 | 兩次工具算對、模型讀回答錯 |
| 計算器不可用 | 12／12 | 12／12 | 正確ASK且未執行，固定模板 |
| 單獨回填工具結果 | 5／6 | 5／6 | 0+1回報1仍答錯 |
| 加法概念說明 | 6／6 | 6／6 | 固定「把兩個數合起來」答案 |
| 原始圖片顏色 | 9／9 | 9／9 | 原題全為green，另有換色對照 |
| 原始圖片形狀 | 0／9 | 0／9 | 綠色方形全部答circle |
| 顏色與音高聯合 | 18／18 | 18／18 | color、pitch兩部分都對，不含形狀 |
| 單獨音高 | 6／6 | 6／6 | low／high各3題，只辨純音 |
| 已提供短資料的問答 | 3／3 | 3／3 | 桌子／書櫃／抽屜各一題，無新增檢索 |
| 缺數量求助 | 3／3 | 3／3 | 固定ASK句子 |
| 密碼拒絕示範 | 3／3 | 3／3 | 本課窄範圍規則 |
| 簡短照抄 | 3／3 | 3／3 | 指定數字與短回答 |
| 全部題目直接合計 | 80／90 | 78／90 | 不是每項任務等權平均 |

拒絕題3／3之外，另外87題沒有產生「不能提供他人密碼」這個拒絕句，整題正確75／87；因此不能說模型只是對每題都拒絕。但這仍只排除本題庫的那種誤拒，沒有測完所有正常問法。圖片成對對照是27／36對都正確，其中顏色9／9、形狀0／9、聯合18／18；換聲18對都正確，詳見[19.6](19.md#19.6)。只要原圖有錯，換後答對也不能把整對列為成功。

同一份最後考卷，也評了之前各站與壓縮版本。這裡的「整題」都要求正常結束、動作規格與最後內容正確，沒有把工具執行成功當成助理答對。表中的CE學生只照標準答案練習；KD學生還會模仿教師對下個文字片段分配的機率，這兩條支線的教法見[19.10](19.md#19.10)。

| 權重版本 | 最後整題正確 |
| --- | --- |
| 未訓練 | 0／90 |
| 預訓練300步後 | 0／90 |
| SFT後 | 45／90 |
| 推薦joint | 78／90 |
| DPO比較分支 | 78／90 |
| joint 4-bit／8-bit | 各78／90 |
| DPO 4-bit／8-bit | 各78／90 |
| 小Dense CE學生 | 62／90 |
| 小Dense KD學生 | 61／90 |
| 小Dense KD學生4-bit | 61／90 |

DPO在最後考卷與joint同分，不會把先前驗證退步的事抹掉，也不會改成用最後考卷重新選模型。推薦joint的決定與依據在[測試前的選擇紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-selection.json)；全版本分數在[正式部署實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_deployment.json)。推薦成品的[完整90題紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint.json)與[原始資料](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/data.json)用相同`id`連起來；其餘版本也保留在同一證據目錄。量化同分卻有少數生成改變，不能用這張總表宣稱逐token相同，詳見[19.10](19.md#19.10)。

效率則是另一種證據：快取的12種預選驗證輸入原始ID一致；一個短照抄題在L4的full／cache中位時間是68.054／59.978毫秒。它沒有提高上表能力分數，也沒有量生成記憶體峰值，完整範圍見[19.9](19.md#19.9)。把能力、速度與檔案大小分開說，學生才能知道下一步該增加哪種資料與檢查。

小Dense支線也已完成，但保留能力需要逐項看。它有79,920個參數，教師是事先選定的DPO分支；不是推薦joint的蒸餾版。下面每欄都用同一份最後題目，4-bit學生是儲存後重新載入的版本。

| 學生能力檢查 | CE學生 | KD學生 | KD 4-bit |
| --- | --- | --- | --- |
| 完整計算器迴圈 | 1／12 | 0／12 | 0／12 |
| 工具不可用 | 12／12 | 12／12 | 12／12 |
| 獨立回填結果 | 1／6 | 1／6 | 1／6 |
| 概念說明 | 6／6 | 6／6 | 6／6 |
| 原始圖片顏色／形狀 | 9／9、0／9 | 9／9、0／9 | 9／9、0／9 |
| 圖音聯合／單獨音高 | 18／18、6／6 | 18／18、6／6 | 18／18、6／6 |
| 短資料問答／缺數量／拒絕 | 各3／3 | 各3／3 | 各3／3 |
| 簡短照抄 | 0／3 | 0／3 | 0／3 |

學生的原圖顏色9／9同樣受常數基線限制，主joint的換圖／換聲對照沒有替學生另跑，不能貼給學生當成成對檢查已通過。這次KD比CE少對的一題涉及錯工具參數及錯讀回，4-bit同分也有6題生成改變，具體例子與檔案大小見[19.10](19.md#19.10)，全部逐題檔見[學生實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_student.json)與其證據目錄。縮小確實完成，原能力卻沒有全部保留，這也是應該交給學生看的結果。

如果你想繼續處理自然照片，先回到[11.14](11.md#11.14)看物件與整張圖的關係，再看[11.15](11.md#11.15)的小筆畫和[11.16](11.md#11.16)的閱讀順序。這些例子回答同一個問題：入口丟掉了哪些資訊？想讓一般人的語音也能提出聊天問題，則回到[12.13](12.md#12.13)分清聽寫和回答，再看[12.14](12.md#12.14)讓逐字稿與打字訊息交給同一聊天核心。想讓工具處理陌生問法更穩，仍回到[B.7](0B.md#B.7)增加獨立問法評估。

[第20章](20.md#20.1)會沿著照片、中文讀字與語音這條路繼續，但換一種起點：接手上游已訓練的圖文底座，再用本課資料微調，語音先經另一個模型轉成文字。它不是把本章的328,128參數MoE加大後接續訓練；原有的圖片與聲音摘要也不會因參數增加就自動找回丟掉的細節。新起點已有多少能力、本課改了哪些數字、在新素材上還錯什麼，都要另外交代和檢查。

兩個專題因此各回答一個問題：本章讓你追蹤零件從零學習、接續訓練與組合的過程；下一章讓你研究如何使用一份成熟底座，練習較貼近日常輸入的任務。下一章採Dense而保留本章MoE，選擇原因放在[20.2](20.md#20.2)。先記住一個資源限制就好：MoE可以少算部分專家，但全部專家的權重通常仍要存放，不能只看啟用參數就判斷弱硬體能否承受。

練習從表裡選一項，寫一個當前小世界以外的輸入與它需要新增的檢查。例如「真實街景裡紅色車輛」不能直接沿用合成色塊準確率；必須先準備有來源和標註的照片、素材家族切分與新評估。接著選擇想走哪條路：在本章的小模型上研究一個新零件，或到第20章接手已訓練的底座。先說清楚要證明什麼，再決定模型與訓練方法。
