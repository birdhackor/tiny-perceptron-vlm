## 19.10 如何把成品帶到較小的電腦？

把一套工具運到小教室，可以換緊湊的包裝，也可以換較小的工具。量化比較像前者：用較少bits近似記錄每個數字。蒸餾比較像後者：先選一個較小學生模型，再讓它向教師學習。本成品的主MoE與便攜學生是不同交付選項，不能給同尺寸權重改名student就宣佈變小。

前置是[量化與反量化](17.md#17.2)、[真實4-bit打包](17.md#17.8)、[學生架構先變小](18.md#18.2)、[蒸餾後再量化](18.md#18.14)與[Dense／MoE結構差別](19.md#19.2)。PTQ是訓練完成後再量化；這一步會改變數字，應重新載入儲存後的版本做任務評估，而不是只量記憶體中的一個副本。

本節的joint是文字、圖片與聲音共同訓練後的版本；DPO則從joint接續，用較好與較差的回答配對學偏好。兩條支線的關係見[19.4的訓練階段](19.md#19.4)。先辨認來源，才能分清我們壓縮或蒸餾的是哪一位教師。

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

素材接頭把圖音特徵轉成語言核心能收的向量；字嵌入把輸入編號查成向量。本成品使用[14.2的RMSNorm](14.md#14.2)：用均方根調整一組特徵的數值大小，不先減掉平均值。完整成品的PTQ格式把非router的Linear權重量化，包含expert、注意力、素材接頭與輸出表；字嵌入、正規化、bias與router保留FP32。每一列輸出權重各存一個刻度，稱per-output-channel。router保留原精度有助於把壓縮影響分開檢查，但其他權重改變後，送進router的特徵也可能改變，所以仍可能選到不同expert。

部署時先讀取打包檔、還原成FP32權重，再以普通浮點運算推論。它減少下載與儲存量，沒有讓載入後整個模型只用4-bit記憶體。工具參數、拒絕、EOS（表示回答結束的專用編號）與多模態成分都應重新檢查。若同問題的FP32版已經答錯，壓縮後仍錯不能寫成「量化沒有影響能力」。

便攜Dense學生每個位置都使用同一組前饋規則，不再由router選expert。它另有真正較小的結構，這次也已實跑：兩層、寬度48、79,920個參數，同一個264編號詞表與同一份資料家族切分。它從隨機權重開始，沒有把MoE直接刪掉幾位expert當成學生。教師事先選定為DPO比較分支，因此這份學生不能稱為推薦joint的蒸餾版；來源核對記在下方學生實報。

推薦joint的兩種PTQ已真的存檔、重新載入，再重跑同一份90題最後檢查。下面的檔案bytes取自部署測試當時的檔案，包含容器與metadata，也就是描述模型結構、來源等附加資訊；後續公開版若移除內部路徑等資訊，檔案大小與指紋可能改變，公開下載要核對19.11的正式清單。

| 推薦joint版本 | 測試當時檔案bytes | 儲存張量bytes | 最後整題正確 |
| --- | --- | --- | --- |
| FP32來源檔 | 1,345,023 | 1,312,512 | 78／90 |
| PTQ 4-bit | 275,381 | 248,864 | 78／90 |
| PTQ 8-bit | 429,365 | 402,720 | 78／90 |

三列各任務的正確數也相同，但4-bit並沒有逐token相同：12個完整計算迴圈裡，原本「請算1加0」讀回後答`111`，4-bit改成`11`，兩個都錯，因此總分不變。整份90題共有1題的生成ID序列改變；8-bit在這90題的生成ID則沒有觀察到差異。這與19.9同一權重的快取一致對照是不同實驗，不能互相替代。

DPO比較分支的FP32／4-bit／8-bit同樣都是78／90；其4-bit有2題生成ID改變，8-bit沒有觀察到差異。因此「同分」只說明這份題庫的評分結果，不等於整個模型數字、所有回答或其他輸入都相同。本輪保留原本就錯的形狀與工具讀回案例，沒有把同分說成量化修好了它們。

儲存規格、來源指紋與量化欄位見[部署實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_deployment.json)；逐題可並排看[joint FP32](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint.json)、[joint 4-bit](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint-ptq4.json)與[joint 8-bit](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint-ptq8.json)。這些檔案載入後仍還原FP32，不代表執行記憶體也縮為八分之一。

學生比較分兩支：CE只學標準答案的交叉熵；KD則同時學標準答案與教師的候選分布，配方為`0.5 CE + 0.5 T² KL(p_T||q_T)`，溫度T=2；p_T與q_T分別是教師與學生在此溫度下的候選分布。`distillation_loss`內部已乘T²，呼叫處不再乘一次。KL量兩份分布的差，詳細直覺見[18.8](18.md#18.8)，混合示範與教師訊號見[18.9](18.md#18.9)。教師保持固定，只有學生更新。

兩支複製同一份隨機Dense起點，固定相同的抽樣種子、批次規則與350次更新，各累計145,163個有效回答位置。這控制了配方；報告沒有逐批保存GPU取樣清單，不能稱做過逐批實錄核對。KD還需要固定教師的前向計算，訓練成本因此不能只用學生參數數量估計；完整計時與來源核對保留在下方實報。

驗證各為59／84，最後整題正確卻是CE 62／90、KD 61／90。這次蒸餾沒有勝過同尺寸、同起點的CE學生。真正差一題的例子是「1+8等於多少？」：CE請求`TOOL:calculator:1+8`，工具回9，最後答9；KD卻請求`TOOL:calculator:1+9`，工具因而回10，模型最後又答11。請求參數和讀回兩層都錯了，不能只看最後總分來說「大致都保留下來」。

下面各分項都是同一份90題考卷的子集；成對數字先列CE、再列KD。[完整計算迴圈](19.md#19.7)包含模型請求、工具執行與模型讀回答案，兩支只有1／12與0／12。獨立回填是直接提供原題與工具結果，省掉先前的請求步驟，兩支都只有1／6。照抄要求原樣重現指定數字，兩支都是0／3；形狀要求辨識圖片是方形或圓形，兩支都是0／9。這些[成品題型](19.md#19.1)各自考不同能力，不能把分項再加進90題總數。

| 小Dense交付版本 | 測試當時檔案bytes | 儲存張量bytes | 最後整題正確 |
| --- | --- | --- | --- |
| CE學生FP32 | 342,451 | 319,680 | 62／90 |
| KD學生FP32 | 342,451 | 319,680 | 61／90 |
| KD學生4-bit | 106,229 | 91,680 | 61／90 |

KD的4-bit也真的存檔、重載；它有6／90題的動作或最後生成ID改變，均發生在原本已錯的題目，總分才仍是61／90。小模型與打包確實減少儲存，這個固定350步支線卻沒有完整保留教師能力。本輪也沒有量學生的推論速度或生成記憶體峰值，所以不能從參數較少推出一個速度數字；這不是對所有Dense、所有蒸餾配方的結論。

所有條件與分數見[學生實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_student.json)，同題可並排看[CE逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/student/test-ce.json)、[KD逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/student/test-kd.json)與[KD 4-bit逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/student/test-kd-ptq4.json)。若你想重做壓縮研究，先保留這組比較與失敗，再為新配方另切驗證／最後考卷；不要看這次最後題目後挑比較好的一次重跑冒充原實驗。

練習只把bits改成8，先預測打包數值增加、刻度和bias仍佔空間，再執行。權重數字變細通常減少近似誤差，但這次單輸入偶然差更小，仍不構成所有任務的保證。

