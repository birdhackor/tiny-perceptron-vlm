## 18.13 圖片／音訊能力能如何蒸餾？

教師看圖時用16個視覺token，學生只用4個，回答三個文字token。若把兩條整段logits直接對上，同一列可能一邊仍在處理圖片、另一邊已經在回答，機率就不具相同意義。前置是[模態展開後的位置](10.md#10.7)、[回答第一token的預測位置](07.md#7.4)與[分布對齊](18.md#18.6)。蒸餾可以比較相同答案，但需要先找到各自真正負責預測答案的位置。

用沒有角色或問題額外位置的玩具序列，教師前文16格、學生4格，答案為A0、A1、A2。A0輸入位置分別16與4，但預測A0的logits來自它前一格，也就是15與3；A1由16與4預測，A2由17與5預測。這個向左一格的對齊非常重要，不能因只在說圖片就忘記下一字預測的規則。

圖中每條橫線連接相同「待預測答案」，不是相同絕對索引。第一列教師query15與學生query3都預測A0，即使前文長度不同，也能比較同一詞表分布。下列兩列同理；答案輸入本身則從16/4開始，和圖裡的Query位置分開。

![前文長度不同時，按同一答案token的預測位置配對](../figures/architecture_modal_answer_alignment.svg)

```python
import torch
from tiny_perceptron.alignment import distillation_kl

teacher_prefix, student_prefix, answer_tokens = 16, 4, 3
t_rows = list(range(teacher_prefix - 1, teacher_prefix + answer_tokens - 1))
s_rows = list(range(student_prefix - 1, student_prefix + answer_tokens - 1))
t_logits = torch.zeros(1, teacher_prefix + answer_tokens, 2)
s_logits = torch.zeros(1, student_prefix + answer_tokens, 2)
t_logits[:, t_rows] = torch.tensor([0.8, 0.2]).log()
t_answer, s_answer = t_logits[:, t_rows], s_logits[:, s_rows]
labels = torch.zeros(1, answer_tokens, dtype=torch.long)
print("預測位置", t_rows, s_rows)
print("對齊形狀", tuple(t_answer.shape), tuple(s_answer.shape))
print("答案KL", round(distillation_kl(s_answer, t_answer, labels, temperature=1).item(), 4))
```

輸入只是人工分數表，沒有載入圖片或聲音。教師選中三格都設比例 `[0.8,0.2]`，學生零分數則均勻；`t_rows`和`s_rows`各自挑出預測相同答案的logits，labels將三格都標成有效。應印出 `[15,16,17]`與 `[3,4,5]`，兩邊形狀同為 `(1,3,2)`，KL0.1927。未對齊時整段長度19與7不同，不能直接按列做分布誤差。

真實模型還含角色、問題、圖像標記或音訊框，答案位置應從展開後的有效labels與預測位移推導，不能固定套16和4。兩邊也需看同一份圖片/聲音與問題，否則學生是在學另一個條件下的教師回答。若比較內部視覺特徵，因數量與維度不同，還需另外設計映射與目標，不是直接複用這個文字KL。

正式實驗接上[11.5的圖片問答](11.md#11.5)與[12.12的圖片加音訊教師](12.md#12.12)。圖片教師用已訓練160步的`all`支線，聯合教師用400步的模型，來源檔案SHA保存在[多模態蒸餾實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/multimodal_distillation.json)的`teacher_provenance`。兩者的語言部分寬64、兩層，整個多模態模型各145,664參數、582,656 bytes浮點數值。學生語言部分寬32、一層，整個模型36,096參數、144,384 bytes，仍是可直接載入的`multimodal-v1`模型。

教師把16×16圖片切成4×4的patch，得到16個視覺token；學生切成8×8的patch，只得到4個，視覺特徵寬度也由16改8。較大的patch每次要讀更多像素，這一組視覺編碼器反由教師1,344個參數增為學生1,664個。整個學生較小主要還包括語言模型與接頭的縮小；圖片前文少12格，也不表示包含角色、問題、音訊與答案的整段序列只剩四分之一。本輪沒有量推論kernel速度或隔離記憶體，不能由token數直接填出加速倍數。

實際對齊可核對兩筆訓練資料。紅方形的`color?`，標準回答是`red`與EOS，四個ID為`[122,109,108,2]`；教師預測列25–28，學生13–16。藍方形加180 Hz音調的`joint?`，答案`square,low`與EOS共11個ID；教師列36–46，學生24–34。這些位置包含真實聊天前文與模態展開，不是前面玩具例的15與3。工具分別從兩邊位移後的有效labels選列，再驗證答案ID完全一致；教師讀同一份圖、聲音、問題和標準回答前文，快取264-ID詞表的答案分數，教師權重保持不動。

圖片資料按同一場景的`shape?`和`color?`家族一起切分，訓練／驗證／測試36／12／12筆，家族18／6／6組，沒有交集。聯合資料24／12／12筆，訓練使用偏移0和180、220、380、420 Hz，驗證偏移1及200、400 Hz，測試偏移2及260、340 Hz；音調高低門檻仍是300 Hz。這兩個測試頻率沒參與聯合階段更新，但先前音訊編碼器已見過，不能說整條路徑從未見過。每個任務把同初始化學生分成普通CE與CE+KL兩支，batch為4、lr=0.003，各更新350次。兩支讀相同批次，圖片任務各8,391個有效回答與EOS目標，聯合任務各16,104個；KL沿用預設T=2、`0.5 CE + 0.5 T² KL`，沒有在這12題上挑設定。

| 同一留出任務 | 已訓練教師答對數／NLL | CE學生答對數／NLL | CE+KL學生答對數／NLL |
| --- | --- | --- | --- |
| 圖片問答，12題、72個真值目標 | 9/12／0.1643 | 9/12／0.0651 | 8/12／0.1216 |
| 圖片加音訊，12題、138個真值目標 | 12/12／0.0008 | 8/12／0.0936 | 6/12／0.3396 |

表中NLL是給共同標準回答前文時的平均代價，答對數則由自由生成的原始ID核對。兩個教師與四個學生在各自12題都EOS12/12，沒有非法控制ID、生成錯誤或跳過題，但KL學生仍各少答對一題與兩題。圖片KL學生對藍方形的`color?`答`red`；聯合兩個學生對紅方形、260 Hz的`square,low`都答`circle,low`。停止正常不能代替顏色與形狀判斷。

再分別遮住輸入，才能追蹤學生用了什麼條件。聯合CE學生從8/12降為空白圖片6/12、空白音訊4/12；KL學生從6/12變為6/12、3/12。只看相同6/12還不夠：逐題比對顯示，KL學生在空白圖片前後的12串原始生成ID完全相同，正常輸入時又對所有形狀都答`circle`。這12題沒有觀察到圖片改變其生成；音調部分卻都12/12，遮聲音後才降為6/12。這是本輪生成行為的具體限制，不能推論所有輸入下內部視覺表示都沒有作用。圖片問答兩支遮圖後都5/12，也需與完整逐題回答一起讀，不能只把遮圖掉分當成已保住全部視覺能力。

圖片教師快取花約0.153秒、檔案262,251 bytes；聯合約0.102秒、315,875 bytes。圖片CE／KL學生更新分別約8.02／9.68秒，聯合10.38／11.96秒，快取成本另算。四個學生浮點數值儲存相同，原始推論檔則分別169,739／169,929／169,887／170,013 bytes；這些容器另有metadata與隨機狀態，公開移除狀態後大小需重新量，不能拿檔案差當權重差。這次只蒸餾答案分布，沒有蒸餾內部圖像特徵，也只檢查合成形狀與純音調，未測自然圖片問答或語音辨識。

在[T.10的多模態準備流程](../training.md#T.10)取得`vqa/`、`joint/`教師目錄，保留`model.pt`和原始`dataset.json`，再跑完整CPU蒸餾：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment multimodal_distillation --device cpu
```

這次GPU正式執行四支合計1,400次更新，連評估與保存約53.01秒，CPU耗時另量。輸出有`vqa-ce.pt`、`vqa-ce_kl.pt`、`joint-ce.pt`、`joint-ce_kl.pt`和教師副本；`model.pt`是聯合CE+KL學生。它保留自己的patch大小與接頭設定，可用原生多模態入口讀取：

```bash
.venv/bin/python -m scripts.infer_modal outputs/course-experiments/course-v1/multimodal_distillation/model.pt --color red --shape square --frequency 260 --prompt 'joint?' --tokens 16 --device cpu
```

這個入口會產生居中的紅方形和260 Hz聲音，物理真值是`square,low`，並輸出模型實際回答、原始ID與EOS。它的圖片偏移預設0，與正式留出題的偏移2不同，因此這條命令是另一次課堂探針，不能代替上表12題分數。練習先並排聯合KL學生正常與遮圖的全部12串ID，再指出六道方形題為何即使音調正確仍不算整題答對。

練習只把student_prefix改成6，保持同樣三個答案。先預測s_rows變 `[5,6,7]`，教師不變，對齊後形狀與KL仍相同，再執行核對。音訊前文框數變化也遵循同樣道理：找到語意相同的答案位置，再比較分布。

