# 14.2 fresh reader 的實際讀圖與閱讀紀錄

身份：/root/v4_review_coordinator/reader_stable_14_2。首次接觸專案，只以具基本數學背景的讀者閱讀所指派的14.2。沒有讀舊review內容、作者／協調者筆記或editorial guides。舊report只做bytes複製，保存在prior-report-unread.json。

真實順序見reading-order.json。先全文讀14.2，再讀它明確連結的4.3與T.8，接著render並view圖、寫預測、执行及讀輸出，最後全文重讀14.2。沒有估算閱讀分鐘。

## 我讀到的問題及推理

問題是控制同一位置三個特徵的尺度時，要不要先消去共同平均。對[2,2,2]，LN先減平均2而得到[0,0,0]；分母雖原本為0，小正數eps讓結果可計算。RMS把平方平均4開根而得到2，不減平均，所以約成[1,1,1]。共同偏正值這項資訊因而在兩種規則中有不同待遇。

對[1,2,3]，平均2；LN的偏差[-1,0,1]平方平均是2/3，分母sqrt(2/3+eps)，所以約[-1.2247,0,1.2247]。RMS的平方和是1+4+9=14，分母sqrt(14/3+eps)約2.1602，所以約[0.4629,0.9258,1.3887]。本節以近似數字忽略微小eps影響，程式則保留它。gamma每格乘、beta每格加，示範取gamma=1、beta=0，讓我可以只比較規則。

x有兩列位置與三欄特徵，形狀(2,3)。F是torch.nn.functional的別名；F.layer_norm的(3,)指明最後三格。x.square()逐格平方，mean(-1,keepdim=True)讓各列自己平均並留下單欄形狀(2,1)，sqrt開根，x除以它時同列三格用同一分母。round(decimals=4)只是印出時四捨五入到四位小數；print兩行的LN與RMS標籤讓我能核對正文。

## 真正的render與view evidence

輸入：course/figures/normalization.svg。
實際命令：inkscape course/figures/normalization.svg --export-type=png --export-width=1600 --export-background=white --export-filename=docs/reader-reviews/artifacts/natural-v4-stable-prefix/14.2/normalization-render.png。
命令成功exit 0；Inkscape有Pango/GTK與animation CSS警告，但輸出完整的靜態圖。
實際view：tools.view_image('/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-stable-prefix/14.2/normalization-render.png')。

我在白底圖上看到標題「每張卡片自己調尺度」，中段由左向右五張淡藍圓角卡片、深藍箭頭：一個token／[1,2,3] → 平均／尺度／只沿feature → LN或RMS／兩種不同規則 → 縮放／偏移／gamma；LN有beta → 同shape輸出／下一個零件。底部是「沿feature軸，不能跨時間偷看別的卡片」。所有字與箭頭均可辨，沒有截字、碰撞或把兩條規則畫成數值相同的輸出。正文緊接圖前的解釋把卡片對應到一個位置的三個特徵、gamma/beta對應乘與加，我可從圖理解處理的順序及保持形狀的意思。圖是共用流程示意，不是兩種公式的詳細演算；演算由旁邊正文與程式提供。

實際使用的前置4.3及T.8沒有Markdown嵌入SVG或其他圖，所以沒有額外必要SVG需要render。

## 詞彙與前置的實際使用

4.3提供標準差的數字例、eps在根號內、每格gamma/beta與不同位置各算自己的平均。已有高中數學的平均、平方、根號與標準差背景，所以沒有追读4.3的3.5；防偷看與BatchNorm只屬4.3的延伸，也沒有追讀3.6或其他章。
T.8讓我讀懂驗證代價是有效下一文字位置的平均猜錯代價、越低越好；訓練／驗證／最後檢查為分开的資料側。表中LN/RMS總參數141,568／141,248也相差320；本節的5處×每處64個beta恰能讓我理解這個參數差。L4是該次實測裝置名稱，本節只需要知道兩組在同一設備比較。T.8寫略過前三步後取中位數，補齊「暖機後」的操作意思。原論文連結是出處，本節已自足解釋RMS規則，因此沒有下載PDF。

圖內token没有另外作切詞定義，但正文已明確說卡片代表一個位置；feature可由特徵、shape可由輸出形狀互相對照。向前、反向、梯度核對、更新没有在14.2教其機制；在本節只是計時包含整個訓練步的範圍名單，我不需要先會計算梯度才能讀懂「13.952毫秒比13.062毫秒大，所以這輪沒有測到加速」。上述詞彙未造成核心推理或練習阻礙，已記錄而未列成必修缺口。

## 練習的預測與核對

執行前的原始預測保存在prediction-before-run.json。x+10變成[[12,12,12],[11,12,13]]。我預測LN兩列不變，因為平均同步加10而相對偏差與變異數不變；RMS第二列改變，因為分母變成sqrt((121+144+169)/3+eps)，约12.0277。全相同第一列RMS仍約全一，eps只造成很小的差。

我以現有.venv離線CPU執行原例與追加比較行，没有取資料、載模型或訓練。實際stdout是execution-output.txt：原始LN/RMS完全吻合正文的四位小數；輸入及兩種輸出均為(2,3)。移動後LN仍為[[0,0,0],[-1.2247,0,1.2247]]，RMS變為[[1,1,1],[0.9146,0.9977,1.0808]]。LN整體allclose=True，RMS整體=False；RMS第一列=True，第二列=False。LN最大絕對差為0，RMS整體為0.4516424834728241。這與預測以及正文的「通常改變」一致。

L4的短訓代價與每步時間是本節引用的既有實測结果。我只是讀懂它们及單種子限制，没有把這次CPU短例當成重跑那些結果。這次判斷只涵蓋14.2可讀性；不是事實核查或全書結論。
