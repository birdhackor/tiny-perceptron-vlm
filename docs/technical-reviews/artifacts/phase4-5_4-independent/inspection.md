# 5.4 本輪獨立來源與執行核對

審閱者：`/root/phase4_factual_coordinator/factual_5_4`，fresh；2026-10-05。未讀任何舊技術／讀者報告的正文或結論，未複製他人判斷。此筆記由本輪親讀與實際 CPU 結果寫成。本文沒有圖，也没有模型成績、樣本／token／訓練步數的實測 claim；不需要讀其他實驗 JSON 來猜來源。`bounded-results.json` 是本輪短檢查結果，已親讀並核對各欄。

## 原始內容與方法

親讀完整 `course/chapters/05.md#5.4`（原始第114–149行），以及本節直接引用的 `course/chapters/01.md#1.12` 的減去負梯度／實際更新說明。本節不是章首，未把章導言當成額外審閱範圍。原稿節位元组與 fence 存在 `original/`；來源 SHA-256 是 `4c530d55ba25de5415f3342a3c7d48827f567655b008ed6b86a0c122b56f82fa`，沒有重排換行。親讀 review method、reader protocol、checker 完整 schema 與 section_facts 完整來源。section_facts 先原樣擷取 fence，注入 build_course 的 bootstrap，在新 CPU process 依序執行；限制 socket、子行程、artifact 外寫檔，這是 Python audit guard 而非 OS sandbox。原程式只有一個 fence，成功執行，沒有 guard event。沒有執行訓練配方。

## 原論文

實際從 `https://arxiv.org/pdf/1412.6980v9` 取得原 PDF。親讀首頁作者／ICLR2015／頁緣 `arXiv:1412.6980v9 [cs.LG] 30 Jan 2017`，第2頁 Algorithm 1 和 §2、第3頁 §2.1 相關更新式與 §3 eq.(1)–(4)。PDF 是作者在 arXiv 的原方法論文，並非來源庫摘要。Algorithm 1 明列 m0=v0=0、t0=0、各向量逐格計算、β1=.9、β2=.999、α=.001、ε=10^-8；更新式是 θ_t=θ_(t−1)−α*m_hat/(sqrt(v_hat)+ε)，ε 在平方根外。m、v 分別估計一階矩與平方梯度二階原始矩（不是中心化 variance），正文「加權平均」正確。§3 eq.(4) 說零初始化帶來 1−β2^t 因子，除去此因子修正初始化偏差；對第一矩同理。本報告不引用論文的收斂定理或效能優勢，也不把本輪玩具測試當作這些定理的證明。

`pdftotext` exit=0，但 stderr 提示 xref num 342 要 reconstruct；原 PDF 第2頁另用 `pdftoppm` 實際渲染並由 view_image 親看 `sources/adam-page2.png`。頁面清楚可讀，已目視確認 Algorithm 1 根號只包含 v_hat，沒有把 ε 算進根號。此 PNG 是原論文核對素材，不是教材不存在的圖。

## 官方原碼/API

下載 PyTorch 官方 `pytorch/pytorch` 的 commit `5c4886908584029761b579af026dcfb627c84070`（與安裝 torch.version.git_version 相同），不是官方最新版猜測。CPU 安裝版自報 `2.14.1+cpu`。`bounded-environment.json` 顯示下載 Adam 與 Optimizer 的 SHA-256 和 installed 本地檔逐 byte 相同。

- `torch-adam-installed-commit.py`：親讀 `Adam.__init__` 第34–40行參數；`_init_group` 第151–186行每 parameter state 的 step、exp_avg、exp_avg_sq 零初始化；第278–325行公式及 args；`_single_tensor_adam` 第413–475行 step+=1、exp_avg.lerp_ 與平方梯度 recurrence；第528–546行 t 次方 bias correction、sqrt(v)/sqrt(1−β2^t)+eps、param.addcdiv_(...,value=−lr/(1−β1^t))。非 AMSGrad、零 weight_decay、maximize=False 的式子與本節相同。測試用 foreach=False/fused=False 限定 CPU單tensor路徑，不擴充成所有後端的浮點完全一致宣稱。
- `torch-optimizer-installed-commit.py`：親讀 `Optimizer.state_dict` 第700–722行，state 是 per-parameter，parameter 值本身沒有存入 optimizer state；`load_state_dict` 第900–906行接收 state_dict。本節說保留 m、v、步數是延續此方法的必要狀態，不是聲稱它們單獨足以重現整個訓練流程。
- `torch-sgd-installed-commit.py`：親讀第155–202行官方演算法與預設參數，第357–379行單tensor路徑。momentum=0、weight_decay=0、maximize=False 時，g不被改動，`param.add_(grad, alpha=-lr)` 正是參數減去lr×g，支持開頭兩格基本梯度下降的比較。
- `torch-tensor-docs-installed-commit.py`：第4872–4900行 `Tensor.sqrt`、`Tensor.square` 導向各自 torch function。`torch-docs-installed-commit.py` 第9582–9600行 `torch.tensor` 預設 requires_grad=False、複製 data、無 autograd history；第10995–11042行 sqrt/square 逐元素產生新 tensor；第12617–12634行 zeros 依 size 產生全零且預設 requires_grad=False。原 fence 所有 g,m,v,m_hat,v_hat,update 都是 shape[2] CPU float32、requires_grad=False、grad=None，既沒有求導，也沒有 parameter 或 optimizer.step；是手算給定梯度的應減去量。

## 實際 CPU 數值核對

原 fence 的 stdout 與章文相同。Decimal45位獨立代入：SGD移動量 `[.001,.100]`；m=`[.1,10]`、v=`[.001,10]`、m_hat=`[1,100]`、v_hat=`[1,10000]`。含 ε 的實數更新是約 `[.0009999999900000001,.0009999999999000000]`。原float32會把差異四捨五入為兩格 .001，所以正文「約」而非「完全相等」是正確。允許 atol=1e-6、rtol=2e-7 核對 float32 moment與推導；float64官方API對照 atol=1e-14、rtol=1e-12。

將原 fence 唯一改為第二格 −100，平方平均逐byte數值相同，m與update第二格變負，實際手動 p-=update 時第二格上升。另把 g全0，有ε時更新全0且有限；去ε時0/0全NaN。四個 torch.optim.Adam 第一格案例（原例、負梯度、零、小梯度 `[1e-10,-2e-10]`）與手算正確式一致；小梯度另與错误的 sqrt(v_hat+ε)比較，兩式明顯不同，確認 ε位置並非印字巧合。

兩步 history測試：g1=`[1,100]`、g2=`[1,−1]`，m2=`[.19,8.9]`、v2=`[.001999,9.991]`、m_hat2≈`[1,46.8421052632]`，應減去更新≈`[.00099999999,.000662580001]`，兩格確實不必同位移。在記憶體內保存 state_dict 再 load 後，第二步 parameter 與持續執行完全相同；只保存parameter而重建零state，第二格結果不同。只做四個首步和一個兩步 tiny vector檢查，無完整訓練、模型／資料下載、GPU、付費或上傳。

## 未解決的作用域疑問

原文最後說「Adam調步幅沒有顛倒下降的基本方向」。若只讀為本節明示的「只追第一步」／零歷史練習，負g確實導致負m_hat，減去负更新使參數上升。但這句沒有在自身標明此作用域，又連向1.12的當前梯度下降方向。原方法一般步由m_hat決定方向，而m_hat不必與當前g同號。上述兩步第二格g=−1、m_hat≈+46.8421，Adam仍讓parameter減少，與當前g的負梯度方向相反。不是程式錯、不是說分母會變負，也不是宣稱Adam必然失敗；待澄清的是不能從首步例子推論一般「仍沿當前梯度下降方向」保證。

判定 revise，建議局部改成：「在本例零歷史的第一步，負梯度使m_hat與應減去量都為負，參數因此增加；後續方向由累積的m_hat決定，未必與當前梯度同號。」不需要改公式或程式。

## 圖／頁面範圍

5.4 沒有 SVG、圖片、圖標籤、素材對應或空間資料流。讀者需比較的兩格位置已由 `[1,100]` 與逐格公式明示，文字及數值足以回答哪格被縮放，不需自行想像圖片。故本節 figure_consistency=not_applicable；沒有將字串搜尋或build成功冒充圖像／桌面手機頁面視覺檢查。1.12的更新文字只用於核對負號語義，沒有審閱其接字表成績或它的圖。
