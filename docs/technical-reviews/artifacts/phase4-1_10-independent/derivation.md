# 1.10 獨立推導與查證範圍

Reviewer: `/root/phase4_factual_coordinator/factual_1_10`，fresh context。

親讀範圍：`course/chapters/01.md` 原始第 1–260 行（含章節導言），第 270–402 行（1.8、1.9、1.10 全節與 1.11 開頭）；1.10 正式範圍第 351–390 行。完整讀取 `docs/review-tools/factual-reviewer-instructions.md`、`scripts/check_technical_reviews.py`、`docs/review-tools/section_facts.py` 與實際 bootstrap。未讀舊 1.10 審閱、歷史正文或結論。

## 鏈式法則與單位

設 `u=c*w`、`L=u²=c²*w²`。對本節可微的實純量函數，

`dL/du=2*u`，`du/dw=c`，故 `dL/dw=(2*u)*c=2*c²*w`。

導數是微小增量比的極限，不是函數值的相除。在 w=3、c=2 時，`u=6`、`L=36`，`L/u=6`，但 `dL/du=12`。所有量都是無物理單位的教學純量；敏感度可寫作每一 w 單位對應多少 L 單位。本例無 batch、類別、平均、token 或樣本軸，PyTorch w、u、loss 都是 shape `[]` 的零維 tensor。

若 `h=0.001`，則 `Δu=c*h=0.002`，

`ΔL=(u+Δu)²-u²=2*u*Δu+(Δu)²=0.024+0.000004=0.024004`。

教材說「附近」「約」的 0.024 是正確的一階近似；它不是精確有限步結果。精確前向差商的分母是 h，為 `0.024004/0.001=24.004`；極限導數是 24。對稱差分分母是 `2h=0.002`，此二次式在實數運算中恰為 24；float64 執行為 23.999999999997357。數值核對事先設定：ΔL 絕對差 1e-12；對稱導數絕對差 1e-10；可表示整數的數值和梯度採精確相等。正文的「每一單位」按局部率解讀，沒有聲稱對有限 Δu=1 精確增加 12。

## 練習与分支

c=3、w=3 時，u=9、L=81、dL/du=18、du/dw=3，dL/dw=54。相對 c=2，梯度比是 `(3/2)²=2.25`，不是 1.5；因為乘數和 u 所在點同時改變。

若同一 w 經兩條路到結果，總導數為各路貢獻的和。檢查 `L=(2*w)²+w²`：w=3 時 L=36+9=45，兩路梯度為 24 與 6，相加為 30。此額外小變化只核對本節補充的路徑相加說明。

## 程式實際做什麼

原 fence 逐 byte 保存且執行成功。`requires_grad=True` 在此正常 grad mode 中讓乘法與平方留下 backward 關係；w 是 leaf，u 的 grad_fn 是 MulBackward0，L 的 grad_fn 是 PowBackward0。原程式自己顯示兩段解析敏感度 12 和 2；autograd 實際算出的總梯度是 w.grad=24，沒有讀取 u.grad。`loss.backward()` 不執行 optimizer 或更新；原碼與 c=3 練習後 w 都保持 3。`.item()` 將單元素 tensor 值取成 Python number；此運算不保留可微關係。`.detach()` 返回無 grad 關係的 tensor，原 graph 不因這次顯示改寫；其儲存可與 u 共享，本節沒有對 detached tensor 做原地寫入。這是求導示範，不是模型訓練或性能評測。

開頭與末尾關於接字分數→機率→答案代價的銜接，與 1.8 的 class-index cross-entropy 相符。官方 CrossEntropyLoss 的未加權、無 label smoothing 公式是 `-log(exp(z_y)/sum_c exp(z_c))`。本節執行的平方代價不代表實際使用這種答案代價訓練了文字模型。

## 圖與工具限制

親读当前 `rewrite-01-chain.svg` 原碼，Chromium 151 在 15 秒上限內未完成，沒有可用 screenshot；保留 stderr 與真實 exit 124 receipt。Inkscape 1.4 成功輸出 640×795 PNG，已用 view_image 親看：向下兩箭頭依序乘 2、平方，w=3→u=6→L=36；下方 12×2=24 與最下方代價36/梯度24標示完整，中文字、乘號、數字無缺失、重疊或裁切。此通過是 SVG 的 Inkscape 實際渲染核對，未聲稱已驗證瀏覽器版面。

官方 docs.pytorch.org 的初次 batch 取得回應 HTTP 403。替代來源全部是 PyTorch 官方 GitHub repository 的 v2.11.0 原始文件/原碼，已親讀下列段落，原始 bytes、URL、日期與 SHA 都保存。實測環境為 Python3.13.5、PyTorch2.14.1+cpu（git 5c4886908584029761b579af026dcfb627c84070），不同於來源文件 v2.11.0；不假定兩版全部行為相同。另親讀本機實際 `torch.Tensor.backward` 程式與 docstring，其本例使用的 scalar/chain-rule/leaf-gradient 契約一致；保存本機來源指紋與函式正文。這次只檢查本節用到的 API 路徑。

實際權威原始段落：

- `docs/source/notes/autograd.rst` 第 11–36、178–202 行：reverse-mode graph、chain rule、requires_grad、leaf .grad。
- `torch/_tensor.py` 第 576–636、806–823 行：Tensor.backward scalar 契約、累積 leaf 梯度、委派 autograd.backward；detach 返回不需梯度 tensor、共享 storage。
- `torch/_torch_docs.py` 第 124–137、9231–9285、10610–10631 行：requires_grad factory argument、leaf tensor/scalar、元素平方。
- `torch/_tensor_docs.py` 第 2821–2838、4975–4982 行：item 單元素 Python number、不支持求導；Tensor.square 對應 torch.square。
- `tools/autograd/derivatives.yaml` 第 229–237、1217–1225、1385–1397 行：加法、常數乘法和冪的 backward 規則登記。
- `torch/csrc/autograd/input_buffer.cpp` 第 94–134、203–217 行：CPU graph input-buffer 收到多次貢獻會用 add 累加。
- `torch/nn/functional.py` 第 3421–3512 行：cross_entropy 接受 unnormalized logits 與 target，委派 backend。
- `torch/nn/modules/loss.py` 第 1194–1237 行：class-index CE 的負對數 softmax 公式與 LogSoftmax+NLLLoss 等價。

未下載資料集或模型、未執行 GPU/完整訓練/上傳/付費/commit；未改教材或 SVG。
