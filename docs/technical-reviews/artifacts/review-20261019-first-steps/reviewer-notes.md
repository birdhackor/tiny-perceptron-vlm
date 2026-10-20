# 核對記錄

## 可見範圍與獨立性

完整閱讀指定凍結原文、技術審閱規約，以及 clear-tutorial 的技術核對準則與 project-context。沒有參與教材撰寫，也沒有讀取其他審閱者的報告、處置、seal、閱讀計畫、提示詞或歷史審閱內容。

起始執行的 `git status --short` 意外列出其他審閱目錄、提示詞及 dispatch-log 的路徑和修改狀態；沒有開啟這些檔案，也沒有看到其結論或正文。後續搜尋只指定教材、程式、測試與 workflow，未搜尋被排除的審閱樹。這次路徑中繼資料暴露列為訊號 C。

## 透明推導

- 四張等機會的卡中三張貓、一張狗，所以 P(貓)=3/4、P(狗)=1/4；兩個互斥而窮盡的候選機率之和是 1，0.9+0.9=1.8 不符合此條件。
- 自然對數是 exp 的反函數。對 0<p≤1，−ln(p)≥0；其導數為 −1/p<0。p→0+ 時 −ln(p)→+∞。若 p1、p2>0，令 p1=e^a、p2=e^b，則 p1p2=e^(a+b)，因此 −ln(p1p2)=−ln(p1)−ln(p2)。兩事件獨立時，聯合機率才可寫成邊際機率的乘積；本例 0.5×0.5=0.25，−ln(0.25)=1.3862943611…。
- 成績列平均：(60+70+80)/3=70，(80+90+100)/3=90。欄平均：(60+80)/2=70、(70+90)/2=80、(80+100)/2=90。把 80 改成 110 後，第一列平均為 80，第三欄平均為 105，其餘不變。
- 矩陣乘法 C[i,j]=Σ_k recipes[i,k]×weights[k,j]。成本列為 2×10+1×20+0×30=40 和 1×10+2×20+1×30=80；甜度為 2×0+1×1+0×4=1 和 1×0+2×1+1×4=6。加上 bias=[5,0] 得 [45,1]、[85,6]；改為 [8,0] 得 [48,1]、[88,6]。
- L(w)=(w−3)²。差商 [L(w+h)−L(w)]/h=2(w−3)+h，因此導數為 2(w−3)。在 w=1 為 −4，在 w=1.4 為 −3.2。L(1)=4，L(1.1)=3.61，差為 −0.39。更新 w'=w−lr×L'(w) 後，lr=0.1、0.6、1.1 分別給 w'=1.4、3.4、5.4 與代價 2.56、0.16、5.76。對此二次式，一步後的代價為 4(1−2lr)²；負梯度只保證適當小步的局部下降，不能保證任意步長改善。
- 3<<4=48，48|5=53，53>>4=3，53&15=5。這是兩個非負四位元整數的打包示例，不把有符號位移或任意大整數強說成同一種儲存配置。
- Python 的 `//` 一般規則是向下取整；本頁用的正整數 `7//2=3` 正確。另實測 `-7//2=-4`、`trunc(-7/2)=-3`，所以不把正文的正整數「捨餘數」例子外推成對負數向零截斷的規則。

## 實際看過的圖

使用 view_image 親看：

- `outputs/course-review-20261019/figure-renders-04/74c93bfd89b0b164573f8dea1506179982c79250b77bfc862102895b02387138.png`。上表六個數與同學／小考方向正確；下方兩種分組、箭頭、dim 標記與結果一一對應正文。箭頭表示分組歸約，沒有誤畫成資料交換。
- `outputs/course-review-20261019/figure-renders-04/e81b9f6de2af7b7d9d8bc9cbe32528794d207d1817014c46624d1619820ca942.png`。橫軸 w、縱軸 L，藍點 (1,4)、橘點 (1.4,2.56)、最低點 (3,0) 與右下的更新箭頭都正確；箭頭是兩次狀態的連接，不作切線解讀。另由 SVG 核對座標 x=75+123w、y=430−34L；161 個折線頂點的最大縱向誤差小於 0.01 pixel，符合 SVG 小數取整。

PNG 僅為指定凍結 SVG 的外部算繪證據，沒有複製或列為 artifact；其指紋存於 verification.json。

## 執行範圍與限制

8 段 Python 原碼、表格例子、練習變體均以 CPU 執行。主要環境為 Python 3.13.16、PyTorch 2.14.1、macOS arm64；數值整數結果精確核對，float32 梯度與更新使用絕對誤差 1e-6。Notebook 另暫建鎖定環境，使用 ipykernel 7.4.0、JupyterLab 4.6.4、nbclient 0.11.0，重用既有 PyTorch，完成 kernel 登記、新 kernel 執行與 Jupyter HTTP contents API 檢查後移除。這不是從空機重裝全部核心依賴的實驗；`uv sync` 使用 dry-run，Notebook 依賴則實際依鎖定輸出安裝。

Notebook 程式輸出正確且沒有 cell error。關閉時 ipykernel 的 psutil 子程序枚舉遇到 `PermissionError: [Errno 1] Operation not permitted (originated from sysctl() malloc 1/3)`；這是程序查詢被沙箱限制，不是檔案寫入被拒。Jupyter 啟動有 yarn / Node 的 `Dynamic require of "util" is not supported` 訊息，仍成功啟動伺服器並回傳指定 Notebook。完整 stderr 與伺服器輸出保存在 notebook-execution.json；不據此聲稱這台環境關閉時完全無警告。

沒有操作瀏覽器圖形介面、Windows PowerShell、Linux 或 Colab runtime。Windows 啟用與長路徑、Colab 登入與 bootstrap 分支，以官方文件／原始碼及目前 repo 命令核對；沒有宣稱跨平台實測通過。Colab 未實測正是原文明示的限制。網站的 CPU 輸出查驗使用公開 build-info、公開實例與發布程式／workflow；沒有重跑全書，也沒有開啟其他審閱的私有執行收據。沒有長時間訓練，只有不更新權重的一步 dry run。

導覽表的每個明列與範圍端點都已展開核對現行章檔；只核對此表所說的實作說明位置，不延伸審查那些章的技術內容。

## 核對工具的初始失敗

以下是本回合前景執行的輸出複製品，不是教材執行錯誤：

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import sys,platform,torch,importlib.metadata; print(sys.version); print(platform.platform()); print(torch.__version__); print({n:importlib.metadata.version(n) for n in ["jupyterlab","ipykernel","nbclient"]})'
3.13.16 (main, Oct  2 2026, 08:32:25) [Clang 21.0.0 (clang-2100.1.1.101)]
macOS-26.4.1-arm64-arm-64bit-Mach-O
2.14.1
importlib.metadata.PackageNotFoundError: No package metadata was found for jupyterlab
```

後續 verification.json 逐套件記錄缺少的 Notebook 套件，另以 notebook-execution.json 保留暫建環境的實際結果。

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-first-steps/inspect_repository.py
    cell = row.split(" | ")[2]
IndexError: list index out of range
```

自製導覽表核對器的欄位索引已修正後重跑，完整結果在 repository-inspection.json。未產生或改寫任何已保存的正式判斷。

## 升級訊號

訊號A：無 — 查核方式：核對凍結原文、freeze-04.json 的指定頁面與 SHA、兩張 SVG／PNG、check_technical_report.py 及 scripts/check_technical_reviews.py；實跑 verify.py、inspect_repository.py、check_notebook_setup.py 與 curl 原始來源。未發現派工單的現況宣稱被反證。
訊號B：無 — 查核方式：派工單未附「已駁回／不要重提」清單，沒有可核對的編號區間。
訊號C：有 — 起始 git status 僅暴露其他審閱檔案的路徑與修改狀態，未讀其內容；跨平台與圖形介面未實跑，已在限制中區分。Notebook 關閉時的程序枚舉受沙箱限制，但指定程式與 contents API 核對完成；未碰到檔案寫入被拒。
訊號D：無 — 未修改教材；在指定原文、圖與本次核對所讀的程式中未見可辨認的裁決編號或修復註解，亦未讀歷史紀錄推測修訂來歷。
