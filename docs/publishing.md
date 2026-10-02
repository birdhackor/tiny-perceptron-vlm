# 教材發布

網站使用 [Zensical](https://zensical.org/) 建置，部署在 GitHub Pages。讀者可用全文搜尋、章節導覽與頁內目錄找內容，切換深淺色模式，並直接複製程式碼。小節頁附實際 CPU 輸出、Colab 入口與 `.ipynb` 下載，需要圖解的地方加入SVG。正文仍是 `course/chapters/` 的單一來源，Notebook 由 `scripts/build_course.py` 產生。30組正式訓練與Flash Attention補充探針已實跑，120份公開權重完成CPU操作驗收；各組結果與限制見[實驗紀錄](course-experiments/README.md)。Colab 自動設定已加入，尚未在 Google 執行環境實機驗證。

## 讀者怎麼使用

1. 開啟 <https://birdhackor.github.io/tiny-perceptron-vlm/>，從第一節開始或選閱讀路線。
2. 在網頁讀說明、看 SVG 與程式輸出。
3. 想改程式時，按本節「在 Colab 動手做」，登入 Google 並執行全部 cell。小實驗使用 CPU 即可。
4. 偏好本機 Jupyter，可下載 notebook，依暖身指南準備 repo 與套件。

Colab 第一個程式格使用最新 `main` 的共用程式；網站與 Colab notebook 連結則指向本次發布的 commit。重新跑舊版實驗時，請改成對應 commit 的共用程式。Colab 執行使用其 Python 與 pip 環境；本機與網站 CI 使用 `uv.lock` 的固定 CPU 環境。

## 建置與驗證

在 repo 根目錄安裝 Notebook 與網站套件，並啟用 `.venv`：

```bash
uv sync --frozen --extra cpu --group notebook --group site
source .venv/bin/activate
python scripts/build_course.py --check
python scripts/check_course_reviews.py
python scripts/check_technical_reviews.py
python docs/review-tools/check_review_round.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
python scripts/check_notebooks.py --mode kernel --workers 1
python scripts/export_course.py --executed outputs/notebooks --revision main
python scripts/check_site.py
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 啟用環境。`site` 群組固定 Zensical 版本；只跑訓練或 Notebook 時不需要它。

`export_course.py` 先把教材整理成 `outputs/zensical/docs/` 的 Markdown，附上核對過的輸出與圖解，再執行真正的 `zensical build --clean --strict`。Markdown 解析、頁面版型、搜尋索引與導覽都由 Zensical 負責。`zensical.toml` 收錄全部小節的導覽；建置會拒絕缺頁或重複項目。新增小節時，需同步更新 `course/lesson-index.json` 與導覽。

匯出後可執行 `python -m zensical serve` 在本機閱讀與測試搜尋。不要以雙擊 HTML 檔案的方式測試全文搜尋：瀏覽器會限制 `file://` 頁面的搜尋 worker。修改教材來源後重新執行匯出指令，便會重建供 Zensical 使用的 Markdown。若沒有網路，正文、SVG 與本機搜尋仍可透過這個伺服器閱讀，公式暫時保留 TeX，連網後由 MathJax 排版。

`use_directory_urls = false` 保留既有的 `1.1.html` 等網址；暖身的 `#W.2` 等錨點也保留。前置連結、舊書籤與 Notebook 中的閱讀連結不必更換。教材的額外 CSS／JavaScript 位於 `course/web/`，只處理圖解放大、公式與閱讀間距；導覽與搜尋使用 Zensical 原生功能。

`--executed` 會核對每個 cell 的原始碼、執行次數與錯誤，拒絕舊教材的結果或未執行副本。發布CI也逐節核對讀者與另一批技術審閱的來源、圖解及證據雜湊，並核對本輪新審閱身分與25份導言；缺少審閱、要求修改或版本已變更而未重審時，發布會停止。只要有一節程式失敗，發布流程也會停止。讀取模式可省略 `--executed`，但不會填入假輸出。

網站產物與已執行 Notebook 留在被 Git 忽略的 `outputs/`；發布時透過 Pages artifact 傳送，不把生成網頁或執行輸出提交到普通 Git。

## GitHub Pages

repo 的 Settings → Pages → Build and deployment → Source 選 **GitHub Actions**。之後 `.github/workflows/pages.yml` 在 `main` 更新或手動觸發時，會重新執行 222 個小節、核對輸出與網站連結，再部署到 Pages。核心 Linux／macOS／Windows CI 保留原本的流程。

網站 CI 使用一個 worker 逐份啟動獨立 kernel。GitHub runner 曾在並行啟動時發生 TCP 連接埠被占用、kernel 未開始執行就退出的情況；順序啟動避免多個 Notebook 同時爭用啟動資源。

若 Pages 已透過 API 設為 `build_type: workflow`，就不必再手動設定。網站的 `build-info.json` 留下發布 commit、節數、是否包含已核對的 CPU 輸出，以及 Zensical 版本。
