# 教材發布

第一版以 GitHub Pages 提供閱讀網站，每節附實際 CPU 輸出、Colab 入口與 `.ipynb` 下載。正文仍是 `course/chapters/` 的單一來源，Notebook 由 `scripts/build_course.py` 產生。模型能力與 GPU 訓練配方尚待實測；Colab 自動設定已加入，尚未在 Google 執行環境實機驗證。

## 讀者怎麼使用

1. 開啟 <https://birdhackor.github.io/tiny-perceptron-vlm/>，從第一節開始或選閱讀路線。
2. 在網頁讀說明、看 SVG 與程式輸出。
3. 想改程式時，按本節「在 Colab 動手做」，登入 Google 並執行全部 cell。小實驗使用 CPU 即可。
4. 偏好本機 Jupyter，可下載 notebook，依暖身指南準備 repo 與套件。

Colab 第一格使用最新 `main` 的共用程式；網站與 Colab notebook 連結則指向本次發布的 commit。重新跑舊版實驗時，請改成對應 commit 的共用程式。Colab 執行使用其 Python 與 pip 環境；本機與網站 CI 使用 `uv.lock` 的固定 CPU 環境。

## 建置與驗證

在 repo 根目錄啟用已安裝的 `.venv`：

```bash
python scripts/build_course.py --check
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
python scripts/check_notebooks.py --mode kernel --workers 3
python scripts/export_course.py --executed outputs/notebooks --revision main
python scripts/check_site.py
```

`--executed` 會核對每個 cell 的原始碼、執行次數與錯誤，拒絕舊教材的結果或未執行副本。只要有一節失敗，發布流程就會停止。讀取模式可省略 `--executed`，但不會填入假輸出。

網站產物與已執行 Notebook 留在被 Git 忽略的 `outputs/`；發布時透過 Pages artifact 傳送，不把生成網頁或執行輸出提交到普通 Git。

## GitHub Pages

repo 的 Settings → Pages → Build and deployment → Source 選 **GitHub Actions**。之後 `.github/workflows/pages.yml` 在 `main` 更新或手動觸發時，會重新執行 222 個小節、核對輸出與網站連結，再部署到 Pages。核心 Linux／macOS／Windows CI 保留原本的流程。

若 Pages 已透過 API 設為 `build_type: workflow`，就不必再手動設定。網站的 `build-info.json` 留下發布 commit、節數與是否包含已核對的 CPU 輸出。
