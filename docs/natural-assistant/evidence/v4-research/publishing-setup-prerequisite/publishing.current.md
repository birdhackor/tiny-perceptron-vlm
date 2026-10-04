# 教材發布

本課的正文是一份來源，網頁與Notebook是兩種使用方式。網頁適合順著說明讀圖與結果；Notebook讓你改一個數字、再看結果怎樣變。兩者由同一份正文產生，讀者不必在兩本內容不同的教材之間來回找。

網站以[Zensical](https://zensical.org/)建置，發布到[GitHub Pages](https://birdhackor.github.io/tiny-perceptron-vlm/)。如果你只想學習，直接從[閱讀指南](../course/README.md)選路線即可；下面的建置步驟給想修改或維護教材的人使用。

## 1. 讀者怎麼使用？

1. 在網頁讀正文、看SVG圖解與已核對的CPU輸出。
2. 想修改短程式時，按本節的Colab入口，或下載`.ipynb`在本機Jupyter開啟。
3. 有程式的Notebook在第一個程式格準備工具，再按順序執行後面的程式格。沒有程式的小節則以閱讀為主。
4. 想重做完整訓練時，再走對應訓練指引，取得那項任務的資料與權重。

網站的Colab連結指向發布commit中的Notebook。其準備程式在Colab首次使用時會clone專案的預設分支；這與Notebook連結固定版本是兩回事。要重做舊版實驗，應先在Colab取得對應commit的專案再繼續準備工具，或在本機checkout那個版本。Colab使用其Python與pip環境；本機網站建置使用專案的`uv.lock`。本課沒有把本機kernel成功寫成Colab實機驗證。

## 2. 從正文產生網站

正文位於`course/chapters/`，操作與附錄則保留在各自的Markdown檔。`scripts/build_course.py`產生Notebook與小節索引，`zensical.toml`提供導覽；`scripts/export_course.py`整理網頁Markdown、圖解與已執行結果，再呼叫真正的Zensical嚴格建置。

先依[W.1](../course/first-steps.md#W.1)取得專案並安裝Git與uv，再到有`pyproject.toml`的專案根目錄安裝工具、啟用環境：

```bash
uv sync --frozen --extra cpu --group dev --group notebook --group site
source .venv/bin/activate
```

Windows PowerShell改用`.venv\Scripts\Activate.ps1`啟用。`notebook`群組提供執行工具，`site`群組提供網站建置，`dev`群組提供檢查與測試；只讀網頁的學生無須安裝它們。

核對教材與執行結果：

```bash
python scripts/build_course.py --check
python scripts/check_course_reviews.py
python scripts/check_technical_reviews.py
python docs/review-tools/check_review_round.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
python scripts/check_notebooks.py --mode kernel --workers 1
python scripts/reading_time.py validate --executed
```

這些檢查要求目錄、正文、圖解、審閱資料、Notebook與閱讀時間對上目前來源。若修改正文，先重審、重估受影響的頁面，不能用舊結果替新內容背書。檢查的支持範圍見[實作驗證](validation.md)。

全部通過後，以目前Git commit作為網站版本：

```bash
python scripts/export_course.py --executed outputs/notebooks --require-reading-times --revision "$(git rev-parse HEAD)"
python scripts/check_site.py
python -m http.server 8000 --directory outputs/site
```

先把這次要發布的教材與核對資料commit，再執行上面的匯出。`--revision`是網站入口的版本標記，不會替你保存尚未commit的修改。`--executed`核對每個程式格與輸出，`--require-reading-times`要求每頁估時仍符合來源；網站檢查再核對內部連結、圖檔、Notebook下載與Colab入口。

開啟`http://127.0.0.1:8000/`查看產物。用HTTP伺服器測試，才能讓搜尋正常使用瀏覽器的網頁機制；直接雙擊HTML檔案會受到不同限制。網址中的`127.0.0.1`指向執行伺服器的那臺電腦。

## 3. 發布到GitHub Pages

專案的Settings → Pages → Build and deployment → Source選擇GitHub Actions。`.github/workflows/pages.yml`在`main`更新或手動觸發時，取得必要資料、核對審閱、逐份啟動獨立CPU kernel、嚴格建置網站並檢查連結，再由Pages部署工作發布。

Notebook數量由當前小節目錄決定，新增小節時也要同步來源、索引與導覽。網站產物與已執行Notebook放在被Git忽略的`outputs/`，透過Pages artifact傳送；正文與核對證據留在Git中。

發布後查看`build-info.json`，核對實際commit、頁面與小節數、CPU輸出狀態及Zensical版本，再實際點開新增頁面與下載入口。工作顯示成功和讀者拿到正確版本都需要確認。資料包、模型配套與網站是三條不同的發布路線，前兩者的規則見[教材資產存放](asset-storage.md)。
