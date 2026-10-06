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

