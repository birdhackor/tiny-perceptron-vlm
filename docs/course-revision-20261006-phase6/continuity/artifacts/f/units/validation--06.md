## 在自己的版本重跑

先按[W.1](../course/first-steps.md#W.1)準備專案與Notebook環境，啟用`.venv`後，在專案根目錄執行：

```bash
python scripts/build_course.py --check
python scripts/check_course_reviews.py
python scripts/check_technical_reviews.py
python docs/review-tools/check_review_round.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
python scripts/check_notebooks.py --mode kernel --workers 1
pytest -ra
```

前四行核對教材與審閱資料；接著註冊kernel，讓每份Notebook由獨立執行環境開始；最後的單元測試檢查程式機制。Notebook執行副本與當次核對寫入`outputs/notebooks/`，它們是工作產物，不是模型能力分數。

如果改了教材，舊報告或舊輸出可能被拒絕；應重讀、重查受影響內容，再重新執行改過的Notebook。若另做模型訓練，則按對應訓練指引保存配置與留出題結果。網站建置與連結核對接著見[教材發布](publishing.md)。本機CPU執行也不等於Colab雲端或所有學生設備已經驗證。
