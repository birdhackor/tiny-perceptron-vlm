## C.2 最後答對代表過程正確嗎？

回答2+2寫「2+2=5；5−1=4」，結尾4對，第一步卻錯。檢查不能只讀最後一個數字，還要分開查每條等式、前後是否接續，以及是不是在解原題。

每步 `(a,b,claimed)` 是宣稱a+b=claimed，減1寫成加-1。先把這份錯解編成兩個三元組：

```python
steps = [(2, 2, 5), (5, -1, 4)]
truth = 2 + 2
valid = [a + b == claimed for a, b, claimed in steps]
linked = [steps[i][0] == steps[i - 1][2] for i in range(1, len(steps))]
print("等式算對", valid)
print("接上前步", linked)
print("最後答案對", steps[-1][2] == truth)
```

等式是False、True；接續True；最後答案True。第二步真的從前步聲稱的5開始，卻不能救第一步。只把第一步claimed改4，等式都True，第二步仍從5開始，接續便False。

即使等式與接續都成立，還要核對原題參數與任務（短程式尚未檢查這一項）：無理由地多加一個數再減回，不能只靠算術局部正確就接受。原題其實只需2+2=4。驗證器是依明確規則核對輸出的程式，應說明它驗了哪些條件，不稱通用文字證明器。

舊模型 `(5+0)+3` 生成 `5+0=5;5+3=10;answer=10`，格式和接續都通過，第二條加法仍錯。這個例子讓我們把「能解析」與「算對」分開；通過窄驗證器也不證明輸出是內部推理的忠實記錄。

舊候選的完整紀錄見[正式報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/reasoning.json)。

