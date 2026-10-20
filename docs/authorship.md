# 教材的撰寫方式

課文說某份標註或判準「由 AI 助理撰寫，沒有逐題經人審核」時，依據是這份作者聲明。

## 作者的說明

2026-10-20，Claude Code 問到第 20 章 AISHELL 問句的回答判準由誰撰寫、作者有沒有審過時，作者回答：

> 我沒親自審過，……整份教材我目前只有指引方向，以及看過一部分之後抓"寫法"問題(不是具體問題)之後整份改寫，所以具體措辭都不會是我審過的

也就是說：作者決定方向，讀過一部分內容後指出寫法上的問題，再請 AI 工具整份改寫。課文、資料標註與判準的具體措辭由 AI 工具寫成，作者沒有逐條審過，本課也沒有安排其他人逐條審核。

## 涵蓋的判準

第 20 章 8 題 AISHELL 問句（驗證、測試各 4 題）的回答判準，寫在 `scripts/prepare_natural_v4_voice_questions.py` 的 `SPEC`；準備程式把它們逐字寫進 `docs/natural-assistant/v4/data/voice-question-sources.json` 每一筆的 `semantic_rubric`。這些判準由 AI 工具撰寫，沒有逐題經人審核。

寫下這份說明時，兩個檔的 SHA-256：

- `scripts/prepare_natural_v4_voice_questions.py`：`6c011e4815e2cb8e80c9c9b9c33f5ae57ada28620bcf6575e3ac6c0b0b96ef8e`
- `docs/natural-assistant/v4/data/voice-question-sources.json`：`0efb9dc7b18fa56d2cb479fb7c01996625c959efcb6633b55e01f4fa0b00b120`
