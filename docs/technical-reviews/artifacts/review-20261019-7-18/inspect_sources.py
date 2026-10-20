"""以記憶體讀取原始來源，只保存取得收據與支持主張的小段摘錄。"""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1

    def handle_data(self, value):
        if not self.skip:
            self.parts.append(value)


entries = [
    {
        "id": "instructgpt",
        "url": "https://arxiv.org/html/2203.02155v1",
        "version": "arXiv:2203.02155v1",
        "locator": "High-level methodology; Models; Evaluation; Who are we aligning to?",
        "excerpt": "We use the output of the RM as a scalar reward.",
        "inspected": "已讀示範的 supervised learning、比較資料訓練 RM、PPO 更新、RM 的 prompt/response 輸入及 scalar 輸出、held-out prompts 評估與標註者偏好的限制。支持一種流程，沒有保證 reward 等於普遍品質。",
    },
    {
        "id": "dpo",
        "url": "https://arxiv.org/html/2305.18290v2",
        "version": "arXiv:2305.18290v2",
        "locator": "Preliminaries; Direct Preference Optimization; DPO outline; Appendix B loss code",
        "excerpt": "without an RL training loop.",
        "inspected": "已讀比較資料的 winner/loser 定義、固定 pi_ref 的目標式及 log probability ratios、DPO gradient 的方向、SFT reference 的實務初始化與無 SFT checkpoint 時先擬合 chosen 的替代方式。它依比較資料直接更新 policy；可靠指標需要資料與任務標準相符。",
    },
    {
        "id": "deepseek",
        "url": "https://arxiv.org/html/2501.12948v1",
        "version": "arXiv:2501.12948v1",
        "locator": "DeepSeek-R1-Zero; Reward Modeling; Cold Start; Distillation v.s. Reinforcement Learning",
        "excerpt": "We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero",
        "inspected": "已讀 base model 直接 RL、GRPO、accuracy/format rule-based rewards、R1 的 long-CoT cold-start SFT 與後續 RL、多階段流程、評估方式與較小模型訓練的限制。無 SFT 路線特指 R1-Zero 的後訓練，不是否認 base model 已預訓練。",
    },
    {
        "id": "ppo_docs",
        "url": "https://spinningup.openai.com/en/latest/algorithms/ppo.html",
        "version": "OpenAI Spinning Up latest，取得日期與原文 SHA-256 見收據",
        "locator": "Background; Key Equations / You Should Know",
        "excerpt": "it is still possible to end up with a new policy which is too far from the old policy",
        "inspected": "已讀 PPO-Penalty 與 PPO-Clip 的區分、clipped surrogate 與正負 advantage 的作用、沒有硬約束的說明，以及 mean KL 門檻 early stopping。抑制過大更新並不保證每次只有小幅變更。",
    },
    {
        "id": "ppo_paper",
        "url": "https://arxiv.org/pdf/1707.06347v2",
        "version": "arXiv:1707.06347v2，2017-08-28",
        "locator": "Clipped Surrogate Objective, equation (7); Adaptive KL Penalty Coefficient; Algorithm",
        "excerpt": "",
        "inspected": "PDF 原文已由瀏覽工具實際讀取，另以 curl 取得同一版本 bytes 核對收據；此程式不解碼 PDF。原文定義 min(r*A,clip(r,1-epsilon,1+epsilon)*A)，對 surrogate clipping 而非把更新後 policy 投影回可行域；adaptive KL penalty 段也說實際 KL 可能顯著偏離目標。",
    },
]

for entry in entries:
    command = ["curl", "-fLsS", "--max-time", "40", entry["url"]]
    result = subprocess.run(command, check=True, capture_output=True)
    raw = result.stdout
    entry.update(
        command_argv=command,
        exit_code=result.returncode,
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        full_response_sha256=hashlib.sha256(raw).hexdigest(),
        full_response_bytes=len(raw),
        stderr=result.stderr.decode("utf-8"),
    )
    if entry["excerpt"]:
        parser = PlainText()
        parser.feed(raw.decode("utf-8"))
        text = re.sub(r"\s+", " ", " ".join(parser.parts))
        assert entry["excerpt"] in text
        assert len(entry["excerpt"].split()) <= 25
        entry["excerpt_exact_match"] = True
    else:
        assert raw.startswith(b"%PDF-")
        entry["pdf_signature_verified"] = True

print(json.dumps({"sources": entries, "full_sources_saved": False}, ensure_ascii=False, indent=2))
