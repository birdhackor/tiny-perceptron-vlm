"""用 curl 取得原始來源；只保存收據與支持主張的短摘錄。"""

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parent
PPO_COMMIT = "038665d62d569055401d91856abb287263096178"
SOURCES = [
    {
        "id": "original_specification",
        "url": "https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/",
        "version": "2020-04-21 文章；本次實際傳回版本以 response_sha256 綁定",
        "locator": "文章開頭定義、積木範例、reward design 與 learned reward model 段落",
        "excerpt": "without achieving the intended outcome",
        "anchors": ["misspecification", "reward design"],
        "inspection_note": "定義以行為是否滿足字面規格、是否達到預期結果為準；積木例顯示高獎勵可與任務完成分離。文章區分 RL 演算法設計和獎勵設計，並說明 learned reward model 仍可能失配。心理欺騙意圖不是該行為定義的必要條件，這是對定義的直接推論。",
    },
    {
        "id": "original_feedback",
        "url": "https://deepmind.google/blog/learning-through-human-feedback/",
        "version": "2017-06-12 文章；本次實際傳回版本以 response_sha256 綁定",
        "locator": "含 susceptible to reward hacking 的 reward predictor 失敗案例段落",
        "excerpt": "reward hacking - or gaming its reward function",
        "anchors": ["discontinued early", "reward predictor"],
        "inspection_note": "研究團隊直接把 reward hacking 與 gaming reward function 並列，描述 reward predictor 的弱點導致行為偏離真正任務；補強本文使用 reward hacking 一詞的對應範圍。",
    },
    {
        "id": "original_ppo",
        "url": f"https://raw.githubusercontent.com/openai/spinningup/{PPO_COMMIT}/docs/algorithms/ppo.rst",
        "version": f"openai/spinningup commit {PPO_COMMIT}",
        "locator": "Background；Key Equations；Advantage is positive／negative；You Should Know",
        "excerpt": "clipping serves as a regularizer",
        "anchors": ["A^{\\pi_{\\theta_k}}(s,a)", "too far"],
        "inspection_note": "逐式核對 L=min(ratio*A, clip(ratio)*A) 與正負 advantage 的說明。裁切針對策略更新的誘因，沒有查核文字是否算對或是否符合真正需求的運算。由公式與本次兩個回答的反例可推知它不會改寫獎勵判準；這不是主張裁切必能阻止任意大的更新。",
    },
]


def main():
    receipts = []
    for source in SOURCES:
        argv = ["curl", "-fsSL", "--compressed", "--max-time", "40", source["url"]]
        response = subprocess.run(argv, check=True, capture_output=True)
        raw = response.stdout
        text = raw.decode("utf-8")
        if "deepmind.google" in source["url"]:
            text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.S)
            text = unescape(re.sub(r"<[^>]+>", " ", text))
        text = " ".join(text.split())
        assert source["excerpt"] in text
        for anchor in source["anchors"]:
            assert anchor in text, (source["id"], anchor)
        receipts.append({**source, "command_argv": argv, "exit_code": response.returncode,
                         "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
                         "response_sha256": hashlib.sha256(raw).hexdigest(), "response_bytes": len(raw),
                         "checked_original": True, "full_response_saved": False})
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-16/fetch-sources.py",
        "environment": {"python": platform.python_version(), "curl": subprocess.run(["curl", "--version"],check=True,capture_output=True,text=True).stdout.splitlines()[0]},
        "result": "原始來源的短摘錄與定位均已核對；不保存第三方全文。",
        "receipts": receipts,
    }
    (ARTIFACT / "sources.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
