import hashlib
import html
import json
import platform
import re
import subprocess


def get(url):
    raw = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "40", url],
        check=True,
        capture_output=True,
    ).stdout
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw.decode())))
    return raw, text


specs = [
    ("efficiency", "https://arxiv.org/html/2110.12894v2", "arXiv:2110.12894v2，2022-03-16", "Potential Disagreement between cost indicators；Making fair comparisons through an efficiency lens；Memory consumption", "no single cost indicator is sufficient.", "核對共享參數、參數／計算匹配、硬體及實作差異、速度與記憶體口徑。這些段落支持應同報品質與成本，也支持輪數與權重大小不能代替延遲及峰值記憶體。"),
    ("oracle", "https://www.ijcai.org/Abstract/16/107", "IJCAI 2016，Bias in Algorithm Portfolio Performance Evaluation，作者摘要", "Abstract，Virtual Best Solver 定義與一次執行的限制", "selects the best solver from a given portfolio of alternatives on a per-instance basis.", "核對每題完美選擇的虛擬最佳解題器，以及單次隨機紀錄不能外推期望效能的限制；正文把 oracle 限於固定紀錄是必要且正確的限定。"),
    ("objectives", "https://www.itl.nist.gov/div898/handbook/pri/section3/pri31.htm", "NIST/SEMATECH e-Handbook of Statistical Methods，讀取當日網站版", "What are the objectives?；Types of designs", "Planning an experiment begins with carefully considering what the objectives (or goals) are", "核對比較前須先確定目的，選擇成品與識別機制效果是不同實驗目標；此文件不支持本專案任何實測數值。"),
    ("tuning", "https://raw.githubusercontent.com/google-research/tuning_playbook/main/README.md", "Google Research Deep Learning Tuning Playbook，main 的讀取快照（完整 bytes SHA-256 隨收據保存）", "Identifying scientific, nuisance, and fixed hyperparameters；Designing the next round of experiments", "make fair comparisons between different settings of the scientific", "核對研究變量、干擾變量與固定變量的區分。資料、訓練與選權重條件需交代；同時不能把所有最佳化超參數硬設同值便稱為普遍公平。正文使用『盡量對齊』及指明比較參數或計算，未做普遍勝負推論。"),
]
items = []
for identifier, url, version, locator, excerpt, note in specs:
    raw, text = get(url)
    normalized = re.sub(r"\s+", " ", excerpt)
    assert normalized in text, identifier
    items.append({"id": identifier, "url": url, "accessed_on": "2026-10-20", "version": version, "full_response_sha256": hashlib.sha256(raw).hexdigest(), "full_response_bytes": len(raw), "locator": locator, "excerpt": excerpt, "inspection_note": note})
print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-18/inspect_primary_sources.py", "environment": {"python": platform.python_version(), "transport": "curl HTTPS"}, "sources": items, "storage": "只保存支持主張的短摘錄、讀取版本與全文雜湊；第三方全文未寫入 repo。"}, ensure_ascii=False, indent=2))
