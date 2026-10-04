"""Fetch authoritative, versioned PPO references for the independent 13.13 audit."""

import hashlib
import json
import platform
from pathlib import Path
from urllib.request import Request, urlopen

OUT = Path("docs/technical-reviews/artifacts")
PREFIX = "fact_v2_13_13"
COMMIT = "cbfd210bb8b08f6bc5c26878c10984b90f516c66"
SOURCES = [
    ("ppo_v2.pdf", "https://arxiv.org/pdf/1707.06347v2"),
    (
        "openai_train_policy.txt",
        f"https://raw.githubusercontent.com/openai/lm-human-preferences/{COMMIT}/lm_human_preferences/train_policy.py",
    ),
    ("spinningup.html", "https://spinningup.openai.com/en/latest/algorithms/ppo.html"),
]

records = []
for filename, url in SOURCES:
    request = Request(url, headers={"User-Agent": "IndependentTechnicalReview/1.0"})
    with urlopen(request, timeout=30) as response:
        data = response.read()
        record = {"url": url, "resolved_url": response.url, "status": response.status}
    path = OUT / f"{PREFIX}_{filename}"
    path.write_bytes(data)
    record.update(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    records.append(record)

result = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_13_fetch.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform()},
    "accessed_on": "2026-10-04",
    "result": "All three authoritative sources downloaded with verified TLS, no environment changes.",
    "sources": records,
}
(OUT / f"{PREFIX}_fetch.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
