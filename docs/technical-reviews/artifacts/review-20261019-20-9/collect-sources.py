"""讀取官方原文，保存版本、雜湊與支持主張的小段摘錄。"""

import json
import platform
import re
from datetime import UTC, datetime
from html import unescape

from audit import HERE, curl, plain, save, sha


minor = ".".join(platform.python_version_tuple()[:2])
specs = [
    {
        "id": "python_sets", "url": f"https://docs.python.org/{minor}/library/stdtypes.html",
        "locator": "Set Types — set, frozenset；set to set comparisons",
        "needles": ["unordered collection", "Two sets are equal if and only if every element of each set is contained in the other"],
        "quote": "Two sets are equal if and only if every element of each set is contained in the other",
    },
    {
        "id": "python_enumerate", "url": f"https://docs.python.org/{minor}/library/functions.html",
        "locator": "enumerate(iterable, start=0)",
        "needles": ["defaults to 0"],
        "quote": "from start which defaults to 0",
    },
    {
        "id": "gqa", "url": "https://cs.stanford.edu/people/dorarad/gqa/about.html",
        "locator": "Question Answering on Image Scene Graphs；場景圖、問題語意與不同題型診斷",
        "needles": ["objects, attributes and relations", "different question types"],
        "quote": "Each image is associated with a scene graph of the image's objects, attributes and relations",
    },
    {
        "id": "validation", "url": "https://scikit-learn.org/1.7/modules/cross_validation.html",
        "locator": "Cross-validation: evaluating estimator performance；Cross-validation iterators for grouped data / Group K-fold",
        "needles": ["evaluation metrics no longer report on generalization performance", "not represented in both testing and training sets"],
        "quote": ["evaluation metrics no longer report on generalization performance.", "the same group is not represented in both testing and training sets."],
    },
    {
        "id": "docci", "url": "https://google.github.io/docci/",
        "locator": "Abstract；Dataset",
        "needles": ["spatial relations, counting", "related or similar"],
        "quote": "crafted to clearly distinguish each image from those that are related or similar.",
    },
]
snapshots = []
for spec in specs:
    downloaded, command = curl(spec["url"])
    text = plain(downloaded)
    for needle in spec["needles"]:
        assert needle in text, (spec["id"], needle)
        position = text.index(needle)
        print(json.dumps({"source": spec["id"], "read_context": text[max(0, position - 150):position + 1000]}, ensure_ascii=False))
    title = re.search(r"<title[^>]*>(.*?)</title>", downloaded.decode("utf-8"), re.S)
    snapshots.append({
        "id": spec["id"], "url": spec["url"], "command": command,
        "download_sha256": sha(downloaded), "download_bytes": len(downloaded),
        "title": unescape(title.group(1)).strip() if title else spec["id"],
        "locator": spec["locator"], "short_excerpt": spec["quote"],
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "full_text_saved": False,
    })
save("primary-sources.json", {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(HERE.relative_to(HERE.parents[3]) / "collect-sources.py"),
    "environment": {"python": platform.python_version(), "device": "cpu"},
    "sources": snapshots,
    "result": "全部原文定位均已找到；只保存小段摘錄，不保存第三方全文。",
    "collector_note": "先前採集器預設的 same elements 措辭不在 Python 3.13 文件中；本次依實際原文 Two sets are equal if and only if 定位。課文實跑與練習變體已在先前採集器到達該定位之前完成。",
})
print(json.dumps({"result": "原始來源採集完成", "sources": len(snapshots)}, ensure_ascii=False))
