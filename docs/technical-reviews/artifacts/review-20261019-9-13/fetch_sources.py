"""以 curl 讀原始來源，只保存定位、原檔雜湊與短摘錄。"""

from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


SPECS = [
    {
        "id": "noul",
        "url": "https://docs.typesafe.ai/primitives/noul",
        "locator": "Noul 的開頭定義與回傳值說明",
        "phrase": "A Noul question asks the TypeSafe model to evaluate a yes/no question and return the probability that the answer is yes.",
        "inspect": ["A Noul answer is a single number", "0 means no and 1 means yes"],
    },
    {
        "id": "dict",
        "url": "https://docs.python.org/3.13/library/stdtypes.html",
        "locator": "Mapping Types — dict；get、d[key]、key in d",
        "phrase": "Return the value for key if key is in the dictionary, else default",
        "inspect": ["it defaults to None", "this method never raises a KeyError", "key not in d Equivalent to not key in d"],
    },
    {
        "id": "control",
        "url": "https://docs.python.org/3.13/tutorial/controlflow.html",
        "locator": "if Statements",
        "phrase": "The keyword ‘ elif ’ is short for ‘else if’",
        "inspect": ["There can be zero or more elif parts"],
    },
    {
        "id": "lists",
        "url": "https://docs.python.org/3.13/tutorial/datastructures.html",
        "locator": "List Comprehensions；依 for 與 if 建構新串列",
        "phrase": "A list comprehension consists of brackets containing an expression followed by a for clause, then zero or more for or if clauses.",
        "inspect": ["The result will be a new list"],
    },
    {
        "id": "distillation",
        "url": "https://arxiv.org/html/1503.02531v1",
        "locator": "Introduction；class probabilities 作為 soft targets 的段落",
        "phrase": "use the class probabilities produced by the cumbersome model as “soft targets” for training the small model.",
        "inspect": ["For this transfer stage", "2 Distillation"],
    },
    {
        "id": "threshold",
        "url": "https://scikit-learn.org/stable/modules/classification_threshold.html",
        "locator": "Important notes regarding the internal cross-validation；cv=prefit 的限制",
        "phrase": "You should never use the same data for training the classifier and tuning the decision threshold due to the risk of overfitting.",
        "inspect": ["new validation set", "internal cross-validation"],
    },
    {
        "id": "calibration",
        "url": "https://proceedings.mlr.press/v70/guo17a.html",
        "locator": "原始論文頁 Abstract；模型機率與實際正確性的區分",
        "phrase": "We discover that modern neural networks, unlike those from a decade ago, are poorly calibrated.",
        "inspect": ["true correctness likelihood", "temperature scaling"],
    },
    {
        "id": "ood",
        "url": "https://openaccess.thecvf.com/content_CVPR_2019/html/Hein_Why_ReLU_Networks_Yield_High-Confidence_Predictions_Far_Away_From_the_CVPR_2019_paper.html",
        "locator": "Abstract；ReLU 型分段線性分類器的遠離訓練資料之高信心行為",
        "phrase": "they produce almost always high confidence predictions far away from the training data.",
        "inspect": ["ReLU type neural networks", "piecewise linear classifier function"],
    },
]


def main():
    receipts = []
    for spec in SPECS:
        command = ["curl", "-fLsS", "--max-time", "30", spec["url"]]
        run = subprocess.run(command, capture_output=True, check=False)
        assert run.returncode == 0, (spec["id"], run.stderr.decode())
        parser = PlainText()
        parser.feed(run.stdout.decode())
        plain = " ".join(" ".join(parser.parts).split())
        assert spec["phrase"] in plain, spec["id"]
        checks = {needle: needle.lower() in plain.lower() for needle in spec["inspect"]}
        assert all(checks.values()), (spec["id"], checks)
        index = plain.index(spec["phrase"])
        receipt = {
            "id": spec["id"],
            "url": spec["url"],
            "command_argv": command,
            "exit_code": run.returncode,
            "accessed_at": datetime.now().astimezone().isoformat(),
            "response_bytes": len(run.stdout),
            "response_sha256": hashlib.sha256(run.stdout).hexdigest(),
            "locator": spec["locator"],
            "excerpt": spec["phrase"],
            "excerpt_plaintext_offset": index,
            "inspection_checks": checks,
            "document_prefix": plain[:100] if spec["id"] == "threshold" else "",
        }
        receipts.append(receipt)
        print(json.dumps({"id": spec["id"], "excerpt": spec["phrase"], "context": plain[max(0, index-250):index+len(spec["phrase"])+650]}, ensure_ascii=False), flush=True)
    destination = Path(__file__).with_name("source-receipts.json")
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"method": "curl 讀取完整原文後，僅保存原檔雜湊、定位與支持主張的短摘錄。第三方全文未寫入 repo。", "sources": receipts}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
