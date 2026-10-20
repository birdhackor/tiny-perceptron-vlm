"""核對凍結章首的導覽、短算例、外部原文及有限世界訓練依據。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import torch

from tiny_perceptron.lesson_products.data import c_rows


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    path = ARTIFACT / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def curl(url):
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--max-time", "30", "--write-out", "\n__CURL_META__%{json}", url,
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    body, metadata = completed.stdout.rsplit(b"\n__CURL_META__", 1)
    info = json.loads(metadata)
    return body, {
        "command": command,
        "http_code": info["http_code"],
        "url_effective": info["url_effective"],
        "content_type": info["content_type"],
        "downloaded_bytes": len(body),
        "response_sha256": sha(body),
        "exit_code": completed.returncode,
        "stderr": completed.stderr.decode(),
    }


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def main():
    torch.set_num_threads(1)
    timestamp = datetime.now().astimezone().isoformat()
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "platform": platform.platform(),
        "curl": subprocess.run(["curl", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
    }
    command = (
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python "
        + Path(__file__).relative_to(ROOT).as_posix()
    )
    freeze_path = "docs/course-review-20261019/freeze-04.json"
    freeze_bytes = (ROOT / freeze_path).read_bytes()
    assert sha(freeze_bytes) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412", "凍結清單版本不符"
    pages = {page["page_id"]: page for page in json.loads(freeze_bytes)["pages"]}
    intro = pages["chapter-09"]
    intro_bytes = (ROOT / intro["snapshot"]).read_bytes()
    assert sha(intro_bytes) == "614054e88d3ce6480d076c995040acd47971c3e570b4a33fc97766a973e5aa71", "章首凍結版本不符"
    chapter = (ROOT / "course/chapters/09.md").read_text()
    assert chapter.split("\n## ", 1)[0] + "\n" == intro_bytes.decode(), "章首目前內容與凍結版本不同"
    assert intro["figures_sha256"] == {} and "![" not in intro_bytes.decode(), "章首圖清單不符"

    selectors = {
        "visible_information": "9.2",
        "false_premise": "9.3",
        "permission": "9.4",
        "related_help": "9.5",
        "wording_split": "9.8",
        "three_candidates": "9.9",
        "four_confidences": "9.10",
        "typed_probability": "9.11",
        "missing_information": "9.13",
        "cost_decision": "9.16",
    }
    runs = {}
    namespaces = {}
    for name, identifier in selectors.items():
        page = pages[identifier]
        raw = (ROOT / page["snapshot"]).read_bytes()
        assert sha(raw) == page["source_sha256"], "支持頁面的凍結指紋不符"
        code = re.findall(r"```python\n(.*?)\n```", raw.decode(), re.S)[0]
        namespace = {}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, page["snapshot"], "exec"), namespace)
        namespaces[name] = namespace
        runs[name] = {
            "snapshot": page["snapshot"], "source_sha256": sha(raw),
            "code_sha256": sha(code.encode()), "stdout": output.getvalue(),
        }

    scores = namespaces["three_candidates"]["scores"]
    candidates = []
    for scale in (1.0, 10.0):
        p = (scores * scale).softmax(-1)
        confidence, prediction = p.max(-1)
        candidates.append({
            "scale": scale, "probabilities": p.tolist(),
            "confidence": confidence.item(), "prediction": prediction.item(),
            "correct": prediction.item() == 1,
        })
    assert scores.shape == (1, 3), "候選維度與教材不同"
    assert all(row["prediction"] == 0 and not row["correct"] for row in candidates), "尺度示例的候選或對錯不同"
    assert abs(candidates[0]["confidence"] - 0.6652409558) < 1e-7, "原尺度信心不符"
    assert 0.9999 < candidates[1]["confidence"] < 1.0, "放大尺度的信心不符"

    four = namespaces["four_confidences"]
    confidence = four["confidence"]
    correct = four["correct"]
    four_result = {
        "confidence": confidence.tolist(), "correct": correct.tolist(),
        "count": len(confidence), "mean_confidence": confidence.mean().item(),
        "accuracy": correct.float().mean().item(), "bins": four["bins"], "ece": four["ece"],
    }
    assert len(confidence) == 4 and len(correct) == 4, "信心與標籤筆數不同"
    assert abs(four_result["mean_confidence"] - 0.75) < 1e-7, "平均信心不符"
    assert four_result["accuracy"] == 0.5 and abs(four_result["ece"] - 0.25) < 1e-7, "正確率或校準差不符"

    links = []
    for label, href in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", intro_bytes.decode()):
        filename, _, anchor = href.partition("#")
        target = (ROOT / "course/chapters" / filename).resolve()
        assert target.is_file(), "章首導覽目標不存在"
        text = target.read_text()
        anchor_found = not anchor or bool(re.search(r"^## " + re.escape(anchor) + r" ", text, re.M))
        assert anchor_found, "章首導覽錨點不存在"
        links.append({"label": label, "href": href, "target": target.relative_to(ROOT).as_posix(), "anchor_found": anchor_found})

    task_worlds = [
        {"task": "card", "application": "K17", "family_id": "review-card"},
        {"task": "address", "family_id": "review-address"},
        {"task": "app", "device": "手機", "family_id": "review-app"},
        {"task": "sum", "x": 1, "y": 2, "z": 3, "family_id": "review-sum"},
        {"task": "boxes", "a": 2, "b": 6, "events": [("swap", 0)], "query": "B", "family_id": "review-boxes"},
        {"task": "none", "family_id": "review-none"},
    ]
    required = {"card": ("application",), "address": (), "app": ("device",), "sum": ("x", "y", "z"), "boxes": ("a", "b", "events", "query")}
    count = 0
    samples = []
    for world in task_worlds:
        for row in c_rows(world, "train"):
            if row["question"] == "choice":
                expected = world["task"] if world["task"] in row["candidates"] else "none"
            else:
                fields = required[world["task"]]
                available = len(fields) - len(row["omitted"])
                expected = (
                    "yes" if available == len(fields) else "no"
                ) if row["question"] == "info" else (
                    "grade2" if available == len(fields) else "grade0" if available == 0 else "grade1"
                )
            assert row["gold"] == expected, "有限世界標籤與獨立欄位計數不符"
            count += 1
            if world["task"] == "card" or (world["task"] == "boxes" and row["omitted"] == ["a"]):
                samples.append({key: row[key] for key in ("task", "state", "question", "candidates", "gold", "omitted")})

    receipt_path = "docs/lesson-products/results/initial-c-execution.json"
    receipt = json.loads((ROOT / receipt_path).read_bytes())
    training = receipt["training"]
    assert receipt["status"] == "completed" and training["random_initialization"] is True, "訓練收據未確認隨機初值完成訓練"
    assert training["steps_this_attempt"] == 800 and training["weights_changed"] is True, "訓練收據未確認參數更新"
    assert training["initial_weights_sha256"] != training["selected_weights_sha256"], "收據的初值與成品指紋相同"
    code_paths = ["tiny_perceptron/lesson_products/" + name for name in ("data.py", "model.py", "training.py", "train.py")]
    code_receipts = []
    for path in code_paths:
        current_sha = sha((ROOT / path).read_bytes())
        assert current_sha == receipt["source_hashes"][path], "目前有限世界實作與訓練收據版本不同"
        code_receipts.append({"path": path, "sha256": current_sha, "matches_training_receipt": True})

    jev_body, jev_retrieval = curl("https://docs.typesafe.ai/introduction.md")
    jev_text = jev_body.decode()
    jev_excerpt = "Jev evaluates typed *questions* against a *state* and returns structured results directly."
    assert jev_excerpt in jev_text, "官方 Introduction 缺少待核對原句"
    assert all(term in jev_text for term in ("Choice", "Noul", "Score", "probability distributions")), "官方有型別結果的範圍不同"

    guo_html, guo_retrieval = curl("https://proceedings.mlr.press/v70/guo17a.html")
    parser = TextOnly()
    parser.feed(guo_html.decode())
    guo_text = " ".join(" ".join(parser.parts).split())
    guo_excerpt = "Confidence calibration – the problem of predicting probability estimates representative of the true correctness likelihood"
    assert guo_excerpt in guo_text, "作者摘要缺少待核對定義"
    _, guo_pdf_retrieval = curl("https://proceedings.mlr.press/v70/guo17a/guo17a.pdf")
    source_receipts = {
        "accessed_at": timestamp, "environment": environment,
        "retention": "第三方全文僅在記憶體讀取，不寫入 repo；只保存短摘錄、定位、回應指紋及審閱者支持範圍。",
        "sources": [
            {"name": "TypeSafe Introduction", "retrieval": jev_retrieval,
             "locator": "Introduction：Jev evaluates typed questions 段落；TypeSafe primitives 表格",
             "excerpt": jev_excerpt,
             "inspection": "官方說明 Jev 對 state 的有型別問題直接回傳結構化結果，並列 Choice、Score、Noul。只用來確認這是 TypeSafe 的外部判斷介面，不採用行銷效益或內部架構主張。"},
            {"name": "On Calibration of Modern Neural Networks", "retrieval": guo_retrieval,
             "pdf_retrieval": guo_pdf_retrieval,
             "locator": "PMLR 70:1321–1330 (2017)，作者摘要；PDF Definitions 的 perfect calibration 定義與其一百題示例",
             "excerpt": guo_excerpt,
             "inspection": "本回合透過官方 PDF 文字檢視實讀 Definitions：給定報出的信心 p，預測正確的條件機率應為 p；有限樣本用分組正確率近似。curl 另確認官方摘要及 PDF 的實際回應指紋。這支持分類信心校準，不支持每一題或整段生成文字必然正確。"},
        ],
    }
    save("original-sources.json", source_receipts)

    file_paths = [
        "course/chapters/09.md", "course/chapters/19.md", "course/glossary.md",
        "tiny_perceptron/alignment.py", receipt_path,
    ] + code_paths
    for page in [intro] + [pages[value] for value in selectors.values()] + [pages["19.16"]]:
        file_paths.append(page["snapshot"])
    result = {
        "command": command, "executed_at": timestamp, "environment": environment,
        "repository_head": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "freeze": {"path": freeze_path, "sha256": sha(freeze_bytes)},
        "inputs": [{"path": path, "sha256": sha((ROOT / path).read_bytes())} for path in dict.fromkeys(file_paths)],
        "introduction": {"matches_frozen_source": True, "has_figures": False, "links": links},
        "short_examples": runs,
        "three_candidates": candidates,
        "four_confidences": four_result,
        "finite_worlds": {"checked_rows": count, "checked_worlds": len(task_worlds), "all_labels_match_independent_rule": True, "samples": samples},
        "training_receipt": {
            "path": receipt_path, "sha256": sha((ROOT / receipt_path).read_bytes()),
            "code_revision": receipt["code_revision"],
            "status": receipt["status"],
            "training": {key: training[key] for key in ("random_initialization", "steps_this_attempt", "selected_step", "weights_changed", "seed", "initial_weights_sha256", "selected_weights_sha256", "environment")},
            "matching_code": code_receipts,
        },
        "result": "所有凍結指紋、導覽目標、短算例與有限世界標籤核對通過；既有訓練收據確認參數更新。",
        "scope": "只支持章首導覽及其明述的有限主張。不重跑訓練、不呼叫商業 API、不驗證全章所有實測成績或一般化能力。",
    }
    save("verification.json", result)
    print(json.dumps({"result": result["result"], "artifact_files": ["original-sources.json", "verification.json"], "checked_finite_rows": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
