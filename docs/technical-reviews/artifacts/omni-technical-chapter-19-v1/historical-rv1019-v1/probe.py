"""本人首判後的兩identity歷史callback；只讀來源與CPU推論，不訓練。"""

from pathlib import Path
from datetime import datetime, UTC
from collections import defaultdict
import gzip
import hashlib
import json
import platform
import re

import torch

from tiny_perceptron.lesson_products.data import arithmetic_worlds, answer, digest, render, split_of
from tiny_perceptron.lesson_products.inference import Product, check_manifest

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).with_name("execution.json")
RAW = Path("/workspace/work/reference/omni-technical-19-17-v1/raw/final-report-v1.predictions.jsonl.gz")
TASK = "/root/omni_technical_coordinator/technical_chapter_19"
REQUEST = "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"
OWN = "docs/course-revision-20261029-omni/reviews/technical/technical_chapter_19/technical-original-v1.json"
CURRENT_FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
ORIGINAL_FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
QUOTE = "**Looped 解題器**讀固定語法的事件或加法題，輸出答案預測（事件的狀態追蹤已通過預先訂的門檻，加法還沒達標）"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_bytes())


def reference(path):
    return {"path": path, "sha256": sha(ROOT / path)}


def node(o, pointer):
    for key in pointer.strip("/").split("/") if pointer else []:
        o = o[int(key)] if isinstance(o, list) else o[key]
    return o


def verified(ref):
    assert sha(ROOT / ref["path"]) == ref["sha256"]
    return load(ref["path"])


started = datetime.now(UTC).isoformat()
own = load(OWN)
assert sha(ROOT / OWN) == "1778278b379fa457f6b7eacacf87f15af0afa63a4b63f17502b23309f0a5d905"
assert own["reviewer_task"] == TASK and not own.get("historical_issue_rechecks")
freeze = load(CURRENT_FREEZE)
page = next(p for p in freeze["pages"] if p["page_id"] == "chapter-19")
old_page = next(p for p in load(ORIGINAL_FREEZE)["pages"] if p["page_id"] == "chapter-19")
snapshot = (ROOT / page["snapshot"]).read_bytes()
chapter = (ROOT / page["source"]).read_bytes()
body = chapter[: re.search(rb"^## ", chapter, re.M).start()]
assert body == snapshot and sha(ROOT / page["snapshot"]) == page["source_sha256"] == own["source_sha256"]
assert old_page["source_sha256"] == page["source_sha256"] and page["figures_sha256"] == {}
assert QUOTE in snapshot.decode() and "算出答案" not in snapshot.decode()
source_pins = [{"path": s["path"], "sha256": sha(ROOT / s["path"]), "matches_original": sha(ROOT / s["path"]) == s["sha256"]} for s in own["sources"] if s["kind"] == "repository_code"]
assert all(s["matches_original"] for s in source_pins)

group = load(REQUEST)["groups"][5]
assert group["old_owner"] == "/root/rv1019_technical_chapter-19" and len(group["items"]) == 2
assert {i["issue_id"] for i in group["items"]} == {"TECH-chapter-19-c6-current-01", "TECH-chapter-19-c6"}
identities = []
for item in group["items"]:
    old_report = verified(item["original_report"])
    original_issue = node(old_report, item["json_pointer"])
    disposition = node(verified(item["disposition_ref"]), item["disposition_ref"]["json_pointer"])
    assert disposition["issue_id"] == item["issue_id"] and disposition["action"] == "fixed"
    assert disposition["final_necessity"] == item["necessity"] == "necessary" and original_issue["necessary"] is True
    assert item["historical_fix_verification"] in disposition["verification"]
    identities.append({"issue_id": item["issue_id"], "original_report": item["original_report"], "json_pointer": item["json_pointer"], "necessity": item["necessity"], "original_issue_sha256": hashlib.sha256(json.dumps(original_issue, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "original_status": original_issue["status"]})
unavailability = node(verified(group["unavailability"]["report"]), group["unavailability"]["json_pointer"])
assert unavailability["reviewer_task"] == group["old_owner"] and unavailability["unavailable"] is True

# 以固定資料世界重建全部test加法題；沒有挑題或重選split。
manifest = load("docs/lesson-products/data-v2-manifest.json")
settings = load("docs/lesson-products/results/settings-freeze-v1.json")
historical = load("docs/lesson-products/results/final-report-v1.json")
assert sha(ROOT / "docs/lesson-products/data-v2-manifest.json") == settings["data_manifest_sha256"] == historical["data_manifest_sha256"]
salt = manifest["config"]["split_salt"]
worlds = [w for w in arithmetic_worlds() if split_of(w["family_id"], salt) == "test"]
assert len(worlds) == 42 and len({w["family_id"] for w in worlds}) == 10
directory = ROOT / "outputs/lesson-products/b"
public_manifest = check_manifest(directory, "b")
release = load("tiny_perceptron/lesson_products/release.json")
assert sha(directory / "artifact-manifest.json") == release["products"]["b"]["manifest_sha256"]
assert public_manifest["settings_freeze_sha256"] == sha(ROOT / "docs/lesson-products/results/settings-freeze-v1.json")
for name, expected in settings["products"]["b"].items():
    assert sha(directory / name) == expected
torch.set_num_threads(1)
model = Product(directory, "b", "cpu")
rows = []
totals = {str(d): 0 for d in [1, 3, 5]}
for world in worlds:
    state, gold = render(world), answer(world)
    predictions = {str(d): model.solve(state, loops=d)["answer"] for d in [1, 3, 5]}
    for depth, predicted in predictions.items():
        totals[depth] += predicted == gold
    rows.append({"id": digest((world["family_id"], state)), "family_id": world["family_id"], "state": state, "gold": gold, "predictions": predictions})

# 另讀指紋吻合的原始逐題logits，核目前重跑與原始逐題答案及門檻。
raw_digest = hashlib.sha256()
raw_sums = {}
boxes = {"count": 0, "correct": 0}
with gzip.open(RAW, "rb") as stream:
    for line in stream:
        raw_digest.update(line)
        row = json.loads(line)
        if row.get("product") != "b" or row.get("split") != "test":
            continue
        predicted = {d: max(range(len(row["scores_by_depth"][d])), key=row["scores_by_depth"][d].__getitem__) for d in ["1", "3", "5"]}
        assert all(predicted[d] == row["answers_by_depth"][d] for d in predicted)
        if row["task"] == "sum":
            raw_sums[row["id"]] = {"gold": row["gold"], "predictions": predicted}
        else:
            boxes["count"] += 1
            boxes["correct"] += predicted["5"] == row["gold"]
assert raw_digest.hexdigest() == historical["predictions"]["sha256"]
assert set(raw_sums) == {r["id"] for r in rows}
assert all(raw_sums[r["id"]] == {"gold": r["gold"], "predictions": r["predictions"]} for r in rows)
assert totals == {"1": 0, "3": 1, "5": 2}
assert boxes == {"count": 2956, "correct": 2928}
assert historical["gates"]["b_primary"] is True and historical["gates"]["b_secondary_target"] is False
assert boxes["correct"] / boxes["count"] >= .90 and totals["5"] / 42 < .85

result = {
    "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK,
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-19-v1/historical-rv1019-v1/probe.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "full_review": reference(OWN), "original_source_freeze": reference(ORIGINAL_FREEZE), "current_source_freeze": reference(CURRENT_FREEZE),
    "source": {"snapshot": page["snapshot"], "sha256": page["source_sha256"], "bytes": len(snapshot), "figure_sha256": {}, "source_quote": QUOTE, "complete_intro_reread": True, "source_pins_unchanged": source_pins},
    "request": {**reference(REQUEST), "json_pointer": "/groups/5"}, "distinct_identities": identities,
    "raw_source": {"path": str(RAW), "gzip_sha256": sha(RAW), "uncompressed_sha256": raw_digest.hexdigest()},
    "public_weights": {"release": reference("tiny_perceptron/lesson_products/release.json"), "manifest_sha256": sha(directory / "artifact-manifest.json"), "model_sha256": sha(directory / "model.pt"), "matches_pretest_freeze": True},
    "arithmetic_reexecution": {"split": "test", "salt": salt, "samples": 42, "families": 10, "seed_from_training_receipt": json.loads((directory / "training-receipt.json").read_bytes())["seed"], "correct_by_depth": totals, "matches_all_original_raw_predictions": True, "rows": rows},
    "historical_tracking_raw_recount": {**boxes, "loops": 5, "accuracy": boxes["correct"] / boxes["count"], "preregistered_threshold": .90, "passes": True},
    "arithmetic_secondary_threshold": {"threshold": .85, "correct": totals["5"], "count": 42, "accuracy": totals["5"] / 42, "passes": False},
    "conclusion": "兩個identity分別仍修妥；神經answer是argmax預測，舊追蹤主門檻通過、加法未達標與目前文字一致。",
    "limitations": "本次重跑全部42道舊有限加法CPU推論，事件分數由歷史原始logits重算；沒有重新訓練、重新推論2956道事件或檢測新擴充能力。無stage seal，這不是新主表承接或全gate結果。",
}
with OUTPUT.open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "sha256": sha(OUTPUT), "identities": [i["issue_id"] for i in identities], "arithmetic": totals, "tracking": boxes, "all_predictions_match_historical": True}, ensure_ascii=False, indent=2))
