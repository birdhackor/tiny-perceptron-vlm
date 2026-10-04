"""Reviewer-authored introduction follow-up, locked to the text personally read."""

import hashlib
import json
import platform
import re
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments.compression import _style_scores
from tiny_perceptron.data import ByteTokenizer

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_18_01_"
TASK = "/root/integration_technical_coordinator/fact_v2_18_01"
BODY_SHA = "b1a20f5fbee7a83cd3c95b7cd52777f24d3fcb1075b17cf4c432f7cae6434d87"
INTRO = """# 第 18 章：蒸餾，讓小模型向教師學習

較小模型的容量有限，我們希望它除了學標準答案，也能從教師取得額外指導；教師是否較有能力，先用同一任務檢查。本章先分清[教師能提供的訊號](18.md#18.1)，再選真正較小的學生，逐步追蹤軟目標、溫度、詞表對齊與KL誤差。後半段檢查教師風格與錯誤、不同架構與模態的對齊，以及蒸餾後再量化的成本。小程式用固定數字或隨機結構核對機制；完整實跑則保留沒有改善的屬性題、數字不正確的風格答案與仍然重複的故事續寫，讓「成功接上教師」和「學生變好」各有證據。

"""
SUMMARY = (
    "本人本次完整親讀第18章標題及首個##之前全部導讀。它以小學生容量有限和教師需先在同一任務接受檢查為動機，"
    "連到18.1的文字/分布訊號區分，再列出小學生、soft targets、溫度、詞表/KL及後半章錯誤/架構/模態/量化的學習順序。"
    "這些章節路線屬導覽，沒有把未親讀的後半節內容當成已重新審查。其方法動機與先前真正親讀的KD §1–2、MiniLLM v6 §1、"
    "18.1及必要前置4.6/7.11/T.4/T.5/T.8/18.6相接；人工訊號與完整訓練品質分開。"
    "本次另直接讀原distillation實報的學生逐題結果：w16 CE/CE+KL各1/10，w32各4/10，"
    "style CE+KL把2+2答成11/10等，MoE教師與Dense學生確有重複the的續寫，支持導讀保留失敗而不保證蒸餾改善。"
    "18.1完整正文SHA與先前親讀snapshot完全一致，六份必要前置也逐節一致；這次核對版本，沒有宣稱重新閱讀那些正文。"
    "沒有重跑模型、GPU或完整訓練。"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    chapter = ROOT / "course/chapters/18.md"
    text = chapter.read_text(encoding="utf-8")
    heading = re.search(r"^## .+$", text, re.M)
    assert text[:heading.start()] == INTRO, "導讀變更後必須本人重新閱讀，不能只刷新hash"
    body = dict(sections(chapter))["18.1"]
    assert hashlib.sha256(body.encode()).hexdigest() == BODY_SHA
    assert body == (OUT / (PREFIX + "section.md")).read_text()
    prerequisite_receipts = []
    for row in json.loads((OUT / (PREFIX + "prerequisites.json")).read_text()):
        source, lesson = row["source"].split("#")
        assert dict(sections(ROOT / source))[lesson] == row["body"]
        prerequisite_receipts.append({"source": row["source"], "sha256": row["sha256"], "unchanged": True})
    report_path = ROOT / "docs/technical-reviews/18.1.json"
    prior = report_path.read_bytes()
    prior_hash = hashlib.sha256(prior).hexdigest()
    history = OUT / (PREFIX + "report_before_intro_" + prior_hash[:16] + ".json")
    if history.exists():
        assert history.read_bytes() == prior
    else:
        history.write_bytes(prior)
    report = json.loads(prior)
    assert report["reviewer_task"] == TASK
    assert "intro_sha256" not in report, "已補導讀的report不得由本初次補審程式再次覆寫"
    snapshot = OUT / (PREFIX + "intro_read.md")
    snapshot.write_text(INTRO, encoding="utf-8")
    original_path = ROOT / "docs/course-experiments/results/distillation.json"
    raw_path = ROOT / "outputs/course-control/37060809775/result.json"
    original = json.loads(original_path.read_text())
    raw = json.loads(raw_path.read_text())
    assert original["results"] == raw["results"]
    tasks = original["results"]["tasks"]
    tok = ByteTokenizer()
    inspected = {}
    for key, task in tasks.items():
        inspected[key] = {}
        for name, branch in task["runs"].items():
            evaluation = branch["test"]
            samples = evaluation["generated_samples"]
            assert len(samples) == evaluation["examples"]
            correct = 0
            for sample in samples:
                ids = sample["generated_ids"]
                raw_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
                exact = raw_ids == tok.encode(sample["expected"])
                assert exact == sample["exact"]
                assert tok.decode(raw_ids) == sample["generated"]
                correct += exact
            assert correct == evaluation["correct"]
            inspected[key][name] = {"correct": correct, "records": len(samples),
                                    "supervised_tokens": evaluation["supervised_tokens"],
                                    "max_new_tokens": evaluation["max_new_tokens"],
                                    "samples": samples}
    attribute = tasks["attributes"]["runs"]
    for width, expected in ((16, 1), (32, 4)):
        ce = attribute[f"w{width}_ce"]
        kl = attribute[f"w{width}_ce_kl"]
        assert ce["test"]["correct"] == kl["test"]["correct"] == expected
        for field in ("initialization_sha256", "batch_plan_sha256", "steps", "effective_supervised_tokens"):
            assert ce["training"][field] == kl["training"][field]
    style_data = json.loads((OUT / (PREFIX + "style_dataset.json")).read_text())
    style = tasks["style_transfer"]["runs"]["w32_ce_kl"]
    scored = _style_scores(style["test"], style_data["test"])
    assert scored["content_correct"] == 3 and scored["examples"] == 27
    assert style["test"]["generated_samples"][0]["question"] == "style=concise; 2+2=?"
    assert style["test"]["generated_samples"][0]["generated"] == "11"
    story = tasks["moe_to_dense"]["runs"]["w32_ce_kl"]["test"]
    assert "the the the" in story["generated_samples"][0]["generated"]
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu",
                   "scope": "JSON/source inspection and pure rubric execution; no model forward, training or GPU run"}
    receipt_path = OUT / (PREFIX + "intro_review.json")
    receipt = {"reviewer_task": TASK, "read_on": "2026-10-04", "intro_sha256": sha(snapshot),
               "intro_summary": SUMMARY, "source_body_sha256": BODY_SHA, "body_unchanged": True,
               "prerequisite_receipts": prerequisite_receipts,
               "prior_report_path": str(history.relative_to(ROOT)), "prior_report_sha256": prior_hash,
               "original_experiment_sha256": sha(original_path), "raw_experiment_sha256": sha(raw_path),
               "historical_environment": {key: original[key] for key in
                                           ("revision", "seed", "device", "torch_version", "python_version", "gpu",
                                            "elapsed_seconds", "timing_scope", "step_scale")},
               "inspected_student_tests": inspected, "style_content_rubric": scored["rubric"],
               "roadmap_link": "18.md#18.1 resolves to the unchanged section personally reviewed earlier",
               "reviewer_decision": "pass: introduction coheres with inspected method prerequisites and preserved empirical failures",
               "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_18_01_intro_review.py",
               "result": "all assertions passed; source/result inspection only", "environment": environment}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    report["intro_sha256"] = sha(snapshot)
    report["intro_summary"] = SUMMARY
    for identifier, path, kind, description in [
        ("intro_read", snapshot, "source_snapshot", "本審閱者本次完整親讀的首##之前導讀，文字已顯式鎖定，非只抓最新hash。"),
        ("intro_execution", receipt_path, "execution", "原實報完整學生test原ID重數、內容rubric與導讀連接核對；無模型重跑。"),
        ("intro_code", Path(__file__).resolve(), "code", "本人導讀补審可重跑程式，導讀文字變更會拒絕hash刷新。"),
        ("intro_prior", history, "source_snapshot", "導讀補審前本人18.1報告原文及版本，完整保留。"),
    ]:
        entry = {"id": identifier, "path": str(path.relative_to(ROOT)), "kind": kind,
                 "sha256": sha(path), "description": description}
        if kind == "execution":
            entry.update(command=receipt["command"], result=receipt["result"], environment=environment)
        report["artifacts"].append(entry)
    report["sources"].append({"id": "intro_runtime", "kind": "execution", "title": "本人導讀及原學生全test結果核對",
                              "verified": True, "artifact_id": "intro_execution"})
    report["claims"][0]["location"] += "；章導讀開頭教師額外指導的動機"
    report["claims"][-1]["location"] += "；章導讀最後區分接上教師與學生變好"
    additions = [
        ("c15", "導讀所說未改善的屬性實跑有真實留出結果保留。",
         "w16 CE/CE+KL均1/10，w32均4/10；matched初始化/抽樣計画/400更新及44,985有效訓練目標。",
         {"test_records_per_branch": 10, "test_valid_tokens_including_eos": 69, "seed": 42,
          "student_steps_per_branch": 400, "student_training_valid_tokens": 44985},
         "只支持固定seed和這兩個寬度的完整test匹配總數沒有提升；逐題有改善與退步，不是所有蒸餾都不改善。"),
        ("c16", "導讀所說風格答案的數字錯誤有真實生成保留。",
         "全部27题原ID与exact重算一致；CE+KL內容3/27；2+2短答11、生動答案前綴10、JSON answer10，皆非4。",
         {"test_records": 27, "test_valid_tokens_including_eos": 546, "seed": 42,
          "student_steps": 300, "max_new_tokens": 96},
         "只證明失敗確有保存。不是以格式/比喻成功替代算術正確；完整27題rubric重新以原資料執行。"),
        ("c17", "導讀所說故事續寫重複片語有真實生成保留。",
         "MoE教師首題tthe the the the the the，w32 CE+KL學生首題the t the the the the th；完整各52题sample仍在原報告。",
         {"test_documents": 52, "test_valid_tokens_including_eos": 41914, "seed": 42, "max_new_tokens": 24,
          "student_steps": 300},
         "52篇/41,914有效目標是原報告完整loss分母；每題只有最多24單位greedy續寫，此處只查重複樣本存在，不用少數樣本估整體故事品質。"),
    ]
    for identifier, statement, observed, denominators, scope in additions:
        report["claims"].append({"id": identifier, "kind": "empirical", "statement": statement,
                                 "location": "course/chapters/18.md首##之前導讀最後一句", "status": "verified",
                                 "evidence": [{"source_id": "intro_runtime", "locator": "inspected_student_tests与style_content_rubric",
                                               "supports": "本人直接讀原報告、逐raw ID重數完整test，保留固定配置及逐題生成。"}],
                                 "artifact_ids": ["intro_execution", "intro_code", "intro_read"], "scope": scope,
                                 "verification": {"method": "executed", "expected": "原實跑保存導讀描述的失敗、完整分母與配置",
                                                  "observed": observed, "details": "解析原實報與原raw結果core相等；不重跑模型；未把檢查程式時間當歷史訓練時間。",
                                                  "denominators": denominators}})
    for name in ("factual_accuracy", "numeric_verification", "source_verification", "limitations"):
        report["checks"][name]["claim_ids"].extend(["c15", "c16", "c17"])
        report["checks"][name]["details"] += " 本人另完整親讀並保存章導讀；原學生結果核對見intro_execution，未重新推論或訓練。"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"verdict": report["verdict"], "intro_sha256": report["intro_sha256"],
                      "prior_report_sha256": prior_hash, "claims": len(report["claims"]),
                      "body_unchanged": True, "prerequisites_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
