"""核對入口頁的來源差異、舊證據完整性，並重跑原有短程式。"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-course"
REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_course/technical-course-original-01.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


previous = load(REPORT)
prior_output = load(BASE / "audit_course-output.json")
prior_site = load(BASE / "audit_site-output.json")
differences = load(BASE / "source-diffs-current-01.json")
notes = {
    "course/chapters/01.md": "係數與梯度改成四倍的措辭，不改字元編號、猜字、代價與更新的內容；字元還原原始程式保持相同。",
    "course/chapters/02.md": "感知器用語改為常見用法；查表、拼接、特徵與下一字預測內容保持相同。",
    "course/chapters/03.md": "有限樣本可能碰巧符合理論值；注意力如何比較並混合資訊的主題不變。",
    "course/chapters/05.md": "收窄字元示例範圍、hashlib 輸入及凍結參數說明；入口頁只依它指向訓練、檢查點與留出評估，均保留。",
    "course/chapters/06.md": "補入 surrogate、空白切分及固定 token 序列的條件；字元、byte、BPE 取捨與切詞入口保留。",
    "course/chapters/07.md": "分清模型與代價對齊、PAD 位置有效範圍及 PPO 目的；對話、長度、MTP、思考投入與後訓練的導覽仍對應原小節。",
    "course/chapters/08.md": "補充測試在更新前後評估與 LoRA 梯度的條件；書店告示、內容／格式／範圍要求的路線不變。",
    "course/chapters/09.md": "修正 round、成本與集中度說法並更新來源讀取日期；候選語義、資訊充分性、校準及決策主題保留。",
    "course/chapters/0A.md": "來源適用性除日期外也看店家及生效時間；外部線索與長文導覽、線索位置範圍不變。",
    "course/chapters/0C.md": "把數字使用規則與 greedy 的退化分布說清楚；多步、抽樣、驗證與訓練回饋主題保留。",
    "course/chapters/10.md": "normalize 不接受整數的措辭收窄；像素到特徵及圖文配對主題保留。",
    "course/chapters/11.md": "明確說明格內總和及欄索引；圖片使用、位置與筆畫損失的主題保留。",
    "course/chapters/12.md": "補入容忍差、整數週期、平均特徵及 pooling 邊界的條件；聲音表示、語音入口與共享對話仍存在。",
    "course/chapters/13.md": "明確說明先裁切 ratio 再乘優勢；獎勵、PPO 與 DPO 對回答傾向的主題保留。",
    "course/chapters/14.md": "收窄函式庫對應範圍，區分元素相等與逐位元比較；位置方法、循環深度及非單調評估路線保留。",
    "course/chapters/15.md": "路由統計排除沒有路由的 Dense 對照；共享計算與專家選擇的對照主題保留。",
    "course/chapters/16.md": "單層 cache 的實際改變位置及取位數措辭更明確；計算重用、效率與草稿驗證的導覽保留。",
    "course/chapters/17.md": "補充 Tensor.to 在型別與裝置相同時可沿用原物件；較少位元與誤差檢查主題保留。",
    "course/chapters/18.md": "收窄 activation、蒸餾溫度與梯度分量說法；教師訊號與較小學生的章目主題保留。",
    "course/chapters/19.md": "明示解題器輸出是預測且加法未達標，收窄 OCR、state_dict、numeric weighting 與計時紀錄；三成品隨機來源、角色、Auto 流程及失敗界線均與原始程式和結果一致。",
    "course/chapters/20.md": "補充環境工具、AI 評分規準、稽核摘要、模板結尾與提示範圍；沿用 Qwen／Whisper 並分開驗收的頁面主張不變。",
    "course/glossary.md": "Unicode 與 PPO 定義增加限制條件；仍是術語回查入口，頁面沒有轉述這些新句子。",
    "course/training.md": "區分工具的裝置選項及預期答案的限制；仍是局部實驗與訓練的操作入口，頁面沒有承諾各命令自動選裝置或一定答對。",
}

integrity = []
for item in [dict(path=str(REPORT.relative_to(ROOT)), sha256="6f91a12a0f6b3a7e783e40abb0bb5883b59d476962336828bed42c685d232012"), *previous["artifacts"]]:
    current = sha(ROOT / item["path"])
    integrity.append({"path": item["path"], "expected_sha256": item["sha256"], "observed_sha256": current, "unchanged": current == item["sha256"]})

source_states = []
for source in previous["sources"]:
    if source["kind"] == "repository_code":
        current = sha(ROOT / source["path"])
        source_states.append({"id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": current, "changed": current != source["sha256"]})

reviewed = []
for row in differences["sources"]:
    actual = subprocess.check_output(row["command"], cwd=ROOT, text=True)
    assert actual == row["diff"]
    source_id = next(x["id"] for x in source_states if x["path"] == row["path"])
    affected = [c["id"] for c in previous["claims"] if any(e["source_id"] == source_id for e in c["evidence"])]
    reviewed.append({"path": row["path"], "source_id": source_id, "current_sha256": sha(ROOT / row["path"]), "diff_sha256": hashlib.sha256(actual.encode()).hexdigest(), "base_matches_original": row["base_matches_original"], "target_matches_current": row["target_matches_current"], "affected_claim_ids": affected, "status": "verified", "inspection_note": notes[row["path"]]})

command = [sys.executable, str(BASE / "audit_course.py")]
process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
current_audit = json.loads(process.stdout)

raw_receipts = []
for key in ("product_training", "moe_training"):
    for row in prior_output[key]:
        observed = sha(ROOT / row["path"])
        raw_receipts.append({"path": row["path"], "expected_sha256": row["sha256"], "observed_sha256": observed, "unchanged": row["sha256"] == observed})
for key in ("product_final", "moe_final"):
    row = prior_output[key]
    observed = sha(ROOT / row["path"])
    raw_receipts.append({"path": row["path"], "expected_sha256": row["sha256"], "observed_sha256": observed, "unchanged": row["sha256"] == observed})

site_url = "https://birdhackor.github.io/tiny-perceptron-vlm/build-info.json"
site_command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "35", site_url]
site_process = subprocess.run(site_command, text=True, capture_output=True, check=True)
site_info = json.loads(site_process.stdout)
site_check = {"url": site_url, "command": site_command, "returncode": site_process.returncode,
              "observed": site_info, "matches_preserved_site_receipt": site_info == prior_site["build_info_after"],
              "response_sha256": hashlib.sha256(site_process.stdout.encode()).hexdigest()}

old_chapters = {x["path"]: x for x in prior_output["chapter_contents"]}
headings = [{"path": x["path"], "title_unchanged": x["title"] == old_chapters[x["path"]]["title"],
             "headings_unchanged": x["headings"] == old_chapters[x["path"]]["headings"]} for x in current_audit["chapter_contents"]]
freeze = ROOT / "docs/course-review-20261019/freeze-06.json"
freeze_record = load(freeze)
page = next(x for x in freeze_record["pages"] if x["page_id"] == "course")

current_audit["recheck"] = {
    "executed_at": datetime.now(timezone.utc).isoformat(),
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-course/recheck_course-current-01.py",
    "child_audit_command": command, "child_audit_returncode": process.returncode,
    "previous_issues": previous["issues"], "immutable_evidence": integrity,
    "repository_sources": source_states, "source_changes": reviewed,
    "raw_training_and_test_receipts": raw_receipts, "site": site_check,
    "chapter_headings": headings,
    "source_diffs": {"path": str((BASE / "source-diffs-current-01.json").relative_to(ROOT)), "sha256": sha(BASE / "source-diffs-current-01.json")},
    "freeze": {"path": str(freeze.relative_to(ROOT)), "sha256": sha(freeze), "page": page},
    "page_text_unchanged": (ROOT / "course/README.md").read_bytes() == (ROOT / page["snapshot"]).read_bytes(),
    "scope": "逐項判斷來源變動對入口頁全部主張的影響；不是替各來源章節新增的每一句話另作完整技術審閱。原始程式重跑仍不包含長時間訓練或雲端寫入。",
}
print(json.dumps(current_audit, ensure_ascii=False, indent=2))
