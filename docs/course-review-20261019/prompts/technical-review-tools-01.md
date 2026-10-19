# 發布閘門工具的審閱（正式審閱）

你是這一輪發布閘門工具的技術審閱者。你的任務名是 `/root/rv1019_technical_review-tools`，寫進報告的 `reviewer_task`。

這是正式審閱：

- 你寫的檔會以 SHA-256 綁住，之後不能改寫；修正後的複查另寫新檔。
- 你沒有參與撰寫這些工具。你要判斷的是：工具照它的註解與設計做事、沒有讓不該通過的紀錄通過、測試真的守著每一條規則。

最終回覆用正體中文。

## 背景

發布閘門（`scripts/check_course_reviews.py`、`scripts/check_technical_reviews.py`、`docs/review-tools/check_review_round.py`）經 `docs/review-tools/incremental_revision.py` 的 `audit_manifest`，先用 `docs/review-tools/baseline_replay.py` 的 `replay` 在暫存硬連結副本重放 2026-10-08 那一輪。commit `cc6afc5d` 讓新一輪的 manifest 用 `baseline_supplements` 帶一份補充還原清單：每一筆把一個舊輪次以 SHA-256 記下、之後又改過的檔，在重放時換回 repo 內保存、SHA 等於舊紀錄的副本。

- 設計：`outputs/course-inline-code-20261017/pub/design/replay-supplements.md`（D-RP1～D-RP5）。
- 實作者的核對腳本與紀錄在 `outputs/course-inline-code-20261017/pub/replay-impl/`，可以參考；但下面的真實重放對照要用你自己寫的程式重做。

## 審什麼

閘門的工具審閱要求下列檔的現行位元組都在你的 `checked_files` 裡：

| 檔 | 和上一輪工具審閱的關係 |
|---|---|
| `docs/review-tools/incremental_revision.py` | `cc6afc5d` 改過，上一輪審的是它的父版本 |
| `docs/review-tools/baseline_replay.py` | 同上 |
| `tests/test_incremental_revision_review.py` | 同上 |
| `docs/course-review-20261019/baseline-supplements.json` | `cc6afc5d` 新增 |
| `docs/review-tools/phase7_review.py` | 上一輪審過，之後沒改 |
| `docs/course-revision-20261016/tools/reader.py` | 同上 |
| `docs/course-revision-20261016/tools/freeze_sources.py` | 同上 |

1. **改過的三個檔與新增的清單**：讀 `cc6afc5d~1..cc6afc5d` 的完整 diff 與這幾個檔的現行全文，至少判斷：
   - 有沒有任何輸入能讓重放寫到暫存樹以外、讀到 repo 以外，或改到硬連結的原檔；
   - 補充能不能把舊紀錄沒記的位元組塞進重放，讓舊審閱照樣通過；
   - 沒帶補充時行為是否與改前相同；
   - `check_tool_audit` 的補充清單綁定是否足夠；
   - 測試是否守著每一條新規則，能力不足時是否只跳過那一項。
2. **沒改的三個檔**：核對現行 SHA-256 與上一輪工具審閱 `docs/course-revision-20261016/reviews/tool-audit-callback-04.json` 的 `checked_files` 記的相同（只讀那個欄位，不讀它的問題與結論），在 `claims` 記下。
3. **真實重放對照**（設計 D-RP4 的驗收條件，不能省）：在乾淨 clone 用獨立 process，對現行版本做五種情況，每種保存指令、結束碼與輸出：
   1. 不帶補充 → 重放回報 `scripts/fetch_natural_data.py` 漂移；
   2. 只補 `scripts/fetch_natural_data.py` → 回報 `docs/selftrained/v2-manifest.json` 漂移；
   3. 兩筆都補 → 重放通過；
   4. 補充某一筆的 `sha256` 與舊表不符 → 以舊身分拒絕；
   5. 某一筆符合舊表，但隔離副本裡它指到的來源位元組被改壞 → 以來源 SHA 拒絕。
   另外核對重放前後，clone 裡全部追蹤檔的 SHA-256 不變。上面的預期結果是設計寫的；實際結果不同時照實記，並寫進 `issues`。
   - clone 放在 `outputs/course-review-20261019/tool-audit/`（被 gitignore）。用 `git clone --no-local` 從這個 repo 複製，設 `GIT_LFS_SKIP_SMUDGE=1`。不要在主工作樹裡改任何檔來做負例。
4. **跑測試**：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_incremental_revision_review.py`，記下結果。

## 能看什麼、能做什麼

- repo 裡的程式、測試、紀錄都可以讀，也可以執行；可以連網查官方文件。
- 完成並存下自己的原始判斷以前，不看其他人對這些工具的審閱報告與複審意見（例如 `outputs/course-inline-code-20261017/pub/reports/`、`pub/findings.tsv`、`docs/course-review-20261019/reviews/` 底下別人的目錄）。

## 寫什麼、寫在哪

- 只寫：
  - 你的報告目錄 `docs/course-review-20261019/reviews/rv1019_technical_review-tools/`；
  - 你的 artifact 目錄 `docs/technical-reviews/artifacts/review-20261019-review-tools/`；
  - 上面的 clone 目錄。
- 不修改其他任何檔；碰到寫入被拒就停下回報。不 commit、不 push、不用 git stash、不觸發任何 GitHub workflow。
- artifact 只放小檔：你寫的程式、指令輸出、SHA 清單。不要把 clone 或大檔放進 artifact 目錄。
- 報告：`docs/course-review-20261019/reviews/rv1019_technical_review-tools/tool-audit-original-01.json`：

```json
{
  "schema_version": 1,
  "report_kind": "incremental_tools_review",
  "review_stage": "technical_software",
  "reviewer_task": "/root/rv1019_technical_review-tools",
  "author_tasks": ["/root"],
  "initial_context": "fresh",
  "created_at": "<寫檔當下的 UTC 時間>",
  "verdict": "pass 或 revise",
  "scope": "<審了什麼、怎麼審>",
  "checked_files": [{"path": "<上表每個檔>", "sha256": "<現行 SHA-256>"}],
  "claims": [{"id": "<編號>", "status": "verified | unresolved | contradicted", "statement": "<主張>", "artifact_ids": ["<artifact id>"]}],
  "executions": [{"command": "<指令>", "exit_code": 0, "result": "<結果摘要>", "environment": {"python": "<版本>", "platform": "<平台>"}}],
  "artifacts": [{"id": "<id>", "path": "<artifact 目錄裡的檔>", "sha256": "<SHA-256>"}],
  "issues": [{"id": "<編號>", "file": "<檔:行>", "severity": "P1 | P2 | P3", "necessary": true, "details": "<哪裡錯、依據、失敗情境>", "suggestion": "<最小修正方向>"}],
  "limits": ["<沒能驗證的範圍>"]
}
```

- 全部規則都核實、五種對照都和預期一致，才寫 `verdict: pass`。有任何會讓不該通過的紀錄通過、或寫出暫存樹以外的問題，就寫 `revise`。

## 回覆

- verdict；`checked_files` 每個檔的 SHA-256。
- 五種對照各一行：預期、實際、artifact。
- issues 各一行；沒有就說沒有，並說明你怎麼確認。
- 報告與每個 artifact 的路徑與 SHA-256。
- 你沒能驗證的範圍。
