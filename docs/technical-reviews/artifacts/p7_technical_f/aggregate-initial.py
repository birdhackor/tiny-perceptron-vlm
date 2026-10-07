"""Assemble this owner's preserved judgments, without rewriting trace/proof history."""
import copy
import datetime
import hashlib
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[4]
F = R / "docs/technical-reviews/artifacts/p7_technical_f"
M = R / "docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json"
manifest = json.loads(M.read_text())
group = next(g for g in manifest["groups"] if g["group"] == "f")
T = "docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/f/6f2c72d53cc64e9d8396b432fb621e7d.jsonl"
rows = [json.loads(x) for x in (R / T).read_text().splitlines()]
assert rows[-1]["event"] == "complete"
notes = [r for r in rows if r["event"] == "checkpoint"]
prefer_v2 = {"20.1", "20.9", "20.10", "B.7", "C.1"}
source_sha_files = {
    "qwen": F / "qwen-pinned-README.md", "whisper": F / "whisper-pinned-README.md",
    "lora": R / "outputs/p7_technical_f/source-local-cache/lora-v2.pdf",
    "qwen-api": F / "qwen-pinned-api.json", "qwen-source": F / "qwen-modeling-4.57.6.py",
    "torch-autograd": F / "pytorch-autograd-2.8.rst", "split": F / "cross-validation.rst",
    "causal": F / "transformers-mask-4.57.6.py", "cer-original": F / "official-cer.py",
    "counter-original": F / "cpython-collections.py", "whisper-api": F / "whisper-pinned-api.json",
    "rfc3629": F / "rfc3629.txt",
}
summaries = {
    "readme": "本人逐段核對閱讀/安裝/訓練入口，287份Notebook、313小節與實際550詞表模型參數成立；新CPU前反向、匿名模型及有限推論已做，v2門檻仍未全部通過，歷史訓練/發布與本輪執行分開。",
    "environment": "本人核對互斥CPU/CUDA extras、uv鎖檔與真CPU前反向，以及九份原公開CPU操作指紋；Linux本機可用不代表Colab/Windows/GPU容量或ASR/助理品質，三意圖音訊特徵與Whisper文字入口分開。",
    "asset-storage": "Git正文、LFS實體與pointer、HF推論配套、本機ignored產物責任分清；實際v2資料包8962檔案與28876/2435/3734行已核，固定SHA只支持同一材料，不能替代能力測試。",
    "curriculum": "本人核對六個固定上游專案的實際相關原碼、DPO/KL/回答labels CPU小探針、25個課程metadata配對與原官方課程topic定位；LoRA/MoE/量化/蒸餾目的分開，未新跑上游訓練或借其品質成績。",
    "validation": "本人核對六種驗證範圍、3734固定test原數字、width16本機wrapper煙測與L4歷史40→80原欄；31584參數CPU親算。CLI/help與原receipt支持實查範圍，未重新執行全課/GPU/發布。",
    "training-assets": "本人核八個實體LFS包、174成员/1447訓練記錄/29.4254656MiB與原授權、真安全解包及CPU逐byte重建。原錄音62/15/30與訓練558含增强路徑、49近重複群組分清，失敗斷言/預寫紀錄及更正全部保留。",
    "publishing": "本人核common Markdown轉換、Colab固定Notebook與默认branch源码區別、真四fixture輸出拒絕及Pages YAML契約；4×4=16固定權重檔與21510B根卡分別核。無新外部發布；unit7截图提前露出unit8已披露，随后正常解鎖核原資料。",
}

pages, proof_refs, corrections = [], [], []
for pid in group["primary_page_ids"]:
    path = F / ("page-" + pid + ("-v2" if pid in prefer_v2 else "") + ".json")
    page = json.loads(path.read_text())
    proof_refs.append(dict(page_id=pid, path=str(path.relative_to(R)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    selected = [r for r in notes if r["page_id"] == pid]
    assert selected
    page["summary"] = summaries.get(pid, page.get("summary", ""))
    assert page["summary"]
    page["summary"] = page["summary"].replace("整頁兩視窗真看", "必要選取區域兩視窗真看")
    page["checkpoint_event_indices"] = [r["event_index"] for r in selected]
    page["checkpoint_unit_indices"] = [r["unit_index"] for r in selected]
    page["missing_visuals"] = page.get("missing_visuals") or selected[-1]["missing_visuals"]
    page["page_visual_check"] = page.get("page_visual_check") or dict(status="unverified", required=False,
        details="No required SVG in this frozen page; actual whole-page layout was not viewed. See preserved reading notes for the specific scope.")
    # Viewport is receipt metadata, not a new view. Both actual image hashes and original receipt remain untouched.
    for ref in page["page_visual_check"].get("artifacts", []):
        if isinstance(ref.get("viewport"), str):
            original = ref["viewport"]
            ref["viewport"] = [int(x) for x in original.split("x")]
            corrections.append(dict(page_id=pid, field="page_visual_check.artifacts.viewport", original=original, correction=ref["viewport"], scope="Metadata type only; path/hash/receipt and actual observation unchanged."))
    if not page["claims"]:
        assert pid == "chapter-0A"
        page["applicability"] = dict(substantive_claims=False, reason="This appendix introduction only states its route and distinguishes input-display exercises from actual model outputs; no external method, quantitative result or program fence is asserted.")
    for c in page["claims"]:
        if c["kind"] == "empirical" and not isinstance(c["verification"].get("denominators"), dict):
            assert isinstance(c.get("denominators"), dict) and c["denominators"]
            c["verification"]["denominators"] = c.pop("denominators")
            corrections.append(dict(page_id=pid, claim_id=c["id"], field="verification.denominators", scope="Moved owner's existing exact denominator dict from top-level into required schema; values unchanged."))
    for a in page["artifacts"]:
        if a.get("kind") == "raw_output":
            assert pid == "training" and a["id"] in ("float-validation", "int4-validation", "int8-validation")
            a["kind"] = "source_snapshot"
            corrections.append(dict(page_id=pid, artifact_id=a["id"], field="kind", original="raw_output", correction="source_snapshot", scope="Allowed metadata type for this owner's raw result snapshot; original file/hash/description and actual execution artifacts unchanged."))
    for s in page["sources"]:
        if s["kind"] in ("paper", "official_docs", "official_source") and not s.get("sha256"):
            encoded = re.search(r"[a-f0-9]{64}", s.get("version", ""))
            if encoded: s["sha256"] = encoded[0]
            elif s["id"] in source_sha_files:
                s["sha256"] = hashlib.sha256(source_sha_files[s["id"]].read_bytes()).hexdigest()
            else: raise ValueError("Missing original source hash " + pid + ":" + s["id"])
        # Full original papers/text stay cache; only URL/revision/hash/location/support survive formal report.
        for key in ("local_pdf", "local_text", "cache_path", "original_pdf_path", "original_text_path"):
            s.pop(key, None)
    pages.append(page)

timestamp = datetime.datetime.now(datetime.timezone.utc)
report = dict(schema_version=1, review_policy="phase7_grouped", batch_id=manifest["batch_id"],
    stage="technical", group="f", reviewer_task="/root/p7_technical_f", reviewer_context="fresh",
    author_tasks=["/root"], manifest_sha256=hashlib.sha256(M.read_bytes()).hexdigest(), verdict="revise",
    created_at=timestamp.isoformat(), trace_files=[dict(path=T, sha256=hashlib.sha256((R/T).read_bytes()).hexdigest())],
    context_page_ids=rows[0]["context_page_ids"], primary_page_count=len(pages), checkpoint_count=len(notes),
    primary_checkpoint_count=sum(r["page_id"] in group["primary_page_ids"] for r in notes),
    completion=dict(session=rows[0]["session"], original_start_at=rows[0]["recorded_at"], actual_complete_at=rows[-1]["recorded_at"],
        complete_event_index=rows[-1]["event_index"], restoration="Repeatedly resumed original session via current-only and saved owner notes; no old checkpoint was recreated."),
    summary="54 primary pages and assigned19.10–19.12 context completed with own progressive checkpoints. Initial revise: historical semantic grading scope, action/posture/state subset label, and ineffective Bash resume precheck remain open. No textbook changes or new external publication.",
    issues=[dict(page_id=p["page_id"], **i) for p in pages for i in p["issues"]],
    preserved_provisional_page_records=proof_refs, report_metadata_corrections=corrections, pages=pages,
    execution_scope=dict(new="Proportionate CPU examples, raw-data arithmetic, limited short learning/quantization, anonymous fixed-file checks, local archive unpack/rebuild, AST/parser/help and necessary actual image views as individually recorded.",
        historical="GPU/Modal training, public test generations, UI real-model runs and release history are original stored raw evidence; rereading/aggregating them is not a new execution of those systems.",
        not_executed=["New GPU or long fixed training", "New Modal remote training", "New HF/model or website publication", "All287 notebook kernels or full test suite", "Colab actual runtime", "Windows/CUDA/MPS target machines", "Full-page every-pixel inspection", "Fresh semantic regrade of all42 final photos"]),
    independence_and_provenance=dict(origin="True originally dispatched fork_turns=none technical owner, distinct from authors and first reader.",
        claim="Not completely blind to historical incidental metadata. Independent semantic judgments were preserved and were not overwritten by old scores or pass flags.",
        disclosures=[
            "20.4 OCR source status/prior peer metadata and20.8 image_review/peer_review/prior_validation.reason were incidentally printed. No peer report/author research REPORT was read; those conclusions are excluded.",
            "20.8 own scene14/28 and facts31/56 were saved before historic per-case grade lookup. Historical scene16/28 remains historical; own1039 aggregate23/28 and facts53/56 differs per case. Candidate EOS/voice failures remain separate valid gates.",
            "Historical first three grading reasons were exposed after own complete semantic scoring; permitted historical raw-score aggregation did not replace it.",
            "C.1 broad raw structures exposed candidate/acceptance field names and some old interpretation metadata; own reconstructed IDs/targets/input masks and selected raw execution fields used instead.",
            "Natural UI/data original manifests included image_review/source metadata; Python package license/status fields incidentally appeared. No approval or status flag used as proof.",
            "Training fixed-text plan status/history paths and broad rg private-audit filename/SHA fragments were seen; actual audit content and author review conclusions were not read as answers.",
            "Readme broad historical raw print exposed completed/approval flags, CI success/test-summary and source-selection field names; raw bytes/tensor fields/argv and personally checked source contracts only support claims.",
            "Curriculum original capstone interpretation/limitation/recommendation snippets were incidentally printed. Own by-task counts, actual pinned source functions and CPU formulas form evidence, not those snippets.",
            "Validation historical checks and sha_verified booleans were printed; historical L4/steps/parameter/difference fields and comparison source were checked independently within stated limits.",
            "Training-assets raw source field names status/checks/training_permission appeared; unpack copied original README/metadata but those were not read as judgments. All eight old Git LFS objects were read only and never altered.",
            "Training-assets unit7 checkpoint prewrote successful62 before underlying failed command returned. Original failed assertion and second group-count assertion stay immutable; unit8 actual-v4 corrects original62/15/30 vs558 derived-inclusive paths/49 groups. No unverified scope is silently changed to pass.",
            "Publishing capture started at05:21:58 before true page end. Desktop image3, viewed about05:22, exposed unread§4 fixed HF revision/16 files/local card. Root notified; unit8 then actually unlocked and independently checked after05:22:55. Mobile three viewed after real unit8 read. No all-blind-reading claim.",
            "Publishing broad rg exposed historical-review explanation in update_course_progress.py and card descriptor redistribution_approved; excluded. Card inspected only headings/bytes/hash and actual README-only publication function.",
            "Truncated raw/source/image outputs were not treated as fully inspected; necessary items were subsequently read/viewed in smaller returns and receipts record only actual tool views.",
            "Source paths supplied by root were raw-location indices only. Public-course-review/author reports, old integrity/closure summaries, raw progressive state and others' current review answers were not intentionally read.",
            "All corrections were appended or assembled in a new report; earlier trace/proofs/failed executions remain unchanged. The report's fresh origin describes dispatch identity, not perfect avoidance of incidental historical metadata.",
        ]),
    corrections_and_failed_probes=[
        "Initial full draft20261007T053013 preserved; own metadata check found only3 unsupported raw_output kind values. New report maps those actual raw JSON snapshots to allowed source_snapshot; no data/judgment/trace changed.",
        "19.12 test count3762→3662 + tools72=3734;20.4 receipt corrected-v2;20.8 unit0 discontinuous quote rejected and true replacement stored.",
        "20.13 guessed hash corrected actual original in next note;A.7 total135 corrected130;B.6 positions151/146 corrected132/127;B.1 import/canonical hash mistakes corrected with original failures retained.",
        "B.3 mutable-list alias made saved first prompt include TOOL_RESULT, so exact initial-prompt claim excluded; original two input sequences still independently checked.",
        "Python wrapper vs actual .venv interpreter versions corrected in env-v2 artifacts; system Python3.12.14 differs from project3.13.5/Torch2.14.1+cpu. No CUDA inferred from CPU fixture.",
        "Natural source guessed script path corrected; pins12 corrected11; original audio36 corrected38; core/outer seconds corrected1728.676332246; checkpoint stores torch/CUDA RNG not generic Python random state.",
        "Training source guesses/system Python missing torch corrected; SFT labels/generation and quantization-only dependency scope corrected in later units; capture JSONL parse failure corrected without deleting first record.",
        "Readme default-vocab256 parameter counts5371843/2212803 excluded in favor of actual550 vocab5447107/2288067. Default train_stage is dry-run unless--train, corrected next unit.",
        "Asset-storage current v2 archive is actual67862431B payload, not133B pointer; Git HEAD still has pointer. HF decorated-source locator corrected by inspect.unwrap file/line.",
        "Full original documents/papers stay local cache; official metadata now binds exact original SHA at already checked URL/version/locator without CI depending on a complete PDF/text snapshot.",
        "Publishing first exporter import failed missing scripts path; corrected-v2 real four-case fixture retained. Unit3 direct dependency overlist corrected atunit4; card URL regex quantities excluded.",
        "Post-completion source lookup guessed missing prepare_vision.py and experiments.json; actual recipe source plan.json and already checked source paths used in new records. Missing lookup did not establish a contract.",
        "First report assembly assumed trace checkpoint had primary bool and raised KeyError before writing any report; corrected to actual assigned primary_page_ids membership.",
    ])
assert len(pages) == 54
assert [i["id"] for i in report["issues"]] == ["F20.8-history-grading-scope", "F20.13-action-subset-scope", "Fnatural-training-resume-guard"]
folder = M.parent / "reports/technical/f"
folder.mkdir(parents=True, exist_ok=True)
target = folder / ("p7_technical_f-initial-" + timestamp.strftime("%Y%m%dT%H%M%S") + ".json")
with target.open("x") as out: out.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(dict(path=str(target.relative_to(R)), sha256=hashlib.sha256(target.read_bytes()).hexdigest(), verdict=report["verdict"], pages=len(pages), checkpoints=len(notes), issues=[i["id"] for i in report["issues"]]), ensure_ascii=False))
