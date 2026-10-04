"""Build the independently inspected 8.9 report from persistent bounded evidence."""

import hashlib
import json
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path.cwd()
ART = "docs/technical-reviews/artifacts/"
PREFIX = "fact_v2_08_09"
audit = json.loads((ROOT / ART / f"{PREFIX}_audit-result.json").read_text())
body = dict(sections(ROOT / "course/chapters/08.md"))["8.9"]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def ref(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


def verify(expected, observed, details, tolerance=None, denominators=None):
    result = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if tolerance is not None:
        result["tolerance"] = tolerance
    if denominators is not None:
        result["denominators"] = denominators
    return result


artifacts = []
artifact_ids = {}
for path in sorted((ROOT / ART).glob(f"{PREFIX}_*")):
    relative = path.relative_to(ROOT).as_posix()
    suffix = path.name.removeprefix(PREFIX + "_")
    identifier = "a_" + suffix.replace(".", "_").replace("-", "_")
    artifact_ids[suffix] = identifier
    kind = "execution" if suffix == "audit-result.json" else "code" if suffix.endswith(".py") or "-code" in suffix else "derivation" if suffix == "derivation.md" else "source_snapshot"
    entry = {"id": identifier, "kind": kind, "path": relative, "sha256": sha(relative), "description": f"8.9 獨立審查持久證據：{suffix}。原始來源/精確原文、凍結原程式、重建且hash匹配資料或當次CPU輸出；用途由逐主張引用定位。"}
    if kind == "execution":
        entry.update(command=audit["command"], result=audit["result"], environment=audit["environment"])
    artifacts.append(entry)
for label, path in [
    ("formal_lora", "docs/course-experiments/results/lora.json"),
    ("formal_style", "docs/course-experiments/results/style.json"),
    ("student_cli", "docs/course-experiments/student-checks/media-and-raw-adapter-cli.json"),
    ("style_download", "checkpoints/course/style/download-manifest.json"),
    ("lora_download", "checkpoints/course/lora/download-manifest.json"),
    ("style_export", "checkpoints/course/style/export-manifest.json"),
    ("lora_export", "checkpoints/course/lora/export-manifest.json"),
]:
    artifacts.append({"id": "a_" + label, "kind": "source_snapshot", "path": path, "sha256": sha(path), "description": "既有正式實驗／公開權重來源指紋證據；親讀後以當次CPU audit核對配置、分母及逐題，不冒稱重新執行歷史CUDA訓練。"})

run_art = artifact_ids["audit-result.json"]
math_art = artifact_ids["derivation.md"]
paper_art = artifact_ids["lora-v2.pdf"]
sources = [{"id": "s_lora", "kind": "paper", "title": "LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685v2", "version": "arXiv:2106.09685v2, 16 October 2021; original PDF header and fetched metadata independently matched", "verified": True, "checked_original": True, "accessed_on": "2026-10-04", "authority_reason": "Hu等人的LoRA原始論文；直接定義固定W0、低秩BA、alpha/r縮放與合併/任務切換。", "inspection_note": "親讀原PDF以pdftotext重抽§4.1 pp4–5 Eq(3)、初始化和alpha/r段、No Additional Inference Latency，以及§4.2 Practical Benefits and Limitations。原文明示W0凍結、A/B同輸入相加、可合併且換不同B'A'；合併後不同任務批次限制也有說明。該來源只支持數學機制，不支持本專案的風格品質。"}]
for identifier, title, path, note in [
    ("s_nograd", "PyTorch no_grad official source", "torch/autograd/grad_mode.py", "親讀class no_grad lines22–85：反向autograd結果requires_grad=False、thread local、factory exception及不適用forward AD；本節普通leaf Parameter的手動fill/copy/clone皆在上下文內。"),
    ("s_tensor", "PyTorch Tensor.copy_ and fill_ official API docstrings", "torch/_tensor_docs.py", "親讀Tensor.clone lines1108–1114、copy_ lines1146–1170（copy source元素、可broadcast/cast）、fill_ lines1901–1909（原tensor全部填指定value）。本節來源shape/dtype一致，所以不依賴broadcast或cast例外。"),
    ("s_torchdocs", "PyTorch clone and equal official API docstrings", "torch/_torch_docs.py", "親讀torch.clone lines2909–2934與torch.equal lines4241–4263：clone複製且通常可微；本節在no_grad內故備份無grad_fn。equal核size與元素，不比較dtype且NaN不相等；本節都是有限同dtype矩陣。"),
    ("s_fp", "PyTorch Numerical accuracy", "docs/source/notes/numerical_accuracy.md", "親讀開頭Numerical accuracy與Batched computations or slice computations：浮點加法/乘法不具結合性、數學相同不保證bitwise、跨release/CPU/GPU也不同。精度說明支持本節近似一致的限制，不能推出任何固定普遍容忍差。"),
]:
    sources.append({"id": identifier, "kind": "official_source", "title": title, "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/" + path, "version": "pytorch/pytorch tag v2.14.1; matches installed torch 2.14.1+cpu", "verified": True, "checked_original": True, "accessed_on": "2026-10-04", "authority_reason": "PyTorch維護者在官方pytorch/pytorch repository發布的固定版本原始API docstrings與精度說明。", "inspection_note": note + " 官方docs網站403後改讀該固定tag原文；快照保存於同前綴artifacts。"})
for identifier, path, version, note in [
    ("s_alignment", "tiny_perceptron/alignment.py", "current worktree; exact file hash also matches formal lora code_sha256", "lines11–26：LoRALinear的a=(rank,in)、b=(out,rank)、forward為base(x)+(xA^TB^T)alpha/rank、merged_weight=W+BA alpha/rank；base bias保存。"),
    ("s_adapters", "tiny_perceptron/adapters.py", "current worktree", "完整讀lines1–145：FP32基模數字指紋、config、alpha/rank、native base同一性、明示module path、entry精確a/b/rank/alpha及shape/finite檢查，全部prepared後才植入。loader允许adapter自己明示的子集，沒有獨立聲稱必須11層；本節完整切換使用兩份實際相同11層payload。"),
    ("s_infer", "scripts/infer.py", "current worktree", "完整讀CLI：原始native checkpoint載入、--adapter先驗證、--chat=user/EOS/assistant、預設temperature0、生成EOS報告；merged檔直接checkpoint不加adapter。"),
    ("s_formal", ART + PREFIX + "_formal-behavior.py.txt", "git object a7cdffec4dc1d2356264f26e37da5705a298877e:scripts/course_experiments/behavior.py; SHA matches formal report", "完整讀LoRA相關§_add_lora、_adapter_state、_restore_adapter、_merge_lora、_fit_adapter、run_lora，並讀資料模板/rubric：訓練11處、凍結base、450次16題AdamW0.003、完整A/B/alpha恢復與9位置probe、1e-6 switch/1e-4 merge threshold。當前behavior有rubric修訂，故歷史測量只引用這份實際hash匹配版本。"),
    ("s_common", ART + PREFIX + "_formal-common.py.txt", "same formal git object; SHA matches report", "split_records按去重family切分；text_examples/render_chat回答遮罩；fit_lm、evaluate_lm完整生成無抽樣題目截取、temperature預設0、96 tokens及有效target總數。"),
    ("s_formal_run", ART + PREFIX + "_formal-run.py.txt", "same formal git object; SHA matches report", "execute讀seed42與set_num_threads(2)、CUDA同步、訓練/evaluation/save包覆計時；timing_scope明確排除image build/startup/HF upload。本節不宣稱速度。"),
    ("s_data", "tiny_perceptron/data.py", "current worktree; matches formal SHA", "ByteTokenizer全部原始UTF8 bytes加8；render_chat只assistant內容加EOS為labels、角色/user皆-100；pad_batch分開attention mask與target ignore。"),
    ("s_model", "tiny_perceptron/model.py", "current worktree; matches formal SHA", "TinyLM輸出logits(batch,time,vocab)；ModelConfig/forward/loss_sum/ generate：264 vocabulary、FP32/native config、有效label分母、greedy/EOS；probe對每個9位置所有264候選比較。"),
]:
    sources.append({"id": identifier, "kind": "repository_code", "title": path, "path": path, "sha256": sha(path), "version": version, "verified": True, "inspection_note": note})
sources.extend([
    {"id": "s_math", "kind": "derivation", "title": "LoRA identity, base binding, alpha ratio and double application", "verified": True, "details": (ROOT / ART / f"{PREFIX}_derivation.md").read_text()},
    {"id": "s_run", "kind": "execution", "title": "Independent 8.9 bounded CPU audit", "verified": True, "artifact_id": run_art},
])

claims = []


def claim(identifier, kind, statement, location, evidence, scope, artifact_list=None, verification=None):
    item = {"id": identifier, "kind": kind, "statement": statement, "location": location, "status": "verified", "evidence": evidence, "artifact_ids": artifact_list or [], "scope": scope}
    if verification:
        item["verification"] = verification
    claims.append(item)


claim("c1", "concept", "同一個固定基模可配不同LoRA；有效權重為W+(alpha/rank)BA，完整切換替換指定adapter的修正。", "第1–2段：基模/adapter與有效矩陣", [ref("s_lora", "§4.1 Eq(3), scaling paragraph; §4.2 Practical Benefits", "固定W0與多套A/B的分支/切換機制"), ref("s_alignment", "LoRALinear.forward/merged_weight lines11–26", "本專案採同一公式")], "線性低秩修正、base bias保留；數學可切換並不保證手設數字是文風。", [paper_art, math_art])
claim("c2", "software", "no_grad內的手動fill/copy與clone不被記成反向訓練圖。", "程式with torch.no_grad與其後第1段解釋", [ref("s_nograd", "class no_grad lines22–85", "本節操作結果不建立反向autograd關係"), ref("s_run", "snippet_checks.original.backup_requires_grad/backup_grad_fn", "實跑clone備份requires_grad=False、grad_fn=None")], "只指這組普通FP32 Parameter操作；no_grad不關掉所有可能的forward AD，也不改factory Parameter例外。", [run_art], verify("備份不帶計算圖", "requires_grad=false;grad_fn=None", "compile/exec精確抽取原節python區塊，直接檢查備份。"))
claim("c3", "software", "fill_改當下矩陣，而clone複製內容，避免後續原地修改一併改壞備份。", "程式兩套adapter_a/b建立與clone說明", [ref("s_tensor", "Tensor.fill_ lines1901–1909; copy_ lines1146–1170", "原地填值及元素copy"), ref("s_torchdocs", "torch.clone lines2909–2934", "獨立資料複製"), ref("s_run", "snippet_checks.original; alias/backup assertion in audit.py", "兩套A/B均不同且備份storage不與原矩陣共享")], "本例clone在no_grad中；一般clone仍可微，獨立storage不等於預設脫離autograd。", [run_art], verify("兩套A/B不同，clone備份保留當下內容", "兩套a/b differences=true; backup_distinct_storage=true;直接alias隨fill_改變而clone不變", "原程式seed0、Linear(4,3)、rank2；額外alias控制實驗測實際storage語義。"))
claim("c4", "numeric", "原程式切回後B平均印0.1，A與B逐元素均恢復並印True。", "程式最後兩個print與輸出解釋", [ref("s_run", "snippet_checks.original", "精確原程式輸出與shape"), ref("s_torchdocs", "torch.equal lines4241–4263", "有限同shape的逐元素比較")], "此4進3出rank2手設FP32算例；只核切換，未做風格訓練。", [run_art, math_art], verify("切回A的B平均 0.1; A兩個矩陣都恢復 True", "B raw mean=0.10000000149011612;round(...,2)=0.1;a_restored/b_restored=true", "B為3×2共六個0.1；精確抽取本節執行，seed0；比較兩張有限同dtype備份。", "display round to 2 decimals exact 0.1;torch.equal exact stored elements"))
claim("c5", "software", "兩個copy_把A/B替換回第一組；程式只在記憶體備份，沒有加總兩組ΔW或保存權重檔。", "兩個copy_後的正文段", [ref("s_tensor", "Tensor.copy_ lines1146–1170", "copy不是相加"), ref("s_run", "精確original-code.txt及snippet_checks", "程式沒有save/serialization；兩個copy均恢復")], "示範設定兩組相同rank/alpha，所以不展示不同設定的檔案切換；不可拿此程式當完整adapter檔案格式。", [run_art, artifact_ids["original-code.txt"]], verify("替換a/b，沒有第三組混合或寫檔", "a_restored=true;b_restored=true;原碼僅clone/fill/copy/print", "親讀整個精確抽取區塊，CPU執行其兩張矩陣結果；沒有訓練或儲存呼叫。"))
claim("c6", "concept", "正式adapter需與基模版本、模組位置、rank與縮放設定綁定；矩陣形狀匹配不足以保证保留原本意義。", "除了數字段與不相容版本段", [ref("s_lora", "§4.1 frozen W0; Eq(3)", "adapter定義在特定固定W0上"), ref("s_math", "base binding derivation item1", "更換W為W+E時有效矩陣多出E"), ref("s_adapters", "base_state_sha256/load_lora_adapter", "本專案嚴格比config與原始FP32指紋")], "是相容性必要條件，不聲稱所有權重不同就必然效果差；對未知基模沒有品質保證。", [math_art, run_art])
claim("c7", "numeric", "固定A、B和rank，alpha4與alpha2相比，前者修正正好放大兩倍。", "切換還需要恢復縮放設定段", [ref("s_lora", "§4.1 scaling paragraph", "scale=alpha/r"), ref("s_math", "item2", "(4/r)/(2/r)=2"), ref("s_run", "snippet_checks.scaling", "rank2兩種alpha修正矩陣")], "固定rank且A/B相同；說的是修正ΔW，不是整個輸出或基模權重變兩倍。", [run_art, math_art], verify("alpha4修正=2*alpha2修正", "FP32 entries=-0.08000000566244125 versus -0.04000000283122063;torch.equal(delta4,2*delta2)=True", "rank2、A全部0.2、B全部-0.1，明算每個BA entry=-0.04；更換alpha後只比較ΔW。", "理想實數比例精確2；這次CPU二進位乘2逐元素相等"))
claim("c8", "software", "檔案需有各明示層的A/B/rank/alpha與縮放；不相容版本或不完整單層矩陣應報出差異。", "正式檔案設定/層名完整性與T.5操作連結", [ref("s_formal", "_adapter_state/_fit_adapter", "訓練保存各層a/b/rank/alpha、base_sha256、config、scaling"), ref("s_adapters", "load_lora_adapter validation/prepared loop", "逐層檢查並在全部通過後植入"), ref("s_run", "rejections;switch_and_merge_cpu.modules", "wrong_base/scaling/missing_b/path/shape均ValueError且無半植入，兩套11處明示")], "loader接納adapter自己明示的合法子集；要判斷完整11層切換仍需與預期清單比對。本節並未宣稱loader自動偵測被整個省略的entry。", [run_art], verify("完整檔通過，不相容與缺單張矩陣拒絕", "兩套11處通過；5個malformed檔均ValueError並no_partial_insertion=true", "用相同原始FP32基模公開檔執行；故障輸入由audit程式逐項構造，未修改共用checkpoint。"))

denominators = {"seed": 42, "base_config": audit["model_config"], "training": audit["historical_training_configuration"], "records_per_adapter": {"train": 49, "validation": 8, "test": 7}, "families": {"train": 28, "validation": 4, "test": 4}, "training_effective_targets": {"concise": 16591, "vivid": 318991, "full_sft": 318991}, "validation_effective_targets": {"concise": 16, "vivid": 352}, "test_effective_targets": {"concise": 14, "vivid": 308}, "probe": {"batches": 1, "positions": 9, "vocabulary": 264, "logits_compared": 2376, "input_ids": [[1, 3, 57, 51, 57, 69, 71, 2, 4]]}, "switch_sequence": "concise→vivid→concise, eval/no_grad", "historical_environment": audit["historical_experiment"], "measurement_repetitions": 1, "warmup": "not a latency benchmark; no timed warmup denominator", "generation": {"temperature": 0, "max_new_tokens": 96, "all_heldout_examples": 30}}
claim("c9", "empirical", "正式短答→比喻→短答的同一1+1對話probe中，所有位置候選分數切回差0，兩套分數不同。", "上一節兩套真訓練adapter的切換段", [ref("s_formal", "run_lora probe/_restore_adapter", "9位置×264全logits、A/B/alpha恢復與兩份實訓權重"), ref("s_run", "historical_switch_merge;switch_and_merge_cpu;data_audit", "逐原始配置/資料hash/targets核對，CPU切回差0且两套差22.810192108154297")], "正式一次CUDA probe數值；CPU獨立支持同一切换機制。這不是全部新题或開放文風品質評估，也不是速度benchmark。", [run_art, "a_formal_lora", "a_formal_style"], verify("historical a_b_a=0且adapter_a_b差非零", "archived CUDA=0/22.810190200805664; independent CPU=0/22.810192108154297", "親讀正式報告和hash匹配原碼；公開檔base指紋匹配，實執行完整2376 logits切换，30題CPU生成IDs同歷史；没有重訓或重跑CUDA。", denominators=denominators))
claim("c10", "concept", "手設或數值一致的切換檢查不證明模型具有良好文風或新題能力。", "第2段末及切換段末的限制", [ref("s_lora", "§4.1 mechanism versus §5 empirical experiments", "方法定義與品質評估是不同證據"), ref("s_fp", "Numerical accuracy introductory section", "數值等效是運算性質"), ref("s_run", "evaluations_cpu_against_historical", "所有15題/adapter實生成核對，算術仍未匹配")], "本節沒有把一次probe稱為普遍風格成功；限制與實驗資料相符。", [run_art, "a_formal_lora"])
claim("c11", "concept", "指定LoRA修正可合進W另存完整模型，普通推論不需再載該adapter。", "若不需要繼續切換段前半", [ref("s_lora", "§4.1 No Additional Inference Latency", "W=W0+BA可直接儲存推論；須套先前alpha/r縮放"), ref("s_math", "first identity", "base(x)+sBAx等於(W+sBA)x並保留bias"), ref("s_formal", "_merge_lora/run_lora saves", "深拷貝base/merged_weight、移除LoRALinear並另存native checkpoint")], "本節線性分支，沒有dropout等需額外處理的adapter操作；已合併推論不驗證泛用速度改善。", [math_art, run_art])
claim("c12", "empirical", "正式同probe比喻adapter與合併模型最大logit差約0.00000668，小於本次0.0001容忍值。", "合併段具體數值", [ref("s_formal", "run_lora merged_difference and threshold", "同probe最大absolute diff与1e-4規約"), ref("s_run", "historical_switch_merge.merge_max_difference;switch_and_merge_cpu", "原始CUDA值6.67572021484375e-06，CPU新合併及公開合併檔也在限內")], "0.00000668是CUDA2.14.1+cu126/L4的既有實測四捨五入值；CPU並不需等於它。容忍差只屬此配置/一次2376 logits probe，不保證任意序列或平台。", [run_art, "a_formal_lora", math_art], verify("archived CUDA merge=6.67572021484375e-06<1e-4", "archived value讀值相同；CPU fresh merge=1.2516975402832031e-05,saved GPU-merged checkpoint=2.7179718017578125e-05，兩者<1e-4", "沒有CUDA重訓；原始hash匹配code說明測量位置與threshold。CPU實載原始base+vivid、重合併和另存native公開merged檔逐logit比較；FP32 CPU重合併weights不逐位等於CUDA合併weights屬預期限制。", denominators=denominators))
claim("c13", "concept", "浮點運算次序改變時，數學等效合併不保證逐位完全相等。", "合併段浮點加總順序句", [ref("s_fp", "Numerical accuracy opening paragraphs", "浮點非結合性、跨版本/backend不保證bitwise"), ref("s_lora", "§4.1 Eq(3)/merge", "僅實數代數的等價"), ref("s_run", "switch_and_merge_cpu", "有限非零合併误差及CPU/GPU合併權重差")], "合併不只改加總，也改矩陣乘法結合/rounding次序；教材的通常不保證完全相等正確，沒有保證固定誤差界。", [run_art])
claim("c14", "software", "合併檔已含修正，應作普通checkpoint用，不能再加同一adapter一次。", "合併段末與T.5連結", [ref("s_math", "item3 double application", "再加非零ΔW形成W+2ΔW"), ref("s_adapters", "base_sha256 check and already-LoRA rejection", "merged基模不同指紋、已adapted拒絕再植入"), ref("s_infer", "main --adapter path", "直接用merged checkpoint省略adapter"), ref("s_run", "rejections.already_merged/already_adapted; CLI", "公開merged再載原adapter ValueError；原始base推論兩種版本正常")], "非零修正的權重會double apply；不聲稱每種input都必定改output。正式loader會拒絕錯base，這裡沒有以重新修改指紋來繞過。", [run_art, math_art, "a_student_cli"], verify("merged用普通checkpoint；不接受重加原adapter", "公開merged-vivid.pt再載adapter被base_sha256拒絕；已LoRA模型被拒絕；兩套原始base CLI正常EOS，回答3與3，像把兩組積木合在一起再數。", "執行load_checkpoint/loader故障驗證；實跑T.5兩條公開raw-adapter CPU CLI。保留base與公開檔SHA及download/export manifests，不將單题當品質總分。"))
claim("c15", "numeric", "刪掉恢復A的copy_後，B平均仍印0.1，但兩張矩陣均恢復的判斷為False。", "最後練習段", [ref("s_run", "snippet_checks.exercise", "按原文只刪一行後實際執行"), ref("s_math", "item4", "保留第二組A而恢復第一組B，平均不能識別漏A")], "固定seed0和原來兩组配置；不是隨意改參數的普遍輸出保證。", [run_art, math_art], verify("切回A的B平均 0.1;A兩個矩陣都恢復 False", "stdout相同；a_restored=false,b_restored=true;B mean raw0.10000000149011612", "同一精確區塊只移除layer.a.copy_(adapter_a['a'])；沒有刪除或重排其他行。", "print顯示精確匹配；stored tensor comparison exact"))

report = {
    "schema_version": 1,
    "review_stage": "technical",
    "lesson_id": "8.9",
    "source": "course/chapters/08.md#8.9",
    "reviewer_task": "/root/integration_technical_coordinator/fact_v2_08_09",
    "reviewer_context": "fresh",
    "source_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    "figure_sha256": {},
    "verdict": "pass",
    "prerequisites_read": ["course/chapters/08.md#8.8", "course/training.md#T.5"],
    "review_scope": "僅獨立審8.9；完整讀technical-review-guide、8.9及明示必要前置8.8/T.5。未讀舊審查結論或作者歷史、未委派、未訓練/付費/改共用環境或教材；author_tasks無可靠派工資料故不猜填。",
    "claims": claims,
    "sources": sources,
    "artifacts": artifacts,
    "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "逐核15項：固定基模不同LoRA、矩陣shape/备份/copy、缩放、切换、合并、不重加與练习均由原始論文、固定版PyTorch來源、專案原碼及CPU執行支持。", "claim_ids": [c["id"] for c in claims]},
        "numeric_verification": {"status": "pass", "details": "精確原區塊與只刪A恢復行的练习分别0.1/True和0.1/False；alpha因子2實算；正式CUDA6.67572021484375e-06正確round為0.00000668且<1e-4。CPU兩種merge最大差1.2516975402832031e-05/2.7179718017578125e-05保留作另環境證據。", "claim_ids": ["c4", "c7", "c9", "c12", "c15"]},
        "figure_consistency": {"status": "not_applicable", "details": "8.9全文沒有圖片/SVG引用，因此figure_sha256={}；8.8作概念前置親讀，但此節不以其圖提出新主張。", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "親讀LoRA arXiv2106.09685v2原PDF§4.1–4.2，metadata確認版本；官方docs403後直接取pytorch/pytorch v2.14.1原API和numerical_accuracy文件。現碼逐讀，歷史lora behavior/common/run/text由正式revision取出且SHA逐份對上code_sha256；報告和全部持久artifact可追溯。", "claim_ids": [c["id"] for c in claims]},
        "limitations": {"status": "pass", "details": "清楚區分手設矩陣與訓練文風、單9位置probe與新題品質、實數恆等式與浮點近似、歷史CUDA值與當次CPU結果。兩套各49/8/7資料與450×16有效target逐hash/sampler核，30題逐生成IDs與既有報告相同；既有24.482532592秒只含experiment train/eval/local saves，排除image build/startup/HF，不新宣稱速度/普遍能力。完整層清單需要外部期望，loader合法子集規約未誤說成自動查全11層；inference requires_grad不當訓練參數數量。", "claim_ids": ["c1", "c2", "c3", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14"]},
    },
}
(ROOT / "docs/technical-reviews/8.9.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"report": "docs/technical-reviews/8.9.json", "verdict": report["verdict"], "source_sha256": report["source_sha256"], "claims": len(claims), "artifacts": len(artifacts)}, ensure_ascii=False))
