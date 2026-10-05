"""Write only this fresh reviewer's own 19.11 report; do not read old reports."""
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
A = Path(__file__).resolve().parent
TASK = "/root/phase4_factual_coordinator/factual_19_11"
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def aid(rel):
    return "a-" + rel.replace("/", "--").replace(".", "_")
env = json.loads((A / "environment.json").read_bytes())
command = json.loads((A / "execution.json").read_bytes())["command"]
artifacts = []
for p in sorted(A.rglob("*")):
    if not p.is_file() or p.name in {"checker.stdout.txt", "final-check.json"}:
        continue
    rel = p.relative_to(A).as_posix()
    kind = "code" if p.suffix == ".py" or p.name.endswith("program.txt") else "source_snapshot"
    if p.name in {"results.json", "stdout.txt", "stderr.txt", "environment.json", "execution.json", "list.stdout.txt", "public-existing-byte-audit.json"}:
        kind = "execution"
    if p.name == "derivations.md":
        kind = "derivation"
    item = {"id": aid(rel), "path": p.relative_to(ROOT).as_posix(), "sha256": digest(p),
            "kind": kind, "description": {
                "read-scope.json": "本人實際原文/AST/JSON pointers/權威原文讀取範圍與未做工作的紀錄。",
                "section.md": "19.11 原始 UTF-8 bytes；不去尾白或正規化換行。",
                "results.json": "本人短 CPU 檢查與既有原始測量重算結果；零模型下載、零新生成/新分數。",
                "public-existing-byte-audit.json": "11 份既有公開權重的原 SHA/數值/config 比較、必要原值及固定公開 URL/provenance；沒有保存新權重。",
                "derivations.md": "bytes/分母/單步與總步數/時間範圍的本人推導。",
                "inputs/course/chapters/19.md": "最初完整章節的 frozen input，僅保存/hash；不是目前整章版本宣告。",
            }.get(rel, "本輪真實原始來源/方法/命令/版本/執行輸出快照：" + rel)}
    if kind == "execution":
        item.update(command=command, result="實際執行 exit=0；原 fence、tiny save/load、10 fail-fast 拒絕、離線 list/fetch fixture、原記錄與 11 現有 checkpoint 核對均完成。詳見 results.json；没有完整訓練或新能力分數。", environment={k: str(v) for k, v in env.items()})
    artifacts.append(item)

sources = []
def repository(identifier, title, rel, note):
    p = A / "inputs" / rel
    sources.append({"id": identifier, "kind": "repository_code", "title": title,
                    "path": p.relative_to(ROOT).as_posix(), "sha256": digest(p),
                    "version": "本輪原檔 SHA-256 " + digest(p), "verified": True,
                    "inspection_note": note})
repository("capstone", "CapstoneModel 與原 save/load/runtime/evaluation 方法", "tiny_perceptron/capstone.py",
           "本人讀 1-121、285-296、354-375、499-598、608-694；AST 先定位。參數/格式/helper 是原方法；無讀作者 review 摘要。")
repository("tokenizer-config", "原 ByteTokenizer 與模型設定", "tiny_perceptron/data.py", "本人讀 ByteTokenizer 1-28：UTF-8 byte +8、264 vocab、8 specials、state()；模型設定另見 model-contract。")
repository("model-contract", "原 ModelConfig / TinyLM 結構與輸出契約", "tiny_perceptron/model.py", "本人讀 ModelConfig 1-28 與 TinyLM 53-89：config、state tensors、logits/cache/auxiliary 契約。")
repository("trainer", "原 capstone train_stage", "scripts/course_experiments/capstone.py", "本人 AST 定位並讀 1-244（未讀 245-275 report 解釋欄位）；resume、資料/排程/code guard、sampler/optimizer/reference、timer、save/export。")
repository("cli", "原 capstone CLI parser / infer / serve dispatch", "scripts/capstone.py", "本人讀 main 完整，CPU 執行真 parser 與 tiny PTQ loader dispatch；training 與 generation 由具名 stub 替換，沒跑完整訓練或公開模型回答。")
repository("fetch", "原下載與 --list API", "scripts/fetch_capstone.py", "本人读完整 fetch/main：固定 revision/token=False、SHA/size/identity、完整資料夾 rename、已存在目標拒絕；實跑 --list 与 no-network fixture。")
repository("release", "原公開 inference allowlist / validators / public manifest builder", "scripts/capstone_release.py", "本人讀 18-241、366-455；architecture/metadata allowlist、student alias、finite-shape forward、anonymous pinned original verification 方法；不讀模型卡作者結果解說。")
repository("quantization", "原 Capstone int4/int8 儲存與 FP32 還原", "tiny_perceptron/capstone_quantization.py", "本人讀 21-189；per-output-channel 原碼、pack metadata、integers.float()*scale、strict load；tiny CPU 實測，沒有速度/記憶體 benchmark。")
repository("ui", "原 serve CPU loader", "tiny_perceptron/capstone_ui.py", "本人讀 181-215；serve 直接調 FP32 load_capstone，PTQ 在開 server 前被拒絕。未讀 PAGE 常數。")
repository("seed", "原 seed_everything", "tiny_perceptron/training.py", "本人 AST 定位并讀 23-27：Python/torch seed；CPU 實測僅同一環境。")
repository("packing", "原 pack_int4 / unpack_int4", "tiny_perceptron/quantization.py", "本人读 29-44：每 uint8 byte 兩個 nibble，pad nibble 与 shape count。")
repository("manifest", "固定公開 11 份檔案的原 manifest", "docs/course-experiments/capstone-public.json", "先讀 key/type 再讀 /repo /revision /models/* 具名 file path/size/SHA/output/format 欄；與 GitHub 1df335... 原件 bytes 完全一致。")
repository("raw-tensor-measurement", "既有公開 11 份推論檔逐欄比較的原測量", "docs/course-experiments/student-checks/capstone-public-tensors.json", "先 key/type，再 /public_repo /public_revision /records/* /command /environment /verification_program；避開 verification 解說欄。本人重新審核 11 rows/IDs/SHA/原 script 與既有 bytes。")
repository("raw-tensor-method", "既有逐欄 torch.equal 原檢查程式", "docs/course-experiments/student-checks/capstone-public-tensors-program.txt", "本人讀原碼完整：token=False pinned hf download、來源 SHA、dtype/shape/value 與 config/protocol equality；此方法不拿答案相似性判等。")
repository("raw-student-sample", "公開 student-kd-int4 的單一原失敗 trace", "docs/course-experiments/student-checks/capstone-public-student.json", "先 key/type，再 /public_revision /stage /checkpoint_sha256 /command /environment /record。沒有讀 scope/作者結果評語；沒有重新生成。")
repository("selection", "joint/DPO 原驗證選取規則與原候選計數", "docs/course-experiments/capstone-selection.json", "先 key/type；只讀 selected_stage、時間/先後欄、criterion 原判準與 candidates 原計量/provenance；未讀 dpo_disposition/recipe_unchanged。")
repository("joint-validation", "joint 的原 84-row validation outputs", "docs/course-experiments/capstone-evidence/joint/validation.json", "先 key/type；按 /count /by_task /records/* 原標準、eos/raw/answer/flags 重算，84 unique cases；無重新生成。")
repository("dpo-validation", "DPO 的原 84-row validation outputs", "docs/course-experiments/capstone-evidence/dpo/validation.json", "同 joint pointers：原 flags/分母/84 case IDs/answer rules 重核，by_task 實際重算，沒有新模型得分。")
repository("deployment-source", "主線 8 份原發布來源檔 provenance", "docs/course-experiments/releases/capstone_deployment.json", "先 key/type；只 /files checkpoint entries 的 path/output/sha256/kind，原 bytes hash 核；不讀 model_card 或 review 結論。")
repository("student-source", "學生 3 份原發布來源檔 provenance", "docs/course-experiments/releases/capstone_student.json", "先 key/type；只 /files checkpoint entries 的 path/output/sha256/kind，原 bytes hash 核；不讀 model_card 或 review 結論。")
fetch_info = json.loads((A / "authorities/fetch-provenance.json").read_bytes())["sources"]
for identifier, filename, title, reason, inspection, kind in [
    ("torch-checkpoint", "saving_loading_models.py", "PyTorch Saving and Loading Models 原 tutorial", "PyTorch 官方 tutorials repository，直接提供 checkpoint 與 warmstart API/方法。", "本人讀 52-68、142-185、256-314、377-415；重建 architecture、model/optimizer state_dict、一般 checkpoint、warmstart。", "official_docs"),
    ("torch-reproducibility", "torch-randomness.rst", "PyTorch v2.8.0 Reproducibility 原文件", "PyTorch 官方 repo 固定 release 原文；提供裝置/版本/seed 範圍。", "本人讀 1-84；同 seed 不保證 CPU/GPU 或平台/版本間逐步相同。", "official_docs"),
    ("torch-rng", "torch-random.py", "PyTorch v2.8.0 RNG 原碼", "PyTorch 官方原碼，定義 CPU get_rng_state/set_rng_state 與 seed API。", "本人讀 1-38；CPU state get/set、manual_seed。安裝版是 2.14.1+cpu，API 支持範圍另經实际短CPU查核；不混稱 v2.8 執行。", "official_source"),
    ("torch-dtype", "tensor_attributes.rst", "PyTorch v2.8.0 dtype 原表", "PyTorch 官方 dtype 表直接規定 float32/int8/uint8 位元寬度。", "本人讀 1-47；32-bit float = torch.float32，int8/uint8 為 8-bit。", "official_docs"),
]:
    original = next(x for x in fetch_info if x["name"] == filename)
    sources.append({"id": identifier, "kind": kind, "title": title, "url": original["url"],
                    "version": original["version"], "authority_reason": reason, "verified": True,
                    "checked_original": True, "accessed_on": "2026-10-05", "inspection_note": inspection,
                    "snapshot_artifact_id": aid("authorities/" + filename)})
sources.append({"id": "dpo-paper", "kind": "paper", "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                "url": "https://arxiv.org/pdf/2305.18290v3", "version": "arXiv:2305.18290v3, 29 Jul 2024",
                "authority_reason": "DPO 提出者的原論文；不是來源庫的結論。", "verified": True,
                "checked_original": True, "accessed_on": "2026-10-05",
                "inspection_note": "本人核原 PDF SHA-256/首頁 v3 與日期，讀 §4 p4 Eq7/政策 θ gradient、p5 DPO outline 的 given πref；PDF/text 永久保存。",
                "snapshot_artifact_id": aid("authorities/dpo-v3.pdf")})
for identifier, filename, title, inspection in [
    ("pinned-manifest", "pinned-capstone-public.json", "教材固定 GitHub revision 的公開 manifest 原件", "schema、11 models、HF revision；完整 bytes 等於本人已讀的現行原 manifest，SHA cef4452d...。"),
    ("pinned-trainer", "pinned-capstone-trainer.py", "教材固定 GitHub revision 的 trainer 原件", "AST 方法位置；完整 bytes 等於本人已讀必要 1-244 原 trainer，SHA 63fcc9d...；不讀 report 結果評語。"),
]:
    original = next(x for x in json.loads((A / "authorities/pinned-repo-provenance.json").read_bytes()) if x["path"] == filename)
    sources.append({"id": identifier, "kind": "official_source", "title": title,
                    "url": original["url"], "version": original["version"], "authority_reason": "本專案維護者的固定原始發布版本，與教材超連結直接對應。",
                    "verified": True, "checked_original": True, "accessed_on": "2026-10-05",
                    "inspection_note": inspection, "snapshot_artifact_id": aid("authorities/" + filename)})
sources += [
    {"id": "public-original-bytes", "kind": "official_source", "title": "原固定 HF revision 的 11 份公開推論 payload",
     "url": "https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed/course/course-integration-v2",
     "version": "33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed; each full SHA/URL recorded in public-existing-byte-audit.json",
     "authority_reason": "本專案維護者固定版本的實際公開 payload；本人核既有 pinned 原快取 bytes 對正式 manifest 與已保存原測量。",
     "verified": True, "checked_original": True, "accessed_on": "2026-10-05",
     "inspection_note": "零下載。本人核 11 payload 與 18 unique companion file SHA/size，load state/config/protocol/provenance、比較全部 necessary tensors/quantized fields、跑原 schema validator 的 finite-shape CPU forward（沒有生成/score）。永久保存原測量/原方法、必要原值/指紋與 pinned 公開 URL；正式引用不以 ignored cache 為唯一原件。",
     "snapshot_artifact_id": aid("public-existing-byte-audit.json")},
    {"id": "own-execution", "kind": "execution", "title": "本人 bounded CPU audit", "verified": True,
     "artifact_id": aid("results.json")},
    {"id": "own-derivation", "kind": "derivation", "title": "本人 bytes/樣本/總步數/時間的推導", "verified": True,
     "details": "derivations.md：bytes 精確整數、11×5=55 refs/18 unique files、75/84 vs71/84、單反例1+2=3與2+9=11、while saved_step<600 的總量與時間檢查範圍。"},
]

claims = []
def claim(identifier, kind, statement, location, scope, refs, *, verification=None, extra_artifacts=()):
    item = {"id": identifier, "kind": kind, "statement": statement, "location": location,
            "scope": scope, "status": "verified", "evidence": [
                {"source_id": sid, "locator": loc, "supports": supports} for sid, loc, supports in refs],
            "artifact_ids": [aid("read-scope.json"), *extra_artifacts]}
    if verification:
        item["verification"] = verification
        item["artifact_ids"] += [aid("results.json"), aid("stdout.txt"), aid("verify_cpu.py")]
    claims.append(item)
def executed(expected, observed, details, denominators=None, tolerance=None):
    v = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if denominators: v["denominators"] = denominators
    if tolerance: v["tolerance"] = tolerance
    return v

claim("c01", "concept", "推論與恢復訓練是不同交付契約；權重需要對應結構/處理契約，續訓還需 optimizer 與進度/隨機状態，檔名 model.pt 本身不能證明用途。",
      "19.11 開場與共同成品交付段", "成熟 checkpoint 一般機制；本課 tokenizer/模態/tool 格式另由原方法支持。新成品多神經元件來源/字表/意圖/失敗紀錄是第5階段交付計畫，並未據此宣稱已完成。",
      [("torch-checkpoint", "52-68, 142-185, 256-314", "推論重建模型結構；完整 checkpoint 包含 optimizer/epoch，model state_dict 不等於完整訓練狀態。"),
       ("capstone", "74-121, 285-296, 550-598, 612-652", "本課實際結構、tokenizer/模態/工具與可選 training fields。")])
claim("c02", "software", "原 Python fence 實際保存/讀回未訓練 step=0 的 sft 標籤推論包；description 比較只核 config/參數數目，不能證明 SFT 或 tensor/任務正確。",
      "19.11 fence-1 與其後說明", "原 fence 被逐 byte 原樣執行；另用 5520-param tiny 狀態驗證 exact state 與 description 的不同作用，未以這些隨機模型評測任務。",
      [("capstone", "description 109-121; save_capstone/load_capstone 612-652", "description 只有結構/numel；stage/step 是存入 metadata，save 沒有訓練更新。"),
       ("tokenizer-config", "ByteTokenizer.state 27-28", "tokenizer state 必须對上 byte protocol。"),
       ("own-execution", "results.json /original_fence /tiny_save_restore /inference_state; stdout 前3行", "真存讀/描述、tensor mutation 反例與 omitted training fields。")],
      verification=executed("capstone-v1, step0, inference_only True, description 相同；tensor 改動可以保持 description 相同。", "原 fence 输出格式 capstone-v1 訓練步數0/只供推論True/參數相同True；tiny tensor mutation 後 description 仍同。", "TemporaryDirectory 結束後無新 .pt 保存；Stage='sft' 未引入 optimizer 或更新。"))
claim("c03", "software", "--list 只列11個公開名字；stage alias/父來源把四站FP32、joint/DPO量化與Dense學生分開；各 folder 的五檔/config/tokenizer 隨正式包裹帶走。",
      "details 公開清單與下載名表", "固定舊合成任務11份推論版本；名稱能下載不代表新成品驗收或能力等價。未讀模型卡作者結果解說，僅原 provenance/格式/檔案規約。",
      [("manifest", "/revision /models/*/{id,format_version,checkpoint,files}", "11 unique IDs 與每份5個 file references。"),
       ("release", "STAGE_IDS 42; public_stage_id 96-118; payload 121-241", "student 需要 explicit branch+Dense config、teacher/data SHA；推論內含 config/tokenizer。"),
       ("fetch", "main 62-91", "--list 分支只 print manifest 不呼叫 fetch/訓練。"),
       ("own-execution", "results.json /public_list; public-existing-byte-audit.json /models/*/{config,tokenizer,provenance,metadata_keys}", "實際離線 --list 與原 payload/來源核對。")],
      verification=executed("11 unique stages；FP32/quantized/student identity 正確；每份5 file refs。", "實跑 list 11列，与 manifest IDs 同順序；payload format/stage/config/student branch 与原 source 一致。", "--list 離線執行，沒有模型下載。普通 JSON/Path/CLI/helper 契約合組核對。"))
claim("c04", "empirical", "正文所稱逐檔匿名核對並讀回全部11份，對應固定revision的既有原測量與真正相符的公開 payload，公開權重保留原來源的數值。",
      "details 第一段與正式檔案說明", "核既有發布/匿名驗證原方法和 measurement，非本輪重新下載。本人只讀既有原 bytes，做 exact numeric/config/hash/schema 檢查；無模型生成/能力分數。",
      [("raw-tensor-measurement", "/records/* /public_revision /verification_program", "11條原測量的 stage/public SHA/source SHA/compared-field list/pass。"),
       ("raw-tensor-method", "same(); token=False hf_hub_download; per-field assert loop", "原 anonymous/pinned、dtype/shape/value torch.equal 方法。"),
       ("deployment-source", "/files checkpoint path/output/sha256", "主線8份原來源指紋。"),
       ("student-source", "/files checkpoint path/output/sha256", "學生3份原來源指紋。"),
       ("public-original-bytes", "public-existing-byte-audit.json /models/* and /files", "本人比對11原 payload、full SHA、config/state/protocol、finite-shape loader。"),
       ("pinned-manifest", "/models/*; original UTF-8 SHA cef4452d...", "教材指定1df335... manifest原 bytes 与本人檢查版本完全一致。")],
      verification=executed("原測量11份各 SHA/size/identity/fields 一致且推論數值不因清理封裝改變。", "11 source-public pairs 全 fields/tensors exact 相同；11 public weights/18 unique files/55 refs 的 SHA與bytes完全吻合。", "原源與公開 hash不同但數值/架構/協定一樣；本輪0下載/0新生成/0新分數。", {"model_pairs": 11, "file_references": 55, "unique_file_paths": 18}),
      extra_artifacts=(aid("public-existing-byte-audit.json"),))
claim("c05", "numeric", "joint正式 model.pt 為1,327,950 bytes及指定 d85cca...SHA；joint-int4為270,855 bytes，不能用舊實驗封裝bytes替代。",
      "details 第二行安裝大小/SHA與量化大小段", "單一固定revision的完整檔案長度，以byte為單位；不是 tensor-only 大小、runtime記憶體或舊source封裝大小。",
      [("manifest", "/models/2/files/0 and /models/4/files/0", "正式 joint/joint-int4 exact bytes/SHA。"),
       ("public-original-bytes", "public-existing-byte-audit.json joint / joint-int4", "本人現有原公開 bytes 的stat/hash符合同版manifest。"),
       ("release", "clean_capstone_payload 121-168", "allowlist 刪除 training/內部metadata，clone保留tensor數值；封裝與內容數值各自核。")],
      verification=executed("1327950與270855 bytes；joint SHA d85cca83cdb4952f4653ef6b58d94562b41403db3a9db2ea31ff57e31246ca16。", "既有原 bytes/hash 精確吻合；joint 原source SHA7476cc...與正式SHA不同，全部 tensor/config 字段相同。", "stat.st_size 整數與 SHA-256，無舍入，正式分母是一檔。", tolerance="精確 byte 整數與完整64hex SHA 相等。"),
      extra_artifacts=(aid("derivations.md"), aid("public-existing-byte-audit.json")))
claim("c06", "software", "此處 FP32為32-bit float；int4/int8是壓縮儲存，載入還原FP32；CLI infer選適用 loader/共用工具loop，serve目前只用FP32 loader。",
      "details FP32/int4/int8說明與量化infer/serve段", "只支持本課原格式。没有推斷更少runtime記憶體/更快，也没有將Dense學生當joint。tiny loader/dispatch可核API，不能核任務能力。",
      [("torch-dtype", "tensor_attributes.rst dtype table 20-34", "torch.float32是32bit float；int8/uint8是8bitstorage。"),
       ("quantization", "restore_quantized_payload 117-184; load 187-189", "unpack/integers.float()*scale、strict float32 model還原。"),
       ("packing", "29-44", "int4兩nibbles/byte與shape。"),
       ("cli", "infer fallback and shared run_assistant, 76-102", "quant load fallback成功后仍同run_assistant路線。"),
       ("ui", "serve 202-205", "只调用 load_capstone，PTQ 在 server建立前拒絕。"),
       ("own-execution", "results.json /tiny_quantized_load /cli_quantized_infer_and_serve", "4/8 storage dtypes、FP32載入、真CLI fallback与serve拒絕；generation stub。")],
      verification=executed("int4uint8/int8int8 storage；restored param全部float32；infer dispatch共用runtime，serve拒絕PTQ。", "两tiny格式载回float32；真infer parser/loader調run_assistant recording stub一次；serve报Unsupported format且未开server。", "未生成公開模型答案，未做速度/記憶體benchmark。"))
claim("c07", "software", "下載器固定revision/token=False，驗證五檔大小/SHA及模型身分後才rename成完整folder；--output更改根資料夾，已下載joint可直接用原路徑。",
      "details fetch命令、目的地與另留一份段", "原正式下載方法與原發布measurement支持匿名性。本輪 no-network fixture只驗證validation/atomicplacement與傳給helper的token=False，沒有把fixture當真正匿名網路下載。若重跑同stage到既存target，原程式拒絕覆寫；正文『若已在就使用』是省略重下指示。",
      [("fetch", "fetch_capstone 24-59; --output 72-79", "temporary完整檔驗證、payload stage/format、rename、target.exists拒絕。"),
       ("release", "build_public_manifest 400-455", "原方法實際 pinned token=False 匿名下載並比SHA/size，對應原正式清單。"),
       ("own-execution", "results.json /fetch_fixture /public_list", "实际原API fixture五檔成功、毀損sha拒絕/不置入、既存target拒絕；helper arguments revision/tokenFalse。")],
      verification=executed("5檔驗證后整folder置入；毀損/既存target拒絕，token=False與固定revision。", "fixture成功完整5檔；毀損沒有 target/joint，existing target保留；記錄每次stub调用皆tokenFalse/固定revision。", "只替換網路helper为本地fixture，实际fetch validation不改。下載路徑根由原parser/Path拼接。"))
claim("c08", "empirical", "joint是既有合成任務驗證後選出的推薦成品，DPO是比較分支，不是新成品必須接受的最後更新。",
      "details --list说明與四站后说明", "原共同84-row validation上的選擇；不轉作一般能力/跨環境因果結論，不把舊模型/下載成績當第5階段新任務完成。",
      [("selection", "/selected_stage /criterion /candidates/{joint,dpo}", "選取規則、joint75/84與DPO71/84來源紀錄。"),
       ("joint-validation", "/records/* exact flags and /by_task", "joint原84 outputs。"),
       ("dpo-validation", "/records/* exact flags and /by_task", "DPO同84 cases原outputs。"),
       ("capstone", "evaluate_rows 499-547", "action+EOS及final exact判準，与合計方法。")],
      verification=executed("同84 unique cases/答案規則，原exact-action+EOS+final規則選較高joint。", "本人重算84 unique cases一致、joint75/84、DPO71/84，by_task与全合計均一致。", "原輸出記錄重算，不重新跑模型，也不測新任務。", {"validation_cases_per_stage": 84, "stages": 2, "joint_correct": 75, "dpo_correct": 71}))
claim("c09", "empirical", "原公開student-kd-int4單一1+2請求錯寫2+9，工具回11，最後模型答12；成功下載/載入不能證明答案正確。",
      "details student-kd-int4反例", "一個固定checkpoint/命令的原始樣本；不以單樣本估學生/teacher/joint整體能力，不宣稱重推得到同答。",
      [("raw-student-sample", "/command /checkpoint_sha256 /record/action_trace /parsed_action /runtime /final_trace /answer", "原1+2命令与原TOOL2+9/11/DIRECT12紀錄，模型SHA對同公開檔。"),
       ("capstone", "parse_action 550-561; calculator_runtime 564-574; run_assistant 577-598", "原工具加法11、runtime輸出是後續模型輸入，最終答案仍模型產生。")],
      verification=executed("1+2應為3；record錯呼2+9，runtime11，final12。", "原trace/runtimeresult/answer精確匹配；calculator_runtime原方法核2+9=11，parse_action吻合。", "只有原測量JSON解析與calculator方法短CPU核對；0模型生成。", {"original_samples": 1}))
claim("c10", "concept", "DPO policy在更新，reference固定；恢復原DPO目標需原reference，不能以當前policy再建reference冒充原續訓。",
      "details DPO續訓說明", "本課stage固定reference的DPO部分，原trainer另有CE anchor；不聲稱純DPO配方或玩具測試重新證明DPO。",
      [("dpo-paper", "§4 p4 Eq7/policy-only θ gradient; p5 DPO outline (given πref,D,β)", "objective把πref視為給定baseline，只優化πθ。"),
       ("capstone", "preference_loss 354-368; frozen_reference 371-375; save 637-638", "requires_gradFalse/no_grad与持久化原reference。"),
       ("trainer", "120; resume 146-150; 207-213", "新stage先建原reference，resume用保存reference覆寫，缺失拒絕；另0.2CEanchor。"),
       ("own-execution", "results.json /tiny_save_restore reference fields; /resume_rejections missing-dpo-reference", "temporary原reference與policy數值不同，但存讀后reference exact保存。")],
      extra_artifacts=(aid("results.json"), aid("authorities/dpo-v3.pdf")))
claim("c11", "software", "四站/續訓CLI分別設定300/1400/600/100與resume總600步、batch24/seed42/CPU；540秒只是循環預算，最後更新与后處理可超過，未完成仍存工作檔。",
      "details 四站 bash recipe與時間說明", "核原recipe及parser/loop/save分支，沒有真正重做正式CPU/GPU訓練或驗收分數。五命令只經真parser dispatch到recording stub；時間界線由原while方法核。",
      [("cli", "train arguments 15-29 and train_stage dispatch 52-67", "steps/batch/seed/device/seconds默認540、input/resume傳遞。"),
       ("seed", "seed_everything 23-27", "新實驗seed設定原方法。"),
       ("trainer", "93-98, 123-135, 176-244", "stage sampler、timer每批前判定、存model-training与export/validation在loop后。"),
       ("pinned-trainer", "1-244 exact same inspected original bytes", "教材固定link確實對同版trainer。"),
       ("own-execution", "results.json /training_cli_argument_dispatch", "五原recipe的實際parser呼叫arguments：seconds540/batch24/seed42。")],
      verification=executed("五原training命令對應原階段/路徑/數字，且訓練函式不執行；預算非整命令wallclock硬上限。", "recording stub接到steps[300,1400,600,100,600]；全部batch24/seed42/CPU/seconds540；完整訓練0次。", "timer/save/export/validation界線逐讀原方法194-244；沒有以未跑的recipe宣稱分數重現。"))
claim("c12", "numeric", "resume --steps600是原總步數，不是另外追加600。",
      "details resume bash后第一句", "while限原固定steps，已完成/未完成以原排程total比較；假設savedstep100剩500為本人算例，不是既有訓練進度。",
      [("trainer", "108-109; 150; 179-192; 195; 218", "要求原requested_steps相同，恢復step，while step<steps，每更新step+=1。"),
       ("own-derivation", "derivations.md savedstep100/total600", "600−100=500，indices100到599；不是追加600。")],
      verification={"method": "hand_calculation", "expected": "原steps600保持總量；savedstep100時剩500。", "observed": "以原while step<600及每步+1代入，600-100=500。", "details": "trainer恢复step而非清零；schedule_changed实际CPU拒絕；這不是完整續訓trajectory測試。", "tolerance": "精確整數相等。"},
      extra_artifacts=(aid("derivations.md"),))
claim("c13", "software", "resume核同站/排程/batch/資料/兩原程式指紋，讀optimizer、CPU/Python/CUDA RNG与sampler進度；seed42不重置保存位置，跨站input则新optimizer/抽樣流程並要求完成父站。",
      "details resume契約與input-checkpoint對比", "只核原trainer明列兩程式hash（不是整依賴環境fingerprint）；只做bounded same-CPU狀態查核，沒有完整GPU同軌跡重放。",
      [("trainer", "100-116, 120-152, 153-173", "data/stage/total/batch/optimizer/hash/reference guards、new optimizer/sampler與restore順序。"),
       ("torch-rng", "get_rng_state/set_rng_state 9-30", "CPU state原API契約。"),
       ("own-execution", "results.json /tiny_save_restore /resume_rejections", "next RNG/sample exact continuation、原reference恢复、10真拒絕branch。")],
      verification=executed("保存後續随机数/样本与原continuation相同；違反資料、站、steps、batch、optimizer、code、DPOreference、不完整父檔条件拒绝。", "5 sampler IDs[0,2,0,0,0]與保存處續行吻合，CPU next3/Python next完全相同；10 fail-fast cases都在training loop前拒絕。", "tiny5520-param一次人工梯度optimizertransaction，0 task training updates。Cross-stage新流程由原初始化/restore分支核；不声称完整recipe trajectory重現。"))
claim("c14", "software", "公開11推論檔缺完整工作狀態、原data_manifest/完成排程，不能用於本課正式resume或串階段input；可把數值作為另寫訓練的新實驗初值。",
      "details 最後段", "限定本課train_stage的正式入口，不宣稱數值不能學習；公開包是推論/comparison交付，新成品權重与操作仍待下一階段回填。",
      [("release", "clean_capstone_payload 121-168; PROVENANCE_KEYS 44-70", "公開allowlist省完整optimizer/RNG/reference/data_manifest/schedule fields，保留fingerprintprovenance。"),
       ("public-original-bytes", "public-existing-byte-audit.json all 11 /metadata_keys and inference_only", "本人核实际公开payload缺这些原训练字段。"),
       ("trainer", "100-116, 136-152", "数据manifest/complete-parent/same-stage训练state缺失就拒绝。"),
       ("torch-checkpoint", "377-415 warmstarting; 256-314 general checkpoint", "用旧权重初始化新模型/训练是warmstart，与完整恢复optimizer/epoch不同。"),
       ("own-execution", "results.json /inference_state /resume_rejections", "推論schema省6训练fields，inference_resume和public_parent_no_manifest拒绝。")],
      verification=executed("公开 payload只有 inference fields/provenance；正式入口拒绝缺训练manifest/state。", "11 actualpublic payload均无optimizer/training_state/RNG/reference与data_manifest/schedule证明；两相应短CPU原gate拒绝。", "不新写训练工程；可warmstart的成熟机制查官方原文，不误称恢复已验证实验。"))
claim("c15", "concept", "逐步重現還受PyTorch版本、裝置與運算實現影響，要區分同環境恢復測試与跨環境重新訓練。",
      "details DPO續訓/裝置說明", "成熟可重現性範圍。本文CPU證據不是GPU或跨版本結果；安裝torch2.14.1+cpu與引用v2.8原API/文件版本分別記錄。",
      [("torch-reproducibility", "1-15; PyTorch RNG seed 29-40; CUDA benchmarking 70-84", "跨release/platform/CPU與GPU不保證完全一致；固定seed只是控制一部分源頭。"),
       ("own-execution", "environment.json; results.json /tiny_save_restore", "實際測的是單一CPU環境之状态与抽樣恢复，不扩大至跨装置。")],
      extra_artifacts=(aid("environment.json"), aid("results.json")))

meta = json.loads((A / "extraction.json").read_bytes())
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "19.11",
    "source": "course/chapters/19.md#19.11", "source_sha256": meta["source_sha256"],
    "reviewer_task": TASK, "reviewer_context": "fresh", "author_tasks": [],
    "author_tasks_known": False, "author_identity_note": "實際原作者canonical身份未知，按派遣與協調者確認記[]，不猜。本人是新單節task，不是本節作者/第一次reader/其他節technical reviewer。",
    "verdict": "pass", "reviewed_on": "2026-10-05", "figure_sha256": {},
    "frozen_input": {"artifact_id": aid("inputs/course/chapters/19.md"),
                     "sha256": digest(A / "inputs/course/chapters/19.md"),
                     "meaning": "最初完整Markdown frozen input的保存指紋；不宣稱目前整章版本。正文實際讀19.11與未用到的末20字元，whole chapter只bytes preservation/hash。"},
    "read_scope_artifact_id": aid("read-scope.json"),
    "reviewer_summary": "本節區分推論包與完整續訓工作狀態；原step0程式真的存/讀但不訓練。details是11份已發布舊合成任務的推論交付、固定reference和正式續訓入口。所有主張按原payload/方法/既有測量與有限CPU證據核，不把舊下載、載入、validation或玩具狀態測試當新成品驗收。",
    "sources": sources, "artifacts": artifacts, "claims": claims, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "claim_ids": [x["id"] for x in claims], "details": "15組實質概念/數值/普通API/原測量逐項核。原權威支持checkpoint與DPO/重現性，原實作支持格式與入口；新成品交付段為計畫。沒有未解實質主張。"},
        "numeric_verification": {"status": "pass", "claim_ids": ["c04", "c05", "c08", "c09", "c11", "c12", "c13"], "details": "精確bytes/SHA、11models/55refs/18unique files、same84case joint75/84 vsDPO71/84、single failure 2+9=11/final12、recipe數字与steps總量、RNG/sampler數值均核。原测量重算不生成新模型分數。"},
        "figure_consistency": {"status": "not_applicable", "claim_ids": [], "details": "原小節沒有引用SVG/其他圖，helper確認0refs。文字表格與明示檔名/階段已足以說明檔案角色；無需想像素材或空間位置。沒有聲稱圖渲染/頁面驗證。"},
        "source_verification": {"status": "pass", "claim_ids": [x["id"] for x in claims], "details": "親核PyTorch原tutorial/RNG/dtype與DPOv3原PDF，不用locator摘要定真假；固定GitHub1df335...manifest/trainer真原件與local bytes相等。普通API合組实跑原fence/fixture/parser。具名JSON pointers/AST讀取範圍保存在read-scope，避開原作者review/結果修正摘要。永久保留原測量/方法/來源URL版本、SHA、真commands/stdout/env及必要原狀態數值；公開原bytes可依固定HTTPS/SHA取得，不以ignoredcache作唯一正式source。官方v2.8原API文件與實際torch2.14.1+cpu版本明確區分。"},
        "limitations": {"status": "pass", "claim_ids": ["c01", "c02", "c04", "c06", "c08", "c09", "c10", "c11", "c13", "c14", "c15"], "details": "零GPU、零模型/訓練資料下載、零完整recipe訓練、零公開模型新生成/新分數；沒有留新權重。只做tiny5520-param保存/人工梯度optimizer狀態交易、原gate/loader/CLI與既有measurement audit。1個失敗不估整體能力；storage不代表速度/運行memory；84-row舊選擇不代表新成品完成；同CPU状态恢复不保证完整stage或跨device轨迹。author canonical未知明确[]，未宣称已识别全部作者。"},
    },
}
(ROOT / "docs/technical-reviews/19.11.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": "docs/technical-reviews/19.11.json", "reviewer_task": TASK,
                  "claims": len(claims), "sources": len(sources), "artifacts": len(artifacts),
                  "source_sha256": report["source_sha256"], "report_sha256": digest(ROOT / "docs/technical-reviews/19.11.json")}, ensure_ascii=False))
