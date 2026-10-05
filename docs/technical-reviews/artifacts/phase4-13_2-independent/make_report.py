"""Write this reviewer's complete new canonical report; never read an earlier report."""
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT / "docs/review-tools"))
from section_facts import original_section

REVIEWER = "/root/phase4_factual_coordinator/factual_13_2"
body, whole, first_line = original_section(ROOT / "course/chapters/13.md", "13.2")
source_sha = hashlib.sha256(body).hexdigest()
assert body == (BASE / "section.md").read_bytes()
assert source_sha == "b03f6bd9b67b59934a77d69391056df215cd57ed21754b90776c5bbbdf5b83b9"
extraction = json.loads((BASE / "extraction.json").read_bytes())
original_env = json.loads((BASE / "environment.json").read_bytes())
cpu_env = json.loads((BASE / "cpu-checks-environment.json").read_bytes())
execution = json.loads((BASE / "bounded-execution.json").read_bytes())
assert all(item["exit_code"] == 0 for item in execution)
assert json.loads((BASE / "execution.json").read_bytes())["exit_code"] == 0
assert original_env["attempted_fences"] == [1] and not original_env["guard_events"]
variants = json.loads((BASE / "cpu-checks-result.json").read_bytes())
natural = json.loads((BASE / "ultrafeedback-verification-result.json").read_bytes())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

env = {"python": cpu_env["python"], "python_executable": cpu_env["python_executable"],
       "torch": cpu_env["torch"], "torch_git_version": cpu_env["torch_git_version"],
       "device": "cpu", "cuda_build": cpu_env["cuda_build"], "cuda_available": cpu_env["cuda_available"],
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
artifacts = []
def artifact(identifier, filename, kind, description, command=None, result=None):
    path = BASE / filename
    assert path.is_file() and not path.is_symlink()
    item = {"id": identifier, "path": path.relative_to(ROOT).as_posix(),
            "sha256": digest(path), "kind": kind, "description": description}
    if kind == "execution":
        item.update(command=command, result=result, environment=env)
    artifacts.append(item)

for identifier, filename, kind, description in [
    ("section", "section.md", "source_snapshot", "13.2 原始 UTF-8 小節 bytes；source_sha256 的對象。"),
    ("frozen-chapter", "frozen-input-chapter13.md", "source_snapshot", "初讀時 frozen input 全章快照；不是目前整章版本宣稱。"),
    ("extraction", "extraction.json", "source_snapshot", "實際小節範圍、fence SHA、空圖清單及 helper 身分。"),
    ("fence-code", "fence-1.py", "code", "未修改的原始 Python fence。"),
    ("bootstrap", "bootstrap.py", "code", "本次原 fence 真執行使用的 repo bootstrap。"),
    ("fence-env", "environment.json", "source_snapshot", "原 fence CPU 版本、裝置、guard 事件與依賴原檔 SHA。"),
    ("fence-stderr", "stderr.txt", "source_snapshot", "原 fence 的實際 stderr（空檔）。"),
    ("variants-code", "cpu_checks.py", "code", "獨立 token oracle、rejected=4、不同長度、mask、條件機率與重複 shift 檢查。"),
    ("variants-result", "cpu-checks-result.json", "derivation", "CPU 變體的真輸出與已宣告條件機率表；不代表模型成績。"),
    ("variants-env", "cpu-checks-environment.json", "source_snapshot", "本次 CPU tensor 檢查環境。"),
    ("variants-stderr", "cpu_checks.py.stderr.txt", "source_snapshot", "CPU 變體真 stderr（空檔）。"),
    ("natural-code", "verify_ultrafeedback.py", "code", "只執行原資料準備 AST 與 raw pointers；排除原 SFT/DPO 訓練語句。"),
    ("natural-result", "ultrafeedback-verification-result.json", "derivation", "100筆、切分 SHA、家族、byte 上限、截短計數及所讀 pointers。"),
    ("natural-stderr", "verify_ultrafeedback.py.stderr.txt", "source_snapshot", "原資料驗證真 stderr（空檔）。"),
    ("bounded-runner", "run_checks.py", "code", "以30秒上限保存每個真命令、exit code、stdout、stderr與CPU離線設定。"),
    ("paper-pdf", "sources/dpo-arxiv-2305.18290v3.pdf", "source_snapshot", "本人核對首頁版本及 §3/§4 的原始論文 PDF。"),
    ("paper-text", "sources/dpo-arxiv-2305.18290v3.txt", "source_snapshot", "pdftotext -layout 的原論文文字，含原頁碼與公式。"),
    ("dpo-trainers", "sources/author-dpo-trainers.py", "source_snapshot", "作者 repo 固定 commit 的原始 trainers.py。"),
    ("dpo-data", "sources/author-dpo-preference_datasets.py", "source_snapshot", "作者 repo 固定 commit 的原始 preference_datasets.py。"),
    ("dpo-head", "sources/dpo-author-repository-head.json", "source_snapshot", "解析 immutable upstream commit 的原 GitHub API 回應。"),
    ("natural-card", "sources/ultrafeedback-binarized-pinned-README.md", "source_snapshot", "本人 HTTPS 取回、與原存檔 SHA 相同的固定版本官方資料卡。"),
    ("fetch-code", "fetch_authority.py", "code", "真權威原檔取得程式，保留 TLS 檢查；未下載資料集或權重。"),
    ("fetch-log", "authority-fetch.json", "source_snapshot", "實際 HTTPS URL、狀態、日期、內容 SHA 與原檔位置。"),
    ("fetch-stdout", "authority-fetch.stdout.txt", "source_snapshot", "官方文件取得與 pinned card 相同比較的真 stdout。"),
    ("fetch-stderr", "authority-fetch.stderr.txt", "source_snapshot", "官方文件取得真 stderr（空檔）。"),
    ("original-behavior", "sources/experiment-original-behavior.py", "code", "原實驗 git revision 的完整原碼；未讀的額外解釋值原樣保留。"),
    ("original-text", "sources/experiment-original-text.py", "code", "原實驗資料 hash/截短/序列化程式，完整 SHA 吻合原測量。"),
    ("original-code-acquisition", "original-code-acquisition.json", "source_snapshot", "真 git show 命令、exit、stdout SHA，與保存原程式逐檔核對。"),
    ("raw-rows", "original-records/train-first-100.jsonl", "source_snapshot", "指定 UltraFeedback 原始100筆；不刪改原欄位。"),
    ("raw-viewer", "original-records/source-viewer-response.json", "source_snapshot", "原官方 rows 回應；檢查具名原始列與 provenance。"),
    ("raw-row-provenance", "original-records/rows.manifest.jsonl", "source_snapshot", "原 row index、revision、split、config provenance。"),
    ("raw-asset", "original-records/asset.json", "source_snapshot", "完整原 asset 檔；只讀原來源、revision、split與原檔 hash pointers。"),
    ("raw-api", "original-records/source-api.json", "source_snapshot", "保存的官方 API 回應；本人只核對 id/sha。"),
    ("raw-experiment", "original-records/original-dpo-result.json", "source_snapshot", "完整原測量；只使用具名原計數/hash/version/steps，不讀解釋值。"),
    ("raw-training-path", "original-training-path-pointers.json", "derivation", "原 pilot SFT 記錄80筆/80步，DPO80步；既有原測量，沒有重訓。"),
    ("recreated-train", "reconstructed-ultrafeedback/ultrafeedback-excerpts/train.jsonl", "derivation", "按原資料準備方法重建80筆；SHA精確等於原 train manifest。"),
    ("recreated-validation", "reconstructed-ultrafeedback/ultrafeedback-excerpts/validation.jsonl", "derivation", "按原方法重建10筆；SHA精確等於原 validation manifest。"),
    ("recreated-test", "reconstructed-ultrafeedback/ultrafeedback-excerpts/test.jsonl", "derivation", "按原方法重建10筆；SHA精確等於原 test manifest。"),
    ("recreated-manifest", "reconstructed-ultrafeedback/ultrafeedback-excerpts/manifest.json", "derivation", "原 _save_splits 產生的本次資料分母/hash，不含模型評測。"),
    ("inspected-data", "sources/inspected-data.py", "source_snapshot", "本人真正讀取的 ByteTokenizer/render_chat/pad_batch 計算區段。"),
    ("inspected-alignment", "sources/inspected-alignment.py", "source_snapshot", "本人真正讀取的 sequence_log_probability 計算區段。"),
    ("inspected-common", "sources/inspected-common.py", "source_snapshot", "本人真正讀取的 split_records/write_json 原方法區段。"),
    ("inspected-code-locators", "inspected-code-locators.json", "source_snapshot", "窄計算快照範圍與完整原碼 SHA。"),
    ("personal-inspection", "personal-source-inspection.md", "derivation", "本人來源定位、版本、支持範圍、实际閱讀範圍及限制。"),
    ("report-builder", "make_report.py", "code", "此 reviewer 的完整新報告生成器；不讀舊 canonical 報告。"),
]:
    artifact(identifier, filename, kind, description)
artifact("fence-run", "execution.json", "execution", "原 fence 的真 bounded worker 執行紀錄。",
         json.loads((BASE / "execution.json").read_bytes())["command"], "exit 0；1/1 原 Python fence 真執行，guard_events=[]。")
artifact("fence-stdout", "stdout.txt", "execution", "原 fence 真輸入/目標序列。",
         json.loads((BASE / "execution.json").read_bytes())["command"], "chosen/rejected 兩行正確輸出，exit 0。")
artifact("bounded-run", "bounded-execution.json", "execution", "兩個短 CPU 檢查的真 argv/exit/環境/耗時。",
         ".venv/bin/python docs/technical-reviews/artifacts/phase4-13_2-independent/run_checks.py", "2/2 子命令 exit 0；timeout 各30秒，無訓練或模型評測。")
artifact("variants-run", "cpu_checks.py.stdout.txt", "execution", "CPU oracle 與變體真 stdout。",
         " ".join(execution[0]["command_argv"]), "exit 0；數列、mask、不等長、rejected=4與條件前文反例全部 assertion 成功。")
artifact("natural-run", "verify_ultrafeedback.py.stdout.txt", "execution", "原資料準備與原測量核對真 stdout。",
         " ".join(execution[1]["command_argv"]), "exit 0；100筆、80/10/10、0家族重疊、三組原SHA精確一致、每欄≤120bytes。")

sources = [
    {"id": "dpo-paper", "kind": "paper", "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
     "url": "https://arxiv.org/pdf/2305.18290v3", "version": "arXiv:2305.18290v3, 29 July 2024", "accessed_on": "2026-10-05",
     "authority_reason": "方法作者的原始論文，首頁作者、標題與 arXiv 版本本人核對。", "verified": True, "checked_original": True,
     "inspection_note": "本人讀 PDF 首頁及 §3 p3 共同提問/偏好樣本/Eq1、§4 p4 Eq7；原 PDF/text 保存在 paper-pdf/paper-text。來源 index 只定位原檔。"},
    {"id": "dpo-trainer-source", "kind": "official_source", "title": "DPO authors' trainers.py",
     "url": "https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py",
     "version": "commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f", "accessed_on": "2026-10-05",
     "authority_reason": "DPO 作者 Eric Mitchell 維護的方法原始實作；固定 commit 而非搜尋摘要。", "verified": True, "checked_original": True,
     "inspection_note": "本人親讀 _get_batch_logps90–115、concatenated_inputs118–142、BasicTrainer.concatenated_forward210–220。原 source 的 labels 未先shift；本repo render_chat 已shift，明確區分契約。"},
    {"id": "dpo-data-source", "kind": "official_source", "title": "DPO authors' preference_datasets.py",
     "url": "https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/preference_datasets.py",
     "version": "commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f", "accessed_on": "2026-10-05",
     "authority_reason": "原 DPO 作者實作的偏好樣本準備 API。", "verified": True, "checked_original": True,
     "inspection_note": "本人親讀 tokenize_batch_element214–277：chosen/rejected 各自加共同prompt及EOS，各自labels複製與prompt遮罩。"},
    {"id": "ultrafeedback-card", "kind": "official_docs", "title": "HuggingFaceH4 UltraFeedback Binarized official dataset card",
     "url": "https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/resolve/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85/README.md",
     "version": "dataset revision 3949bf5f8c17c394422ccfab0c31ea9c20bdeb85", "accessed_on": "2026-10-05",
     "authority_reason": "原資料預處理者 HuggingFaceH4 发布的固定版本資料卡；HTTPS取回內容SHA與原資產一致。", "verified": True, "checked_original": True,
     "inspection_note": "本人讀 YAML schema/splits、Dataset Description、Data Splits與record schema：GPT-4評完整completion；chosen為highest overall_score，rejected取其餘；train_prefs/test_prefs為官方split。"},
]
for identifier, path, version, inspected in [
    ("repo-chat", ROOT / "tiny_perceptron/data.py", "current source SHA; also identical to original experiment data.py", "AST先定位；本人讀3–28、46–86：ByteTokenizer、shifted、render_chat、pad_batch。"),
    ("repo-logps", ROOT / "tiny_perceptron/alignment.py", "current source SHA", "AST先定位；本人只讀sequence_log_probability29–35：不再shift、只累加有效labels。"),
    ("repo-split", ROOT / "scripts/course_experiments/common.py", "same source SHA as original experiment revision 8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d", "本人讀write_json33–38與split_records50–70，原始完整SHA吻合原测量。"),
    ("original-pilot", BASE / "sources/experiment-original-behavior.py", "git revision 8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d; SHA matches raw code_sha256", "AST先定位；本人讀_pair_examples552–562、_natural_dpo_pilot655–685，排除返回scope解釋；原資料計算語句656–678真執行，訓練語句不執行。"),
    ("original-text-method", BASE / "sources/experiment-original-text.py", "git revision 8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d; SHA matches raw code_sha256", "本人讀_json_bytes/_digest/_save_splits39–62、_asset_rows75–88、_utf8_prefix374–375；必要函式用AST取出真執行。"),
]:
    sources.append({"id": identifier, "kind": "repository_code", "title": path.name + " personally inspected original contract",
                    "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "version": version,
                    "verified": True, "inspection_note": inspected})
for identifier, artifact_id, title in [("executed-fence", "fence-run", "Actual original 13.2 Python fence CPU run"),
                                      ("executed-variants", "variants-run", "Independent bounded CPU calculations"),
                                      ("executed-natural", "natural-run", "Original raw-data and data-preparation verification")]:
    sources.append({"id": identifier, "kind": "execution", "title": title, "artifact_id": artifact_id, "verified": True})
sources += [
    {"id": "byte-calculation", "kind": "derivation", "title": "Independent arithmetic and UTF-8 byte IDs", "verified": True,
     "details": "1+1=2；ASCII/UTF-8 '1','+','=','?','2','3','4' bytes為49,43,61,63,50,51,52，+8得57,51,69,71,58,59,60；CPU獨立oracle核對完整sequence。"},
    {"id": "length-bound", "kind": "derivation", "title": "Pilot input length and truncation scope", "verified": True,
     "details": "每欄≤120 bytes時，完整對話token stream長≤1bos+1user+120prompt+1eos+1assistant+120answer+1eos=245；render_chat x刪最後位置所以≤244，低於原pilot max_length256。UTF-8 decode(ignore)移去不完整末codepoint；不保證grapheme或語義資訊完整。"},
]

def evidence(source_id, locator, supports):
    return {"source_id": source_id, "locator": locator, "supports": supports}
claims = [
    {"id": "preference-fields", "kind": "concept", "statement": "prompt 是共同前文，chosen/rejected 是這次同題的偏好標記，並非模型自行發現的真值。",
     "location": "course/chapters/13.md:37", "scope": "開場1+1算例由人指定2較佳、3較差；不推廣成UltraFeedback必然由人類評分或chosen必然真確。", "status": "verified",
     "evidence": [evidence("dpo-paper", "§3 p3, yw≻yl|x and Eq(1), preference dataset D", "同一x有兩個外部偏好標記completion；偏好不是絕對真值。"),
                  evidence("byte-calculation", "1+1=2", "本算例選2而非3有明確算術判準。"),
                  evidence("ultrafeedback-card", "Dataset Description; chosen/rejected schema", "外部資料標籤由完整回答的GPT-4評分及binarization生成，與小算例人工標記分清。")],
     "artifact_ids": ["section", "paper-pdf", "natural-card", "personal-inspection"]},
    {"id": "byte-ids-and-exercise", "kind": "software", "statement": "預設ByteTokenizer為UTF-8每byte加8，邊界IDs1/3/4/2；兩份輸入精確如文，rejected改4後普通ID60且chosen不变。",
     "location": "course/chapters/13.md:42–55,61", "scope": "本repo預設render_chat/ByteTokenizer的具體契約，UTF-8一個字元可佔多個byte/token。", "status": "verified",
     "evidence": [evidence("repo-chat", "ByteTokenizer14–28; render_chat54–68", "預設tok及角色/boundary IDs、UTF-8+8計算。"),
                  evidence("byte-calculation", "listed byte values and +8", "數字2/3/4編號58/59/60；完整提問bytes手算。"),
                  evidence("executed-fence", "stdout.txt, both lines", "原fence真印出chosen與rejected正確輸入。"),
                  evidence("executed-variants", "cpu_checks.py oracle(), pairs for 2/3/4/300/四", "獨立byte構造與變體核對，chosen sequence保持不變。")],
     "artifact_ids": ["fence-code", "fence-run", "fence-stdout", "variants-code", "variants-run", "variants-result", "inspected-data"],
     "verification": {"method": "executed", "expected": "chosen x=[1,3,57,51,57,69,71,2,4,58]；rejected末項59；改4末項60。",
                      "observed": "原fence exit0；CPU五種答案全部精確等於獨立oracle，rejected=4末項60。", "details": "原碼未改執行；CPU整數比較精確相等，UTF-8多byte變體同樣核對。"}},
    {"id": "single-shift-and-mask", "kind": "software", "statement": "兩份y前八項-100，最後是答案58/59與EOS2；索引8助手預測答案，最後數字預測EOS，render_chat已做下一位置對齊，不能再shift。",
     "location": "course/chapters/13.md:57", "scope": "-100忽略計分/損失位置，前文仍在輸入中；此repo labels 已shift，不能套用上游未shift labels的API再shift。", "status": "verified",
     "evidence": [evidence("repo-chat", "render_chat54–68; return ids[:-1], targets[1:]", "已對齊一次，assistant內容及EOS是有效目標。"),
                  evidence("repo-logps", "sequence_log_probability29–35", "labels!=-100遮罩、gather後sum，不再shift。"),
                  evidence("dpo-trainer-source", "_get_batch_logps90–115", "上游原labels在score helper內才shift，-100忽略；與本repo前移契約明確不同。"),
                  evidence("executed-fence", "stdout.txt target lists", "兩組真目標前八項-100，末兩項58或59、2。"),
                  evidence("executed-variants", "valid_indices, ignored_positions_invariant, extra_shift_wrong_assistant_target", "有效索引8/9；改忽略位置logits不改分；第二次shift會把EOS錯放助手位置。")],
     "artifact_ids": ["fence-run", "fence-stdout", "variants-code", "variants-run", "variants-result", "inspected-alignment", "dpo-trainers"],
     "verification": {"method": "executed", "expected": "y=[-100]*8+[58或59,2]，有效indices8/9；再shift不保持正確位置。",
                      "observed": "原fence精確吻合；CPU有效indices8/9，忽略位置變化分數相等；再shift使索引8的target錯變2。", "details": "整數序列精確比較；手工條件表用float64，log分數容忍差1e-12。"}},
    {"id": "candidate-own-prefix", "kind": "concept", "statement": "每份答案須用其自己的token前文計機率；chosen輸入不能配rejected目標，不等長也各自對齊；換掉rejected的提問會失去同題比較。",
     "location": "course/chapters/13.md:39,59,61", "scope": "條件機率和配對資料契約，不宣稱任意數值差均可識別題目難度；CPU反例為宣告概率表，不是模型成績。", "status": "verified",
     "evidence": [evidence("dpo-paper", "§3 conditional yw≻yl|x; §4 Eq(7)", "同x的兩份completion各有π(y|x)，更換x不再是同一條件下的偏好比較。"),
                  evidence("dpo-data-source", "tokenize_batch_element214–277", "同prompt分別prepends於各自chosen/rejected，各自完整input和labels。"),
                  evidence("dpo-trainer-source", "concatenated_inputs118–142; concatenated_forward210–220", "各自input/labels對齊，不等長pad後才合batch，EOS沿各自prefix預測。"),
                  evidence("executed-variants", "conditional_logits; padding_shapes; own/crossed scores", "宣告p(3|assistant)=.3、p(EOS|3)=.8、p(EOS|2)=.2，正確pair概率.24，交叉input得到.06；不等長[2,12]批次成功。")],
     "artifact_ids": ["paper-pdf", "dpo-data", "dpo-trainers", "variants-code", "variants-run", "variants-result"]},
    {"id": "natural-source-and-splits", "kind": "empirical", "statement": "UltraFeedback小包100筆按提問家族自行切成80/10/10，並非官方測試集。",
     "location": "course/chapters/13.md:66", "scope": "指定dataset revision的第一100筆train_prefs與指定原實驗；只重建原資料準備，不重訓或評測原模型。", "status": "verified",
     "evidence": [evidence("ultrafeedback-card", "YAML dataset_info/splits; Data Splits", "官方train_prefs與test_prefs各自存在，repo100筆來自train_prefs，後來10筆test是自切。"),
                  evidence("original-pilot", "_natural_dpo_pilot655–679", "source_asset到prompt_id家族，對全部raw筆資料先準備再split_records。"),
                  evidence("repo-split", "split_records50–70", "seed42按完整家族切80%/90%邊界，不按答案分拆。"),
                  evidence("executed-natural", "raw source rows/viewer; original result /results/ultrafeedback_pilot/source_records and /data/{train,validation,test}/{records,families,sha256,path}", "真核对100筆/100家族、80/10/10、無跨組，三份序列化SHA精確等於原測量。")],
     "artifact_ids": ["natural-code", "natural-run", "natural-result", "raw-rows", "raw-viewer", "raw-row-provenance", "raw-asset", "raw-api", "raw-experiment", "original-behavior", "original-code-acquisition", "recreated-train", "recreated-validation", "recreated-test", "recreated-manifest"],
     "verification": {"method": "executed", "expected": "100筆原始train_prefs，自切train80/validation10/test10，家族不跨組，資料版本對得起原測量。",
                      "observed": "100原列與viewer逐筆相同；100家族；80/10/10；家族重疊0；三組sha與原manifest全部精確一致。",
                      "details": "原行为/text git show SHA吻合code_sha256；執行原準備AST656–678和原split/serialization，沒執行fit或載權重。",
                      "denominators": {"source_records": 100, "source_families": 100, "train_records": 80, "validation_records": 10, "test_records": 10, "family_overlap": 0}}},
    {"id": "utf8-excerpt-contract", "kind": "software", "statement": "提問和两篇回答各截至最多120 UTF-8 bytes，保留完整UTF-8字元，使此ByteTokenizer資料可容納於小pilot。",
     "location": "course/chapters/13.md:66", "scope": "完整UTF-8 codepoint，不保證grapheme/完整推理或標籤有效；原pilot max_length256。", "status": "verified",
     "evidence": [evidence("original-text-method", "_utf8_prefix374–375", "encode[:120].decode(errors='ignore')保留可完整解碼的byte前綴。"),
                  evidence("original-pilot", "_natural_dpo_pilot655–685", "prompt及各side最後assistant答案分別截短120，max_length256。"),
                  evidence("executed-natural", "fields; multibyte boundary assertions", "300個excerpt每欄≤120；中/emoji/é邊界測試成功；prompt/chosen/rejected分別68/88/90筆被截短。"),
                  evidence("length-bound", "245 stream tokens, 244 shifted x tokens", "有界byte列加邊界後，input必≤244<256，支持小pilot容納。")],
     "artifact_ids": ["natural-code", "natural-run", "natural-result", "original-text", "original-behavior", "personal-inspection"],
     "verification": {"method": "executed", "expected": "100組×3欄全部≤120 UTF-8 bytes，邊界不留下不完整codepoint，x最多244。",
                      "observed": "三欄excerpt_max_bytes均120；68/88/90筆截短；119 A+你、118 A+emoji、é×61三個邊界預測全吻合。",
                      "details": "對照原方法完整資料準備而非重新讀模型；input長度上限從render_chat邊界計數推導，整數精確。"}},
    {"id": "excerpt-label-and-quality-limits", "kind": "concept", "statement": "截短可能刪掉原評分者看到的關鍵理由，來源chosen不保證截短後仍適用；這個小包只作資料/訓練通路支線，不支持自然聊天品質。",
     "location": "course/chapters/13.md:66", "scope": "信息損失與證據適用範圍；不判定特定一列label已逆轉，也不拿低loss或已跑步數冒充聊天品質。", "status": "verified",
     "evidence": [evidence("ultrafeedback-card", "Dataset Description, GPT-4 scores full model completions and overall_score selection", "chosen標記根據原完整completion的評分，未提供120byte excerpt重新評分。"),
                  evidence("original-text-method", "_utf8_prefix374–375", "prefix裁切有實際資訊丟失途徑。"),
                  evidence("executed-natural", "fields.truncated_records and raw counts", "真有68/88/90筆各欄內容被裁切；重建只驗證資料通路/原測量版本，沒有新聊天驗收。"),
                  evidence("dpo-paper", "§3 preference dataset conditional on full y; Eq(1)", "偏好樣本的標籤針對被比較的回答；改寫/截掉y不能自動保證原相對標籤仍適用。")],
     "artifact_ids": ["natural-card", "paper-pdf", "natural-result", "natural-run", "raw-training-path", "personal-inspection"]},
]
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "13.2", "source": "course/chapters/13.md#13.2",
    "source_sha256": source_sha, "verdict": "pass", "reviewer_task": REVIEWER, "reviewer_context": "fresh",
    "reviewed_on": "2026-10-05", "figure_sha256": {}, "artifacts": artifacts, "sources": sources, "claims": claims, "issues": [],
    "read_scope": {"section": f"course/chapters/13.md:{first_line}–{first_line + len(body.splitlines()) - 1}",
                   "prerequisite": "13.1 原節，lines7–34", "chapter_intro": "not required: 13.2 is not the first section",
                   "previous_reviews_read": False, "author_result_interpretations_read": False,
                   "details_artifact_id": "personal-inspection"},
    "frozen_input": {"meaning": "complete Markdown bytes saved at first extraction; not a current whole-chapter fingerprint",
                     "artifact_id": "frozen-chapter", "sha256": digest(BASE / "frozen-input-chapter13.md")},
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "7組實質主張核對共同prompt、外部標記、byte IDs、mask/shift、各自前文、原小包和截短限制；未發現需修訂的實質錯誤。", "claim_ids": [c["id"] for c in claims]},
        "numeric_verification": {"status": "pass", "details": "原fence和變體精確核對IDs58/59/60及有效位置8/9；原100筆→80/10/10及家族/原SHA真計算吻合；每欄120byte；手工條件表log差為log4，float64誤差≤1e-12。", "claim_ids": ["byte-ids-and-exercise", "single-shift-and-mask", "candidate-own-prefix", "natural-source-and-splits", "utf8-excerpt-contract"]},
        "figure_consistency": {"status": "not_applicable", "details": "原節沒有圖/SVG引用，extraction.svg_references=[]；文字已列完整輸入、目標與索引，此事實審核不需判讀畫面。沒有宣稱已做不存在的圖render。", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "本人讀原DPO v3與作者固定commit兩個API、固定revision HF資料卡，HTTPS card SHA與原資產一致；原Git實作SHA吻合原測量，raw pointers與原資料逐項核對。installed CPU torch2.14.1+cpu；原既有記錄torch2.14.1+cu126/cuda，沒有將CPU重建說成重訓。", "claim_ids": [c["id"] for c in claims]},
        "limitations": {"status": "pass", "details": "區分手工/CPU機制與訓練品質；相同提問限制、UTF-8 codepoint與語义完整、完整回答偏好與截短後標記均有明確scope。未載入既有模型、下載訓練資料、重訓、GPU或自然聊天重評。", "claim_ids": ["preference-fields", "candidate-own-prefix", "natural-source-and-splits", "utf8-excerpt-contract", "excerpt-label-and-quality-limits"]},
    },
    "limitations": ["No new training, GPU run, existing-model inference or natural-chat evaluation.",
                    "UTF-8 codepoint preservation does not certify grapheme clusters, reasoning completeness or cropped preference labels.",
                    "Original training steps are raw existing-record provenance, not a new execution or quality result.",
                    "No figure is referenced; no visual rendering or viewport check is claimed."],
}
destination = ROOT / "docs/technical-reviews/13.2.json"
temporary = BASE / "new-report.json"
temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
temporary.replace(destination)
# Confirm the canonical file is this newly generated reviewer's report before any checker invocation.
written = json.loads(destination.read_bytes())
assert written["reviewer_task"] == REVIEWER and written["reviewer_context"] == "fresh"
assert written["source_sha256"] == source_sha and written["lesson_id"] == "13.2" and written["verdict"] == "pass"
result = {"canonical_path": destination.relative_to(ROOT).as_posix(), "canonical_reviewer_task": written["reviewer_task"],
          "source_sha256": source_sha, "report_sha256": digest(destination), "claims": len(claims), "artifacts": len(artifacts),
          "verdict": written["verdict"], "canonical_identity_confirmed": True}
(BASE / "canonical-generation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
