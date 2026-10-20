"""將已核對的原始判斷寫入新檔；拒絕覆寫既有正式報告。"""

import hashlib
import json
from collections import Counter
from pathlib import Path

AREA = Path(__file__).resolve().parent
ROOT = AREA.parents[3]
REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_19-19/technical-19-19-original-01.json"
EXEC = json.loads((AREA / "verification.json").read_bytes())
SUPP = json.loads((AREA / "supplement-verification.json").read_bytes())
HTTP = json.loads((AREA / "http-verification.json").read_bytes())
ORIGINALS = json.loads((AREA / "official-source-excerpts.json").read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


sources = []
repo_specs = [
    ("release", "tiny_perceptron/lesson_products/release.json", "repo、revision、products.prefix 與 manifest_sha256；兩項公開 manifest 與本次下載吻合。"),
    ("main", "tiny_perceptron/lesson_products/__main__.py", "main 的子命令、預設 CPU、目的目錄、question、loops、effort、port 與 Product 載入分支；實際執行 CLI。"),
    ("delivery", "tiny_perceptron/lesson_products/delivery.py", "download_product 的固定 revision、manifest 與逐檔驗證；serve 的 127.0.0.1 綁定與三個 POST 入口。"),
    ("inference", "tiny_perceptron/lesson_products/inference.py", "check_manifest、Product.__init__、solve、judge、EFFORT、auto_solve；核對檢查順序、實際讀出及輸出欄位。"),
    ("model", "tiny_perceptron/lesson_products/model.py", "Config、Tokenizer、LoopedSolver、CandidateScorer、validate_candidates；模型用 PyTorch 建構，輸入與輪數有上限。"),
    ("data", "tiny_perceptron/lesson_products/data.py", "answer 的 swap、REQUIRED、judgment、CANDIDATES、MEANINGS、QUESTIONS；核對交換與缺欄標籤。"),
    ("training", "tiny_perceptron/lesson_products/training.py", "checkpoint 保存 model、optimizer、sampler_rng、torch_rng、cuda_rng；restore 逐項恢復，並以小模型保存／還原查核。"),
    ("train", "tiny_perceptron/lesson_products/train.py", "train 的隨機建構、seed、資料指紋、policy 選擇、model.pt 匯出及 training-receipt；不載入外部預訓練編碼器。"),
    ("evaluation", "tiny_perceptron/lesson_products/evaluation.py", "freeze_settings 以 b_logits 取得各輪結果，auto_quality 綜合 C 路由、B 正確性與實際輪數，再選凍結 Auto 規則。"),
    ("local", "scripts/lesson_products/train_local.py", "main 的 cpu/mps/cuda 選擇、新目錄守門、配方驗證與 train 呼叫；只執行 help 與既有輸出目錄拒絕路徑。"),
    ("export", "scripts/selftrained/train.py", "export_inference 只從 checkpoint[model] 取得張量寫 model.safetensors；optimizer、RNG 與 sampler 留在訓練檢查點。"),
    ("moe", "tiny_perceptron/selftrained/inference.py", "InferenceAssistant 驗證 manifest、from-scratch metadata、完整模型鍵／形狀／dtype，再 load_state_dict 與 eval。"),
    ("chat", "scripts/selftrained/chat.py", "MoE 公開 CPU 推論 CLI；本次只執行 --help，未重新下載或生成 MoE 回答。"),
    ("readme", "docs/lesson-products/README.md", "固定 HF 版本、候選語義 ID、三種 typed question、本機 JSON API、重訓與新輸出目錄指引。"),
    ("ignore", ".gitignore", "data、checkpoints、outputs 與權重副檔名的忽略規則；git ls-files 確認程式與生成器被追蹤。"),
    ("warmup", "course/first-steps.md", "第一次開啟教材的 CPU 安裝、專案根目錄與 .venv 啟用；以現有 .venv 執行本次命令，未重新安裝依賴。"),
    ("publish", "scripts/lesson_products/publish.py", "publish_remote 以私密環境變數取得登入資料，receipt 僅記公開 revision、逐檔 SHA／大小及狀態，錯誤先經 redact_error。未執行發布。"),
    ("publication", "scripts/lesson_products/publication.py", "redact_error 遮蔽具秘密類型名稱的環境值與 URL；以合成資料執行，不列印真實秘密。"),
    ("pubreceipt", "docs/lesson-products/results/publication-v1.json", "既有 completed 發布收據的 public revision 與 anonymous_downloads；本次以新下載逐檔對照 B/C 的大小及 SHA，沒有沿用其能力分數。"),
]
for key, path, note in repo_specs:
    sources.append({"id": "s_" + key, "kind": "repository_code", "title": path, "path": path,
                    "sha256": sha(ROOT / path), "version": SUPP["commit"] + "；本回合工作樹內容以 SHA-256 固定",
                    "verified": True, "inspection_note": note})

notes = {
    "serialization": "實讀 Serialized file format for torch.save、torch.load with weights_only=True、weights_only security、allowlist。支持 ZIP64／pickle 張量格式與受限反序列化；沒有將它宣稱為全面防止惡意檔案的保證。另對照本機 PyTorch 2.14.1 的 torch.load docstring 與實際純張量參數包。",
    "eval": "實讀 Module.eval 與 Module.train：eval 等同 train(False)，只影響特定模組，沒有更改使用者的抽樣程序。另讀本機 2.14.1 docstring，執行 Dropout 與 multinomial 短例。",
    "randomness": "實讀 Reproducibility 開頭、PyTorch random number generator 與跨平台限制；seed、版本及硬體都與重現條件有關，不保證跨裝置位元相同。",
    "mps": "實讀 MPS backend 開頭與裝置建立：macOS GPU 透過 Metal Performance Shaders Graph 與 kernels 運算；不由本次 CPU 結果推出 MPS 正式訓練已驗證。",
    "checkpoint": "實讀 What is a state_dict、Save/Load state_dict 及 Saving & Loading a General Checkpoint for Inference and/or Resuming Training；推論參數與 optimizer state 分開。頁面標示教程 2.14.0+cu130，當日 HTML 指紋記在來源收據。",
    "hf": "實讀 hf_hub_download 的特定 revision 範例與 snapshot_download；revision 可為 branch、tag 或 commit hash，且應使用完整 commit hash。這只支持 HF 版本定位；本專案逐檔 SHA 驗證另核對自己的 delivery 實作及新下載。",
    "http": "實讀 RFC 9110 的 Introduction 與 HTTP Architecture：HTTP 是應用層 request/response 協定。此節對本機網頁／程式請求的簡化說明成立，不把 HTTP 限於瀏覽器。",
}
for original in ORIGINALS:
    assert original["exit_code"] == 0 and original["excerpts"]
    key = original["id"]
    sources.append({"id": "s_" + key + "_official", "kind": "official_docs", "title": key + " 官方原文",
                    "url": original["url"], "version": original["version"] + "；HTML SHA-256 " + original["original_sha256"],
                    "verified": True, "checked_original": True, "accessed_on": "2026-10-20",
                    "authority_reason": "由 PyTorch／Hugging Face 官方或 RFC Editor 維護，分別定義其 API、版本定位與 HTTP 協定。",
                    "inspection_note": notes[key]})

artifacts = []


def artifact(identifier, filename, kind, description, execution=None):
    value = {"id": identifier, "kind": kind, "path": (AREA / filename).relative_to(ROOT).as_posix(),
             "sha256": sha(AREA / filename), "description": description}
    if execution is not None:
        value.update(command=execution["command"], result=execution["result"], environment=execution["environment"])
    artifacts.append(value)


artifact("a_probe_code", "probe.py", "code", "匿名下載、封鎖網路的 CLI、參數包、manifest 篡改與小模型 checkpoint 查核程式。")
artifact("a_execution", "verification.json", "execution", "本次固定 HF 版本的下載及 CPU 操作；包含首次 8765 埠占用的原始錯誤。", EXEC)
artifact("a_http_code", "http_probe.py", "code", "改用空閒埠驗證同一 HTTP 服務；不接觸已在運作的服務。")
artifact("a_http", "http-verification.json", "execution", "本機空閒埠的 health 與 solve／judge／auto 真實 HTTP 請求回應。", HTTP)
artifact("a_supplement_code", "supplement.py", "code", "CLI help、新輸出目錄拒絕、Git 追蹤、五個固定診斷題與第一個錯誤見證搜尋。")
artifact("a_supplement", "supplement-verification.json", "execution", "CPU CLI 及保存契約查核；有限輸入錯誤見證支持原文能力限制。",
         {"command": SUPP["command"], "result": "所有 help exit code 0；既有輸出目錄 exit code 2 且保持空目錄；找到 0+1+6 被答為 12 的見證。", "environment": SUPP["environment"]})
artifact("a_originals_code", "read_originals.py", "code", "curl 讀原始來源、計算原始 HTML SHA，僅保存短摘錄與讀取定位。")
artifact("a_originals", "official-source-excerpts.json", "source_snapshot", "七份官方原文的 HTTPS URL、實讀版本、日期、curl 命令、全文 SHA 與短摘錄；沒有保存第三方全文。")
publication_command = """PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python - <<'PY' > docs/technical-reviews/artifacts/review-20261019-19-19/publication-verification.json
import hashlib,json,os,platform
from pathlib import Path
from unittest.mock import patch
from scripts.lesson_products.publication import redact_error
area=Path('docs/technical-reviews/artifacts/review-20261019-19-19')
pubpath=Path('docs/lesson-products/results/publication-v1.json')
pub=json.loads(pubpath.read_bytes())['execution']
run=json.loads((area/'verification.json').read_bytes())
expected={x['path']:x for x in pub['anonymous_downloads']}
comparisons=[]
for product,record in run['models'].items():
 for name,metadata in record['files'].items():
  before=expected[product+'/'+name]
  comparisons.append({'path':product+'/'+name,'sha256_equal':before['sha256']==metadata['sha256'],'bytes_equal':before['bytes']==metadata['bytes']})
with patch.dict(os.environ,{'REVIEW_PROBE_TOKEN':'synthetic_value'},clear=True):
 redacted=redact_error(ValueError('synthetic_value https://example.invalid/private?credential=synthetic_value'))
result={'environment':{'python':platform.python_version(),'device':'cpu'},'publication_receipt':{'path':str(pubpath),'sha256':hashlib.sha256(pubpath.read_bytes()).hexdigest(),'status':pub['status'],'public':pub['public']},'comparisons_to_fresh_downloads':comparisons,'redaction_probe':redacted,'scope':'只讀既有公開發布收據與本次下載指紋，另以合成字串驗證遮蔽函式；沒有上傳、讀取金鑰值或啟動遠端工作。'}
print(json.dumps(result,ensure_ascii=False,indent=2))
PY"""
pub = json.loads((AREA / "publication-verification.json").read_bytes())
artifact("a_publication", "publication-verification.json", "execution", "將既有上傳後公開驗證指紋與本次下載的十二檔逐一比對，並執行合成秘密／URL 遮蔽。",
         {"command": publication_command, "result": "十二檔 SHA 與大小均一致；合成輸入被遮蔽為 [REDACTED] [URL]。", "environment": pub["environment"]})
artifact("a_report_code", "build_report.py", "code", "整理原始技術判斷與證據引用；以 exclusive create 防止覆寫正式報告。")

claims = []


def evidence(source, locator, supports):
    return {"source_id": "s_" + source, "locator": locator, "supports": supports}


def claim(kind, statement, location, scope, refs, observed=None, expected=None, aids=None, denominators=None):
    value = {"id": f"c{len(claims)+1:02}", "kind": kind, "statement": statement, "location": location,
             "status": "verified", "evidence": refs, "artifact_ids": aids or [], "scope": scope}
    if kind in ("numeric", "software", "empirical"):
        value["verification"] = {"method": "executed", "expected": expected or statement,
                                 "observed": observed, "details": "核對所引程式契約，並讀取 artifact 中的本回合命令、輸入、輸出與環境；未將既有作者實驗視為本次重跑。"}
        if kind == "numeric":
            value["verification"]["tolerance"] = "整數與類別精確相等；不比較執行時間。"
        if denominators:
            value["verification"]["denominators"] = denominators
    claims.append(value)


claim("concept", "同名模型再次訓練後，輸出及適合的 Auto 規則可能改變。", "開頭「同名模型經過再次訓練」", "可能改變，不是每次重訓必然改變；Auto 依指定版本的實際表現選規則。",
      [evidence("randomness_official", "Reproducibility；PyTorch random number generator", "訓練亂數、版本及平台會影響重現條件。"), evidence("evaluation", "freeze_settings；auto_quality", "Auto 規則直接依 B 各深度與 C 的預測選擇，換權重後不能假定同一取捨。")], aids=["a_originals", "a_execution"])
claim("concept", "可使用的版本應共同鎖定權重、架構設定、tokenizer 與決策規則；網址本身不識別檔案內容。", "開頭「權重、網路設定、tokenizer及決策規則」及末段「整個決策流程」", "鎖定可信 release 與內容指紋；SHA 比對不等同來源身分或一般攻擊安全性的保證。",
      [evidence("hf_official", "Download a single file；revision", "同一 repo 可取不同 revision；完整 commit hash 固定版本。"), evidence("release", "products；manifest_sha256", "各成品 manifest 與固定 HF revision 綁定。"), evidence("inference", "check_manifest；Product.__init__", "必須共同驗證模型、設定、tokenizer 與 policy／calibration。")], aids=["a_originals", "a_execution"])
claim("software", "release.json 固定 HF commit、成品目錄 prefix 與各 manifest 指紋。", "「新成品用兩層清單配對版本」", "本次公開 B／C 固定成品。", [evidence("release", "repo；revision；products", "記錄完整 commit、prefix 與 manifest SHA。")], "兩次匿名下載回報 revision=2c6a1062f3e66a602cf83eda6c703456ec18bca0，manifest 分別與 release 相符。", aids=["a_execution"])
claim("software", "各 artifact manifest 列出推論相關檔案大小與 SHA-256，載入前會核對。", "「各成品的 artifact manifest 再列」", "拒絕不相符檔案；不聲稱自行改寫 manifest 後仍能驗證發布者身分。", [evidence("inference", "check_manifest", "逐檔核對 SHA、bytes、必要檔案與路徑。"), evidence("delivery", "download_product", "先驗證 manifest 指紋再下載它列出的檔案。")], "B／C 各五項 manifest 成員全數通過；改動 config.json 一個空白即拒絕：成品檔案指紋不符。", aids=["a_execution"])
claim("software", "推論入口先驗證成品，再以對應設定重建模型與載入權重，實際模型處於評估模式。", "「推論時先核對，再建立網路」", "實作在建構後先 eval 再 load_state_dict，載入不改 training 旗標；因此推論時的狀態與原文敘述一致。", [evidence("inference", "Product.__init__", "check_manifest 在網路建構與 torch.load 前，並呼叫 eval。")], "B、C 的所有 module.training 都為 false，純張量參數成功載入；破壞設定時先於建網路被拒。", aids=["a_execution"])
claim("concept", "評估模式關閉某些訓練行為，但不會自動使抽樣輸出固定。", "「評估模式關閉某些訓練行為」", "eval 影響 Dropout／BatchNorm 等；仍需另管 RNG 與讀出策略。本次 B/C 本身用 argmax，沒有宣稱它們會抽樣。", [evidence("eval_official", "Module.eval；Module.train", "eval 等同 train(False)，只影響特定模組。"), evidence("randomness_official", "PyTorch random number generator", "顯式抽樣仍使用 RNG。")], aids=["a_originals", "a_execution"])
claim("concept", "讀出方式、輪數及裝置屬於重現結果時應記錄的條件。", "「讀出方式、輪數及裝置仍須一起記」", "記錄條件仍不保證不同版本、裝置得到位元相同結果。", [evidence("randomness_official", "Reproducibility", "CPU/GPU 與版本之間不保證完全重現。"), evidence("inference", "Product.solve；Product.judge", "B/C 的 argmax、溫度、輪數與裝置參與實際運算。")], aids=["a_originals", "a_execution"])
claim("software", "首次下載成功與離線載入推論是兩項不同檢查。", "「首次下載成功與真正離線載入推論」", "本次於 CPU 分開驗證下載與在程序內禁止 socket 連線後的本機推論，不代表其他環境都已驗證。", [evidence("delivery", "download_product", "只有下載路徑使用 urllib 連線。"), evidence("inference", "Product.__init__；solve；judge", "推論從本機路徑讀檔。")], "匿名 download b、c 成功；封鎖 socket.connect／create_connection 後 solve、judge、auto 與截短 Auto 均 exit code 0。", aids=["a_execution"])
claim("software", "MoE 助理沿用既有獨立公開 CPU 操作入口。", "「MoE 助理沿用…公開CPU操作」", "查核既有連結所指 CLI 與 safe export 契約；本次未重跑 MoE 完整下載與生成。", [evidence("chat", "main；--device", "MoE 有獨立 chat.py 入口。"), evidence("moe", "InferenceAssistant", "支援 CPU，驗證公開 safetensors 後載入。")], "scripts/selftrained/chat.py --help exit code 0，列出公開 repo、revision、manifest 與 CPU 選項。", aids=["a_supplement"])
claim("software", "Looped 解題器接受有限文字與輪數。", "「Looped 解題器接受有限文字與輪數」", "B 只接受資訊完整的兩盒／三數語法，CLI loops 為 1–6；不是任意文字解題器。", [evidence("main", "solve 子命令", "state 必填，loops 由 1–6 選擇。"), evidence("inference", "parse_state；Product.solve", "驗證語法與完整資訊才執行。")], "solve --help 列 --state／--loops；指定交換範例 loops=3 回傳 answer=2、loops=3；任意聊天文字被拒絕。", aids=["a_execution", "a_supplement"])
claim("software", "機率判斷器接受狀態、問題與候選。", "「機率判斷器接受狀態、問題與候選」", "候選需為對應 typed question 的已知語義 ID，不能任意添加語義。", [evidence("main", "judge 子命令", "提供 state、question 與 candidates。"), evidence("model", "validate_candidates", "驗證至少兩個不重複、合法的候選。")], "judge --help 列出三參數；實際判斷回傳候選機率與語義。", aids=["a_supplement", "a_execution"])
claim("software", "Auto 同時載入 B、C 與凍結規則，並接受 effort。", "「Auto 指令同時載入」", "effort=auto 使用 B 的規則；手動 effort 使用固定檔位，但仍先經 C 的兩次判斷。", [evidence("main", "auto 分支", "建立兩個 Product 並傳入 effort。"), evidence("inference", "Product.__init__；auto_solve", "讀取 auto-policy.json，依 effort 選輪數。")], "交換範例 auto 回傳 choice=boxes、info=yes、requested_loops=3；manual medium 可用。", aids=["a_execution"])
claim("software", "用完成權重作推論不等於正在訓練。", "「使用本課已訓練完成的權重作推論」", "指本節 solve／judge／auto；不涉及重新訓練 CLI。", [evidence("inference", "Product.solve；Product.judge", "方法使用 inference_mode，沒有 backward 或 optimizer.step。")], "三種 CLI 以公開既有 model.pt 完成載入與推論；參數包沒有 optimizer 狀態。", aids=["a_execution"])
claim("software", "使用本課完成權重不代表借用外部成熟神經模型。", "「也不代表借用了外部成熟神經模型」", "B/C 的 from-scratch 實作與固定作者訓練收據吻合；不是本次重新執行完整訓練來證明。", [evidence("train", "train 的建構與匯出分支", "從 seed 初始化 LoopedSolver／CandidateScorer，匯出自己選定的 state_dict。"), evidence("model", "LoopedSolver；CandidateScorer", "直接建構 PyTorch 層，不呼叫外部 pretrained 載入器。"), evidence("moe", "InferenceAssistant.__init__", "MoE 另驗證 all-neural-weights-random 的匯出契約。")], "本次下載兩個 training-receipt 均 random_initialization=true，模型參數與自己的 Config 成功匹配。", aids=["a_execution"])
claim("software", "本機重訓入口提供 cpu、mps、cuda 裝置選擇。", "「再選CPU、Apple裝置…或NVIDIA GPU」", "需裝置及相應後端可用；本次只跑 CPU 介面查核，沒有完成 MPS/CUDA 正式訓練。", [evidence("local", "main 的 --device 與 train 呼叫", "三種 device 選項傳給訓練器。"), evidence("train", "train；training-receipt.environment", "模型搬至指定裝置並記硬體。")], "train_local --help 列 {cpu,mps,cuda}；公開 B/C 收據記作者使用 NVIDIA L4。", aids=["a_supplement", "a_execution"])
claim("concept", "MPS 指 Metal Performance Shaders，是 Apple 裝置的 GPU 運算後端。", "「MPS（Metal Performance Shaders）」", "指 PyTorch 的 macOS MPS device，硬體與作業系統須支援。", [evidence("mps_official", "MPS backend 開頭與 mps device", "官方明定 macOS GPU、Metal Performance Shaders Graph 與 kernels。")], aids=["a_originals"])
claim("software", "訓練配方與收據記錄 seed、更新步數、資料版本、檢查點選擇及硬體。", "「訓練配方另記seed」", "對本節 B/C 成品的保存格式與固定公開收據進行查核，沒有重新驗證其完整訓練過程。", [evidence("train", "metadata；training-receipt", "包含 seed、last_step、selected_step、data_manifest_sha256 與 environment。")], "B 收據 seed=7421、last/selected=1000；C seed=9041、last/selected=800；兩者都含資料 SHA、torch/python/device/gpu。", aids=["a_execution"])
claim("concept", "可精確接續的檢查點需要最佳化器與隨機狀態；推論權重無法恢復原本訓練進度。", "「可續訓的檢查點還要有」與「只下載它無法恢復」", "指本專案有狀態的 AdamW 與抽題流程；可以用推論權重新開訓練，但不能稱原樣無縫續訓。", [evidence("checkpoint_official", "Saving & Loading a General Checkpoint for Inference and/or Resuming Training", "只存 model.state_dict 不包含 optimizer state。"), evidence("randomness_official", "PyTorch random number generator", "RNG 影響後續抽樣，需恢復相應狀態。"), evidence("training", "checkpoint；restore", "保存並恢復 optimizer 與 sampler／Torch／CUDA RNG。")], aids=["a_execution", "a_originals"])
claim("software", "程式碼與生成器進 Git，原始資料、檢查點及權重沿既有儲存方式排除於普通 Git。", "「程式碼與生成器保存在Git」", "指本節操作使用的資料／模型輸出；不把本專案授權、以 LFS 保存的固定教學包排除規則混為一談。", [evidence("ignore", "原始資料快取、權重與實驗輸出規則", "忽略 data、checkpoints、outputs 與模型副檔名。")], "git ls-files 列出資料生成器、模型、訓練、推論與交付程式；下載到指定暫存目錄且未把權重列為永久 artifact。", aids=["a_supplement", "a_execution"])
claim("software", "下載與上傳驗證保留可公開的檔案 hash 與結果。", "「下載與上傳驗證會保留」", "本次只讀既有發布收據並對照新下載；合成遮蔽測試只驗證已實作規則，不聲稱稽核所有歷史日誌或實際上傳。", [evidence("delivery", "download_product 回傳與 manifest", "下載核對公開 SHA 並回報 revision、manifest 與檔案數。"), evidence("publish", "publish_remote 的 receipt／anonymous_downloads", "上傳後匿名固定 revision 重取並存逐檔 SHA、大小、狀態；錯誤遮蔽。"), evidence("publication", "redact_error", "遮蔽秘密類型環境值與 URL。"), evidence("pubreceipt", "execution.public；execution.anonymous_downloads", "保存實際已發布結果的公開指紋。")], "既有發布收據與本次新下載十二檔大小、SHA 全相同；合成秘密與 URL 被遮蔽為 [REDACTED] [URL]。", aids=["a_execution", "a_publication"])
claim("software", "本節終端機命令可在根目錄的 CPU 環境執行；download 取得 B/C 後，其餘指令讀本機檔案。", "bash 命令前說明與整個命令區塊", "使用既有 .venv，未重做 W.1 安裝；為遵守寫入範圍，僅加 --destination／--b／--c 指向本次 artifact 暫存子目錄。", [evidence("warmup", "第一次開啟教材：CPU 安裝與 .venv 啟用", "啟用後 python 指向專案虛擬環境。"), evidence("main", "main 的各命令分支", "命令名稱、參數及預設 CPU 與原文一致。")], "於根目錄用 .venv Python 成功執行兩次下載及三種推論；推論子程序明確封鎖網路。", aids=["a_execution", "a_supplement"])
claim("numeric", "A=2、B=6 交換後查詢 B，正確答案是 2。", "「例中的交換把A原來的2移到B」", "單一兩盒交換題；不是模型整體正確率。", [evidence("data", "answer 的 swap 分支", "(a,b) 由 (2,6) 變成 (6,2)，查詢 B 得 2。")], "獨立世界規則函式與公開 B loops=3 都得到 2；HTTP solve 同為 2。", aids=["a_execution", "a_http"])
claim("software", "卡片進度缺申請編號時，資訊充分性的正確類別為 no。", "「卡片進度缺申請編號」", "依本課有限世界的必要欄位契約；不是對所有卡片服務的外部規定。", [evidence("data", "REQUIRED.card；judgment 的 info 分支", "application 必填；缺欄時 available 少於 needed，gold=no。")], "judgment 得 no；公開 C 的 info CLI 與 HTTP judge 也選 no。", aids=["a_execution", "a_http"])
claim("software", "--question 可選 choice、info、rubric，分別是任務／服務選擇、資訊充分性、規準等級。", "「--question 可選」", "含 boxes/sum 等有限任務；規準固定為必要欄位完整程度，不是開放式主觀品質評分。", [evidence("data", "QUESTIONS；CANDIDATES；MEANINGS", "三種問題、候選意義與 grade 的語義固定。"), evidence("main", "judge 的 --question", "CLI 限定三選項。")], "三個問題均實際呼叫 c.judge 成功，回傳對應 question、候選機率與語義；help 列三選項。", aids=["a_execution", "a_supplement"])
claim("empirical", "能接受有限輸入格式不保證模型答對其他有限輸入。", "「模型在其他有限輸入上仍可能答錯」", "只用一個實際錯誤見證支持『可能答錯』，不由搜尋後選出的錯題估計正式 test 正確率。", [evidence("inference", "Product.solve", "語法驗證只決定可否接受輸入，答案由模型 argmax 產生。")], "字典序搜尋第 14 題 三數加法;x=0;y=1;z=6 的 gold=7，公開 B 三輪答 12；之前五個固定診斷題都正確。", aids=["a_supplement"], denominators={"fixed_diagnostic_inputs": 5, "witness_search_inputs": 14, "calls_per_input_per_phase": 1, "loops": 3, "device": "cpu", "weights": "固定公開 B；推論不重新設訓練 seed", "selection": "0–6 的三數字典序，第一錯即停；兩診斷流程可重疊", "split": "本次人工診斷，非獨立 heldout 評估"})
claim("software", "Looped 解題器輸出答案與實際輪數。", "「Looped 解題器輸出答案與實際輪數」", "loops 指本次 forward 執行次數，不代表答案正確或推理深度的語義保證。", [evidence("inference", "Product.solve 回傳", "回傳 answer 與傳入 model 的 loops。"), evidence("model", "LoopedSolver.forward", "迴圈確實執行指定次數。")], "CLI solve 回傳 answer=2、loops=3；受限 Auto 的 solution.loops=2。", aids=["a_execution"])
claim("software", "機率判斷器輸出候選機率與集中度。", "「機率判斷器輸出候選機率與集中度」", "機率針對當前候選集合；集中度不等於答對機率。", [evidence("inference", "Product.judge", "softmax 後輸出 probabilities，concentration=(n*max(p)-1)/(n-1)。")], "三種 typed question 均有 probabilities、concentration、meanings；info 範例 no=1.0，集中度=1.0。", aids=["a_execution"])
claim("software", "Auto 記錄實際兩次判斷及解題輪數。", "「Auto 另記實際使用的兩次判斷」", "先做 choice、info；拒絕或不支援時 b_loops=0，接受時才執行 B。", [evidence("inference", "auto_solve", "兩次 c.judge 寫入 choice／info，實際輪數寫入 b_loops。")], "auto 的 CLI 與 HTTP 都有 choice、info、requested_loops、b_loops 及 solution.loops，值與實際請求相符。", aids=["a_execution", "a_http"])
claim("numeric", "medium 要求三輪，max-loops=2 將本次執行截為兩輪並標 budget_limited。", "「--effort medium…--max-loops 2」", "在本例 C 接受且 B 可解時成立；若資訊不足／不支援，Auto 會先拒絕而不執行 B。", [evidence("inference", "EFFORT；auto_solve", "requested=3、actual=min(3,2)=2，actual<requested 時回 budget_limited。")], "CLI 與 HTTP 均 requested_loops=3、b_loops=2、solution.loops=2、status=budget_limited。", aids=["a_execution", "a_http"])
claim("software", "成品操作頁列出固定 HF 版本、候選 ID、三種問題、JSON 服務與重訓命令。", "「成品操作…列出」", "只查核連結目標所承諾的操作內容，不把操作頁引用的正式分數視為本節新能力主張。", [evidence("readme", "先使用完成權重；重做訓練", "各項內容與本節連結承諾相符。")], "讀取 README 並以 CLI help、三種 judge、HTTP、train_local help 對照名稱與選項，均一致。", aids=["a_execution", "a_http", "a_supplement"])
claim("software", "serve --port 可在 127.0.0.1 啟動本機 HTTP 服務，文件使用 8765。", "「可用serve --port 8765在本機127.0.0.1」", "指定埠必須可用；本機 8765 已被其他程序占用，不能由此判為服務實作錯誤。改用可用的 52810 驗證同一入口。", [evidence("delivery", "serve；ThreadingHTTPServer", "服務明確綁定 (127.0.0.1,port)。"), evidence("main", "serve 的 --port", "預設 8765，允許自訂。")], "8765 試啟動得到 Address already in use；改用 52810 啟動成功，health 回 ready，三個 POST 回 200。", aids=["a_execution", "a_http"])
claim("concept", "HTTP 是讓網頁與程式傳送請求的協定。", "「HTTP是網頁與程式之間傳送請求的協定」", "正文描述此示範用途；HTTP 本身是通用應用層 request/response 協定，不限網頁瀏覽器。", [evidence("http_official", "Introduction；HTTP Architecture", "定義 HTTP 應用層與 request/response 語義。")], aids=["a_originals", "a_http"])
claim("software", "本機服務接受與 CLI 相同的 solve、judge、auto 有限語法請求，不是一般聊天服務。", "「讓別的程式用網址送出同樣的…」", "JSON 的 state 仍需通過 parse_state；HTTP 只是另一個傳輸入口。", [evidence("delivery", "Handler.do_POST", "三個 endpoint 直接呼叫相同 Product 方法或 auto_solve。"), evidence("inference", "parse_state", "拒絕任意聊天文字。")], "三個 HTTP JSON 請求回 200，答案、no 與 budget_limited 與 CLI 相同；聊天文字 parse_state 被拒絕。", aids=["a_execution", "a_http"])
claim("software", "完整本機訓練應使用自己的新輸出目錄。", "「完整訓練使用自己的新輸出目錄」", "本機 wrapper 在資料生成或訓練前拒絕已存在目錄；沒有實際啟動正式訓練。", [evidence("local", "main 的 options.output.exists", "既有輸出目錄直接 parser.error。")], "用已存在的空暫存目錄呼叫 train_local 得 exit code 2，提示輸出需另立新目錄；目錄仍空。", aids=["a_supplement"])
claim("software", "B/C 推論權重叫 model.pt，MoE 公開推論權重叫 model.safetensors，屬不同成品格式。", "「這兩個新成品的推論權重叫」", "具體指本課這些成品；副檔名本身不能保證任意來源檔案的格式或安全性。", [evidence("inference", "Product.__init__", "載入 model.pt。"), evidence("export", "export_inference", "只將 checkpoint[model] 張量輸出為 model.safetensors。"), evidence("moe", "InferenceAssistant.__init__", "使用 safetensors 讀取並檢查完整模型 state。")], "本次 B/C 實際下載與載入 model.pt；MoE 的獨立 CLI help 可執行，匯出與載入程式均指向 model.safetensors。", aids=["a_execution", "a_supplement"])
claim("concept", "此處 model.pt 使用 PyTorch 的序列化檔案格式。", "「model.pt是PyTorch自己的檔案格式」", "指 torch.save 產生的本次檔案；通常為含 pickle metadata 與張量 storage 的 ZIP64，不能把所有 .pt 都認作相同格式。", [evidence("serialization_official", "Saving and loading tensors；Serialized file format for torch.save", "說明 torch.save 的張量儲存格式及慣用 .pt 副檔名。"), evidence("train", "torch.save(selected.state_dict())", "公開權重由 torch.save 匯出。")], aids=["a_originals", "a_execution"])
claim("software", "本節 B/C 權重只存模型參數，程式用 torch.load(weights_only=True) 載入。", "「這裡只存模型參數，程式用torch.load」", "本次 Config 的 state_dict 全是參數張量；不是泛稱所有 PyTorch state_dict 都無 buffer 或只有參數。", [evidence("train", "model.pt 匯出", "保存 selected.state_dict。"), evidence("inference", "Product.__init__ 的 torch.load", "明確設定 weights_only=True。")], "B 為 18 個、C 為 30 個 tensor entries，容器皆 OrderedDict；沒有 optimizer／rng 鍵且成功載入模型。", aids=["a_execution"])
claim("concept", "weights_only=True 使用受限反序列化，只還原張量、字典等允許資料，不任意動態載入檔案指定物件。", "「只還原張量與字典這類基本資料」", "本專案未擴充 safe_globals，且公開權重為已驗 SHA 的純張量 state_dict；不把這句擴張成防 DoS、記憶體漏洞或所有惡意檔案的保證。", [evidence("serialization_official", "torch.load with weights_only=True；weights_only security；weights_only allowlist", "限制可建構類型及函式，禁止任意動態 import，仍有明列安全限制。")], aids=["a_originals", "a_execution"])
claim("software", "凍結 Auto 選擇器依舊版 B 各輪表現及成本關係制定，所以換 B 權重需要重新驗證。", "練習答案「選擇器原本根據舊Looped解題器各檔的表現制定」", "重新驗證是依賴關係要求，不宣稱規則或成本一定改變；目前規則也考慮 C 的路由結果。", [evidence("evaluation", "b_logits；auto_quality；freeze_settings", "先計算 B 不同輪數預測，再以端到端正確性與實際平均輪數選規則。"), evidence("inference", "rule_depth；auto_solve", "部署時使用綁定的規則與目前 B 權重。")], "實際載入 manifest 綁定的 auto-policy，auto 交換範例依當前規則選 3 輪；程式檢查確認規則的訓練來源依賴各深度 B 預測。", aids=["a_execution", "a_supplement"])

assert len(claims) == 39
assert all(item["sha256_equal"] and item["bytes_equal"] for item in pub["comparisons_to_fresh_downloads"])
assert SUPP["failure_witness"]["witness"] is not None
assert all(command["exit_code"] == 0 for command in SUPP["commands"])
assert all(command["exit_code"] == 0 for command in EXEC["offline_commands"])
assert len(HTTP["requests"]) == 3 and all(request["status"] == 200 for request in HTTP["requests"])

ids = [item["id"] for item in claims]
report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "19.19",
    "source": "course/chapters/19.md#19.19", "reviewer_task": "/root/rv1019_technical_19-19",
    "reviewer_context": "fresh", "author_tasks": ["/root"],
    "source_sha256": "c76c544f270aacd42aa41851cd9fec33cfd1dc9d147a9bb6d0bc9992dcd68439",
    "source_freeze": {"path": "docs/course-review-20261019/freeze-04.json", "sha256": "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"},
    "figure_sha256": {}, "verdict": "pass", "claims": claims, "sources": sources,
    "artifacts": artifacts, "issues": [],
    "claim_counts": dict(Counter(item["kind"] for item in claims)),
    "coverage_note": "已按句核對全部實質主張；先用小設定、學生不必重做正式實驗、公開資料應有來源授權及不公開私人金鑰是操作與保存要求，不冒充新的實測成績。連結頁面只核對本節實際引用的必要契約，未把其餘教材納入整頁背書。",
    "independence": "完成與保存此原始判斷前，未讀其他角色報告、指定禁止的舊審閱目錄或 outputs；本節無圖，未讀 outputs。",
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "逐句拆成 39 項主張，數值、格式、CLI、模型載入、檢查點與版本依賴均有直接依據，沒有未核實或矛盾主張。", "claim_ids": ids},
        "numeric_verification": {"status": "pass", "details": "交換以 (2,6)→(6,2) 核對 gold=2；medium 與硬上限以 min(3,2)=2 對照 CLI/HTTP。錯誤見證僅證明可能答錯，列明診斷母體與搜尋方式，不估計品質率或性能。", "claim_ids": ["c22", "c25", "c29"]},
        "figure_consistency": {"status": "not_applicable", "details": "凍結原文無圖引用，figure_sha256={}；無需讀取或渲染圖片。", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "以 curl 實讀七份官方原文並保存版本、日期、定位與全文 SHA 的短摘錄收據。PyTorch 官方文件版本 2.14，另對照實測 2.14.1 的 installed docstring；repo 程式均以全檔 SHA 固定。", "claim_ids": ids},
        "limitations": {"status": "pass", "details": "CPU 小例成功不外推到 MPS/CUDA 或一般聊天能力；模型仍有合法語法內的錯誤。8765 埠被占用屬環境限制，已以同入口可用埠完成服務驗證。weights_only 不被解讀為全面安全保證；推論包不冒稱精確續訓。", "claim_ids": ids},
    },
    "unverified_scope": [
        "未重跑正式訓練、跨 seed 能力比較、MPS 或 CUDA 訓練與效能，也未重跑完整正式測試集。",
        "未重新安裝 W.1 CPU 環境；本回合使用既有 Python 3.13.16、PyTorch 2.14.1、CPU。",
        "MoE 只核對公開 CLI 與 safetensors 匯出／載入契約，未重跑它的完整公開下載與生成；本節未宣稱其新答案或品質。",
        "預設 8765 在本機已被占用，該埠本次未啟動成功；同一服務在空閒埠 52810 的三個入口均已驗證。",
        "未執行上傳或稽核所有歷史秘密；只對照公開上傳後指紋與新下載，並查核合成遮蔽案例。",
        "讀者理解審閱由其他角色負責；本報告不判斷易讀性，也不是學生測試。",
    ],
}
assert sha(ROOT / "docs/course-review-20261019/freeze-01/sources/19.19.md") == report["source_sha256"]
assert sha(ROOT / report["source_freeze"]["path"]) == report["source_freeze"]["sha256"]
with REPORT.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": REPORT.relative_to(ROOT).as_posix(), "sha256": sha(REPORT), "verdict": report["verdict"], "claim_counts": report["claim_counts"]}, ensure_ascii=False))
