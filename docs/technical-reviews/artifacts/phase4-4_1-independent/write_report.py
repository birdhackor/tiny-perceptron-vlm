"""Own independently written claim mapping, not imported from prior reports."""
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
hashfile = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
facts = json.loads((OUT / "probe-result.json").read_text())
extraction = json.loads((OUT / "original/extraction.json").read_text())
intro = json.loads((OUT / "intro-receipt.json").read_text())
env = {"python": facts["python"], "torch": facts["torch"], "torch_git_version": facts["torch_git_version"], "device": "cpu", "cuda_build": "None", "cuda_available": "False"}
artifacts = []
artifact_map = {}
for p in sorted(OUT.rglob("*")):
    if not p.is_file() or "__pycache__" in p.parts or p.name == "checker-receipt.json":
        continue
    local = p.relative_to(OUT).as_posix()
    ident = local.replace("/", "--").replace(".", "-")
    artifact_map[local] = ident
    kind = "code" if p.suffix == ".py" else "source_snapshot"
    obj = {"id": ident, "path": p.relative_to(ROOT).as_posix(), "sha256": hashfile(p), "kind": kind, "description": f"4.1 independent permanent evidence: {local}"}
    if local in ("original/execution.json", "probe-execution.json"):
        run = json.loads(p.read_text())
        obj.update(kind="execution", command=run["command"], result=f"exit_code={run['exit_code']}; original output and probe result retained separately", environment=env)
    elif local == "probe-result.json":
        obj["kind"] = "derivation"
        obj["description"] = "實跑數值、Decimal 逐格加法、座標軸、單位、索引、梯度及目前模型入口的核對結果"
    artifacts.append(obj)

def a(*paths):
    return [artifact_map[path] for path in paths]

def original_source(ident, title, name, kind, version, authority, inspection):
    receipt = next(x for x in json.loads((OUT / "sources/acquisition.json").read_text()) if x["filename"] == name)
    return {"id": ident, "kind": kind, "title": title, "verified": True, "url": receipt["url"], "version": version,
            "authority_reason": authority, "checked_original": True, "accessed_on": "2026-10-05", "inspection_note": inspection,
            "snapshot_path": (OUT / "sources" / name).relative_to(ROOT).as_posix(), "snapshot_sha256": receipt["sha256"]}

commit = "5c4886908584029761b579af026dcfb627c84070"
sources = [
    original_source("transformer", "Attention Is All You Need", "vaswani-2017-v7.pdf", "paper", "arXiv:1706.03762v7", "作者的原始 Transformer 論文與版本化 arXiv PDF。", "親讀 PDF 抽取原文 pp.5–6 §§3.2.3–3.5：非法注意連結於 softmax 前遮蔽；字與位置表示同維度相加；可用 learned/fixed 位置；原論文主設定採 sinusoidal，並非本節的手填小數。"),
    original_source("nope", "Transformer Language Models without Positional Encodings Still Learn Positional Information", "nope-2022-v1.pdf", "paper", "arXiv:2203.16634v1 (2022-03-30)", "作者的 NoPos 實驗及位置探測原始論文。", "親讀 pp.1–2 Abstract/Introduction/§2 以及 p.3 §5：明確介紹每個绝對位置的向量加至 token embedding；learned embedding 是被訓練的位置表示；NoPos 可獲得隱式位置資訊。因果注意藉可見前綴數量產生位置線索是作者 conjecture/hypothesis，未改寫成所有模型的保證。"),
    original_source("gpt2", "OpenAI GPT-2 official model source", "gpt2-model.py", "official_source", "openai/gpt-2 commit 9b63575ef42771a015060c964af2c3da4cf7c8ab (2024-01-26)", "OpenAI 發布的 GPT-2 模型原始程式，固定 commit。", "親讀完整 src/model.py，重點 attention_mask L58–67、positions_for L139–142、model L145–177。wpe/wte 都是 get_variable 的 random-normal 初始化，兩個 gather 結果相加，再經 Transformer；不能把本節等差小数當 GPT-2 位置值。"),
    original_source("embedding", "PyTorch nn.Embedding original source and docstring", "pytorch-sparse.py", "official_source", f"PyTorch installed 2.14.1+cpu; git {commit}", "PyTorch 官方 repo 以安裝 CPU wheel 的 git_version 定位；下載 bytes 與 wheel 的 nn/modules/sparse.py 精確相同。", "親讀 Embedding L14–202：固定字典按索引查列、output (*,H)、weight (num_embeddings,embedding_dim)、Parameter requires_grad=not _freeze、reset_parameters 的 normal_ 及 forward 的 F.embedding。未把 padding_idx 的特別規則誤用於本節未設 padding_idx 的表。"),
    original_source("torch-api", "PyTorch torch API original docstrings", "pytorch-torch-docs.py", "official_docs", f"PyTorch installed 2.14.1+cpu; git {commit}", "PyTorch 官方原始 docstrings，與安裝檔 exact-byte 核對。", "親讀 torch.add L401–449、torch.equal L4242–4262、torch.arange L9680–9727：逐元素加法並可 broadcast；equal 比大小與元素；arange 區間 [start,end)、整數參數預設 int64。本節同寬是表示設計，不泛化成 PyTorch 禁止所有 broadcasting。"),
    original_source("tensor-api", "PyTorch Tensor mutation original docstrings", "pytorch-tensor-docs.py", "official_docs", f"PyTorch installed 2.14.1+cpu; git {commit}", "PyTorch 官方 Tensor API 原始 docstrings，與安裝檔 exact-byte 核對。", "親讀 copy_ L1147–1170 和 zero_ L6282–6288：前者將 src 逐元素複製至 self，後者就地填零。教材只有手動指定權重，沒有 optimizer 更新。"),
    original_source("grad-mode", "PyTorch no_grad original source and docstring", "pytorch-grad-mode.py", "official_source", f"PyTorch installed 2.14.1+cpu; git {commit}", "PyTorch 官方 no_grad 實作，與安裝檔 exact-byte 核對。", "親讀 no_grad L22–85，範圍內停用梯度記錄、__exit__ 恢復之前 grad mode；不把暫時停記錄誤稱永久凍結 Parameter。對照執行中的 AddBackward0 與 scope 內/外 requires_grad。"),
    {"id": "model", "kind": "repository_code", "title": "Current TinyLM absolute-position input path", "path": "tiny_perceptron/model.py", "sha256": hashfile(ROOT / "tiny_perceptron/model.py"), "version": "2026-10-05 current checkout bytes; permanent identical model.py snapshot", "verified": True, "inspection_note": "親讀完整 model.py；本節關係是 L60–61 同寬 token/position embeddings、L71–79 lookup/offset/cap/addition。讀 attention.py L10–17、L32–69 確認 causal 許可與 learned table 是不同操作。TinyLM 增加 batch 軸後進第一 block 的值已用 pre-hook 實跑核對。"},
    {"id": "attention", "kind": "repository_code", "title": "Current causal permission implementation", "path": "tiny_perceptron/attention.py", "sha256": hashfile(ROOT / "tiny_perceptron/attention.py"), "version": "2026-10-05 current checkout bytes", "verified": True, "inspection_note": "親讀完整 attention.py；L10–17 使用 key_position <= query_position 產生 True 許可；L20–27 softmax 前 masked_fill；L49–69 執行注意力並可選 rotary。"},
    {"id": "original-run", "kind": "execution", "title": "Exact current 4.1 Python fence under CPU helper", "verified": True, "artifact_id": artifact_map["original/execution.json"]},
    {"id": "probe-run", "kind": "execution", "title": "Independent bounded 4.1 counterfactuals and decimal check", "verified": True, "artifact_id": artifact_map["probe-execution.json"]},
    {"id": "addition", "kind": "derivation", "title": "Independent exact decimal elementwise sums", "verified": True, "details": "以 Decimal 分開建立字表與位置表的三排數值，逐位置、逐特徵加：1+0=1；0+0.1=0.1、1+0.2=1.2、0+0.3=0.3；1+0.2=1.2、0+0.4=0.4、0+0.6=0.6。float32 與精確 decimal 的最大絕對差為 4.76837158203125e-8。沒有量綱與評測分母。"},
]

def ev(s, loc, supports):
    return {"source_id": s, "locator": loc, "supports": supports}

def claim(ident, kind, text, loc, scope, evidence, ids=(), verification=None):
    obj = {"id": ident, "kind": kind, "statement": text, "location": loc, "scope": scope, "status": "verified", "evidence": evidence, "artifact_ids": a(*ids)}
    if verification:
        obj["verification"] = verification
    return obj

def executed(expected, observed, details, tolerance=None):
    obj = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if tolerance:
        obj["tolerance"] = tolerance
    return obj

claims = [
    claim("absolute-table", "concept", "相同 token ID 會查到同一字向量；本節另按絕對位置索引查一個同寬可調向量，與字向量相加，讓相同字處於不同格時有可不同的輸入表示。", "04.md L7–9, L34, L47", "查表輸入層的線索；不是所有位置方法的統一公式，也未稱加法本身已理解語序。兩表同寬是此向量配方，torch.add 的一般 broadcasting 不在此主張。", [
        ev("embedding", "nn.Embedding L14–40 and L189–198", "ID 只作 lookup，並無位置輸入；weight 是可學習固定字典，各索引取對應同寬向量。"),
        ev("transformer", "§3.5 p.6 first paragraph", "位置與 input embedding 同 dmodel，兩者相加，可選 learned 或 fixed。原論文的具體 sinusoidal 公式不同於本例，不混稱。"),
        ev("gpt2", "model.py L152–157", "實際自回歸模型把 wte 的 token lookup 與 wpe 的 absolute-position lookup 相加。"),
    ], ("sources/vaswani-2017-v7.pdf", "sources/pytorch-sparse.py", "sources/gpt2-model.py")),
    claim("table-arithmetic", "numeric", "表格的三個和是 [1,0,0]、[0.1,1.2,0.3]、[1.2,0.4,0.6]；兩次貓原字向量相同，但加位置後不同。", "04.md L11–15, L26–34", "三個位置、每位置三個 unitless feature；無 batch 軸、無樣本或 token 比率分母。四位小數印出值是 float32 的顯示。", [
        ev("addition", "Decimal derivation in probe.py and probe-result.json independent_decimal_sums", "逐格代數加法的精確十進位值獨立於 torch 的 float32 對照。"),
        ev("original-run", "original/stdout.txt and original/execution.json", "實際原碼 print 的位置與三排和；原 assert 已通過。"),
        ev("probe-run", "probe-result.json original_values/max_abs_error_from_exact_decimal", "實值 shape [3,3]；對精確 Decimal 最大差 4.76837158203125e-8，位置與 equal 比較精確。"),
    ], ("original/execution.json", "original/stdout.txt", "probe-execution.json", "probe-result.json"), executed("上述三排精確 decimal sums；cat token features equal、position sums unequal。", "三排與表一致；最大 decimal 誤差 4.76837158203125e-8；兩項 cat 比較均符合。", "軸是 [position,feature]；本節沒有約分或平均。", "浮點值 atol=1e-6、rtol=0；整數、形狀和 torch.equal 判定為精確相等。")),
    claim("api-and-autograd", "software", "Embedding 建表、copy_ 手填、arange(3) 產生 int64 [0,1,2]、lookup/add 保持 [3,3]、equal 檢查元素；no_grad 只包手填，x 仍有 AddBackward0，相加可被求導。原程式未執行 backward 或參數更新。", "04.md L18–31, L34, L47", "原 fence 的完整 API 群組。驗證後另加的 x.sum().backward() 是工具診斷，不改寫教材為訓練實驗；AddBackward0 名稱限已測安裝版。", [
        ev("embedding", "Embedding Shape/Attributes L33–40; __init__ L162–170; forward L189–198", "四/五列三維 table、input [3] → output [3,3]；預設可求導。"),
        ev("torch-api", "torch.arange L9680–9727; torch.add L401–449; torch.equal L4242–4262", "arange 的終點不包含且預設整數 int64；add 對應每格和；equal 驗值與大小。"),
        ev("tensor-api", "Tensor.copy_ L1147–1170; Tensor.zero_ L6282–6288", "copy_ 就地載入範例权重；zero_ 是練習清零的真實 API。"),
        ev("grad-mode", "no_grad L22–85", "enter 暫停 grad、exit 復原，沒有永久凍結兩表。"),
        ev("original-run", "original/fence-1.py; stdout.txt; execution.json", "原碼實跑成功，只有前向/print/assert，stdout 顯示 AddBackward0。"),
        ev("probe-run", "probe-result.json original_shape/original_grad_fn/after_assignment_requires_grad/backward_e_grad/backward_p_grad", "原前向 e/p.grad 都 None；scope 外 x.requires_grad=True，之後診斷能回傳兩表梯度；scope 內變化 x_no_grad 為 False。"),
    ], ("original/fence-1.py", "original/execution.json", "probe-execution.json", "probe-result.json"), executed("手填不留 graph；離開 no_grad 的 lookup/add 建圖；原輸出 [3,3] 並有 AddBackward0。", "所有 API、形狀、grad mode 及 AddBackward0 符合；原碼沒有更新；診斷 backward 兩表均有梯度。", "PyTorch 2.14.1+cpu git 5c488690…；四份下載原碼與 installed bytes 精確相同。")),
    claim("learned-not-arithmetic", "concept", "位置小數為人指定的展示，learned absolute position table 可以亂數初始化並與 token table 一起求導學習；沒有逐格等差遞增規定。", "04.md L36", "僅 learned absolute embedding 路線；不替 fixed sinusoidal、RoPE 或其他方法下初始化結論。本節原 forward 不包含訓練更新。", [
        ev("gpt2", "model.py L152–157", "wpe 是 random-normal 初始化的模型 variable、wte 亦然；两者相加後共同參與模型。"),
        ev("embedding", "Embedding.__init__ L162–170; reset_parameters L182–184", "PyTorch 預設 Parameter 可求導、normal_ 初始化，沒有單調或等差條件。"),
        ev("nope", "§2 p.2 Learned Embeddings paragraph", "原文直接將 learned embeddings 定義為 trained for absolute positions。"),
        ev("probe-run", "probe-result.json backward_e_grad/backward_p_grad/arbitrary_position_values/arbitrary_output", "短 backward 支持本例兩表可一起獲梯度；任意非等差表也完成相同 lookup/add。這不是訓練成效的證據。"),
    ], ("sources/gpt2-model.py", "sources/pytorch-sparse.py", "sources/nope-2022-v1.pdf", "probe-execution.json", "probe-result.json")),
    claim("zero-position-exercise", "software", "在手填區加 p.weight.zero_() 並改 equal 斷言後，兩次貓相加結果相同；本例初始表示的差異來自位置列。", "04.md L40", "比較第一個注意力層之前的 x；不聲稱零位置表的完整因果模型永遠無法利用順序。", [
        ev("tensor-api", "Tensor.zero_ L6282–6288", "zero_ 將五列位置表就地填零。"),
        ev("probe-run", "exercise-zero-position.py; probe-result.json zero_exercise_values/swapped_position_values", "真正修改並執行原 fence 的練習；三排變成字表，cat equal 斷言成功；只反轉位置索引會交換兩個 cat 和而不改字表。"),
    ], ("exercise-zero-position.py", "probe-execution.json", "probe-result.json"), executed("清零後 x=[[1,0,0],[0,1,0],[1,0,0]]；x[0]==x[2]。", "練習原碼副本實跑通過，三排和 exact equal 字向量。", "no_grad 範圍內真實 zero_；交換位置反例核查位置這個輸入來源。")),
    claim("mask-is-different", "concept", "因果許可限制可讀前文，位置向量則改變各位置輸入特徵；許可範圍也可能提供一些順序線索，所以本節不把 explicit position 當成唯一可能的順序來源。", "04.md L38", "區分兩種操作；NoPos 論文將 causal 可見數量的機制寫成假說，教材也只說能取得一些線索，不是性能保證。", [
        ev("transformer", "§3.2.3 p.5; §3.5 p.6", "decoder 對非法連結設定 -inf，與加入 input positional vector 的位置不同。"),
        ev("nope", "Abstract and Introduction p.1; §5 p.3", "NoPos 隱式位置探測支持『可有位置線索』；可見前綴數量是原作者猜想，不外推任意模型能力。"),
        ev("attention", "attention_mask L10–17; manual_attention L20–27", "目前程式以 key<=query 給許可，再遮蔽 score，沒有把 mask 直接當本節相加向量。"),
        ev("probe-run", "probe-result.json causal_allowed", "三格許可矩陣的可見數是1、2、3；只是操作結構核對，不代替 NoPos 實驗。"),
    ], ("sources/vaswani-2017-v7.pdf", "sources/nope-2022-v1.pdf", "probe-execution.json", "probe-result.json")),
    claim("table-bounds", "software", "五列位置表有效位置是0到4；位置5越界，表格同寬相加才是本節的逐格表示。", "04.md L42, L47", "nn.Embedding(5,3) 本例的非負索引契約；不是長文品質或一般 broadcasting 的承諾。", [
        ev("embedding", "Embedding num_embeddings/weight/Shape L20–40; forward L189–198", "固定字典大小決定可查表列數，lookup 不自動擴表。"),
        ev("torch-api", "torch.add L401–449", "一般 add 可广播，但本文配方特徵向量同寬；測試選 non-singleton 3 vs2。"),
        ev("probe-run", "probe-result.json bounds_errors/incompatible_width_error", "位置4成功，5及-1都 IndexError；三格字向量加兩格位置向量 RuntimeError。"),
    ], ("probe-execution.json", "probe-result.json"), executed("0–4可查列；5越界；3維+2維拒絕。", "row4數字 exact equal；5、-1為IndexError；3 vs2出 shape RuntimeError。", "CPU 即時查表，沒有截斷、回捲或增加位置表。")),
    claim("capacity-is-not-competence", "concept", "增加表的列數只增加可用索引；不保證各位置已充分訓練、模型已懂語序或長文能用好位置。", "04.md L36, L42", "教材主張是限制，沒有報告任何 long-context 成績。其 RoPE/視窗延伸句是後續方法路標，此節沒有相应性能或验收聲明。", [
        ev("embedding", "Embedding L20–40 and L162–198", "num_embeddings 只是 parameter/lookup 容量，程式沒有長文能力驗收。"),
        ev("nope", "§2 p.2 learned vs fixed; §5 p.3 probing analysis", "位置表示的輸入定義與經訓練後的表示探測是分開問題，未給本節手填向量任何語序能力。"),
        ev("probe-run", "probe-result.json unused_p_rows/backward_p_grad", "有限診斷只用0–2，3–4雖能索引卻沒有梯度；支持容量不等於本次收到訓練訊號。不能由 toy backward 推斷成熟模型品質。"),
    ], ("sources/pytorch-sparse.py", "sources/nope-2022-v1.pdf", "probe-execution.json", "probe-result.json")),
    claim("current-implementation", "software", "本節 learned absolute lookup/add 起點符合目前預設 TinyLM：同寬兩表，按位置相加，超過配置長度拒絕。", "04.md L9, L34, L42, L47; implementation cross-check", "小節範例無 batch；TinyLM 有 [batch,position,feature]。只核輸入第一 block 的數值與索引容量，不驗收其語言品質。rotary 選項在 model.py 不建此位置表，符合後續方法會不同的路標。", [
        ev("model", "TinyLM L60–61, L71–79", "預設兩表寬度均c.width；position arange及加法一致；offset+t 上限有明確錯誤。rotary=True 時 table 為None。"),
        ev("probe-run", "probe-result.json tiny_lm_first_block_shape/tiny_lm_first_block_values/length_error", "實跑三字模型捕捉第一 block 之前 [1,3,3]，去掉 batch 軸後與教材每格和一致；六格長度拒絕。"),
    ], ("model.py", "probe-execution.json", "probe-result.json"), executed("第一 block 輸入 [1,3,3] 且與表格每格一致；六格超過max_length5拒絕。", "hook捕捉值與原表atol1e-6一致，長度6為ValueError。", "同一份現行 model.py 已保存hash；此限定CPU前向約2秒，沒有訓練。")),
]
report = {
 "schema_version": 1, "review_stage": "technical", "lesson_id": "4.1", "source": "course/chapters/04.md#4.1", "source_sha256": extraction["source_sha256"],
 "reviewer_task": "/root/phase4_factual_coordinator/factual_4_1", "reviewer_context": "fresh", "verdict": "pass", "figure_sha256": {},
 "intro_sha256": intro["sha256"], "intro_summary": intro["intro_summary"],
 "read_scope": {"introduction": "course/chapters/04.md original UTF-8 bytes before 4.1 (L1–4)", "section": "4.1 all current bytes (L5–50), table, Python fence, exercise and details", "incidental_context": "The current chapter read displayed 4.2, 4.3, 4.4 and the opening of 4.5 (through L180); only 4.1 is adjudicated in this report. No neighboring SVG was opened or judged.", "implementation": ["tiny_perceptron/model.py all; focus L60–79", "tiny_perceptron/attention.py all; focus attention_mask and forward"], "original_sources": "Own fetched original PDFs/docstrings/source bytes, inspected passages stated per source; no previous review/report/history text or judgments read", "figures": "none referenced in current 4.1"},
 "claims": claims, "sources": sources, "artifacts": artifacts, "issues": [],
 "checks": {
  "factual_accuracy": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "親核 learned absolute lookup 與 add、初始化/求導、因果許可差異、容量及能力限制；未發現需改教材的實質錯誤。"},
  "numeric_verification": {"status": "pass", "claim_ids": ["table-arithmetic"], "details": "獨立 Decimal 逐格和與 CPU float32 對照，最大差4.76837158203125e-8；axes [position,feature]，unitless，無 metric 分母。"},
  "figure_consistency": {"status": "not_applicable", "claim_ids": [], "details": "raw 4.1 未引用任何 SVG 或其他圖片；表格逐格數值由 table-arithmetic 核查。沒有將4.2的殘差圖誤列成4.1。"},
  "source_verification": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "直接取固定 arXiv PDFs、OpenAI GPT-2 commit及與安裝 wheel git對應的PyTorch原始資料，親讀locator並自寫各supports。原資料URL、版本、HTTP回條和SHA都永久保存；PyTorch4個檔與installed bytes exact相同。"},
  "limitations": {"status": "pass", "claim_ids": ["learned-not-arithmetic","zero-position-exercise","mask-is-different","capacity-is-not-competence","current-implementation"], "details": "原碼只是前向展示；額外 backward 僅診斷可求導、無optimizer或成績。無GPU、資料/模型下載、訓練或上傳。NoPos causal機制保持作者hypothesis範圍；未把零初始位置表示外推為完整模型永遠無順序訊號。"},
 },
}
(ROOT / "docs/technical-reviews/4.1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": "docs/technical-reviews/4.1.json", "claims": len(claims), "artifacts": len(artifacts), "source_sha256": extraction["source_sha256"], "intro_sha256": intro["sha256"], "verdict": "pass"}, indent=2))
