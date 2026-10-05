"""Write this reviewer's complete new canonical report without reading the old one."""

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
TASK = "/root/phase4_factual_coordinator/factual_17_4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original_env = json.loads((BASE / "original/environment.json").read_text())
cpu = json.loads((BASE / "cpu/stdout.json").read_text())
cpu_exec = json.loads((BASE / "cpu/execution.json").read_text())
original_exec = json.loads((BASE / "original/execution.json").read_text())
metadata = json.loads((BASE / "original/extraction.json").read_text())
inspection = json.loads((BASE / "inspection.json").read_text())
assert inspection["reviewer_task"] == TASK

# Recheck raw section bytes immediately before writing this own report.
import importlib.util
spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
current, _, first_line = facts.original_section(ROOT / "course/chapters/17.md", "17.4")
assert sha(BASE / "original/section.md") == metadata["source_sha256"]
assert hashlib.sha256(current).hexdigest() == metadata["source_sha256"]
assert sha(BASE / "inputs/frozen-chapter-17.md") == metadata["source_file_sha256"]
assert sha(BASE / "inputs/quantization.py") == cpu["original_module_sha256"]

env_original = {k: str(original_env[k]) for k in ("python", "torch", "torch_git_version", "cuda_build", "cuda_available", "device_requested")}
env_cpu = cpu["environment"]
artifact_ids = {}
artifacts = []
for p in sorted(BASE.rglob("*")):
    if not p.is_file():
        continue
    assert not p.is_symlink()
    relative = p.relative_to(BASE).as_posix()
    identifier = relative.replace("/", "_").replace(".", "_").replace("-", "_")
    artifact_ids[relative] = identifier
    kind = "code" if p.suffix == ".py" else "source_snapshot"
    item = {"id": identifier, "kind": kind, "path": p.relative_to(ROOT).as_posix(),
            "sha256": sha(p), "description": f"本輪 17.4 永久原始證據／版本紀錄：{relative}"}
    if relative == "inspection.json":
        item.update(kind="derivation", description="本人實際讀取範圍、公式推導、原始來源 inspection 和範圍判斷；未沿用舊判定。")
    if relative in ("original/execution.json", "original/stdout.txt", "original/stderr.txt", "original/environment.json"):
        item.update(kind="execution", command=original_exec["command"],
                    result="exit 0；scale/zero 0.2 5；碼 [0,5,10,15]；還原 [-1,0,1,2]；stderr 空；無 guard event。",
                    environment=env_original)
    if relative in ("cpu/execution.json", "cpu/stdout.json", "cpu/stderr.txt"):
        item.update(kind="execution", command=cpu_exec["command"],
                    result="exit 0；五組量化變體、全零限制、round/clamp、容器大小及 16 個 signed packer 值的往返斷言全部通過；無訓練。",
                    environment=env_cpu)
    artifacts.append(item)


def aid(path):
    return artifact_ids[path]


def official(identifier, kind, title, file, url, version, note):
    return {"id": identifier, "kind": kind, "title": title, "url": url,
            "version": version, "accessed_on": "2026-10-05", "verified": True,
            "checked_original": True, "authority_reason": "PyTorch 專案自行維護的官方原始碼／版本化 API 文件。",
            "inspection_note": note, "snapshot_artifact_id": aid("sources/" + file),
            "snapshot_sha256": sha(BASE / "sources" / file)}


sources = [
    official("observer", "official_source", "PyTorch UniformQuantizationObserverBase and MinMaxObserver", "pytorch-v2.8.0-observer.py",
             "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/ao/quantization/observer.py", "v2.8.0",
             "本人先 AST 定位 _calculate_qparams，再讀 lines 344-422 與 MinMaxObserver 435-498。實作先 min(min,0)/max(max,0)，以 quant_max-quant_min 作分母；else 分支正 epsilon、round offset、clamp zero-point。未將 docstring 的退化預设取代實際分支。"),
    official("affine", "official_source", "PyTorch FakeQuantize affine quantize/dequantize contract", "pytorch-v2.8.0-fake_quantize.py",
             "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/ao/quantization/fake_quantize.py", "v2.8.0",
             "本人讀 FakeQuantize lines 130-160：完整 clamp(round(x/scale+zero_point)) 然後減 zero_point 乘 scale；zero_point 被定義為浮點 0 對應的 quantized value。這是公式來源，沒有宣稱本节在訓練。"),
    official("round", "official_docs", "PyTorch torch.round API", "pytorch-2.8-torch.round.html",
             "https://docs.pytorch.org/docs/2.8/generated/torch.round.html", "PyTorch documentation 2.8",
             "本人讀 torch.round anchor、HTML 1586-1623；nearest integer、half to even、dtype 保留、decimals。API 行為另外在已安装 2.14.1+cpu 實行核對；未混稱版本一致。"),
    official("tensor", "official_docs", "PyTorch Tensor dtype and conversion API", "pytorch-2.8-tensors.html",
             "https://docs.pytorch.org/docs/2.8/tensors.html", "PyTorch documentation 2.8",
             "本人讀 Data types HTML 1613-1631，uint8 是 unsigned 8-bit；float HTML 2507-2509 等同 to(float32)；to 3375-3377 做 dtype/device conversion；tolist 3411-3413 回傳 nested list。原 fence CPU 執行核對這組相關 API。"),
    {"id": "repo_quant", "kind": "repository_code", "title": "本輪 frozen repository quantization.py 原始契約",
     "path": (BASE / "inputs/quantization.py").relative_to(ROOT).as_posix(), "sha256": sha(BASE / "inputs/quantization.py"),
     "version": "frozen from repository HEAD d14ad30f9794f7d6ec83e3aed0f02dc359e29eac; exact bytes fb3afbb775d25e3d19a2843171e228e3342422c6fd9056fe0a0af1d057dce206",
     "verified": True, "inspection_note": "先 AST 定位，再親讀 quantize_symmetric lines 8-17、quantize_affine 20-26、pack_int4 29-36、unpack_int4 39-44。本人核對原件與永久副本 SHA；只核規約及已執行的有界小值，不評測完整模型。"},
    {"id": "original_execution", "kind": "execution", "title": "未修改的 17.4 Python fence 真實 CPU 執行",
     "verified": True, "artifact_id": aid("original/execution.json")},
    {"id": "cpu_variants", "kind": "execution", "title": "本人有界 CPU 算例、容器與 packer 變體",
     "verified": True, "artifact_id": aid("cpu/stdout.json")},
    {"id": "math", "kind": "derivation", "title": "本人獨立整數格與 affine 解碼推導", "verified": True,
     "details": "四位元碼有 2^4=16 個；0..15 只有 15 個相鄰間隔。寬 2-(-1)=3，scale=3/15=1/5；zero=-(-1)/(1/5)=5，解碼 (q-5)/5。練習寬 1-(-2)=3，zero=10，解碼 (q-10)/5。q=zero 解碼恰為0；scale單位是浮點值/碼間隔，zero 是碼，分母並非樣本數。包含0後 low<=0<=high；全零時二者0，未guard的scale0無法作除數。"},
]


def ev(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}


claims = [
    {"id": "affine_grid", "kind": "concept", "status": "verified", "location": "17.4 lines 109-111, 130",
     "statement": "zero-point 是代表浮點零的整數碼；scale 決定格距，還原為 (碼-zero)*scale；仿射量化還包含 round/clamp 的近似。",
     "scope": "本節 uniform affine 示範，正 scale、整數 zero；非格點仍可能有捨入誤差，未主張任意浮點可無損還原。",
     "evidence": [ev("affine", "FakeQuantize lines 130-147", "官方 affine 公式、scale 及 zero-point 定義。"),
                  ev("math", "q=zero => (q-zero)*scale=0; adjacent q difference => scale", "零的位置及相鄰浮點格距由代數直接核實。"),
                  ev("cpu_variants", "/cases/2 (off_grid)", "0.07、1.01 映回0、1，存在0.07、約0.01誤差。")],
     "artifact_ids": [aid("sources/pytorch-v2.8.0-fake_quantize.py"), aid("cpu/stdout.json")]},
    {"id": "original_numbers", "kind": "numeric", "status": "verified", "location": "17.4 lines 109, 128",
     "statement": "四位元有16碼、15個間隔；[-1,2] 用 scale 0.2、zero5；原 x 得碼 [0,5,10,15] 並還原 [-1,0,1,2]。",
     "scope": "四筆手工輸入，各項恰在該格；數字是算例，不是模型成績或資料平均。",
     "evidence": [ev("observer", "_calculate_qparams lines 369-370 and 402-405", "range/(qmax-qmin) 與 integer offset 公式。"),
                  ev("math", "2^4=16; 15-0=15; 3/15=1/5; q=[0,5,10,15]", "本人有理數推導，核對單位與分母。"),
                  ev("original_execution", "original/stdout.txt lines 1-3", "原 fence 實際打印完全相符。")],
     "artifact_ids": [aid("original/execution.json"), aid("original/stdout.txt"), aid("cpu/stdout.json")],
     "verification": {"method": "executed", "expected": "scale=0.2、zero=5；q=[0,5,10,15]；restored=[-1,0,1,2]；16碼15間隔。",
                      "observed": "原fence與變體原例完全符合；torch.equal(restored,x) 成立；Fraction 驗證scale=1/5。",
                      "details": "原 x shape(4,)；min/max是整個tensor的純量；15是碼間隔數。無平均、無分數分母。",
                      "tolerance": "碼、zero、間隔數與torch.equal精確相等；數學刻度用Fraction精確1/5，Python浮點打印0.2。"}},
    {"id": "original_fence", "kind": "software", "status": "verified", "location": "17.4 Python fence lines 114-125",
     "statement": "fence 執行 tensor 建立、全體min/max取純量、round/clamp、to(uint8)、float 後解碼及tolist打印；沒有梯度或參數更新。",
     "scope": "一組相關普通API合組覆蓋；uint8存碼後先轉浮點才減zero，避免有號解碼在unsigned容器進行。CPU直接示範，不是原生int4儲存或推論。",
     "evidence": [ev("round", "torch.round anchor, HTML lines 1586-1623", "逐元素nearest round及decimals API。"),
                  ev("tensor", "Tensor.float / Tensor.to / Tensor.tolist; HTML 2507-2509, 3375-3377, 3411-3413", "dtype转换及显示契约。"),
                  ev("original_execution", "original/fence-1.py, original/environment.json, original/execution.json", "全部原 fence 在 CPU/離線guard下成功執行，无guard事件。"),
                  ev("cpu_variants", "/round_half_to_even, /clipped_codes, /cases/0/container_bytes", "round ties、clamp endpoint及uint8 1byte实际核對。")],
     "artifact_ids": [aid("original/fence-1.py"), aid("original/execution.json"), aid("cpu/stdout.json")],
     "verification": {"method": "executed", "expected": "原 fence exit0並打印正文三行；round/clamp/conversion運作，無訓練動作。",
                      "observed": "exit0；stdout三行符合；uint8 element_size1，float还原dtype float32；midpoints得到[-0,0,2,2]，clipped碼[0,0,5,15,15]。",
                      "details": "實際Python3.13.5、torch2.14.1+cpu，cuda_build None、available False；原碼385 bytes SHA保存。官方API引用2.8，未把它寫成執行環境版本。"}},
    {"id": "exercise_numbers", "kind": "numeric", "status": "verified", "location": "17.4 line 134",
     "statement": "x 改為 [-2,0,1]，寬度仍3，scale仍0.2；zero為10，碼為[0,10,15]。",
     "scope": "三筆練習值；僅改端點位置，說明格距與零點的不同工作；各值恰在格。",
     "evidence": [ev("math", "3/15=1/5; -(-2)/(1/5)=10; [-2,0,1]/(1/5)+10", "本人代入練習數字。"),
                  ev("cpu_variants", "/cases/1 (exercise); /rational_derivation/exercise_zero", "CPU練習變體與有理數核對相同。")],
     "artifact_ids": [aid("cpu/variants.py"), aid("cpu/stdout.json")],
     "verification": {"method": "executed", "expected": "scale0.2、zero10、q[0,10,15]、restored[-2,0,1]。",
                      "observed": "三項碼、zero與restored精確符合；Fraction推導zero10。",
                      "details": "range寬3；15碼間隔；x shape(3,)；容器3bytes，與量化格距的數學分母無關。",
                      "tolerance": "整數、torch.equal及Fraction精確相等；浮點scale打印0.2。"}},
    {"id": "zero_range", "kind": "concept", "status": "verified", "location": "17.4 lines 128, 130",
     "statement": "min(...,0)、max(...,0)使選定區間包含浮點零；全零區間需要特別處理scale，原例只示範非零寬度。",
     "scope": "有限非空示範值；本fence不宣稱是通用量化函數。全零失败正是文中揭示的边界，不判為未揭示程式缺陷。",
     "evidence": [ev("observer", "_calculate_qparams lines 369-370, 402-405", "官方同樣包含零並為scale加正epsilon。"),
                  ev("math", "low=min(min(x),0)<=0<=max(max(x),0)=high", "包含零的代數及low=high=0時除零。"),
                  ev("cpu_variants", "/cases/3, /cases/4, /all_zero", "正值區間補0、负值区间补0；原全零抛ZeroDivisionError，repo helper正scale還原零。")],
     "artifact_ids": [aid("cpu/stdout.json"), aid("inputs/quantization.py")]},
    {"id": "container_format", "kind": "concept", "status": "verified", "location": "17.4 line 130",
     "statement": "torch.uint8 是一byte無號容器；只使用0..15不等於已把兩碼打包到一byte。",
     "scope": "本fence每元素保留一個uint8；四位元是碼集合的16個選擇，實體容器仍是8bit。",
     "evidence": [ev("tensor", "Data types #data-types, HTML lines 1613-1631", "torch.uint8 是unsigned 8-bit。"),
                  ev("cpu_variants", "/cases/0/container_bytes and variants.py element_size assertions", "四碼實際4bytes，非2bytes。"),
                  ev("original_execution", "original/fence-1.py lines 8-9", "只有to(uint8)與float解碼，沒有打包位元操作。")],
     "artifact_ids": [aid("original/fence-1.py"), aid("cpu/stdout.json")]},
    {"id": "later_packer", "kind": "software", "status": "verified", "location": "17.4 line 132; relevant 17.8 context",
     "statement": "後續pack_int4 容纳 -8..7 的signed值，加8映成儲存碼；它與本節unsigned affine碼及zero-point須按各自規則對齊。",
     "scope": "只核本repo實際打包器與此段格式提醒；未核後節模型storage實測，也未宣稱所有4-bit格式通用或可直接互換。",
     "evidence": [ev("repo_quant", "pack_int4 lines 29-36; unpack_int4 lines 39-44", "范围检验、+8映码、low/high nibble及-8反映。"),
                  ev("cpu_variants", "/packer", "16個signed值往返一致；signed零雙碼打成136 (0x88)；本節含10、15的affine碼直接输入被拒。")],
     "artifact_ids": [aid("inputs/quantization.py"), aid("cpu/variants.py"), aid("cpu/stdout.json")],
     "verification": {"method": "executed", "expected": "signed -8..7往返不变；零存成nibble8；0..15 affine碼不可盲傳signed packer。",
                      "observed": "全16值往返精確相同，8bytes；零雙碼byte136；含10/15輸入ValueError 'int4 可儲存 -8 到 7'。",
                      "details": "AST先定位後親讀原契約；运行调用实际repo helper，其完整SHA与永久副本一致。"}},
]

report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "17.4",
    "source": "course/chapters/17.md#17.4", "source_sha256": metadata["source_sha256"],
    "reviewer_task": TASK, "reviewer_context": "fresh", "verdict": "pass", "figure_sha256": {},
    "frozen_input": {"path": (BASE / "inputs/frozen-chapter-17.md").relative_to(ROOT).as_posix(),
                     "sha256": metadata["source_file_sha256"], "meaning": "最初本輪fence擷取時的完整Markdown原始bytes快照；不是目前整章版本宣稱。"},
    "actual_reading": inspection["actual_reading"], "sources": sources, "claims": claims,
    "artifacts": artifacts, "issues": [],
    "checks": {
        "factual_accuracy": {"status": "pass", "details": "逐實質主張核affine公式、zero定義、uniform刻度、uint8容器及後續packer本地格式；原例與边界一致。", "claim_ids": [x["id"] for x in claims]},
        "numeric_verification": {"status": "pass", "details": "本人Fraction核16碼/15間隔、3/15與zero5/10；實行原fence及练习，碼与浮點還原精確相同；off-grid可见rounding損失。非模型得分，無實測样本分母。", "claim_ids": ["original_numbers", "exercise_numbers"]},
        "figure_consistency": {"status": "not_applicable", "details": "17.4完整原文無圖、SVG或圖形數字；extraction svg_references=[]。未聲稱render任何圖。", "claim_ids": []},
        "source_verification": {"status": "pass", "details": "親讀PyTorch v2.8.0官方observer/fake_quantize及2.8 API原頁，版本URL/locator/inspection/下載command/SHA永久保存；量化helper原件/副本SHA相同。沒有以來源庫摘要或舊報告替代原來源。", "claim_ids": ["affine_grid", "original_fence", "zero_range", "container_format", "later_packer"]},
        "limitations": {"status": "pass", "details": "原碼為encoding/decoding示範、无backward/optimizer/training；全零scale0需guard已揭示，原变体實際验证。無本節既有結果JSON、無模型成績、無GPU/模型資料下載/完整重評。官方参考2.8與实际2.14.1+cpu版本清楚區別。", "claim_ids": ["affine_grid", "original_fence", "zero_range", "container_format", "later_packer"]}
    },
    "scope_limits": [inspection["external_version_limit"], inspection["figures"], inspection["empirical_json"], inspection["boundary"]],
}
target = ROOT / "docs/technical-reviews/17.4.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
own = json.loads(target.read_text())
assert own["reviewer_task"] == TASK and own["source_sha256"] == metadata["source_sha256"]
print(json.dumps({"own_canonical": target.relative_to(ROOT).as_posix(), "reviewer_task": TASK,
                  "source_sha256": own["source_sha256"], "report_sha256": sha(target),
                  "claims": len(claims), "artifacts": len(artifacts), "verdict": own["verdict"]}, ensure_ascii=False))
