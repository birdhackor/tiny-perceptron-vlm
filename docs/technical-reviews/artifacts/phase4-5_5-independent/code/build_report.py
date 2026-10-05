"""Serialize this reviewer's own claim-by-claim findings; never ingest previous reports."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
prefix = BASE.relative_to(ROOT).as_posix()
execution = json.loads((BASE / "results/bounded-checks.json").read_text())
assert execution["all_assertions_passed"] is True
env = {str(k): str(v) for k, v in execution["environment"].items()}
commit = env["torch_git_commit"]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def artifact_id(path):
    return "a-" + re.sub(r"[^a-zA-Z0-9_-]", "-", path)

original_command = ".venv/bin/python -I docs/review-tools/section_facts.py course/chapters/05.md#5.5 --output /tmp/phase4-5_5-facts-20261005 --execute --timeout 45"
bounded_command = "timeout 45s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_5-independent/code/bounded_checks.py > docs/technical-reviews/artifacts/phase4-5_5-independent/results/bounded-stdout.txt 2> docs/technical-reviews/artifacts/phase4-5_5-independent/results/bounded-stderr.txt"
execution_files = {
    "original/execution.json": (original_command, "Exit 0; original 1 Python fence, stdout tensor([1.9600]); no guard events."),
    "original/stdout.txt": (original_command, "Original fence output tensor([1.9600]); assertion passed."),
    "original/stderr.txt": (original_command, "Exit 0; empty stderr."),
    "results/bounded-checks.json": (bounded_command, "Exit 0; all assertions passed for original and both exercises, Decimal arithmetic, L2/AdamW state, SGD control, task-loss counterexample, prior-history boundary, explicit groups and each small-fence API."),
    "results/bounded-stdout.txt": (bounded_command, "Exit 0; raw stdout JSON agrees with bounded-checks.json."),
    "results/bounded-stderr.txt": (bounded_command, "Exit 0; empty stderr."),
    "results/installed-environment.json": ("OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -I docs/technical-reviews/artifacts/phase4-5_5-independent/code/inspect_installed.py", "Exit 0; PyTorch 2.14.1+cpu, CPU, CUDA unavailable; exact installed implementation snapshots and hashes."),
}
artifacts = []
for path in sorted(BASE.rglob("*")):
    if not path.is_file() or "__pycache__" in path.parts:
        continue
    relative = path.relative_to(BASE).as_posix()
    if relative.startswith("results/checker") or relative == "results/artifact-manifest.json":
        continue
    kind = "code" if path.suffix == ".py" else "source_snapshot"
    if relative in execution_files:
        kind = "execution"
    item = {"id": artifact_id(relative), "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path), "kind": kind, "description": "Permanent independent 5.5 evidence: " + relative}
    if relative in execution_files:
        command, result = execution_files[relative]
        item.update(command=command, result=result, environment=env)
    artifacts.append(item)

def official(identifier, title, module, snapshot, note):
    return {"id": identifier, "kind": "official_source", "title": title, "verified": True,
            "url": f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{module}",
            "version": "PyTorch upstream commit " + commit + "; installed CPU wheel 2.14.1+cpu",
            "authority_reason": "Original implementation/docstring from the PyTorch project's official pytorch/pytorch repository, at the exact commit reported by the installed CPU wheel.",
            "accessed_on": "2026-10-05", "checked_original": True, "inspection_note": note,
            "snapshot_path": prefix + "/" + snapshot, "snapshot_sha256": sha(BASE / snapshot)}

sources = [
    {"id": "paper-adamw", "kind": "paper", "title": "Decoupled Weight Decay Regularization — Loshchilov & Hutter", "verified": True,
     "url": "https://arxiv.org/pdf/1711.05101v3", "version": "arXiv:1711.05101v3, 4 January 2019; ICLR 2019",
     "authority_reason": "Authors' original versioned paper fetched directly from arXiv; title, authors, printed arXiv ID/version/date and ICLR publication statement personally checked.",
     "accessed_on": "2026-10-05", "checked_original": True,
     "inspection_note": "Read title/version, Abstract/Introduction and §2 printed pp.1–4; Eq.(1), Proposition 1, Algorithm 2, Proposition 2 and explanatory paragraphs. Rendered and actually viewed PDF p.3 to distinguish purple L2 gradient term from green decoupled decay term. Paper's schedule/decay coefficient convention is not equated literally with PyTorch API symbols.",
     "snapshot_path": prefix + "/sources/adamw-1711.05101v3.pdf", "snapshot_sha256": sha(BASE / "sources/adamw-1711.05101v3.pdf")},
    official("torch-adamw", "PyTorch AdamW exact official implementation and algorithm documentation", "torch/optim/adamw.py", "sources/pytorch-adamw-installed-commit.py", "Personally read AdamW constructor lines 19–48, doc formula 59–108 and functional dispatch 158–179. SHA matches installed adamw.py exactly; decay is decoupled and lr-scaled, with m/v initialized to zero."),
    official("torch-adam", "PyTorch Adam/AdamW update internals at installed commit", "torch/optim/adam.py", "sources/pytorch-matching-torch--optim--adam.py", "Personally read _init_group 150–189, step 214–272, _single_tensor_adam 413–475 and 528–546: only non-None grads enter update, decoupled mul_(1-lr*weight_decay) precedes moment updates, coupled L2 term enters grad, and biased moments feed corrected step. Downloaded file is byte-identical to installed adam.py."),
    official("torch-optimizer", "PyTorch Optimizer gradient clearing and parameter-group contracts", "torch/optim/optimizer.py", "sources/pytorch-matching-torch--optim--optimizer.py", "Read params contract 267–269, zero_grad doc/body 1048–1093 and add_param_group 1127–1187. Docs explicitly distinguish grad=0 from None; groups carry explicit per-group options. Byte matches installed file."),
    official("torch-parameter", "PyTorch nn.Parameter official definition", "torch/nn/parameter.py", "sources/pytorch-matching-torch--nn--parameter.py", "Read Parameter 30–57: Tensor subclass, registration behavior and requires_grad=True default. Byte matches installed file."),
    official("torch-small-api", "PyTorch official tensor, zeros_like and allclose documentation source", "torch/_torch_docs.py", "sources/pytorch-matching-torch--_torch_docs.py", "Read torch.tensor 9582–9610 (copy creates leaf), zeros_like 12647–12677 (same size/dtype/device, zero-filled, requires_grad=False), allclose 834–865 (rtol=1e-5/atol=1e-8 and criterion). Exact commit matches execution wheel's git identity; these doc contracts were also independently exercised."),
    official("torch-detach", "PyTorch Tensor.detach official docstring source", "torch/_tensor.py", "sources/pytorch-matching-torch--_tensor.py", "Read detach 798–813: returned tensor does not require grad and shares storage; local API check exercised both properties."),
    {"id": "bert-groups", "kind": "official_source", "title": "Google Research BERT original optimization policy", "verified": True,
     "url": "https://raw.githubusercontent.com/google-research/bert/eedf5716ce1268e56f0a50264a88cafad334ac61/optimization.py",
     "version": "google-research/bert commit eedf5716ce1268e56f0a50264a88cafad334ac61",
     "authority_reason": "Original BERT authors' official Google Research repository, pinned to immutable commit independently obtained through the repository's official GitHub commit API; inspected original source, not a copied interpretation.",
     "accessed_on": "2026-10-05", "checked_original": True,
     "inspection_note": "Read header, create_optimizer 59–65, apply_gradients 108–157 and _do_use_weight_decay 159–167. Explicit excludes LayerNorm/layer_norm/bias; decay is outside m/v update. Only existence of that exclusion policy is used; TensorFlow BERT optimizer was not executed and its bias-correction behavior is not claimed identical to PyTorch.",
     "snapshot_path": prefix + "/sources/bert-optimization.py", "snapshot_sha256": sha(BASE / "sources/bert-optimization.py")},
    {"id": "validation-doc", "kind": "official_docs", "title": "scikit-learn official Cross-validation: evaluating estimator performance", "verified": True,
     "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst",
     "version": "scikit-learn 1.7.2 documentation source tag",
     "authority_reason": "Official scikit-learn maintained documentation source; supplies validation/test separation and hyperparameter-selection contract, independent of this optimizer's implementation.",
     "accessed_on": "2026-10-05", "checked_original": True,
     "inspection_note": "Personally read lines 10–98, especially 60–70 on hyperparameter tuning risk and held-out validation before final test evaluation. Source supports choosing a hyperparameter using validation; it supplies no benchmark for this scalar example.",
     "snapshot_path": prefix + "/sources/sklearn-cross-validation-1.7.2.rst", "snapshot_sha256": sha(BASE / "sources/sklearn-cross-validation-1.7.2.rst")},
    {"id": "own-execution", "kind": "execution", "title": "Fresh bounded scalar CPU checks", "verified": True, "artifact_id": artifact_id("results/bounded-checks.json")},
    {"id": "original-execution", "kind": "execution", "title": "Original 5.5 fence execution under section_facts CPU/offline guard", "verified": True, "artifact_id": artifact_id("original/execution.json")},
    {"id": "own-derivation", "kind": "derivation", "title": "Independent Decimal calculation and quadratic counterexample", "verified": True,
     "details": "For fresh m=v=0 and g=0, Adam direction is zero. Decimal: 1−0.1*0.2=0.98; 2*0.98=1.96; 2−1.96=0.04. For task f=(w−2)^2, f'(2)=0 but f(1.96)=0.0016>f(2)=0. For penalty (λ/2)w² with λ=.2, gradient .4 enters m=.04/v=.00016; first Adam step w=2−.1*.4/(.4+1e-8)=1.9000000025. Full actual CPU/assertion evidence is in bounded_checks.py and bounded-checks.json."},
]

def evidence(source, locator, supports):
    return {"source_id": source, "locator": locator, "supports": supports}

def verification(expected, observed, details, tolerance=None):
    value = {"method": "executed", "expected": expected, "observed": observed, "details": details}
    if tolerance is not None:
        value["tolerance"] = tolerance
    return value

bounded = artifact_id("results/bounded-checks.json")
original_exec = artifact_id("original/execution.json")
code = artifact_id("code/bounded_checks.py")
claims = [
    {"id": "c1-decoupled-update", "kind": "concept", "status": "verified",
     "statement": "權重衰減是與任務梯度方向分開的回拉；本段 PyTorch AdamW 先 w←(1−ηλ)w，再減去 Adam 方向步子。",
     "location": "course/chapters/05.md:152–154",
     "scope": "本節 η 是 PyTorch lr、λ 是 weight_decay；精確因子由官方 API 原碼支持。原论文 Algorithm 2 将基础学习率 α 和排程 η_t 分开，衰减 λ 的数值约定不同；不声称符号逐字相同。新建 optimizer 的零历史让本例方向为零。",
     "evidence": [evidence("paper-adamw", "§2 printed pp.2–3; Eq.(1), Algorithm 2 lines 6–12 and paragraph after Proposition 2; rendered PDF p.3", "权重衰减独立于损失梯度缩放/m/v；图中紫/绿项分别属于 L2/AdamW，两者分开。"), evidence("torch-adamw", "AdamW.__doc__ lines 60–89; constructor 35–48", "官方算法在 m/v 更新之外先减 γλθ；AdamW 设置 decoupled_weight_decay=True。"), evidence("torch-adam", "_single_tensor_adam lines 413–419, 456–475, 528–546", "实际执行顺序为参数乘 1-lr*weight_decay，再按任务梯度更新 m/v 与 Adam 步子。")],
     "artifact_ids": [artifact_id("sources/adamw-1711.05101v3.pdf"), artifact_id("sources/adamw-algorithm2-page3.png"), artifact_id("sources/pytorch-adamw-installed-commit.py"), artifact_id("sources/pytorch-matching-torch--optim--adam.py"), bounded]},
    {"id": "c2-scalar-numbers", "kind": "numeric", "status": "verified",
     "statement": "w=2、lr=.1、weight_decay=.2，新建 AdamW 且有零梯度时，因子 .98，剩 1.96，缩小 .04。",
     "location": "course/chapters/05.md:154, 160–165, 168",
     "scope": "一格 float32 参数第一步；没有任务、数据或模型成绩。数字按实际原码的全零初始 m/v 解释。",
     "evidence": [evidence("own-derivation", "Decimal factor/remaining/shrink derivation; bounded-checks.json checks.arithmetic", "独立代入 ηλ、原始权重与缩小量。"), evidence("original-execution", "original/execution.json exit_code=0; original/stdout.txt; attempted_fences=[1]", "原 fence 不修改而真正执行，打印 tensor([1.9600]) 且原断言成功。"), evidence("own-execution", "fence_runs[0], checks.arithmetic", "复执行原 bytes；观察精确 float32 值与 m=v=0；独立 Decimal 核对。")],
     "artifact_ids": [original_exec, bounded, code],
     "verification": verification("因子=.98、值=1.96、差=.04；fresh m=v=0", "1.9600000381469727；abs error 3.814697269177714e-8；Decimal .98/1.96/.04 精确", "原 fence 与独立原样执行均通过。无单位/分母混淆；单参数单步；打印四小数不代替实际数值。", "独立 abs<=1e-6；原 torch.allclose 的 rtol=1e-5、atol=1e-8")},
    {"id": "c3-original-software", "kind": "software", "status": "verified",
     "statement": "程式建立一格可调参数，zeros_like 给同形状零梯度，step 真正更新参数；detach/allclose 用于读取与核对。",
     "location": "course/chapters/05.md:157–165, 168",
     "scope": "逐项覆盖 import torch/from torch import nn、torch.tensor、nn.Parameter、AdamW 参数列表/lr/weight_decay、w.grad 赋值、zeros_like、step、detach、print、allclose/assert；原程式是手动给梯度后的单步参数更新，没有 backward 或训练。",
     "evidence": [evidence("torch-parameter", "Parameter lines 30–57", "Parameter 是 Tensor subclass，预设 requires_grad=True。"), evidence("torch-small-api", "torch.tensor lines 9582–9610; zeros_like 12647–12677; allclose 834–865", "建立一元素 tensor；同形状/型别/装置零 tensor；allclose 检查误差界。"), evidence("torch-adamw", "AdamW constructor lines 19–48; Args 99–108", "参数 iterable 与显式 lr/weight_decay 被传给 decoupled Adam。"), evidence("torch-adam", "step lines 214–272; actual parameter mul_/addcdiv_ lines 419, 546", "step 调度并实际就地修改参数。"), evidence("torch-detach", "Tensor.detach lines 798–813", "detach 返回无梯度追踪且共享 storage 的读取视图。"), evidence("original-execution", "original stdout/execution; original/fence-1.py exact bytes", "整个 fence 原样执行，print 与断言均成功。"), evidence("own-execution", "checks.api_contracts and fence_runs", "运行每个 API 契约并验证 allclose 会拒绝未缩小的2。")],
     "artifact_ids": [original_exec, bounded, code, artifact_id("original/fence-1.py")],
     "verification": verification("一元素 CPU float32 Parameter；零梯度与 detach 契约吻合；step 更新为约1.96并通过 assert", "全部契约断言通过；Parameter is_leaf/requires_grad=True，zeros 同 shape/dtype/device、requires_grad=False；detach不追踪梯度且共享storage", "实际执行原始 fence、必要练习变体与 API check；没有以语法检查或模拟程序代替执行。")},
    {"id": "c4-zero-vs-none-exercises", "kind": "software", "status": "verified",
     "statement": "本段 PyTorch AdamW 看到 grad=None 会跳过参数；零梯度仍可衰减。新建 optimizer 下改衰减0或改 grad=None，w 都维持2。",
     "location": "course/chapters/05.md:170, 174; linked course/chapters/01.md#1.13",
     "scope": "PyTorch 2.14.1+cpu，实际代码/练习各从 w=2 的新建 optimizer 开始；不能泛化为有历史时零梯度也只有衰减。暖状态边界另做独立检查，None 连历史都不推进。",
     "evidence": [evidence("torch-adam", "_init_group lines 150–189; step 238–269", "只有 p.grad is not None 的参数进入 params_with_grad 和状态初始化/更新。"), evidence("torch-optimizer", "zero_grad documentation lines 1048–1063; body 1082–1093", "官方明确 0 与 None 的 optimizer step 行为不同；支持与1.13清梯度的区别。"), evidence("own-execution", "fence_runs[1] and [2], checks.history_boundary", "教材两项实际修改并调整 expected assert：no-decay w=2/state step1；None w=2/state{}；暖状态零梯度仍有动量，但None保持参数及历史。")],
     "artifact_ids": [bounded, code, artifact_id("code/fence-no-decay.py"), artifact_id("code/fence-none-gradient.py"), artifact_id("original/context-1.13.md")],
     "verification": verification("no-decay + zero gradient ->2；decay .2 + None ->2且参数被跳过", "两个局部原码变体输出 tensor([2.])、断言通过；None 无 state；暖状态None亦不变化", "没有只是读取源码推断；实际执行两个练习，并额外核验既有动量不适用本节的'只衰减'情境。")},
    {"id": "c5-l2-enters-history", "kind": "concept", "status": "verified",
     "statement": "把权重平方惩罚加进任务代价，其梯度进入 Adam 的 m/v 并被缩放；AdamW 的衰减与这份历史分开。",
     "location": "course/chapters/05.md:172 first sentence",
     "scope": "成熟方法由原论文和官方原码支持；自己的单步反例只检查机制。显式惩罚用(λ/2)w²使梯度λw，不把正则化系数命名差异误作新结论。",
     "evidence": [evidence("paper-adamw", "§2 printed pp.2–3; Proposition 1/2; Algorithm 2 lines 6–12; paragraph after Proposition 2", "L2梯度与任务梯度先相加后自适应缩放；decoupled只自适应任务梯度；标准SGD的等价性有自己的范围。"), evidence("torch-adam", "_single_tensor_adam lines 416–428 then 456–475; AdamW constructor sets decoupled flag", "coupled分支把λp加到grad后更新m/v；decoupled直接改变p且不加入grad。"), evidence("own-execution", "checks.l2_vs_adamw and checks.sgd_control", "实际backward给total grad=.4；Adam m=.04/v=.00016、w=1.9000000025；AdamW m=v=0、w≈1.96；plainSGD控制为1.96。")],
     "artifact_ids": [artifact_id("sources/adamw-1711.05101v3.pdf"), artifact_id("sources/pytorch-matching-torch--optim--adam.py"), bounded, code]},
    {"id": "c6-loss-and-validation", "kind": "concept", "status": "verified",
     "statement": "缩小有用的大权重可能暂时提高任务代价；衰减强度应使用验证资料选择。",
     "location": "course/chapters/05.md:172 second sentence",
     "scope": "'可能'仅需存在反例，未声称本节真实训练/验证质量。验证资料选择是超参数评估原则；没有测试集选参或一项固定最佳强度的主张。",
     "evidence": [evidence("torch-adamw", "AdamW doc algorithm lines 75–89; weight_decay argument 108", "独立衰减即使任务梯度为零仍改变参数；weight_decay为显式可选超参数。"), evidence("own-derivation", "f(w)=(w−2)^2 at w=2 then w=1.96", "任务梯度0处减小参数可把任务代价0提升为.0016的构造反例。"), evidence("own-execution", "checks.task_loss_can_increase", "按实际缩小后float32值计算task loss，真实观察0→0.0015999969482436427。"), evidence("validation-doc", "official cross_validation.rst tag1.7.2 lines 60–70", "比较超参数时用held-out validation，最终test评估独立。")],
     "artifact_ids": [bounded, code, artifact_id("sources/sklearn-cross-validation-1.7.2.rst")]},
    {"id": "c7-explicit-exclusion-policy", "kind": "concept", "status": "verified",
     "statement": "bias或正规化倍率有时另外分组并不衰减，规则必须明确。",
     "location": "course/chapters/05.md:172 third sentence",
     "scope": "'有时'表示政策选项；不宣称普遍最优或 PyTorch AdamW 默认自动排除。BERT权威例证展示排除bias/LayerNorm；PyTorch可通过显式 groups落实。",
     "evidence": [evidence("bert-groups", "create_optimizer lines59–65; _do_use_weight_decay lines159–167", "Google原始BERT实现明确排除LayerNorm/layer_norm/bias，证明此政策实际存在。"), evidence("torch-optimizer", "_params_doc lines267–269; add_param_group doc1133–1135 and defaults1181–1187", "明确参数组与每组optimization options的正式API契约。"), evidence("own-execution", "checks.explicit_groups", "真正执行AdamW组：weight_decay .2的weight→1.96，weight_decay0的bias/norm_scale→2。")],
     "artifact_ids": [artifact_id("sources/bert-optimization.py"), artifact_id("sources/pytorch-matching-torch--optim--optimizer.py"), bounded, code]},
]

report = {
    "schema_version": 1, "review_stage": "technical", "lesson_id": "5.5", "source": "course/chapters/05.md#5.5",
    "source_sha256": sha(BASE / "original/section.md"), "figure_sha256": {},
    "reviewer_task": "/root/phase4_factual_coordinator/factual_5_5", "reviewer_context": "fresh", "author_tasks": [],
    "verdict": "pass", "reviewed_on": "2026-10-05", "issues": [],
    "read_scope": {"selected": "5.5 complete raw section lines150–181", "related": "Chapter05 lines60–149 (end5.2,5.3,5.4), complete1.13, method/schema/helper/review protocol, raw bootstrap contract and personally read primary-source passages listed below", "chapter_introduction": "Not read; 5.5 is not the chapter's first section", "legacy_reports": "No previous technical/reader report body or conclusions read; locator-only PDF index used only for paths/hash/printed IDs."},
    "artifacts": artifacts, "sources": sources, "claims": claims,
    "checks": {
        "factual_accuracy": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "Seven substantive claims independently covered by primary originals, bounded CPU execution and explicit support scope. Weight decay vs squared-loss penalty and AdamW's coefficient convention match current text; no unresolved substantive discrepancy."},
        "numeric_verification": {"status": "pass", "claim_ids": ["c2-scalar-numbers"], "details": "Original fence executed unchanged; independent Decimal .98/1.96/.04 and actual float32 error checked. Both exercises executed with corrected expectations; bounded state, gradient and task-loss checks passed. No empirical benchmark JSON/sample/token/step claims are present to verify."},
        "figure_consistency": {"status": "not_applicable", "claim_ids": [], "details": "Extraction found zero SVG or embedded image references in 5.5. Scalar decay/grad-state distinction has no image geometry or correspondence requiring course figure. No desktop/mobile page-render claim. Original paper PDF page3 was actually rendered and viewed for colored algorithm-term reading, retained as source evidence."},
        "source_verification": {"status": "pass", "claim_ids": ["c1-decoupled-update", "c3-original-software", "c4-zero-vs-none-exercises", "c5-l2-enters-history", "c6-loss-and-validation", "c7-explicit-exclusion-policy"], "details": "Fresh direct verified-HTTPS originals: arXiv v3 with title/version personally checked; official PyTorch matching wheel commit (four installed source files byte-equal), pinned Google BERT source, official scikit-learn validation docs. Each claim has original passage/API/line locators and supports; no source-summary reuse."},
        "limitations": {"status": "pass", "claim_ids": [c["id"] for c in claims], "details": "Only short CPU scalar executions; no datasets, weights, GPU, full training or uploads. Zero-gradient-only-decay claim is specific to fresh optimizer state as the text/fence shows; warmed-history countercheck recorded. Coefficient convention belongs to PyTorch API, not literal paper symbol equivalence. Exclusions are explicit choices, no universal best-policy or generalization claim. No legacy empirical JSON referenced in this section; no fabrication or retraining. No course image rendering applicability and no browser layout verification claimed."},
    },
}
(BASE / "results/artifact-manifest.json").write_text(json.dumps({"files": [{"path": a["path"], "sha256": a["sha256"]} for a in artifacts]}, indent=2) + "\n")
(ROOT / "docs/technical-reviews/5.5.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": "docs/technical-reviews/5.5.json", "verdict": report["verdict"], "claims": len(claims), "artifacts": len(artifacts), "source_sha256": report["source_sha256"]}, indent=2))
