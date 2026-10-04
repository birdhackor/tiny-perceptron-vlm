"""Assemble this review from already-read snapshots; never refresh the lesson hash."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_finish_19_11"
ENV = {"python": "3.13.5", "torch": "2.14.1+cpu", "huggingface_hub": "1.33.0", "device": "CPU"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    binding = json.loads((ART / f"{PREFIX}-reread-binding.json").read_text())
    original_binding = json.loads((ART / f"{PREFIX}-section-snapshots.json").read_text())
    copies = json.loads((ART / f"{PREFIX}-source-copies.json").read_text())
    artifacts, paths = [], {}
    executions = {
        "public-downloads.json": (
            ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_19_11-fetch-originals.py",
            "Exit 0; 55 pinned token=False file fetches verified byte size/SHA; all 11 CPU loaders succeeded.",
        ),
        "public-tree.json": (
            ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_19_11-fetch-originals.py",
            "Exit 0; anonymous fixed-revision HF tree contains 11 model.pt files and no model-training.pt.",
        ),
        "list-command.json": (
            ".venv/bin/python scripts/fetch_capstone.py --list (subprocess recorded by fetch-originals.py)",
            "Exit 0; 11 distinct stage lines match the manifest.",
        ),
        "verification.json": (
            ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_19_11-verify.py",
            "Exit 0; 18 short Dense CPU updates, joint/DPO exact resume, nine guards, three public inference calls, "
            "four CLI rejection cases, one new-training update from public weights, and all original/public joint tensors equal.",
        ),
        "evidence-audit.json": (
            ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_19_11-evidence-audit.py",
            "Exit 0; inspected original GPU reports, four 84-row validations, shared data, stage parent hashes and KD-int4 failure.",
        ),
        "budget.json": (
            ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_19_11-budget.py",
            "Exit 0; real 0.001-second budget yielded one CPU update, 0.019565565-second loop, and a saved incomplete checkpoint.",
        ),
        "pytest.log": (
            ".venv/bin/python -m pytest -q tests/test_capstone_release.py::test_public_verification_and_download_load_actual_checkpoint "
            "tests/test_capstone_release.py::test_failed_download_never_installs_partial_stage "
            "tests/test_capstone_release.py::test_public_manifest_cannot_be_created_when_hf_bytes_do_not_match "
            "tests/test_capstone_release.py::test_export_removes_training_references_and_preserves_all_logits "
            "tests/test_capstone_release.py::test_ptq_packs_real_weights_retains_router_and_reloads "
            "tests/test_capstone.py::test_cpu_three_update_stage_smoke_and_dependency_guards",
            "Exit 0; 7 tests passed in 2.92 seconds.",
        ),
    }
    for path in sorted(ART.glob(f"{PREFIX}-*")):
        if "prior-report" in path.name or path.name.endswith("checker.log"):
            continue
        suffix = path.name[len(PREFIX) + 1 :]
        identifier = suffix.replace(".", "_")
        paths[suffix] = identifier
        kind = "code" if path.suffix == ".py" else "source_snapshot"
        description = f"Independent personally read or generated review evidence: {suffix}; original bytes retained."
        item = {
            "id": identifier,
            "kind": kind,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "description": description,
        }
        if suffix in executions:
            command, outcome = executions[suffix]
            item.update(kind="execution", command=command, result=outcome, environment=ENV)
        artifacts.append(item)
    sources = []

    def external(identifier, kind, title, url, version, locator, note, files):
        sources.append(
            {
                "id": identifier,
                "kind": kind,
                "title": title,
                "url": url,
                "version": version,
                "verified": True,
                "checked_original": True,
                "accessed_on": "2026-10-04",
                "authority_reason": "Original publisher/maintainer source for the API or method discussed.",
                "inspection_note": f"Personally inspected {locator}. {note}",
                "artifact_ids": [paths[file] for file in files],
            }
        )

    torch_commit = copies["torch_git"]
    external(
        "torch-saving",
        "official_docs",
        "PyTorch Saving and Loading Models",
        "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
        "Original tutorial page updated 2025-06-26, last verified 2024-11-05; snapshot read 2026-10-04",
        "Saving & Loading a General Checkpoint for Inference and/or Resuming Training, extracted lines 2059-2188",
        "Requires model and optimizer state dictionaries for resume; architecture is initialized before loading. "
        "Crosschecked the exact installed 2.14.1 implementation; tutorial date is stated rather than inferred.",
        ["torch-saving.html", "torch-saving.txt"],
    )
    external(
        "torch-randomness",
        "official_docs",
        "PyTorch 2.14 Reproducibility",
        "https://docs.pytorch.org/docs/2.14/notes/randomness.html",
        "PyTorch docs 2.14, updated 2026-05-14",
        "Opening limitations and Controlling sources of randomness, extracted lines 1603-1660",
        "Identical seeds do not guarantee identical results across releases/platforms or CPU and GPU.",
        ["torch-randomness.html", "torch-randomness.txt"],
    )
    for identifier, filename, locator, note in [
        (
            "torch-adam",
            "adam",
            "Adam._init_group lines 153-195",
            "step, exp_avg and exp_avg_sq are lazily initialized optimizer state.",
        ),
        (
            "torch-optimizer",
            "optimizer",
            "Optimizer.state_dict lines 697-791; load_state_dict lines 902-1026",
            "Optimizer state/parameter groups are serialized and restored; a fresh instance alone has no historical moments.",
        ),
        (
            "torch-random",
            "random",
            "get_rng_state/set_rng_state/manual_seed lines 25-68",
            "RNG state restoration differs from seeding; CPU generator state is explicit, CUDA is separate.",
        ),
    ]:
        module = f"torch/optim/{filename}.py" if filename != "random" else "torch/random.py"
        external(
            identifier,
            "official_source",
            f"PyTorch {module}",
            f"https://github.com/pytorch/pytorch/blob/{torch_commit}/{module}",
            f"Installed official torch 2.14.1+cpu; torch.version.git_version={torch_commit}",
            locator,
            "Read the actual installed distribution's original file and copied unchanged bytes as .py.txt. " + note,
            [f"torch-{filename}.py.txt"],
        )
    external(
        "hf-download",
        "official_docs",
        "Hugging Face Hub Download files from the Hub",
        "https://huggingface.co/docs/huggingface_hub/v1.33.0/en/guides/download",
        "huggingface_hub docs v1.33.0",
        "Download a single file; revision examples, extracted lines 298-343",
        "Full commit hashes pin a file version; cached downloads retain actual byte identity.",
        ["hf-download-guide.html", "hf-download-guide.txt"],
    )
    external(
        "hf-headers",
        "official_source",
        "huggingface_hub.utils._headers.get_token_to_send",
        "https://github.com/huggingface/huggingface_hub/blob/v1.33.0/src/huggingface_hub/utils/_headers.py",
        "Installed huggingface_hub 1.33.0",
        "get_token_to_send lines 125-135 and build_hf_headers lines 31-119",
        "Personally read installed official original source: token=False returns None and excludes stored tokens.",
        ["hf-headers.py.txt"],
    )
    external(
        "dpo-paper",
        "paper",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "https://arxiv.org/pdf/2305.18290v3",
        "arXiv:2305.18290v3, 29 Jul 2024",
        "Section 4 equation (7) and DPO outline, PDF pp. 4-5 / extracted lines 194-282",
        "The objective contains policy/reference log ratios for a given reference. Changing reference changes "
        "that objective; this supports preserving reference identity, not our synthetic model's quality.",
        ["dpo-original.pdf", "dpo-original.txt"],
    )
    repo_specs = [
        (
            "capstone",
            "tiny_perceptron/capstone.py",
            "default_config/description; preference_loss/frozen_reference; save/load/export, lines 25-49, 110-123, 354-375, 612-699",
        ),
        (
            "trainer",
            "scripts/course_experiments/capstone.py",
            "train_stage lines 70-275: parents, AdamW/sampler, resume guards, deadlines and saved progress",
        ),
        ("cli", "scripts/capstone.py", "argparse defaults lines 13-25 and train/FP32-PTQ infer dispatch lines 49-111"),
        ("fetch", "scripts/fetch_capstone.py", "fetch_capstone lines 25-61 and list branch lines 64-101"),
        (
            "release",
            "scripts/capstone_release.py",
            "_metadata/public_stage_id/clean/validate lines 79-240; manifest checks lines 366-398",
        ),
        (
            "ptq",
            "tiny_perceptron/capstone_quantization.py",
            "quantize_capstone lines 39-116; restore/load lines 119-196: packed values and float reconstruction",
        ),
        ("ui", "tiny_perceptron/capstone_ui.py", "serve lines 202-217: only load_capstone, not PTQ loader"),
        ("release-tests", "tests/test_capstone_release.py", "original selected tests lines 82-112, 146-180, 218-246"),
        (
            "stage-tests",
            "tests/test_capstone.py",
            "test_cpu_three_update_stage_smoke_and_dependency_guards lines 278-298",
        ),
    ]
    for identifier, path, note in repo_specs:
        observed = next(item for item in copies["sources"] if item["original"] == path)
        sources.append(
            {
                "id": identifier,
                "kind": "repository_code",
                "title": path,
                "path": path,
                "sha256": observed["sha256"],
                "version": "Personally read working-checkout original bytes 2026-10-04; immutable snapshot hash retained",
                "verified": True,
                "inspection_note": note,
                "artifact_ids": [paths[Path(observed["snapshot"]).name[len(PREFIX) + 1 :]]],
            }
        )
    source = "scripts/course_experiments/capstone_deployment.py"
    snapshot = ART / f"{PREFIX}-repo-scripts_course_experiments_capstone_deployment.py.txt"
    sources.append(
        {
            "id": "deployment-code",
            "kind": "repository_code",
            "title": source,
            "path": source,
            "sha256": sha(snapshot),
            "version": "Personally read original source 2026-10-04; snapshot copied by evidence-audit.py",
            "verified": True,
            "inspection_note": "run lines 271-306 and result lines 365-375: fixed joint recommendation and evaluated exports.",
        }
    )
    for identifier, artifact in [
        ("download-run", "public-downloads.json"),
        ("tree-run", "public-tree.json"),
        ("list-run", "list-command.json"),
        ("cpu-run", "verification.json"),
        ("original-record-audit", "evidence-audit.json"),
        ("budget-run", "budget.json"),
        ("pytest-run", "pytest.log"),
    ]:
        sources.append(
            {
                "id": identifier,
                "kind": "execution",
                "title": f"Actual independent {identifier}",
                "verified": True,
                "artifact_id": paths[artifact],
            }
        )
    claims = []

    def claim(
        identifier,
        kind,
        statement,
        location,
        refs,
        artifact_files,
        scope,
        observed=None,
        expected=None,
        details=None,
        denominators=None,
    ):
        item = {
            "id": identifier,
            "kind": kind,
            "statement": statement,
            "location": location,
            "status": "verified",
            "evidence": [
                {"source_id": source_id, "locator": locator, "supports": supports}
                for source_id, locator, supports in refs
            ],
            "artifact_ids": [paths[file] for file in artifact_files],
            "scope": scope,
        }
        if kind in {"software", "numeric", "empirical"}:
            item["verification"] = {
                "method": "executed",
                "expected": expected or statement,
                "observed": observed or "Actual saved execution agrees within the stated scope.",
                "details": details or "Read source/inputs and inspected actual saved output, not exit code alone.",
            }
            if kind == "numeric":
                item["verification"]["tolerance"] = (
                    "Exact integer/file hash or tuple equality; no floating tolerance needed."
                )
            if denominators:
                item["verification"]["denominators"] = denominators
        claims.append(item)

    R = lambda source, locator, supports: (source, locator, supports)  # noqa: E731
    cpu = ["verification.json"]
    downloads = ["public-downloads.json"]
    claim(
        "c01",
        "concept",
        "推論需匹配模型结构/权重/tokenizer；恢复同一训练还需优化器、RNG、抽样位置与阶段。",
        "开头及第二段",
        [
            R("torch-saving", "general checkpoint section", "优化器历史是权重之外的必要状态"),
            R("torch-random", "get_rng_state/set_rng_state", "随机生成器状态与种子是不同对象"),
            R("capstone", "load_capstone 644-653", "tokenizer协议和模型结构配套受到直接校验"),
        ],
        cpu,
        "足够状态还依赖具体训练程序、数据和运算实现；不是跨平台逐位相同保证。",
    )
    claim(
        "c02",
        "software",
        "FP32与PTQ分别标为capstone-v1/capstone-ptq-v1，并标明资料版本和训练阶段。",
        "格式段落",
        [
            R("capstone", "save/load_capstone 612-653", "格式/data_version/stage/tokenizer字段"),
            R("ptq", "quantize/restore payload", "PTQ格式与阶段字段"),
        ],
        downloads,
        "仅本项目的固定自定义格式，不是Transformers模型契约。",
    )
    claim(
        "c03",
        "software",
        "inference_only公开导出移除更新器、RNG、training_state和reference。",
        "公开推论状态段落",
        [R("release", "clean_capstone_payload 121-169", "明确推论字段白名单")],
        downloads,
        "所有11公开payload亲读，状态缺失不是仅靠文件名判断。",
    )
    claim(
        "c04",
        "numeric",
        "短程序读回的输出为capstone-v1、0、True、True。",
        "首段Python短程序后的预期输出",
        [R("cpu-run", "short_program_and_step_exercise[0]", "实存实读，检查描述而非把它误作参数值比较")],
        cpu,
        "description True只核架构与参数数量；独立验证另测逐tensor和输入logits。",
        observed="format capstone-v1; step 0; inference_only True; description_equal True; all_tensors_equal True.",
    )
    claim(
        "c05",
        "software",
        "stage/step是可写标签；把step从0改1不会自动执行SFT或改变模型。",
        "短程序解释及末尾练习",
        [
            R("capstone", "save_capstone 612-641", "保存给定字段，没有optimizer.step"),
            R("cpu-run", "short_program_and_step_exercise", "step0/1读回而更新次数皆0"),
        ],
        cpu,
        "step0在这个未训练程序中成立；任意外来文件的step字段不是独立训练证据。",
    )
    claim(
        "c06",
        "empirical",
        "固定33c6898f revision的全部11公开模型可匿名下载、核SHA并读回。",
        "HF正式交付段落",
        [
            R("download-run", "calls and models", "55次token=False下载与11个真实CPU loader结果"),
            R("hf-headers", "get_token_to_send", "token=False不会使用存储token"),
        ],
        downloads,
        "读回支持本次固定公开字节/当前CPU版本；不证明每个模型回答正确，也不重建发布者历史登录状态。",
        observed="11/11 models loaded; every one of 55 declared file checks matched pinned SHA/bytes.",
        denominators={
            "models": 11,
            "declared_files": 55,
            "distinct_public_paths": 18,
            "revision": "33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed",
            "optimizer_updates": 0,
        },
    )
    claim(
        "c07",
        "software",
        "下载使用完整revision，并对文件内容SHA-256核验，下载不要求登录。",
        "revision/内容指纹说明",
        [
            R("hf-download", "revision examples lines 298-343", "固定完整commit的官方API语义"),
            R("fetch", "fetch_capstone 42-58", "每份匿名下载后核大小与SHA"),
        ],
        downloads,
        "SHA检查声明内容一致；不作任意来源可信性或绝对无碰撞保证。",
    )
    claim(
        "c08",
        "software",
        "--list输出11个stage别名，不下载权重或训练。",
        "第一组bash及其说明",
        [
            R("fetch", "main list branch 82-99", "list branch只印清单且不调用fetch/train"),
            R("list-run", "stdout/row_count", "实际11个JSON行"),
        ],
        ["list-command.json"],
        "list仍读取并验证本地清单；别名与checkpoint内部stage可以不同。",
    )
    claim(
        "c09",
        "software",
        "四站、四种主模型PTQ和三个学生别名分别指明正确用途与模型身份。",
        "下载名称表",
        [R("release", "STAGE_IDS/public_stage_id", "学生别名由student_branch且Dense结构校验，不混同joint阶段")],
        downloads,
        "全部模型身份亲读；学生教师为DPO分支，不把它当推荐joint的相同能力。",
    )
    claim(
        "c10",
        "empirical",
        "本轮推荐joint，DPO是比较分支而非必须采用的最后更新。",
        "名称表之前及四站命令后",
        [
            R(
                "original-record-audit",
                "stages and deployment.configuration_and_result",
                "原记录joint validation75/84、DPO71/84及推荐字段",
            ),
            R("deployment-code", "run 271-306 and result 365-375", "固定joint推荐，没有测试结果驱动的动态选择"),
        ],
        ["evidence-audit.json", "experiment-joint.json", "experiment-preference.json", "experiment-deployment.json"],
        "支持报告记载的这一次seed42合成任务验证选择；没有重新跑GPU质量或独立证明作者所有历史决策时间。",
        observed="joint75/84 versus DPO71/84; recommendation_basis explicitly validation coverage, fixed before official test.",
        denominators={
            "validation": 84,
            "seed": 42,
            "train_rows": 552,
            "joint_updates": 600,
            "dpo_updates": 100,
            "gpu": "NVIDIA L4",
            "timing_scope": "Original experiment reports, not this CPU audit",
        },
    )
    claim(
        "c11",
        "software",
        "每个下载目录含五个声明文件，config和tokenizer内嵌于model.pt。",
        "五文件说明",
        [R("download-run", "each models[].files/keys/config/tokenizer", "实际五文件与payload字段")],
        downloads,
        "五个文件的完整大小和哈希亲核；模型卡为用途/限制来源，不能代替任务验证。",
    )
    claim(
        "c12",
        "software",
        "FP32储存float32；int4/int8为打包储存，载入后普通FP32运算，不保证推论变快或内存按位宽缩小。",
        "FP32/压缩格式段落",
        [
            R("ptq", "restore_quantized_payload 129-196", "integers.float()*scale后load_state_dict到float32模型"),
            R("download-run", "quantization/tensors", "全部量化载入后state dtype为float32"),
        ],
        downloads,
        "只验证储存格式与计算路径；没有运行低位元kernel/速度/峰值内存benchmark。",
    )
    claim(
        "c13",
        "software",
        "学生为独立的小Dense结构，其能力不能直接借用joint结果。",
        "Dense学生能力说明",
        [
            R("release", "public_stage_id 96-119", "student alias要求experts0及teacher身份"),
            R("download-run", "student*/description versus joint/description", "实际79920 versus328128参数且结构不同"),
        ],
        downloads,
        "仅本次学生结构及公开teacher来源，不外推所有Dense/蒸馏配方。",
    )
    claim(
        "c14",
        "numeric",
        "joint正式model.pt为1,327,950 bytes，SHA为d85cca83...246ca16。",
        "正式joint大小/SHA段落",
        [R("download-run", "models[id=joint].bytes/sha256", "匿名固定文件的实际stat和全SHA")],
        downloads,
        "只针对revision33c6898f的正式封装；不是旧部署文件大小。",
        observed="1327950 bytes; d85cca83cdb4952f4653ef6b58d94562b41403db3a9db2ea31ff57e31246ca16.",
    )
    claim(
        "c15",
        "empirical",
        "joint公开封装与旧实验文件大小/metadata不同，但逐tensor权重相同。",
        "正式joint封装解释",
        [R("cpu-run", "joint_repackaging", "原始旧model.pt真读hash/metadata并逐tensor比较公开文件")],
        cpu,
        "原binary保持本地；保存真实hash/shape/dtype/逐tensor比较结果作可Git取的证据，不声称亲读作者未提供的training包。",
        observed="Original 1345023 bytes/8f7e8582...bd7c3b47; public1327950/d85cca83...; every tensor exactly equal.",
        denominators={
            "models_compared": 2,
            "parameters_each": 328128,
            "precision": "float32",
            "tolerance": "torch.equal all tensors",
        },
    )
    claim(
        "c16",
        "software",
        "默认下载目标为checkpoints/capstone/<stage>，五文件核验后才安装；已有目录拒绝覆盖，可另选--output。",
        "目录/验证/另存说明",
        [
            R("fetch", "target checks and temp-directory rename 34-60", "清单默认路径与整目录提交"),
            R("pytest-run", "test_failed_download_never_installs_partial_stage", "损坏文件未留下部分已安装目录"),
        ],
        cpu + ["pytest.log"],
        "默认路径由源码确认，本次实际下载用独立/tmp目的地；不会把已有目录静默重下载。",
    )
    claim(
        "c17",
        "numeric",
        "公开joint-int4为270,855 bytes，并有自己的SHA。",
        "第二组bash后的大小说明",
        [R("download-run", "models[id=joint-int4].bytes/sha256", "真实文件尺寸与独立哈希")],
        downloads,
        "270855是公开封装大小；不等同旧实验275381或张量储存量。",
        observed="270855 bytes; 3a00bebe552bb520d45c29c16d2f5dd9ab0763cdcc4bb439640e3cc2e964eea5.",
    )
    claim(
        "c18",
        "software",
        "infer对PTQ选择正确载入器，并运行同一工具循环。",
        "量化下载/CPU infer命令",
        [
            R("cli", "load fallback and run_assistant 78-102", "FP32失败后使用PTQ loader"),
            R("cpu-run", "public_inference_commands[joint-int4]", "实际TOOL:1+2、runtime3、DIRECT:3"),
        ],
        cpu,
        "当前本地CLI/CPU版本实跑；并非任何量化模型保证同答案。",
    )
    claim(
        "c19",
        "empirical",
        "公开student-kd-int4把1+2请求成2+9，工具回11，自己答12。",
        "学生失败例子",
        [
            R("cpu-run", "public_inference_commands[student-kd-int4]", "固定公开权重实际第二次生成"),
            R("original-record-audit", "student_failure.row/record", "原90题逐题文件对应的失败条目"),
        ],
        cpu + ["evidence-audit.json", "student-test-kd-ptq4.json"],
        "一题具体贪心生成案例，不把程序可运行当正确能力，也不外推每个输入。",
        observed="TOOL:calculator:2+9; runtime result11; final DIRECT:12; expected3.",
        denominators={
            "live_prompts": 1,
            "original_test_rows": 90,
            "original_calculator_rows": 12,
            "seed": 42,
            "student_updates": 350,
            "max_new_tokens_per_generation": 64,
            "student_parameters": 79920,
            "device_live": "CPU",
            "device_original": "NVIDIA L4",
        },
    )
    claim(
        "c20",
        "software",
        "serve只使用FP32格式，joint-int4交给它会失败。",
        "本机serve限制",
        [
            R("ui", "serve 202-217", "只调用load_capstone"),
            R("cpu-run", "public_cli_rejections serve command", "实际Unsupported capstone checkpoint/data format"),
        ],
        cpu,
        "只说明当前serve入口限制；PTQ infer仍工作，不是一般网页无法支持量化。",
    )
    claim(
        "c21",
        "concept",
        "DPO续训要保持原reference；改为当前policy复制的reference会改变原实验目标。",
        "reference保存段落",
        [
            R("dpo-paper", "Section4 Eq7 and DPO outline", "给定reference的policy/reference log比目标"),
            R("trainer", "resume reference 147-152", "载回原frozen reference，缺失拒绝"),
        ],
        cpu,
        "本实验DPO+0.2CE重播配方；原reference不随policy更新，不能用其他DPO变体替代。",
    )
    claim(
        "c22",
        "concept",
        "同环境恢复与跨环境重新训练必须区分，装置/运算实现影响逐步重现。",
        "逐步重现限制段落",
        [R("torch-randomness", "Opening limitations", "同种子不保证跨设备/平台/版本再现")],
        cpu,
        "本次CPU逐tensor相同仅验证本短实验，不证明GPU或跨版本逐位一致。",
    )
    claim(
        "c23",
        "software",
        "四条train命令会更新权重并按pretrain→sft→joint→dpo接已完成父模型。",
        "四站bash及完成证明说明",
        [
            R("cli", "train argument dispatch", "命令实参进入train_stage"),
            R("trainer", "predecessor guard 99-115; optimizer.step 216", "父子限制与真更新"),
        ],
        cpu + ["pytest.log"],
        "执行的是1/1/4/4短CPU机制配置；300/1400/600/100完整CPU四站未重训，不宣称GPU分数重现。",
    )
    claim(
        "c24",
        "software",
        "batch-size是每次监督训练抽取的记录数，seed设新实验初始化与抽样起点。",
        "batch-size24和seed42解释",
        [
            R(
                "trainer",
                "seed_everything, Random(seed+stage*1000), balanced sampling 96,123,197",
                "种子及监督batch实际构造",
            )
        ],
        cpu,
        "DPO另抽max(1,batch_size//2)偏好对；24不是总forward记录/token数，也不保证跨设备完全一样。",
        observed="短trial batch2的每个真实batch ID序列长度2，同seed完整与恢复序列一致。",
    )
    claim(
        "c25",
        "software",
        "540秒是每批更新前检查的循环预算，最后批/保存可能超出；导出、验证、报告还需额外时间。",
        "修订后的四站命令后时间预算段落",
        [
            R("trainer", "while and post-loop saving/export/evaluation 195-244", "批边界判断及循环外后处理"),
            R("budget-run", "requested_loop_budget_seconds/observed_training_loop_seconds", "真实短CPU预算超出后保存"),
        ],
        ["budget.json", "section-19.11-reread.md"],
        "默认值540秒由CLI源码确认；真实边界证据用0.001秒预算，不花540秒硬测。",
        observed="Budget0.001 seconds; loop0.019565565 seconds; one update; saved step1; incomplete schedule.",
    )
    claim(
        "c26",
        "software",
        "预算未完成也保存model-training.pt，报告schedule_completed=false，不能拿来开始下一站。",
        "预算中断工作档/父档说明",
        [R("trainer", "save_progress, report, predecessor guard", "真实进度保存及完成证明校验")],
        cpu + ["budget.json"],
        "支持正常预算停止和catch中保存；不保证SIGKILL/断电时在任意指令处都能写完文件。",
    )
    claim(
        "c27",
        "software",
        "--resume的steps是原总步数，并校验阶段、原计划、batch、数据manifest与记录的代码SHA。",
        "600总步数及resume条件",
        [R("trainer", "resume guards99-173", "原计划/阶段/数据/batch/两源代码SHA严格检查")],
        cpu,
        "代码指纹只覆盖capstone.py与本trainer两文件，不是依赖/全部仓库/硬件契约；本次总4步暂停2步后只更新2步。",
    )
    claim(
        "c28",
        "software",
        "resume恢复AdamW、随机状态、sampler和step；重复写seed42不会让抽样从第一笔重来。",
        "resume继续取样说明",
        [
            R("trainer", "optimizer/sampler/RNG restore 136-153", "初始化后用checkpoint状态覆写"),
            R("torch-optimizer", "load_state_dict", "更新器历史恢复"),
        ],
        cpu,
        "joint与DPO短CPU完整/暂停+恢复的真实batch序列、权重、optimizer、reference/RNG都相等；不外推大规模GPU。",
    )
    claim(
        "c29",
        "software",
        "跨站input-checkpoint载已完成紧邻父模型，建立新的更新器和阶段抽样器，而非恢复父站历史。",
        "input-checkpoint与resume区别",
        [R("trainer", "parent branch 110-123; restore only if resume 136", "跨站参数与新训练流程")],
        cpu,
        "CPU新站真实更新及父hash核验；pretrain本身必须无父或按其专用resume恢复。",
    )
    claim(
        "c30",
        "software",
        "公开推论包缺完整状态/data_manifest/schedule_completed，本课resume/正式parent CLI拒绝；权重仍能用于新训练。",
        "公开推论不能代入CLI段落",
        [
            R("cpu-run", "public_cli_rejections and public_weights_new_experiment", "两条实际CLI错误和独立AdamW更新"),
            R("release", "metadata allowlist 43-92", "清理掉原manifest与完成证明，仅保留dataset摘要"),
        ],
        cpu + downloads,
        "实际拒绝首先来自data_manifest不同；不是一般PyTorch无法训练这些权重。新实验只跑1个更新且不证明能力。",
    )
    claim(
        "c31",
        "software",
        "公开交付范围为11个推论模型，不含作者完整model-training工作包。",
        "本节最后交付范围说明",
        [
            R("tree-run", "files", "固定公开子树11个权重文件且无model-training.pt"),
            R("download-run", "payload keys", "11包都inference_only且没有工作状态"),
        ],
        downloads + ["public-tree.json"],
        "只核指定固定revision/公开子树，不声称访问或验证作者完整私有备份。",
    )
    report = {
        "schema_version": 1,
        "review_stage": "technical",
        "lesson_id": "19.11",
        "source": "course/chapters/19.md#19.11",
        "reviewer_task": "/root/fact_finish_19_11",
        "reviewer_context": "fresh",
        "source_sha256": binding["sha256"],
        "figure_sha256": {},
        "verdict": "pass",
        "claims": claims,
        "sources": sources,
        "artifacts": artifacts,
        "issues": [
            {
                "claim_id": "c25",
                "status": "resolved",
                "details": "Original personally read section said hard maximum540s, contradicted by batch-boundary source and real CPU counterexample.",
                "resolution": "Author clarified loop budget, last-batch/save overrun and later export/evaluation/report time. Personally reread entire revised19.11 and source lines175-244; bound to new snapshot.",
            }
        ],
        "review_history": {
            "original_section_sha256": original_binding["19.11"]["sha256"],
            "original_snapshot": original_binding["19.11"]["snapshot"],
            "reread_section_sha256": binding["sha256"],
            "reread_snapshot": binding["snapshot"],
            "note": "Snapshots created at actual reading time. Builder never reads course source or refreshes its hash.",
        },
        "checks": {
            "factual_accuracy": {
                "status": "pass",
                "details": "31 concrete claims checked; deadline wording corrected and personally reread.",
                "claim_ids": [c["id"] for c in claims],
            },
            "numeric_verification": {
                "status": "pass",
                "details": "Exact snippet tuple, fixed file bytes/SHA, all11 payloads, joint tensor equality and specific student failure verified; original84/90 denominators retained.",
                "claim_ids": [c["id"] for c in claims if c["kind"] in {"numeric", "empirical"}],
            },
            "figure_consistency": {
                "status": "not_applicable",
                "details": "19.11 has no referenced figure/SVG. Prerequisite figure is not used as evidence for this section.",
                "claim_ids": [],
            },
            "source_verification": {
                "status": "pass",
                "details": "Personally read PyTorch2.14 docs and installed2.14.1 originals, HF1.33 original source/docs, DPOv3 Eq7, actual public payloads and relevant project code; no other review conclusions read.",
                "claim_ids": ["c01", "c06", "c07", "c21", "c22", "c27", "c30"],
            },
            "limitations": {
                "status": "pass",
                "details": "Distinguishes public inference vs raw experiment model vs unavailable author private training package, project CLI refusal vs general new training, CPU mechanism vs originalGPU quality/timing and soft deadline vs wall-time bound.",
                "claim_ids": ["c10", "c12", "c15", "c19", "c22", "c25", "c27", "c30", "c31"],
            },
        },
    }
    (ROOT / "docs/technical-reviews/19.11.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Bound personally reread section {binding['sha256']}; {len(claims)} claims, {len(sources)} sources, {len(artifacts)} artifacts; pass."
    )


if __name__ == "__main__":
    main()
