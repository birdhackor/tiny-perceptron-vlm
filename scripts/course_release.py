"""明確的學生檔案匯出；不把完整訓練存檔或未審閱資料直接公開。"""

import hashlib
import json
import re
from pathlib import Path

MIT_TEXT = """MIT License

Copyright (c) 2026 birdhackor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
GENERATED_NAMES = {
    "README.md",
    "export-manifest.json",
    "download-manifest.json",
    "evaluation.json",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
}


def safe_relative(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or path.as_posix() != value:
        raise ValueError("檔案需要正規化的 repo-relative 路徑")
    return path


def validate_base(base):
    if not (
        re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", base.get("repo", ""))
        and re.fullmatch(r"[0-9a-f]{40}", base.get("revision", ""))
        and re.fullmatch(r"[0-9a-f]{64}", base.get("sha256", ""))
    ):
        raise ValueError("LoRA adapter 需要公開且固定 revision/hash 的 base_checkpoint")
    safe_relative(base.get("filename", ""))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_approval(approval, experiment_id, batch_id, checkpoint_repo):
    if (
        approval.get("schema_version") != 1
        or approval.get("approved") is not True
        or approval.get("reviewed") is not True
    ):
        raise ValueError("公開清單必須經 root 具體審閱並標記 approved/reviewed")
    if approval.get("experiment_id") != experiment_id or approval.get("batch_id") != batch_id:
        raise ValueError("審閱清單與 experiment/batch 不相符")
    if not re.fullmatch(r"[0-9a-f]{40}", approval.get("revision", "")):
        raise ValueError("審閱清單需要完整的訓練程式 revision")
    source = approval.get("private_source", {})
    if source.get("repo") != checkpoint_repo or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
        raise ValueError("需要指定此私有 repo 的固定 HF revision")
    prefix = Path(source.get("prefix", ""))
    if prefix.is_absolute() or ".." in prefix.parts or prefix.parts[:3] != ("course", batch_id, experiment_id):
        raise ValueError("私有來源 prefix 必須屬於此 experiment/batch")
    card = approval.get("model_card", {})
    if not all(card.get(key) for key in ("summary", "scope", "limitations")):
        raise ValueError("需要已審閱的 model_card summary/scope/limitations")
    if "repository_index" in approval:
        index = approval["repository_index"]
        if not isinstance(index, dict) or set(index) != {"content", "sha256"}:
            raise ValueError("repository_index 必須恰有已審閱的 content 與 sha256")
        content, digest = index["content"], index["sha256"]
        if not isinstance(content, str) or not content.strip() or not re.search(r"(?m)^#{1,6}[ \t]+\S", content):
            raise ValueError("repository_index content 必須是含標題的非空 Markdown 全文")
        try:
            encoded = content.encode("utf-8")
        except UnicodeError as error:
            raise ValueError("repository_index content 必須是有效 UTF-8") from error
        if not 16 <= len(encoded) <= 262144 or any(
            ord(character) < 32 and character not in "\n\r\t" for character in content
        ):
            raise ValueError("repository_index Markdown 長度需在 16 bytes 至 256 KiB，且不能含控制字元")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or hashlib.sha256(encoded).hexdigest() != digest
        ):
            raise ValueError("repository_index sha256 必須精確匹配 content 的 UTF-8 bytes")
    if approval.get("pku_derived") or approval.get("visibility") == "private":
        raise ValueError("PKU 衍生分支依本項目政策保持私有")
    files = approval.get("files", [])
    if not files or len({item["path"] for item in files}) != len(files):
        raise ValueError("公開檔案清單必須非空且不重複")
    reserved = {
        "result.json",
        "upload-manifest.json",
        "failure.json",
        "current-run.json",
        "approved-public-exports.json",
        "result-attestation.json",
        "approval-attestation.json",
    }
    names = {item["path"] for item in files}
    outputs = set()
    for item in files:
        path = safe_relative(item["path"])
        output = safe_relative(item.get("output", item["path"]))
        if (
            path.name in reserved
            or ".verified-bases" in path.parts
            or output.name in GENERATED_NAMES
            or output.as_posix() in outputs
        ):
            raise ValueError("禁止公開原始執行結果、控制檔或無效路徑")
        outputs.add(output.as_posix())
        if "pku" in path.as_posix().lower() or item.get("pku_derived") or item.get("visibility") == "private":
            raise ValueError("PKU 衍生檔案依本項目政策保持私有")
        if item.get("kind") not in ("checkpoint", "metadata", "dataset", "license"):
            raise ValueError("未知公開檔案 kind")
        if path.suffix in (".pt", ".pth", ".ckpt", ".safetensors") and item["kind"] != "checkpoint":
            raise ValueError("權重必須經 checkpoint 推論匯出器")
        if item.get("base_checkpoint"):
            validate_base(item["base_checkpoint"])
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError("每個公開檔案需有精確的 SHA-256")
        if item.get("redistribution_approved") is not True or not item.get("license"):
            raise ValueError("每個檔案需明示已核對的公開授權")
        companions = item.get("companions", []) + ([item["tokenizer_file"]] if item.get("tokenizer_file") else [])
        if any(name not in names for name in companions):
            raise ValueError("必要 companion/tokenizer 必須一起列入審閱清單")
    return source


def inference_payload(saved, provenance, specification=None, companion_hashes=None):
    """依明確格式保留推論欄位；無格式的自訂模型需要 architecture 說明。"""
    specification = specification or {}
    companion_hashes = companion_hashes or {}
    architecture = specification.get("architecture", {})
    architecture_type = architecture.get("type") if isinstance(architecture, dict) else architecture
    version = saved.get("format_version")
    if version in (1, "quantized-v1", "multimodal-v1"):
        fields = ("format_version", "config", "model", "modal_config", "tokenizer", "bits", "task")
    elif version == "simple-v1" or (version is None and "kind" in saved and "vocabulary" in saved):
        fields = ("model", "vocabulary", "context", "width", "kind")
        version = "simple-v1"
    elif version == "lora-v1":
        fields = ("format_version", "adapter", "config", "base_sha256", "scaling")
        base = specification.get("base_checkpoint", {})
        validate_base(base)
    elif architecture_type in ("VisionEncoder", "AudioEncoder"):
        fields = ("encoder", "classifier", "config", "classes")
        version = "encoder-v1"
        if "encoder" not in saved or "config" not in saved:
            raise ValueError("encoder 公開檔需要 final encoder/config，不能使用未完成的 training 檔")
    elif architecture_type in ("Contrastive", "FinitePolicy"):
        fields = ("model",)
        version = "contrastive-v1" if specification["architecture"]["type"] == "Contrastive" else "finite-policy-v1"
    else:
        raise ValueError("未知 checkpoint 格式；需明確 architecture 或另行匯出 companion，不能猜測")
    clean = {key: saved[key] for key in fields if key in saved}
    clean["format_version"] = version
    if "tokenizer" in clean:
        allowed = ("type", "vocab_size", "sha256", "tokenizer_sha256", "specials", "special_ids", "special_token_ids")
        clean["tokenizer"] = {key: value for key, value in clean["tokenizer"].items() if key in allowed}
    if "modal_config" in clean:
        allowed = ("vision_width", "audio_width", "image_size", "patch_size", "bands")
        clean["modal_config"] = {key: value for key, value in clean["modal_config"].items() if key in allowed}
    if version == "lora-v1":
        clean["base_checkpoint"] = {
            key: specification["base_checkpoint"][key] for key in ("repo", "revision", "filename", "sha256")
        }
    if "architecture" in specification:
        clean["architecture"] = specification["architecture"]
    original = saved.get("metadata", {})
    # 只保留解碼、adapter 綁定與模型用途；不複製 data/samples/report/teacher。
    keys = (
        "tokenizer",
        "tokenizer_sha256",
        "tokenizer_file_sha256",
        "tokenizer_vocab_size",
        "special_ids",
        "special_token_ids",
        "base_sha256",
        "adapter",
        "scaling",
        "experiment",
        "task",
        "scope",
    )
    metadata = {key: original[key] for key in keys if key in original}
    if "tokenizer" in metadata and not isinstance(metadata["tokenizer"], str):
        raise ValueError("metadata tokenizer 只接受可核驗的類型名稱")
    for key in ("adapter", "scaling", "experiment", "task", "scope"):
        if key in metadata and not isinstance(metadata[key], str):
            raise ValueError("模型用途及 adapter metadata 只接受文字，不能包含未知資料")
    metadata.update(provenance)
    tokenizer_file = specification.get("tokenizer_file")
    if tokenizer_file:
        expected = companion_hashes[tokenizer_file]
        for value in (
            metadata.get("tokenizer_sha256"),
            metadata.get("tokenizer_file_sha256"),
            clean.get("tokenizer", {}).get("sha256"),
        ):
            if value is not None and value != expected:
                raise ValueError("原始 checkpoint 的 tokenizer SHA 與核准 companion 不相符")
        metadata["tokenizer_sha256"] = companion_hashes[tokenizer_file]
        metadata["tokenizer_file"] = specification.get("tokenizer_output", tokenizer_file)
        binding = dict(clean.get("tokenizer", {}))
        binding["sha256"] = companion_hashes[tokenizer_file]
        clean["tokenizer"] = binding
    clean["metadata"] = metadata
    return clean


def validate_export(path):
    """實際重建自訂模型或 native loader，避免發布學生根本載不進去的檔。"""
    import torch
    from torch import nn

    from scripts.infer_simple import load_simple_checkpoint
    from tiny_perceptron.model import ModelConfig, TinyLM
    from tiny_perceptron.multimodal import AudioEncoder, VisionEncoder
    from tiny_perceptron.training import load_checkpoint

    saved = torch.load(path, map_location="cpu", weights_only=True)
    version = saved["format_version"]
    if version in (1, "quantized-v1", "multimodal-v1"):
        load_checkpoint(path)
    elif version == "simple-v1":
        load_simple_checkpoint(path)
    elif version == "lora-v1":
        model = TinyLM(ModelConfig(**saved["config"]))
        for name, state in saved["adapter"].items():
            if state["rank"] < 1 or state["a"].shape[0] != state["rank"] or state["b"].shape[1] != state["rank"]:
                raise ValueError("LoRA adapter 的 rank/A/B 形狀不一致")
            linear = model.get_submodule(name)
            if (
                not isinstance(linear, nn.Linear)
                or state["a"].shape[1] != linear.in_features
                or state["b"].shape[0] != linear.out_features
            ):
                raise ValueError("LoRA adapter 與 base 模型層形狀不一致")
    elif version == "encoder-v1":
        model = (
            VisionEncoder(**saved["config"])
            if saved["architecture"]["type"] == "VisionEncoder"
            else AudioEncoder(**saved["config"])
        )
        model.load_state_dict(saved["encoder"], strict=True)
        if "classifier" in saved:
            width = saved["config"]["width"]
            classifier = nn.Linear(width, len(saved["classes"]))
            classifier.load_state_dict(saved["classifier"], strict=True)
    elif version == "contrastive-v1":
        from scripts.course_experiments.modalities import _Contrastive

        architecture = saved["architecture"]
        if (
            architecture.get("vision_config") != {"width": 16, "image_size": 16, "patch_size": 4}
            or architecture.get("text_vocab_size") != 264
            or architecture.get("temperature") != 0.1
            or architecture.get("pooling") != "mean_then_l2"
        ):
            raise ValueError("Contrastive 必須明示訓練時的 encoder/tokenizer/temperature/pooling")
        _Contrastive().load_state_dict(saved["model"], strict=True)
    elif version == "finite-policy-v1":
        from scripts.course_experiments.applications import _FinitePolicy

        architecture = saved["architecture"]
        if not all(key in architecture for key in ("actions", "input_features")):
            raise ValueError("有限動作策略需要明確的 actions 與 input_features")
        model = _FinitePolicy()
        model.load_state_dict(saved["model"], strict=True)
        if len(architecture["actions"]) != model.net[-1].out_features:
            raise ValueError("策略 action 數與模型輸出不一致")
        if (
            architecture.get("input_features") != "three six-way one-hot operands"
            or architecture.get("hidden_width") != 48
            or architecture.get("activation") != "tanh"
        ):
            raise ValueError("策略需要與訓練一致的 input_features/hidden_width/activation")
    else:
        raise ValueError("沒有此格式的推論驗證器")
    return version


def write_model_card(path, approval, repo, prefix, weights_revision=None):
    card = approval["model_card"]
    limitations = card["limitations"] if isinstance(card["limitations"], list) else [card["limitations"]]
    lines = [
        f"# {approval['experiment_id']}：教學模型",
        "",
        card["summary"],
        "",
        card["scope"],
        "",
        "限制：",
        "",
        *[f"- {item}" for item in limitations],
        "",
        f"訓練程式版本：`{approval['revision']}`。授權逐檔列於 export-manifest.json。",
        "",
        "權重許可及上游來源聲明見 LICENSE 與 THIRD_PARTY_NOTICES.md；程式碼採 MIT。",
        "",
        "下載的 checkpoint 已移除 optimizer、RNG 與內嵌的訓練參照模型；自訂模型的 architecture 保留在檔案中。清單若列有教師基準，它是可單獨載入的推論權重。",
        "",
    ]
    if card.get("evaluation"):
        lines += [
            "已審閱的實測資料：",
            "",
            "```json",
            json.dumps(card["evaluation"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    if weights_revision:
        lines += ["固定版本下載：", ""]
        for item in approval["files"]:
            filename = item.get("output", item["path"])
            lines.append(
                f"- [{filename}](https://huggingface.co/{repo}/resolve/{weights_revision}/{prefix}/{filename}?download=true)"
            )
    inference = card.get("inference") or approval.get("inference")
    if inference:
        lines += [
            "",
            "使用設定：",
            "",
            "```json",
            json.dumps(inference, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
        if isinstance(inference, dict) and inference.get("commands"):
            lines += ["在專案根目錄執行：", "", "```bash", *inference["commands"], "```", ""]
    if card.get("inference_variants"):
        lines += ["各權重的使用設定：", ""]
        for filename, settings in card["inference_variants"].items():
            lines += [f"### {filename}", "", "```json", json.dumps(settings, ensure_ascii=False, indent=2), "```", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_notices(destination, approval):
    """由已審閱的來源聲明生成文件；不把訓練結果中的樣本搬進 notice。"""
    card = approval["model_card"]
    default = card.get("weights_license", "MIT")
    licenses = {item["license"] for item in approval["files"] if item["kind"] == "checkpoint"} or {default}
    lines = ["Weights licenses", "", "Code: MIT. The license of each weight file follows:", ""]
    lines += [f"- {item.get('output', item['path'])}: {item['license']}" for item in approval["files"]]
    if any(value.lower() == "mit" for value in licenses):
        lines += ["", MIT_TEXT]
    if any(value.upper() == "CC-BY-SA-4.0" for value in licenses):
        lines += [
            "",
            "CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/legalcode.en",
            "This project elects this license for these weights; it does not claim training always creates Adapted Material.",
        ]
    unsupported = {value for value in licenses if value.lower() != "mit" and value.upper() != "CC-BY-SA-4.0"}
    if unsupported and not card.get("weights_license_text"):
        raise ValueError("其他權重許可需要提供已審閱的 weights_license_text")
    if card.get("weights_license_text"):
        lines += ["", card["weights_license_text"]]
    (destination / "LICENSE").write_text("\n".join(lines) + "\n", encoding="utf-8")
    notices = [
        "# Third-party notices",
        "",
        "Training-data provenance; these licenses do not change the course code license.",
        "",
    ]
    for data in card.get("training_data", []):
        notices += [f"## {data['source']}", ""]
        for key in ("revision", "license", "url", "modifications", "attribution", "redistributed_raw_data"):
            if key in data:
                notices.append(f"{key}: {data[key]}")
        notices.append("")
    extra = card.get("third_party_notices", [])
    if isinstance(extra, str):
        extra = [extra]
    notices += [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, indent=2) for item in extra]
    (destination / "THIRD_PARTY_NOTICES.md").write_text("\n".join(notices) + "\n", encoding="utf-8")


def validate_lora_base(directory, clean):
    import torch

    base_pointer = clean["base_checkpoint"]
    file = directory / ".verified-bases" / f"{base_pointer['sha256']}.pt"
    if file_sha256(file) != base_pointer["sha256"]:
        raise ValueError("LoRA 的公開 base 檔案 SHA 不相符")
    saved = torch.load(file, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1 or saved.get("config") != clean["config"]:
        raise ValueError("LoRA 的公開 base 必須是相同 config 的未量化 TinyLM")
    digest = hashlib.sha256()
    for name, value in sorted(saved["model"].items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    if digest.hexdigest() != clean["base_sha256"]:
        raise ValueError("LoRA 訓練時的 base state identity 與公開 base 不相符")


def build_export(directory, destination, approval, provenance):
    import shutil

    import torch

    directory, destination = Path(directory), Path(destination)
    validate_approval(approval, approval["experiment_id"], approval["batch_id"], approval["private_source"]["repo"])
    destination.mkdir(parents=True, exist_ok=True)
    companion_hashes = {item["path"]: item["sha256"] for item in approval["files"]}
    companion_outputs = {item["path"]: item.get("output", item["path"]) for item in approval["files"]}
    files = []
    for item in approval["files"]:
        source = directory / item["path"]
        if (
            source.is_symlink()
            or not source.resolve().is_relative_to(directory.resolve())
            or file_sha256(source) != item["sha256"]
        ):
            raise ValueError("審閱後來源檔案 SHA 或路徑已變更")
        output = item.get("output", item["path"])
        target = destination / output
        if (
            Path(output).is_absolute()
            or ".." in Path(output).parts
            or not target.resolve().is_relative_to(destination.resolve())
        ):
            raise ValueError("公開輸出路徑無效")
        target.parent.mkdir(parents=True, exist_ok=True)
        if item["kind"] == "checkpoint":
            specification = dict(item)
            if item.get("tokenizer_file"):
                specification["tokenizer_output"] = companion_outputs[item["tokenizer_file"]]
            clean = inference_payload(
                torch.load(source, map_location="cpu", weights_only=True), provenance, specification, companion_hashes
            )
            if clean["format_version"] == "lora-v1":
                validate_lora_base(directory, clean)
            if clean.get("config", {}).get("vocab_size", 264) != 264 and not item.get("tokenizer_file"):
                raise ValueError("非 byte 模型必須公開配對 tokenizer companion")
            if clean["format_version"] in (1, "multimodal-v1", "quantized-v1"):
                from tiny_perceptron.tokenization import load_tokenizer

                tokenizer_path = directory / item["tokenizer_file"] if item.get("tokenizer_file") else None
                load_tokenizer(tokenizer_path, clean["config"]["vocab_size"], clean)
            torch.save(clean, target)
            validate_export(target)
        elif item["kind"] in ("metadata", "dataset", "license"):
            shutil.copyfile(source, target)
        else:
            raise ValueError("不支援的公开檔案 kind")
        files.append(
            {
                "output": output,
                "sha256": file_sha256(target),
                "bytes": target.stat().st_size,
                "license": item["license"],
                "source_sha256": item["sha256"],
            }
        )
    manifest = {"schema_version": 1, "files": files, "provenance": provenance}
    (destination / "export-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if approval["model_card"].get("evaluation"):
        (destination / "evaluation.json").write_text(
            json.dumps(approval["model_card"]["evaluation"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    write_notices(destination, approval)
    return files
