"""從已發布的固定版本 manifest 生成 HF 根 README 預覽；不執行任何發布。"""

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path
from urllib.parse import quote

from scripts.course_release import safe_relative

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/course-experiments/public-models.json"
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def _validate_model(model):
    if not isinstance(model, dict):
        raise ValueError("每個已發布模型必須是 object")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", model.get("id", "")):
        raise ValueError("已發布模型 id 無效")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model.get("repo", "")):
        raise ValueError("已發布模型需要 HF owner/repo")
    if not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
        raise ValueError("已發布模型需要固定的完整 HF revision，不能使用 main")
    files = model.get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError("已發布模型的檔案清單不能為空")
    outputs = set()
    for item in files:
        safe_relative(item["path"])
        safe_relative(item["output"])
        if any(ord(character) < 32 for character in item["path"] + item["output"]):
            raise ValueError("已發布檔名不能含控制字元")
        if item["output"] in outputs:
            raise ValueError("已發布模型的 output 檔名重複")
        outputs.add(item["output"])
        if (
            not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
            or type(item.get("bytes")) is not int
            or item["bytes"] < 1
        ):
            raise ValueError("已發布檔案需要 SHA-256 與正整數 bytes")
    if not {"README.md", "LICENSE", "export-manifest.json"}.issubset(outputs):
        raise ValueError("已發布模型需要模型卡、LICENSE 與逐檔匯出清單")
    if not any(Path(item["output"]).suffix in WEIGHT_SUFFIXES for item in files):
        raise ValueError("已發布模型至少需要一份權重")


def _url(model, item, download=False):
    route = "resolve" if download else "blob"
    url = f"https://huggingface.co/{model['repo']}/{route}/{model['revision']}/{quote(item['path'], safe='/')}"
    return url + "?download=true" if download else url


def _inference_command(model):
    inference = model.get("inference", {})
    if not inference:
        return None
    script, checkpoint = inference.get("script"), inference.get("checkpoint")
    if not isinstance(script, str) or safe_relative(script).parts[0] != "scripts" or not script.endswith(".py"):
        raise ValueError("公開推論設定需要 scripts/*.py")
    outputs = {item["output"] for item in model["files"]}
    if checkpoint not in outputs:
        raise ValueError("公開推論 checkpoint 不在此模型的已發布檔案內")
    folder = Path("checkpoints/course") / model["id"]
    argv = ["uv", "run", "--extra", "cpu", "python", script, (folder / checkpoint).as_posix()]
    flags = {
        "prompt",
        "tokens",
        "temperature",
        "chat",
        "tokenizer",
        "adapter",
        "image",
        "audio",
        "color",
        "shape",
        "frequency",
        "device",
        "resample_audio",
        "json",
        "cache",
    }
    unknown = set(inference) - flags - {"script", "checkpoint"}
    if unknown:
        raise ValueError(f"公開推論設定含尚未支援的 CLI 欄位：{sorted(unknown)}")
    for name, value in inference.items():
        if name in ("script", "checkpoint") or value is False or value is None:
            continue
        argv.append("--" + name.replace("_", "-"))
        if value is True:
            continue
        if type(value) not in (str, int, float):
            raise ValueError("公開推論參數需是文字、數字或 boolean")
        if name in ("tokenizer", "adapter"):
            if value not in outputs:
                raise ValueError("公開推論 companion 不在此模型的已發布檔案內")
            value = (folder / value).as_posix()
        argv.append(str(value))
    return shlex.join(argv)


def build_index(manifest):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), list):
        raise ValueError("需要 schema_version=1 的 public-models.json")
    models = manifest["models"]
    if not models:
        raise ValueError("沒有已發布模型，不能生成模型發布索引")
    if len({model.get("id") for model in models}) != len(models):
        raise ValueError("public-models.json 的模型 id 重複")
    for model in models:
        _validate_model(model)
    lines = [
        "# Tiny Perceptron：課程教學模型",
        "",
        "這些 tiny 模型用來練習從零訓練、下載與推論。"
        "各模型的任務、實測結果與限制以其模型卡為準；小型受控任務的結果不能推廣為通用能力或高性能保證。",
        "",
        "本頁僅列出課程 `docs/course-experiments/public-models.json` 中已發布的模型，"
        "不代表所有課程實驗都已完成。下載連結固定在各次發布的 HF commit；根 README 的後續更新不改變那些版本。",
        "",
        "程式碼採 **MIT**。**權重與資料按檔案適用個別授權**；下載前請閱讀各模型的 "
        "`LICENSE`、`THIRD_PARTY_NOTICES.md` 與 `export-manifest.json`，不能把程式碼的 MIT 當成所有檔案的權重授權。",
        "",
        "| 已發布模型 | 模型卡 | 固定 HF revision | 逐檔授權 |",
        "| --- | --- | --- | --- |",
    ]
    for model in models:
        by_output = {item["output"]: item for item in model["files"]}
        lines.append(
            f"| `{model['id']}` | [任務與實測限制]({_url(model, by_output['README.md'])}) "
            f"| [`{model['revision'][:12]}`](https://huggingface.co/{model['repo']}/commit/{model['revision']}) "
            f"| [LICENSE]({_url(model, by_output['LICENSE'])}) / "
            f"[逐檔清單]({_url(model, by_output['export-manifest.json'])}) |"
        )
    lines += [
        "",
        "## 在課程 checkout 下載與推論",
        "",
        "使用本課程程式碼與配對的推論腳本。下載器只取選定模型，匿名下載並核對固定 revision、檔案大小與 SHA-256。",
        "",
        "```bash",
        "uv sync --extra cpu",
        "uv run --extra cpu python scripts/fetch_course_models.py --list",
        "```",
        "",
        "下面列出公開清單中已提供的推論設定；生成輸出仍需核對，不從下載成功推論模型能力。",
        "RAG 的文字推論範例直接提供文件；工具模型的文字推論只產生請求，沒有自行執行計算。"
        "檢索、嚴格驗證、工具執行與結果回填仍是外層程式的工作；完整操作請依各模型卡的流程。",
    ]
    for model in models:
        lines += [
            "",
            f"### {model['id']}",
            "",
            "```bash",
            shlex.join(
                ["uv", "run", "--extra", "cpu", "python", "scripts/fetch_course_models.py", "--model", model["id"]]
            ),
        ]
        inference = _inference_command(model)
        if inference:
            lines.append(inference)
        lines += ["```", ""]
        if inference is None:
            lines += ["此項公開清單未指定推論 CLI；請依模型卡的使用設定與 architecture 載入。", ""]
        lines += ["固定版本權重：", ""]
        for item in model["files"]:
            if Path(item["output"]).suffix in WEIGHT_SUFFIXES:
                label = item["output"].replace("[", "\\[").replace("]", "\\]")
                lines.append(
                    f"- [{label}]({_url(model, item, download=True)}) · {item['bytes']:,} bytes · SHA-256 `{item['sha256']}`"
                )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, help="儲存可供 root 審閱的 Markdown；省略時印至 stdout")
    parser.add_argument("--index-json", type=Path, help="另存 repository_index content/sha256；不代表已核准")
    args = parser.parse_args()
    content = build_index(json.loads(args.manifest.read_text(encoding="utf-8")))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    if args.index_json:
        args.index_json.parent.mkdir(parents=True, exist_ok=True)
        args.index_json.write_text(
            json.dumps(
                {"content": content, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
